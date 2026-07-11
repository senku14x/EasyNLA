"""Multi-layer AV-SFT — three-slot verbalizer warm-start (plan §6.1, §6.3).

Mirrors `nla.train_sft --mode av` but injects THREE activations at THREE markers
(prev/centre/next) via `injection_multi.register_multislot_hook`, over the
three-slot AV prompt. Cross-entropy on response tokens only. RAW vectors are
injected (the Karvonen hook norm-matches internally — plan §4 Rev 2).

wandb: loss, lr, grad_norm, response tokens, and `mean_cjk` — the CJK-fraction
canary (CLAUDE.md): if injection silently fails the actor sees the literal
marker char and free-associates Chinese, so mean_cjk spiking is the loudest
smoke signal for a broken injection path.

peft / bitsandbytes / wandb are imported lazily so the loss step
(`av_compute_loss`) is unit-testable on a tiny model with no extra deps.
"""

import argparse
import math
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from multilayer_nla.datasets import detect_av_slots, load_av_sft_dataset, prepare_av_chunk_multi
from multilayer_nla.injection_multi import register_multislot_hook


def build_lr_lambda(warmup_steps, total_steps, min_lr_ratio):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return fn


def cjk_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if "CJK" in unicodedata.name(c, "")) / len(text)


def av_compute_loss(model, batch_ids, attn, loss_mask, vectors, vectors_ref, prompt_lens=None):
    """Forward with the three-slot injection active, CE on response tokens only.

    Factored out so it can be plumbing-tested on a tiny model: sets vectors_ref
    for the hook, forwards, shifts-by-one, masks to the response, returns
    (loss, n_response_tokens). The hook reads vectors_ref[0]; we always clear it.
    prompt_lens (optional) bounds injection to the prompt span (the gold response is
    marker-free, so harmless here — it keeps the payload protocol identical to RL).
    """
    vectors_ref[0] = {"vectors": vectors, "prompt_lens": prompt_lens}
    try:
        logits = model(input_ids=batch_ids, attention_mask=attn).logits.float()
    finally:
        vectors_ref[0] = None
    shift_logits = logits[:, :-1].contiguous()
    shift_targets = batch_ids[:, 1:].to(shift_logits.device).contiguous()
    shift_mask = loss_mask[:, 1:].to(shift_logits.device).contiguous()
    V = shift_logits.size(-1)
    per_tok = F.cross_entropy(
        shift_logits.view(-1, V), shift_targets.view(-1), reduction="none",
    ).view(shift_targets.shape)
    n_resp = shift_mask.sum().clamp(min=1)
    loss = (per_tok * shift_mask).sum() / n_resp
    return loss, int(n_resp.item())


@torch.no_grad()
def evaluate_av(model, eval_rows, tokenizer, inject_char, inj_id, vectors_ref, device,
                max_len, batch_size, slot_cols, max_batches=None):
    """Held-out response CE on docs unseen in training (no grad), token-weighted mean."""
    was_training = model.training
    model.eval()
    tot_loss, tot_resp, nb = 0.0, 0, 0
    for cs in range(0, len(eval_rows), batch_size):
        chunk = eval_rows[cs:cs + batch_size]
        ids, attn, loss_mask, vectors, plens = prepare_av_chunk_multi(
            chunk, tokenizer, inject_char, inj_id, device, max_len=max_len, slot_cols=slot_cols)
        loss, n_resp = av_compute_loss(model, ids, attn, loss_mask, vectors, vectors_ref, plens)
        tot_loss += loss.item() * n_resp
        tot_resp += n_resp
        nb += 1
        if max_batches and nb >= max_batches:
            break
    if was_training:
        model.train()
    return tot_loss / max(tot_resp, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--parquet", required=True, help="av_sft parquet (prompt + response + 3 activations)")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--min-lr", type=float, default=3e-6)
    p.add_argument("--lr-warmup-steps", type=int, default=50)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--use-lora", action="store_true", default=False)
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--ram-dtype", choices=["float32", "float16"], default="float32",
                   help="in-RAM dtype for loaded activations; float16 halves resident memory "
                        "for big-k permutation datasets (batch prep upcasts to fp32)")
    p.add_argument("--eval-parquet", default=None,
                   help="optional held-out AV parquet (response CE on unseen docs); the headline AV "
                        "metric is end-to-end (evaluate_e2e), not this CE. The condition lives in the "
                        "data (av_in_* slot layers), so there is no --condition flag.")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=25, help="batches per held-out eval pass (caps cost)")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="mlnla-qwen3-8b")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16

    # Marker char/id straight from the tokenizer (reuses the single-layer auto-pick
    # + committed cache). Multi-slot uses the per-row count guard, not the
    # single-marker neighbor check, so no sidecar neighbor verification is needed.
    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    from nla.datagen.injection_tokens import find_injection_token
    inject_char, inj_id = find_injection_token(tokenizer)
    slot_cols = detect_av_slots(args.parquet)
    k = len(slot_cols)
    print(f"[av] marker {inject_char!r} -> id {inj_id}; k={k} AV-input slots {slot_cols}")

    quant_config = None
    if args.quant == "4bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=dtype, attn_implementation=args.attn_implementation,
        quantization_config=quant_config, device_map=({"": 0} if quant_config else None),
    )
    if quant_config is None:
        model = model.to(device)
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from nla.utils.arch_adapters import resolve_attn_target_modules
        if quant_config is not None:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.gradient_checkpointing)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", use_rslora=True,
            target_modules=resolve_attn_target_modules(model.config)))
        model.print_trainable_parameters()

    vectors_ref = [None]
    register_multislot_hook(model, vectors_ref, inj_id, k, layer_idx=1)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    ram_dtype = np.float16 if args.ram_dtype == "float16" else np.float32
    rows = load_av_sft_dataset(args.parquet, n_max=args.max_rows, slot_cols=slot_cols,
                               ram_dtype=ram_dtype)
    print(f"[av] {len(rows)} rows")
    eval_rows = None
    if args.eval_parquet:
        eval_rows = load_av_sft_dataset(args.eval_parquet, slot_cols=slot_cols, ram_dtype=ram_dtype)
        print(f"[av] held-out eval: {len(eval_rows)} rows from {args.eval_parquet}")

    try:
        import bitsandbytes as bnb
        optim_cls = bnb.optim.AdamW8bit
    except ImportError:
        optim_cls = torch.optim.AdamW
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = optim_cls(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, build_lr_lambda(args.lr_warmup_steps, args.num_steps, args.min_lr / max(args.lr, 1e-12)))

    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    rng = np.random.default_rng(args.seed)
    perm = list(range(len(rows)))
    rng.shuffle(perm)
    cursor = 0
    grad_accum = args.gradient_accumulation_steps

    for step in range(args.num_steps):
        t0 = time.time()
        optim.zero_grad()
        accum_loss, accum_resp = 0.0, 0
        for _ in range(grad_accum):
            if cursor + args.batch_size > len(perm):
                rng.shuffle(perm); cursor = 0
            chunk = [rows[i] for i in perm[cursor:cursor + args.batch_size]]
            cursor += args.batch_size
            ids, attn, loss_mask, vectors, plens = prepare_av_chunk_multi(
                chunk, tokenizer, inject_char, inj_id, device, max_len=args.max_len, slot_cols=slot_cols)
            loss, n_resp = av_compute_loss(model, ids, attn, loss_mask, vectors, vectors_ref, plens)
            (loss / grad_accum).backward()
            accum_loss += loss.item(); accum_resp += n_resp
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optim.step(); sched.step()

        # CJK canary on a freshly sampled batch's would-be generations is too slow;
        # cheap proxy: fraction in the training responses (should be ~0; a spike in
        # generated text at eval is the real signal — logged by the RL trainer).
        mean_loss = accum_loss / grad_accum
        log = {
            "step": step, "loss": mean_loss, "lr": sched.get_last_lr()[0],
            "grad_norm": float(grad_norm), "resp_tokens": accum_resp,
            "wall_s": time.time() - t0,
        }
        print(f"step {step:04d} | loss {mean_loss:.4f} | lr {log['lr']:.2e} | "
              f"grad {log['grad_norm']:.2f} | resp {accum_resp} | t {log['wall_s']:.1f}s", flush=True)
        if not args.no_wandb:
            import wandb
            wandb.log(log, step=step)

        if eval_rows is not None and ((step + 1) % args.eval_every == 0 or (step + 1) == args.num_steps):
            ev_loss = evaluate_av(model, eval_rows, tokenizer, inject_char, inj_id, vectors_ref,
                                  device, args.max_len, args.batch_size, slot_cols, args.eval_batches)
            print(f"  [eval] held-out loss {ev_loss:.4f}", flush=True)
            if not args.no_wandb:
                import wandb
                wandb.log({"eval/loss": ev_loss}, step=step)

        if (step + 1) % args.save_every == 0 or (step + 1) == args.num_steps:
            out = save_dir / f"iter_{step + 1:07d}"
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out)); tokenizer.save_pretrained(str(out))
            print(f"[save] -> {out}")
    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
