"""Multi-layer AR-SFT — multi-tap reconstructor warm-start (plan §6.2, §6.3).

Truncated backbone through block 25 + three depth heads (models_multi). Loss =
three-target MSE on √d-normalized (pred, gold). Mirrors `nla.train_sft --mode ar`
but reconstructs all three depths at once.

wandb plots (what you'll watch go up): per-tap state FVE — fve/prev, fve/centre,
fve/next — plus fve/overall, and the matching per-tap losses.

peft / bitsandbytes / wandb lazy-imported; `ar_compute_loss` + `per_tap_mse` are
unit-testable on a tiny model with no extra deps.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from nla.schema import compute_predict_mean_baselines, normalize_activation
from multilayer_nla.datasets import (
    AR_LAYER_TO_TARGET_COL,
    AR_TARGET_COL_TO_NAME,
    SLOT_COLUMNS,
    load_ar_sft_dataset,
    prepare_ar_chunk_multi,
)
from multilayer_nla.models_multi import (
    DEFAULT_TAP_LAYERS,
    init_multitap_critic_from_base,
    multitap_predict,
    three_target_loss,
)
from multilayer_nla.train_av_multi import build_lr_lambda

SLOT_NAMES = ("prev", "centre", "next")


def ar_compute_loss(model, batch_ids, attn, gold, mse_scale):
    """multitap_predict -> three_target_loss. Returns (loss, pred[B,n_taps,d])."""
    pred = multitap_predict(model, batch_ids, attn, mse_scale)
    loss = three_target_loss(pred, gold, mse_scale)
    return loss, pred


def per_tap_mse(pred, gold, mse_scale):
    """Per-tap normalized MSE [n_taps] — mean over batch and feature dims."""
    assert pred.shape[1] == gold.shape[1], (
        f"per_tap_mse tap/target mismatch: pred {tuple(pred.shape)} vs gold {tuple(gold.shape)} "
        f"(would broadcast). Pass target_cols matching the model's tap_layers.")
    pn = normalize_activation(pred, mse_scale)
    gn = normalize_activation(gold, mse_scale)
    return ((pn - gn) ** 2).mean(dim=(0, 2))


def _per_tap_baselines(rows, mse_scale, cols=SLOT_COLUMNS):
    """Predict-the-mean MSE baseline per tap (the FVE denominator, §8), one per
    target column in `cols` (subset for a single-/two-tap AR)."""
    bls = []
    for c in cols:
        acts = torch.tensor(np.stack([r[c] for r in rows[:4000]]), dtype=torch.float32)
        _, rawvar = compute_predict_mean_baselines(acts, mse_scale)
        bls.append(rawvar)
    return bls


@torch.no_grad()
def evaluate_ar(model, eval_rows, tokenizer, mse_scale, baselines, device,
                max_len, batch_size, max_batches=None, target_cols=SLOT_COLUMNS):
    """Held-out per-tap MSE + FVE (no grad). The FVE denominator is the eval set's
    OWN predict-mean baseline (its target variance), so this is generalization FVE on
    documents unseen in training. `target_cols` must match the model's tap_layers.
    Returns (mse[list], fve[list], mean_loss)."""
    was_training = model.training
    model.eval()
    mse_acc = torch.zeros(len(baselines), device=device)
    loss_acc, nb = 0.0, 0
    for cs in range(0, len(eval_rows), batch_size):
        chunk = eval_rows[cs:cs + batch_size]
        ids, attn, gold = prepare_ar_chunk_multi(chunk, tokenizer, device, max_len=max_len,
                                                 target_cols=target_cols)
        loss, pred = ar_compute_loss(model, ids, attn, gold, mse_scale)
        mse_acc += per_tap_mse(pred, gold, mse_scale)
        loss_acc += loss.item()
        nb += 1
        if max_batches and nb >= max_batches:
            break
    if was_training:
        model.train()
    mse = (mse_acc / max(nb, 1)).tolist()
    fve = [1.0 - m / bl for m, bl in zip(mse, baselines)]
    return mse, fve, loss_acc / max(nb, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--parquet", required=True, help="ar_sft parquet (prompt + 3 activations)")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--tap-layers", default="23,24,25", help="comma-sep tap blocks (center l-1,l,l+1)")
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
    p.add_argument("--strip-final-norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--ram-dtype", choices=["float32", "float16"], default="float32",
                   help="in-RAM dtype for loaded activations; float16 halves resident memory "
                        "for big-k permutation datasets (batch prep upcasts to fp32)")
    p.add_argument("--eval-parquet", default=None,
                   help="held-out AR eval parquet (ar_dev.parquet / ar_test.parquet). Reports per-tap "
                        "gold FVE on docs unseen in AR training: gold explanation -> shared AR -> fixed "
                        "[L23,L24,L25]. Baseline is the eval split's own predict-the-mean (not train rows).")
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
    tap_layers = tuple(int(x) for x in args.tap_layers.split(","))
    # The AR reconstruction target is fixed within {L23,L24,L25}; map each tap to its
    # target column so a single-/two-tap AR (capacity ablation) pulls ONLY that column.
    assert all(l in AR_LAYER_TO_TARGET_COL for l in tap_layers), (
        f"AR tap layers must be within the fixed target {{23,24,25}}; got {tap_layers}. "
        f"The §7 invariant fixes the AR target at [L23,L24,L25] for every condition.")
    target_cols = tuple(AR_LAYER_TO_TARGET_COL[l] for l in tap_layers)
    tap_names = tuple(AR_TARGET_COL_TO_NAME[c] for c in target_cols)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)

    quant_config = None
    if args.quant == "4bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=dtype,
        )
    model = init_multitap_critic_from_base(
        args.base_ckpt, tap_layers, dtype, quant_config,
        device_map=({"": 0} if quant_config else None),
        strip_final_norm=args.strip_final_norm,
    )
    if quant_config is None:
        model = model.to(device)

    from nla.utils.arch_adapters import resolve_attn_target_modules
    lora_target_modules = resolve_attn_target_modules(model.backbone.config)
    if args.use_lora:
        from peft import LoraConfig, inject_adapter_in_model, prepare_model_for_kbit_training
        if quant_config is not None:
            model.backbone = prepare_model_for_kbit_training(model.backbone, use_gradient_checkpointing=False)
        inject_adapter_in_model(LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", use_rslora=True,
            target_modules=lora_target_modules), model.backbone)
        for n_, p_ in model.named_parameters():
            p_.requires_grad_(("lora_" in n_) or n_.startswith("heads."))
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[ar] LoRA-injected; trainable {n_tr / 1e6:.1f}M (lora + {len(tap_layers)} heads)")
    model.train()

    ram_dtype = np.float16 if args.ram_dtype == "float16" else np.float32
    rows = load_ar_sft_dataset(args.parquet, n_max=args.max_rows, ram_dtype=ram_dtype)
    d_model = int(np.asarray(rows[0][SLOT_COLUMNS[0]]).shape[-1])
    mse_scale = math.sqrt(d_model)
    print(f"[ar] {len(rows)} rows, d_model={d_model}, mse_scale={mse_scale:.3f}, taps={tap_layers}")
    baselines = _per_tap_baselines(rows, mse_scale, target_cols)
    print("[ar] per-tap predict-mean baselines: " +
          ", ".join(f"{nm}={b:.4f}" for nm, b in zip(tap_names, baselines)))
    eval_rows = eval_baselines = None
    if args.eval_parquet:
        eval_rows = load_ar_sft_dataset(args.eval_parquet, ram_dtype=ram_dtype)
        eval_baselines = _per_tap_baselines(eval_rows, mse_scale, target_cols)
        print(f"[ar] held-out eval: {len(eval_rows)} rows from {args.eval_parquet}; baselines " +
              ", ".join(f"{nm}={b:.4f}" for nm, b in zip(tap_names, eval_baselines)))

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
    rng = np.random.default_rng(args.seed)
    perm = list(range(len(rows)))
    rng.shuffle(perm)
    cursor = 0
    grad_accum = args.gradient_accumulation_steps

    for step in range(args.num_steps):
        t0 = time.time()
        optim.zero_grad()
        accum_loss = 0.0
        tap_mse_acc = torch.zeros(len(tap_layers), device=device)
        for _ in range(grad_accum):
            if cursor + args.batch_size > len(perm):
                rng.shuffle(perm); cursor = 0
            chunk = [rows[i] for i in perm[cursor:cursor + args.batch_size]]
            cursor += args.batch_size
            ids, attn, gold = prepare_ar_chunk_multi(chunk, tokenizer, device, max_len=args.max_len,
                                                     target_cols=target_cols)
            loss, pred = ar_compute_loss(model, ids, attn, gold, mse_scale)
            (loss / grad_accum).backward()
            accum_loss += loss.item()
            tap_mse_acc += per_tap_mse(pred.detach(), gold, mse_scale) / grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optim.step(); sched.step()

        mean_loss = accum_loss / grad_accum
        tap_mse = tap_mse_acc.tolist()
        log = {"step": step, "loss": mean_loss, "lr": sched.get_last_lr()[0],
               "grad_norm": float(grad_norm), "wall_s": time.time() - t0}
        fve_overall = 0.0
        for nm, m, bl in zip(tap_names, tap_mse, baselines):
            fve = 1.0 - m / bl
            log[f"loss/{nm}"] = m
            log[f"fve/{nm}"] = fve
            fve_overall += fve / len(tap_layers)
        log["fve/overall"] = fve_overall
        fve_str = "/".join(f"{log[f'fve/{nm}']*100:.1f}" for nm in tap_names)
        print(f"step {step:04d} | loss {mean_loss:.4f} | FVE {'/'.join(tap_names)} "
              f"{fve_str}% | lr {log['lr']:.2e} | t {log['wall_s']:.1f}s", flush=True)
        if not args.no_wandb:
            import wandb
            wandb.log(log, step=step)

        if eval_rows is not None and ((step + 1) % args.eval_every == 0 or (step + 1) == args.num_steps):
            ev_mse, ev_fve, ev_loss = evaluate_ar(
                model, eval_rows, tokenizer, mse_scale, eval_baselines,
                device, args.max_len, args.batch_size, args.eval_batches, target_cols=target_cols)
            elog = {"eval/loss": ev_loss, "eval/fve/overall": sum(ev_fve) / len(ev_fve)}
            for nm, m, f in zip(tap_names, ev_mse, ev_fve):
                elog[f"eval/loss/{nm}"] = m
                elog[f"eval/fve/{nm}"] = f
            ev_str = "/".join(f"{f*100:.1f}" for f in ev_fve)
            print(f"  [eval] held-out FVE {'/'.join(tap_names)} {ev_str}% "
                  f"| overall {elog['eval/fve/overall']*100:.1f}% | loss {ev_loss:.4f}", flush=True)
            if not args.no_wandb:
                import wandb
                wandb.log(elog, step=step)

        if (step + 1) % args.save_every == 0 or (step + 1) == args.num_steps:
            out = save_dir / f"iter_{step + 1:07d}"
            out.mkdir(parents=True, exist_ok=True)
            from safetensors.torch import save_file
            import json
            sd = {n: p_.detach().cpu().contiguous() for n, p_ in model.named_parameters()
                  if ("lora_" in n) or n.startswith("heads.")}
            save_file(sd, str(out / "ar_multitap.safetensors"))
            (out / "ar_meta.json").write_text(json.dumps({
                "tap_layers": list(tap_layers), "target_cols": list(target_cols),
                "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                "quant": args.quant, "strip_final_norm": args.strip_final_norm,
                "mse_scale": mse_scale, "d_model": d_model,
                "target_modules": lora_target_modules,
            }, indent=2))
            tokenizer.save_pretrained(str(out))
            print(f"[save] -> {out}")
    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
