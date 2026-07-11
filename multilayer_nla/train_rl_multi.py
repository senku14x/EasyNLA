"""Multi-layer RL (GRPO) — k-slot av_in rollout + fixed-target reward (plan §6.3, §8).

DECOUPLED (k-general, like evaluate_e2e): the AV INPUT is the parquet's `av_in_*` slots
(k∈{1,2,3}, auto-detected — so ANY condition/combo RLs correctly), and the AR TARGET is the
fixed activation_prev/centre/next sliced to the AR ckpt's `tap_layers`. The injection count
follows the data (k markers); the reconstruction target follows the AR checkpoint.

Extends the self-contained single-GPU GRPO trainer (nla/train_rl_self_contained)
to the coherent-patch NLA:
  - Actor: base + AV-SFT LoRA ("default", trainable) + a FROZEN "reference"
    adapter loaded from the SAME AV-SFT ckpt. KL is taken against that reference
    via set_adapter("reference") — NOT disable_adapter() (Fix 4, plan §6.3): the
    policy is a LoRA, so disabling it would anchor KL to pre-SFT Qwen and undo
    AV-SFT. Because default == reference at init, step-0 KL == 0 (asserted).
  - Critic: MultiTapCriticModel (AR-LoRA + 3 heads), frozen eval (optional co-train).
  - Reward = three_target_reward = -(1/3d) Σ_j ||û^j - u^j||^2  (TASK REWARD ONLY;
    no auxiliary losses — plan §6.3). KL is an optimization constraint, β=0.01.

The GRPO surrogate math lives in `grpo_surrogate` (pure, unit-tested incl. the
KL==0-when-reference-equals-actor property); peft/bnb/wandb are lazy so the math
is testable without them.
"""

import argparse
import math
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nla.schema import compute_predict_mean_baselines, normalize_activation
from multilayer_nla.datasets import (
    AR_LAYER_TO_TARGET_COL,
    AR_TARGET_COL_TO_NAME,
    AR_TARGET_COLUMNS,
    apply_chat_template_no_think,
    build_av_prompt,
    detect_av_slots,
    fill_ar_prompt,
)
from multilayer_nla.injection_multi import register_multislot_hook
from multilayer_nla.models_multi import (
    multitap_predict,
    three_target_reward,
)

FAILED_EXTRACTION_REWARD = -2.0  # orthogonal-equivalent worst case under √d-normalized MSE
# AV INPUT = av_in_* slots (k∈{1,2,3}, detected per parquet); AR TARGET = the fixed
# activation_prev/centre/next, sliced to the critic's taps. Decoupled exactly like evaluate_e2e,
# so ANY condition/combo RLs correctly (inject k av_in vectors; reconstruct the fixed target).


def cjk_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if "CJK" in unicodedata.name(c, "")) / len(text)


# ---- GRPO math (pure, unit-tested) ----

def compute_group_advantages(rewards: torch.Tensor, groups: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Group-relative advantage A_ij = (r_ij - mean_j) / (std_j + eps), per prompt."""
    adv = torch.zeros_like(rewards)
    for g in range(n_groups):
        mask = groups == g
        if mask.sum() == 0:
            continue
        gr = rewards[mask]
        mu = gr.mean()
        sd = gr.std() if gr.numel() > 1 else torch.tensor(1.0, device=rewards.device)
        adv[mask] = (gr - mu) / (sd + 1e-6)
    return adv


def grpo_surrogate(new_lp, old_lp, ref_lp, advantage, clip_eps=0.2, kl_beta=0.01):
    """Per-sample GRPO loss: clipped surrogate + β·KL(k3) against the reference.

    KL uses δ = ref_lp - new_lp, kl = exp(δ) - δ - 1 (≥0). When the reference
    equals the actor (ref_lp == new_lp), δ == 0 -> kl == 0 — this is the Fix-4
    step-0 property: actor initialized from the AV-SFT reference => KL == 0.

    Returns (loss_scalar, kl_mean_detached, clip_frac).
    """
    old_lp = old_lp.detach()
    ref_lp = ref_lp.detach()
    ratio = torch.exp(new_lp - old_lp)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    delta = ref_lp - new_lp
    kl = torch.exp(delta) - delta - 1.0
    per_tok = -(surrogate - kl_beta * kl)
    clip_frac = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean()
    return per_tok.mean(), kl.mean().detach(), clip_frac.detach()


# ---- data ----

def load_rl_dataset_multi(parquet_path, n_max=None):
    """Rows with av_in (acts [k,d] = the INJECTED input) + gold ([3,d] = the fixed
    L23/24/25 target). k = number of `av_in_*` slots (detected). Mirrors
    evaluate_e2e.load_eval_rows: RL injects exactly what the AV was trained on and
    reconstructs the fixed target — decoupled, so any condition/combo RLs correctly.
    Returns (rows, k)."""
    import pyarrow.parquet as pq
    slot_cols = detect_av_slots(parquet_path)          # av_in_* in slot order (k of them)
    cols = [*slot_cols, *AR_TARGET_COLUMNS]
    pf = pq.ParquetFile(parquet_path)
    rows = []
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        take = rg.num_rows if n_max is None else min(n_max - len(rows), rg.num_rows)
        rg = rg.slice(0, take)

        def to_np(name):
            col = rg.column(name).combine_chunks()
            return (col.flatten().to_numpy(zero_copy_only=False)
                    .astype(np.float32).reshape(len(col), -1))

        av = {c: to_np(c) for c in slot_cols}
        gd = {c: to_np(c) for c in AR_TARGET_COLUMNS}
        for i in range(take):
            rows.append({"acts": np.stack([av[c][i] for c in slot_cols]),          # [k, d] av_in (inject)
                         "gold": np.stack([gd[c][i] for c in AR_TARGET_COLUMNS])})  # [3, d] fixed target
    return rows, len(slot_cols)


def build_prompt_text(k, inject_char, tokenizer, prompt_msgs=None):
    """The k-marker AV prompt with the injection placeholder swapped for the marker char.

    prompt_msgs: the rl parquet's STORED prompt (datasets.load_stored_prompt) — pass it
    so RL rolls out with exactly the prompt the AV was SFT'd on, for ANY k / template
    (permutation-grid conditions use the neutral template, which the legacy k-keyed
    build_av_prompt cannot reconstruct). None falls back to the legacy template.
    enable_thinking=False — identical to AV-SFT prep, or the actor sees a different
    prompt than it learned (and Qwen3 burns the budget on <think>)."""
    from nla.schema import INJECT_PLACEHOLDER
    msgs = [
        {**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inject_char)}
        if isinstance(m.get("content"), str) else m
        for m in (prompt_msgs if prompt_msgs is not None else build_av_prompt(k))
    ]
    return apply_chat_template_no_think(tokenizer, msgs)


@torch.no_grad()
def rollout_multislot(actor, tokenizer, prompt_text, acts_k, vectors_ref, inj_id,
                      group_size, max_new_tokens, temperature, device, eos_ids):
    """Generate group_size samples; inject the SAME k raw av_in activations at the k
    markers of every sample. acts_k: [k, d] (k = the AV's av_in slot count). Returns
    per-sample dicts."""
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    plen = prompt_t.shape[1]
    batched = prompt_t.expand(group_size, -1).contiguous()
    # [G, k, d] -> [G*k, d] example-major: matches the row-major k-marker scan.
    v = torch.as_tensor(acts_k, dtype=torch.float32, device=device)  # accepts list/ndarray/tensor
    k = v.shape[0]
    v_batch = v.unsqueeze(0).expand(group_size, -1, -1).reshape(group_size * k, -1).contiguous()
    # Injection is bounded to the prompt span (prompt_lens). generate() injects only on
    # the prefill (the prompt; response tokens are cache steps where the hook no-ops),
    # so the whole prefill is the prompt. We deliberately do NOT suppress the marker
    # during sampling: that renormalizes the sampled token's probability and would
    # desync old/new/ref log-probs in GRPO. A marker the actor emits in the response is
    # simply not an injection site (prompt-bounded), and a degenerate marker-spamming
    # response just earns poor reward.
    vectors_ref[0] = {"vectors": v_batch,
                      "prompt_lens": torch.full((group_size,), plen, dtype=torch.long, device=device)}
    try:
        gen = actor.generate(
            input_ids=batched, attention_mask=torch.ones_like(batched),
            max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
            top_p=1.0, top_k=0, repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True, output_logits=True,
        )
    finally:
        vectors_ref[0] = None
    full_ids, scores = gen.sequences, gen.logits
    out = []
    for g in range(group_size):
        resp_ids = full_ids[g, plen:].tolist()
        n_real = next((i + 1 for i, t in enumerate(resp_ids) if t in eos_ids), len(resp_ids))
        resp_ids = resp_ids[:n_real]
        old_logp = []
        for t, step_logits in enumerate(scores):
            if t >= n_real:
                break
            old_logp.append(F.log_softmax(step_logits[g].float(), -1)[resp_ids[t]].item())
        out.append({
            "text": tokenizer.decode(resp_ids, skip_special_tokens=True),
            "full_ids": full_ids[g, : plen + n_real], "prompt_len": plen,
            "old_logp": torch.tensor(old_logp, dtype=torch.float32),
        })
    return out


def score_with_multitap_critic(critic, tokenizer, explanations, golds, mse_scale, device, max_len=1024):
    """Reward per sample. None for failed extraction / over-length critic prompt."""
    rewards = []
    for expl, gold in zip(explanations, golds):
        if expl is None:
            rewards.append(None)
            continue
        ids = tokenizer.encode(fill_ar_prompt(expl), add_special_tokens=False)
        if not 0 < len(ids) <= max_len:
            rewards.append(None)
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = multitap_predict(critic, x, None, mse_scale)  # [1, k, d]
        g = torch.as_tensor(gold, dtype=torch.float32, device=device).unsqueeze(0)  # [1, k, d]
        r = three_target_reward(pred, g, mse_scale).item()
        rewards.append(r if math.isfinite(r) else None)
    return rewards


@torch.no_grad()
def greedy_eval(actor, tokenizer, critic, eval_rows, eval_baselines,
                mse_scale, inject_char, vectors_ref, eos_ids, device,
                max_new_tokens=150, batch_size=32, prompt_text=None):
    """Held-out GREEDY eval (do_sample=False) on the 'default' policy → the 'real' number.

    Same pipeline as the training rollout but greedy: inject the k av_in slots → AV greedy-gen →
    extract → frozen AR → FVE vs the fixed target (rows carry pre-sliced 'gold'). The training
    loop logs SAMPLED (T=1) FVE, which runs ~10pp BELOW this; watch THIS climb. Baselines are
    the held-out split's own. `prompt_text` (from main) is the SAME rendered prompt the
    rollout uses (stored-in-parquet when available). Returns success-only +
    failure-penalized FVE + extraction rate.
    """
    from nla.schema import extract_explanation
    was_training = actor.training
    actor.eval()
    k = eval_rows[0]["acts"].shape[0]
    if prompt_text is None:
        prompt_text = build_prompt_text(k, inject_char, tokenizer)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    rewards = []  # per-row three_target_reward (= -MSE) or None (failed extraction)
    b = float(np.mean(eval_baselines))
    for cs in range(0, len(eval_rows), batch_size):
        chunk = eval_rows[cs:cs + batch_size]
        B = len(chunk)
        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device).expand(B, -1).contiguous()
        plen = prompt_t.shape[1]
        acts_bk = np.stack([r["acts"] for r in chunk])                              # [B, k, d]
        v_batch = torch.as_tensor(acts_bk, dtype=torch.float32, device=device).reshape(B * k, -1)
        vectors_ref[0] = {"vectors": v_batch,
                          "prompt_lens": torch.full((B,), plen, dtype=torch.long, device=device)}
        try:
            gen = actor.generate(
                input_ids=prompt_t, attention_mask=torch.ones_like(prompt_t),
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True)
        finally:
            vectors_ref[0] = None
        seqs = gen.sequences
        expls, golds = [], []
        for r_ in range(B):
            resp_ids = seqs[r_, plen:].tolist()
            n_real = next((i + 1 for i, t in enumerate(resp_ids) if t in eos_ids), len(resp_ids))
            text = tokenizer.decode(resp_ids[:n_real], skip_special_tokens=True)
            e = extract_explanation(text)
            expls.append(e if (e and e.strip()) else None)
            golds.append(chunk[r_]["gold"])          # fixed target, pre-sliced to the AR's taps in main()
        rewards.extend(score_with_multitap_critic(critic, tokenizer, expls, golds, mse_scale, device))
    if was_training:
        actor.train()
    valid = [r for r in rewards if r is not None]
    succ_fve = (1.0 - (-float(np.mean(valid))) / b) if valid else float("nan")
    pen = [r if r is not None else -b for r in rewards]  # fail = predict-the-mean = per-row FVE 0
    pen_fve = (1.0 - (-float(np.mean(pen))) / b) if rewards else float("nan")
    return {"fve_greedy": succ_fve, "pen_fve_greedy": pen_fve,
            "extraction_rate": len(valid) / max(len(rewards), 1), "n": len(rewards)}


def _grpo_update(actor, optim, tokenizer, full_ids_list, prompt_lens, acts_list,
                 old_logps, advantages, vectors_ref, device, micro_batch,
                 clip_eps, kl_beta, max_grad_norm):
    """Fused micro-batched forward+loss+backward. new_lp via "default" adapter,
    ref_lp via "reference" (= frozen AV-SFT) — Fix 4. Returns (loss, grad_norm, kl, clip)."""
    optim.zero_grad()
    n = len(full_ids_list)
    losses, kls, clips = [], [], []
    advantages = advantages.detach()
    for cs in range(0, n, micro_batch):
        idxs = list(range(cs, min(cs + micro_batch, n)))
        bs = len(idxs)
        T = max(full_ids_list[i].numel() for i in idxs)
        pad = tokenizer.eos_token_id
        batch_ids = torch.full((bs, T), pad, dtype=torch.long, device=device)
        attn = torch.zeros((bs, T), dtype=torch.long, device=device)
        for r, i in enumerate(idxs):
            L = full_ids_list[i].numel()
            batch_ids[r, :L] = full_ids_list[i].to(device)
            attn[r, :L] = 1
        # vectors: each row's k av_in acts -> [bs*k, d] example-major
        v = torch.stack([torch.as_tensor(acts_list[i], dtype=torch.float32, device=device) for i in idxs])
        v_batch = v.reshape(bs * v.shape[1], -1)   # v: [bs, k, d]
        # Bound injection to each row's prompt span: this forward is over prompt+response,
        # but rollout injected ONLY at the prompt prefill — markers in the generated
        # response are NOT injection sites (and would otherwise trip the count guard).
        plens_t = torch.tensor([prompt_lens[i] for i in idxs], dtype=torch.long, device=device)
        payload = {"vectors": v_batch, "prompt_lens": plens_t}

        vectors_ref[0] = payload
        try:
            new_logits = actor(input_ids=batch_ids, attention_mask=attn).logits
        finally:
            vectors_ref[0] = None
        new_logp = F.log_softmax(new_logits.float(), -1)
        vectors_ref[0] = payload
        try:
            with torch.no_grad():
                actor.set_adapter("reference")
                ref_logits = actor(input_ids=batch_ids, attention_mask=attn).logits
        finally:
            actor.set_adapter("default")
            vectors_ref[0] = None
        ref_logp = F.log_softmax(ref_logits.float(), -1)
        del ref_logits

        chunk_losses = []
        for r, i in enumerate(idxs):
            L = full_ids_list[i].numel()
            p_len = prompt_lens[i]
            if L <= p_len:
                continue
            tgt = batch_ids[r, p_len:L]
            pidx = torch.arange(p_len - 1, L - 1, device=device)
            nlp = new_logp[r].index_select(0, pidx).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            rlp = ref_logp[r].index_select(0, pidx).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            olp = old_logps[i].to(device)
            if nlp.numel() == 0 or olp.numel() != nlp.numel():
                continue
            sl, kl, cf = grpo_surrogate(nlp, olp, rlp, advantages[i], clip_eps, kl_beta)
            chunk_losses.append(sl)
            kls.append(kl.item())
            clips.append(cf.item())
        del new_logits, ref_logp
        if not chunk_losses:
            del new_logp
            continue
        cl = torch.stack(chunk_losses).sum() / n
        cl.backward()
        losses.append(cl.item() * n / len(chunk_losses))
        del new_logp
    gn = torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], max_grad_norm)
    gn = gn.item() if hasattr(gn, "item") else float(gn)
    if math.isfinite(gn):
        optim.step()
    else:
        optim.zero_grad(set_to_none=True)
        print(f"[grpo] non-finite grad ({gn}) — skip step", flush=True)
    return (float(np.mean(losses)) if losses else 0.0, gn,
            float(np.mean(kls)) if kls else 0.0, float(np.mean(clips)) if clips else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", required=True, help="AV-SFT LoRA dir (policy init + frozen KL reference)")
    p.add_argument("--ar-ckpt", required=True,
                   help="AR multitap dir (ar_multitap.safetensors + ar_meta.json). The AR's tap_layers "
                        "set the reconstruction target; the AV input is the parquet's av_in_* slots (k).")
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B")
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--rl-parquet", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-steps", type=int, default=500)
    p.add_argument("--batch-prompts", type=int, default=16)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.01)  # plan §6.3 Fix 4
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--logp-micro-batch", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--eval-parquet", default=None,
                   help="held-out DEV shard (rl_dev_<cond>.parquet, same columns as --rl-parquet) for "
                        "the periodic GREEDY eval = the 'real' FVE. Omit to skip (log sampled FVE only).")
    p.add_argument("--eval-every", type=int, default=50, help="greedy held-out eval cadence (steps; includes step 0)")
    p.add_argument("--eval-rows", type=int, default=256, help="cap held-out eval rows (speed vs noise)")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--no-kl-step0-check", action="store_true",
                   help="skip the Fix-4 assertion that step-0 KL≈0 (only when intentionally resuming)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="mlnla-qwen3-8b")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    assert args.temperature == 1.0, "old_logp comes from raw logits; GRPO ratio is only valid at T=1"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    from nla.datagen.injection_tokens import find_injection_token
    inject_char, inj_id = find_injection_token(tokenizer)

    quant_config = None
    if args.quant == "4bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        quantization_config=quant_config, device_map=({"": 0} if quant_config else None))
    if quant_config is None:
        base = base.to(device)
    from peft import PeftModel, prepare_model_for_kbit_training
    if quant_config is not None:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    # Fix 4: policy "default" = AV-SFT (trainable); "reference" = SAME AV-SFT, frozen.
    actor = PeftModel.from_pretrained(base, args.av_ckpt, adapter_name="default", is_trainable=True)
    actor.load_adapter(args.av_ckpt, adapter_name="reference")
    actor.set_adapter("default")
    for n_, p_ in actor.named_parameters():
        if ".reference." in n_:
            p_.requires_grad_(False)
    actor.train()

    vectors_ref = [None]
    k = len(detect_av_slots(args.rl_parquet))   # av_in_* slot count = markers injected per example
    register_multislot_hook(actor, vectors_ref, inj_id, k, layer_idx=1)
    print(f"[rl] k={k} av_in slot(s) injected per example (from {Path(args.rl_parquet).name})")
    eos_ids = {tokenizer.eos_token_id}
    _gc = getattr(getattr(actor, "generation_config", None), "eos_token_id", None)
    if _gc is not None:
        eos_ids.update(_gc if isinstance(_gc, (list, tuple)) else [_gc])
    eos_ids.discard(None)

    # ---- critic: multitap AR (AR-LoRA + 3 heads) ----
    import json
    from safetensors.torch import load_file
    from peft import LoraConfig, inject_adapter_in_model
    from multilayer_nla.models_multi import init_multitap_critic_from_base
    ar_meta = json.loads((Path(args.ar_ckpt) / "ar_meta.json").read_text())
    tap_layers = tuple(ar_meta["tap_layers"])
    mse_scale = ar_meta.get("mse_scale") or math.sqrt(ar_meta["d_model"])
    crit_quant = quant_config if ar_meta.get("quant") == "4bit" else None
    critic = init_multitap_critic_from_base(
        args.base_ckpt, tap_layers, torch.bfloat16, crit_quant,
        device_map=({"": 0} if crit_quant else None),
        strip_final_norm=ar_meta.get("strip_final_norm", True))
    if crit_quant is None:
        critic = critic.to(device)
    inject_adapter_in_model(LoraConfig(
        r=ar_meta["lora_r"], lora_alpha=ar_meta["lora_alpha"], lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", use_rslora=True, target_modules=ar_meta["target_modules"]), critic.backbone)
    _sd = load_file(str(Path(args.ar_ckpt) / "ar_multitap.safetensors"))
    miss, unexp = critic.load_state_dict(_sd, strict=False)
    assert not unexp, f"AR load: unexpected keys {unexp[:3]}"
    for p_ in critic.parameters():
        p_.requires_grad_(False)
    critic.eval()
    print(f"[rl] critic taps={tap_layers} mse_scale={mse_scale:.3f}; loaded {len(_sd)} AR tensors")

    # AR TARGET = the fixed activation_prev/centre/next, sliced to the critic's taps (tap layer ->
    # target column via AR_LAYER_TO_TARGET_COL). Independent of the AV INPUT (the k av_in slots) —
    # exactly like evaluate_e2e, so any condition/combo RLs with the right target.
    tap_cols = [AR_LAYER_TO_TARGET_COL[l] for l in tap_layers]
    tap_idx = [AR_TARGET_COLUMNS.index(tc) for tc in tap_cols]
    tap_names = [AR_TARGET_COL_TO_NAME[tc] for tc in tap_cols]
    print(f"[rl] AR reconstructs {tap_names} (= layers {list(tap_layers)}) from k={k} av_in input")

    rows, k2 = load_rl_dataset_multi(args.rl_parquet, n_max=args.max_rows)
    assert k2 == k, f"detected k={k} for the hook but the loader returned k={k2}"
    for r in rows:
        r["gold"] = r["gold"][tap_idx]     # [n_tap, d] fixed target aligned to the AR's taps
    # predict-mean FVE baselines over the TARGET taps only (so fve/overall matches the reward)
    baselines = []
    for j in range(len(tap_idx)):
        g = torch.tensor(np.stack([r["gold"][j] for r in rows[:4000]]), dtype=torch.float32)
        _, rawvar = compute_predict_mean_baselines(g, mse_scale)
        baselines.append(rawvar)
    print(f"[rl] {len(rows)} rows; per-target baselines " +
          ", ".join(f"{nm}={b:.4f}" for nm, b in zip(tap_names, baselines)))

    # held-out GREEDY eval set (the 'real' FVE — the training log is SAMPLED, ~10pp lower).
    eval_rows, eval_baselines = None, None
    if args.eval_parquet:
        eval_rows, eval_k = load_rl_dataset_multi(args.eval_parquet, n_max=args.eval_rows)
        assert eval_k == k, f"eval parquet has k={eval_k} av_in slots but train has k={k}"
        for r in eval_rows:
            r["gold"] = r["gold"][tap_idx]
        eval_baselines = []
        for j in range(len(tap_idx)):  # predict-the-mean baselines from the EVAL split's targets only
            g = torch.tensor(np.stack([r["gold"][j] for r in eval_rows]), dtype=torch.float32)
            _, rv = compute_predict_mean_baselines(g, mse_scale)
            eval_baselines.append(rv)
        print(f"[rl] greedy held-out eval: {len(eval_rows)} rows from {Path(args.eval_parquet).name}, "
              f"every {args.eval_every} steps (metrics logged as eval/*)")

    try:
        import bitsandbytes as bnb
        optim = bnb.optim.AdamW8bit([p for p in actor.parameters() if p.requires_grad],
                                    lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    except ImportError:
        optim = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],
                                  lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    cursor = 0

    from nla.schema import extract_explanation
    from multilayer_nla.datasets import load_stored_prompt
    # Prefer the rl parquet's STORED prompt (permutation-grid conditions carry the
    # neutral template); fall back to the legacy k-keyed template for sweep parquets.
    stored_msgs = load_stored_prompt(args.rl_parquet)
    prompt_text = build_prompt_text(k, inject_char, tokenizer, prompt_msgs=stored_msgs)
    print(f"[rl] rollout prompt: {'stored-in-parquet' if stored_msgs else f'legacy build_av_prompt({k})'}")

    for step in range(args.num_steps):
        t0 = time.time()
        if cursor + args.batch_prompts > len(order):
            rng.shuffle(order); cursor = 0
        batch = order[cursor:cursor + args.batch_prompts]
        cursor += args.batch_prompts

        actor.eval()
        full_ids, plens, acts_l, expls, texts, groups, old_lps, golds = [], [], [], [], [], [], [], []
        for gi, ri in enumerate(batch):
            row = rows[ri]
            resp = rollout_multislot(actor, tokenizer, prompt_text, row["acts"], vectors_ref, inj_id,
                                     args.group_size, args.max_new_tokens, args.temperature, device, eos_ids)
            for r in resp:
                full_ids.append(r["full_ids"]); plens.append(r["prompt_len"]); acts_l.append(row["acts"])
                expls.append(extract_explanation(r["text"])); texts.append(r["text"])
                groups.append(gi); old_lps.append(r["old_logp"].to(device))
                golds.append(row["gold"])  # AR target = fixed prev/centre/next (sliced to taps); input = k av_in

        rewards = score_with_multitap_critic(critic, tokenizer, expls, golds, mse_scale, device)
        valid = [r for r in rewards if r is not None]
        rfilled = [FAILED_EXTRACTION_REWARD if r is None else r for r in rewards]
        rt = torch.tensor(rfilled, dtype=torch.float32, device=device)
        adv = compute_group_advantages(rt, torch.tensor(groups, device=device), args.batch_prompts)

        actor.train()
        loss, gn, kl, clip = _grpo_update(
            actor, optim, tokenizer, full_ids, plens, acts_l, old_lps, adv, vectors_ref, device,
            args.logp_micro_batch, args.clip_eps, args.kl_beta, args.max_grad_norm)

        # Fix-4 step-0 guard: actor init == reference => KL must be ~0.
        if step == 0 and not args.no_kl_step0_check:
            assert abs(kl) < 1e-2, (
                f"step-0 KL={kl:.4f} != 0 — the KL reference is NOT the AV-SFT init "
                f"(likely fell back to LoRA-disabled base). Fix the reference adapter (plan §6.3 Fix 4)."
            )
            print(f"[rl] Fix-4 OK: step-0 KL={kl:.2e} ≈ 0 (reference == AV-SFT init)")

        mean_reward = float(np.mean(valid)) if valid else float("nan")
        fve_overall = 1.0 - (-mean_reward) / float(np.mean(baselines)) if valid else float("nan")
        mean_cjk = sum(cjk_fraction(t) for t in texts) / max(len(texts), 1)
        log = {
            "step": step, "loss": loss, "grad_norm": gn, "kl_mean": kl, "clip_frac": clip,
            "reward_mean": mean_reward, "fve/overall": fve_overall,
            "extraction_rate": len(valid) / max(len(rewards), 1), "mean_cjk": mean_cjk,
            "advantage_mean": adv.mean().item(), "wall_s": time.time() - t0,
        }
        print(f"step {step:04d} | loss {loss:.4f} | r {mean_reward:.3f} | FVE {fve_overall*100:.1f}% "
              f"| kl {kl:.4f} | clip {clip:.2%} | ext {log['extraction_rate']:.0%} | "
              f"cjk {mean_cjk:.3f} | t {log['wall_s']:.0f}s", flush=True)
        if texts:  # show a sample rollout so extraction failures / CJK are visible
            _s0 = texts[0][:200].replace("\n", " ")
            print(f"    [sample extracted={expls[0] is not None}] {_s0!r}", flush=True)

        # periodic GREEDY held-out eval — the number to actually watch (train FVE above is SAMPLED).
        if eval_rows is not None and (step % args.eval_every == 0 or (step + 1) == args.num_steps):
            ev = greedy_eval(actor, tokenizer, critic, eval_rows, eval_baselines,
                             mse_scale, inject_char, vectors_ref, eos_ids, device,
                             max_new_tokens=args.max_new_tokens, prompt_text=prompt_text)
            log["eval/fve_greedy"] = ev["fve_greedy"]
            log["eval/pen_fve_greedy"] = ev["pen_fve_greedy"]
            log["eval/extraction_rate"] = ev["extraction_rate"]
            print(f"    [GREEDY held-out] FVE {ev['fve_greedy']*100:.1f}% "
                  f"(pen {ev['pen_fve_greedy']*100:.1f}%) | ext {ev['extraction_rate']:.0%} "
                  f"| n={ev['n']}  <- the real number; sampled FVE above runs ~10pp lower", flush=True)

        if not args.no_wandb:
            import wandb
            wandb.log(log, step=step)

        if (step + 1) % args.save_every == 0 or (step + 1) == args.num_steps:
            out = save_dir / f"iter_{step + 1:06d}"
            out.mkdir(parents=True, exist_ok=True)
            actor.save_pretrained(str(out))
            print(f"[save] -> {out}")
    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
