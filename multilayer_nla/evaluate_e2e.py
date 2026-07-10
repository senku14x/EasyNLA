"""End-to-end evaluator for the §7 SFT control sweep (no training, no RL update).

For one condition's checkpoints on a held-out rl_<bucket>_<cond>.parquet:

    av_in_* activations
      -> AV generation (greedy, injection active)
      -> explanation extraction (<explanation>...</explanation>)
      -> shared AR reconstruction
      -> FVE against the FIXED targets [L23,L24,L25] (activation_prev/centre/next)

The AV input has k slots (k=3 local/duplicate/wide, k=1 single); the AR target is the
same fixed triplet for every condition. Predict-the-mean baselines are computed from
THIS eval split's targets only (never AR training rows).

Reports (JSON summary + per-example JSONL):
  fve_prev/centre/next, fve_overall                          (successful extractions only)
  pen_fve_prev/centre/next, pen_fve_overall                  (failures := mean-predictor, FVE 0)
  successful_extraction_rate, failed_generation_count
  mean_generated_tokens, median_generated_tokens
  bootstrap CIs on fve_overall + pen_fve_overall (resampling DOCUMENTS, not rows)
  shuffled_pen_fve_overall                                   (gens permuted across docs -> ~0)

Run:
  python -m multilayer_nla.evaluate_e2e --base-ckpt Qwen/Qwen3-8B \
      --av-ckpt $CKPT/av_local/iter_0001000 --ar-ckpt $CKPT/ar/iter_0001000 \
      --eval-parquet $SWEEP/rl_dev_local.parquet --condition local \
      --out $EVAL/dev_local.jsonl --summary $EVAL/dev_local.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from nla.schema import (
    compute_predict_mean_baselines,
    extract_explanation,
    normalize_activation,
)
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
from multilayer_nla.models_multi import init_multitap_critic_from_base, multitap_predict

SLOT_NAMES = ("prev", "centre", "next")


# ----------------------------------------------------------------- data

def load_eval_rows(parquet_path):
    """Rows with av_in (acts [k,d]), gold ([3,d] = L23/24/25), doc_id, src_row_id.

    `src_row_id` is written by build_sweep.py (per-condition eval sets); a
    build_from_published rl.parquet lacks it — fall back to the global row index
    so the per-example JSONL stays joinable to the source parquet either way.
    """
    slot_cols = detect_av_slots(parquet_path)
    pf = pq.ParquetFile(parquet_path)
    has_src = "src_row_id" in pf.schema_arrow.names
    cols = ["doc_id", *(["src_row_id"] if has_src else []), *slot_cols, *AR_TARGET_COLUMNS]
    rows = []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)

        def to_np(name):
            c = t.column(name).combine_chunks()
            return c.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(c), -1)

        av = {c: to_np(c) for c in slot_cols}
        gd = {c: to_np(c) for c in AR_TARGET_COLUMNS}
        dids = t.column("doc_id").to_pylist()
        srcs = (t.column("src_row_id").to_pylist() if has_src
                else list(range(len(rows), len(rows) + t.num_rows)))
        for i in range(t.num_rows):
            rows.append({
                "doc_id": dids[i],
                "src_row_id": int(srcs[i]),
                "acts": np.stack([av[c][i] for c in slot_cols]),      # [k, d]
                "gold": np.stack([gd[c][i] for c in AR_TARGET_COLUMNS]),  # [3, d]
            })
    return rows, len(slot_cols)


# ----------------------------------------------------------------- models

def load_actor(base_ckpt, av_ckpt, k, quant, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.datagen.injection_tokens import find_injection_token
    tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
    inject_char, inj_id = find_injection_token(tokenizer)
    quant_config = _quant_cfg(quant)
    base = AutoModelForCausalLM.from_pretrained(
        base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        quantization_config=quant_config, device_map=({"": 0} if quant_config else None))
    if quant_config is None:
        base = base.to(device)
    from peft import PeftModel
    actor = PeftModel.from_pretrained(base, av_ckpt, adapter_name="default", is_trainable=False)
    actor.set_adapter("default")
    actor.eval()
    vectors_ref = [None]
    register_multislot_hook(actor, vectors_ref, inj_id, k, layer_idx=1)
    eos_ids = {tokenizer.eos_token_id}
    _gc = getattr(getattr(actor, "generation_config", None), "eos_token_id", None)
    if _gc is not None:
        eos_ids.update(_gc if isinstance(_gc, (list, tuple)) else [_gc])
    eos_ids.discard(None)
    return actor, tokenizer, inject_char, inj_id, vectors_ref, eos_ids


def load_critic(base_ckpt, ar_ckpt, quant, device):
    import json as _json
    from safetensors.torch import load_file
    from peft import LoraConfig, inject_adapter_in_model
    ar_meta = _json.loads((Path(ar_ckpt) / "ar_meta.json").read_text())
    tap_layers = tuple(ar_meta["tap_layers"])
    mse_scale = ar_meta.get("mse_scale") or math.sqrt(ar_meta["d_model"])
    quant_config = _quant_cfg(quant) if ar_meta.get("quant") == "4bit" else None
    critic = init_multitap_critic_from_base(
        base_ckpt, tap_layers, torch.bfloat16, quant_config,
        device_map=({"": 0} if quant_config else None),
        strip_final_norm=ar_meta.get("strip_final_norm", True))
    if quant_config is None:
        critic = critic.to(device)
    inject_adapter_in_model(LoraConfig(
        r=ar_meta["lora_r"], lora_alpha=ar_meta["lora_alpha"], lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", use_rslora=True, target_modules=ar_meta["target_modules"]), critic.backbone)
    sd = load_file(str(Path(ar_ckpt) / "ar_multitap.safetensors"))
    _, unexp = critic.load_state_dict(sd, strict=False)
    assert not unexp, f"AR load: unexpected keys {unexp[:3]}"
    for p_ in critic.parameters():
        p_.requires_grad_(False)
    critic.eval()
    return critic, mse_scale


def _quant_cfg(quant):
    if quant != "4bit":
        return None
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_storage=torch.bfloat16)


# ----------------------------------------------------------------- generate + score

@torch.no_grad()
def generate_batch(actor, tokenizer, prompt_text, acts_bk, vectors_ref, eos_ids, device, max_new_tokens):
    """One greedy generation per row. acts_bk: [B,k,d] injected example-major at the
    B identical k-marker prompts. Returns (texts[B], n_tokens[B])."""
    B, k = acts_bk.shape[0], acts_bk.shape[1]
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device).expand(B, -1).contiguous()
    plen = prompt_t.shape[1]
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
    texts, lens = [], []
    for b in range(B):
        resp_ids = seqs[b, plen:].tolist()
        n_real = next((i + 1 for i, t in enumerate(resp_ids) if t in eos_ids), len(resp_ids))
        resp_ids = resp_ids[:n_real]
        texts.append(tokenizer.decode(resp_ids, skip_special_tokens=True))
        lens.append(len(resp_ids))
    return texts, lens


@torch.no_grad()
def ar_sqerr(critic, tokenizer, expl, gold, mse_scale, device, max_len=1024):
    """Per-tap √d-normalized squared error [3] for one explanation vs gold[3,d].
    None if extraction failed (expl is None) or the critic prompt is over-length."""
    if expl is None:
        return None
    ids = tokenizer.encode(fill_ar_prompt(expl), add_special_tokens=False)
    if not 0 < len(ids) <= max_len:
        return None
    x = torch.tensor([ids], dtype=torch.long, device=device)
    pred = multitap_predict(critic, x, None, mse_scale)              # [1,3,d] raw
    g = torch.as_tensor(gold, dtype=torch.float32, device=device).unsqueeze(0)
    pn, gn = normalize_activation(pred, mse_scale), normalize_activation(g, mse_scale)
    return ((pn - gn) ** 2).mean(dim=(0, 2)).tolist()               # [3]


@torch.no_grad()
def ar_sqerr_batch(critic, tokenizer, expls, golds, mse_scale, device, max_len=1024, batch_size=64):
    """BATCHED per-tap √d-normalized squared error [3] per row; None for a failed
    extraction (expl is None) or an over-length critic prompt. Right-pads + passes the
    attention mask so multitap_predict taps each row's true last token (it uses
    attn.sum(1)-1). Aligned with the input order. ~B× fewer forwards than per-row."""
    out = [None] * len(expls)
    items = []  # (orig_idx, ids) for scorable rows
    for i, e in enumerate(expls):
        if e is None:
            continue
        ids = tokenizer.encode(fill_ar_prompt(e), add_special_tokens=False)
        if 0 < len(ids) <= max_len:
            items.append((i, ids))
    pad = tokenizer.eos_token_id
    for cs in range(0, len(items), batch_size):
        chunk = items[cs:cs + batch_size]
        T = max(len(ids) for _, ids in chunk)
        bids = torch.full((len(chunk), T), pad, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), T), dtype=torch.long, device=device)
        for r, (_, ids) in enumerate(chunk):
            bids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attn[r, :len(ids)] = 1
        pred = multitap_predict(critic, bids, attn, mse_scale)                       # [bs,k,d]
        g = torch.stack([torch.as_tensor(golds[oi], dtype=torch.float32, device=device)
                         for oi, _ in chunk])                                        # [bs,k,d]
        assert pred.shape[1] == g.shape[1], (
            f"AR tap/target mismatch: pred {tuple(pred.shape)} vs gold {tuple(g.shape)} — gold must be "
            f"sliced to the AR's taps (main() does this); refusing to broadcast.")
        pn, gn = normalize_activation(pred, mse_scale), normalize_activation(g, mse_scale)
        se = ((pn - gn) ** 2).mean(dim=2)                                            # [bs,3]
        for r, (oi, _) in enumerate(chunk):
            out[oi] = se[r].tolist()
    return out


# ----------------------------------------------------------------- FVE aggregation

def _fve(mse_per_tap, baselines):
    fv = [1.0 - m / b for m, b in zip(mse_per_tap, baselines)]
    return fv, sum(fv) / len(fv)


def aggregate(errs, baselines):
    """errs: list of [3] per-row sqerr or None (failed). Returns success-only and
    failure-penalized (failed := baseline, i.e. per-row FVE 0) tap MSE + FVE."""
    ok = [e for e in errs if e is not None]
    n_taps = len(baselines)
    succ_mse = [float(np.mean([e[j] for e in ok])) if ok else float("nan") for j in range(n_taps)]
    pen_rows = [e if e is not None else list(baselines) for e in errs]
    pen_mse = [float(np.mean([e[j] for e in pen_rows])) for j in range(n_taps)]
    succ_fve, succ_ov = _fve(succ_mse, baselines)
    pen_fve, pen_ov = _fve(pen_mse, baselines)
    return {
        "n_total": len(errs), "n_success": len(ok),
        "fve": succ_fve, "fve_overall": succ_ov,
        "pen_fve": pen_fve, "pen_fve_overall": pen_ov,
    }


def bootstrap_overall(errs, doc_ids, baselines, n_boot, seed, penalized):
    """Resample DOCUMENTS (not rows) with replacement; recompute the overall FVE each
    time. Returns (lo, hi) 95% percentile CI. penalized=True -> failures count as 0."""
    by_doc = {}
    for e, d in zip(errs, doc_ids):
        by_doc.setdefault(d, []).append(e)
    docs = list(by_doc.keys())
    if not docs:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_taps = len(baselines)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(docs), size=len(docs), replace=True)
        rows = [e for idx in pick for e in by_doc[docs[idx]]]
        if penalized:
            rows = [e if e is not None else list(baselines) for e in rows]
        else:
            rows = [e for e in rows if e is not None]
        if not rows:
            continue
        mse = [float(np.mean([r[j] for r in rows])) for j in range(n_taps)]
        vals.append(_fve(mse, baselines)[1])
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ----------------------------------------------------------------- driver

def evaluate(actor, tokenizer, critic, inject_char, inj_id, vectors_ref, eos_ids,
             rows, k, mse_scale, device, *, max_new_tokens=150, batch_size=32,
             shuffle_seed=0, prompt_msgs=None):
    """Run the full pipeline; return (per_example list, generated_texts, expls).

    prompt_msgs: the AV chat prompt to generate with. Pass the eval parquet's
    stored prompt (load_stored_prompt) so generation matches AV training for
    ANY k / template; None falls back to the legacy k-keyed build_av_prompt.
    """
    from nla.schema import INJECT_PLACEHOLDER
    if prompt_msgs is None:
        prompt_msgs = build_av_prompt(k)
    prompt_text = apply_chat_template_no_think(
        tokenizer, [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inject_char)}
                    for m in prompt_msgs])

    texts, lens = [], []
    nb = (len(rows) + batch_size - 1) // batch_size
    for bi, cs in enumerate(range(0, len(rows), batch_size)):
        chunk = rows[cs:cs + batch_size]
        acts_bk = np.stack([r["acts"] for r in chunk])  # [B,k,d]
        t, ln = generate_batch(actor, tokenizer, prompt_text, acts_bk, vectors_ref,
                               eos_ids, device, max_new_tokens)
        texts.extend(t); lens.extend(ln)
        if bi % 5 == 0 or bi == nb - 1:
            print(f"    gen batch {bi + 1}/{nb}  ({len(texts)}/{len(rows)} rows)", flush=True)
    print(f"    scoring {len(rows)} explanations (batched)...", flush=True)

    expls = []
    for t in texts:
        e = extract_explanation(t)
        expls.append(e if (e and e.strip()) else None)  # empty/whitespace == failed extraction
    errs = ar_sqerr_batch(critic, tokenizer, expls, [r["gold"] for r in rows],
                          mse_scale, device, batch_size=batch_size)
    return texts, lens, expls, errs


def _doc_derangement(doc_ids, seed):
    """A permutation p with doc_ids[p[i]] != doc_ids[i] wherever possible, so the
    shuffled control pairs each doc's target with ANOTHER document's generation
    (the spec's negative control is across documents, not rows)."""
    n = len(doc_ids)
    perm = list(np.random.default_rng(seed).permutation(n))
    for i in range(n):
        if doc_ids[perm[i]] == doc_ids[i]:
            for j in range(n):
                if doc_ids[perm[j]] != doc_ids[i] and doc_ids[perm[i]] != doc_ids[j]:
                    perm[i], perm[j] = perm[j], perm[i]
                    break
    return perm


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B")
    p.add_argument("--av-ckpt", required=True, help="AV-SFT LoRA dir for this condition")
    p.add_argument("--ar-ckpt", required=True, help="shared AR multitap dir")
    p.add_argument("--eval-parquet", required=True, help="rl_<bucket>_<cond>.parquet")
    p.add_argument("--condition", required=True)
    p.add_argument("--out", required=True, help="per-example JSONL")
    p.add_argument("--summary", required=True, help="summary JSON")
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--shuffle-control", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rows, k = load_eval_rows(args.eval_parquet)
    from multilayer_nla.datasets import load_stored_prompt
    stored_prompt = load_stored_prompt(args.eval_parquet)
    print(f"[eval:{args.condition}] {len(rows)} rows, k={k} AV slots, ar_target=[L23,L24,L25], "
          f"prompt={'stored-in-parquet' if stored_prompt else f'legacy build_av_prompt({k})'}")

    actor, tokenizer, inject_char, inj_id, vectors_ref, eos_ids = load_actor(
        args.base_ckpt, args.av_ckpt, k, args.quant, device)
    critic, mse_scale = load_critic(args.base_ckpt, args.ar_ckpt, args.quant, device)
    # Align the gold targets to the AR's taps: a 3-tap AR scores [L23,L24,L25]; a single-/two-tap
    # AR (e.g. the L24-only reconstructor) scores ONLY its tap(s). load_eval_rows loaded gold as
    # [prev,centre,next]; slice to the AR's taps so a 1-tap pred isn't silently broadcast vs 3 cols.
    tap_cols = [AR_LAYER_TO_TARGET_COL[l] for l in critic.tap_layers]
    tap_idx = [AR_TARGET_COLUMNS.index(tc) for tc in tap_cols]
    tap_names = [AR_TARGET_COL_TO_NAME[tc] for tc in tap_cols]
    for r in rows:
        r["gold"] = r["gold"][tap_idx]          # [k, d] aligned to the AR's taps
    # predict-the-mean baselines from THIS eval split's tapped targets only, at the critic's
    # mse_scale (NEVER from AR training rows).
    baselines = [compute_predict_mean_baselines(
        torch.tensor(np.stack([r["gold"][j] for r in rows]), dtype=torch.float32), mse_scale)[1]
        for j in range(len(tap_idx))]

    texts, lens, expls, errs = evaluate(
        actor, tokenizer, critic, inject_char, inj_id, vectors_ref, eos_ids, rows, k,
        mse_scale, device, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        prompt_msgs=stored_prompt)

    agg = aggregate(errs, baselines)
    doc_ids = [r["doc_id"] for r in rows]
    boot_succ = bootstrap_overall(errs, doc_ids, baselines, args.bootstrap, args.seed, penalized=False)
    boot_pen = bootstrap_overall(errs, doc_ids, baselines, args.bootstrap, args.seed, penalized=True)

    # shuffled control: permute generated explanations across docs; FVE must collapse.
    shuffled_pen = float("nan")
    if args.shuffle_control and len(rows) > 1:
        perm = _doc_derangement([r["doc_id"] for r in rows], args.seed + 1)
        sh_errs = ar_sqerr_batch(critic, tokenizer, [expls[perm[i]] for i in range(len(rows))],
                                 [rows[i]["gold"] for i in range(len(rows))],
                                 mse_scale, device, batch_size=args.batch_size)
        shuffled_pen = aggregate(sh_errs, baselines)["pen_fve_overall"]

    n_failed = sum(1 for e in errs if e is None)
    summary = {
        "condition": args.condition,
        "ar_target_layers": list(critic.tap_layers),
        "av_ckpt": args.av_ckpt, "ar_ckpt": args.ar_ckpt, "eval_parquet": args.eval_parquet,
        "n_total": agg["n_total"], "n_success": agg["n_success"],
        "successful_extraction_rate": agg["n_success"] / max(agg["n_total"], 1),
        "failed_generation_count": n_failed,
        **{f"fve_{nm}": agg["fve"][i] for i, nm in enumerate(tap_names)},
        "fve_overall": agg["fve_overall"],
        **{f"pen_fve_{nm}": agg["pen_fve"][i] for i, nm in enumerate(tap_names)},
        "pen_fve_overall": agg["pen_fve_overall"],
        "fve_overall_ci95": list(boot_succ), "pen_fve_overall_ci95": list(boot_pen),
        "shuffled_pen_fve_overall": shuffled_pen,
        "mean_generated_tokens": float(np.mean(lens)) if lens else 0.0,
        "median_generated_tokens": float(np.median(lens)) if lens else 0.0,
        "baselines": baselines, "mse_scale": mse_scale, "bootstrap": args.bootstrap,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2))

    with open(args.out, "w") as f:
        for r, t, ln, e, err in zip(rows, texts, lens, expls, errs):
            per_tap_fve = (None if err is None
                           else [1.0 - err[j] / baselines[j] for j in range(len(baselines))])
            f.write(json.dumps({
                "doc_id": r["doc_id"], "condition": args.condition, "src_row_id": r["src_row_id"],
                "generated_text": t, "parse_success": e is not None,
                **{f"fve_{nm}": (per_tap_fve[i] if per_tap_fve else None) for i, nm in enumerate(tap_names)},
                "fve_overall": (None if per_tap_fve is None else sum(per_tap_fve) / len(per_tap_fve)),
                "generation_length": ln,  # generated TOKENS (matches mean/median_generated_tokens)
            }) + "\n")

    print(f"[eval:{args.condition}] ext={summary['successful_extraction_rate']:.1%} "
          f"FVE {'/'.join(tap_names)} {'/'.join(f'{v*100:.1f}' for v in agg['fve'])}% "
          f"overall {agg['fve_overall']*100:.1f}% | penalized {agg['pen_fve_overall']*100:.1f}% "
          f"| shuffled {shuffled_pen*100:.1f}% | tok~{summary['mean_generated_tokens']:.0f}")
    print(f"[eval:{args.condition}] summary -> {args.summary}  per-example -> {args.out}")


if __name__ == "__main__":
    main()
