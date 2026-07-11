"""Warmstart improvement without API cost: score / regenerate AV-SFT labels
against the frozen AR.

The verbalizer→gold gap is the dominant bottleneck (§7: the ~4pp condition
spread sits ~20pp under the AR-gold ceiling; §8 recipe closed it to ~9pp).
Two levers, both label-side, both API-free:

  score-gold   Score every EXISTING gold label with the frozen AR against the
               fixed target: reward = -(1/kd)Σ||û-u||² (the RL reward). Writes
               the input + a `gold_ar_reward` column, prints the reward
               distribution, and (with --filter-quantile q) also writes a
               filtered parquet dropping the bottom q of labels. Rationale
               (data curation): labels that the reconstructor cannot map back
               toward the target carry little activation information — SFT on
               them teaches style, not content.

  bon          Best-of-N rejection-sampling self-distillation. For each row:
               sample N explanations from an AV-SFT checkpoint (temperature
               sampling, injection active), score each with the frozen AR
               against the FIXED target, keep the best (gold label included in
               the argmax by default). Writes an av-SFT-format parquet for a
               continuation SFT round. This is ReST-style RL-lite — one GPU,
               no GRPO — and it also DE-LAYER-BLINDS the warmstart: the
               published labels only ever described the single L24 vector,
               but BoN selection scores against the full fixed target, so the
               surviving text is selected for multi-layer information.

Both modes need av parquets that carry the fixed target columns — build with
`build_conditions ... --av-with-targets` (or feed the bank av_sft file to
score-gold with --targets-from-bank L23,L24,L25).

CAUTION (the private-code risk, research-notes §10.1): BoN optimizes text
against the SAME AR that evaluation uses. Held-out e2e FVE gains after
BoN-SFT are only meaningful against the FROZEN AR on held-out docs, and any
faithfulness claim additionally needs the independent-reconstructor check.
Track text quality (nla text_judges / explanation diversity) across rounds —
reward-hacked templates show up there first.

Usage (vast box):
  python -m multilayer_nla.distill_av --mode score-gold \\
      --in $COND/av_lay3.parquet --out $COND/av_lay3_scored.parquet \\
      --base-ckpt Qwen/Qwen3-8B --ar-ckpt $CKPT/ar_3tap/iter_0003000 \\
      --filter-quantile 0.2
  python -m multilayer_nla.distill_av --mode bon \\
      --in $COND/av_lay3.parquet --out $COND/av_lay3_bon8.parquet \\
      --base-ckpt Qwen/Qwen3-8B --av-ckpt $CKPT/av_lay3/iter_0001000 \\
      --ar-ckpt $CKPT/ar_3tap/iter_0003000 --n-samples 8 --max-rows 50000
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nla.schema import extract_explanation, wrap_explanation

REWARD_COL = "gold_ar_reward"


# ------------------------------------------------------------------ pure helpers

def reward_from_taps(per_tap_sqerr) -> float | None:
    """[k] per-tap normalized sqerr -> scalar reward -(mean), or None (failed)."""
    if per_tap_sqerr is None:
        return None
    r = -float(np.mean(per_tap_sqerr))
    return r if math.isfinite(r) else None


def select_best(gold_reward, sample_rewards, *, include_gold: bool = True):
    """Argmax over {gold?} ∪ samples. Returns (source, index, reward) where
    source is 'gold' or 'sample'; index is the winning sample index (or -1 for
    gold). Failed candidates (None) never win. Falls back to gold when every
    sample failed — even if the gold score itself is None (un-scorable gold is
    still the only trainable label). Ties break toward gold, then the earliest
    sample (deterministic)."""
    best = ("gold", -1, gold_reward if include_gold else None)
    for i, r in enumerate(sample_rewards):
        if r is None:
            continue
        cur = best[2]
        if cur is None or r > cur:
            best = ("sample", i, r)
    if best[0] == "gold" and best[2] is None and include_gold:
        return ("gold", -1, None)   # nothing scorable — keep the gold label
    return best


def expand_for_bon(acts_bk: np.ndarray, n_samples: int) -> np.ndarray:
    """[B, k, d] -> [B*N*k, d] with each row's k-slot block repeated N times
    CONSECUTIVELY (row-major: r0s0, r0s1, ..., r0s{N-1}, r1s0, ...) — matching a
    prompt batch built by repeating each row N consecutive times, so the
    injection scan pairs sample j of row i with row i's vectors."""
    B, k, d = acts_bk.shape
    return np.repeat(acts_bk, n_samples, axis=0).reshape(B * n_samples * k, d)


# ------------------------------------------------------------------ data

def _load_rows(parquet_path, n_max=None, need_targets=True, targets_from_bank=None):
    """Rows with prompt, response, av_in_* slots ([k,d]), gold targets ([3,d]),
    doc_id (+src_row_id when present). Targets come from the in-file fixed
    target columns, or (bank av_sft input) from activation_L{...} via
    --targets-from-bank layers."""
    from multilayer_nla.datasets import AR_TARGET_COLUMNS, detect_av_slots
    pf = pq.ParquetFile(parquet_path)
    names = pf.schema_arrow.names
    if targets_from_bank:
        tgt_cols = [f"activation_L{L}" for L in targets_from_bank]
        missing = [c for c in tgt_cols if c not in names]
        assert not missing, f"{parquet_path} lacks {missing} for --targets-from-bank"
        slot_cols = None  # bank file has no av_in_*; score-gold doesn't inject
    else:
        tgt_cols = list(AR_TARGET_COLUMNS)
        if need_targets:
            missing = [c for c in tgt_cols if c not in names]
            assert not missing, (
                f"{parquet_path} lacks the fixed target columns {missing} — rebuild the av "
                f"dataset with build_conditions --av-with-targets (AV-SFT ignores them; "
                f"distill_av needs them to score against the fixed target)."
            )
        slot_cols = detect_av_slots(parquet_path) if any(n.startswith("av_in_") for n in names) else None
    cols = ["response", *tgt_cols]
    for opt in ("prompt", "doc_id", "src_row_id"):
        if opt in names:
            cols.append(opt)
    if slot_cols:
        cols += slot_cols
    rows = []
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        take = rg.num_rows if n_max is None else min(n_max - len(rows), rg.num_rows)
        rg = rg.slice(0, take)

        def to_np(name):
            c = rg.column(name).combine_chunks()
            return (c.flatten().to_numpy(zero_copy_only=False)
                    .astype(np.float32).reshape(len(c), -1))

        tg = {c: to_np(c) for c in tgt_cols}
        av = {c: to_np(c) for c in (slot_cols or [])}
        plain = {c: rg.column(c).to_pylist() for c in cols if c not in tgt_cols and c not in (slot_cols or [])}
        for i in range(take):
            row = {c: plain[c][i] for c in plain}
            row["gold"] = np.stack([tg[c][i] for c in tgt_cols])
            if slot_cols:
                row["acts"] = np.stack([av[c][i] for c in slot_cols])
            rows.append(row)
    return rows, (len(slot_cols) if slot_cols else 0)


def _score_explanations(critic, tokenizer, expls, golds, mse_scale, device, batch_size=64):
    """Batched reward per explanation (None = failed extraction / over-length)."""
    from multilayer_nla.evaluate_e2e import ar_sqerr_batch
    errs = ar_sqerr_batch(critic, tokenizer, expls, golds, mse_scale, device,
                          batch_size=batch_size)
    return [reward_from_taps(e) for e in errs]


def _slice_gold_to_taps(rows, critic):
    from multilayer_nla.datasets import AR_LAYER_TO_TARGET_COL, AR_TARGET_COLUMNS
    tap_cols = [AR_LAYER_TO_TARGET_COL[l] for l in critic.tap_layers]
    tap_idx = [AR_TARGET_COLUMNS.index(tc) for tc in tap_cols]
    for r in rows:
        if r["gold"].shape[0] == len(AR_TARGET_COLUMNS):
            r["gold"] = r["gold"][tap_idx]
    return tap_idx


# ------------------------------------------------------------------ modes

def run_score_gold(args, device="cuda"):
    import torch
    from transformers import AutoTokenizer
    from multilayer_nla.evaluate_e2e import load_critic
    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    critic, mse_scale = load_critic(args.base_ckpt, args.ar_ckpt, args.quant, device)
    tgl = [int(x) for x in args.targets_from_bank.split(",")] if args.targets_from_bank else None
    rows, _ = _load_rows(args.inp, n_max=args.max_rows, targets_from_bank=tgl)
    if tgl:
        # bank targets arrive in --targets-from-bank order; map them to tap order
        from multilayer_nla.datasets import AR_LAYER_TO_TARGET_COL  # noqa: F401  (order doc)
        order = {L: i for i, L in enumerate(tgl)}
        idx = [order[l] for l in critic.tap_layers]
        for r in rows:
            r["gold"] = r["gold"][idx]
    else:
        _slice_gold_to_taps(rows, critic)
    print(f"[distill:score-gold] {len(rows)} rows; critic taps {list(critic.tap_layers)}")

    expls = [extract_explanation(r["response"]) for r in rows]
    rewards = _score_explanations(critic, tokenizer, expls, [r["gold"] for r in rows],
                                  mse_scale, device, batch_size=args.score_batch)
    valid = np.array([r for r in rewards if r is not None], dtype=np.float64)
    n_fail = sum(1 for r in rewards if r is None)
    qs = {f"q{int(q * 100):02d}": float(np.quantile(valid, q))
          for q in (0.05, 0.25, 0.5, 0.75, 0.95)} if valid.size else {}
    print(f"[distill:score-gold] scored {valid.size}/{len(rows)} "
          f"(fail {n_fail}) | mean {valid.mean() if valid.size else float('nan'):.4f} | {qs}")

    # Stream the input through (a bank av_sft file can carry 11 wide layer
    # columns — never materialize it whole), appending the per-row reward, and
    # optionally a filtered copy (reward >= the q-quantile threshold).
    thr = float(np.quantile(valid, args.filter_quantile)) if (args.filter_quantile and valid.size) else None
    fpath = (str(Path(args.out).with_suffix("")) + f".top{int((1 - args.filter_quantile) * 100)}.parquet"
             if thr is not None else None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = fwriter = None
    n_kept = 0
    off = 0
    try:
        for batch in pq.ParquetFile(args.inp).iter_batches(batch_size=4096):
            if off >= len(rows):
                break
            t = pa.Table.from_batches([batch]).slice(0, len(rows) - off)
            r_slice = rewards[off:off + t.num_rows]
            t = t.append_column(REWARD_COL, pa.array(r_slice, pa.float32()))
            if writer is None:
                writer = pq.ParquetWriter(args.out, t.schema)
            writer.write_table(t)
            if thr is not None:
                keep = [i for i, r in enumerate(r_slice) if r is not None and r >= thr]
                if keep:
                    ft = t.take(keep)
                    if fwriter is None:
                        fwriter = pq.ParquetWriter(fpath, ft.schema)
                    fwriter.write_table(ft)
                    n_kept += len(keep)
            off += t.num_rows
    finally:
        if writer is not None:
            writer.close()
        if fwriter is not None:
            fwriter.close()
    print(f"[distill:score-gold] -> {args.out}")

    summary = {"n": len(rows), "n_failed": n_fail, "reward_mean": float(valid.mean()) if valid.size else None,
               "quantiles": qs, "ar_ckpt": args.ar_ckpt, "in": args.inp}
    if thr is not None:
        summary.update({"filter_quantile": args.filter_quantile, "threshold": thr,
                        "kept": n_kept, "filtered_path": fpath})
        print(f"[distill:score-gold] filtered (reward >= {thr:.4f}): {n_kept}/{len(rows)} -> {fpath}")
    Path(args.out + ".distill_meta.json").write_text(json.dumps(summary, indent=2))


@torch.no_grad()
def _bon_generate(actor, tokenizer, prompt_ids, acts_bk, vectors_ref, eos_ids, device,
                  n_samples, max_new_tokens, temperature):
    """B rows -> B*N sampled texts (each row's N samples consecutive)."""
    B, k, _ = acts_bk.shape
    BN = B * n_samples
    prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device).expand(BN, -1).contiguous()
    plen = prompt_t.shape[1]
    v = torch.as_tensor(expand_for_bon(acts_bk, n_samples), dtype=torch.float32, device=device)
    vectors_ref[0] = {"vectors": v,
                      "prompt_lens": torch.full((BN,), plen, dtype=torch.long, device=device)}
    try:
        gen = actor.generate(
            input_ids=prompt_t, attention_mask=torch.ones_like(prompt_t),
            max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
            top_p=1.0, top_k=0, pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True)
    finally:
        vectors_ref[0] = None
    seqs = gen.sequences
    texts = []
    for r in range(BN):
        resp_ids = seqs[r, plen:].tolist()
        n_real = next((i + 1 for i, t in enumerate(resp_ids) if t in eos_ids), len(resp_ids))
        texts.append(tokenizer.decode(resp_ids[:n_real], skip_special_tokens=True))
    return texts


def run_bon(args, device="cuda"):
    import torch
    from multilayer_nla.evaluate_e2e import load_actor, load_critic
    rows, k = _load_rows(args.inp, n_max=args.max_rows)
    assert k >= 1, f"{args.inp} has no av_in_* slots — BoN needs the built av_<cond> parquet"
    actor, tokenizer, inject_char, inj_id, vectors_ref, eos_ids = load_actor(
        args.base_ckpt, args.av_ckpt, k, args.quant, device)
    critic, mse_scale = load_critic(args.base_ckpt, args.ar_ckpt, args.quant, device)
    _slice_gold_to_taps(rows, critic)
    print(f"[distill:bon] {len(rows)} rows, k={k}, N={args.n_samples}, T={args.temperature}; "
          f"critic taps {list(critic.tap_layers)}")

    from multilayer_nla.datasets import apply_chat_template_no_think
    from nla.schema import INJECT_PLACEHOLDER
    prompt_msgs = rows[0].get("prompt")
    assert prompt_msgs, f"{args.inp} lacks the stored prompt column"
    prompt_text = apply_chat_template_no_think(
        tokenizer, [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inject_char)}
                    for m in prompt_msgs])
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    B = args.gen_batch
    out_resp, out_src, out_reward, out_gold_r = [], [], [], []
    n_gold_kept = 0
    for cs in range(0, len(rows), B):
        chunk = rows[cs:cs + B]
        acts_bk = np.stack([r["acts"] for r in chunk])
        texts = _bon_generate(actor, tokenizer, prompt_ids, acts_bk, vectors_ref, eos_ids,
                              device, args.n_samples, args.max_new_tokens, args.temperature)
        golds_rep = [r["gold"] for r in chunk for _ in range(args.n_samples)]
        expls = [extract_explanation(t) for t in texts]
        expls = [e if (e and e.strip()) else None for e in expls]
        s_rewards = _score_explanations(critic, tokenizer, expls, golds_rep, mse_scale,
                                        device, batch_size=args.score_batch)
        gold_expls = [extract_explanation(r["response"]) for r in chunk]
        g_rewards = _score_explanations(critic, tokenizer, gold_expls, [r["gold"] for r in chunk],
                                        mse_scale, device, batch_size=args.score_batch)
        for i, row in enumerate(chunk):
            samp = s_rewards[i * args.n_samples:(i + 1) * args.n_samples]
            src, idx, rew = select_best(g_rewards[i], samp, include_gold=not args.no_gold)
            if src == "gold":
                out_resp.append(row["response"])
                n_gold_kept += 1
            else:
                out_resp.append(wrap_explanation(expls[i * args.n_samples + idx]))
            out_src.append(src)
            out_reward.append(rew)
            out_gold_r.append(g_rewards[i])
        done = min(cs + B, len(rows))
        print(f"  {done}/{len(rows)} | gold kept {n_gold_kept}/{done} "
              f"({100.0 * n_gold_kept / done:.0f}%)", flush=True)

    # Stream the source through (never materialize the full slot columns),
    # swapping in the selected responses + provenance columns.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    off = 0
    try:
        for batch in pq.ParquetFile(args.inp).iter_batches(batch_size=4096):
            if off >= len(rows):
                break
            t = pa.Table.from_batches([batch]).slice(0, len(rows) - off)
            m = t.num_rows
            t = (t.drop_columns(["response"])
                 .append_column("response", pa.array(out_resp[off:off + m], pa.string()))
                 .append_column("bon_source", pa.array(out_src[off:off + m], pa.string()))
                 .append_column("bon_reward", pa.array(out_reward[off:off + m], pa.float32()))
                 .append_column(REWARD_COL, pa.array(out_gold_r[off:off + m], pa.float32())))
            if writer is None:
                writer = pq.ParquetWriter(args.out, t.schema)
            writer.write_table(t)
            off += m
    finally:
        if writer is not None:
            writer.close()
    vg = [r for r in out_gold_r if r is not None]
    vb = [r for r in out_reward if r is not None]
    print(f"[distill:bon] -> {args.out}")
    print(f"[distill:bon] gold kept {n_gold_kept}/{len(rows)} "
          f"({100.0 * n_gold_kept / len(rows):.1f}%) | mean gold reward "
          f"{np.mean(vg) if vg else float('nan'):.4f} -> selected {np.mean(vb) if vb else float('nan'):.4f}")
    Path(args.out + ".distill_meta.json").write_text(json.dumps({
        "n": len(rows), "n_samples": args.n_samples, "temperature": args.temperature,
        "gold_kept": n_gold_kept, "mean_gold_reward": float(np.mean(vg)) if vg else None,
        "mean_selected_reward": float(np.mean(vb)) if vb else None,
        "av_ckpt": args.av_ckpt, "ar_ckpt": args.ar_ckpt, "in": args.inp,
        "note": "selected reward is measured by the SAME frozen AR used for selection — "
                "an optimistic estimate by construction; the honest number is held-out "
                "e2e FVE after the continuation SFT round.",
    }, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["score-gold", "bon"], required=True)
    p.add_argument("--in", dest="inp", required=True,
                   help="av parquet with targets (build_conditions --av-with-targets), or for "
                        "score-gold, the bank av_sft parquet with --targets-from-bank")
    p.add_argument("--out", required=True)
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B")
    p.add_argument("--ar-ckpt", required=True, help="frozen AR multitap dir (the scorer)")
    p.add_argument("--av-ckpt", default=None, help="AV-SFT LoRA dir (bon mode)")
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--targets-from-bank", default=None,
                   help="score-gold on a BANK av_sft file: comma-sep target layers whose "
                        "activation_L{k} columns are the fixed target (e.g. 23,24,25)")
    p.add_argument("--filter-quantile", type=float, default=None,
                   help="score-gold: also write a filtered parquet dropping the bottom q "
                        "of labels by AR reward (e.g. 0.2)")
    p.add_argument("--n-samples", type=int, default=8, help="bon: samples per row (N)")
    p.add_argument("--temperature", type=float, default=1.0, help="bon: sampling temperature")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--gen-batch", type=int, default=8, help="bon: rows per generate call (xN sequences)")
    p.add_argument("--score-batch", type=int, default=64)
    p.add_argument("--no-gold", action="store_true",
                   help="bon: exclude the gold label from the argmax (pure on-policy selection)")
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()

    if args.mode == "score-gold":
        run_score_gold(args)
    else:
        assert args.av_ckpt, "--mode bon needs --av-ckpt"
        run_bon(args)


if __name__ == "__main__":
    main()
