"""Post-build integrity checks for a regenerated bank (CPU, no torch, no model).

  1. sidecar: present, RAW-storage invariant (`norm: none`), schema v2.
  2. columns: activation_L{k} for every save-layer; window_L{k} + window_size
     for every window-layer (when windows were captured).
  3. window internal parity: window_L{k}[:, -1, :] == activation_L{k} (slot W-1
     is the labeled position p). Exact after casting both to the window dtype:
     the final-token vector (fp32 view of the bf16 compute) and the window's
     last slot (fp16 cast of the same bf16 value) denote the SAME real number,
     which fp16 represents exactly within range — so equality is exact, not
     approximate. Only checked for window-layers that are also save-layers.
  4. finiteness: no NaN/Inf anywhere (a float16 overflow would show here).
  5. cross-parity of the p-slot vector against a reference:
       (a) the in-file stored `activation_vector` at `activation_layer` (kept
           when the bank was built without --drop-stored-activation), and/or
       (b) --existing <ref.parquet>: an external known-good bank, joined on
           (doc_id, n_raw_tokens).
     Gate = per-layer MEDIAN cosine > 0.999 (bf16 batching noise sits ~0.9999;
     a systematic off-position/wrong-layer regen shifts EVERY row so the median
     collapses). --strict requires float16-exact equality instead (only when
     both forwards are bit-identical, e.g. batch-size 1 on both sides).

    python -m multilayer_nla.verify_bank --bank bank.parquet \
        [--existing ref.parquet] [--strict] [--sample 5000]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pyarrow.parquet as pq
import yaml

COS_MATCH = 0.999


def _reshape(col, inner_last):
    """FixedSizeList column -> [n, inner_last] or [n, k, inner_last] float32."""
    c = col.combine_chunks()
    flat = c.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    per = flat.size // len(c)
    return flat.reshape(len(c), inner_last) if per == inner_last else flat.reshape(len(c), -1, inner_last)


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)


def verify_bank(bank: str, existing: str | None = None, sample: int = 5000,
                strict: bool = False, *, log=print) -> int:
    """Returns the number of failed checks (0 == PASS). Pure I/O + numpy."""
    fails = 0
    meta_path = bank + ".mlnla_meta.yaml"
    try:
        meta = yaml.safe_load(open(meta_path))
    except FileNotFoundError:
        log(f"FAIL: sidecar {meta_path} missing (the contract)")
        return 1
    save_layers = meta["save_layers"]
    window_layers = meta.get("window_layers") or []
    W, d = int(meta.get("window_size") or 0), meta["d_model"]
    log(f"bank={bank}\n  save={save_layers} window={window_layers} W={W} d={d} "
        f"act_dtype={meta.get('activation_dtype')} win_dtype={meta.get('window_dtype')} "
        f"rows_out={meta['rows_out']} norm={meta['norm']}")
    if meta["norm"] != "none":
        log("FAIL: RAW-storage invariant violated (norm != none)")
        fails += 1

    pf = pq.ParquetFile(bank)
    names = pf.schema_arrow.names
    for k in save_layers:
        if f"activation_L{k}" not in names:
            log(f"FAIL: missing column activation_L{k}")
            fails += 1
    if W >= 1:
        for k in window_layers:
            if f"window_L{k}" not in names:
                log(f"FAIL: missing column window_L{k}")
                fails += 1
        if "window_size" not in names:
            log("FAIL: missing window_size column")
            fails += 1
    if fails:
        return fails

    cols = [c for c in ["doc_id", "n_raw_tokens", "activation_vector", "activation_layer",
                        "window_size"] if c in names]
    cols += [f"activation_L{k}" for k in save_layers]
    cols += [f"window_L{k}" for k in window_layers if W >= 1]
    t = pf.read_row_group(0, columns=cols)
    n = min(sample, t.num_rows)
    t = t.slice(0, n)

    if W >= 1:
        ws = set(t.column("window_size").to_pylist())
        if ws != {W}:
            log(f"FAIL: window_size column {ws} != sidecar {W}")
            fails += 1

    # 3+4: finiteness everywhere; last-slot parity where both columns exist.
    for k in save_layers:
        act = _reshape(t.column(f"activation_L{k}"), d)
        if not np.isfinite(act).all():
            log(f"FAIL: non-finite values in activation_L{k}")
            fails += 1
    for k in window_layers if W >= 1 else []:
        win = _reshape(t.column(f"window_L{k}"), d).reshape(n, W, d)
        if not np.isfinite(win).all():
            log(f"FAIL: non-finite values in window_L{k} (float16 overflow?)")
            fails += 1
            continue
        if k in save_layers:
            act = _reshape(t.column(f"activation_L{k}"), d)
            # both sides in the window dtype: exact equality expected (see doc).
            if not np.array_equal(win[:, -1, :].astype(np.float16),
                                  act.astype(np.float16)):
                log(f"FAIL: activation_L{k} != window_L{k}[:, -1, :] (last slot != p)")
                fails += 1
            else:
                log(f"  ok L{k}: window last slot == activation_L{k}, all finite")
        else:
            log(f"  ok L{k}: window finite (no matching activation_L{k} to compare)")

    # 5a: in-file stored-vector cross-parity.
    if "activation_vector" in t.schema.names and "activation_layer" in t.schema.names:
        lays = set(t.column("activation_layer").to_pylist())
        if len(lays) == 1 and (sl := int(next(iter(lays)))) in save_layers:
            stored = _reshape(t.column("activation_vector"), d)
            regen = _reshape(t.column(f"activation_L{sl}"), d)
            cos = _cos(stored, regen)
            med, mn = float(np.median(cos)), float(cos.min())
            if med > COS_MATCH:
                log(f"  ok stored-vector parity L{sl}: median cos {med:.5f} (min {mn:.5f})")
            else:
                log(f"FAIL: stored-vector parity L{sl}: median cos {med:.4f} — regen at the "
                    f"wrong position / layer / model")
                fails += 1

    # 5b: external reference cross-parity, joined on (doc_id, n_raw_tokens).
    if existing:
        expf = pq.ParquetFile(existing)
        ex_names = expf.schema_arrow.names
        ex_cols = [c for c in ["doc_id", "n_raw_tokens"] if c in ex_names]
        ex_cols += [f"activation_L{k}" for k in save_layers if f"activation_L{k}" in ex_names]
        ex = expf.read_row_group(0, columns=ex_cols)
        key_new = list(zip(t.column("doc_id").to_pylist(), t.column("n_raw_tokens").to_pylist()))
        ex_idx = {kk: i for i, kk in enumerate(zip(ex.column("doc_id").to_pylist(),
                                                   ex.column("n_raw_tokens").to_pylist()))}
        shared = [(i, ex_idx[kk]) for i, kk in enumerate(key_new) if kk in ex_idx]
        log(f"  cross-parity: {len(shared)} shared keys vs reference "
            f"({'STRICT float16-exact' if strict else f'median cosine > {COS_MATCH}'})")
        if not shared:
            log("FAIL: no shared (doc_id, n_raw_tokens) keys vs reference")
            fails += 1
        else:
            ni = np.array([i for i, _ in shared])
            ei = np.array([j for _, j in shared])
            for k in save_layers:
                if f"activation_L{k}" not in ex.schema.names:
                    log(f"  (reference has no activation_L{k} — skip)")
                    continue
                a = _reshape(t.column(f"activation_L{k}"), d)[ni]
                b = _reshape(ex.column(f"activation_L{k}"), d)[ei]
                exact = np.array_equal(a.astype(np.float16), b.astype(np.float16))
                cos = _cos(a, b)
                mincos, medcos = float(cos.min()), float(np.median(cos))
                if strict:
                    if exact:
                        log(f"  ok L{k}: p-slot == reference (float16-exact, {len(cos)} rows)")
                    else:
                        mad = float(np.abs(a - b).max())
                        log(f"  FAIL L{k}: not float16-exact under --strict (min cos {mincos:.5f}, "
                            f"max|Δ| {mad:.3g}) — forwards not bit-identical (batch/padding?) "
                            f"or wrong position.")
                        fails += 1
                elif exact or medcos > COS_MATCH:
                    log(f"  ok L{k}: p-slot matches reference (median cos {medcos:.5f}, min {mincos:.5f}"
                        f"{' — float16-exact' if exact else ''})")
                else:
                    log(f"  FAIL L{k}: p-slot != reference (min cos {mincos:.4f}, median {medcos:.4f}) "
                        f"— gather at the WRONG position, or wrong --max-length/model.")
                    fails += 1

    log("\n" + ("PASS" if fails == 0 else f"FAIL ({fails} problems)"))
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True, help="bank parquet to verify")
    ap.add_argument("--existing", default=None, help="known-good reference bank for cross-parity")
    ap.add_argument("--sample", type=int, default=5000, help="rows to check")
    ap.add_argument("--strict", action="store_true",
                    help="require float16-exact cross-parity (bit-identical forwards only)")
    args = ap.parse_args()
    return 1 if verify_bank(args.bank, args.existing, args.sample, args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
