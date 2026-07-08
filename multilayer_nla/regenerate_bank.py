"""Regenerate a multi-layer × multi-position activation bank for published
(labeled) NLA rows — ONE GPU pass, supersedes the parent repo's two-pass
`regenerate_multilayer_activations.py` + `multitoken_nla/build_window_bank.py`.

The published warmstart parquets (EasyNLA `asher577/easynla-warmstart-data`,
nanoNLA `ceselder/qwen3-8b-nla-L24-finefineweb-100k`) already contain the real
LABELS — the AV `response` explanation, the AR critic `prompt`, the RL prompt,
the exact `detokenized_text_truncated` prefix, `n_raw_tokens`, and `doc_id`.
The activation is a deterministic function of (exact prefix, layer): the label
only ever depended on the TEXT (stage 2 feeds the API model just the prefix),
so the published explanation is exactly as valid for any layer/position slice
of that prefix as for the original single layer-24 vector. We regenerate the
activations locally and inherit the labels for free — no API, no paid labeling.

Per input row, one forward over `detokenized_text_truncated` captures:

    activation_L{k}   FixedSizeList[d]      final-token (== labeled position p)
                                            vector, for every k in --save-layers
    window_L{k}       FixedSizeList[W*d]    the last W positions' vectors
                                            [p-W+1 .. p], slot-major [W, d],
                                            slot 0 = oldest, slot W-1 == p,
                                            for every k in --window-layers
    window_size       int32                 constant W
    center_layer      int64                 provenance (build centers come later)

plus EVERY published column carried through untouched. RAW vectors only
(`norm="none"` — the invariant); windows default to float16 (lossless vs the
bf16 compute; range-guarded), final-token vectors default to float32.

Guards (all on by default):
  * round-trip: the re-encoded prefix must reproduce `n_raw_tokens`, or the
    final token is NOT the position the label describes. Hard fail unless
    --max-drop-frac tolerates (and drops+logs) rare per-row tokenizer drift.
  * stored-vector cross-parity: when the input carries the original
    `activation_vector` (+ its `activation_layer`), the regenerated
    activation_L{that layer} must match it per chunk (median cosine >= 0.999;
    bf16 batching noise sits ~0.9999, an off-by-one position or wrong layer
    collapses it). This closes the gap the count-only round-trip leaves open
    (same-count retokenization drift) and catches wrong --max-length, wrong
    model revision, and wrong hook semantics in one shot.
  * short-window: rows with < W real tokens can't fill the window; dropped and
    logged (expected ~0: stage-0's _MIN_POSITION=50 guarantees >= 51 tokens).
  * float16 range: enforced before every fp16 cast (extractor + writer).

Usage (see multilayer_nla/README.md for the full runbook):
    python -m multilayer_nla.regenerate_bank \\
        --in $PUB/av_sft_train.parquet --out $BANK/av_sft_train.parquet \\
        --base-model Qwen/Qwen3-8B --save-layers 19-29 \\
        --window 8 --window-layers 23,24,25 --max-length 4096
    # storage math without touching the GPU:  add --dry-run
    # fan out:  --num-shards N --shard-index i  (per-shard files; merge after)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen._common import add_storage_args, load_class, make_storage, parse_kwargs

TEXT_COL = "detokenized_text_truncated"
FLOAT16_MAX = 65504.0
_DTYPES = {"float16": np.float16, "float32": np.float32}
# Median-cosine gate for the stored-vector cross-parity. Correct position under
# bf16 batching noise sits ~0.9999; a systematic off-position/wrong-layer regen
# collapses the MEDIAN (every row moves), which is why the median is the gate
# and per-row stragglers are only counted.
PARITY_MEDIAN_MIN = 0.999


def layer_col(k: int) -> str:
    return f"activation_L{k}"


def window_col(k: int) -> str:
    return f"window_L{k}"


def parse_layers(spec: str) -> list[int]:
    """'19-29' or '19,24,29' or '19-21,25,27-29' -> sorted unique ints."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    assert out, f"empty/invalid layer spec: {spec!r}"
    return sorted(out)


def select_keep_indices(results: list[dict], n_raw_tokens, *,
                        max_drop_frac: float = 0.0, row_offset: int = 0):
    """Which chunk rows survive the guards, and why the rest were dropped. Pure.

    Drops (a) short-window rows (result['valid'] is False) always, and
    (b) round-trip-mismatch rows (re-encoded token count != stored n_raw_tokens)
    when `n_raw_tokens` is provided. Round-trip drops hard-fail unless their
    fraction is within `max_drop_frac` — a systematic mismatch is almost always
    a --max-length misconfiguration that would silently place the final token
    off the labeled position.

    Returns (keep_idx, info) with info = {n_short, n_roundtrip, roundtrip_frac}.
    """
    n = len(results)
    short = {i for i, r in enumerate(results) if not r.get("valid", True)}
    roundtrip: set[int] = set()
    if n_raw_tokens is not None:
        roundtrip = {i for i, (r, nrt) in enumerate(zip(results, n_raw_tokens))
                     if len(r["token_ids"]) != nrt}
        frac = len(roundtrip) / n if n else 0.0
        if frac > max_drop_frac:
            first = sorted(roundtrip)[0]
            raise AssertionError(
                f"{len(roundtrip)} rows fail the tokenization round-trip "
                f"(first: input row {row_offset + first} re-encoded to "
                f"{len(results[first]['token_ids'])} tokens, stage-0 had "
                f"{n_raw_tokens[first]}). Drop fraction {frac:.4%} exceeds "
                f"--max-drop-frac {max_drop_frac:.4%}. Check --max-length matches "
                f"the original extraction (4096 for the published datasets)."
            )
    drop = short | roundtrip
    keep_idx = [i for i in range(n) if i not in drop]
    return keep_idx, {
        "n_short": len(short),
        "n_roundtrip": len(roundtrip),
        "roundtrip_frac": (len(roundtrip) / n if n else 0.0),
    }


def _fsl(mat: np.ndarray, inner_len: int) -> pa.Array:
    """[n, inner_len] -> FixedSizeList[inner_len], dtype from the array."""
    assert mat.ndim == 2 and mat.shape[1] == inner_len, (
        f"_fsl got {mat.shape}, expected (n, {inner_len})"
    )
    flat = np.ascontiguousarray(mat).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat), inner_len)


def stored_parity(table: pa.Table, results: list[dict], d_model: int):
    """Cross-parity: regenerated final-token vector vs the input's ORIGINAL
    `activation_vector` at its `activation_layer`. Pure (numpy). Returns
    (median_cos, min_cos, n_below, layer) or None when the input has no stored
    vector / the stored layer wasn't regenerated. `table` and `results` must be
    pre-filtered to the same kept rows.

    The stored vector was computed by the original stage-0 forward (full doc,
    original batch shape); the regenerated one comes from the truncated prefix
    re-forwarded in a new batch. Causal attention makes them the same
    computation modulo bf16 kernel/batch noise — cosine ~0.9999. An off-by-one
    gather, wrong layer, wrong model, or same-count retokenization drift breaks
    the match, which is exactly what this guard is for.
    """
    names = table.schema.names
    if "activation_vector" not in names or table.num_rows == 0:
        return None
    if "activation_layer" in names:
        lays = set(table.column("activation_layer").to_pylist())
        assert len(lays) == 1, f"activation_layer not constant in chunk: {sorted(lays)[:5]}"
        stored_layer = int(lays.pop())
    else:
        return None
    if stored_layer not in results[0]["final"]:
        return None
    col = table.column("activation_vector").combine_chunks()
    stored = (col.flatten().to_numpy(zero_copy_only=False)
              .astype(np.float32).reshape(len(col), d_model))
    regen = np.stack([np.asarray(r["final"][stored_layer], dtype=np.float32)
                      for r in results])
    num = (stored * regen).sum(axis=1)
    den = np.linalg.norm(stored, axis=1) * np.linalg.norm(regen, axis=1) + 1e-8
    cos = num / den
    return float(np.median(cos)), float(cos.min()), int((cos < PARITY_MEDIAN_MIN).sum()), stored_layer


def assemble_bank_table(table: pa.Table, results: list[dict], save_layers: list[int],
                        window_layers: list[int], window: int, d_model: int,
                        act_dtype, win_dtype, center: int,
                        *, drop_stored_activation: bool = False) -> pa.Table:
    """Append activation_L{k} (+ window_L{k}, window_size) columns; carry every
    input column through (optionally dropping the now-redundant stored
    activation_vector). `table` and `results` must be pre-filtered to the SAME
    kept rows and aligned 1:1. Pure (numpy/pyarrow) — unit-testable offline with
    fabricated results.
    """
    n = table.num_rows
    assert len(results) == n, f"results ({len(results)}) != table rows ({n})"
    out = table
    if drop_stored_activation and "activation_vector" in out.schema.names:
        out = out.drop_columns(["activation_vector"])

    for k in save_layers:
        assert layer_col(k) not in out.schema.names, f"input already has {layer_col(k)}"
        mat = np.stack([np.asarray(r["final"][k], dtype=np.float32) for r in results])
        assert mat.shape == (n, d_model), (
            f"layer {k}: built {mat.shape}, expected ({n}, {d_model})"
        )
        if act_dtype == np.float16:
            mx = float(np.abs(mat).max()) if n else 0.0
            assert np.isfinite(mx) and mx < FLOAT16_MAX, (
                f"layer {k}: max|coord|={mx:.1f} exceeds float16 range; "
                f"use --activation-dtype float32."
            )
        out = out.append_column(layer_col(k), _fsl(mat.astype(act_dtype, copy=False), d_model))

    do_window = window >= 1 and window_layers
    if do_window:
        for k in window_layers:
            assert window_col(k) not in out.schema.names, f"input already has {window_col(k)}"
            blocks = [np.asarray(r["window"][k], dtype=win_dtype).reshape(window, d_model)
                      for r in results]
            win = (np.stack([b.reshape(-1) for b in blocks]) if n
                   else np.zeros((0, window * d_model), dtype=win_dtype))
            out = out.append_column(window_col(k), _fsl(win, window * d_model))
        out = out.append_column("window_size", pa.array([window] * n, pa.int32()))

    if "center_layer" not in out.schema.names:
        out = out.append_column("center_layer", pa.array([center] * n, pa.int64()))
    return out


def bucketed_extract(texts, n_raw_tokens, extract_fn, *, length_bucket: bool = False):
    """Run `extract_fn(texts)`, optionally sorting the chunk ascending by
    n_raw_tokens first (less dynamic-padding waste) then scattering results back
    to the original order. Output order/content identical either way. Pure."""
    if not length_bucket:
        return extract_fn(texts)
    assert n_raw_tokens is not None and len(n_raw_tokens) == len(texts), (
        "--length-bucket requires the n_raw_tokens column aligned to texts"
    )
    order = sorted(range(len(texts)), key=lambda i: n_raw_tokens[i])  # stable ascending
    res_sorted = extract_fn([texts[i] for i in order])
    assert len(res_sorted) == len(texts), (
        f"extractor returned {len(res_sorted)} results for {len(texts)} texts"
    )
    out = [None] * len(texts)
    for j, i in enumerate(order):
        out[i] = res_sorted[j]
    return out


def storage_estimate(n_rows: int, d_model: int, save_layers, window_layers,
                     window: int, act_dtype, win_dtype) -> dict:
    """Per-row and total appended bytes (the carried input columns come free)."""
    act_b = np.dtype(act_dtype).itemsize
    win_b = np.dtype(win_dtype).itemsize
    per_row = len(save_layers) * d_model * act_b
    if window >= 1:
        per_row += len(window_layers) * window * d_model * win_b
    return {"per_row_bytes": per_row, "total_gb": per_row * n_rows / 1e9, "n_rows": n_rows}


def _infer_center(pf: pq.ParquetFile, explicit: int | None) -> int:
    """center-layer from CLI, else the (constant) center_layer/activation_layer column."""
    if explicit is not None:
        return explicit
    names = pf.schema_arrow.names
    for col in ("center_layer", "activation_layer"):
        if col in names:
            vals = set(pf.read(columns=[col]).column(col).to_pylist())
            assert len(vals) == 1, (
                f"{col} is not constant across rows ({sorted(vals)[:5]}...) — "
                f"pass --center-layer explicitly."
            )
            return int(vals.pop())
    raise SystemExit(
        "no --center-layer given and neither center_layer nor activation_layer "
        "column present — cannot infer the center."
    )


def _shard_out(out: str, idx: int, n: int) -> str:
    """Per-shard output path so parallel jobs never clobber one --out."""
    if n <= 1:
        return out
    p = Path(out)
    return str(p.with_name(f"{p.stem}.shard{idx:02d}of{n:02d}{p.suffix}"))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="inp", required=True,
                   help="published labeled parquet (prefix text + labels; vectors optional)")
    p.add_argument("--out", required=True, help="output bank parquet (labels preserved)")
    p.add_argument("--base-model", required=True, help="HF base model, e.g. Qwen/Qwen3-8B")
    p.add_argument("--center-layer", type=int, default=None,
                   help="center block l (provenance + default save window). Default: read from "
                        "the center_layer/activation_layer column (24 for the published sets).")
    p.add_argument("--save-layers", default=None,
                   help="final-token layers to archive, e.g. '19-29'. Wider is FREE on compute "
                        "(one forward) — cost is storage. Default: the center triplet {l-1,l,l+1}.")
    p.add_argument("--window", type=int, default=8,
                   help="token positions per window row (W). 0 disables window capture entirely.")
    p.add_argument("--window-layers", default=None,
                   help="layers to save [W,d] windows for (subset of interest, e.g. '23,24,25'). "
                        "Default: the center triplet. Ignored when --window 0.")
    p.add_argument("--activation-dtype", choices=list(_DTYPES), default="float32",
                   help="storage dtype for activation_L{k} (float32 default — matches the "
                        "parent-repo banks; float16 halves storage, range-guarded).")
    p.add_argument("--window-dtype", choices=list(_DTYPES), default="float16",
                   help="storage dtype for window_L{k} (float16 default: lossless vs the bf16 "
                        "compute, 2x smaller, range-guarded).")
    p.add_argument("--chunk-size", type=int, default=512, help="rows per read/write (bounds memory)")
    p.add_argument("--batch-size", type=int, default=16, help="model forward batch size")
    p.add_argument("--max-length", type=int, default=4096,
                   help="extractor context cap — MUST match the original stage-0 extraction "
                        "(4096 for the published datasets). Too small right-truncates long rows "
                        "and regenerates at the wrong position; the n_raw_tokens round-trip "
                        "check turns that into a hard error.")
    p.add_argument("--length-bucket", action="store_true", default=False,
                   help="sort each chunk by n_raw_tokens before the forward (speed); output "
                        "byte-identical, just faster. Requires n_raw_tokens.")
    p.add_argument("--no-roundtrip-check", action="store_true",
                   help="disable the n_raw_tokens round-trip guard (NOT recommended)")
    p.add_argument("--max-drop-frac", type=float, default=0.0,
                   help="tolerate up to this fraction of round-trip-mismatch rows by DROPPING "
                        "(and logging) them; default 0.0 hard-fails on any mismatch. Use e.g. "
                        "1e-3 on a large run where rare per-row tokenizer drift is expected.")
    p.add_argument("--no-stored-parity", action="store_true",
                   help="skip the stored activation_vector cross-parity guard (only when the "
                        "input genuinely has no comparable stored vector)")
    p.add_argument("--drop-stored-activation", action="store_true",
                   help="drop the input's single-layer activation_vector column from the output "
                        "after parity passes (it is redundant with activation_L{that layer}; "
                        "saves d*4 bytes/row)")
    p.add_argument("--drop-input-bank-cols", action="store_true",
                   help="drop any activation_L*/window_L*/window_size columns already in the "
                        "input before appending — lets you re-window an existing bank shard "
                        "(the window is re-forwarded from the prefix text either way).")
    p.add_argument("--max-rows", type=int, default=None, help="cap input rows (pilot subsetting)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="data-parallel fan-out: split the input into N contiguous shards "
                        "(one GPU/job each). Each shard writes its own parquet; merge after.")
    p.add_argument("--shard-index", type=int, default=0, help="which shard [0, num-shards)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the storage estimate + config and exit (no model load)")
    p.add_argument("--extractor-cls",
                   default="multilayer_nla.extract_multilayer.MultiLayerHFExtractor")
    p.add_argument("--extractor-kwargs", default=None, help="JSON dict of extra extractor kwargs")
    add_storage_args(p)
    args = p.parse_args()

    import re
    import yaml

    storage = make_storage(args)
    pf = pq.ParquetFile(storage.open_read(args.inp))
    names = pf.schema_arrow.names
    assert TEXT_COL in names, (
        f"input lacks {TEXT_COL!r} — cannot regenerate without the source prefix"
    )
    if not args.no_roundtrip_check:
        assert "n_raw_tokens" in names, (
            "input lacks n_raw_tokens — the round-trip guard cannot run. Either use a "
            "published parquet that has it or pass --no-roundtrip-check (NOT recommended)."
        )

    center = _infer_center(pf, args.center_layer)
    assert center - 1 >= 0, f"center-layer={center} has no l-1 block"
    save_layers = parse_layers(args.save_layers) if args.save_layers else [center - 1, center, center + 1]
    assert min(save_layers) >= 0, f"--save-layers has a negative layer: {save_layers}"
    window = args.window
    window_layers = (parse_layers(args.window_layers) if args.window_layers
                     else [center - 1, center, center + 1]) if window >= 1 else []
    triplet = {center - 1, center, center + 1}
    if not triplet.issubset(save_layers):
        print(f"WARNING: --save-layers {save_layers} omits part of the center-{center} triplet "
              f"{sorted(triplet)} — build_from_published --center {center} will fail until you "
              f"widen the window or build at a center whose triplet IS saved.", flush=True)

    _BANK_RE = re.compile(r"^(activation|window)_L\d+$|^window_size$")
    clash = [c for c in names if _BANK_RE.match(c)]
    if clash and not args.drop_input_bank_cols:
        raise SystemExit(
            f"input already has bank columns {clash[:4]}... — pass --drop-input-bank-cols to "
            f"re-bank an existing shard (re-forwards everything from the prefix text)."
        )
    read_cols = [c for c in names if not _BANK_RE.match(c)] if args.drop_input_bank_cols else None

    act_dtype = _DTYPES[args.activation_dtype]
    win_dtype = _DTYPES[args.window_dtype]

    assert args.num_shards >= 1 and 0 <= args.shard_index < args.num_shards, (
        f"bad shard config: shard_index={args.shard_index}, num_shards={args.num_shards}"
    )
    total = pf.metadata.num_rows
    if args.max_rows is not None:
        total = min(total, args.max_rows)
    per = (total + args.num_shards - 1) // args.num_shards
    lo = args.shard_index * per
    hi = min(total, lo + per)

    # Storage math BEFORE any GPU work — the d_model guess for --dry-run comes
    # from the stored activation_vector width when present, else printed as
    # per-unit. (The live run uses the extractor's true d_model.)
    d_guess = None
    if "activation_vector" in names:
        f = pf.schema_arrow.field("activation_vector")
        if pa.types.is_fixed_size_list(f.type):
            d_guess = f.type.list_size
    est = storage_estimate(hi - lo, d_guess or 4096, save_layers, window_layers,
                           window, act_dtype, win_dtype)
    print(f"[bank] rows [{lo}, {hi}) of {total} | save L{save_layers[0]}-L{save_layers[-1]} "
          f"({len(save_layers)} layers, {args.activation_dtype}) | window "
          f"{'OFF' if window < 1 else f'W={window} on {window_layers} ({args.window_dtype})'}",
          flush=True)
    print(f"[bank] appended storage ~{est['per_row_bytes'] / 1024:.0f} KB/row "
          f"-> ~{est['total_gb']:.1f} GB this shard"
          f"{' (d_model assumed 4096 for the estimate)' if d_guess is None else ''}", flush=True)
    if args.dry_run:
        print("[bank] --dry-run: exiting before model load.")
        return

    user_kwargs = parse_kwargs(args.extractor_kwargs)
    assert "model_name" not in user_kwargs, "pass --base-model, not model_name in --extractor-kwargs"
    extractor = load_class(args.extractor_cls)(
        model_name=args.base_model, batch_size=args.batch_size,
        max_length=args.max_length, **user_kwargs,
    )
    assert hasattr(extractor, "extract_bank"), (
        f"{args.extractor_cls} has no extract_bank(); the unified regeneration needs it"
    )
    import torch
    from nla.utils.arch_adapters import resolve_decoder_layers
    d_model = extractor.d_model
    n_layers = len(resolve_decoder_layers(extractor.model))
    for li in save_layers + window_layers:
        assert li < n_layers, f"layer {li} out of range for a {n_layers}-block model"
    torch_win_dtype = torch.float16 if args.window_dtype == "float16" else torch.float32

    out_path = _shard_out(args.out, args.shard_index, args.num_shards)
    if args.num_shards > 1:
        print(f"[shard {args.shard_index}/{args.num_shards}] -> {out_path}", flush=True)
    storage.ensure_parent(out_path)

    writer = None
    g = 0                 # global input-row offset
    n_in = n_out = n_short = n_roundtrip = 0
    parity_medians: list[float] = []
    parity_below = 0
    parity_layer = None
    import time
    t0 = time.time()
    n_target = hi - lo
    try:
        for batch in pf.iter_batches(batch_size=args.chunk_size, columns=read_cols):
            if g >= hi:
                break
            blen = batch.num_rows
            o_lo, o_hi = max(lo, g), min(hi, g + blen)
            if o_lo < o_hi:
                tbl = pa.Table.from_batches([batch]).slice(o_lo - g, o_hi - o_lo)
                n_in += tbl.num_rows
                texts = tbl.column(TEXT_COL).to_pylist()
                nrt = (tbl.column("n_raw_tokens").to_pylist()
                       if "n_raw_tokens" in tbl.schema.names else None)
                results = bucketed_extract(
                    texts, nrt,
                    lambda t: extractor.extract_bank(
                        t, save_layers, window_layers, window, window_dtype=torch_win_dtype),
                    length_bucket=args.length_bucket,
                )
                keep, info = select_keep_indices(
                    results, None if args.no_roundtrip_check else nrt,
                    max_drop_frac=args.max_drop_frac, row_offset=o_lo)
                n_short += info["n_short"]
                n_roundtrip += info["n_roundtrip"]
                tbl = tbl.take(keep)
                results = [results[i] for i in keep]

                if not args.no_stored_parity and tbl.num_rows:
                    par = stored_parity(tbl, results, d_model)
                    if par is not None:
                        med, mn, n_below, parity_layer = par
                        parity_medians.append(med)
                        parity_below += n_below
                        assert med >= PARITY_MEDIAN_MIN, (
                            f"[bank] stored-vector cross-parity FAILED: chunk median cos "
                            f"{med:.5f} < {PARITY_MEDIAN_MIN} vs the input's activation_vector "
                            f"(L{parity_layer}, min {mn:.4f}). The regenerated final-token "
                            f"vector is NOT the activation the label describes — wrong "
                            f"--max-length / model revision / hook semantics. Nothing written "
                            f"is trustworthy; fix the config and re-run."
                        )

                tbl = assemble_bank_table(
                    tbl, results, save_layers, window_layers, window, d_model,
                    act_dtype, win_dtype, center,
                    drop_stored_activation=args.drop_stored_activation)
                if writer is None:
                    writer = pq.ParquetWriter(storage.open_write(out_path), tbl.schema)
                writer.write_table(tbl)
                n_out += tbl.num_rows
                el = time.time() - t0
                rate = n_in / el if el > 0 else 0.0
                eta = (n_target - n_in) / rate if rate > 0 else 0.0
                print(f"  {n_out} written | {n_in}/{n_target} in | {rate:.0f} rows/s | "
                      f"ETA {eta / 60:.1f} min", flush=True)
            g += blen
    finally:
        if writer is not None:
            writer.close()

    parity_summary = None
    if parity_medians:
        parity_summary = {
            "layer": parity_layer,
            "chunk_median_cos_min": float(min(parity_medians)),
            "chunk_median_cos_median": float(np.median(parity_medians)),
            "rows_below_threshold": parity_below,
            "threshold": PARITY_MEDIAN_MIN,
        }
        print(f"[bank] stored-vector cross-parity OK vs activation_vector (L{parity_layer}): "
              f"worst chunk median cos {parity_summary['chunk_median_cos_min']:.5f}; "
              f"{parity_below}/{n_out} rows below {PARITY_MEDIAN_MIN} (per-row drift, kept).",
              flush=True)
    elif not args.no_stored_parity:
        print("[bank] NOTE: no stored activation_vector to cross-check — the round-trip "
              "count guard is the only position check. Run verify_regen_parity on a slice.",
              flush=True)

    from multilayer_nla.manifest import build_manifest
    meta = {
        "kind": "mlnla_bank",
        "schema_version": 2,
        "base_model": args.base_model,
        "d_model": d_model,
        "center_layer": center,
        "save_layers": save_layers,
        "activation_dtype": args.activation_dtype,
        "window_size": window if window_layers else 0,
        "window_layers": window_layers,
        "window_dtype": args.window_dtype,
        "window_layout": "slot-major [W, d]; slot 0 = oldest (p-W+1), slot W-1 = p (labeled token)",
        "norm": "none",
        "max_length": args.max_length,
        "source": args.inp,
        "rows_in": n_in,
        "rows_out": n_out,
        "dropped_short_window": n_short,
        "dropped_roundtrip": n_roundtrip,
        "stored_parity": parity_summary,
        "activation_cols": [layer_col(k) for k in save_layers],
        "window_cols": [window_col(k) for k in window_layers],
        "manifest": build_manifest(stage="regenerate_bank", tokenizer=extractor.tokenizer,
                                   extra={"base_model": args.base_model}),
    }
    meta_path = out_path + ".mlnla_meta.yaml"
    storage.write_text(meta_path, yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))

    print(f"wrote {n_out}/{n_in} rows -> {out_path}")
    print(f"  dropped: {n_short} short-window, {n_roundtrip} round-trip "
          f"({100.0 * n_roundtrip / n_in if n_in else 0:.3f}%) — kept rows are position-exact")
    print(f"  sidecar -> {meta_path}")
    if args.num_shards > 1:
        print(f"[shard {args.shard_index}/{args.num_shards}] done. Merge all shards with e.g.:\n"
              f"  python -c \"import pyarrow.parquet as pq, glob; "
              f"pq.write_table(pq.ParquetDataset(sorted(glob.glob('PREFIX.shard*of*.parquet'))).read(), "
              f"'MERGED.parquet', row_group_size=4096)\"")
    print(json.dumps({"rows_out": n_out, "save_layers": save_layers,
                      "window": window if window_layers else 0,
                      "window_layers": window_layers, "d_model": d_model}, indent=2))


if __name__ == "__main__":
    main()
