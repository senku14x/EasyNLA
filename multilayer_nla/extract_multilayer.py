"""Multi-layer / multi-position activation extraction (one forward per batch).

`MultiLayerHFExtractor` extends EasyNLA's `HFExtractor` with two capture modes:

  * `extract_multi(texts, layer_indices, final_token_only=False)` — per text,
    the hidden state at each requested decoder block: `[seq_len, d]` (or the
    final real token's `[d]` with `final_token_only=True`). This is the
    multi-layer analog of `extract()` and what the parity verifiers compare.

  * `extract_bank(texts, final_layers, window_layers, window, window_dtype)` —
    the UNIFIED bank capture: final-token vectors for every layer in
    `final_layers` PLUS a `[window, d]` left-window (the last `window` real
    token positions, oldest slot first, slot W-1 == the final token) for every
    layer in `window_layers`, all from ONE forward. Supersedes the two-pass
    regenerate + build_window_bank flow of the parent repo.

Correctness notes (the load-bearing details):

  * `layer_index=K` captures the OUTPUT of decoder block K — identical to
    `HFExtractor.extract()` semantics (== HF `hidden_states[K+1]`). Bitwise
    parity with the single-layer path is checked by `verify_center_parity` /
    `verify_regen_parity` — run them before trusting any downstream number.
  * The hook on the HIGHEST requested layer raises `_CaptureComplete` after
    storing its capture, short-circuiting the forward exactly like the EasyNLA
    single-layer extractor: blocks above max(requested) and the lm_head
    ([B, S, vocab] logits — the dominant VRAM cost) never run. Lower layers'
    outputs are already computed and captured when the abort fires, so captured
    values are unaffected by where the abort happens.
  * Right padding is set by the base class and REQUIRED: the final real token
    of row i sits at index `attention_mask[i].sum() - 1`. The window gather is
    `[len-W, len)` per row, done ON GPU, so only `[B, W, d]` moves to CPU.
  * Windows are cast to `window_dtype` (default float16) ON GPU before the
    transfer. The base model computes in bf16 (8-bit mantissa exponent range);
    float16 (10-bit mantissa, max 65504) is lossless vs that compute EXCEPT for
    magnitude overflow, which the pre-cast range guard turns into a loud error,
    never a silent inf. Final-token vectors are returned float32.
  * Rows shorter than `window` get `valid=False` (window unfillable) — the
    caller drops + logs them. Stage-0's `_MIN_POSITION = 50` means any labeled
    position has >= 51 tokens of context, so windows <= ~50 never drop rows.

RAW vectors everywhere (`norm="none"`) — normalization is a training/eval-time
decision, never an extraction-time one.

The CLI (below) is the fresh-corpus stage-0 path: sample keyed-RNG positions
per document (byte-identical to single-layer stage0) and store the contiguous
{center-1, center, center+1} triplet. For regenerating activations for rows
whose prefix text + labels are already published, use `regenerate_bank.py`.
"""

import argparse
import json
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from datasets import Dataset, load_dataset
from tqdm import tqdm

from nla.utils.arch_adapters import resolve_decoder_layers
from nla.datagen._common import add_storage_args, load_class, make_storage, parse_kwargs
from nla.datagen.extractors import HFExtractor, _CaptureComplete
# Reuse the EXACT keyed-RNG sampler + min-position + dataset-id helpers so a
# multi-layer run is positionally identical to a single-layer run (invariant).
from nla.datagen.stage0_extract import _MIN_POSITION, _dataset_id, _sample_positions
from multilayer_nla.manifest import build_manifest

FLOAT16_MAX = 65504.0


class MultiLayerHFExtractor(HFExtractor):
    """HFExtractor that captures several decoder-block outputs in one forward.

    Inherits model load, right-padding tokenization, batching, sdpa attention,
    and d_model from EasyNLA's HFExtractor. The single-layer `extract()` is
    left untouched so the parity verifiers can compare both paths on the same
    model instance.
    """

    def _register_multi_hooks(self, layer_indices: list[int]):
        layers = resolve_decoder_layers(self.model)
        assert len(set(layer_indices)) == len(layer_indices), (
            f"duplicate layer indices in {layer_indices}"
        )
        for li in layer_indices:
            assert 0 <= li < len(layers), (
                f"layer_index={li} out of range for model with {len(layers)} layers"
            )
        self._captured_multi: dict[int, torch.Tensor] = {}
        handles = []
        last_li = max(layer_indices)

        def make_hook(li: int):
            def hook(_module, _inputs, output):
                # Transformer blocks return tuples; first element is the hidden
                # state. .clone() (not bare .detach()) — storage may be reused.
                h = output[0] if isinstance(output, tuple) else output
                self._captured_multi[li] = h.detach().clone()
                if li == last_li:
                    # All lower requested layers have already fired (forward is
                    # sequential); skip the remaining blocks + lm_head.
                    raise _CaptureComplete
            return hook

        for li in layer_indices:
            handles.append(layers[li].register_forward_hook(make_hook(li)))
        return handles

    def _forward_captured(self, input_ids, attention_mask, layer_indices):
        """One forward with the multi hooks active; asserts every hook fired."""
        self._captured_multi = {}
        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except _CaptureComplete:
            pass
        for li in layer_indices:
            assert li in self._captured_multi, (
                f"forward hook on decoder block {li} did not fire — wrong module path?"
            )
            assert self._captured_multi[li].shape[-1] == self.d_model, (
                f"layer {li}: captured width {self._captured_multi[li].shape[-1]} "
                f"!= d_model {self.d_model}"
            )

    @torch.no_grad()
    def extract_multi(self, texts: list[str], layer_indices: list[int],
                      *, final_token_only: bool = False) -> list[dict[str, Any]]:
        """Per-text {token_ids, hidden}. `hidden[li]` is [seq_len, d] float32 by
        default; with `final_token_only=True` it is the final real token's [d]
        (keeps CPU memory flat when capturing a wide layer window over long
        prefixes)."""
        handles = self._register_multi_hooks(layer_indices)
        try:
            return self._extract_multi_impl(texts, layer_indices,
                                            final_token_only=final_token_only)
        finally:
            for h in handles:
                h.remove()

    def _extract_multi_impl(self, texts, layer_indices, *, final_token_only):
        results: list[dict[str, Any]] = []
        for start in range(0, len(texts), self.batch_size):
            sub = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                sub, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length, add_special_tokens=True,
            )
            device = self.model.get_input_embeddings().weight.device
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            self._forward_captured(input_ids, attention_mask, layer_indices)

            lengths = attention_mask.sum(dim=1)  # [B], on device
            lengths_cpu = lengths.cpu().tolist()
            if final_token_only:
                # Select the final REAL token ON GPU, then move only [B, d] per
                # layer to CPU. Right padding: last real token at index len-1.
                last = (lengths - 1).clamp_min(0)                       # [B]
                bidx = torch.arange(input_ids.shape[0], device=last.device)
                final = {li: self._captured_multi[li][bidx, last].float().cpu()  # [B, d]
                         for li in layer_indices}
                for i, seq_len in enumerate(lengths_cpu):
                    results.append({
                        "token_ids": input_ids[i, :seq_len].cpu().tolist(),
                        "hidden": {li: final[li][i] for li in layer_indices},  # [d]
                    })
            else:
                hidden = {li: self._captured_multi[li].float().cpu() for li in layer_indices}
                for i, seq_len in enumerate(lengths_cpu):
                    results.append({
                        "token_ids": input_ids[i, :seq_len].cpu().tolist(),
                        "hidden": {li: hidden[li][i, :seq_len].clone() for li in layer_indices},
                    })
        return results

    @torch.no_grad()
    def extract_bank(self, texts: list[str], final_layers: list[int],
                     window_layers: list[int], window: int,
                     *, window_dtype: torch.dtype = torch.float16) -> list[dict[str, Any]]:
        """UNIFIED capture: final-token vector for every `final_layers` layer +
        `[window, d]` left-window for every `window_layers` layer, one forward.

        Returns per text:
            {token_ids: list[int],
             final:  {li: Tensor[d] float32}          for li in final_layers,
             window: {li: Tensor[window, d] window_dtype}  for li in window_layers,
             valid:  bool}   # False iff seq has < window real tokens

        `window_layers` need not be a subset of `final_layers`; hooks are
        registered on the union. `window=0` (or empty window_layers) disables
        window capture and never marks rows invalid.
        """
        assert window >= 0, f"window must be >= 0, got {window}"
        do_window = window >= 1 and len(window_layers) > 0
        all_layers = sorted(set(final_layers) | (set(window_layers) if do_window else set()))
        assert all_layers, "no layers requested"
        handles = self._register_multi_hooks(all_layers)
        try:
            return self._extract_bank_impl(texts, final_layers,
                                           window_layers if do_window else [],
                                           window, window_dtype, all_layers)
        finally:
            for h in handles:
                h.remove()

    def _extract_bank_impl(self, texts, final_layers, window_layers, window,
                           window_dtype, all_layers):
        results: list[dict[str, Any]] = []
        do_window = bool(window_layers)
        for start in range(0, len(texts), self.batch_size):
            sub = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                sub, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length, add_special_tokens=True,
            )
            device = self.model.get_input_embeddings().weight.device
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            self._forward_captured(input_ids, attention_mask, all_layers)

            B = input_ids.shape[0]
            lengths = attention_mask.sum(dim=1)                        # [B] on device
            last = (lengths - 1).clamp_min(0)
            bidx = torch.arange(B, device=device)
            final = {li: self._captured_multi[li][bidx, last].float().cpu()  # [B, d]
                     for li in final_layers}

            valid = torch.ones(B, dtype=torch.bool, device=device)
            per_layer_win: dict[int, torch.Tensor] = {}
            if do_window:
                offsets = torch.arange(window, device=device)          # [W]
                starts = lengths - window                              # [B]; <0 => too short
                valid = starts >= 0
                win_idx = (starts.clamp_min(0)[:, None] + offsets[None, :])  # [B, W]
                # Bound so a degenerate all-short batch (S < W) can't index out
                # of range; such rows are invalid and dropped downstream.
                win_idx = win_idx.clamp_(max=input_ids.shape[1] - 1)
                wbidx = bidx[:, None].expand(B, window)
                for li in window_layers:
                    w = self._captured_multi[li][wbidx, win_idx]       # [B, W, d]
                    if window_dtype == torch.float16 and bool(valid.any()):
                        # Guard BEFORE the cast, over valid rows only (invalid
                        # rows carry clamp-garbage that must not false-positive).
                        mx = w[valid].abs().max()
                        assert torch.isfinite(mx) and float(mx) < FLOAT16_MAX, (
                            f"layer {li}: max|coord|={float(mx):.1f} exceeds float16 "
                            f"range ({FLOAT16_MAX}). Re-run with --window-dtype float32."
                        )
                    per_layer_win[li] = w.to(window_dtype).cpu()       # [B, W, d]

            lengths_cpu = lengths.cpu().tolist()
            valid_cpu = valid.cpu().tolist()
            for i in range(B):
                seq_len = lengths_cpu[i]
                results.append({
                    "token_ids": input_ids[i, :seq_len].cpu().tolist(),
                    "final": {li: final[li][i] for li in final_layers},          # [d] fp32
                    "window": {li: per_layer_win[li][i] for li in window_layers},  # [W, d]
                    "valid": bool(valid_cpu[i]),
                })
        return results


def _schema(d_model: int, keep_text: bool) -> pa.Schema:
    fields = [("n_raw_tokens", pa.int64())]
    if keep_text:
        fields.append(("detokenized_text_truncated", pa.string()))
    fields += [
        ("activation_prev", pa.list_(pa.float32(), d_model)),
        ("activation_centre", pa.list_(pa.float32(), d_model)),
        ("activation_next", pa.list_(pa.float32(), d_model)),
        ("center_layer", pa.int64()),
        ("doc_id", pa.string()),
    ]
    return pa.schema(fields)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-model", required=True, help="HF model name/path (also the extractor provenance key)")
    p.add_argument("--corpus", required=True, help="HF dataset name or local .parquet path")
    p.add_argument("--corpus-config", default=None)
    p.add_argument("--corpus-split", default="train")
    p.add_argument("--corpus-start", type=int, default=0)
    p.add_argument("--corpus-length", type=int, required=True, help="number of documents to process")
    p.add_argument("--text-column", default="text")
    p.add_argument("--center-layer", type=int, default=24,
                   help="center block l; patch = {l-1, l, l+1}. Default 24 (Qwen3-8B 2/3 depth).")
    p.add_argument("--positions-per-doc", type=int, default=10)
    p.add_argument("--chunk-size", type=int, default=256, help="docs per extraction call / parquet write granularity")
    p.add_argument("--seed", type=int, default=42, help="position-sampling seed (keyed per doc_id)")
    p.add_argument("--keep-text", action=argparse.BooleanOptionalAction, default=True,
                   help="keep detokenized_text_truncated (needed to later regenerate/window these "
                        "rows; --no-keep-text shrinks the parquet for a probe-only run)")
    p.add_argument("--extractor-cls", default="multilayer_nla.extract_multilayer.MultiLayerHFExtractor")
    p.add_argument("--extractor-kwargs", default=None, help="JSON dict of extra extractor kwargs")
    p.add_argument("--output", required=True, help="output parquet path")
    add_storage_args(p)
    args = p.parse_args()

    center = args.center_layer
    layers = [center - 1, center, center + 1]
    assert center - 1 >= 0, f"center-layer={center} has no l-1 block"

    storage = make_storage(args)

    user_kwargs = parse_kwargs(args.extractor_kwargs)
    assert "model_name" not in user_kwargs, (
        "pass --base-model, not --extractor-kwargs '{\"model_name\": ...}' "
        "(kwargs would silently win and poison the sidecar provenance)."
    )
    extractor_kwargs = {"model_name": args.base_model, **user_kwargs}
    extractor = load_class(args.extractor_cls)(**extractor_kwargs)
    assert hasattr(extractor, "extract_multi"), (
        f"{args.extractor_cls} has no extract_multi(); multi-layer extraction needs it"
    )
    d_model = extractor.d_model
    tokenizer = extractor.tokenizer
    n_layers = len(resolve_decoder_layers(extractor.model))
    assert center + 1 < n_layers, (
        f"center-layer={center} needs block {center + 1}, but model has {n_layers} blocks"
    )
    keep_text = args.keep_text
    schema = _schema(d_model, keep_text)

    special_ids = set(tokenizer.all_special_ids)
    pad_id_to_check = (
        tokenizer.pad_token_id
        if (tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id)
        else None
    )

    import os
    if args.corpus.endswith(".parquet") and os.path.exists(args.corpus):
        ds = Dataset.from_parquet(args.corpus)
    else:
        ds = load_dataset(args.corpus, name=args.corpus_config, split=args.corpus_split)
    assert isinstance(ds, Dataset), (
        f"expected a concrete Dataset, got {type(ds).__name__} — pass an explicit --corpus-split"
    )
    ds = ds.select(range(args.corpus_start, args.corpus_start + args.corpus_length))

    storage.ensure_parent(args.output)
    row_count = 0
    n_docs_skipped = 0
    n_docs_short_sampled = 0

    with pq.ParquetWriter(storage.open_write(args.output), schema) as writer:
        for chunk_start in tqdm(range(0, len(ds), args.chunk_size), desc="chunks"):
            chunk = ds.select(range(chunk_start, min(chunk_start + args.chunk_size, len(ds))))
            texts = chunk[args.text_column]
            results = extractor.extract_multi(texts, layers)

            # Vectorized row build: accumulate per-doc numpy slices, then build
            # the FixedSizeList columns from contiguous buffers once per chunk
            # (no per-float Python boxing).
            prev_parts, centre_parts, next_parts = [], [], []
            nrt_col, did_col, cl_col, text_col = [], [], [], []
            for doc_offset, res in enumerate(results):
                doc_idx = args.corpus_start + chunk_start + doc_offset
                doc_id = f"{args.corpus}:{args.corpus_split}:{doc_idx}"
                token_ids = res["token_ids"]
                if pad_id_to_check is not None:
                    assert pad_id_to_check not in token_ids, (
                        f"pad_token_id {pad_id_to_check} found in token_ids for {doc_id} — "
                        f"the extractor's [:seq_len] slice is broken; all positions suspect."
                    )
                positions = _sample_positions(
                    token_ids, args.positions_per_doc, special_ids, doc_id, args.seed,
                )
                if not positions:
                    n_docs_skipped += 1
                    continue
                if len(positions) < args.positions_per_doc:
                    n_docs_short_sampled += 1
                pos_idx = torch.as_tensor(positions, dtype=torch.long)
                h = res["hidden"]
                # advanced-index -> [n_pos, d] float32. RAW vectors — the
                # normalization decision lives downstream.
                prev_parts.append(h[center - 1][pos_idx].numpy())
                centre_parts.append(h[center][pos_idx].numpy())
                next_parts.append(h[center + 1][pos_idx].numpy())
                for pos in positions:
                    nrt_col.append(pos + 1)
                    did_col.append(doc_id)
                    cl_col.append(center)
                    if keep_text:
                        text_col.append(
                            tokenizer.decode(token_ids[: pos + 1], skip_special_tokens=True)
                        )

            if did_col:
                def _fsl(parts):
                    flat = np.concatenate(parts, axis=0).reshape(-1).astype(np.float32, copy=False)
                    return pa.FixedSizeListArray.from_arrays(pa.array(flat), d_model)
                cols = {
                    "n_raw_tokens": pa.array(nrt_col, pa.int64()),
                    "activation_prev": _fsl(prev_parts),
                    "activation_centre": _fsl(centre_parts),
                    "activation_next": _fsl(next_parts),
                    "center_layer": pa.array(cl_col, pa.int64()),
                    "doc_id": pa.array(did_col, pa.string()),
                }
                if keep_text:
                    cols["detokenized_text_truncated"] = pa.array(text_col, pa.string())
                writer.write_table(pa.table(cols, schema=schema))
                row_count += len(did_col)

    corpus_slice = {"start": args.corpus_start, "length": args.corpus_length}
    manifest = build_manifest(
        stage="stage0_multilayer_extract",
        tokenizer=tokenizer,
        extra={
            "base_model": args.base_model,
            "layer_triplet": layers,
            "center_layer": center,
            "corpus": args.corpus,
            "corpus_slice": corpus_slice,
            "position_seed": args.seed,
            "positions_per_doc": args.positions_per_doc,
            "d_model": d_model,
        },
    )
    meta = {
        "kind": "mlnla_dataset",
        "schema_version": 1,
        "stage": "base_multilayer",
        "base_model": args.base_model,
        "d_model": d_model,
        "center_layer": center,
        "layers": layers,                 # output-of-block indices (stage0 semantics)
        "layer_offsets": [-1, 0, 1],
        "norm": "none",                   # RAW — invariant
        "corpus": args.corpus,
        "corpus_slice": corpus_slice,
        "positions_per_doc": args.positions_per_doc,
        "seed": args.seed,
        "row_count": row_count,
        "keep_text": keep_text,
        "dataset_id": _dataset_id(args.base_model, center, args.corpus, corpus_slice),
        "manifest": manifest,
    }
    meta_path = args.output + ".mlnla_meta.yaml"
    storage.write_text(meta_path, yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))

    print(f"wrote {row_count} rows ({d_model}-dim x3 layers {layers}) -> {args.output}")
    print(f"  skipped {n_docs_skipped} docs (too short / all-special past position {_MIN_POSITION})")
    print(f"  short-sampled {n_docs_short_sampled} docs (< {args.positions_per_doc} valid positions)")
    print(f"sidecar -> {meta_path}")
    print(json.dumps({"row_count": row_count, "layers": layers, "d_model": d_model}, indent=2))


if __name__ == "__main__":
    main()
