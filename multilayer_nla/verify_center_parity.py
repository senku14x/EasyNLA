"""Plan §12.5 step 1: verify the multi-layer center tap a^(l) is byte-identical
to legacy single-layer extraction at layer l.

This is the decisive correctness gate for the three-hook plumbing. The multi-layer
extractor captures {l-1, l, l+1} in one forward; its center tap MUST equal what
`nla.datagen.stage0_extract` (via `HFExtractor.extract`) produces at layer l, or
every downstream FVE number is built on a different vector than the controls.

Both paths run on the SAME model instance, SAME tokenization, SAME hook
mechanism (register_forward_hook -> output[0]), and the forward is deterministic
(eval, no dropout, use_cache=False). So we expect EXACT equality; we report the
max abs diff and fail if it exceeds --atol (default 0, i.e. bitwise).

Run on a small number of real corpus docs (no sampling — compares the full
per-position hidden states). GPU/H200 only (loads the base model).
"""

import argparse
import os

import torch
from datasets import Dataset, load_dataset

from nla.utils.arch_adapters import resolve_decoder_layers
from multilayer_nla.extract_multilayer import MultiLayerHFExtractor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True)
    p.add_argument("--corpus", required=True, help="HF dataset name or local .parquet path")
    p.add_argument("--corpus-config", default=None)
    p.add_argument("--corpus-split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--center-layer", type=int, default=24)
    p.add_argument("--n-docs", type=int, default=16, help="docs to compare (full sequences)")
    p.add_argument("--atol", type=float, default=0.0,
                   help="max allowed abs diff between center tap and legacy. Default 0 (bitwise).")
    p.add_argument("--extractor-kwargs", default=None)
    args = p.parse_args()

    center = args.center_layer
    triplet = [center - 1, center, center + 1]

    kwargs = {}
    if args.extractor_kwargs:
        import json
        kwargs = json.loads(args.extractor_kwargs)
    extractor = MultiLayerHFExtractor(model_name=args.base_model, **kwargs)
    n_layers = len(resolve_decoder_layers(extractor.model))
    assert 0 <= center - 1 and center + 1 < n_layers, (
        f"center {center} invalid for {n_layers}-layer model"
    )

    if args.corpus.endswith(".parquet") and os.path.exists(args.corpus):
        ds = Dataset.from_parquet(args.corpus)
    else:
        ds = load_dataset(args.corpus, name=args.corpus_config, split=args.corpus_split)
    texts = ds.select(range(min(args.n_docs, len(ds))))[args.text_column]

    print(f"[parity] center layer {center}, triplet {triplet}, {len(texts)} docs")

    # Legacy single-layer path (what stage0_extract calls) and the multi path,
    # on the SAME extractor/model instance.
    legacy = extractor.extract(texts, center)               # list[ExtractionResult]
    multi = extractor.extract_multi(texts, triplet)         # list[{token_ids, hidden}]

    assert len(legacy) == len(multi), f"doc count mismatch: {len(legacy)} vs {len(multi)}"

    max_abs = 0.0
    total_positions = 0
    for i, (lr, mr) in enumerate(zip(legacy, multi)):
        assert lr.token_ids == mr["token_ids"], f"doc {i}: token_ids differ (tokenization drift)"
        center_tap = mr["hidden"][center]                   # [seq_len, d]
        leg = lr.hidden_states                              # [seq_len, d]
        assert center_tap.shape == leg.shape, (
            f"doc {i}: shape {tuple(center_tap.shape)} vs legacy {tuple(leg.shape)}"
        )
        diff = (center_tap - leg).abs().max().item()
        max_abs = max(max_abs, diff)
        total_positions += leg.shape[0]

    print(f"[parity] compared {total_positions} positions across {len(legacy)} docs")
    print(f"[parity] max abs diff (center tap vs legacy) = {max_abs:.3e}")
    if max_abs <= args.atol:
        print(f"[parity] PASS — center tap matches legacy layer-{center} extraction "
              f"(<= atol {args.atol:g})")
    else:
        raise SystemExit(
            f"[parity] FAIL — max abs diff {max_abs:.3e} > atol {args.atol:g}. "
            f"The multi-layer center tap diverges from legacy extraction; downstream "
            f"FVE would be built on a different vector than the single-layer control."
        )


if __name__ == "__main__":
    main()
