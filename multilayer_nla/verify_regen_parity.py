"""Smoke gate: the regenerate path's bank activation_L{c} (final-token, GPU-gathered)
== legacy single-layer extraction at layer c, on ACTUAL published prefixes.

verify_center_parity.py proves the multi-hook center tap matches legacy on the
full per-position sequence (corpus-sampling path). THIS proves the regenerate
path specifically — final-token-only capture with the on-GPU gather
(extract_multi(final_token_only=True), the Blocker-1 fix) — reproduces exactly
what the single-layer `tools/regenerate_activations.py` would compute
(HFExtractor.extract(...).hidden_states[-1]) on the same `detokenized_text_truncated`
rows. If these match bitwise, the bank's L{c} vector is the real layer-c
activation the published label describes.

GPU/H200 only (loads the base model). Run on a small slice of the downloaded
published parquet (which has detokenized_text_truncated + n_raw_tokens):

    python -m multilayer_nla.verify_regen_parity \\
        --base-model Qwen/Qwen3-8B --parquet $PUB/av_sft.parquet \\
        --center-layer 24 --n-rows 64 --max-length 4096
"""

import argparse

import pyarrow.parquet as pq

from nla.utils.arch_adapters import resolve_decoder_layers
from multilayer_nla.extract_multilayer import MultiLayerHFExtractor

TEXT_COL = "detokenized_text_truncated"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True)
    p.add_argument("--parquet", required=True, help="published slim parquet (has detokenized_text_truncated)")
    p.add_argument("--center-layer", type=int, default=24)
    p.add_argument("--n-rows", type=int, default=64)
    p.add_argument("--max-length", type=int, default=4096, help="MUST match the published extraction")
    p.add_argument("--atol", type=float, default=0.0, help="max allowed abs diff (default 0 = bitwise)")
    p.add_argument("--extractor-kwargs", default=None)
    args = p.parse_args()

    center = args.center_layer
    kwargs = {"max_length": args.max_length}
    if args.extractor_kwargs:
        import json
        kwargs.update(json.loads(args.extractor_kwargs))
    extractor = MultiLayerHFExtractor(model_name=args.base_model, **kwargs)
    n_layers = len(resolve_decoder_layers(extractor.model))
    assert 0 <= center - 1 and center + 1 < n_layers, f"center {center} invalid for {n_layers} layers"

    pf = pq.ParquetFile(args.parquet)
    assert TEXT_COL in pf.schema_arrow.names, f"{args.parquet} lacks {TEXT_COL!r}"
    tbl = next(pf.iter_batches(batch_size=args.n_rows, columns=[TEXT_COL]))
    texts = tbl.column(TEXT_COL).to_pylist()
    print(f"[regen-parity] center {center}, {len(texts)} published rows, max_length {args.max_length}")

    # Regenerate path: final-token-only, on-GPU gather (the Blocker-1 fix).
    multi = extractor.extract_multi(texts, [center], final_token_only=True)   # hidden[center] = [d]
    # Legacy single-layer path (what tools/regenerate_activations.py uses): final token.
    legacy = extractor.extract(texts, center)                                 # .hidden_states = [seq, d]
    assert len(multi) == len(legacy) == len(texts)

    max_abs = 0.0
    for i, (m, lr) in enumerate(zip(multi, legacy)):
        assert m["token_ids"] == lr.token_ids, f"row {i}: token_ids differ (tokenization drift)"
        bank = m["hidden"][center]              # [d]  (regenerate path)
        leg = lr.hidden_states[-1]              # [d]  (final token of legacy)
        assert bank.shape == leg.shape, f"row {i}: shape {tuple(bank.shape)} vs {tuple(leg.shape)}"
        max_abs = max(max_abs, (bank - leg).abs().max().item())

    print(f"[regen-parity] max abs diff (bank L{center} final token vs legacy) = {max_abs:.3e}")
    if max_abs <= args.atol:
        print(f"[regen-parity] PASS — the GPU-gathered final-token bank vector matches legacy "
              f"layer-{center} extraction (<= atol {args.atol:g}).")
    else:
        raise SystemExit(
            f"[regen-parity] FAIL — max abs diff {max_abs:.3e} > atol {args.atol:g}. The "
            f"final-token gather diverges from legacy extraction; the bank's L{center} vector "
            f"is not the activation the published label describes."
        )


if __name__ == "__main__":
    main()
