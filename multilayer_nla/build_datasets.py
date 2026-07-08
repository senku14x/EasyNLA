"""Build multi-layer training parquets from a base 3-layer extraction.

    base_multilayer.parquet  --split-->  av_sft / ar_sft / rl  (carry 3 activations)
                             --explain-> +api_explanation  (reuse nla.datagen.stage2)
                             --assemble-> training parquets (prompt[/response] + 3 acts)

Two ways to get explanations:
  * REAL: run `nla.datagen.stage2_api_explain` on the av_sft / ar_sft split
    parquets between `split` and `assemble` (it append_column's api_explanation
    and carries the three activation columns through). Then `assemble` picks up
    that column.
  * DUMMY (--dummy-explanations): a deterministic templated explanation from the
    source tail — for the point-6 PLUMBING smoke only (no API, no cost). Do NOT
    report FVE from a dummy run.

The multi-layer trainers auto-pick the marker from the tokenizer and use the
per-row count guard, so these training parquets need NO sidecar (unlike the
single-layer contract).
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nla.schema import wrap_explanation
from multilayer_nla.datasets import (
    SLOT_COLUMNS,
    build_av_prompt,
    fill_ar_prompt,
    split_by_document,
)

EXPLANATION_COL = "api_explanation"
TEXT_COL = "detokenized_text_truncated"


def dummy_explanation(text: str) -> str:
    """Deterministic non-API explanation for the plumbing smoke (NOT for FVE)."""
    toks = (text or "").strip().split()
    tail = " ".join(toks[-4:]) if toks else "the prefix"
    return (
        "The passage maintains its established topic and register. "
        "A syntactic unit is in progress and expects grammatical completion. "
        f"The final tokens (\"{tail}\") sit mid-clause and constrain the next token to a continuation."
    )


def _explanations_for(table, *, dummy: bool) -> list[str | None]:
    if EXPLANATION_COL in table.schema.names:
        return table.column(EXPLANATION_COL).to_pylist()
    if dummy:
        assert TEXT_COL in table.schema.names, (
            f"--dummy-explanations needs the {TEXT_COL!r} column (extract with --keep-text)"
        )
        return [dummy_explanation(t) for t in table.column(TEXT_COL).to_pylist()]
    raise SystemExit(
        f"no {EXPLANATION_COL!r} column and --dummy-explanations not set. Run "
        f"nla.datagen.stage2_api_explain on the split first, or pass --dummy-explanations."
    )


def assemble(split_parquet: str, mode: str, out_path: str, *, dummy: bool = False) -> int:
    """Write one training parquet. Returns row count.

    av : prompt (3-marker list[struct]) + response (<explanation>..) + 3 acts
    ar : prompt (suffix-anchored critic str)            + 3 acts
    rl : prompt (3-marker list[struct])                 + 3 acts   (no explanation)
    """
    assert mode in ("av", "ar", "rl")
    table = pq.read_table(split_parquet)
    n_in = table.num_rows

    cols = {c: table.column(c) for c in SLOT_COLUMNS}
    cols["doc_id"] = table.column("doc_id")

    if mode == "rl":
        prompt = build_av_prompt()
        cols["prompt"] = pa.array([prompt] * table.num_rows)
        n_out = table.num_rows
    else:
        expls = _explanations_for(table, dummy=dummy)
        keep = [i for i, e in enumerate(expls) if e and e.strip()]
        if len(keep) < n_in:
            table = table.take(keep)
            cols = {c: table.column(c) for c in SLOT_COLUMNS}
            cols["doc_id"] = table.column("doc_id")
            expls = [expls[i] for i in keep]
        n_out = table.num_rows
        if mode == "av":
            cols["prompt"] = pa.array([build_av_prompt()] * n_out)
            cols["response"] = pa.array([wrap_explanation(e) for e in expls], pa.string())
        else:  # ar
            cols["prompt"] = pa.array([fill_ar_prompt(e) for e in expls], pa.string())

    out = pa.table(cols)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, out_path, row_group_size=4096)
    print(f"[assemble:{mode}] {split_parquet} -> {out_path}  ({n_out}/{n_in} rows"
          + (" )" if mode == "rl" else f", {n_in - n_out} dropped empty-explanation)"))
    return n_out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["split", "assemble", "all"], required=True)
    p.add_argument("--base", help="base_multilayer.parquet (for split / all)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fracs", default="0.25,0.25,0.5", help="av,ar,rl doc-split fractions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits-dir", default=None,
                   help="for --stage assemble: dir with av_sft/ar_sft/rl[_explained].parquet")
    p.add_argument("--dummy-explanations", action="store_true",
                   help="templated explanations, NO API — plumbing smoke only")
    args = p.parse_args()

    out = Path(args.out_dir)
    fracs = tuple(float(x) for x in args.fracs.split(","))

    if args.stage in ("split", "all"):
        assert args.base, "--base required for split"
        split_by_document(args.base, str(out / "splits"), fracs=fracs,
                          names=("av_sft", "ar_sft", "rl"), seed=args.seed)

    if args.stage == "split":
        return

    splits = Path(args.splits_dir) if args.splits_dir else (out / "splits")

    def pick(name):
        # prefer an explained parquet if stage2 produced one
        for cand in (splits / f"{name}_explained.parquet", splits / f"{name}.parquet"):
            if cand.exists():
                return str(cand)
        raise SystemExit(f"missing split parquet for {name} in {splits}")

    counts = {
        "av": assemble(pick("av_sft"), "av", str(out / "av_sft.parquet"), dummy=args.dummy_explanations),
        "ar": assemble(pick("ar_sft"), "ar", str(out / "ar_sft.parquet"), dummy=args.dummy_explanations),
        "rl": assemble(pick("rl"), "rl", str(out / "rl.parquet")),
    }
    (out / "build_manifest.json").write_text(json.dumps({
        "base": args.base, "fracs": list(fracs), "seed": args.seed,
        "dummy_explanations": args.dummy_explanations, "counts": counts,
    }, indent=2))
    print(f"[build] av/ar/rl -> {out}  counts={counts}"
          + ("  [DUMMY explanations — plumbing only]" if args.dummy_explanations else ""))


if __name__ == "__main__":
    main()
