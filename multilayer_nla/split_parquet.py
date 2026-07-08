"""Materialize a document-level split of a built training parquet.

Thin CLI over `datasets.split_by_document` for the flat published subsets
(ceselder av_sft/ar_sft/rl have no train/val split). Routes every row by
doc_bucket(seed, doc_id) — all positions of a doc land in ONE bucket — carries
every column, and writes a JSON manifest (per-bucket doc/row counts + doc-set
hashes, disjointness asserted by construction).

Determinism note: the bucket is a pure hash of (seed, doc_id), so re-running
after a different --center build (or on a re-regenerated bank) reproduces the
SAME doc buckets — eval docs stay fixed across center sweeps by construction.

Usage:
    python -m multilayer_nla.split_parquet --in $TRAIN/av_sft.parquet \\
        --out-dir $TRAIN --prefix av_sft --fracs 0.9,0.1 --names train,val --seed 42
    # -> $TRAIN/av_sft_train.parquet + $TRAIN/av_sft_val.parquet + manifest
"""

import argparse
import hashlib
import json
from pathlib import Path

from multilayer_nla.datasets import split_by_document


def _sha(strings) -> str:
    h = hashlib.sha256()
    for s in sorted(strings):
        h.update(s.encode())
        h.update(b"\0")
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True, help="built training parquet (has doc_id)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", required=True,
                   help="output basename prefix; files are <prefix>_<name>.parquet")
    p.add_argument("--fracs", default="0.9,0.1")
    p.add_argument("--names", default="train,val")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    fracs = tuple(float(x) for x in args.fracs.split(","))
    names = tuple(f"{args.prefix}_{x.strip()}" for x in args.names.split(","))
    assert abs(sum(fracs) - 1.0) < 1e-6, f"--fracs must sum to 1, got {sum(fracs)}"
    assert len(fracs) == len(names), "--fracs and --names must align"

    paths = split_by_document(args.inp, args.out_dir, fracs=fracs, names=names, seed=args.seed)

    # Manifest: per-bucket doc sets, hashed, disjointness asserted.
    import pyarrow.parquet as pq
    docs = {}
    for nm, path in paths.items():
        docs[nm] = set(pq.read_table(path, columns=["doc_id"]).column("doc_id").to_pylist())
    ordered = list(docs)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            inter = docs[a] & docs[b]
            assert not inter, f"{len(inter)} doc_id(s) in both {a} and {b} — split broken"
    manifest = {
        "source": args.inp,
        "seed": args.seed,
        "fracs": list(fracs),
        "outputs": paths,
        "n_docs": {nm: len(docs[nm]) for nm in ordered},
        "doc_set_sha256": {nm: _sha(docs[nm]) for nm in ordered},
    }
    out = Path(args.out_dir) / f"{args.prefix}_split_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[split-parquet] {args.inp} -> {list(paths.values())}")
    print(f"[split-parquet] docs={manifest['n_docs']} (disjoint OK) manifest -> {out}")


if __name__ == "__main__":
    main()
