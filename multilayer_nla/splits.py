"""Document-level train/dev/test splits for the §7 SFT control sweep.

Two INDEPENDENT splits, each a stable hash of doc_id (reusing datasets.doc_bucket,
so a whole document lands in exactly one bucket — never split a doc across buckets,
no row of an eval doc leaks into training):

  rl  source -> rl_train (80%) / rl_dev (10%) / rl_test (10%)   [end-to-end eval]
  ar  source -> ar_train (80%) / ar_dev (10%) / ar_test (10%)   [AR-only gold eval]

The partition is IMPLICIT in the hash (doc_bucket(doc_id, fracs, seed)); we do NOT
materialize per-bucket copies of the ~180 GB activation bank. This tool scans only
the doc_id column, tallies doc/row counts per bucket, hashes each bucket's doc_id
set, asserts zero overlap, and writes a JSON manifest. build_sweep.py applies the
SAME doc_bucket to route rows when it constructs the per-bucket datasets, so the
on-disk datasets match the manifest by construction.

Optionally locks a deterministic dev/test SUBSET (the N docs with smallest
hash(seed,'subset',doc_id) — a uniform sample, fixed BEFORE any results are seen)
for the expensive end-to-end rollout eval.

Run:
  python -m multilayer_nla.splits --source 'regen/rl.shard*.parquet'     --name rl \
      --out-dir $SWEEP --seed 42 --fracs 0.8,0.1,0.1 --dev-subset 256 --test-subset 1000
  python -m multilayer_nla.splits --source 'regen/ar_sft.shard*.parquet' --name ar \
      --out-dir $SWEEP --seed 42 --fracs 0.8,0.1,0.1
"""

import argparse
import glob
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from multilayer_nla.datasets import doc_bucket

DEFAULT_FRACS = (0.8, 0.1, 0.1)
DEFAULT_NAMES = ("train", "dev", "test")


def expand_source(source: str) -> list:
    """A path or glob -> sorted list of parquet paths (errors if nothing matches)."""
    paths = sorted(glob.glob(source)) if any(c in source for c in "*?[") else [source]
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        raise SystemExit(f"no files match --source {source!r}")
    return paths


def scan_doc_buckets(source_paths, fracs, names, seed):
    """Scan ONLY the doc_id column; return (doc_to_bucket, row_counts).

    doc_to_bucket: {doc_id: bucket_name} (a doc maps to exactly one bucket).
    row_counts:    {bucket_name: n_rows}.
    """
    doc_to_bucket, row_counts = {}, {nm: 0 for nm in names}
    for p in source_paths:
        pf = pq.ParquetFile(p)
        if "doc_id" not in pf.schema_arrow.names:
            raise SystemExit(f"{p} has no doc_id column — cannot split by document.")
        for rg in range(pf.num_row_groups):
            for d in pf.read_row_group(rg, columns=["doc_id"]).column("doc_id").to_pylist():
                nm = names[doc_bucket(d, fracs, seed)]
                doc_to_bucket[d] = nm
                row_counts[nm] += 1
    return doc_to_bucket, row_counts


def _sha(strings) -> str:
    """Order-independent sha256 of a set of strings (a stable record of the bucket)."""
    h = hashlib.sha256()
    for s in sorted(strings):
        h.update(s.encode()); h.update(b"\0")
    return h.hexdigest()


def locked_subset(doc_ids, n, seed):
    """Deterministic uniform sample: the n doc_ids with smallest hash(seed,'subset',doc).

    Sorting by a hash (not the raw id) avoids clustering by id prefix, and is fixed by
    (seed, n) alone — so the subset is locked before any results are seen.
    """
    ordered = sorted(doc_ids, key=lambda d: hashlib.sha256(f"{seed}|subset|{d}".encode()).hexdigest())
    return sorted(ordered if (n is None or n >= len(ordered)) else ordered[:n])


def build_split_manifest(source_paths, name, out_dir, *, fracs=DEFAULT_FRACS,
                         names=DEFAULT_NAMES, seed=42, dev_subset=None, test_subset=None):
    doc_to_bucket, row_counts = scan_doc_buckets(source_paths, fracs, names, seed)
    buckets = {nm: [] for nm in names}
    for d, nm in doc_to_bucket.items():
        buckets[nm].append(d)

    # Prove pairwise disjointness (it holds by construction; assert it anyway).
    seen = {}
    for nm, ds in buckets.items():
        for d in ds:
            assert d not in seen, f"{name}: doc {d} routed to both {seen[d]} and {nm}"
            seen[d] = nm

    manifest = {
        "name": name,
        "seed": seed,
        "fracs": list(fracs),
        "names": list(names),
        "sources": [str(p) for p in source_paths],
        "n_docs": {nm: len(buckets[nm]) for nm in names},
        "n_rows": {nm: row_counts[nm] for nm in names},
        "doc_set_sha256": {nm: _sha(buckets[nm]) for nm in names},
        "n_docs_total": len(doc_to_bucket),
        "n_rows_total": sum(row_counts.values()),
    }
    # Locked deterministic subsets for expensive end-to-end eval (dev=names[1], test=names[2]).
    subsets = {}
    if dev_subset and len(names) > 1:
        subsets[names[1]] = locked_subset(buckets[names[1]], dev_subset, seed)
    if test_subset and len(names) > 2:
        subsets[names[2]] = locked_subset(buckets[names[2]], test_subset, seed)
    if subsets:
        manifest["locked_subsets"] = subsets
        manifest["locked_subset_sizes"] = {nm: len(ds) for nm, ds in subsets.items()}
        manifest["locked_subset_sha256"] = {nm: _sha(ds) for nm, ds in subsets.items()}

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(out_dir) / f"{name}_split_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[split:{name}] seed={seed} fracs={fracs} -> {out}")
    print(f"[split:{name}] docs={manifest['n_docs']}  rows={manifest['n_rows']}")
    if subsets:
        print(f"[split:{name}] locked subsets: " +
              ", ".join(f"{nm}={len(ds)}" for nm, ds in subsets.items()))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="parquet path or glob (regenerated rl/ar_sft shards)")
    p.add_argument("--name", required=True, help="split name prefix written as <name>_split_manifest.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fracs", default="0.8,0.1,0.1")
    p.add_argument("--names", default="train,dev,test")
    p.add_argument("--dev-subset", type=int, default=None,
                   help="lock a deterministic N-doc subset of the dev bucket (cheap ckpt selection)")
    p.add_argument("--test-subset", type=int, default=None,
                   help="lock a deterministic N-doc subset of the test bucket (final eval if full too big)")
    args = p.parse_args()
    fracs = tuple(float(x) for x in args.fracs.split(","))
    names = tuple(x.strip() for x in args.names.split(","))
    assert abs(sum(fracs) - 1.0) < 1e-6, f"--fracs must sum to 1, got {sum(fracs)}"
    assert len(fracs) == len(names), "--fracs and --names must align"
    build_split_manifest(expand_source(args.source), args.name, args.out_dir,
                         fracs=fracs, names=names, seed=args.seed,
                         dev_subset=args.dev_subset, test_subset=args.test_subset)


if __name__ == "__main__":
    main()
