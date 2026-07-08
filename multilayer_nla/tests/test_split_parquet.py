"""CLI wrapper test for split_parquet (split_by_document itself is covered by
test_datasets): output files, prefix naming, manifest contents, doc
disjointness, and determinism of the doc buckets across re-runs.

Run: python -m pytest multilayer_nla/tests/test_split_parquet.py -q
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from multilayer_nla import split_parquet


def _write_built(path: str, n_docs=40, rows_per_doc=3, d=4):
    rng = np.random.default_rng(0)
    doc_ids, vals = [], []
    for di in range(n_docs):
        for _ in range(rows_per_doc):
            doc_ids.append(f"doc:{di}")
            vals.append(rng.standard_normal(d).astype(np.float32))
    flat = np.ascontiguousarray(np.stack(vals)).reshape(-1)
    pq.write_table(pa.table({
        "doc_id": pa.array(doc_ids, pa.string()),
        "activation_centre": pa.FixedSizeListArray.from_arrays(pa.array(flat), d),
        "prompt": pa.array([f"p{i}" for i in range(len(doc_ids))]),
    }), path)


def _run(argv):
    old = sys.argv
    sys.argv = ["split_parquet"] + argv
    try:
        split_parquet.main()
    finally:
        sys.argv = old


def test_split_outputs_manifest_and_disjointness():
    with tempfile.TemporaryDirectory() as td:
        inp = f"{td}/av_sft.parquet"
        _write_built(inp)
        _run(["--in", inp, "--out-dir", td, "--prefix", "av_sft",
              "--fracs", "0.8,0.2", "--names", "train,val", "--seed", "42"])
        tr = pq.read_table(f"{td}/av_sft_train.parquet")
        va = pq.read_table(f"{td}/av_sft_val.parquet")
        assert tr.num_rows + va.num_rows == 120
        assert tr.schema.names == va.schema.names  # every column carried
        # doc-level: no doc appears in both; each doc's rows stay together
        tr_docs, va_docs = set(tr.column("doc_id").to_pylist()), set(va.column("doc_id").to_pylist())
        assert not (tr_docs & va_docs)
        m = json.loads(Path(f"{td}/av_sft_split_manifest.json").read_text())
        assert m["seed"] == 42
        assert m["n_docs"]["av_sft_train"] == len(tr_docs)
        assert m["n_docs"]["av_sft_val"] == len(va_docs)


def test_same_seed_reproduces_buckets():
    with tempfile.TemporaryDirectory() as td:
        inp = f"{td}/x.parquet"
        _write_built(inp)
        _run(["--in", inp, "--out-dir", f"{td}/a", "--prefix", "x", "--seed", "7"])
        _run(["--in", inp, "--out-dir", f"{td}/b", "--prefix", "x", "--seed", "7"])
        ma = json.loads(Path(f"{td}/a/x_split_manifest.json").read_text())
        mb = json.loads(Path(f"{td}/b/x_split_manifest.json").read_text())
        assert ma["doc_set_sha256"] == mb["doc_set_sha256"]  # identical doc buckets


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print("\nALL PASSED")


if __name__ == "__main__":
    _run_all()
