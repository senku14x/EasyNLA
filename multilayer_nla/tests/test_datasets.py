"""Validate multi-vector dataset loading (plan §6.1, point 2): the doc-level
split preserves ALL THREE activation columns and never splits a doc; the
slot-stacking matches the injection scan order.

Run:  python -m multilayer_nla.tests.test_datasets
"""

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from multilayer_nla.datasets import (
    SLOT_COLUMNS,
    av_in_columns,
    av_user_content,
    build_av_prompt,
    detect_av_slots,
    doc_bucket,
    split_by_document,
    stack_slot_vectors,
)


def _make_base_parquet(path, n_docs=120, ppd=5, d=8):
    rng = np.random.default_rng(0)
    prev, centre, nxt, dids, nrt, cl = [], [], [], [], [], []
    for doc in range(n_docs):
        for p in range(ppd):
            # encode (doc, pos, slot) into the values so we can verify exact preservation
            prev.append(np.full(d, doc * 1000 + p, dtype=np.float32))
            centre.append(np.full(d, doc * 1000 + p + 0.5, dtype=np.float32))
            nxt.append(np.full(d, doc * 1000 + p + 0.25, dtype=np.float32))
            dids.append(f"c:train:{doc}")
            nrt.append(p + 60)
            cl.append(24)

    def fsl(arrs):
        flat = np.concatenate([a.reshape(1, -1) for a in arrs]).reshape(-1).astype(np.float32)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat), d)

    schema = pa.schema([
        ("n_raw_tokens", pa.int64()),
        ("activation_prev", pa.list_(pa.float32(), d)),
        ("activation_centre", pa.list_(pa.float32(), d)),
        ("activation_next", pa.list_(pa.float32(), d)),
        ("center_layer", pa.int64()),
        ("doc_id", pa.string()),
    ])
    tbl = pa.table({
        "n_raw_tokens": pa.array(nrt, pa.int64()),
        "activation_prev": fsl(prev),
        "activation_centre": fsl(centre),
        "activation_next": fsl(nxt),
        "center_layer": pa.array(cl, pa.int64()),
        "doc_id": pa.array(dids, pa.string()),
    }, schema=schema)
    pq.write_table(tbl, path, row_group_size=97)  # odd size -> rows split across groups
    return n_docs, ppd, d


def test_split_doc_level_and_preserves_all_columns():
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "base.parquet")
        n_docs, ppd, d = _make_base_parquet(base)
        paths = split_by_document(base, str(Path(tmp) / "splits"),
                                  fracs=(0.25, 0.25, 0.5),
                                  names=("av_sft", "ar_sft", "rl"), seed=42)

        base_tbl = pq.read_table(base)
        # 1) all three activation columns survive in every split
        for nm, p in paths.items():
            t = pq.read_table(p)
            assert set(SLOT_COLUMNS).issubset(set(t.schema.names)), f"{nm} dropped activation columns"
            assert t.schema.names == base_tbl.schema.names, f"{nm} schema changed"

        # 2) partition is exact: every base row appears in exactly one split, no loss
        total = sum(pq.read_table(p).num_rows for p in paths.values())
        assert total == base_tbl.num_rows == n_docs * ppd

        # 3) document-level: every doc's rows are entirely within one split
        doc_to_split = {}
        for nm, p in paths.items():
            for did in set(pq.read_table(p).column("doc_id").to_pylist()):
                assert did not in doc_to_split, f"doc {did} appears in {nm} AND {doc_to_split[did]}"
                doc_to_split[did] = nm
        assert len(doc_to_split) == n_docs

        # 4) values preserved exactly for a sampled doc (centre column carries doc*1000+p+0.5)
        for nm, p in paths.items():
            t = pq.read_table(p)
            if t.num_rows == 0:
                continue
            dids = t.column("doc_id").to_pylist()
            nrts = t.column("n_raw_tokens").to_pylist()
            col = t.column("activation_centre").combine_chunks()
            centre = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(col), -1)
            for i in range(min(5, t.num_rows)):
                doc = int(dids[i].split(":")[-1])
                pos = nrts[i] - 60
                assert np.allclose(centre[i], doc * 1000 + pos + 0.5), f"value corrupted in {nm}"

        # 5) deterministic
        paths2 = split_by_document(base, str(Path(tmp) / "splits2"),
                                   fracs=(0.25, 0.25, 0.5), seed=42)
        for nm in paths:
            assert (pq.read_table(paths[nm]).column("doc_id").to_pylist()
                    == pq.read_table(paths2[nm]).column("doc_id").to_pylist())


def test_doc_bucket_routing_and_balance():
    fracs = (0.25, 0.25, 0.5)
    # same doc_id -> same bucket (document-level guarantee at the routing level)
    assert doc_bucket("c:train:5", fracs, 42) == doc_bucket("c:train:5", fracs, 42)
    counts = [0, 0, 0]
    for i in range(3000):
        counts[doc_bucket(f"c:train:{i}", fracs, 42)] += 1
    frac = [c / 3000 for c in counts]
    assert abs(frac[0] - 0.25) < 0.04 and abs(frac[1] - 0.25) < 0.04 and abs(frac[2] - 0.5) < 0.04, frac


def test_stack_slot_vectors_order():
    d = 4
    cols = av_in_columns(3)  # av_in_0/1/2 (positional AV-input slots)
    rows = [
        {cols[0]: np.arange(d) + 0.0, cols[1]: np.arange(d) + 10.0, cols[2]: np.arange(d) + 20.0},
        {cols[0]: np.arange(d) + 100.0, cols[1]: np.arange(d) + 110.0, cols[2]: np.arange(d) + 120.0},
    ]
    v = stack_slot_vectors(rows, cols)  # [B*k, d] = [6, d]
    assert v.shape == (6, d)
    # example-major, slot order [0,1,2] — the exact order the injection scan
    # (row-major over markers) walks.
    assert np.allclose(v[0], np.arange(d) + 0.0)
    assert np.allclose(v[1], np.arange(d) + 10.0)
    assert np.allclose(v[2], np.arange(d) + 20.0)
    assert np.allclose(v[3], np.arange(d) + 100.0)
    assert np.allclose(v[4], np.arange(d) + 110.0)
    assert np.allclose(v[5], np.arange(d) + 120.0)
    # k=1 (single condition): one slot per example, example-major
    c1 = av_in_columns(1)
    v1 = stack_slot_vectors([{c1[0]: np.arange(d) + 7.0}, {c1[0]: np.arange(d) + 8.0}], c1)
    assert v1.shape == (2, d)
    assert np.allclose(v1[0], np.arange(d) + 7.0) and np.allclose(v1[1], np.arange(d) + 8.0)


def test_av_prompt_slot_counts():
    """The AV prompt carries exactly k marker placeholders: 3 for local/duplicate/wide,
    2 for the k=2 combos (e.g. [L20,L24] / [L24,L28]), 1 for single. Marker-count drift
    breaks injection, so k not in {1,2,3} must raise."""
    from nla.schema import INJECT_PLACEHOLDER
    assert av_user_content(3).count(INJECT_PLACEHOLDER) == 3
    assert av_user_content(2).count(INJECT_PLACEHOLDER) == 2
    assert av_user_content(1).count(INJECT_PLACEHOLDER) == 1
    assert build_av_prompt(3)[0]["content"].count(INJECT_PLACEHOLDER) == 3
    assert build_av_prompt(2)[0]["content"].count(INJECT_PLACEHOLDER) == 2
    assert build_av_prompt(1)[0]["content"].count(INJECT_PLACEHOLDER) == 1
    for bad in (0, 4, 5):
        try:
            av_user_content(bad)
            assert False, f"k={bad} must raise"
        except ValueError:
            pass


def test_detect_av_slots_order_and_isolation():
    """detect_av_slots returns the av_in_* columns in slot order (0,1,2,...) and
    ignores everything else (the prompt, the response, the AR-target columns)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "av.parquet")
        d = 4

        def fsl(vals):
            flat = np.asarray(vals, np.float32).reshape(-1)
            return pa.FixedSizeListArray.from_arrays(pa.array(flat), d)

        tbl = pa.table({
            "prompt": pa.array(["p", "p"]),
            "response": pa.array(["a", "b"]),
            "av_in_0": fsl([[1] * d, [2] * d]),
            "av_in_2": fsl([[5] * d, [6] * d]),          # out of order on disk
            "av_in_1": fsl([[3] * d, [4] * d]),
            "activation_centre": fsl([[9] * d, [9] * d]),  # AR-target col -> must be ignored
        })
        pq.write_table(tbl, path)
        assert detect_av_slots(path) == ["av_in_0", "av_in_1", "av_in_2"]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} dataset tests passed.")


if __name__ == "__main__":
    _run_all()
