"""Validate dataset assembly (plan §6.1, point 2): split parquet -> av/ar/rl
training parquets in the exact format the trainers' loaders expect. Dummy
explanations (no API). Run: python -m multilayer_nla.tests.test_build_datasets
"""

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from multilayer_nla.build_datasets import assemble, dummy_explanation
from multilayer_nla.datasets import (
    INJECT_PLACEHOLDER,
    N_SLOTS,
    SLOT_COLUMNS,
    load_ar_sft_dataset,
    load_av_sft_dataset,
)


def _make_split(path, n=40, d=8, with_explanation_col=False, n_empty=0):
    rng = np.random.default_rng(0)
    acts = {c: [rng.standard_normal(d).astype(np.float32) for _ in range(n)] for c in SLOT_COLUMNS}

    def fsl(arrs):
        flat = np.concatenate([a.reshape(1, -1) for a in arrs]).reshape(-1).astype(np.float32)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat), d)

    cols = {
        "detokenized_text_truncated": pa.array([f"some web text number {i} that ends" for i in range(n)]),
        **{c: fsl(acts[c]) for c in SLOT_COLUMNS},
        "center_layer": pa.array([24] * n, pa.int64()),
        "doc_id": pa.array([f"c:train:{i}" for i in range(n)], pa.string()),
    }
    if with_explanation_col:
        expl = [f"real explanation {i}" for i in range(n)]
        for i in range(n_empty):
            expl[i] = ""  # these must be dropped
        cols["api_explanation"] = pa.array(expl, pa.string())
    pq.write_table(pa.table(cols), path)
    return acts


def test_assemble_av_format():
    with tempfile.TemporaryDirectory() as tmp:
        split = str(Path(tmp) / "av_split.parquet")
        acts = _make_split(split)
        out = str(Path(tmp) / "av_sft.parquet")
        n = assemble(split, "av", out, dummy=True)
        assert n == 40
        rows = load_av_sft_dataset(out)
        assert len(rows) == 40
        r = rows[0]
        # prompt is a chat message list with exactly N_SLOTS markers
        content = r["prompt"][0]["content"]
        assert content.count(INJECT_PLACEHOLDER) == N_SLOTS
        # response wrapped in explanation tags
        assert r["response"].startswith("<explanation>") and r["response"].rstrip().endswith("</explanation>")
        # all three activations present and preserved
        for j, c in enumerate(SLOT_COLUMNS):
            assert np.allclose(r[c], acts[c][0])


def test_assemble_ar_format():
    with tempfile.TemporaryDirectory() as tmp:
        split = str(Path(tmp) / "ar_split.parquet")
        _make_split(split)
        out = str(Path(tmp) / "ar_sft.parquet")
        assemble(split, "ar", out, dummy=True)
        rows = load_ar_sft_dataset(out)
        assert len(rows) == 40
        # ar prompt is a string with the suffix anchor and the explanation inside
        assert isinstance(rows[0]["prompt"], str)
        assert rows[0]["prompt"].endswith("<summary>")
        assert "<text>" in rows[0]["prompt"]
        assert all(c in rows[0] for c in SLOT_COLUMNS)


def test_assemble_rl_keeps_all_rows_no_explanation():
    with tempfile.TemporaryDirectory() as tmp:
        split = str(Path(tmp) / "rl_split.parquet")
        _make_split(split)
        out = str(Path(tmp) / "rl.parquet")
        n = assemble(split, "rl", out)  # rl needs no explanation, dummy not required
        assert n == 40
        t = pq.read_table(out)
        assert "prompt" in t.schema.names and "response" not in t.schema.names
        assert set(SLOT_COLUMNS).issubset(set(t.schema.names))


def test_real_explanation_column_used_and_empty_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        split = str(Path(tmp) / "av_split.parquet")
        _make_split(split, n=40, with_explanation_col=True, n_empty=5)
        out = str(Path(tmp) / "av_sft.parquet")
        n = assemble(split, "av", out, dummy=False)  # uses api_explanation, drops 5 empty
        assert n == 35
        rows = load_av_sft_dataset(out)
        assert all("real explanation" in r["response"] for r in rows)


def test_dummy_explanation_nonempty():
    assert len(dummy_explanation("a b c d e").strip()) > 20
    assert len(dummy_explanation("").strip()) > 20  # robust to empty text


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} build-datasets tests passed.")


if __name__ == "__main__":
    _run_all()
