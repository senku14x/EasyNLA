"""build_from_published: regenerated published rows -> our 3-slot training parquets.

Validates the per-subset adaptation (prompt swap for av/rl, verbatim canonical
critic prompt for ar, label preservation) and that the output round-trips through
the real trainers' loaders. Pure pyarrow + tmp parquet — no model, no API.
Run: python -m multilayer_nla.tests.test_build_from_published
"""

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from multilayer_nla.build_from_published import (
    _assert_docs_disjoint,
    _subset_inputs,
    assemble_published,
    build_one,
)
from multilayer_nla.datasets import (
    AR_CRITIC_TEMPLATE,
    INJECT_PLACEHOLDER,
    N_SLOTS,
    SLOT_COLUMNS,
    fill_ar_prompt,
    load_ar_sft_dataset,
    load_av_sft_dataset,
)

D = 8


def _fsl(arrs):
    flat = np.concatenate([a.reshape(1, -1) for a in arrs]).reshape(-1).astype(np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat), D)


def _triplet_cols(n, seed=0):
    rng = np.random.default_rng(seed)
    acts = {c: [rng.standard_normal(D).astype(np.float32) for _ in range(n)] for c in SLOT_COLUMNS}
    return {c: _fsl(acts[c]) for c in SLOT_COLUMNS}, acts


def _published_table(n, mode, *, with_response=True, with_prompt=True, with_expl=False):
    cols, acts = _triplet_cols(n)
    cols["doc_id"] = pa.array([f"doc:{i}" for i in range(n)], pa.string())
    cols["center_layer"] = pa.array([24] * n, pa.int64())  # written by the regen step
    cols["n_raw_tokens"] = pa.array([50 + i for i in range(n)], pa.int64())
    if mode == "av" and with_response:
        cols["response"] = pa.array([f"<explanation>\nfeat {i}\n</explanation>" for i in range(n)])
    if mode == "av" and with_prompt:
        # published single-marker actor prompt (chat list) — must be discarded
        cols["prompt"] = pa.array([[{"role": "user", "content": "old single-marker ㊗ prompt"}]] * n)
    if mode == "ar" and with_prompt:
        cols["prompt"] = pa.array([fill_ar_prompt(f"feat {i}") for i in range(n)], pa.string())
    if mode == "rl" and with_prompt:
        cols["prompt"] = pa.array([[{"role": "user", "content": "old single-marker ㊗ prompt"}]] * n)
    if with_expl:
        cols["api_explanation"] = pa.array([f"feat {i}" for i in range(n)], pa.string())
    return pa.table(cols), acts


def test_av_prompt_swapped_response_kept():
    table, acts = _published_table(6, "av")
    out = assemble_published(table, "av")
    prompts = out.column("prompt").to_pylist()
    # our three-marker prompt now, not the published single-marker one
    assert prompts[0][0]["content"].count(INJECT_PLACEHOLDER) == N_SLOTS
    resp = out.column("response").to_pylist()
    assert all(r.startswith("<explanation>") for r in resp)
    for c in SLOT_COLUMNS:
        assert c in out.schema.names


def test_av_round_trips_through_loader():
    with tempfile.TemporaryDirectory() as tmp:
        table, acts = _published_table(5, "av")
        out_path = str(Path(tmp) / "av_sft.parquet")
        # write a regenerated input, run build_one, then load with the real loader
        in_path = str(Path(tmp) / "regen_av.parquet")
        pq.write_table(table, in_path)
        n = build_one(in_path, "av", out_path)
        assert n == 5
        rows = load_av_sft_dataset(out_path)
        assert len(rows) == 5
        r = rows[0]
        assert r["prompt"][0]["content"].count(INJECT_PLACEHOLDER) == N_SLOTS
        assert r["response"].startswith("<explanation>")
        for j, c in enumerate(SLOT_COLUMNS):
            assert np.allclose(r[c], acts[c][0])


def test_ar_prompt_kept_verbatim_and_canonical():
    table, _ = _published_table(4, "ar")
    out = assemble_published(table, "ar")
    prompts = out.column("prompt").to_pylist()
    assert all(p.startswith("Summary of the following text:") for p in prompts)
    assert all(p.rstrip().endswith("<summary>") for p in prompts)
    # byte-identical to our shared template (== RL-time critic format)
    assert prompts[0] == AR_CRITIC_TEMPLATE.format(explanation="feat 0")


def test_ar_round_trips_through_loader():
    with tempfile.TemporaryDirectory() as tmp:
        table, acts = _published_table(5, "ar")
        in_path = str(Path(tmp) / "regen_ar.parquet")
        out_path = str(Path(tmp) / "ar_sft.parquet")
        pq.write_table(table, in_path)
        build_one(in_path, "ar", out_path)
        rows = load_ar_sft_dataset(out_path)
        assert len(rows) == 5
        assert isinstance(rows[0]["prompt"], str)
        assert rows[0]["prompt"].rstrip().endswith("<summary>")
        assert all(c in rows[0] for c in SLOT_COLUMNS)


def test_rl_prompt_swapped_no_response():
    table, _ = _published_table(4, "rl")
    out = assemble_published(table, "rl")
    assert "response" not in out.schema.names
    prompts = out.column("prompt").to_pylist()
    assert prompts[0][0]["content"].count(INJECT_PLACEHOLDER) == N_SLOTS
    assert set(SLOT_COLUMNS).issubset(set(out.schema.names))


def test_av_explanation_fallback_wraps():
    # no response column, but api_explanation present -> wrapped into <explanation>
    table, _ = _published_table(3, "av", with_response=False, with_expl=True)
    out = assemble_published(table, "av")
    resp = out.column("response").to_pylist()
    assert all(r.startswith("<explanation>") and "feat" in r for r in resp)


def test_ar_explanation_fallback_fills_canonical():
    # no prompt column, but api_explanation present -> filled with canonical template
    table, _ = _published_table(3, "ar", with_prompt=False, with_expl=True)
    out = assemble_published(table, "ar")
    prompts = out.column("prompt").to_pylist()
    assert all(p.startswith("Summary of the following text:") for p in prompts)
    assert all(p.rstrip().endswith("<summary>") for p in prompts)


def test_ar_rejects_prompt_without_summary_anchor():
    cols, _ = _triplet_cols(2)
    cols["prompt"] = pa.array(["no anchor here", "still none"], pa.string())
    table = pa.table(cols)
    try:
        assemble_published(table, "ar")
    except AssertionError as e:
        assert "<summary>" in str(e)
    else:
        raise AssertionError("expected rejection of ar prompts lacking the <summary> anchor")


def _archive_cols(n, layers, seed=1):
    """activation_L{k} archive columns (what the generalized regen writes)."""
    rng = np.random.default_rng(seed)
    vecs = {k: [rng.standard_normal(D).astype(np.float32) for _ in range(n)] for k in layers}
    cols = {f"activation_L{k}": _fsl(vecs[k]) for k in layers}
    return cols, vecs


def test_resolve_triplet_from_archive_selects_center():
    layers = list(range(19, 30))  # 11-layer window
    cols, vecs = _archive_cols(4, layers)
    cols["response"] = pa.array(["<explanation>\nfeat\n</explanation>"] * 4)
    cols["doc_id"] = pa.array([f"doc:{i}" for i in range(4)], pa.string())
    table = pa.table(cols)

    out24 = assemble_published(table, "av", center=24)  # -> L23/L24/L25
    for i in range(4):
        assert np.allclose(out24.column("activation_prev").to_pylist()[i], vecs[23][i])
        assert np.allclose(out24.column("activation_centre").to_pylist()[i], vecs[24][i])
        assert np.allclose(out24.column("activation_next").to_pylist()[i], vecs[25][i])
    assert out24.column("center_layer").to_pylist() == [24] * 4

    out22 = assemble_published(table, "av", center=22)  # re-slice, no re-extraction
    assert np.allclose(out22.column("activation_centre").to_pylist()[0], vecs[22][0])
    assert np.allclose(out22.column("activation_prev").to_pylist()[0], vecs[21][0])
    assert out22.column("center_layer").to_pylist() == [22] * 4


def test_center_outside_window_raises():
    cols, _ = _archive_cols(2, list(range(19, 30)))
    cols["response"] = pa.array(["<explanation>\nf\n</explanation>"] * 2)
    table = pa.table(cols)
    try:
        assemble_published(table, "av", center=10)  # L9/L10/L11 not saved, no legacy triplet
    except SystemExit as e:
        assert "neither" in str(e)
    else:
        raise AssertionError("expected SystemExit when center is outside the saved window")


def test_provenance_columns_carried_through():
    # doc_id / center_layer / n_raw_tokens survive so the inherited split + center
    # stay auditable downstream (and a wrong-corpus mix would be visible).
    for mode in ("av", "ar", "rl"):
        table, _ = _published_table(4, mode)
        out = assemble_published(table, mode)
        for prov in ("doc_id", "center_layer", "n_raw_tokens"):
            assert prov in out.schema.names, f"{mode}: dropped provenance column {prov}"
        assert out.column("center_layer").to_pylist() == [24] * 4


def test_build_one_streams_archive_in_batches():
    # the 11-layer archive must build via streaming (column-projected, batched) —
    # never read whole — and select the right triplet across batch boundaries.
    with tempfile.TemporaryDirectory() as tmp:
        layers = list(range(19, 30))
        cols, vecs = _archive_cols(10, layers)
        cols["response"] = pa.array([f"<explanation>\nfeat {i}\n</explanation>" for i in range(10)])
        cols["doc_id"] = pa.array([f"doc:{i}" for i in range(10)], pa.string())
        in_path = str(Path(tmp) / "regen_av.parquet")
        pq.write_table(pa.table(cols), in_path)
        out_path = str(Path(tmp) / "av_sft.parquet")
        n = build_one(in_path, "av", out_path, center=24, batch_size=4)  # -> 3 batches
        assert n == 10
        rows = load_av_sft_dataset(out_path)
        assert len(rows) == 10
        for i in (0, 4, 5, 9):  # across batch boundaries
            assert np.allclose(rows[i]["activation_prev"], vecs[23][i])
            assert np.allclose(rows[i]["activation_centre"], vecs[24][i])
            assert np.allclose(rows[i]["activation_next"], vecs[25][i])
        # the 8 unused archive layers are NOT carried into the training parquet
        assert "activation_L19" not in pq.read_table(out_path).schema.names


def test_ar_rejects_noncanonical_prefix():
    # right <summary> suffix but WRONG prefix -> must be rejected (the old endswith
    # check would have wrongly accepted it; AR-SFT must match RL's fill_ar_prompt).
    cols, _ = _triplet_cols(2)
    cols["prompt"] = pa.array(["Different lead-in: <text>x</text> <summary>",
                               "Another: <text>y</text> <summary>"], pa.string())
    table = pa.table(cols)
    try:
        assemble_published(table, "ar")
    except AssertionError as e:
        assert "canonical" in str(e)
    else:
        raise AssertionError("expected rejection of a non-canonical AR prompt prefix")


def test_build_one_multi_shard_concatenates():
    # the shards ARE the layer bank — build streams across all of them into one
    # training parquet (no wide merge), selecting the center triplet per shard.
    with tempfile.TemporaryDirectory() as tmp:
        layers = list(range(19, 30))
        paths, vecs_by_shard = [], {}
        for sh in range(2):
            cols, vecs = _archive_cols(6, layers, seed=sh)
            cols["response"] = pa.array([f"<explanation>\nf{sh}_{i}\n</explanation>" for i in range(6)])
            cols["doc_id"] = pa.array([f"doc:{sh}:{i}" for i in range(6)], pa.string())
            p = str(Path(tmp) / f"av_sft.shard0{sh}of02.parquet")
            pq.write_table(pa.table(cols), p)
            paths.append(p); vecs_by_shard[sh] = vecs
        out = str(Path(tmp) / "av_sft.parquet")
        n = build_one(paths, "av", out, center=24, batch_size=4)
        assert n == 12
        rows = load_av_sft_dataset(out)
        assert len(rows) == 12
        assert np.allclose(rows[0]["activation_centre"], vecs_by_shard[0][24][0])
        assert np.allclose(rows[6]["activation_centre"], vecs_by_shard[1][24][0])


def test_subset_inputs_prefers_shards():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for sh in range(3):
            (d / f"av_sft.shard0{sh}of03.parquet").write_bytes(b"x")
        got = _subset_inputs(d, "av_sft")
        assert len(got) == 3 and all("shard" in g for g in got)
        assert _subset_inputs(d, "ar_sft") == []  # absent subset -> [] (caller decides)


def test_discover_subsets_handles_train_val_naming():
    # EasyNLA warmstart files are av_sft_train/av_sft_val etc.; nanoNLA-published
    # are flat av_sft. discover_subsets must find both and refuse an empty subset.
    from multilayer_nla.build_from_published import discover_subsets
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for f in ("av_sft_train", "av_sft_val", "ar_sft_train", "rl_train"):
            (d / f"{f}.parquet").write_bytes(b"x")
        got = discover_subsets(d)
        assert [(m, f) for m, f, _ in got] == [
            ("av", "av_sft_train"), ("av", "av_sft_val"),
            ("ar", "ar_sft_train"), ("rl", "rl_train")]
        (d / "av_sft_train.parquet").unlink(); (d / "av_sft_val.parquet").unlink()
        try:
            discover_subsets(d)
        except SystemExit as e:
            assert "av_sft" in str(e)
        else:
            raise AssertionError("expected refusal when a logical subset has no variant")


def test_docs_disjoint_pass_and_overlap_raises():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        def w(name, ids):
            pq.write_table(pa.table({"doc_id": pa.array(ids, pa.string())}), str(d / f"{name}.parquet"))
        w("av_sft", ["a", "a", "b"]); w("ar_sft", ["c", "d"]); w("rl", ["e"])
        names = ("av_sft", "ar_sft", "rl")
        assert _assert_docs_disjoint(d, names) == {"av_sft": 2, "ar_sft": 2, "rl": 1}
        w("rl", ["e", "b"])  # 'b' leaks from av_sft -> must raise
        try:
            _assert_docs_disjoint(d, names)
        except AssertionError as e:
            assert "shared" in str(e)
        else:
            raise AssertionError("expected disjointness failure on a leaked doc_id")


def test_missing_triplet_refused():
    table = pa.table({"response": pa.array(["<explanation>\nx\n</explanation>"])})
    try:
        assemble_published(table, "av")
    except SystemExit as e:
        assert "triplet" in str(e)
    else:
        raise AssertionError("expected refusal when no activation columns are present")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} build-from-published tests passed.")


if __name__ == "__main__":
    _run_all()
