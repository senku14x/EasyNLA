"""Unified bank regeneration core: published prefix -> per-layer final-token
activations + per-layer position windows, one pass.

Pure (numpy/pyarrow/torch-cpu) — no model. Validates the assemble step (both
activation_L{k} and window_L{k} columns), the round-trip + short-window keep
logic, the stored-vector cross-parity guard, dtype handling (fp16 exactness +
range guard), the length-bucket scatter, and the on-GPU gather math the
extractor uses.

Run: python -m pytest multilayer_nla/tests/test_regenerate_bank.py -q
"""

import numpy as np
import pyarrow as pa

from multilayer_nla.regenerate_bank import (
    PARITY_MEDIAN_MIN,
    assemble_bank_table,
    bucketed_extract,
    layer_col,
    parse_layers,
    select_keep_indices,
    storage_estimate,
    stored_parity,
    window_col,
    _shard_out,
)

D = 8
W = 4
TRIPLET = [23, 24, 25]
WIDE = list(range(19, 30))  # 11-layer archive


def _make_table(n, n_raw, *, stored_from=None, stored_layer=24):
    cols = {
        "detokenized_text_truncated": pa.array([f"web prefix {i}" for i in range(n)]),
        "n_raw_tokens": pa.array(n_raw, pa.int64()),
        "doc_id": pa.array([f"doc:{i}" for i in range(n)], pa.string()),
        "response": pa.array([f"<explanation>\nfeat {i}\n</explanation>" for i in range(n)]),
    }
    if stored_from is not None:
        # the "published" single-layer vector: [n, D] float32
        flat = np.ascontiguousarray(stored_from).reshape(-1).astype(np.float32)
        cols["activation_vector"] = pa.FixedSizeListArray.from_arrays(pa.array(flat), D)
        cols["activation_layer"] = pa.array([stored_layer] * n, pa.int64())
    return pa.table(cols)


def _make_results(n_raw, final_layers, window_layers=(), *, window=W, seed=0, valid=None):
    rng = np.random.default_rng(seed)
    results = []
    for i, seq_len in enumerate(n_raw):
        results.append({
            "token_ids": list(range(seq_len)),
            "final": {li: rng.standard_normal(D).astype(np.float32) for li in final_layers},
            "window": {li: rng.standard_normal((window, D)).astype(np.float16)
                       for li in window_layers},
            "valid": True if valid is None else valid[i],
        })
    return results


# ---------------------------------------------------------------- assemble

def test_appends_final_and_window_columns():
    n_raw = [63, 65, 64]
    table = _make_table(len(n_raw), n_raw)
    results = _make_results(n_raw, TRIPLET, TRIPLET)
    out = assemble_bank_table(table, results, TRIPLET, TRIPLET, W, D,
                              np.float32, np.float16, center=24)
    for li in TRIPLET:
        assert layer_col(li) in out.schema.names
        assert window_col(li) in out.schema.names
    assert "window_size" in out.schema.names
    assert set(out.column("window_size").to_pylist()) == {W}
    # final vectors land as given; window flattens slot-major with last slot == p
    for li in TRIPLET:
        acts = out.column(layer_col(li)).to_pylist()
        wins = out.column(window_col(li)).to_pylist()
        for i, r in enumerate(results):
            assert np.allclose(acts[i], r["final"][li])
            got = np.asarray(wins[i], dtype=np.float32).reshape(W, D)
            assert np.array_equal(got.astype(np.float16), r["window"][li])
            # slot W-1 is the labeled position p == the final-token vector,
            # exactly, after casting the fp32 final to fp16 (bf16-lossless).
            # fabricated data here is fp16-native so equality is on the cast.


def test_window_disabled_leaves_no_window_columns():
    n_raw = [60, 61]
    table = _make_table(2, n_raw)
    results = _make_results(n_raw, TRIPLET, ())
    out = assemble_bank_table(table, results, TRIPLET, [], 0, D,
                              np.float32, np.float16, center=24)
    assert not any(c.startswith("window_") for c in out.schema.names)
    for li in TRIPLET:
        assert layer_col(li) in out.schema.names


def test_wide_archive_11_layers():
    n_raw = [60, 60]
    table = _make_table(2, n_raw)
    out = assemble_bank_table(table, _make_results(n_raw, WIDE), WIDE, [], 0, D,
                              np.float32, np.float16, center=24)
    assert len([c for c in out.schema.names if c.startswith("activation_L")]) == 11


def test_center_layer_appended_and_preserved():
    n_raw = [60]
    out = assemble_bank_table(_make_table(1, n_raw), _make_results(n_raw, TRIPLET),
                              TRIPLET, [], 0, D, np.float32, np.float16, center=24)
    assert out.column("center_layer").to_pylist() == [24]


def test_preserves_published_label_columns():
    n_raw = [60, 60]
    table = _make_table(2, n_raw)
    out = assemble_bank_table(table, _make_results(n_raw, TRIPLET, TRIPLET),
                              TRIPLET, TRIPLET, W, D, np.float32, np.float16, center=24)
    assert out.column("response").to_pylist() == table.column("response").to_pylist()
    assert out.column("doc_id").to_pylist() == table.column("doc_id").to_pylist()


def test_double_regeneration_refused():
    n_raw = [60]
    table = _make_table(1, n_raw)
    once = assemble_bank_table(table, _make_results(n_raw, TRIPLET), TRIPLET, [], 0, D,
                               np.float32, np.float16, center=24)
    try:
        assemble_bank_table(once, _make_results(n_raw, TRIPLET), TRIPLET, [], 0, D,
                            np.float32, np.float16, center=24)
    except AssertionError as e:
        assert "already has" in str(e)
    else:
        raise AssertionError("expected refusal to re-add an existing activation_L{k} column")


def test_drop_stored_activation():
    n_raw = [60, 60]
    stored = np.random.default_rng(1).standard_normal((2, D)).astype(np.float32)
    table = _make_table(2, n_raw, stored_from=stored)
    results = _make_results(n_raw, TRIPLET)
    kept = assemble_bank_table(table, results, TRIPLET, [], 0, D,
                               np.float32, np.float16, center=24)
    assert "activation_vector" in kept.schema.names
    dropped = assemble_bank_table(table, results, TRIPLET, [], 0, D,
                                  np.float32, np.float16, center=24,
                                  drop_stored_activation=True)
    assert "activation_vector" not in dropped.schema.names


def test_fp16_activation_range_guard():
    n_raw = [60]
    table = _make_table(1, n_raw)
    results = _make_results(n_raw, TRIPLET)
    results[0]["final"][24][:] = 70000.0  # exceeds float16 max 65504
    try:
        assemble_bank_table(table, results, TRIPLET, [], 0, D,
                            np.float16, np.float16, center=24)
    except AssertionError as e:
        assert "float16" in str(e)
    else:
        raise AssertionError("expected fp16 range guard to fire")


def test_fp16_cast_path_equivalence():
    # verify_bank's exactness assumption: for bf16-representable values, casting
    # the fp32 view to fp16 equals casting the bf16 value to fp16 directly.
    import torch
    x = torch.randn(1000, dtype=torch.bfloat16) * 100
    via_fp32 = x.float().numpy().astype(np.float16)
    direct = x.to(torch.float16).numpy()
    assert np.array_equal(via_fp32, direct)


# ---------------------------------------------------------------- keep logic

def test_roundtrip_mismatch_raises():
    n_raw = [60, 61]
    results = _make_results(n_raw, TRIPLET)
    results[1]["token_ids"] = list(range(99))  # re-encoded 99 != stored 61
    try:
        select_keep_indices(results, n_raw)
    except AssertionError as e:
        assert "round-trip" in str(e)
    else:
        raise AssertionError("expected AssertionError on n_raw_tokens mismatch")


def test_roundtrip_drop_within_threshold():
    n_raw = [60, 61, 62, 63]
    results = _make_results(n_raw, TRIPLET)
    results[2]["token_ids"] = list(range(99))
    keep, info = select_keep_indices(results, n_raw, max_drop_frac=0.5)
    assert keep == [0, 1, 3]
    assert info["n_roundtrip"] == 1 and info["n_short"] == 0


def test_roundtrip_drop_over_threshold_still_raises():
    n_raw = [60, 61, 62, 63]
    results = _make_results(n_raw, TRIPLET)
    results[1]["token_ids"] = list(range(99))
    results[2]["token_ids"] = list(range(99))
    try:
        select_keep_indices(results, n_raw, max_drop_frac=0.1)
    except AssertionError as e:
        assert "exceeds" in str(e)
    else:
        raise AssertionError("expected raise when drop fraction exceeds max_drop_frac")


def test_short_window_rows_always_dropped():
    n_raw = [60, 2, 61]
    results = _make_results(n_raw, TRIPLET, TRIPLET, valid=[True, False, True])
    keep, info = select_keep_indices(results, n_raw)
    assert keep == [0, 2]
    assert info["n_short"] == 1


def test_no_roundtrip_when_nrt_none():
    n_raw = [60]
    results = _make_results(n_raw, TRIPLET)
    results[0]["token_ids"] = list(range(99))  # mismatch, but nrt=None -> no check
    keep, info = select_keep_indices(results, None)
    assert keep == [0] and info["n_roundtrip"] == 0


# ---------------------------------------------------------------- stored parity

def test_stored_parity_match():
    n_raw = [60, 61, 62]
    results = _make_results(n_raw, TRIPLET)
    stored = np.stack([r["final"][24] for r in results])  # exactly the regen vectors
    table = _make_table(3, n_raw, stored_from=stored, stored_layer=24)
    med, mn, n_below, layer = stored_parity(table, results, D)
    assert layer == 24 and med > 0.99999 and mn > 0.99999 and n_below == 0


def test_stored_parity_catches_wrong_vector():
    n_raw = [60, 61, 62, 63]
    results = _make_results(n_raw, TRIPLET)
    rng = np.random.default_rng(7)
    stored = rng.standard_normal((4, D)).astype(np.float32)  # unrelated vectors
    table = _make_table(4, n_raw, stored_from=stored, stored_layer=24)
    med, mn, n_below, layer = stored_parity(table, results, D)
    assert med < PARITY_MEDIAN_MIN  # the caller's assert would fire


def test_stored_parity_none_without_column_or_layer():
    n_raw = [60]
    results = _make_results(n_raw, TRIPLET)
    assert stored_parity(_make_table(1, n_raw), results, D) is None
    # stored layer outside the regenerated set -> None (nothing to compare)
    stored = np.stack([r["final"][24] for r in results])
    table = _make_table(1, n_raw, stored_from=stored, stored_layer=5)
    assert stored_parity(table, results, D) is None


# ---------------------------------------------------------------- misc helpers

def test_storage_estimate_math():
    est = storage_estimate(1000, 4096, list(range(19, 30)), [23, 24, 25], 8,
                           np.float32, np.float16)
    # 11 layers * 4096 * 4B + 3 layers * 8 * 4096 * 2B = 180224 + 196608 = 376832 B/row
    assert est["per_row_bytes"] == 11 * 4096 * 4 + 3 * 8 * 4096 * 2
    assert abs(est["total_gb"] - est["per_row_bytes"] * 1000 / 1e9) < 1e-9


def test_gather_last_real_token_equals_slice_last():
    # guards the on-GPU gather's indexing: under right padding,
    # captured[batch, len-1] == captured[i, :len_i][-1].
    import torch
    B, T, d = 4, 6, 5
    captured = torch.randn(B, T, d)
    attn = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    lengths = attn.sum(1)
    last = (lengths - 1).clamp_min(0)
    bidx = torch.arange(B)
    gathered = captured[bidx, last]
    for i in range(B):
        assert torch.allclose(gathered[i], captured[i, : lengths[i]][-1])


def test_window_gather_math():
    # the [B, W] window gather: rows with len >= W get [len-W, len); the last
    # window slot equals the final real token; short rows are invalid.
    import torch
    B, T, d, w = 3, 10, 4, 4
    captured = torch.randn(B, T, d)
    lengths = torch.tensor([10, 5, 3])          # row 2 is too short for w=4
    offsets = torch.arange(w)
    starts = lengths - w
    valid = starts >= 0
    win_idx = (starts.clamp_min(0)[:, None] + offsets[None, :]).clamp_(max=T - 1)
    bidx = torch.arange(B)[:, None].expand(B, w)
    win = captured[bidx, win_idx]               # [B, w, d]
    assert valid.tolist() == [True, True, False]
    for i in range(2):
        L = lengths[i].item()
        assert torch.allclose(win[i], captured[i, L - w:L])
        assert torch.allclose(win[i, -1], captured[i, L - 1])  # slot W-1 == p


def test_length_bucket_preserves_order_and_content():
    texts = ["aaaaa", "b", "ccccccccc", "ddd"]
    nrt = [5, 1, 9, 3]
    seen = []

    def toy(ts):
        seen.append(list(ts))
        return [{"tok": t, "final": {24: [float(len(t))]}} for t in ts]

    unb = bucketed_extract(texts, nrt, toy, length_bucket=False)
    buc = bucketed_extract(texts, nrt, toy, length_bucket=True)
    assert seen[1] == ["b", "ddd", "aaaaa", "ccccccccc"]
    assert [r["tok"] for r in unb] == texts
    assert [r["tok"] for r in buc] == texts
    assert buc == unb


def test_length_bucket_requires_n_raw_tokens():
    try:
        bucketed_extract(["a", "b"], None, lambda t: [{} for _ in t], length_bucket=True)
    except AssertionError as e:
        assert "n_raw_tokens" in str(e)
    else:
        raise AssertionError("expected failure when n_raw_tokens is missing")


def test_parse_layers():
    assert parse_layers("19-29") == list(range(19, 30))
    assert parse_layers("19,24,29") == [19, 24, 29]
    assert parse_layers("19-21,25,27-29") == [19, 20, 21, 25, 27, 28, 29]
    assert parse_layers("24") == [24]


def test_shard_out_paths():
    assert _shard_out("/x/av.parquet", 0, 1) == "/x/av.parquet"
    assert _shard_out("/x/av.parquet", 2, 8) == "/x/av.shard02of08.parquet"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} regenerate-bank tests passed.")


if __name__ == "__main__":
    _run_all()
