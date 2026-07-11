"""Validate the permutation-grid condition builder (offline, synthetic bank).

Encodes a recognizable value per (doc, pos, layer, position-offset) into a tiny
synthetic bank WITH window columns (W=4), runs splits + build_all over a grid
mixing 1-token×3-layer, 3-token×1-layer, dup, pooled, and shufctx conditions,
then proves:
  - av_in_* carry the RIGHT (layer, offset) source per slot (incl. window slices);
  - the AR target is byte-identical for EVERY condition (fixed-target discipline);
  - the stored prompt is the neutral template with exactly k markers, identical
    across same-k conditions;
  - shufctx keeps the final slot true and deranges the context slots;
  - pooling averages the slot sources;
  - the preflight passes, and CATCHES a poisoned build (wrong window offset).

Run: python -m pytest multilayer_nla/tests/test_build_conditions.py -q
"""

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nla.schema import wrap_explanation
from multilayer_nla.datasets import av_in_columns, fill_ar_prompt, av_user_content_neutral
from multilayer_nla import build_conditions, splits
from multilayer_nla.conditions import (
    PERMUTATION_GRID,
    parse_condition,
    parse_conditions,
    validate_against_bank,
)

ACT_LAYERS = list(range(19, 30))
WIN_LAYERS = [23, 24, 25]
W = 4
D = 6


def _val(doc, pos, L, off=0):
    """Unique per (doc, pos, layer, offset) so a slot's source is readable back."""
    return float(doc) * 1000.0 + float(pos) + float(L) * 1e-4 + float(off) * 17.0


def _make_bank(bank_dir, subset, n_docs=120, ppd=3, with_response=False, with_prompt=False):
    n = n_docs * ppd
    dids, resp, prompt = [], [], []
    act = {L: np.zeros((n, D), np.float32) for L in ACT_LAYERS}
    win = {L: np.zeros((n, W, D), np.float32) for L in WIN_LAYERS}
    r = 0
    for doc in range(n_docs):
        did = f"{subset}:{doc}"
        for pos in range(ppd):
            for L in ACT_LAYERS:
                act[L][r] = _val(doc, pos, L, 0)
            for L in WIN_LAYERS:
                for j in range(W):                      # slot W-1 == offset 0 == p
                    win[L][r, W - 1 - j] = _val(doc, pos, L, -j)
            dids.append(did)
            if with_response:
                resp.append(wrap_explanation(f"expl {doc}.{pos}"))
            if with_prompt:
                prompt.append(fill_ar_prompt(f"expl {doc}.{pos}"))
            r += 1
    tbl = {"doc_id": pa.array(dids)}
    for L in ACT_LAYERS:
        tbl[f"activation_L{L}"] = pa.FixedSizeListArray.from_arrays(
            pa.array(act[L].reshape(-1)), D)
    for L in WIN_LAYERS:
        tbl[f"window_L{L}"] = pa.FixedSizeListArray.from_arrays(
            pa.array(win[L].reshape(-1).astype(np.float32)), W * D)
    if with_response:
        tbl["response"] = pa.array(resp)
    if with_prompt:
        tbl["prompt"] = pa.array(prompt)
    Path(bank_dir).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(tbl), str(Path(bank_dir) / f"{subset}.parquet"), row_group_size=53)


def _col(path, name):
    t = pq.read_table(path, columns=[name]).column(name).combine_chunks()
    return t.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(t), -1)


def _expected(path, layer, off, ppd=3):
    srcs = pq.read_table(path, columns=["src_row_id"]).column("src_row_id").to_pylist()
    out = np.zeros((len(srcs), D), np.float32)
    for i, s in enumerate(srcs):
        out[i] = _val(s // ppd, s % ppd, layer, off)
    return out


# ------------------------------------------------------------------ grammar

def test_parse_condition_grammar():
    c = parse_condition("tok3=L24@-2,L24@-1,L24@0")
    assert c.name == "tok3" and c.slots == ((24, -2), (24, -1), (24, 0)) and c.k == 3
    c = parse_condition("mean3=L23@0,L24@0,L25@0|pool")
    assert c.pool and c.k == 1 and c.layers == (23, 24, 25)
    c = parse_condition("sh=L24@-1,L24@0|shufctx")
    assert c.shufctx
    with pytest.raises(ValueError):
        parse_condition("bad=L24@+1")
    with pytest.raises(AssertionError):
        parse_condition("noslots=")
    with pytest.raises(AssertionError):
        parse_condition("dupname=L24@0|nonsense")
    grid = parse_conditions(PERMUTATION_GRID)
    assert {"single", "dup3", "lay3", "tok3", "mix4", "dup4", "tok3_shufctx"} <= set(grid)


def test_layer_grid_needs_only_final_token_layers():
    # every LAYER_GRID slot is at offset 0 -> validates against a WINDOWLESS bank
    # (activation_L19..29 only), and only touches layers in the stored band.
    from multilayer_nla.conditions import LAYER_GRID
    grid = parse_conditions(LAYER_GRID)
    for c, v in grid.items():
        assert all(off == 0 for _, off in v.slots), f"{c}: layer grid must be all @0"
        validate_against_bank(v, activation_layers=set(range(19, 30)),
                              window_layers=set(), window_size=0)   # no windows needed
        assert all(19 <= L <= 29 for L in v.layers), f"{c}: layer outside stored band"


def test_validate_against_bank_errors():
    kw = dict(activation_layers={23, 24, 25}, window_layers={24}, window_size=4)
    validate_against_bank(parse_condition("ok=L24@-3,L24@0"), **kw)
    with pytest.raises(SystemExit):   # missing activation layer
        validate_against_bank(parse_condition("x=L30@0"), **kw)
    with pytest.raises(SystemExit):   # missing window layer
        validate_against_bank(parse_condition("x=L23@-1"), **kw)
    with pytest.raises(SystemExit):   # offset beyond W
        validate_against_bank(parse_condition("x=L24@-4"), **kw)


def test_neutral_template_marker_count():
    for k in (1, 2, 3, 4, 9):
        body = av_user_content_neutral(k, placeholder="<M>")
        assert body.count("<M>") == k


# ------------------------------------------------------------------ end to end

GRID = ("single=L24@0; dup3=L24@0,L24@0,L24@0; lay3=L23@0,L24@0,L25@0; "
        "tok3=L24@-2,L24@-1,L24@0; mix4=L23@-1,L23@0,L25@-1,L25@0; "
        "mean3=L23@0,L24@0,L25@0|pool; tok3_shufctx=L24@-2,L24@-1,L24@0|shufctx")


@pytest.fixture(scope="module")
def built():
    with tempfile.TemporaryDirectory() as tmp:
        bank = Path(tmp) / "bank"
        _make_bank(bank, "av_sft", with_response=True)
        _make_bank(bank, "ar_sft", with_prompt=True)
        _make_bank(bank, "rl")
        out = Path(tmp) / "cond"
        splits.build_split_manifest([str(bank / "rl.parquet")], "rl", str(out), seed=42)
        splits.build_split_manifest([str(bank / "ar_sft.parquet")], "ar", str(out), seed=42)
        conds = parse_conditions(GRID)
        build_conditions.build_all(str(bank), str(out), conds,
                                   str(out / "rl_split_manifest.json"),
                                   str(out / "ar_split_manifest.json"))
        yield out, conds


def test_slots_carry_right_layer_and_offset(built):
    out, conds = built
    for cname in ("single", "dup3", "lay3", "tok3", "mix4"):
        v = conds[cname]
        for f in (out / f"av_{cname}.parquet", out / f"rl_dev_{cname}.parquet"):
            for i, (L, off) in enumerate(v.slots):
                got = _col(f, f"av_in_{i}")
                want = _expected(f, L, off)
                assert np.array_equal(got, want), f"{f.name} slot {i}: wrong (layer,offset) pour"


def test_fixed_target_identical_across_all_conditions(built):
    out, conds = built
    for bucket in ("dev", "test"):
        ref = None
        for c in conds:
            p = out / f"rl_{bucket}_{c}.parquet"
            tgt = np.concatenate([_col(p, tc) for tc in
                                  ("activation_prev", "activation_centre", "activation_next")], axis=1)
            if ref is None:
                ref = tgt
            else:
                assert np.array_equal(tgt, ref), f"{p.name}: target moved with the condition"


def test_prompt_stored_and_identical_within_k(built):
    out, conds = built
    prompts = {}
    for c, v in conds.items():
        if v.shufctx:
            continue
        p0 = pq.read_table(out / f"av_{c}.parquet", columns=["prompt"]).column("prompt")[0].as_py()
        body = p0[0]["content"]
        assert body == av_user_content_neutral(v.k)      # neutral template, k markers
        prompts.setdefault(v.k, body)
        assert prompts[v.k] == body                      # identical within k
    # rl files store the prompt too (what evaluate_e2e / RL generate with)
    p0 = pq.read_table(out / "rl_dev_tok3.parquet", columns=["prompt"]).column("prompt")[0].as_py()
    assert p0[0]["content"] == av_user_content_neutral(3)


def test_pool_is_mean_of_slot_sources(built):
    out, _ = built
    f = out / "av_mean3.parquet"
    got = _col(f, "av_in_0")
    want = np.mean([_expected(f, L, 0) for L in (23, 24, 25)], axis=0).astype(np.float32)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6)
    assert "av_in_1" not in pq.ParquetFile(f).schema_arrow.names


def test_shufctx_keeps_final_true_and_deranges_context(built):
    out, _ = built
    sh, par = out / "rl_dev_tok3_shufctx.parquet", out / "rl_dev_tok3.parquet"
    assert not (out / "av_tok3_shufctx.parquet").exists()   # eval-only: no AV train file
    assert np.array_equal(_col(sh, "av_in_2"), _col(par, "av_in_2"))   # final slot true
    a0, b0 = _col(sh, "av_in_0"), _col(par, "av_in_0")
    frac_same = float(np.mean(np.all(a0 == b0, axis=1)))
    assert frac_same < 0.01
    # deranged slots are still REAL slot values from other rows of the same file
    pool = {tuple(row) for row in b0}
    assert all(tuple(row) in pool for row in a0)


def test_shufctx_refused_for_av_training(built):
    out, conds = built
    with pytest.raises(AssertionError, match="EVAL-ONLY"):
        build_conditions.build_av([str(out.parent / "bank" / "av_sft.parquet")],
                                  str(out / "should_not_exist.parquet"),
                                  conds["tok3_shufctx"])


def test_preflight_catches_wrong_offset_pour(built):
    out, conds = built
    # poison: overwrite tok3's av_in_1 (should be L24@-1) with the offset-0 column
    p = out / "rl_dev_tok3.parquet"
    t = pq.read_table(p)
    poisoned = t.set_column(t.schema.names.index("av_in_1"), "av_in_1", t.column("av_in_2"))
    bad = out / "rl_dev_tok3_poison.parquet"
    pq.write_table(poisoned, bad)
    try:
        import shutil
        backup = out / "rl_dev_tok3_backup.parquet"
        shutil.move(str(p), str(backup))
        shutil.move(str(bad), str(p))
        with pytest.raises(AssertionError):
            build_conditions.assert_conditions(out, {"tok3": conds["tok3"]})
    finally:
        shutil.move(str(p), str(bad))
        shutil.move(str(backup), str(p))


def test_dev_test_docs_disjoint(built):
    out, _ = built
    dev = set(pq.read_table(out / "rl_dev_single.parquet", columns=["doc_id"]).column("doc_id").to_pylist())
    tst = set(pq.read_table(out / "rl_test_single.parquet", columns=["doc_id"]).column("doc_id").to_pylist())
    assert dev and tst and not (dev & tst)


def _run_all():
    import sys
    sys.exit(pytest.main([__file__, "-q"]))


if __name__ == "__main__":
    _run_all()
