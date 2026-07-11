"""Offline tests for the evaluator's FVE aggregation/bootstrap + dev checkpoint
selection — the parts that don't need a GPU or a model.

Run:  python -m multilayer_nla.tests.test_eval_select
"""

import numpy as np

from multilayer_nla.evaluate_e2e import aggregate, bootstrap_overall
from multilayer_nla.select_and_report import select


def test_aggregate_success_and_penalized():
    baselines = [1.0, 1.0, 1.0]
    errs = [[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], None]  # 2 success, 1 failed
    agg = aggregate(errs, baselines)
    assert agg["n_total"] == 3 and agg["n_success"] == 2
    # success-only mse = 0.3 -> FVE 0.7
    assert np.allclose(agg["fve"], [0.7, 0.7, 0.7]) and np.isclose(agg["fve_overall"], 0.7)
    # penalized: failed row counts as baseline (FVE 0); mse = (0.2+0.4+1.0)/3
    pen = 1.0 - (0.2 + 0.4 + 1.0) / 3.0
    assert np.allclose(agg["pen_fve"], [pen, pen, pen]) and np.isclose(agg["pen_fve_overall"], pen)


def test_all_failed_is_penalized_zero_and_success_nan():
    baselines = [2.0, 2.0, 2.0]
    agg = aggregate([None, None], baselines)
    assert np.allclose(agg["pen_fve"], [0.0, 0.0, 0.0]) and np.isclose(agg["pen_fve_overall"], 0.0)
    assert all(np.isnan(x) for x in agg["fve"])  # no successful rows


def test_bootstrap_resamples_documents():
    baselines = [1.0, 1.0, 1.0]
    errs = [[0.1, 0.1, 0.1]] * 10
    docs = [f"d{i // 2}" for i in range(10)]  # 5 docs, 2 rows each
    lo, hi = bootstrap_overall(errs, docs, baselines, n_boot=200, seed=0, penalized=True)
    # identical errs -> FVE is exactly 0.9 under every resample
    assert np.isclose(lo, 0.9) and np.isclose(hi, 0.9)


def test_doc_derangement_avoids_same_doc():
    from multilayer_nla.evaluate_e2e import _doc_derangement
    doc_ids = [f"d{i // 3}" for i in range(30)]  # 10 docs, 3 rows each
    perm = _doc_derangement(doc_ids, seed=0)
    assert sorted(perm) == list(range(30)), "not a valid permutation"
    # the shuffled control must pair each doc's target with ANOTHER doc's generation
    assert all(doc_ids[perm[i]] != doc_ids[i] for i in range(30))


def test_select_uses_dev_grid_only():
    grid = {
        "local":     {500: {500: 0.30, 1000: 0.40}, 1000: {500: 0.50, 1000: 0.55}},
        "duplicate": {500: {500: 0.20, 1000: 0.25}, 1000: {500: 0.30, 1000: 0.28}},
        "wide":      {500: {500: 0.35, 1000: 0.45}, 1000: {500: 0.52, 1000: 0.60}},
        "single":    {500: {500: 0.10, 1000: 0.15}, 1000: {500: 0.20, 1000: 0.22}},
    }
    ar, av, diag = select(grid, ("local", "duplicate", "wide", "single"))
    assert ar == 1000, diag
    # at AR 1000, the better AV per condition:
    assert av == {"local": 1000, "duplicate": 500, "wide": 1000, "single": 1000}


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} eval/select tests passed.")


if __name__ == "__main__":
    _run_all()
