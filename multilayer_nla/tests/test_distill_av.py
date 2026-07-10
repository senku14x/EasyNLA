"""Offline tests for distill_av's pure decision logic (no GPU/model).

The generation/scoring paths run on the box; what must be provably right
offline is (a) best-of-N selection semantics — failed candidates never win,
gold fallback, tie-breaking — and (b) the BoN vector-expansion order, which is
the injection-correctness-critical piece: a wrong interleave silently pairs
sample j with the WRONG row's activations (the exact class of bug the research
notes call out for packing/reordering).

Run: python -m pytest multilayer_nla/tests/test_distill_av.py -q
"""

import numpy as np

from multilayer_nla.distill_av import expand_for_bon, reward_from_taps, select_best


def test_select_best_prefers_higher_reward():
    assert select_best(-1.0, [-0.5, -2.0]) == ("sample", 0, -0.5)
    assert select_best(-0.1, [-0.5, -2.0]) == ("gold", -1, -0.1)


def test_select_best_ignores_failed_samples():
    assert select_best(-1.0, [None, -0.4, None]) == ("sample", 1, -0.4)
    assert select_best(-0.2, [None, None]) == ("gold", -1, -0.2)


def test_select_best_gold_fallback_when_everything_fails():
    assert select_best(None, [None, None]) == ("gold", -1, None)


def test_select_best_tie_goes_to_gold_then_earliest_sample():
    assert select_best(-0.5, [-0.5, -0.5]) == ("gold", -1, -0.5)          # tie -> gold
    assert select_best(None, [-0.5, -0.5], include_gold=True) == ("sample", 0, -0.5)
    assert select_best(-9.9, [-0.5, -0.5], include_gold=False) == ("sample", 0, -0.5)


def test_select_best_no_gold_mode():
    # gold excluded from the argmax even when it scores best
    assert select_best(-0.01, [-0.5, -0.4], include_gold=False) == ("sample", 1, -0.4)


def test_reward_from_taps():
    assert reward_from_taps([0.2, 0.4]) == -0.30000000000000004 or abs(
        reward_from_taps([0.2, 0.4]) + 0.3) < 1e-12
    assert reward_from_taps(None) is None
    assert reward_from_taps([float("inf"), 0.1]) is None


def test_expand_for_bon_pairs_samples_with_their_row():
    B, k, d, N = 3, 2, 4, 5
    acts = np.arange(B * k * d, dtype=np.float32).reshape(B, k, d)
    flat = expand_for_bon(acts, N)                     # [B*N*k, d]
    assert flat.shape == (B * N * k, d)
    # prompt batch layout: row i's N samples are consecutive; each sample's k
    # slot vectors must be row i's, in slot order.
    for i in range(B):
        for j in range(N):
            for s in range(k):
                got = flat[(i * N + j) * k + s]
                assert np.array_equal(got, acts[i, s]), (i, j, s)


def test_expand_for_bon_n1_identity():
    acts = np.random.default_rng(0).standard_normal((4, 3, 5)).astype(np.float32)
    assert np.array_equal(expand_for_bon(acts, 1), acts.reshape(-1, 5))


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print("\nALL PASSED")


if __name__ == "__main__":
    _run_all()
