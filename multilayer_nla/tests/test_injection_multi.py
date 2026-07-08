"""Validate the multi-slot injection core (plan §6.1) — runs on CPU, no model
download. Covers: correct norm-matched values at the right positions in the
right slot order, the per-row and total count guards, and that the hook actually
perturbs a tiny model's output.

Run:  python -m multilayer_nla.tests.test_injection_multi
"""

import torch

from multilayer_nla.injection_multi import (
    inject_multislot_in_residual,
    register_multislot_hook,
)

INJ = 777


def test_injects_correct_vectors_in_scan_order():
    torch.manual_seed(0)
    B, S, d, k = 2, 9, 8, 3
    ids = torch.randint(0, 50, (B, S))
    ids[ids == INJ] = 0
    marker_pos = {0: [1, 4, 7], 1: [2, 3, 6]}  # ascending within each row
    for b, ps in marker_pos.items():
        for p in ps:
            ids[b, p] = INJ
    resid = torch.randn(B, S, d)
    vectors = torch.randn(B * k, d)  # example-major, slot order

    out = inject_multislot_in_residual(ids, resid, vectors, INJ, k)

    # each marker got h_p + ||h_p|| * v/||v||, with v walked row-major
    vec_idx = 0
    for b in range(B):
        for p in sorted(marker_pos[b]):
            h_p = resid[b, p]
            v = vectors[vec_idx]
            expected = h_p + h_p.norm() * v / (v.norm() + 1e-9)
            assert torch.allclose(out[b, p], expected, atol=1e-5), f"slot mismatch at ({b},{p})"
            vec_idx += 1
    assert vec_idx == B * k

    # non-marker positions untouched; input resid not mutated
    is_marker = ids == INJ
    for b in range(B):
        for p in range(S):
            if not is_marker[b, p]:
                assert torch.equal(out[b, p], resid[b, p]), f"non-marker ({b},{p}) changed"


def test_per_row_count_guard_fires():
    B, S, d, k = 2, 8, 4, 3
    ids = torch.zeros(B, S, dtype=torch.long)
    ids[0, [1, 3, 5]] = INJ          # 3 markers (ok)
    ids[1, [2, 4]] = INJ             # 2 markers (wrong)
    try:
        inject_multislot_in_residual(ids, torch.randn(B, S, d), torch.randn(B * k, d), INJ, k)
        raise AssertionError("expected RuntimeError on per-row count mismatch")
    except RuntimeError as e:
        assert "exactly k=3" in str(e), str(e)


def test_total_vector_count_guard_fires():
    B, S, d, k = 2, 8, 4, 3
    ids = torch.zeros(B, S, dtype=torch.long)
    ids[0, [1, 3, 5]] = INJ
    ids[1, [2, 4, 6]] = INJ
    try:
        inject_multislot_in_residual(ids, torch.randn(B, S, d), torch.randn(B * k - 1, d), INJ, k)
        raise AssertionError("expected AssertionError on B*k vector-count mismatch")
    except AssertionError as e:
        assert "vectors" in str(e).lower(), str(e)


def test_hook_perturbs_tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=200, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2)
    model = LlamaForCausalLM(cfg).eval()
    inj, k = 150, 3
    vref = [None]
    register_multislot_hook(model, vref, inj, k, layer_idx=1)

    B, S = 2, 10
    ids = torch.randint(0, 140, (B, S))
    for b in range(B):
        ids[b, [1, 4, 7]] = inj  # exactly 3 markers/row

    with torch.no_grad():
        base = model(input_ids=ids).logits          # vref None -> hook no-op
        vref[0] = torch.randn(B * k, cfg.hidden_size)
        injected = model(input_ids=ids).logits
        vref[0] = None
    assert not torch.allclose(base, injected), "injection hook had no effect on the model output"


def test_prompt_bounded_injection_ignores_response_markers():
    """Reproduces the [9,3] RL crash: row 0 = 3 prompt markers + 6 response markers,
    row 1 = 3 prompt. With prompt_lens delimiting the prompt, only the 3 prompt sites
    per row are injected and the response markers are untouched — no exception."""
    torch.manual_seed(0)
    B, S, d, k = 2, 14, 8, 3
    plen = 6
    ids = torch.randint(0, 50, (B, S)); ids[ids == INJ] = 0
    prompt_marker = {0: [1, 3, 5], 1: [0, 2, 4]}                # all < plen
    for b, ps in prompt_marker.items():
        for p in ps:
            ids[b, p] = INJ
    ids[0, [6, 8, 9, 10, 11, 12]] = INJ                        # 6 stray response markers (>= plen)
    resid = torch.randn(B, S, d)
    vectors = torch.randn(B * k, d)
    prompt_lens = torch.tensor([plen, plen])

    out = inject_multislot_in_residual(ids, resid, vectors, INJ, k, prompt_lens=prompt_lens)

    vec_idx = 0
    for b in range(B):
        for p in sorted(prompt_marker[b]):                     # exactly the 3 prompt slots
            v = vectors[vec_idx]
            expected = resid[b, p] + resid[b, p].norm() * v / (v.norm() + 1e-9)
            assert torch.allclose(out[b, p], expected, atol=1e-5), f"prompt slot ({b},{p}) wrong"
            vec_idx += 1
    assert vec_idx == B * k
    for p in [6, 8, 9, 10, 11, 12]:                            # response markers NOT injected
        assert torch.equal(out[0, p], resid[0, p]), f"response marker pos {p} was injected"


def test_unbounded_counts_response_markers_and_raises():
    """Without prompt_lens the same row-0 sequence has 9 markers -> count guard fires.
    (Pre-fix behavior; prompt-bounding is what makes RL full-sequence scoring safe.)"""
    B, S, d, k = 2, 14, 8, 3
    ids = torch.zeros(B, S, dtype=torch.long)
    ids[0, [1, 3, 5]] = INJ; ids[0, [6, 8, 9, 10, 11, 12]] = INJ   # 9 total
    ids[1, [0, 2, 4]] = INJ                                        # 3
    try:
        inject_multislot_in_residual(ids, torch.randn(B, S, d), torch.randn(B * k, d), INJ, k)
        raise AssertionError("expected RuntimeError on 9 markers without prompt bounding")
    except RuntimeError as e:
        assert "exactly k=3" in str(e)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} injection tests passed.")


if __name__ == "__main__":
    _run_all()
