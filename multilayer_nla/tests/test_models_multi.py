"""Validate the multi-tap AR + three-target reward (plan §6.2, §6.3, point 4-5).
CPU, tiny in-memory Llama — no download.

Run:  python -m multilayer_nla.tests.test_models_multi
"""

import math

import torch

from nla.schema import normalize_activation
from multilayer_nla.models_multi import (
    MultiTapCriticModel,
    multitap_predict,
    three_target_loss,
    three_target_reward,
)


def _tiny_backbone(num_layers=6, d=16):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=128, hidden_size=d, intermediate_size=2 * d,
                      num_hidden_layers=num_layers, num_attention_heads=2, num_key_value_heads=2)
    m = LlamaForCausalLM(cfg).eval()
    m.model.norm = torch.nn.Identity()      # strip final norm (heads see raw residual)
    m.lm_head = torch.nn.Identity()
    return m, d


def test_forward_captures_taps_and_matches_block_outputs():
    torch.manual_seed(0)
    backbone, d = _tiny_backbone(num_layers=6)
    taps = (3, 4, 5)
    model = MultiTapCriticModel(backbone, tap_layers=taps, d_model=d)

    # independent manual hook on block 4 to verify the tap captures that block's output
    ref = {}
    h = backbone.model.layers[4].register_forward_hook(
        lambda _m, _i, o: ref.__setitem__("c", (o[0] if isinstance(o, tuple) else o).clone()))

    ids = torch.randint(0, 120, (2, 7))
    tap = model(ids, attention_mask=torch.ones_like(ids))
    h.remove()

    assert set(tap.keys()) == set(taps)
    for l in taps:
        assert tap[l].shape == (2, 7, d)
    assert torch.allclose(tap[4], ref["c"]), "tap[4] is not block-4's output"


def test_multitap_predict_shape_and_identity_init():
    torch.manual_seed(1)
    backbone, d = _tiny_backbone(num_layers=6)
    taps = (3, 4, 5)
    model = MultiTapCriticModel(backbone, tap_layers=taps, d_model=d)
    mse_scale = math.sqrt(d)

    B, T = 3, 9
    ids = torch.randint(0, 120, (B, T))
    attn = torch.ones(B, T, dtype=torch.long)
    pred = multitap_predict(model, ids, attn, mse_scale)
    assert pred.shape == (B, len(taps), d)

    # identity init: pred_j == normalize(tap_j_last) (head = I, normalize-before-head)
    tap = model(ids, attn)
    last_idx = attn.sum(1) - 1
    ar = torch.arange(B)
    for j, l in enumerate(taps):
        expect = normalize_activation(tap[l][ar, last_idx].float(), mse_scale)
        assert torch.allclose(pred[:, j], expect, atol=1e-5), f"tap {l} identity-init mismatch"

    # each predicted vector is ~sqrt(d) in norm (so the cosine identity holds, §6.3)
    norms = pred.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, mse_scale), atol=1e-3)


def test_three_target_loss_and_reward_math():
    torch.manual_seed(2)
    B, n, d = 4, 3, 8
    mse_scale = math.sqrt(d)
    pred = torch.randn(B, n, d)
    gold = torch.randn(B, n, d)

    pn = normalize_activation(pred, mse_scale)
    gn = normalize_activation(gold, mse_scale)

    # loss = mean over all elements of normalized squared error
    loss = three_target_loss(pred, gold, mse_scale)
    assert torch.allclose(loss, ((pn - gn) ** 2).mean(), atol=1e-6)
    # perfect reconstruction -> 0
    assert three_target_loss(gold, gold, mse_scale).abs() < 1e-6

    # reward is per-sample [B], = -(1/(n*d)) Σ ||.||^2
    r = three_target_reward(pred, gold, mse_scale)
    assert r.shape == (B,)
    assert torch.allclose(r, -((pn - gn) ** 2).mean(dim=(1, 2)), atol=1e-6)
    assert three_target_reward(gold, gold, mse_scale).abs().max() < 1e-6
    # reward equals the loss (up to sign) when averaged over the batch
    assert torch.allclose(r.mean(), -loss, atol=1e-6)


def test_heads_are_trainable_and_distinct():
    backbone, d = _tiny_backbone(num_layers=6)
    model = MultiTapCriticModel(backbone, tap_layers=(3, 4, 5), d_model=d)
    assert len(model.heads) == 3
    # heads are independent modules (not the same object)
    assert model.heads["3"] is not model.heads["4"]
    # head weights carry grad (trainable); backbone params exist too
    head_params = sum(p.numel() for p in model.heads.parameters())
    assert head_params == 3 * d * d
    assert all(p.requires_grad for p in model.heads.parameters())


def test_logits_to_keep_preserves_decoder_layer_outputs():
    """regen passes logits_to_keep=1 to skip the all-position lm_head logits (~60GB +
    a big vocab matmul at S=4096). The decoder hidden states the hooks capture MUST be
    unchanged — that's what makes the speedup safe (bitwise-identical taps)."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=2)
    m = LlamaForCausalLM(cfg).eval()
    ids = torch.randint(0, 120, (2, 7))
    cap = []
    h = m.model.layers[3].register_forward_hook(
        lambda _mod, _in, out: cap.append((out[0] if isinstance(out, tuple) else out).clone()))
    with torch.no_grad():
        m(input_ids=ids)                       # full forward (all-position logits)
        m(input_ids=ids, logits_to_keep=1)     # last-token logits only (the regen path)
    h.remove()
    assert torch.allclose(cap[0], cap[1], atol=0), "logits_to_keep changed a decoder layer output"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} multi-tap AR tests passed.")


if __name__ == "__main__":
    _run_all()
