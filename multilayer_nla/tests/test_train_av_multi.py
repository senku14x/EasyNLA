"""Plumbing test for the AV-SFT loss step (plan §6.1): three-slot injection +
response-only CE + backprop, on a tiny in-memory model (no tokenizer/download).

Run:  python -m multilayer_nla.tests.test_train_av_multi
"""

import torch

from multilayer_nla.injection_multi import register_multislot_hook
from multilayer_nla.train_av_multi import av_compute_loss, build_lr_lambda, cjk_fraction


def _tiny_lm(d=16, layers=4, vocab=200):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=d, intermediate_size=2 * d,
                      num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=2)
    return LlamaForCausalLM(cfg).eval()


def test_av_loss_injects_and_backprops():
    torch.manual_seed(0)
    model = _tiny_lm()
    d = model.config.hidden_size
    inj, k = 150, 3
    vref = [None]
    register_multislot_hook(model, vref, inj, k, layer_idx=1)

    B, S = 2, 12
    ids = torch.randint(0, 140, (B, S))
    for b in range(B):
        ids[b, [1, 4, 7]] = inj                       # exactly 3 markers/row
    attn = torch.ones(B, S, dtype=torch.long)
    loss_mask = torch.zeros(B, S)
    loss_mask[:, 8:] = 1.0                             # response region after the markers
    vectors = torch.randn(B * k, d)

    loss, n_resp = av_compute_loss(model, ids, attn, loss_mask, vectors, vref)
    assert torch.isfinite(loss) and loss.item() > 0
    assert n_resp == int(loss_mask[:, 1:].sum().item())
    assert vref[0] is None, "vectors_ref must be cleared after the forward"

    # gradient flows to the model
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)

    # injection actually changed the loss vs. no-injection (vref stays None)
    with torch.no_grad():
        base_logits = model(input_ids=ids, attention_mask=attn).logits.float()
        vref[0] = vectors
        inj_logits = model(input_ids=ids, attention_mask=attn).logits.float()
        vref[0] = None
    assert not torch.allclose(base_logits, inj_logits), "three-slot injection had no effect"


def test_av_loss_fires_count_guard_on_wrong_marker_count():
    model = _tiny_lm()
    d = model.config.hidden_size
    inj, k = 150, 3
    vref = [None]
    register_multislot_hook(model, vref, inj, k, layer_idx=1)
    B, S = 2, 12
    ids = torch.randint(0, 140, (B, S))
    ids[ids == inj] = 0
    ids[0, [1, 4, 7]] = inj      # 3 markers
    ids[1, [2, 5]] = inj         # 2 markers (wrong) -> hook should raise
    attn = torch.ones(B, S, dtype=torch.long)
    loss_mask = torch.zeros(B, S); loss_mask[:, 8:] = 1.0
    try:
        av_compute_loss(model, ids, attn, loss_mask, torch.randn(B * k, d), vref)
        raise AssertionError("expected RuntimeError from the per-row marker guard")
    except RuntimeError as e:
        assert "exactly k=3" in str(e)
    finally:
        assert vref[0] is None


def test_lr_schedule_and_cjk():
    fn = build_lr_lambda(warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    assert fn(0) == 0.0
    assert abs(fn(10) - 1.0) < 1e-9          # peak at warmup end
    assert abs(fn(100) - 0.1) < 1e-6         # decays to min ratio
    assert fn(5) < fn(10)                    # warming up
    assert cjk_fraction("hello") == 0.0
    assert cjk_fraction("中文") > 0.9        # the marker-leak canary catches CJK


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} AV-SFT plumbing tests passed.")


if __name__ == "__main__":
    _run_all()
