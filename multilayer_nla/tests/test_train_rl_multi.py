"""GRPO math tests for the multi-layer RL trainer (plan §6.3 Fix 4, §8).
Pure torch — no peft/model. Run: python -m multilayer_nla.tests.test_train_rl_multi
"""

import torch

from multilayer_nla.train_rl_multi import (
    compute_group_advantages,
    grpo_surrogate,
    rollout_multislot,
)


def test_fix4_kl_zero_when_reference_equals_actor():
    """The Fix-4 step-0 property: actor initialized from the AV-SFT reference =>
    ref_lp == new_lp => KL == 0. This is what the trainer asserts at step 0."""
    torch.manual_seed(0)
    new_lp = torch.randn(6)
    old_lp = new_lp.clone()          # on-policy (step 0)
    ref_lp = new_lp.clone()          # reference == actor at init
    loss, kl, clip = grpo_surrogate(new_lp, old_lp, ref_lp, torch.tensor(1.0), kl_beta=0.01)
    assert abs(kl.item()) < 1e-7, f"KL should be 0 when reference==actor, got {kl.item()}"
    # ratio==1, kl==0, A=1 -> per-token loss = -(A - 0) = -1
    assert abs(loss.item() - (-1.0)) < 1e-5


def test_kl_positive_when_reference_differs():
    new_lp = torch.zeros(6)
    old_lp = new_lp.clone()
    ref_lp = torch.full((6,), -0.5)  # reference != actor
    _, kl, _ = grpo_surrogate(new_lp, old_lp, ref_lp, torch.tensor(0.0), kl_beta=0.01)
    assert kl.item() > 0.0           # k3 estimator is >= 0, strictly > 0 here


def test_clip_fraction_saturates_outside_band():
    new_lp = torch.zeros(4)
    old_lp = torch.full((4,), -2.0)  # ratio = exp(2) >> 1 + eps
    ref_lp = new_lp.clone()
    _, _, clip = grpo_surrogate(new_lp, old_lp, ref_lp, torch.tensor(1.0), clip_eps=0.2)
    assert clip.item() == 1.0


def test_surrogate_pessimistic_clipping():
    # positive advantage, ratio pushed high -> min() takes the clipped term
    new_lp = torch.zeros(3)
    old_lp = torch.full((3,), -1.0)  # ratio = e ≈ 2.718, clipped to 1.2
    ref_lp = new_lp.clone()
    loss, _, _ = grpo_surrogate(new_lp, old_lp, ref_lp, torch.tensor(1.0), clip_eps=0.2, kl_beta=0.0)
    # surrogate = min(2.718*1, 1.2*1) = 1.2 -> loss = -1.2
    assert abs(loss.item() - (-1.2)) < 1e-4


def test_group_advantages_normalize_per_group():
    rewards = torch.tensor([1., 2., 3., 10., 10., 10.])
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    adv = compute_group_advantages(rewards, groups, n_groups=2)
    # group 0: mean 2, unbiased std 1 -> ~[-1, 0, 1]
    assert torch.allclose(adv[:3], torch.tensor([-1., 0., 1.]), atol=1e-3)
    # group 1: zero variance -> advantages ~ 0 (no NaN from /0)
    assert torch.allclose(adv[3:], torch.zeros(3), atol=1e-3)
    assert torch.isfinite(adv).all()


def test_rollout_payload_is_prompt_bounded_no_suppress():
    """rollout_multislot must (a) set the {vectors, prompt_lens} payload so injection is
    bounded to the prompt span, and (b) NOT pass suppress_tokens (which would renormalize
    the sampled token prob and desync GRPO old/new/ref log-probs). Fake actor/tokenizer,
    no model."""
    INJ = 9
    vref = [None]

    class FakeTok:
        eos_token_id = 0
        def encode(self, s, add_special_tokens=False):
            return [5, 6, 7]                       # plen = 3
        def decode(self, ids, skip_special_tokens=True):
            return "resp"

    class FakeGen:
        pass

    class FakeActor:
        def __init__(self, vref):
            self.vref = vref; self.kw = None; self.payload = None
        def generate(self, **kw):
            self.kw = kw
            self.payload = self.vref[0]            # rollout set this right before generate
            G = kw["input_ids"].shape[0]
            o = FakeGen()
            o.sequences = torch.cat([kw["input_ids"], torch.tensor([[8, 0]] * G)], dim=1)
            o.logits = (torch.zeros(G, 16), torch.zeros(G, 16))
            return o

    actor = FakeActor(vref)
    acts3 = [[0.1] * 4, [0.2] * 4, [0.3] * 4]      # [3, d]
    out = rollout_multislot(actor, FakeTok(), "prompt", acts3, vref, INJ, 2, 2, 1.0, "cpu", {0})
    assert isinstance(actor.payload, dict) and "vectors" in actor.payload
    assert actor.payload["prompt_lens"].tolist() == [3, 3]     # plen=3, group_size=2
    assert "suppress_tokens" not in actor.kw                   # reverted; injection is prompt-bounded
    assert vref[0] is None                                     # cleared after the forward
    assert len(out) == 2 and all("full_ids" in r and "old_logp" in r for r in out)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} RL GRPO-math tests passed.")


if __name__ == "__main__":
    _run_all()
