"""Multi-tap AR (reconstructor) + three-target reward for the three-layer NLA
(plan §6.2, §6.3, §8, point 4-5).

The AR reads ONLY the explanation text z and reconstructs all three depth states:
    z -> [u^(l-1), u^(l), u^(l+1)]  =  [u^23, u^24, u^25]   (center l = 24)

Architecture (plan §6.2 Rev 2 — heads tap their OWN depth, not one final vector):
  - one backbone truncated through block 25 (so block 25's output exists);
  - final RMSNorm stripped (the heads see the RAW residual stream — same as the
    single-layer NLACriticModel and as extraction);
  - a forward hook on each tap block {23,24,25} captures its OUTPUT — identical
    semantics to extract_multilayer (gold a^K = output of block K), so the AR
    predicts the same kind of vector it is graded against;
  - three Linear(d,d) heads, identity-init, one per tap. Each reads its tap's
    last-token hidden state.

Predict (plan §6.3): normalize the tapped hidden to mse_scale BEFORE the head
(bounds the head's input norm — the bf16+Adam blow-up fix from the single-layer
path) then apply the head. At identity init this is just normalize(tap_last).

Loss / reward (point 5): both predictions and targets are √d-normalized, then
    L_state = (1/3d) Σ_j ||û^j - u^j||^2 ,   r = -L_state .
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nla.utils.arch_adapters import resolve_decoder_layers, resolve_text_config
from nla.schema import normalize_activation

# Tap layers for center l=24: [prev, centre, next] = [23, 24, 25].
# Order MUST match datasets.SLOT_COLUMNS so pred[:, j] lines up with gold[:, j].
DEFAULT_TAP_LAYERS = (23, 24, 25)


class MultiTapCriticModel(nn.Module):
    """Truncated backbone + per-depth taps + three reconstruction heads.

    `backbone` must already be truncated to (max(tap_layers)+1) blocks with its
    final norm stripped (use `init_multitap_critic_from_base`). Hooks capture each
    tap block's output WITHOUT detaching, so gradients flow head -> backbone for
    AR training.
    """

    def __init__(self, backbone, tap_layers=DEFAULT_TAP_LAYERS, d_model=None):
        super().__init__()
        self.backbone = backbone
        self.tap_layers = list(tap_layers)
        if d_model is None:
            d_model = resolve_text_config(backbone.config).hidden_size
        self.d_model = d_model
        self.heads = nn.ModuleDict({
            str(l): nn.Linear(d_model, d_model, bias=False) for l in self.tap_layers
        })
        with torch.no_grad():
            for l in self.tap_layers:
                self.heads[str(l)].weight.copy_(torch.eye(d_model))
        self._captured: dict[int, torch.Tensor] = {}
        self._handles = []
        layers = resolve_decoder_layers(backbone)
        for l in self.tap_layers:
            assert 0 <= l < len(layers), (
                f"tap layer {l} out of range for backbone with {len(layers)} blocks "
                f"(truncate to >= {max(self.tap_layers) + 1})"
            )
            self._handles.append(layers[l].register_forward_hook(self._make_hook(l)))

    def _make_hook(self, l):
        def hook(_m, _inp, output):
            # output of block l (post-MLP, post-residual-add) — keep grad attached.
            self._captured[l] = output[0] if isinstance(output, tuple) else output
        return hook

    def _inner(self):
        bb = self.backbone
        if hasattr(bb, "model"):
            return bb.model
        if hasattr(bb, "transformer"):
            return bb.transformer
        raise AssertionError(f"{type(bb).__name__}: no .model/.transformer inner module")

    def forward(self, input_ids, attention_mask=None):
        """Run the backbone; return {tap_layer: [B, T, d]} captured block outputs."""
        self._captured = {}
        self._inner()(input_ids=input_ids, attention_mask=attention_mask)
        for l in self.tap_layers:
            assert l in self._captured, f"tap hook for block {l} did not fire"
        return dict(self._captured)

    def head_state_dict(self):
        """Just the three head weights (for AR-LoRA-style checkpoints)."""
        return {f"heads.{l}.weight": self.heads[str(l)].weight.detach().cpu()
                for l in self.tap_layers}

    def load_head_state_dict(self, sd):
        for l in self.tap_layers:
            self.heads[str(l)].weight.data.copy_(sd[f"heads.{l}.weight"].to(self.heads[str(l)].weight.device))


def init_multitap_critic_from_base(base_ckpt, tap_layers=DEFAULT_TAP_LAYERS, dtype=torch.bfloat16,
                                   quant_config=None, device_map=None, max_memory=None,
                                   strip_final_norm=True):
    """Build a MultiTapCriticModel from a base checkpoint.

    Reuses `nla.train_sft.init_critic_from_base` for the (validated) truncation +
    final-norm strip + lm_head drop + 4-bit/QLoRA placement, then discards its
    single value head and wraps the backbone with the three depth heads.
    Truncates to max(tap_layers)+1 blocks so the highest tap's output exists.
    """
    from nla.train_sft import init_critic_from_base
    num_layers = max(tap_layers) + 1
    base_critic = init_critic_from_base(
        base_ckpt, num_layers, dtype, quant_config,
        device_map=device_map, max_memory=max_memory, strip_final_norm=strip_final_norm,
    )
    backbone = base_critic.backbone  # truncated, norm-stripped, lm_head=Identity
    d_model = resolve_text_config(backbone.config).hidden_size
    model = MultiTapCriticModel(backbone, tap_layers, d_model)
    # Align heads with the backbone's last-block placement (matches single-layer path).
    last = next(resolve_decoder_layers(backbone)[-1].parameters())
    with torch.no_grad():
        for l in tap_layers:
            model.heads[str(l)].to(device=last.device, dtype=dtype)
            model.heads[str(l)].weight.copy_(torch.eye(d_model, dtype=dtype))
    print(f"[multitap-critic] truncated to {num_layers} blocks, taps {list(tap_layers)}, "
          f"{len(tap_layers)} identity-init heads (d={d_model})")
    return model


def multitap_predict(model, input_ids, attention_mask, mse_scale):
    """Per-tap last-token prediction. Returns [B, n_taps, d] in tap_layers order.

    pred_j = head_j(normalize(tap_j_last, mse_scale)). At identity init this is
    just normalize(tap_j_last) — the AR's prediction is the truncated forward's
    layer-j residual, which is what we want before any head learning (plan §6.3).
    """
    tap = model(input_ids, attention_mask)
    B = input_ids.shape[0]
    if attention_mask is not None:
        last_idx = attention_mask.sum(dim=1) - 1
    else:
        last_idx = torch.full((B,), input_ids.shape[1] - 1, device=input_ids.device)
    ar = torch.arange(B, device=input_ids.device)
    preds = []
    for l in model.tap_layers:
        last_h = tap[l][ar, last_idx].float()
        last_h_norm = normalize_activation(last_h, mse_scale)
        head = model.heads[str(l)]
        preds.append(head(last_h_norm.to(head.weight.dtype)).float())
    return torch.stack(preds, dim=1)  # [B, n_taps, d]


def _assert_tap_target_aligned(pred, gold):
    """pred[B,k,d] and gold[B,k,d] must have the SAME tap count, else F.mse_loss /
    the (pred-gold) subtraction silently BROADCASTS (e.g. a 1-tap pred against a
    3-target gold trains the head on the mean of all three). Fail loud instead."""
    assert pred.shape[1] == gold.shape[1], (
        f"AR tap/target mismatch: pred {tuple(pred.shape)} vs gold {tuple(gold.shape)}. "
        f"Pass target_cols matching the model's tap_layers (AR_LAYER_TO_TARGET_COL) so a "
        f"single-/two-tap AR pulls only its target column(s) — otherwise the loss broadcasts."
    )


def three_target_loss(pred, gold, mse_scale):
    """L_state = (1/kd) Σ_j ||û^j - u^j||^2, averaged over the batch (plan §6.3).

    pred, gold: [B, n_taps, d] (gold is RAW; both normalized to mse_scale here so
    the cosine identity holds — plan §6.3 note). Mean over all elements equals
    the per-sample (1/kd)Σ_j averaged over B. n_taps may be 1 (capacity ablation).
    """
    _assert_tap_target_aligned(pred, gold)
    pred_n = normalize_activation(pred, mse_scale)
    gold_n = normalize_activation(gold, mse_scale)
    return F.mse_loss(pred_n, gold_n)


def three_target_reward(pred, gold, mse_scale):
    """Per-sample reward r = -(1/kd) Σ_j ||û^j - u^j||^2 (point 5). Returns [B]."""
    _assert_tap_target_aligned(pred, gold)
    pred_n = normalize_activation(pred, mse_scale)
    gold_n = normalize_activation(gold, mse_scale)
    return -((pred_n - gold_n) ** 2).mean(dim=(1, 2))
