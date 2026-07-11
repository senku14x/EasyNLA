"""Multi-slot Karvonen injection for the three-layer AV (plan §6.1).

Generalizes `nla.injection.karvonen_inject_in_residual` from ONE marker/example
to K markers/example (K=3: prev/centre/next depth slots). The AV prompt repeats
the SAME marker char K times — slot identity is by ORDER, not by a distinct
token — so this scans for that one marker id and injects K vectors per example.

Vector layout (the contract the dataset side must honor):
    vectors is [B*K, d], the row-major flatten of [B, K, d] where the K axis is
    [a^(l-1), a^(l), a^(l+1)] = [prev, centre, next] in PROMPT order.
    The scan is row-major (batch row, then ascending sequence position), so
    `vec_idx` walks example b's K markers in prompt order before moving to
    example b+1 — matching that flatten exactly.

Guards (this is the "assert exactly 3B valid injection sites" of plan §6.1):
    - EXACTLY K markers per batch row (catches a stray marker in the response,
      template drift, or a wrong K) — fires per-row so the diagnosis is precise.
    - B*K markers total == vectors.shape[0].

RAW vectors in; each is norm-matched to the residual it lands on (plan §4 Rev 2,
the AV reads raw activations):
    h'_p = h_p + ||h_p|| * v / ||v||
Identical math to the single-slot path — only the count/order bookkeeping is new.
"""

import torch


def inject_multislot_in_residual(
    input_ids: torch.Tensor,
    resid: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    k: int,
    prompt_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """ADD-norm-matched injection at K marker positions per example.

    input_ids: [B, S] full token stream (prompt[+response]).
    resid: [B, S, d] residual to modify (cloned; original untouched).
    vectors: [B*K, d] in example-major, prompt-slot order (see module docstring).
    inj_id: the marker token id (same char for all K slots).
    k: markers expected per example.
    prompt_lens: [B] (optional). When given, ONLY markers at positions < prompt_lens[b]
        are injection sites. This is required for RL: GRPO re-forwards the full
        prompt+response sequence, but the rollout injected ONLY at the prompt prefill
        (response tokens are generated one-at-a-time, where the hook no-ops), so a
        marker the actor happened to emit in the RESPONSE is NOT an injection site.
        Bounding to the prompt span keeps the training-forward injection identical to
        the rollout's, and stops response markers from shifting slot order / tripping
        the count guard / crashing the update. When None, all markers count (AV-SFT,
        whose gold response is marker-free).

    Returns the modified residual. Raises if the per-row marker count (within the
    prompt span when bounded) != k or the total != vectors.shape[0].
    """
    B, S = input_ids.shape
    assert resid.shape[:2] == (B, S), (
        f"input_ids {tuple(input_ids.shape)} and resid {tuple(resid.shape[:2])} batch/seq must match"
    )
    assert vectors.ndim == 2 and vectors.shape[1] == resid.shape[-1], (
        f"vectors must be [B*K, d], got {tuple(vectors.shape)}, d={resid.shape[-1]}"
    )
    assert vectors.shape[0] == B * k, (
        f"expected B*k = {B}*{k} = {B * k} vectors, got {vectors.shape[0]}"
    )

    marker = input_ids == inj_id
    if prompt_lens is not None:
        # Only markers inside each row's prompt span are injection sites.
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)              # [1, S]
        marker = marker & (pos < prompt_lens.to(input_ids.device).unsqueeze(1))  # [B, S]
    per_row = marker.sum(dim=1)
    if not torch.all(per_row == k):
        where = "the prompt span" if prompt_lens is not None else "each example"
        raise RuntimeError(
            f"[inject_multislot] {where} must have exactly k={k} markers; "
            f"got per-row counts {per_row.tolist()}. Cause: AV-prompt template drift, "
            f"a wrong k/tokenizer, or (if unbounded) a stray marker in the response."
        )

    out = resid.clone()
    vectors = vectors.to(out.device, out.dtype)
    matches = marker.nonzero()  # [B*k, 2] (b, p), row-major: b ascending, then p ascending
    vec_idx = 0
    for b, p in matches.tolist():
        # Clone the slice before reading — out[b, p] is a view into out's storage;
        # the in-place write below would otherwise trip autograd's
        # "modified by inplace op" at backward (same fix as the single-slot path).
        h_p = out[b, p].clone()
        v = vectors[vec_idx]
        v_unit = v / (v.norm() + 1e-9)
        out[b, p] = h_p + h_p.norm() * v_unit
        vec_idx += 1
    assert vec_idx == vectors.shape[0], (
        f"[inject_multislot] injected {vec_idx} != {vectors.shape[0]} vectors (internal bug)"
    )
    return out


def register_multislot_hook(model, vectors_ref, inj_id: int, k: int, layer_idx: int = 1):
    """Attach the embed-capture + layer-`layer_idx` injection hooks for K-slot AV.

    Mirrors `nla.train_sft._register_karvonen_hook` but routes through
    `inject_multislot_in_residual`. `vectors_ref[0]` holds the current forward's
    injection payload (set by the caller before each forward, cleared after):
      - None                                              → no-op
      - {"vectors": [B*K, d], "prompt_lens": [B] | None}  → inject; prompt_lens
        bounds injection to the prompt span (RL: the generated response may contain
        marker tokens that are NOT injection sites — see inject_multislot_in_residual)
      - a bare [B*K, d] tensor (legacy)                   → inject, no prompt bounding

    No-op when there's no full sequence to inject into (seq_len < 2, e.g. the
    single-token cache steps during RL `generate()`) or when no marker is present
    — so the same hook is safe for both SFT forwards and autoregressive rollout.
    """
    state = {"input_ids": None}

    def embed_hook(module, args, kwargs, output):
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        state["input_ids"] = ids
        return output

    def layer_hook(module, args, output):
        if isinstance(output, tuple):
            resid, *rest = output
        else:
            resid, rest = output, None
        input_ids = state["input_ids"]
        if input_ids is None or resid.shape[1] < 2:
            return output
        payload = vectors_ref[0]
        if payload is None:
            return output
        if isinstance(payload, dict):
            v, plens = payload.get("vectors"), payload.get("prompt_lens")
        else:  # legacy bare tensor — no prompt bounding
            v, plens = payload, None
        if v is None or v.shape[0] == 0:
            return output
        ids = input_ids.to(resid.device)
        if (ids == inj_id).sum().item() == 0:
            return output  # cache step / no marker — nothing to inject
        injected = inject_multislot_in_residual(
            ids, resid, v.to(resid.device), inj_id, k,
            prompt_lens=(plens.to(resid.device) if plens is not None else None),
        )
        if rest is None:
            return injected
        return (injected, *rest)

    model.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
    target = model.base_model if hasattr(model, "base_model") else model
    while hasattr(target, "model") and not hasattr(target, "layers"):
        target = target.model
    target.layers[layer_idx].register_forward_hook(layer_hook)
