"""Multi-vector dataset loading for the three-layer NLA (plan §6.1, §7, point 2).

Three responsibilities, all preserving ALL THREE activation columns
(prev/centre/next) end to end — never collapsing to a single `activation_vector`:

  1. The AV prompt with three depth slots (the same marker char x3; slot identity
     is by order). `INJECT_PLACEHOLDER` x3, swapped to the marker char at prep.
  2. `split_by_document` — document-level 3-way split (av_sft / ar_sft / rl) that
     routes every position of a doc to ONE bucket and carries every column.
  3. `stack_slot_vectors` / `prepare_av_chunk_multi` — build the [B*K, d] vector
     tensor in the exact scan order `inject_multislot_in_residual` expects
     ([a_prev, a_centre, a_next] per example, example-major), plus the AV-SFT
     (input_ids, attn, response-only loss_mask).

The base parquet (from extract_multilayer.py) has columns:
    n_raw_tokens, [detokenized_text_truncated], activation_prev/centre/next,
    center_layer, doc_id.
AV/AR SFT parquets add `prompt` (and AV adds `response`) downstream; this module
loads + preps them.
"""

import hashlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nla.schema import INJECT_PLACEHOLDER

# ---- AV prompt: k depth slots, same marker char (order = slot) ----
# Kept as module constants so the builder and training prep use the SAME strings —
# drift moves the marker positions and breaks injection. Across the §7 conditions
# ONLY the AV *input* changes: which layers fill the slots, and k=3 (local /
# duplicate / wide) vs k=1 (single). The AR reconstruction target is INDEPENDENT
# and ALWAYS the [L23,L24,L25] triplet (AR_TARGET_COLUMNS below).
AV_USER_TEMPLATE = (
    "Earlier depth: {m}\n"
    "Centre depth: {m}\n"
    "Later depth: {m}\n"
    "Describe the information represented across these local computational stages."
)
# One-marker variant for the `single` condition (standard single-layer baseline):
# same task framing, one slot. `single` thus differs from the 3-marker conditions
# in marker structure as well as layer count (hence a secondary comparison).
AV_USER_TEMPLATE_SINGLE = (
    "Depth: {m}\n"
    "Describe the information represented at this local computational stage."
)
# Two-marker variant (k=2 conditions, e.g. AV input [L20,L24] or [L24,L28]): two distinct
# input layers in ascending slot order. Mirrors the 3-marker framing, drops the Centre line.
AV_USER_TEMPLATE_DUAL = (
    "Earlier depth: {m}\n"
    "Later depth: {m}\n"
    "Describe the information represented across these local computational stages."
)
N_SLOTS = 3


def av_in_columns(k: int) -> list:
    """Positional AV-input slot column names (slot identity is by order)."""
    return [f"av_in_{i}" for i in range(k)]


def detect_av_slots(parquet_path: str) -> list:
    """The AV-input slot columns present in a parquet, in slot order.

    Prefers the sweep's positional av_in_* columns. Falls back to the legacy
    prev/centre/next triplet (build_from_published / build_datasets output, used by
    the preserved smoke pipeline) so old-format AV parquets still load as k=3.
    """
    names = pq.ParquetFile(parquet_path).schema_arrow.names
    cols = sorted((n for n in names if n.startswith("av_in_")),
                  key=lambda s: int(s.rsplit("_", 1)[-1]))
    if cols:
        return cols
    if all(c in names for c in SLOT_COLUMNS):
        return list(SLOT_COLUMNS)
    raise AssertionError(f"{parquet_path} has no av_in_* or {SLOT_COLUMNS} AV-input slot columns")


# AR reconstruction targets — ALWAYS these three (the center-24 triplet
# L23/L24/L25), for EVERY condition. The AV-input slots (av_in_*) are a SEPARATE,
# condition-specific set; keeping the names distinct is what guarantees the AR
# target never silently changes when the AV input does.
SLOT_COLUMNS = ("activation_prev", "activation_centre", "activation_next")
AR_TARGET_COLUMNS = SLOT_COLUMNS

# Map an AR tap (center-relative absolute layer) to its fixed target column. The AR
# reconstruction target is always within {L23,L24,L25}; a single- or two-tap AR
# (the capacity-ablation experiment) MUST pull only the matching target columns, or
# three_target_loss silently broadcasts pred[B,k,d] against gold[B,3,d] and trains the
# k heads against the mean of all three targets. Keep these two maps in lockstep.
AR_LAYER_TO_TARGET_COL = {23: "activation_prev", 24: "activation_centre", 25: "activation_next"}
AR_TARGET_COL_TO_NAME = {"activation_prev": "prev", "activation_centre": "centre", "activation_next": "next"}


def av_user_content(k: int = N_SLOTS, placeholder: str = INJECT_PLACEHOLDER) -> str:
    if k == 1:
        tmpl = AV_USER_TEMPLATE_SINGLE
    elif k == 2:
        tmpl = AV_USER_TEMPLATE_DUAL
    elif k == N_SLOTS:
        tmpl = AV_USER_TEMPLATE
    else:
        raise ValueError(f"unsupported AV slot count k={k}; expected 1, 2, or {N_SLOTS}")
    return tmpl.format(m=placeholder)


def build_av_prompt(k: int = N_SLOTS, placeholder: str = INJECT_PLACEHOLDER) -> list:
    """The constant AV chat prompt (one user turn with k marker slots)."""
    return [{"role": "user", "content": av_user_content(k, placeholder)}]


def apply_chat_template_no_think(tokenizer, msgs, *, add_generation_prompt=True) -> str:
    """Chat-template the AV prompt with Qwen3 thinking DISABLED (enable_thinking=False).

    Qwen3 is a thinking model: with thinking on it emits <think>...</think> and burns
    the whole token budget reasoning *about* the prompt, never reaching <explanation>.
    The NLA AV is a verbalizer, not a reasoner — we want the answer directly. The
    kwarg is an unused template var on non-thinking archs (Llama/Gemma), so this is
    harmless there. MUST be used identically at AV-SFT train time and RL rollout
    time, or the actor sees a different prompt than it was trained on.
    """
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
        )


# ---- Document-level split (plan §3 invariant, point 2) ----

def doc_bucket(doc_id: str, fracs: tuple[float, ...], seed: int) -> int:
    """Route a doc_id to a bucket index by hashing (seed, doc_id) -> [0,1).

    All positions of a doc share its doc_id, so they all land in one bucket.
    """
    h = hashlib.sha256(f"{seed}|{doc_id}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / float(1 << 64)
    cum = 0.0
    for i, f in enumerate(fracs):
        cum += f
        if u < cum:
            return i
    return len(fracs) - 1


def split_by_document(base_parquet: str, out_dir: str, *,
                      fracs: tuple[float, ...] = (0.25, 0.25, 0.5),
                      names: tuple[str, ...] = ("av_sft", "ar_sft", "rl"),
                      seed: int = 42) -> dict[str, str]:
    """Stream the base multi-layer parquet, route each row to a bucket by doc_id,
    and write one parquet per bucket carrying EVERY column (all three activation
    columns + provenance). Returns {name: path}.

    Document-level (never splits a doc across buckets) and seed-deterministic.
    """
    from pathlib import Path
    assert len(fracs) == len(names), "fracs and names must align"
    assert abs(sum(fracs) - 1.0) < 1e-6, f"fracs must sum to 1, got {sum(fracs)}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(base_parquet)
    schema = pf.schema_arrow
    paths = {nm: str(out / f"{nm}.parquet") for nm in names}
    writers = {nm: pq.ParquetWriter(paths[nm], schema) for nm in names}
    counts = {nm: 0 for nm in names}
    try:
        for rg_idx in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg_idx)
            doc_ids = tbl.column("doc_id").to_pylist()
            buckets = np.fromiter((doc_bucket(d, fracs, seed) for d in doc_ids),
                                  dtype=np.int64, count=len(doc_ids))
            for bi, nm in enumerate(names):
                mask = pa.array(buckets == bi)
                sub = tbl.filter(mask)
                if sub.num_rows:
                    writers[nm].write_table(sub)
                    counts[nm] += sub.num_rows
    finally:
        for w in writers.values():
            w.close()
    print(f"[split] {base_parquet} -> " + ", ".join(f"{nm}:{counts[nm]}" for nm in names))
    return paths


# ---- Slot-vector stacking (the correctness-critical bit) ----

def stack_slot_vectors(rows: list, slot_cols) -> np.ndarray:
    """Stack each row's AV-input activations into [B*k, d] in injection scan order.

    For row b: [row[slot_cols[0]], ..., row[slot_cols[k-1]]] -> rows 0..k-1.
    Flatten is example-major ([b0 slots..., b1 slots..., ...]), matching
    inject_multislot_in_residual's row-major marker walk. `slot_cols` is the ordered
    av_in_* list (k=3 for local/duplicate/wide, k=1 for single).
    """
    per_row = np.stack(
        [np.stack([np.asarray(r[c], dtype=np.float32) for c in slot_cols])
         for r in rows]
    )  # [B, k, d]
    B, k, d = per_row.shape
    return per_row.reshape(B * k, d)


# ---- Loading + AV-SFT chunk prep ----

def load_av_sft_dataset(parquet_path: str, n_max: int | None = None,
                        slot_cols=None) -> list:
    """Load AV-SFT rows: prompt (list[msg]) + response (str) + k AV-input activations.

    The condition lives in the DATA: `slot_cols` (default: the av_in_* columns
    present) are this dataset's AV-input layers — [L23,L24,L25] for local, [L24]x3
    for duplicate, [L20,L24,L28] for wide, [L24] for single. No load-time transform.
    Activations via flatten->numpy; slices row-groups so n_max is exact.
    """
    if slot_cols is None:
        slot_cols = detect_av_slots(parquet_path)
    cols = ["prompt", "response", *slot_cols]
    pf = pq.ParquetFile(parquet_path)
    rows: list = []
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        take = rg.num_rows if n_max is None else min(n_max - len(rows), rg.num_rows)
        rg = rg.slice(0, take)
        prompts = rg.column("prompt").to_pylist()
        responses = rg.column("response").to_pylist()

        def to_np(name):
            col = rg.column(name).combine_chunks()
            return (col.flatten().to_numpy(zero_copy_only=False)
                    .astype(np.float32).reshape(len(col), -1))

        acts = {c: to_np(c) for c in slot_cols}
        for i in range(take):
            rows.append({
                "prompt": prompts[i],
                "response": responses[i],
                **{c: acts[c][i] for c in slot_cols},
            })
    return rows


def prepare_av_chunk_multi(rows: list, tokenizer, inject_char: str, inj_id: int,
                           device, *, max_len: int = 1024, slot_cols=None):
    """Build (input_ids, attn, loss_mask, vectors[B*k, d], prompt_lens[B]) for a multi-slot AV-SFT batch.

    `slot_cols` is the ordered av_in_* list for this dataset (k = len). prompt_lens
    lets the injection hook bound injection to the prompt span (the gold response is
    marker-free, so belt-and-suspenders here, but keeps the payload protocol == RL).

    - prompt: chat-templated, with INJECT_PLACEHOLDER x k swapped to inject_char.
    - per-example guard: assert exactly k marker tokens in the prompt (complements
      the hook's per-row check; catches template/tokenizer drift before the forward).
    - response: trailing EOS so the model learns to stop; loss_mask = 1 on response
      tokens only (CE is response-only).
    - vectors: stack_slot_vectors -> [B*k, d] in scan order.
    """
    if slot_cols is None:
        avs = sorted((c for c in rows[0] if c.startswith("av_in_")),
                     key=lambda s: int(s.rsplit("_", 1)[-1]))
        slot_cols = avs if avs else [c for c in SLOT_COLUMNS if c in rows[0]]
    k = len(slot_cols)
    full_ids_list, prompt_lens = [], []
    for row in rows:
        msgs = [
            {**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inject_char)}
            if isinstance(m.get("content"), str) else m
            for m in row["prompt"]
        ]
        prompt_str = apply_chat_template_no_think(tokenizer, msgs)
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        n_mark = sum(1 for t in prompt_ids if t == inj_id)
        assert n_mark == k, (
            f"AV prompt has {n_mark} marker tokens, expected k={k}. Template drift or "
            f"the marker char split into multiple tokens under this tokenizer."
        )
        resp = row["response"] + (tokenizer.eos_token or "")
        resp_ids = tokenizer.encode(resp, add_special_tokens=False)
        full = prompt_ids + resp_ids
        if len(full) > max_len:
            full = full[:max_len]  # truncate response from the right; prompt (markers) is preserved
        full_ids_list.append(torch.tensor(full, dtype=torch.long))
        prompt_lens.append(len(prompt_ids))

    bs = len(full_ids_list)
    T = max(t.numel() for t in full_ids_list)
    pad_id = tokenizer.eos_token_id
    batch_ids = torch.full((bs, T), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bs, T), dtype=torch.long, device=device)
    loss_mask = torch.zeros((bs, T), dtype=torch.float32, device=device)
    for i, t in enumerate(full_ids_list):
        L = t.numel()
        batch_ids[i, :L] = t.to(device)
        attn[i, :L] = 1
        loss_mask[i, prompt_lens[i]:L] = 1  # response-only (in target-token space; CE shift applied in loss)

    vectors = torch.tensor(stack_slot_vectors(rows, slot_cols), dtype=torch.float32, device=device)
    prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long, device=device)
    return batch_ids, attn, loss_mask, vectors, prompt_lens_t


# ---- AR-SFT: text z -> three activations ----
# Suffix-anchored critic prompt (plan §6.2): the AR reads ONLY the explanation,
# formatted into this template ending in a fixed suffix, and taps the LAST token.
# No marker / injection on the AR side.
#
# This string is IDENTICAL to nla.datagen.stage3_build._DEFAULT_CRITIC_TEMPLATE,
# which built the `ar_sft.prompt` column of the published warmstart dataset
# (ceselder/qwen3-8b-nla-L24-finefineweb-100k). Matching it byte-for-byte lets us
# reuse the published AR prompts verbatim — we only regenerate the activation
# triplet (see regenerate_multilayer.py / build_from_published.py). It is also the
# SINGLE source used by BOTH AR-SFT (build_datasets / build_from_published) and
# RL-time critic scoring (train_rl_multi via fill_ar_prompt), so the §6.2 invariant
# "AR-SFT critic input format == RL-time critic input format" holds by construction.
# The suffix after {explanation} — "</text> <summary>" — is unchanged, so the
# last-token suffix anchor is identical to the old (prefix-less) template.
AR_CRITIC_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"


def fill_ar_prompt(explanation: str) -> str:
    return AR_CRITIC_TEMPLATE.format(explanation=explanation)


def load_ar_sft_dataset(parquet_path: str, n_max: int | None = None) -> list:
    """Load AR-SFT rows: prompt (filled critic text, str) + the 3 fixed targets.

    Targets are AR_TARGET_COLUMNS (activation_prev/centre/next == L23/L24/L25) for
    EVERY condition — the AR reconstructs the same local state regardless of what the
    AV saw. No condition transform.
    """
    cols = ["prompt", *AR_TARGET_COLUMNS]
    pf = pq.ParquetFile(parquet_path)
    rows: list = []
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        take = rg.num_rows if n_max is None else min(n_max - len(rows), rg.num_rows)
        rg = rg.slice(0, take)
        prompts = rg.column("prompt").to_pylist()

        def to_np(name):
            col = rg.column(name).combine_chunks()
            return (col.flatten().to_numpy(zero_copy_only=False)
                    .astype(np.float32).reshape(len(col), -1))

        acts = {c: to_np(c) for c in AR_TARGET_COLUMNS}
        for i in range(take):
            rows.append({"prompt": prompts[i], **{c: acts[c][i] for c in AR_TARGET_COLUMNS}})
    return rows


def prepare_ar_chunk_multi(rows: list[dict], tokenizer, device, *, max_len: int = 1024,
                           target_cols=SLOT_COLUMNS):
    """(input_ids, attn, gold[B, len(target_cols), d]) for an AR-SFT batch.

    Critic prompt tokenized with add_special_tokens=False (matches RL-time
    scoring; True would prepend BOS on Llama/Gemma -> train/reward mismatch).
    Over-length rows are SKIPPED (right-truncation would cut the <summary>
    suffix and the last-token tap would land mid-text). gold is the RAW
    activations for `target_cols` stacked in order — normalized in the loss.

    `target_cols` MUST match the model's tap_layers (via AR_LAYER_TO_TARGET_COL),
    so a single-/two-tap AR pulls only its target column(s). Default = the full
    [prev, centre, next] triplet (the shipped 3-tap AR).
    """
    ids_list, kept = [], []
    n_skipped = 0
    for row in rows:
        ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        if len(ids) > max_len:
            n_skipped += 1
            continue
        ids_list.append(torch.tensor(ids, dtype=torch.long))
        kept.append(row)
    assert ids_list, f"all {len(rows)} AR rows exceeded max_len={max_len}"
    if n_skipped:
        print(f"[ar] skipped {n_skipped}/{len(rows)} rows over max_len (suffix anchor)")
    bs = len(ids_list)
    T = max(t.numel() for t in ids_list)
    pad = tokenizer.eos_token_id
    batch_ids = torch.full((bs, T), pad, dtype=torch.long, device=device)
    attn = torch.zeros((bs, T), dtype=torch.long, device=device)
    for i, t in enumerate(ids_list):
        batch_ids[i, :t.numel()] = t.to(device)
        attn[i, :t.numel()] = 1
    gold = torch.tensor(
        np.stack([[np.asarray(r[c], dtype=np.float32) for c in target_cols] for r in kept]),
        dtype=torch.float32, device=device,
    )  # [B, len(target_cols), d]
    return batch_ids, attn, gold
