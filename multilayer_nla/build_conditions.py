"""Build permutation-grid condition datasets (layers × token positions) from the
unified bank — the generalization of the parent repo's build_sweep.py.

Every condition is an ordered list of (layer, position-offset) SLOTS
(conditions.py grammar: `L24@0`, `L23@-2`, flags `|pool` / `|shufctx`). The AV
input = the condition's slots, materialized as positional `av_in_*` columns;
the AR reconstruction target is ALWAYS the fixed --ar-target-layers triplet at
the labeled position p, for EVERY condition. Slot sources:

    L{k}@0    <- activation_L{k}                       (final-token archive)
    L{k}@-j   <- window_L{k}[:, W-1-j, :]              (position p-j)

Outputs (into --out-dir; same shape as the parent sweep so the § 7 muscle
memory carries over):

  ar_common.parquet / ar_dev.parquet / ar_test.parquet   shared AR train + gold eval
      prompt(canonical critic) + activation_prev/centre/next + doc_id
  av_<cond>.parquet                                      AV-SFT train, per condition
      prompt(k-marker, STORED) + response + av_in_* + doc_id + src_row_id
  rl_dev_<cond>.parquet / rl_test_<cond>.parquet         end-to-end eval, per condition
      prompt(STORED) + av_in_* + activation_prev/centre/next + doc_id + src_row_id

The stored `prompt` column is what evaluate_e2e / train_rl_multi generate with
(datasets.load_stored_prompt), so ANY k works end to end. Template: `neutral`
(default; one uniform wording for every k — conditions at the same k differ
ONLY in the injected vectors) or `legacy` (the §7 depth wording, k<=3 only,
for reproducing old artifacts; absolute FVE is not comparable across templates).

`|shufctx` conditions are EVAL-ONLY controls (refused for AV training): all
slots except the LAST are doc-deranged across rows — evaluate them with the
PARENT condition's AV checkpoint; window ≈ shufctx means the context slots
aren't being used as this-document context.

Preflight (assert_conditions) enforces, before any GPU minute is spent:
row identity across all conditions, prompt identity within each k, slot
equal-iff-same-(layer,offset), absolute slot identity vs the in-file targets,
byte-identical fixed targets across conditions, bucket + corpus disjointness,
shufctx derangement, and (with --base-ckpt) rendered marker counts.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.schema import EXPLANATION_OPEN, wrap_explanation
from multilayer_nla.conditions import (
    Condition,
    parse_conditions,
    validate_against_bank,
)
from multilayer_nla.datasets import (
    AR_TARGET_COLUMNS,
    av_in_columns,
    build_av_prompt,
    build_av_prompt_neutral,
    doc_bucket,
)
from multilayer_nla.build_from_published import (
    EXPLANATION_COL,
    PROMPT_COL,
    RESPONSE_COL,
    _AR_PREFIX,
    _AR_SUFFIX,
    _subset_inputs,
)

DEFAULT_AR_TARGET_LAYERS = [23, 24, 25]
# assert_conditions reads at most this many rows of the fat vector columns per
# file (row identity / prompt / doc checks still scan full string columns).
PREFLIGHT_VEC_SAMPLE = 2048


def _layer_col(L: int) -> str:
    return f"activation_L{L}"


def _window_col(L: int) -> str:
    return f"window_L{L}"


# ------------------------------------------------------------------ bank probe

def probe_bank(bank_path: str) -> dict:
    """Bank geometry from the SCHEMA alone: d_model, activation layers, window
    layers, window size. No sidecar dependency (works on any shard)."""
    sch = pq.ParquetFile(bank_path).schema_arrow
    act, win = {}, {}
    for f in sch:
        if f.name.startswith("activation_L") and pa.types.is_fixed_size_list(f.type):
            act[int(f.name[len("activation_L"):])] = f.type.list_size
        elif f.name.startswith("window_L") and pa.types.is_fixed_size_list(f.type):
            win[int(f.name[len("window_L"):])] = f.type.list_size
    assert act, f"{bank_path}: no activation_L* columns — not a bank parquet"
    d = set(act.values())
    assert len(d) == 1, f"{bank_path}: inconsistent activation widths {d}"
    d_model = d.pop()
    W = 0
    if win:
        ws = {size // d_model for size in win.values()}
        assert len(ws) == 1 and all(size % d_model == 0 for size in win.values()), (
            f"{bank_path}: window column sizes {win} not a clean multiple of d={d_model}"
        )
        W = ws.pop()
    return {"d_model": d_model, "activation_layers": set(act),
            "window_layers": set(win), "window_size": W}


# ------------------------------------------------------------------ slot pours

def _col_mat(t: pa.Table, name: str, d_last: int) -> np.ndarray:
    """FixedSizeList column -> [n, ...] float32 with trailing dim d_last."""
    c = t.column(name).combine_chunks()
    flat = c.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    per = flat.size // len(c) if len(c) else d_last
    return flat.reshape(len(c), per // d_last, d_last) if per != d_last else flat.reshape(len(c), d_last)


def slot_arrays(t: pa.Table, cond: Condition, d: int, W: int) -> list:
    """Materialize the condition's slots from a bank batch -> list of [n, d]
    float32 arrays in slot order. Reshapes each needed window column ONCE and
    slices every offset from it. Pure numpy — unit-testable offline."""
    win_cache: dict = {}
    out = []
    for layer, off in cond.slots:
        if off == 0:
            out.append(_col_mat(t, _layer_col(layer), d))
        else:
            if layer not in win_cache:
                w = _col_mat(t, _window_col(layer), d)          # [n, W, d]
                assert w.ndim == 3 and w.shape[1] == W, (
                    f"window_L{layer}: got {w.shape}, expected (n, {W}, {d})"
                )
                win_cache[layer] = w
            out.append(np.ascontiguousarray(win_cache[layer][:, W - 1 + off, :]))
    return out


def _fsl(mat: np.ndarray, d: int) -> pa.Array:
    flat = np.ascontiguousarray(mat.astype(np.float32, copy=False)).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat), d)


def cond_prompt(cond: Condition, template: str) -> list:
    if template == "neutral":
        return build_av_prompt_neutral(cond.k)
    assert template == "legacy", f"unknown template {template!r}"
    return build_av_prompt(cond.k)   # raises for k not in {1,2,3} — by design


def _need_cols(bank_path: str, cond: Condition, target_layers) -> list:
    """Projection: exactly the bank columns this build needs."""
    cols = ["doc_id"]
    for layer, off in cond.slots:
        cols.append(_layer_col(layer) if off == 0 else _window_col(layer))
    for L in target_layers:
        cols.append(_layer_col(L))
    return list(dict.fromkeys(cols))


def _bucket_keep(doc_ids, bucket_idx, fracs, seed, subset=None):
    return np.fromiter(
        ((doc_bucket(d_, fracs, seed) == bucket_idx) and (subset is None or d_ in subset)
         for d_ in doc_ids),
        dtype=bool, count=len(doc_ids),
    )


# ------------------------------------------------------------------ builders

def build_av(bank_paths, out_path, cond: Condition, *, template="neutral",
             d=None, W=None, batch_size=2048, with_targets=None) -> int:
    """av_<cond>.parquet: stored k-marker prompt + response label + av_in_* +
    doc_id + src_row_id. AV trains on ALL av_sft bank rows (selection happens on
    the rl dev split downstream). src_row_id = global ordinal over the bank
    stream, identical across conditions (the row-identity invariant).

    with_targets: also write the fixed activation_prev/centre/next target
    columns (a list of target layers). AV-SFT itself never reads them; they are
    what distill_av needs to SCORE responses against the fixed target (gold
    scoring / best-of-N self-distillation). Costs 3*d*4 bytes/row."""
    assert not cond.shufctx, (
        f"{cond.name}: shufctx is an EVAL-ONLY control — never train an AV on "
        f"deranged context (build rl_dev/rl_test variants instead)."
    )
    geo = probe_bank(bank_paths[0])
    d = d or geo["d_model"]
    W = W or geo["window_size"]
    validate_against_bank(cond, activation_layers=geo["activation_layers"],
                          window_layers=geo["window_layers"], window_size=geo["window_size"])
    if with_targets:
        missing = [L for L in with_targets if L not in geo["activation_layers"]]
        if missing:
            raise SystemExit(f"av --with-targets: bank lacks activation_L{missing}")
    schema_names = pq.ParquetFile(bank_paths[0]).schema_arrow.names
    has_resp, has_expl = RESPONSE_COL in schema_names, EXPLANATION_COL in schema_names
    if not (has_resp or has_expl):
        raise SystemExit(f"av bank needs a {RESPONSE_COL!r} or {EXPLANATION_COL!r} label column")
    label_col = RESPONSE_COL if has_resp else EXPLANATION_COL
    proj = list(dict.fromkeys(_need_cols(bank_paths[0], cond, with_targets or []) + [label_col]))
    prompt = cond_prompt(cond, template)
    slot_cols = av_in_columns(cond.k)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    for p in bank_paths:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=batch_size, columns=proj):
            t = pa.Table.from_batches([batch])
            m = t.num_rows
            raw = t.column(label_col).to_pylist()
            resp = raw if has_resp else [wrap_explanation(e) for e in raw]
            bad = [i for i, r in enumerate(resp) if not r or EXPLANATION_OPEN not in r]
            assert not bad, (f"av: {len(bad)} rows have an empty/unwrapped response "
                             f"(first idx {bad[0]}); expected the published <explanation> label")
            slots = slot_arrays(t, cond, d, W)
            cols = {
                "prompt": pa.array([prompt] * m),
                "response": pa.array(resp, pa.string()),
                "doc_id": t.column("doc_id"),
                "src_row_id": pa.array(list(range(n, n + m)), pa.int64()),
            }
            if cond.pool:
                cols["av_in_0"] = _fsl(np.mean(slots, axis=0), d)
            else:
                for sc, arr in zip(slot_cols, slots):
                    cols[sc] = _fsl(arr, d)
            if with_targets:
                for tc, L in zip(AR_TARGET_COLUMNS, with_targets):
                    cols[tc] = t.column(_layer_col(L))
            out = pa.table(cols)
            if writer is None:
                writer = pq.ParquetWriter(out_path, out.schema)
            writer.write_table(out)
            n += m
    if writer is not None:
        writer.close()
    print(f"[cond:av] {Path(out_path).name}  ({n} rows, {cond.describe()}, k={cond.k}, "
          f"template={template})")
    return n


def build_ar(bank_paths, out_path, target_layers, *, bucket_idx=None, fracs=None,
             seed=42, batch_size=2048) -> int:
    """ar_<bucket>.parquet: canonical critic prompt (verbatim) + fixed targets +
    doc_id. Condition-independent — built once, shared by every condition."""
    assert len(target_layers) == len(AR_TARGET_COLUMNS), "AR needs exactly 3 target layers"
    geo = probe_bank(bank_paths[0])
    missing = [L for L in target_layers if L not in geo["activation_layers"]]
    if missing:
        raise SystemExit(f"ar: bank lacks activation_L{missing} — widen --save-layers")
    schema_names = pq.ParquetFile(bank_paths[0]).schema_arrow.names
    if PROMPT_COL not in schema_names:
        raise SystemExit(f"ar bank needs the published critic {PROMPT_COL!r} column")
    proj = list(dict.fromkeys([_layer_col(L) for L in target_layers] + ["doc_id", PROMPT_COL]))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    for p in bank_paths:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=batch_size, columns=proj):
            t = pa.Table.from_batches([batch])
            prompts = t.column(PROMPT_COL).to_pylist()
            bad = [i for i, pr in enumerate(prompts)
                   if not (pr and pr.startswith(_AR_PREFIX) and pr.endswith(_AR_SUFFIX))]
            assert not bad, (f"ar: {len(bad)} prompts are NOT the canonical critic template "
                             f"(first idx {bad[0]}: {prompts[bad[0]]!r})")
            cols = {"prompt": pa.array(prompts, pa.string()), "doc_id": t.column("doc_id")}
            for tc, L in zip(AR_TARGET_COLUMNS, target_layers):
                cols[tc] = t.column(_layer_col(L))
            out = pa.table(cols)
            if bucket_idx is not None:
                keep = _bucket_keep(out.column("doc_id").to_pylist(), bucket_idx, fracs, seed)
                out = out.filter(pa.array(keep))
            if out.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(out_path, out.schema)
                writer.write_table(out)
                n += out.num_rows
    if writer is not None:
        writer.close()
    print(f"[cond:ar] {Path(out_path).name}  ({n} rows, targets={target_layers}, bucket={bucket_idx})")
    return n


def build_rl_eval(bank_paths, out_path, cond: Condition, target_layers, bucket_idx,
                  fracs, seed, *, template="neutral", subset=None, batch_size=2048,
                  max_rows_in_memory=200_000) -> int:
    """rl_<bucket>_<cond>.parquet: stored prompt + av_in_* + fixed targets +
    doc_id + src_row_id, filtered to bucket_idx (dev/test, optional locked
    subset). shufctx conditions doc-derange the non-final slots ACROSS the kept
    rows (deterministic in `seed`), keeping the final slot true."""
    geo = probe_bank(bank_paths[0])
    d, W = geo["d_model"], geo["window_size"]
    validate_against_bank(cond, activation_layers=geo["activation_layers"],
                          window_layers=geo["window_layers"], window_size=geo["window_size"])
    missing = [L for L in target_layers if L not in geo["activation_layers"]]
    if missing:
        raise SystemExit(f"rl-eval: bank lacks activation_L{missing} — widen --save-layers")
    proj = _need_cols(bank_paths[0], cond, target_layers)
    prompt = cond_prompt(cond, template)
    slot_cols = av_in_columns(cond.k)

    # Collect kept rows in memory (eval sets are capped by design — dev/test
    # subsets are a few thousand rows), because shufctx needs a global
    # derangement over the final row set, not per-batch.
    kept_slots, kept_tgts, kept_docs, kept_src = [], [], [], []
    seen = 0
    for p in bank_paths:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=batch_size, columns=proj):
            t = pa.Table.from_batches([batch])
            m = t.num_rows
            doc_ids = t.column("doc_id").to_pylist()
            keep = _bucket_keep(doc_ids, bucket_idx, fracs, seed, subset)
            if keep.any():
                idx = np.nonzero(keep)[0]
                tk = t.take(pa.array(idx))
                slots = slot_arrays(tk, cond, d, W)          # list of [nk, d]
                kept_slots.append(np.stack(slots, axis=1))   # [nk, n_slots, d]
                kept_tgts.append(np.stack(
                    [_col_mat(tk, _layer_col(L), d) for L in target_layers], axis=1))
                kept_docs.extend(doc_ids[i] for i in idx)
                kept_src.extend(int(seen + i) for i in idx)
            seen += m
    n = len(kept_docs)
    assert n <= max_rows_in_memory, (
        f"rl-eval {cond.name}: {n} kept rows exceeds the in-memory cap "
        f"{max_rows_in_memory} — use a locked dev/test subset (splits.py) or raise the cap."
    )
    if n == 0:
        print(f"[cond:rl-eval] {Path(out_path).name}: 0 rows kept (empty bucket?) — nothing written")
        return 0
    S = np.concatenate(kept_slots, axis=0)     # [n, n_slots, d]
    G = np.concatenate(kept_tgts, axis=0)      # [n, 3, d]

    if cond.shufctx:
        from multilayer_nla.evaluate_e2e import _doc_derangement
        perm = _doc_derangement(kept_docs, seed + 7)
        # non-final slots come from ANOTHER document's row; final slot stays true.
        S = S.copy()
        S[:, :-1, :] = S[np.asarray(perm), :-1, :]

    cols = {
        "prompt": pa.array([prompt] * n),
        "doc_id": pa.array(kept_docs, pa.string()),
        "src_row_id": pa.array(kept_src, pa.int64()),
    }
    if cond.pool:
        cols["av_in_0"] = _fsl(S.mean(axis=1), d)
    else:
        for i, sc in enumerate(slot_cols):
            cols[sc] = _fsl(S[:, i, :], d)
    for j, tc in enumerate(AR_TARGET_COLUMNS):
        cols[tc] = _fsl(G[:, j, :], d)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), out_path, row_group_size=4096)
    print(f"[cond:rl-eval] {Path(out_path).name}  ({n} rows, {cond.describe()}, "
          f"targets={target_layers}, bucket={bucket_idx})")
    return n


# ------------------------------------------------------------------ preflight

def _vec_sample(path, name, n=PREFLIGHT_VEC_SAMPLE):
    """First n rows of a FixedSizeList column as [n, d] float32 (bounded memory)."""
    pf = pq.ParquetFile(path)
    got = next(pf.iter_batches(batch_size=n, columns=[name])).column(name)
    # RecordBatch columns are plain Arrays (no combine_chunks, unlike Table columns).
    flat = got.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    return flat.reshape(len(got), -1)


def assert_conditions(out_dir: Path, conds: dict, *, target_layers=None) -> dict:
    """Preflight invariants over built condition datasets. Raises on violation.

    Vector checks run on the first PREFLIGHT_VEC_SAMPLE rows (mis-pours are
    systematic — every row is wrong — so a prefix sample is decisive); string /
    id / doc checks scan full columns.
    """
    out_dir = Path(out_dir)
    target_layers = target_layers or DEFAULT_AR_TARGET_LAYERS
    tgt_of = dict(zip(target_layers, AR_TARGET_COLUMNS))
    report = {}
    train_conds = {c: v for c, v in conds.items() if not v.shufctx}

    # 1) row identity across ALL train conditions (same bank stream => same rows).
    base_key = None
    for c in train_conds:
        p = out_dir / f"av_{c}.parquet"
        t = pq.read_table(p, columns=["doc_id", "src_row_id"])
        key = (t.column("doc_id").to_pylist(), t.column("src_row_id").to_pylist())
        if base_key is None:
            base_key = key
        else:
            assert key == base_key, f"av_{c}: source rows differ across conditions"
    if base_key:
        report["av_rows"] = len(base_key[0])

    # 2) prompt identity within each k (the same-k comparison contract).
    by_k: dict = {}
    for c, v in train_conds.items():
        p0 = pq.read_table(out_dir / f"av_{c}.parquet", columns=["prompt"]).column("prompt")[0].as_py()
        by_k.setdefault(v.k, []).append((c, p0))
    for k, entries in by_k.items():
        c0, ref = entries[0]
        for c, p0 in entries[1:]:
            assert p0 == ref, f"av_{c}: prompt differs from av_{c0} at the same k={k}"

    # 3) slot content: within a condition, slots byte-equal iff same (layer, offset);
    #    absolute identity vs in-file targets on the rl_dev files.
    for c, v in train_conds.items():
        if v.pool:
            continue
        p = out_dir / f"av_{c}.parquet"
        slots = {i: _vec_sample(p, f"av_in_{i}") for i in range(v.k)}
        for i in range(v.k):
            for j in range(i + 1, v.k):
                same = np.array_equal(slots[i], slots[j])
                if v.slots[i] == v.slots[j]:
                    assert same, f"av_{c}: slots {i},{j} share {v.slots[i]} but differ"
                else:
                    assert not same, f"av_{c}: slots {i},{j} are {v.slots[i]}/{v.slots[j]} but equal"
    for c, v in conds.items():
        p = out_dir / f"rl_dev_{c}.parquet"
        if not p.exists() or v.pool:
            continue
        last_true_only = v.shufctx  # only the final slot is guaranteed true
        for i, (L, off) in enumerate(v.slots):
            if last_true_only and i != v.k - 1:
                continue
            if off == 0 and L in tgt_of:
                assert np.array_equal(_vec_sample(p, f"av_in_{i}"), _vec_sample(p, tgt_of[L])), (
                    f"rl_dev_{c}: av_in_{i} (L{L}@0) != target {tgt_of[L]} — wrong pour")
            elif off != 0 and L in tgt_of:
                assert not np.array_equal(_vec_sample(p, f"av_in_{i}"), _vec_sample(p, tgt_of[L])), (
                    f"rl_dev_{c}: av_in_{i} (L{L}@{off}) byte-equals the position-p target — "
                    f"the window slice landed on the wrong position")

    # 4) fixed targets byte-identical + same source rows across conditions, per bucket.
    for bucket in ("dev", "test"):
        base_rows = base_tgt = ref_name = None
        for c in conds:
            p = out_dir / f"rl_{bucket}_{c}.parquet"
            if not p.exists():
                continue
            ids = pq.read_table(p, columns=["doc_id", "src_row_id"])
            rows = (ids.column("doc_id").to_pylist(), ids.column("src_row_id").to_pylist())
            tgt = {tc: _vec_sample(p, tc) for tc in AR_TARGET_COLUMNS}
            if base_rows is None:
                base_rows, base_tgt, ref_name = rows, tgt, c
            else:
                assert rows == base_rows, f"rl_{bucket}_{c}: source rows differ from rl_{bucket}_{ref_name}"
                for tc in AR_TARGET_COLUMNS:
                    assert np.array_equal(tgt[tc], base_tgt[tc]), (
                        f"rl_{bucket}_{c}: target {tc} not identical to rl_{bucket}_{ref_name} "
                        f"(the target must be FIXED across conditions)")
        if base_rows is not None:
            report[f"rl_{bucket}_rows"] = len(base_rows[0])

    # 5) shufctx: non-final slots must actually be deranged vs the parent condition.
    for c, v in conds.items():
        if not v.shufctx:
            continue
        parent = next((pc for pc, pv in train_conds.items()
                       if pv.slots == v.slots and not pv.pool), None)
        p = out_dir / f"rl_dev_{c}.parquet"
        if parent is None or not p.exists():
            continue
        pp = out_dir / f"rl_dev_{parent}.parquet"
        a_last = _vec_sample(p, f"av_in_{v.k - 1}")
        b_last = _vec_sample(pp, f"av_in_{v.k - 1}")
        assert np.array_equal(a_last, b_last), (
            f"rl_dev_{c}: final slot differs from rl_dev_{parent} — shufctx must keep it true")
        a0, b0 = _vec_sample(p, "av_in_0"), _vec_sample(pp, "av_in_0")
        frac_same = float(np.mean(np.all(a0 == b0, axis=1)))
        assert frac_same < 0.01, (
            f"rl_dev_{c}: {frac_same:.1%} of context slots identical to the parent — "
            f"derangement did not apply")

    # 6) split + corpus disjointness.
    def _docs(name):
        pth = out_dir / name
        return (set(pq.read_table(pth, columns=["doc_id"]).column("doc_id").to_pylist())
                if pth.exists() else set())
    any_cond = next(iter(train_conds))
    dev_docs, test_docs = _docs(f"rl_dev_{any_cond}.parquet"), _docs(f"rl_test_{any_cond}.parquet")
    inter = dev_docs & test_docs
    assert not inter, f"rl_dev and rl_test share {len(inter)} docs"
    corpora = {"av": _docs(f"av_{any_cond}.parquet"),
               "ar": _docs("ar_common.parquet") | _docs("ar_dev.parquet") | _docs("ar_test.parquet"),
               "rl": dev_docs | test_docs}
    cn = list(corpora)
    for x in range(len(cn)):
        for y in range(x + 1, len(cn)):
            inter = corpora[cn[x]] & corpora[cn[y]]
            assert not inter, f"{cn[x]} and {cn[y]} corpora share {len(inter)} docs (split leak)"

    print(f"[cond:preflight] OK  {report}")
    return report


def assert_marker_counts(out_dir, conds: dict, base_ckpt) -> None:
    """Render each condition's stored AV prompt through the REAL tokenizer +
    chat template; assert exactly k markers (the runtime invariant, lifted to
    preflight so template/tokenizer drift dies before training)."""
    from transformers import AutoTokenizer
    from nla.datagen.injection_tokens import find_injection_token
    from nla.schema import INJECT_PLACEHOLDER
    from multilayer_nla.datasets import apply_chat_template_no_think
    tok = AutoTokenizer.from_pretrained(base_ckpt)
    inject_char, inj_id = find_injection_token(tok)
    for c, v in conds.items():
        p = Path(out_dir) / (f"av_{c}.parquet" if not v.shufctx else f"rl_dev_{c}.parquet")
        if not p.exists():
            continue
        msgs = pq.read_table(p, columns=["prompt"]).column("prompt")[0].as_py()
        msgs = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inject_char)} for m in msgs]
        ids = tok.encode(apply_chat_template_no_think(tok, msgs), add_special_tokens=False)
        n_mark = sum(1 for t in ids if t == inj_id)
        assert n_mark == v.k, (
            f"{c}: rendered prompt has {n_mark} markers, expected k={v.k} "
            f"(template/tokenizer drift would break injection)")
    print("[cond:preflight] marker counts OK (rendered k matches each condition)")


# ------------------------------------------------------------------ orchestration

def _bucket_idx(names, bucket):
    assert bucket in names, f"bucket {bucket!r} not in split names {names}"
    return list(names).index(bucket)


def _read_split(manifest_path):
    m = json.loads(Path(manifest_path).read_text())
    return (m["seed"], tuple(m["fracs"]), tuple(m["names"]),
            {k: set(v) for k, v in m.get("locked_subsets", {}).items()})


def build_all(in_dir, out_dir, conds: dict, rl_manifest, ar_manifest, *,
              ar_target_layers=None, template="neutral", batch_size=2048,
              allow_existing=False, marker_check_ckpt=None, av_with_targets=False):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ar_target_layers = ar_target_layers or DEFAULT_AR_TARGET_LAYERS
    existing = list(out_dir.glob("*.parquet"))
    if existing and not allow_existing:
        raise SystemExit(f"--out-dir {out_dir} already holds {len(existing)} parquet(s); "
                         f"use a fresh dir or --allow-existing.")

    rl_seed, rl_fracs, rl_names, rl_sub = _read_split(rl_manifest)
    ar_seed, ar_fracs, ar_names, _ = _read_split(ar_manifest)
    av_bank = _subset_inputs(in_dir, "av_sft")
    ar_bank = _subset_inputs(in_dir, "ar_sft")
    rl_bank = _subset_inputs(in_dir, "rl")
    for nm, b in (("av_sft", av_bank), ("ar_sft", ar_bank), ("rl", rl_bank)):
        assert b, f"no {nm} bank parquet/shards in {in_dir}"

    counts = {"ar": {}, "av": {}, "rl_dev": {}, "rl_test": {}}
    counts["ar"]["common"] = build_ar(ar_bank, str(out_dir / "ar_common.parquet"), ar_target_layers,
                                      bucket_idx=_bucket_idx(ar_names, "train"), fracs=ar_fracs,
                                      seed=ar_seed, batch_size=batch_size)
    for b in ("dev", "test"):
        counts["ar"][b] = build_ar(ar_bank, str(out_dir / f"ar_{b}.parquet"), ar_target_layers,
                                   bucket_idx=_bucket_idx(ar_names, b), fracs=ar_fracs,
                                   seed=ar_seed, batch_size=batch_size)

    for c, v in conds.items():
        if not v.shufctx:
            counts["av"][c] = build_av(av_bank, str(out_dir / f"av_{c}.parquet"), v,
                                       template=template, batch_size=batch_size,
                                       with_targets=(ar_target_layers if av_with_targets else None))
        for b in ("dev", "test"):
            counts[f"rl_{b}"][c] = build_rl_eval(
                rl_bank, str(out_dir / f"rl_{b}_{c}.parquet"), v, ar_target_layers,
                _bucket_idx(rl_names, b), rl_fracs, rl_seed,
                template=template, subset=rl_sub.get(b), batch_size=batch_size)

    report = assert_conditions(out_dir, conds, target_layers=ar_target_layers)
    if marker_check_ckpt:
        assert_marker_counts(out_dir, conds, marker_check_ckpt)
    (out_dir / "conditions_build_manifest.json").write_text(json.dumps({
        "in_dir": str(in_dir),
        "conditions": {c: v.describe() for c, v in conds.items()},
        "template": template,
        "ar_target_layers": ar_target_layers,
        "rl_split_manifest": str(rl_manifest),
        "ar_split_manifest": str(ar_manifest),
        "counts": counts,
        "preflight": report,
        "note": "AR target fixed for every condition; the condition lives in av_in_* "
                "(layers x positions) + the stored prompt. shufctx = eval-only control.",
    }, indent=2))
    print(f"[cond] all datasets -> {out_dir}")
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["all", "av", "rl-eval", "ar", "preflight"], required=True)
    p.add_argument("--conditions", help="';'-joined condition specs, e.g. "
                                        "'single=L24@0; tok3=L24@-2,L24@-1,L24@0' "
                                        "(see conditions.py; PERMUTATION_GRID is the pre-registered set)")
    p.add_argument("--conditions-preset", choices=["permutation_grid", "legacy_sweep"],
                   help="named preset instead of --conditions")
    p.add_argument("--in", dest="inp", help="bank shard glob (single-mode)")
    p.add_argument("--in-dir", help="bank dir with av_sft/ar_sft/rl parquets or shards (--mode all)")
    p.add_argument("--out", help="output parquet (single-mode)")
    p.add_argument("--out-dir", help="output dir (--mode all / preflight)")
    p.add_argument("--condition", help="which condition (single-mode av / rl-eval)")
    p.add_argument("--ar-target-layers", default="23,24,25",
                   help="fixed AR target layers (position p) — identical for every condition")
    p.add_argument("--template", choices=["neutral", "legacy"], default="neutral",
                   help="AV prompt family. neutral = uniform any-k wording (default for the "
                        "permutation grid); legacy = §7 depth wording (k<=3, old-artifact reproduction)")
    p.add_argument("--bucket", help="split bucket (train|dev|test) for single-mode ar / rl-eval")
    p.add_argument("--rl-split-manifest", help="splits.py manifest for the rl bank")
    p.add_argument("--ar-split-manifest", help="splits.py manifest for the ar bank")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--allow-existing", action="store_true")
    p.add_argument("--av-with-targets", action="store_true",
                   help="also write the fixed target columns into av_<cond>.parquet — needed "
                        "by distill_av (gold scoring / best-of-N); AV-SFT ignores them")
    p.add_argument("--base-ckpt", default=None,
                   help="if set, preflight renders each stored prompt through this tokenizer "
                        "and asserts the marker count == k")
    args = p.parse_args()

    if args.conditions_preset:
        assert not args.conditions, "pass --conditions or --conditions-preset, not both"
        from multilayer_nla.conditions import LEGACY_SWEEP, PERMUTATION_GRID
        args.conditions = (PERMUTATION_GRID if args.conditions_preset == "permutation_grid"
                           else "; ".join(LEGACY_SWEEP.values()))
    conds = parse_conditions(args.conditions) if args.conditions else None

    def _layers(s):
        return [int(x) for x in s.split(",")]

    if args.mode == "all":
        assert conds and args.in_dir and args.out_dir and args.rl_split_manifest and args.ar_split_manifest, \
            "--mode all needs --conditions[-preset] --in-dir --out-dir --rl-split-manifest --ar-split-manifest"
        build_all(args.in_dir, args.out_dir, conds, args.rl_split_manifest, args.ar_split_manifest,
                  ar_target_layers=_layers(args.ar_target_layers), template=args.template,
                  batch_size=args.batch_size, allow_existing=args.allow_existing,
                  marker_check_ckpt=args.base_ckpt, av_with_targets=args.av_with_targets)
        return

    if args.mode == "preflight":
        assert conds and args.out_dir, "--mode preflight needs --conditions[-preset] and --out-dir"
        assert_conditions(Path(args.out_dir), conds, target_layers=_layers(args.ar_target_layers))
        if args.base_ckpt:
            assert_marker_counts(args.out_dir, conds, args.base_ckpt)
        print("[cond:preflight] PASSED")
        return

    import glob as _glob
    assert args.inp and args.out, f"--mode {args.mode} needs --in and --out"
    ins = sorted(_glob.glob(args.inp)) or [args.inp]
    if args.mode == "ar":
        seed, fracs, names, _ = (_read_split(args.ar_split_manifest) if args.ar_split_manifest
                                 else (42, (0.8, 0.1, 0.1), ("train", "dev", "test"), {}))
        bidx = _bucket_idx(names, args.bucket) if args.bucket else None
        build_ar(ins, args.out, _layers(args.ar_target_layers), bucket_idx=bidx,
                 fracs=fracs, seed=seed, batch_size=args.batch_size)
        return
    assert conds and args.condition and args.condition in conds, \
        f"--mode {args.mode} needs --conditions[-preset] and a --condition from them"
    v = conds[args.condition]
    if args.mode == "av":
        build_av(ins, args.out, v, template=args.template, batch_size=args.batch_size)
    else:
        assert args.bucket and args.rl_split_manifest, "--mode rl-eval needs --bucket + --rl-split-manifest"
        seed, fracs, names, sub = _read_split(args.rl_split_manifest)
        build_rl_eval(ins, args.out, v, _layers(args.ar_target_layers),
                      _bucket_idx(names, args.bucket), fracs, seed, template=args.template,
                      subset=sub.get(args.bucket), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
