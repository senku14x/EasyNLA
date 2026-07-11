"""Adapt regenerated published rows into our three-slot training parquets.

Pipeline (NO API, NO paid labeling — see regenerate_multilayer_activations.py):

    HF: ceselder/qwen3-8b-nla-L24-finefineweb-100k   (text + LABELS, no vectors)
        av_sft / ar_sft / rl
            |  regenerate_multilayer_activations.py   (+ activation_L{k} window)
            v
        regenerated av_sft / ar_sft / rl              (labels + layer-window archive)
            |  build_from_published.py  (THIS)        (--center selects the triplet)
            v
        training parquets the multilayer_nla trainers load

`--center c` selects activation_L{c-1,c,c+1} from the archive and renames them to
activation_prev/centre/next (what the trainers read). Re-probing a different
center later is just a re-run of THIS step on the same archive — no re-extraction.
A legacy parquet that already has activation_prev/centre/next is accepted as-is.

What this step changes vs. the published row, per subset:

  av : REPLACE the published single-marker actor prompt with our three-marker
       prompt (build_av_prompt()); KEEP the published `response` (the real
       <explanation>...</explanation> label) verbatim. + 3 activations.
  ar : KEEP the published `prompt` verbatim — it is the canonical critic template
       (`Summary of the following text: <text>{explanation}</text> <summary>`),
       byte-identical to our AR_CRITIC_TEMPLATE, so AR-SFT == RL-time scoring.
       + 3 activations.
  rl : REPLACE the prompt with our three-marker prompt; no response. + 3 acts.

The document-level av/ar/rl split is INHERITED from the published dataset (the
same split the released checkpoints used) — we do not re-split. The §7 condition
flags (coherent / duplicate / single / mismatched) act on these identical labeled
rows at train/inject time, not here.

Trainers auto-pick the marker from the tokenizer and use the per-row count guard,
so these parquets need NO sidecar.
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nla.schema import EXPLANATION_OPEN, wrap_explanation
from multilayer_nla.datasets import (
    AR_CRITIC_TEMPLATE,
    SLOT_COLUMNS,
    build_av_prompt,
    fill_ar_prompt,
)

EXPLANATION_COL = "api_explanation"
RESPONSE_COL = "response"
PROMPT_COL = "prompt"
# The canonical critic template split around {explanation}: a published AR prompt
# is valid iff it == AR_CRITIC_TEMPLATE.format(explanation=X) for some X, i.e. it
# starts with this prefix and ends with this suffix (so AR-SFT == RL-time scoring).
_AR_PREFIX, _AR_SUFFIX = AR_CRITIC_TEMPLATE.split("{explanation}")


def _layer_col(k: int) -> str:
    return f"activation_L{k}"


def _triplet_names(schema_names, center: int) -> list:
    """Source column names for the `center` triplet, from the SCHEMA (no data load).

    Prefers the archive (activation_L{c-1,c,c+1}); falls back to a legacy
    prev/centre/next parquet. Raises if neither is present (e.g. --save-layers
    did not cover this center). Single source of truth for triplet precedence.
    """
    arch = [_layer_col(center - 1), _layer_col(center), _layer_col(center + 1)]
    if all(c in schema_names for c in arch):
        return arch
    if all(c in schema_names for c in SLOT_COLUMNS):
        return list(SLOT_COLUMNS)
    raise SystemExit(
        f"input has neither the archive layers {arch} nor the legacy triplet "
        f"{list(SLOT_COLUMNS)} — run regenerate_multilayer_activations.py first "
        f"(and ensure --save-layers covered center {center})."
    )


def _resolve_triplet(table: pa.Table, center: int) -> list:
    """The `center` triplet as (prev, centre, next) columns from a loaded table."""
    return [table.column(c) for c in _triplet_names(table.schema.names, center)]


def assemble_published(table: pa.Table, mode: str, center: int = 24) -> pa.Table:
    """Adapt one regenerated published subset into a training parquet (pure pyarrow).

    av : prompt(3-marker) + response(published <explanation> label) + 3 acts
    ar : prompt(published canonical critic, verbatim)              + 3 acts
    rl : prompt(3-marker)                                          + 3 acts

    The triplet is selected for `center` from the activation_L{k} archive (or a
    legacy prev/centre/next parquet) and written as activation_prev/centre/next.
    """
    assert mode in ("av", "ar", "rl"), mode
    n = table.num_rows
    names = table.schema.names

    prev, centre, nxt = _resolve_triplet(table, center)
    cols = {SLOT_COLUMNS[0]: prev, SLOT_COLUMNS[1]: centre, SLOT_COLUMNS[2]: nxt}
    # center_layer records THIS triplet's center (the build center, which may
    # differ from the archive's nominal center on a layer-sweep re-run).
    cols["center_layer"] = pa.array([center] * n, pa.int64())
    # Carry provenance so the inherited split stays auditable downstream (trainers
    # ignore these; they let a post-hoc check confirm av/ar/rl are doc-disjoint and
    # make an accidental wrong-corpus mix visible).
    for prov in ("doc_id", "n_raw_tokens"):
        if prov in names:
            cols[prov] = table.column(prov)

    if mode in ("av", "rl"):
        # Discard the published single-marker actor prompt; use our 3-marker prompt.
        cols[PROMPT_COL] = pa.array([build_av_prompt()] * n)

    if mode == "av":
        if RESPONSE_COL in names:
            resp = table.column(RESPONSE_COL).to_pylist()
        elif EXPLANATION_COL in names:
            resp = [wrap_explanation(e) for e in table.column(EXPLANATION_COL).to_pylist()]
        else:
            raise SystemExit(
                f"av needs a {RESPONSE_COL!r} or {EXPLANATION_COL!r} column to source the label"
            )
        bad = [i for i, r in enumerate(resp) if not r or EXPLANATION_OPEN not in r]
        assert not bad, (
            f"{len(bad)} av rows have an empty/unwrapped response (first idx {bad[0]}). "
            f"Expected the published <explanation>...</explanation> label."
        )
        cols[RESPONSE_COL] = pa.array(resp, pa.string())

    elif mode == "ar":
        if PROMPT_COL in names and pa.types.is_string(table.schema.field(PROMPT_COL).type):
            prompt = table.column(PROMPT_COL).to_pylist()
            # EXACT canonical check (not just "ends with <summary>"): the published
            # AR prompt must equal AR_CRITIC_TEMPLATE.format(explanation=X) — i.e.
            # have our prefix AND suffix — or AR-SFT trains on a different critic
            # input than RL scores with (RL builds fill_ar_prompt(explanation)).
            bad = [i for i, p in enumerate(prompt)
                   if not (p and p.startswith(_AR_PREFIX) and p.endswith(_AR_SUFFIX))]
            assert not bad, (
                f"{len(bad)} ar prompts are NOT the canonical critic template (first idx "
                f"{bad[0]}). RL scores with fill_ar_prompt(explanation), so the published AR "
                f"prompt MUST equal AR_CRITIC_TEMPLATE.format(...) exactly (prefix "
                f"{_AR_PREFIX!r} + suffix {_AR_SUFFIX!r}). Got: {prompt[bad[0]]!r}"
            )
        elif EXPLANATION_COL in names:
            prompt = [fill_ar_prompt(e) for e in table.column(EXPLANATION_COL).to_pylist()]
        else:
            raise SystemExit(
                f"ar needs a string {PROMPT_COL!r} (published critic prompt) or "
                f"{EXPLANATION_COL!r} column to source the critic input"
            )
        cols[PROMPT_COL] = pa.array(prompt, pa.string())

    return pa.table(cols)


def _subset_inputs(in_dir: Path, name: str) -> list:
    """Authoritative layer-bank inputs for a subset: the SHARDS if present (we never
    materialize a wide merged archive), else a single file. [] when absent."""
    shards = sorted(in_dir.glob(f"{name}.shard*of*.parquet"))
    if shards:
        return [str(s) for s in shards]
    single = in_dir / f"{name}.parquet"
    if single.exists():
        return [str(single)]
    return []


# --mode all discovers each logical subset under both naming schemes: the
# nanoNLA-published flat names (av_sft.parquet) and the EasyNLA warmstart's
# doc-level train/val files (av_sft_train.parquet / av_sft_val.parquet). Every
# variant found is built; at least one variant per logical subset must exist.
_SUBSET_SUFFIXES = ("", "_train", "_val")


def discover_subsets(in_dir: Path) -> list:
    """[(mode, fname, in_paths)] for every subset variant present in in_dir."""
    found = []
    for mode, base in (("av", "av_sft"), ("ar", "ar_sft"), ("rl", "rl")):
        variants = [(mode, f"{base}{suf}", _subset_inputs(in_dir, f"{base}{suf}"))
                    for suf in _SUBSET_SUFFIXES]
        variants = [(m, f, p) for m, f, p in variants if p]
        if not variants:
            raise SystemExit(
                f"no {base}[{'|'.join(s or '<flat>' for s in _SUBSET_SUFFIXES)}] parquet "
                f"(or shards) in {in_dir} — regenerate_bank must run on every subset first."
            )
        found.extend(variants)
    return found


def build_one(in_paths, mode: str, out_path: str, center: int = 24,
              batch_size: int = 4096) -> int:
    """Stream regenerated shard(s) -> one training parquet, batch by batch.

    `in_paths` is a path or list of shard paths (the authoritative layer bank —
    NEVER merged into one wide file). Projects ONLY the columns build needs (the
    {c-1,c,c+1} triplet + per-mode label/provenance) so the 8 unused archive
    layers are never read and the full ~180 GB archive never lands in RAM.
    Mirrors split_by_document's streaming idiom; appends every shard through one
    ParquetWriter.
    """
    if isinstance(in_paths, str):
        in_paths = [in_paths]
    assert in_paths, "build_one got no input parquets"
    schema_names = pq.ParquetFile(in_paths[0]).schema_arrow.names
    tri = _triplet_names(schema_names, center)
    extra = [c for c in (PROMPT_COL, RESPONSE_COL, EXPLANATION_COL, "doc_id", "n_raw_tokens")
             if c in schema_names]
    projection = list(dict.fromkeys(tri + extra))  # dedup, preserve order

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n = 0
    try:
        for in_path in in_paths:
            for batch in pq.ParquetFile(in_path).iter_batches(batch_size=batch_size, columns=projection):
                out = assemble_published(pa.Table.from_batches([batch]), mode, center)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, out.schema)
                writer.write_table(out)
                n += out.num_rows
    finally:
        if writer is not None:
            writer.close()
    src = in_paths[0] if len(in_paths) == 1 else f"{len(in_paths)} shards"
    print(f"[from-published:{mode}] center={center} {src} -> {out_path}  "
          f"({n} rows, streamed; read {len(projection)} of {len(schema_names)} cols)")
    return n


def _assert_docs_disjoint(out_dir: Path, names) -> dict:
    """One-time guard: every built subset's doc_id set must be pairwise disjoint
    with every other (av/ar/rl split AND the doc-level train/val split are both
    by document). Catches a wrong-corpus mix or a split bug before training.
    Reads only the doc_id column. Returns per-subset unique-doc counts."""
    sets, counts = {}, {}
    for name in names:
        p = out_dir / f"{name}.parquet"
        if "doc_id" not in pq.ParquetFile(p).schema_arrow.names:
            print(f"[from-published] doc_id absent in {name} — skipping disjointness check")
            return {}
        s = set(pq.read_table(p, columns=["doc_id"]).column("doc_id").to_pylist())
        sets[name], counts[name] = s, len(s)
    ordered = list(sets)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            inter = sets[a] & sets[b]
            assert not inter, (
                f"{len(inter)} doc_id(s) shared between {a} and {b} (e.g. {sorted(inter)[:3]}). "
                f"All subsets must be document-disjoint (av/ar/rl split and train/val split are "
                f"both doc-level) — a leak means mixed subsets or the wrong corpus."
            )
    print(f"[from-published] doc-disjoint OK across {len(ordered)} subsets: unique docs {counts}")
    return counts


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["av", "ar", "rl", "all"], required=True)
    p.add_argument("--center", type=int, default=24,
                   help="center layer c; selects activation_L{c-1,c,c+1} from the archive as "
                        "the training triplet. Re-run with a different --center to sweep centers "
                        "without re-extracting (the §5 layer-selection sweep).")
    p.add_argument("--in", dest="inp", help="regenerated published parquet or glob (for av/ar/rl)")
    p.add_argument("--out", help="output training parquet (for av/ar/rl)")
    p.add_argument("--in-dir", help="for --mode all: dir with regenerated av_sft/ar_sft/rl "
                                    "(shards *.shardNNofMM.parquet are used directly — no merge)")
    p.add_argument("--out-dir", help="for --mode all: output dir")
    args = p.parse_args()

    if args.mode == "all":
        assert args.in_dir and args.out_dir, "--mode all needs --in-dir and --out-dir"
        ind, outd = Path(args.in_dir), Path(args.out_dir)
        outd.mkdir(parents=True, exist_ok=True)
        subsets = discover_subsets(ind)
        counts = {}
        for mode, fname, in_paths in subsets:
            counts[fname] = build_one(in_paths, mode, str(outd / f"{fname}.parquet"), args.center)
        docs = _assert_docs_disjoint(outd, [f for _, f, _ in subsets])
        (outd / "build_manifest.json").write_text(json.dumps({
            "source": "build_from_published",
            "in_dir": str(ind),
            "counts": counts,
            "unique_docs": docs,
            "center_layer": args.center,
            "split": "inherited from the published subsets (not re-split); doc-disjoint asserted",
            "note": "labels reused from the published warmstart dataset; "
                    "activations regenerated locally (no API).",
        }, indent=2))
        print(f"[from-published] {[f for _, f, _ in subsets]} (center {args.center}) -> {outd}  "
              f"counts={counts}")
    else:
        assert args.inp and args.out, "--mode av/ar/rl needs --in and --out"
        import glob
        ins = sorted(glob.glob(args.inp)) or [args.inp]  # expand a shard glob if given
        build_one(ins, args.mode, args.out, args.center)


if __name__ == "__main__":
    main()
