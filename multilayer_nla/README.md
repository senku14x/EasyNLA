# multilayer_nla — multi-layer × multi-position NLA on the EasyNLA core

Port of the nanoNLA `multilayer_nla` (+ `multitoken_nla`) experiment packages,
adapted to EasyNLA's core: `nla.utils.arch_adapters`, the short-circuit
`HFExtractor` (upper layers + lm_head never run), arch-aware LoRA targets.
Never mutates `nla/` — the single-layer path is untouched.

**Lineage / prior results (from the parent repo, Qwen3-8B, SFT-only, held-out
test):** multi-layer AV input beats single-layer, significant but modest
(local−duplicate **+1.6pp** FVE, k=3); layer combos saturate past
k=2-with-L24 (L20,L24→L24 = 0.583 best, AR-gold ceiling ≈ 0.67); the
**verbalizer, not the AR, is the bottleneck**. Full records live in the parent
repo's `EXPERIMENT_REPORT.md` / `SWEEP_STATUS.md` / `docs/layer_combos_explained.md`.

## What's here

| script | role | runs on |
|---|---|---|
| `regenerate_bank.py` | **published labels → multi-layer × multi-position bank, ONE GPU pass** | GPU |
| `verify_bank.py` | post-build integrity gate (schema, finiteness, last-slot parity, stored-vector + reference cross-parity) | CPU |
| `verify_center_parity.py` | multi-hook center tap == legacy single-layer extraction (bitwise) | GPU |
| `verify_regen_parity.py` | final-token gather == legacy final token on published prefixes (bitwise) | GPU |
| `conditions.py` | **slot-spec grammar**: `L24@0`, `L23@-2`, flags `\|pool` `\|shufctx`; the pre-registered `PERMUTATION_GRID` | — |
| `build_conditions.py` | **bank → per-condition datasets for ANY layers×positions grid** (fixed target, stored prompt, preflight) | CPU |
| `build_from_published.py` | bank → plain av/ar/rl training parquets for a chosen `--center` (the no-conditions warmstart path) | CPU |
| `split_parquet.py` | doc-level train/val materialization of a built parquet (+ manifest) | CPU |
| `splits.py` | doc-level train/dev/test manifests (+ locked eval subsets) — feeds build_conditions | CPU |
| `train_ar_multi.py` | multi-tap AR (truncated backbone + per-depth heads) SFT warm-start | GPU |
| `train_av_multi.py` | k-slot AV SFT warm-start (any k; memoized prompt tokenization; `--ram-dtype float16`) | GPU |
| `train_rl_multi.py` | single-GPU GRPO, **k-general** (injects the parquet's `av_in_*`, fixed target sliced to the AR's taps, greedy held-out eval) | GPU |
| `distill_av.py` | **warmstart improvement, API-free**: gold-label AR scoring/filtering + best-of-N self-distillation | GPU |
| `evaluate_e2e.py` | held-out end-to-end FVE: AV text → AR → fixed targets, bootstrap CIs over documents, shuffled control | GPU |
| `eval_ar_gold.py` | AR-only gold ceiling (localizes verbalizer vs reconstructor bottleneck) | GPU |
| `extract_multilayer.py` | fresh-corpus stage-0 (keyed-RNG positions) + the `MultiLayerHFExtractor` everything above uses | GPU |
| `datasets.py` `injection_multi.py` `models_multi.py` | k-slot prompt/loaders (legacy + neutral templates), multi-marker injection, multi-tap critic | — |
| `tests/` | 117 offline tests incl. a CPU end-to-end extractor smoke + a synthetic-bank build_conditions round trip | anywhere |

Not ported (completed-phase / superseded, in the parent repo if needed):
`build_sweep.py` (superseded by `build_conditions.py` — every §7 condition is
expressible in the slot grammar, `conditions.LEGACY_SWEEP`), `analyze_sweep.py`,
`headroom.py` (Gate 0), `progressive_reader/`, ops/cluster scripts.

## The unified bank (why one pass)

The parent repo needed two GPU passes over the corpus: one for final-token
multi-layer vectors (`regenerate_multilayer_activations`), a second for
multi-position windows (`multitoken_nla/build_window_bank`). The forward
already computes everything — `regenerate_bank.py` captures both in one pass:

```
activation_L{k}   [d]    final-token (labeled position p), k in --save-layers
window_L{k}       [W·d]  last W positions [p-W+1 … p], slot-major, slot W-1 == p,
                         k in --window-layers
```

plus every published column carried through (labels inherited for free — the
explanation only ever depended on the prefix TEXT, so it is as valid for any
layer/position slice of that prefix as for the original L24 vector).

**Guards** (all default-on): `n_raw_tokens` round-trip (hard fail; the
retokenized prefix must land the final token exactly on the labeled position);
**stored-vector cross-parity** — when the input carries the original
`activation_vector` (the EasyNLA warmstart does), the regenerated
`activation_L{that layer}` must match it (median cosine ≥ 0.999 per chunk;
this catches wrong `--max-length`, wrong model revision, wrong hook semantics,
and same-count retokenization drift in one shot — the strongest guard, and it
costs nothing); short-window drop (≈0 rows given stage-0's `_MIN_POSITION=50`);
float16 range checks before every fp16 cast.

**Storage math** (Qwen3-8B, d=4096; `--dry-run` prints exact numbers). The
ceselder set is ~1M rows total (av ~250k / ar ~250k / rl ~500k):

| columns | per row | full ceselder run |
|---|---:|---:|
| `activation_L19..29` fp32 (11 layers) | 176 KB | ~176 GB (all 1M rows — matches the old bank) |
| `activation_L19..29` fp16 | 88 KB | ~88 GB |
| `window_L{23,24,25}` W=8 fp16 | 197 KB | ~148 GB (av+rl rows only) |
| `activation_L23..25` fp32 only (minimal) | 48 KB | ~48 GB |

The AR only ever reads final-token targets → **`--window 0` on `ar_sft`**.
Windows go on the AV-side subsets (`av_sft`, `rl`). If disk is tight: windows
on `av_sft` only, or a `--max-rows 40000` window pilot (the pre-registered
multitoken pilot size) alongside a window-free full archive.

## Data source (decided 2026-07-08): the ceselder lineage

The rerun uses **`ceselder/qwen3-8b-nla-L24-finefineweb-100k`** — the same
labels the §7/§8 sweeps and the existing `senku21x/qwen3-8b-nla-multilayer-L19-29`
bank were built from, so all previous numbers stay comparable (AR-gold ceiling
≈ 0.67 under the converged recipe is the sanity anchor a fresh AR retrain
should land near).

Consequence: ceselder rows carry **no stored `activation_vector`**, so the
in-run stored-vector parity guard skips (it prints a NOTE). The compensating
gates are MANDATORY, not optional: `verify_regen_parity` (bitwise, on real
published rows) and `verify_bank --existing` against a shard of the OLD
L19-29 bank (joined on `doc_id` + `n_raw_tokens` — an off-position or
wrong-layer rerun collapses the median cosine).

The tooling also accepts `asher577/easynla-warmstart-data` (EasyNLA's own
warmstart; stored vectors present → the parity guard arms automatically, and
`_train`/`_val` files are discovered as-is) — but it is a DIFFERENT lineage
(newer Sonnet 4.6 labels, 371k rows): never mix the two in one experiment, and
re-establish baselines before comparing anything across them.

## Runbook (vast box)

```bash
git clone https://github.com/senku14x/EasyNLA && cd EasyNLA && git checkout multi_layer_nla
python -m venv .venv && source .venv/bin/activate && pip install -e . && pip install bitsandbytes
export HF_HOME=/data/hf  # big disk

# 0. download the published labeled subsets (text + labels, no vectors)
PUB=/data/pub
python - <<'PY'
import os
from datasets import load_dataset
os.makedirs("/data/pub", exist_ok=True)
for name in ("av_sft", "ar_sft", "rl"):
    ds = load_dataset("ceselder/qwen3-8b-nla-L24-finefineweb-100k", name, split="train")
    ds.to_parquet(f"/data/pub/{name}.parquet"); print(name, ds.num_rows)
PY
# schema echo BEFORE anything else — expect detokenized_text_truncated,
# n_raw_tokens, doc_id, prompt[, response]; NO activation_vector:
python - <<'PY'
import pyarrow.parquet as pq, glob
for f in sorted(glob.glob("/data/pub/*.parquet")):
    print(f, pq.ParquetFile(f).metadata.num_rows, pq.ParquetFile(f).schema_arrow.names)
PY
# reference for cross-parity: ONE shard of the old L19-29 bank
huggingface-cli download senku21x/qwen3-8b-nla-multilayer-L19-29 --repo-type dataset \
    --include "*av_sft*shard00*" --local-dir /data/oldbank   # adjust pattern to the repo layout

# 1. storage estimate first (no GPU) — then the real runs
BANK=/data/mlnla/bank
python -m multilayer_nla.regenerate_bank --in $PUB/av_sft.parquet --out $BANK/av_sft.parquet \
    --base-model Qwen/Qwen3-8B --save-layers 19-29 --window 8 --window-layers 23,24,25 --dry-run

for s in av_sft rl; do                                      # AV-side: windows ON
  python -m multilayer_nla.regenerate_bank --in $PUB/$s.parquet --out $BANK/$s.parquet \
      --base-model Qwen/Qwen3-8B --save-layers 19-29 --window 8 --window-layers 23,24,25 \
      --max-length 4096 --batch-size 16 --length-bucket --max-drop-frac 1e-3
done
python -m multilayer_nla.regenerate_bank --in $PUB/ar_sft.parquet --out $BANK/ar_sft.parquet \
    --base-model Qwen/Qwen3-8B --save-layers 19-29 --window 0 \
    --max-length 4096 --batch-size 16 --length-bucket --max-drop-frac 1e-3
# multi-GPU: add --num-shards N --shard-index i per job (the tool prints the
# merge one-liner). Pilot first: --max-rows 2000 on av_sft, then verify, then full.

# 2. integrity gates (cheap; MANDATORY on this lineage — no stored vectors in-run)
python -m multilayer_nla.verify_regen_parity --base-model Qwen/Qwen3-8B \
    --parquet $PUB/av_sft.parquet --center-layer 24 --n-rows 64 --max-length 4096
python -m multilayer_nla.verify_bank --bank $BANK/av_sft.parquet \
    --existing /data/oldbank/<shard>.parquet          # old bank = known-good reference
python -m multilayer_nla.verify_bank --bank $BANK/ar_sft.parquet
python -m multilayer_nla.verify_bank --bank $BANK/rl.parquet

# 3. bank → training parquets for center 24 (re-run with a different --center
#    to sweep centers WITHOUT re-extracting), then doc-level train/val split
TRAIN=/data/mlnla/train_c24
python -m multilayer_nla.build_from_published --mode all --center 24 --in-dir $BANK --out-dir $TRAIN
for s in av_sft ar_sft rl; do
  python -m multilayer_nla.split_parquet --in $TRAIN/$s.parquet --out-dir $TRAIN \
      --prefix $s --fracs 0.9,0.1 --names train,val --seed 42
done
# (same seed ⇒ same doc buckets on any future --center rebuild — eval docs stay fixed)

# 4. warm-start: shared multi-tap AR first, then the AV
python -m multilayer_nla.train_ar_multi --base-ckpt Qwen/Qwen3-8B \
    --parquet $TRAIN/ar_sft_train.parquet --eval-parquet $TRAIN/ar_sft_val.parquet \
    --save-dir /data/ckpt/ar_3tap --tap-layers 23,24,25 --use-lora --quant none \
    --num-steps 3000 --batch-size 64 --gradient-accumulation-steps 4 --lr 1e-4 \
    --wandb-project easynla-multilayer --wandb-name ar_3tap
python -m multilayer_nla.train_av_multi --base-ckpt Qwen/Qwen3-8B \
    --parquet $TRAIN/av_sft_train.parquet --eval-parquet $TRAIN/av_sft_val.parquet \
    --save-dir /data/ckpt/av_local --use-lora --quant none \
    --num-steps 1000 --batch-size 64 --wandb-project easynla-multilayer --wandb-name av_local

# 5. held-out numbers (never report training-rollout FVE)
python -m multilayer_nla.eval_ar_gold --base-ckpt Qwen/Qwen3-8B \
    --ar-ckpt /data/ckpt/ar_3tap/iter_0003000 --eval-parquet $TRAIN/ar_sft_val.parquet \
    --summary /data/eval/ar_gold_val.json
#   sanity anchor: the §8 converged recipe put AR-gold ≈ 0.67 on this lineage —
#   a big deviation means recipe drift, investigate before training the AV further.
python -m multilayer_nla.evaluate_e2e --base-ckpt Qwen/Qwen3-8B \
    --av-ckpt /data/ckpt/av_local/iter_0001000 --ar-ckpt /data/ckpt/ar_3tap/iter_0003000 \
    --eval-parquet $TRAIN/rl_val.parquet --condition local \
    --out /data/eval/e2e_val.jsonl --summary /data/eval/e2e_val.json
```

Recipe notes from the parent repo's converged runs: the AR level is
hyperparameter-sensitive — 3000 steps, effective batch ~256, raised LR, and
bf16 (`--quant none`) moved AR-gold 0.624→0.67. AV at 1000 steps / batch 64.
The two `--eval-parquet` flags give train-time held-out curves; the step-5
evals are the reportable numbers.

## Permutation-grid runbook (after the bank exists)

```bash
# 0. doc-level split manifests over the rl + ar banks (locked eval subsets)
COND=/data/mlnla/cond_grid
python -m multilayer_nla.splits --source "$BANK/rl.parquet" --name rl --out-dir $COND \
    --seed 42 --fracs 0.8,0.1,0.1 --dev-subset 256 --test-subset 1000
python -m multilayer_nla.splits --source "$BANK/ar_sft.parquet" --name ar --out-dir $COND --seed 42

# 1. build every condition dataset (CPU; re-runnable for any new grid without GPU)
python -m multilayer_nla.build_conditions --mode all --conditions-preset permutation_grid \
    --in-dir $BANK --out-dir $COND \
    --rl-split-manifest $COND/rl_split_manifest.json --ar-split-manifest $COND/ar_split_manifest.json \
    --av-with-targets --base-ckpt Qwen/Qwen3-8B
# custom grids: --conditions "tok5=L24@-4,L24@-3,L24@-2,L24@-1,L24@0; dup5=L24@0,L24@0,L24@0,L24@0,L24@0"
# (validated against the bank: any slot L{k}@-j needs window_L{k} with W > j)

# 2. ONE shared AR (never retrained per condition), then one AV per condition
python -m multilayer_nla.train_ar_multi --base-ckpt Qwen/Qwen3-8B \
    --parquet $COND/ar_common.parquet --eval-parquet $COND/ar_dev.parquet \
    --save-dir /data/ckpt/ar_3tap --tap-layers 23,24,25 --use-lora --quant none \
    --num-steps 3000 --batch-size 64 --gradient-accumulation-steps 4 --lr 1e-4
CONDS="single dup3 lay3 tok3 mix4 dup4 tok5 tok5w dup5"
for c in $CONDS; do
  python -m multilayer_nla.train_av_multi --base-ckpt Qwen/Qwen3-8B \
      --parquet $COND/av_$c.parquet --save-dir /data/ckpt/av_$c --use-lora --quant none \
      --num-steps 1000 --batch-size 64 --wandb-name av_$c
done

# 3. eval: dev for selection, test ONCE; each shufctx arm evaluated with its
#    PARENT condition's AV (tok3_shufctx <- av_tok3, tok5_shufctx <- av_tok5, ...)
for c in $CONDS; do
  python -m multilayer_nla.evaluate_e2e --base-ckpt Qwen/Qwen3-8B \
      --av-ckpt /data/ckpt/av_$c/iter_0001000 --ar-ckpt /data/ckpt/ar_3tap/iter_0003000 \
      --eval-parquet $COND/rl_test_$c.parquet --condition $c \
      --out /data/eval/test_$c.jsonl --summary /data/eval/test_$c.json
done
for c in tok3 tok5 tok5w; do
  python -m multilayer_nla.evaluate_e2e --base-ckpt Qwen/Qwen3-8B \
      --av-ckpt /data/ckpt/av_$c/iter_0001000 --ar-ckpt /data/ckpt/ar_3tap/iter_0003000 \
      --eval-parquet $COND/rl_test_${c}_shufctx.parquet --condition ${c}_shufctx \
      --out /data/eval/test_${c}_shufctx.jsonl --summary /data/eval/test_${c}_shufctx.json
done

# 4. optional warmstart improvement round (see below), then re-train + re-eval
python -m multilayer_nla.distill_av --mode score-gold --in $COND/av_lay3.parquet \
    --out $COND/av_lay3_scored.parquet --ar-ckpt /data/ckpt/ar_3tap/iter_0003000 --filter-quantile 0.2
python -m multilayer_nla.distill_av --mode bon --in $COND/av_tok3.parquet \
    --out $COND/av_tok3_bon8.parquet --av-ckpt /data/ckpt/av_tok3/iter_0001000 \
    --ar-ckpt /data/ckpt/ar_3tap/iter_0003000 --n-samples 8 --max-rows 50000
```

## Invariants (the contract — do not break)

- **RAW storage everywhere** (`norm="none"`); normalization only at injection
  (norm-matched add) and loss (√d) time.
- **Document-level splits only**; never split one doc's positions across
  buckets; disjointness is asserted, not assumed.
- **AR target ≠ AV input.** The AR reconstruction target stays fixed (the
  center triplet / `activation_prev/centre/next`) no matter what the AV sees
  (`av_in_*` or the triplet columns). Distinct names are the mechanism.
- **AV emits text only; AR reads text only.** No activation crosses between
  them — `local`'s av_in == ar_target is a round trip through language.
- **The marker is scanned inside the hook**, never precomputed; per-row marker
  count is asserted (k markers per prompt).
- **`--max-length` must match the original extraction (4096)** — the
  round-trip + parity guards enforce this; do not disable them.
- **Never train on dev/test; never select on test.** Predict-the-mean
  baselines come from the eval split only. Bootstrap resamples documents;
  the shuffled control permutes across documents and must collapse.
- Loudest injection smoke test: grep generated text for CJK (a silent
  injection failure makes the actor free-associate Chinese). The mechanism
  check is the per-row marker-count guard, which RL cannot erode.

## The permutation grid (pre-registered)

**Primary question.** At a MATCHED slot count k, does position diversity
(several token positions, one layer) carry more recoverable information
through the language bottleneck than layer diversity (several layers, one
position)? Prior §7/§8 results say layer diversity is real but small
(+1.6pp; adjacent layers are largely redundant — consistent with the SAE
literature's slowly-drifting residual features across depth). Adjacent
POSITIONS carry genuinely different content, so the prediction to beat is
`tok3 > lay3 > dup3 > single`. The main alternative: late-position states
already summarize their context (the progressive-reader-style null), giving
`tok3 ≈ dup3`.

`conditions.PERMUTATION_GRID` (build with `--conditions-preset permutation_grid`):

| condition | slots | role |
|---|---|---|
| `single` | L24@0 | baseline (k=1) |
| `dup3` | L24@0 ×3 | marker-count control (k=3, zero extra info) |
| `lay3` | L23@0, L24@0, L25@0 | layer diversity (k=3) — §7 `local` |
| `tok3` | L24@-2, L24@-1, L24@0 | **position diversity (k=3)** |
| `mix4` | L23@-1, L23@0, L25@-1, L25@0 | positions × layers interaction (k=4) |
| `dup4` | L24@0 ×4 | marker-count control at k=4 |
| `tok5` | L24@-4 … L24@0 | contiguous 5-position window (k=5) |
| `tok5w` | L24@{-7,-4,-2,-1,0} | **log-spaced 5 positions, full W=8 reach** (k=5) |
| `dup5` | L24@0 ×5 | marker-count control at k=5 |
| `tok{3,5,5w}_shufctx` | parent \| shufctx | eval-only: context slots from another doc, final true |

Every condition reconstructs the SAME fixed target (default [L23,L24,L25]@p);
same-k conditions share ONE neutral prompt (identical text — only the vectors
differ); dev selects checkpoints, test is touched once. Decision rules
(paired doc-bootstrap, evaluate_e2e):
- `tok3 − dup3` CI excludes 0 AND `tok3 > tok3_shufctx` ⇒ position context is
  used as this-document context (proceed to wider windows / RL on the winner).
- `tok3 ≈ dup3 ≈ single` ⇒ the multitoken null: late positions are collinear
  for this channel — report it and stop investing in position slots.
- **Dose-response**: `single → tok3 → tok5` each vs its dup control — a rising
  paired-Δ curve says window width keeps paying; flat says it saturates by k=3.
- **Adjacency vs span at k=5**: `tok5w − tok5` CI excludes 0 ⇒ decorrelated
  (log-spaced) positions beat contiguous ones — the position-space analog of
  §7's stride-2 layer result. A `tok5w` win motivates re-banking with
  `--window 16/32` for longer reach (offsets beyond p-7 need a wider bank;
  everything here builds from the existing W=8 bank on CPU).
- `lay3 − dup3` should reproduce §7's +1.6pp under the neutral template
  (a recipe sanity anchor, not a new claim).

## Layer grid (free — uses the full stored L19-29 band)

`--conditions-preset layer_grid`: every slot is at position p (offset 0), so it
needs only `activation_L{k}` (no windows) and any layer in the 11-layer bank is
fair game. All conditions reconstruct the SAME fixed [L23,L24,L25]@p target;
only the AV **input layers** vary. This is the §7/§8 layer question at full
resolution:

| condition | input layers @p | tests |
|---|---|---|
| `single` / `dup2` / `dup3` / `dup5` | L24 ×{1,2,3,5} | marker-count controls |
| `lay2` / `lay3` | 23,25 / 23,24,25 | adjacent to target |
| `far2` | 19, 29 | the two extremes (max decorrelation, k=2) |
| `wide` | 20, 24, 28 | wide span (k=3) — §7 `wide` |
| `s2lo` / `s2hi` | 19,21,23 / 20,22,24 | stride-2 (k=3) — §7 stride |
| `band5` / `spread5` | 21-25 / 19,22,24,26,29 | adjacency vs span (k=5) |

Head-to-heads (paired doc-bootstrap, matched k): `wide`/`s2hi` − `lay3`
(does span beat adjacency, as §7 found?); `band5` − `spread5` (same at k=5);
`single→lay3→band5` each vs its dup (does depth keep paying, or plateau by
k=2-with-L24 as §8 saw?); `far2` − `lay2` (extremes vs adjacent pair).

Changing the reconstruction **target** center (not input) is the one thing the
bank does NOT make free — the shared AR is fixed to reconstruct L23/24/25, so a
target sweep needs a retrained AR (`AR_LAYER_TO_TARGET_COL` + `--tap-layers`).

## Warmstart improvement (the actual bottleneck)

SFT imitates gold, and the gold ceiling is what caps the warm start. Two
API-free levers in `distill_av.py`, both scored by the frozen AR against the
FIXED target:

1. **`--mode score-gold`** — score every published label; report the reward
   distribution; `--filter-quantile 0.2` drops the worst 20% (labels the
   reconstructor can't map toward the target teach style, not content).
2. **`--mode bon`** — best-of-N self-distillation (ReST-style RL-lite): sample
   N explanations per row from the SFT AV, keep the best-scoring (gold
   included in the argmax), SFT again on the winners. Besides raising reward,
   this **de-layer-blinds the labels**: published explanations only ever
   described the single L24 vector, but BoN selects against the full fixed
   multi-layer target.

Caution (research-notes §10.1): BoN optimizes text against the SAME AR that
scores it. The honest number is held-out e2e FVE after the continuation SFT
round, cross-checked with an independently trained AR before any faithfulness
language; watch explanation diversity / text_judges for reward-hacked
templates.

## Known limitations

- `train_rl_multi.py` is single-GPU (k-general now, but NOT EasyNLA's fast
  distributed vLLM path). Wiring multi-slot injection into `vllm-lens` is open
  work — warm-start + SFT-level comparisons don't need it.
- Extraction batch composition differs from the original stage-0 run, so
  stored-vs-regenerated vectors match to bf16 batching noise (cos ~0.9999),
  not bitwise. The verifiers' `--strict` mode is only for batch-identical
  forwards.
- Neutral-template FVE is NOT comparable to §7/§8 legacy-template numbers;
  re-run baselines inside each experiment.
- Tested against `transformers==4.57.1` (the repo pin). v5 breaks
  `apply_chat_template` usage in the trainers.
