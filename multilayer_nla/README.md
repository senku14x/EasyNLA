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
| `build_from_published.py` | bank → av/ar/rl training parquets for a chosen `--center` (re-slice, no re-extraction) | CPU |
| `splits.py` | doc-level train/dev/test manifests (+ locked eval subsets) | CPU |
| `train_ar_multi.py` | multi-tap AR (truncated backbone + per-depth heads) SFT warm-start | GPU |
| `train_av_multi.py` | k-slot AV (multi-marker Karvonen injection) SFT warm-start | GPU |
| `train_rl_multi.py` | single-GPU GRPO (⚠ legacy fixed 3-slot scheme, not `av_in_*`-aware) | GPU |
| `evaluate_e2e.py` | held-out end-to-end FVE: AV text → AR → fixed targets, bootstrap CIs over documents, shuffled control | GPU |
| `eval_ar_gold.py` | AR-only gold ceiling (localizes verbalizer vs reconstructor bottleneck) | GPU |
| `extract_multilayer.py` | fresh-corpus stage-0 (keyed-RNG positions) + the `MultiLayerHFExtractor` everything above uses | GPU |
| `datasets.py` `injection_multi.py` `models_multi.py` | k-slot prompt/loaders, multi-marker injection, multi-tap critic | — |
| `tests/` | 96 offline tests incl. a CPU end-to-end smoke on a tiny in-memory model | anywhere |

Not ported (completed-phase / sweep-specific, in the parent repo if needed):
`build_sweep.py`, `analyze_sweep.py`, `select_and_report`-driven §7 sweep
harness (a trimmed `select_and_report.py` IS here), `headroom.py` (Gate 0),
`progressive_reader/`, ops/cluster scripts.

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

**Storage math** (Qwen3-8B, d=4096; `--dry-run` prints exact numbers):

| columns | per row |
|---|---:|
| `activation_L19..29` fp32 (11 layers) | 176 KB |
| `activation_L19..29` fp16 | 88 KB |
| `window_L{23,24,25}` W=8 fp16 | 197 KB |
| `activation_L23..25` fp32 only (minimal) | 48 KB |

The AR only ever reads final-token targets → **`--window 0` on `ar_sft`**.
Windows go on the AV-side subsets (`av_sft`, `rl`).

## Runbook (vast box)

```bash
git clone https://github.com/senku14x/EasyNLA && cd EasyNLA && git checkout multi_layer_nla
python -m venv .venv && source .venv/bin/activate && pip install -e . && pip install bitsandbytes
export HF_HOME=/data/hf  # big disk

# 0. download the published warmstart data (labels + stored L24 vectors + prefixes)
huggingface-cli download asher577/easynla-warmstart-data --repo-type dataset --local-dir /data/pub
# echo the schema BEFORE anything else — expect detokenized_text_truncated,
# n_raw_tokens, activation_vector, activation_layer, doc_id, prompt[, response]:
python - <<'PY'
import pyarrow.parquet as pq, glob
for f in sorted(glob.glob("/data/pub/*.parquet")):
    print(f, pq.ParquetFile(f).schema_arrow.names, pq.ParquetFile(f).metadata.num_rows)
PY

# 1. storage estimate first (no GPU) — then the real runs
BANK=/data/mlnla/bank
python -m multilayer_nla.regenerate_bank --in /data/pub/av_sft_train.parquet --out $BANK/av_sft_train.parquet \
    --base-model Qwen/Qwen3-8B --save-layers 19-29 --window 8 --window-layers 23,24,25 --dry-run

for s in av_sft_train av_sft_val rl_train rl_val; do        # AV-side: windows ON
  python -m multilayer_nla.regenerate_bank --in /data/pub/$s.parquet --out $BANK/$s.parquet \
      --base-model Qwen/Qwen3-8B --save-layers 19-29 --window 8 --window-layers 23,24,25 \
      --max-length 4096 --batch-size 16 --length-bucket --max-drop-frac 1e-3
done
for s in ar_sft_train ar_sft_val; do                        # AR-side: final-token only
  python -m multilayer_nla.regenerate_bank --in /data/pub/$s.parquet --out $BANK/$s.parquet \
      --base-model Qwen/Qwen3-8B --save-layers 19-29 --window 0 \
      --max-length 4096 --batch-size 16 --length-bucket --max-drop-frac 1e-3
done
# (adjust the subset list to whatever step 0 printed; multi-GPU: add
#  --num-shards N --shard-index i per job — the tool prints the merge one-liner)

# 2. integrity gates (cheap; run them EVERY time)
python -m multilayer_nla.verify_bank --bank $BANK/av_sft_train.parquet
python -m multilayer_nla.verify_regen_parity --base-model Qwen/Qwen3-8B \
    --parquet /data/pub/av_sft_train.parquet --center-layer 24 --n-rows 64 --max-length 4096

# 3. bank → training parquets for center 24 (re-run with a different --center
#    to sweep centers WITHOUT re-extracting)
TRAIN=/data/mlnla/train_c24
python -m multilayer_nla.build_from_published --mode all --center 24 --in-dir $BANK --out-dir $TRAIN

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

## Windowed data downstream (pre-registered, NOT yet built)

The bank's `window_L{k}` columns are inert until an AV consumes a W-slot
window. The pilot design (from the parent repo, kept as the pre-registration):
three arms varying ONLY the AV input against the same frozen AR + fixed target
at p — `single` (W=1), `window` (true W-window), `dup` (p-vector ×W: same
marker count, zero extra information — the load-bearing control), plus a
shuffled-window control. Decision rule: `window − dup` paired doc-bootstrap CI
excludes 0 AND `window > shuffled` ⇒ context helps; `window ≈ dup ≈ single` ⇒
null (adjacent late positions are collinear; report either outcome).

## Known limitations

- `train_rl_multi.py` is the parent repo's fixed 3-slot GRPO (SLOT_COLUMNS
  scheme); it is NOT `av_in_*`-aware and does NOT use EasyNLA's fast
  distributed vLLM path. Wiring multi-slot injection into `vllm-lens` is open
  work — warm-start + SFT-level comparisons don't need it.
- Extraction batch composition differs from the original stage-0 run, so
  stored-vs-regenerated vectors match to bf16 batching noise (cos ~0.9999),
  not bitwise. The verifiers' `--strict` mode is only for batch-identical
  forwards.
- Tested against `transformers==4.57.1` (the repo pin). v5 breaks
  `apply_chat_template` usage in the trainers.
