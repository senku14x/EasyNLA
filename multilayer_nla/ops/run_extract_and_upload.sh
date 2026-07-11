#!/usr/bin/env bash
# =============================================================================
# EasyNLA multilayer_nla — one-shot bank extraction + HF upload (vast box).
#
# Does everything from a bare instance:
#   1. clone the repo @ multi_layer_nla + install
#   2. huggingface + wandb login
#   3. download the ceselder published labels (av_sft / ar_sft / rl)
#   4. PREFLIGHT: verify_regen_parity (fail fast — 1 min, before hours of GPU)
#   5. extract the multi-layer x multi-position bank (L19-29 + W=8 windows),
#      fanned out across every visible GPU
#   6. verify each bank subset (verify_bank)
#   7. upload the ENTIRE bank to a NEW private HF dataset repo, under bank/
#
# The bank supports the full permutation grid incl. tok5w (needs W>=8) and any
# center in 20-28 (11 layers saved). Downstream steps (splits, build_conditions,
# training) are NOT here — this script's deliverable is the uploaded bank.
#
# Run:
#   export HF_TOKEN=hf_xxx WANDB_API_KEY=xxx
#   export HF_BANK_REPO=<your-hf-namespace>/easynla-qwen3-8b-multilayer-bank
#   bash run_extract_and_upload.sh
# Idempotent: re-run to resume (skips subsets whose bank files already exist).
# =============================================================================
set -euo pipefail

# ---- config (all env-overridable) ------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/senku14x/EasyNLA.git}"
BRANCH="${BRANCH:-multi_layer_nla}"
ROOT="${ROOT:-$HOME}"                       # where to clone
DATA="${DATA:-$HOME/mlnla}"                 # POINT THIS AT YOUR BIG DISK
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
SRC_DATASET="${SRC_DATASET:-ceselder/qwen3-8b-nla-L24-finefineweb-100k}"
SUBSETS="${SUBSETS:-av_sft ar_sft rl}"      # which published configs to pull

# bank geometry — matches the permutation grid (tok5w reaches p-7 => W>=8)
SAVE_LAYERS="${SAVE_LAYERS:-19-29}"
WINDOW="${WINDOW:-8}"
WINDOW_LAYERS="${WINDOW_LAYERS:-23,24,25}"
ACT_DTYPE="${ACT_DTYPE:-float32}"           # float16 ~halves upload (range-guarded)
WIN_DTYPE="${WIN_DTYPE:-float16}"
MAX_LENGTH="${MAX_LENGTH:-4096}"            # MUST match the published extraction
BATCH_SIZE="${BATCH_SIZE:-16}"              # lower if you OOM
MAX_DROP_FRAC="${MAX_DROP_FRAC:-1e-3}"      # tolerate rare per-row tokenizer drift

# HF upload target — a NEW private dataset repo; the bank lands under bank/
HF_BANK_REPO="${HF_BANK_REPO:?set HF_BANK_REPO=<namespace>/<repo> (the new bank dataset)}"
HF_BANK_PATH="${HF_BANK_PATH:-bank}"        # folder inside the repo

# GPU fan-out: shards per subset = #visible GPUs (override NSHARDS to force)
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)"; [ "$NGPU" -ge 1 ] || NGPU=1
NSHARDS="${NSHARDS:-$NGPU}"

PUB="$DATA/published"
BANK="$DATA/bank"
export HF_HOME="${HF_HOME:-$DATA/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
# The IPC/large-batch paths want the legacy allocator, not expandable_segments.
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true

log() { printf '\n\033[1;36m[%(%H:%M:%S)T] %s\033[0m\n' -1 "$*"; }

# ---- 1. clone + install -----------------------------------------------------
REPO_DIR="$ROOT/EasyNLA"
if [ ! -d "$REPO_DIR/.git" ]; then
  log "cloning $REPO_URL @ $BRANCH -> $REPO_DIR"
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
  log "repo exists; fetching $BRANCH"
  git -C "$REPO_DIR" fetch origin "$BRANCH"
  git -C "$REPO_DIR" checkout "$BRANCH"
  git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
fi
cd "$REPO_DIR"
log "commit: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"

if ! python -c "import multilayer_nla" 2>/dev/null; then
  log "pip install -e . (+ bitsandbytes, hf_transfer)"
  pip install -q -e .
  pip install -q bitsandbytes hf_transfer || true
fi

# ---- 2. logins --------------------------------------------------------------
log "huggingface login"
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
  echo "HF_TOKEN unset — interactive login:"; huggingface-cli login
fi
log "wandb login"
if [ -n "${WANDB_API_KEY:-}" ]; then wandb login "$WANDB_API_KEY"; else wandb login || true; fi

# ---- 3. download published labels + schema echo -----------------------------
mkdir -p "$PUB" "$BANK"
log "downloading published labels from $SRC_DATASET -> $PUB"
SRC_DATASET="$SRC_DATASET" PUB="$PUB" SUBSETS="$SUBSETS" python - <<'PY'
import os
from datasets import load_dataset
src, pub = os.environ["SRC_DATASET"], os.environ["PUB"]
for name in os.environ["SUBSETS"].split():
    out = f"{pub}/{name}.parquet"
    if os.path.exists(out):
        print(f"  skip {name} (exists)"); continue
    ds = load_dataset(src, name, split="train")
    ds.to_parquet(out); print(f"  {name}: {ds.num_rows} rows -> {out}")
PY

log "schema echo (expect detokenized_text_truncated, n_raw_tokens, doc_id, prompt[, response]; NO activation_vector)"
PUB="$PUB" python - <<'PY'
import glob, pyarrow.parquet as pq, os
for f in sorted(glob.glob(f"{os.environ['PUB']}/*.parquet")):
    pf = pq.ParquetFile(f)
    print(f"  {os.path.basename(f)}: {pf.metadata.num_rows} rows | {pf.schema_arrow.names}")
PY

# ---- 4. PREFLIGHT: fail fast before the long extraction ---------------------
log "PREFLIGHT verify_regen_parity on 64 av_sft rows (bitwise final-token gather vs legacy)"
python -m multilayer_nla.verify_regen_parity \
    --base-model "$BASE_MODEL" --parquet "$PUB/av_sft.parquet" \
    --center-layer 24 --n-rows 64 --max-length "$MAX_LENGTH"

# storage estimate (no GPU) so you see the size before committing hours + upload
log "storage estimate (dry-run)"
python -m multilayer_nla.regenerate_bank --dry-run \
    --in "$PUB/av_sft.parquet" --out "$BANK/av_sft.parquet" --base-model "$BASE_MODEL" \
    --save-layers "$SAVE_LAYERS" --window "$WINDOW" --window-layers "$WINDOW_LAYERS" \
    --activation-dtype "$ACT_DTYPE" --window-dtype "$WIN_DTYPE"

# ---- 5. extract the bank ----------------------------------------------------
# av_sft + rl carry position windows (the AV input); ar_sft is final-token only
# (the AR only ever reconstructs the target at p).
extract_subset() {
  local subset="$1" window="$2" winlayers="$3"
  if compgen -G "$BANK/${subset}.parquet" >/dev/null || compgen -G "$BANK/${subset}.shard*of*.parquet" >/dev/null; then
    log "skip extract $subset (bank file exists)"; return
  fi
  local common=(--in "$PUB/${subset}.parquet" --out "$BANK/${subset}.parquet"
    --base-model "$BASE_MODEL" --save-layers "$SAVE_LAYERS"
    --activation-dtype "$ACT_DTYPE" --window-dtype "$WIN_DTYPE"
    --max-length "$MAX_LENGTH" --batch-size "$BATCH_SIZE"
    --length-bucket --max-drop-frac "$MAX_DROP_FRAC")
  if [ "$window" -ge 1 ]; then common+=(--window "$window" --window-layers "$winlayers"); else common+=(--window 0); fi

  if [ "$NSHARDS" -le 1 ]; then
    log "extract $subset (1 GPU)"
    python -m multilayer_nla.regenerate_bank "${common[@]}"
  else
    log "extract $subset across $NSHARDS GPUs"
    local pids=()
    for ((g=0; g<NSHARDS; g++)); do
      CUDA_VISIBLE_DEVICES="$g" python -m multilayer_nla.regenerate_bank \
        "${common[@]}" --num-shards "$NSHARDS" --shard-index "$g" \
        >"$BANK/${subset}.shard${g}.log" 2>&1 &
      pids+=($!)
    done
    local fail=0
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then fail=1; echo "  shard $i FAILED (see $BANK/${subset}.shard${i}.log)"; fi
    done
    [ "$fail" -eq 0 ] || { echo "extract $subset had a failed shard"; exit 1; }
  fi
}

for s in $SUBSETS; do
  case "$s" in
    ar_sft) extract_subset "$s" 0 "$WINDOW_LAYERS" ;;   # final-token only
    *)      extract_subset "$s" "$WINDOW" "$WINDOW_LAYERS" ;;
  esac
done

# ---- 6. verify each subset --------------------------------------------------
for s in $SUBSETS; do
  f="$BANK/${s}.parquet"; [ -f "$f" ] || f="$(ls "$BANK/${s}.shard"*of*.parquet 2>/dev/null | head -1)"
  [ -n "$f" ] && [ -f "$f" ] || { echo "no bank file for $s"; exit 1; }
  log "verify_bank $s ($(basename "$f"))"
  python -m multilayer_nla.verify_bank --bank "$f"
done

# ---- 7. upload the ENTIRE bank to a NEW HF dataset repo ---------------------
log "writing bank datacard"
HF_BANK_REPO="$HF_BANK_REPO" SRC_DATASET="$SRC_DATASET" BASE_MODEL="$BASE_MODEL" \
SAVE_LAYERS="$SAVE_LAYERS" WINDOW="$WINDOW" WINDOW_LAYERS="$WINDOW_LAYERS" \
ACT_DTYPE="$ACT_DTYPE" WIN_DTYPE="$WIN_DTYPE" COMMIT="$(git rev-parse --short HEAD)" \
BANK="$BANK" python - <<'PY'
import os
card = f"""# {os.environ['HF_BANK_REPO']}

Multi-layer x multi-position activation bank for EasyNLA `multilayer_nla`.

- **base model**: {os.environ['BASE_MODEL']}
- **labels/prefixes from**: {os.environ['SRC_DATASET']} (activations regenerated locally; no API)
- **final-token layers** (`activation_L{{k}}`): {os.environ['SAVE_LAYERS']} ({os.environ['ACT_DTYPE']})
- **position windows** (`window_L{{k}}`): layers {os.environ['WINDOW_LAYERS']}, W={os.environ['WINDOW']} ({os.environ['WIN_DTYPE']}); slot W-1 == labeled position p
- **repo commit**: {os.environ['COMMIT']} (branch multi_layer_nla)
- **norm**: none (RAW). Consume with `multilayer_nla.build_conditions` / `build_from_published`.

Subsets av_sft/ar_sft/rl are document-disjoint (the published split). ar_sft is
final-token only; av_sft/rl carry the position windows.
"""
open(f"{os.environ['BANK']}/README.md", "w").write(card)
print(card)
PY

log "creating HF dataset repo $HF_BANK_REPO (private)"
HF_BANK_REPO="$HF_BANK_REPO" python - <<'PY'
import os
from huggingface_hub import create_repo
create_repo(os.environ["HF_BANK_REPO"], repo_type="dataset", private=True, exist_ok=True)
print("repo ready")
PY

log "uploading $BANK -> $HF_BANK_REPO:$HF_BANK_PATH (this is the big one)"
huggingface-cli upload "$HF_BANK_REPO" "$BANK" "$HF_BANK_PATH" \
    --repo-type dataset --commit-message "multilayer bank: L$SAVE_LAYERS, W$WINDOW@$WINDOW_LAYERS"

log "DONE. Bank uploaded to https://huggingface.co/datasets/$HF_BANK_REPO/tree/main/$HF_BANK_PATH"
echo "Next (downstream, not in this script): multilayer_nla.splits -> build_conditions --conditions-preset permutation_grid -> train."
