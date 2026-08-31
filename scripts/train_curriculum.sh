#!/usr/bin/env bash
# Sequential seven-stage curriculum:
#
#   copy -> add -> mult -> sub -> div -> seq -> nest
#
# One model is trained continuously; every stage initializes from the
# previous stage's checkpoint. Checkpoints land in checkpoints/<task>.pt,
# training logs in results/<task>/train.log.
#
# Run from the release root on one GPU:
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_curriculum.sh [batch_size] [num_workers] [start_stage]
#
# batch_size defaults to 2000 (bf16; fits every stage on an 80GB+ GPU —
# reduce it on smaller cards), num_workers to 16 (CPU trajectory-generation
# workers; training is data-bound, so more CPU workers help). Pass a stage
# name as the third argument to resume mid-curriculum, e.g.
#   bash scripts/train_curriculum.sh 2000 16 sub
set -euo pipefail
cd "$(dirname "$0")/.."

BATCH="${1:-2000}"
WORKERS="${2:-16}"
START="${3:-copy}"

STAGES=(copy add mult sub div seq nest)

stage_index() {
  local s
  for i in "${!STAGES[@]}"; do
    s="${STAGES[$i]}"
    if [[ "$s" == "$1" ]]; then echo "$i"; return; fi
  done
  echo "unknown stage: $1" >&2; exit 1
}
START_IDX="$(stage_index "$START")"

run_stage() {
  local name="$1"; shift
  mkdir -p "results/$name"
  echo "=== stage $name ==="
  python -u "$@" 2>&1 | tee "results/$name/train.log"
}

for i in "${!STAGES[@]}"; do
  [[ "$i" -lt "$START_IDX" ]] && continue
  case "${STAGES[$i]}" in
    copy)
      run_stage copy -m supervised.train.train_copy \
        --batch_size "$BATCH" --num_workers "$WORKERS" \
        --out_name copy.pt
      ;;
    add)
      run_stage add -m supervised.train.train_binary_op --op + \
        --init_ckpt checkpoints/copy.pt --out_name add.pt \
        --n_per_combo 100000 \
        --batch_size "$BATCH" --num_workers "$WORKERS"
      ;;
    mult)
      run_stage mult -m supervised.train.train_binary_op --op '*' \
        --init_ckpt checkpoints/add.pt --out_name mult.pt \
        --n_per_combo 10000 \
        --batch_size "$BATCH" --num_workers "$WORKERS"
      ;;
    sub)
      run_stage sub -m supervised.train.train_binary_op --op - \
        --init_ckpt checkpoints/mult.pt --out_name sub.pt \
        --n_per_combo 10000 \
        --batch_size "$BATCH" --num_workers "$WORKERS"
      ;;
    div)
      run_stage div -m supervised.train.train_binary_op --op / \
        --init_ckpt checkpoints/sub.pt --out_name div.pt \
        --n_per_combo 10000 \
        --batch_size "$BATCH" --num_workers "$WORKERS"
      ;;
    seq)
      run_stage seq -m supervised.train.train_sequential --task_kind seq \
        --init_ckpt checkpoints/div.pt --out_name seq.pt \
        --n_per_combo 10000 \
        --batch_size "$BATCH" --num_workers "$WORKERS"
      ;;
    nest)
      run_stage nest -m supervised.train.train_sequential --task_kind seq_paren \
        --init_ckpt checkpoints/seq.pt --out_name nest.pt \
        --n_per_combo 10000 \
        --batch_size "$BATCH" --num_workers "$WORKERS"
      ;;
  esac
done

echo "=== curriculum complete: checkpoints/{copy,add,mult,sub,div,seq,nest}.pt ==="
