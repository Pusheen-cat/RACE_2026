#!/usr/bin/env bash
# Train all stages of ONE ablation arm sequentially on one GPU, chained in
# the main-curriculum order:
#
#   copy (from scratch) -> add (from the arm's copy.pt) -> mult (from the
#   arm's add.pt)
#
# The copy stage is skipped for edit_no_cleanup: on copy the no-cleanup
# pipeline is bit-identical to the clean one (copy tasks carry '=' in the
# task string, so the done-stage cleanup never fires). Its add stage
# therefore initializes from the MAIN curriculum's checkpoints/copy.pt —
# the equivalent checkpoint — which must exist (train the main curriculum's
# copy stage first). All other arms are fully self-contained.
#
# Run from the release root:
#
#   CUDA_VISIBLE_DEVICES=0 bash ablation/scripts/train_arm.sh <arm> \
#       [batch_size] [num_workers] [start_stage]
#
#   arm         one of: pe_noinv pe_nope pe_rope ctl_no_b ctl_no_c edit_no_cleanup
#   batch_size  default 2000 (bf16; reduce on smaller GPUs — the cosine
#               schedule adapts via the step-count estimate)
#   num_workers CPU trajectory-generation workers, default 16
#   start_stage copy | add | mult — resume mid-arm, default copy
#
# Per-stage task counts follow the main curriculum: copy 250 k/length
# (10 M), add 100 k/cell (10 M — the long stage), mult 10 k/cell (1 M).
# Checkpoints land in ablation/checkpoints/<arm>/<stage>.pt, logs in
# ablation/results/<arm>/<stage>/train.log.
#
# NOTE (edit_no_cleanup): its binary-op stages use length-bucketed batching
# whose --mem_K constant is calibrated for ~90 GB GPUs; on smaller cards
# scale it down proportionally, e.g. append `--mem_K 1.4e7` for ~46 GB by
# running the module directly (see ablation/train/train_binary_op_ablation.py).
set -euo pipefail
cd "$(dirname "$0")/../.."

ARM="${1:?usage: bash ablation/scripts/train_arm.sh <arm> [batch] [workers] [start_stage]}"
BATCH="${2:-2000}"
WORKERS="${3:-16}"
START="${4:-copy}"

case "$ARM" in
  pe_noinv|pe_nope|pe_rope|ctl_no_b|ctl_no_c|edit_no_cleanup) ;;
  *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac

STAGES=(copy add mult)
stage_index() {
  for i in "${!STAGES[@]}"; do
    if [[ "${STAGES[$i]}" == "$1" ]]; then echo "$i"; return; fi
  done
  echo "unknown stage: $1" >&2; exit 1
}
START_IDX="$(stage_index "$START")"

run_stage() {
  local name="$1"; shift
  mkdir -p "ablation/results/$ARM/$name"
  echo "=== arm $ARM stage $name ==="
  python -u "$@" 2>&1 | tee "ablation/results/$ARM/$name/train.log"
}

for i in "${!STAGES[@]}"; do
  [[ "$i" -lt "$START_IDX" ]] && continue
  case "${STAGES[$i]}" in
    copy)
      if [[ "$ARM" == "edit_no_cleanup" ]]; then
        echo "=== arm $ARM stage copy: SKIPPED (bit-identical to the baseline on copy) ==="
        continue
      fi
      run_stage copy -m ablation.train.train_copy_ablation --arm "$ARM" \
        --batch_size "$BATCH" --num_workers "$WORKERS"
      ;;
    add)
      if [[ "$ARM" == "edit_no_cleanup" && ! -f checkpoints/copy.pt ]]; then
        echo "checkpoints/copy.pt not found — edit_no_cleanup's add stage" >&2
        echo "initializes from the main curriculum's copy checkpoint; train" >&2
        echo "that stage first (scripts/train_curriculum.sh)." >&2
        exit 1
      fi
      run_stage add -m ablation.train.train_binary_op_ablation --op + \
        --arm "$ARM" --batch_size "$BATCH" --num_workers "$WORKERS"
      ;;
    mult)
      run_stage mult -m ablation.train.train_binary_op_ablation --op '*' \
        --arm "$ARM" --batch_size "$BATCH" --num_workers "$WORKERS"
      ;;
  esac
done

echo "=== arm $ARM training complete: ablation/checkpoints/$ARM/{copy,add,mult}.pt ==="
