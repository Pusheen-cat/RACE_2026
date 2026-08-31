#!/usr/bin/env bash
# Evaluate ONE ablation arm (or the baseline) on all three tasks:
#
#   copy  — lengths 1..500, 100 samples/length
#   add   — 100×100 frontier eval (even×even grid, 10 samples/cell)
#   mult  — same frontier protocol, op '*'
#
# The three task evals are distributed round-robin over the given GPU list
# and run concurrently (one process per task). Every eval is streaming and
# resumable — rerun the same command to continue after an interruption. For
# finer-grained parallelism WITHIN one frontier eval, launch
# ablation.eval.eval_binary_op_ablation manually with --shard r/n (see its
# docstring).
#
# Run from the release root:
#
#   bash ablation/scripts/eval_arm.sh <arm> <gpus>
#
#   bash ablation/scripts/eval_arm.sh pe_noinv 0,1,2    # one GPU per task
#   bash ablation/scripts/eval_arm.sh baseline 4        # all tasks on GPU 4
#
#   arm   one of: pe_noinv pe_nope pe_rope ctl_no_b ctl_no_c
#         edit_no_cleanup baseline
#
# `baseline` evaluates the MAIN curriculum's checkpoints/{copy,add,mult}.pt
# under the identical protocol (same seeded tasks) for direct comparability.
# The copy eval is skipped for edit_no_cleanup (no copy arm — see
# train_arm.sh). Results land in ablation/results/<arm>/<task>/accuracy.csv,
# logs next to them in eval.log.
set -uo pipefail
cd "$(dirname "$0")/../.."

ARM="${1:?usage: bash ablation/scripts/eval_arm.sh <arm> <gpus, e.g. 0,1,2>}"
GPUS_CSV="${2:?usage: bash ablation/scripts/eval_arm.sh <arm> <gpus, e.g. 0,1,2>}"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

case "$ARM" in
  pe_noinv|pe_nope|pe_rope|ctl_no_b|ctl_no_c|edit_no_cleanup|baseline) ;;
  *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac

TASKS=(copy add mult)
if [[ "$ARM" == "edit_no_cleanup" ]]; then
  TASKS=(add mult)
  echo "arm $ARM: copy eval skipped (no copy arm — baseline-equivalent on copy)"
fi

PIDS=()
NAMES=()
k=0
for task in "${TASKS[@]}"; do
  GPU="${GPUS[$((k % ${#GPUS[@]}))]}"
  k=$((k + 1))
  mkdir -p "ablation/results/$ARM/$task"
  LOG="ablation/results/$ARM/$task/eval.log"
  echo "=== eval $ARM/$task on GPU $GPU (log: $LOG) ==="
  case "$task" in
    copy)
      CUDA_VISIBLE_DEVICES="$GPU" python -u -m ablation.eval.eval_copy_ablation \
        --arm "$ARM" >> "$LOG" 2>&1 &
      ;;
    mult)
      CUDA_VISIBLE_DEVICES="$GPU" python -u -m ablation.eval.eval_binary_op_ablation \
        --arm "$ARM" --op '*' >> "$LOG" 2>&1 &
      ;;
    add)
      CUDA_VISIBLE_DEVICES="$GPU" python -u -m ablation.eval.eval_binary_op_ablation \
        --arm "$ARM" --op + >> "$LOG" 2>&1 &
      ;;
  esac
  PIDS+=($!)
  NAMES+=("$task")
done

RC=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "=== eval $ARM/${NAMES[$i]} finished ==="
  else
    echo "=== eval $ARM/${NAMES[$i]} FAILED (see ablation/results/$ARM/${NAMES[$i]}/eval.log) ===" >&2
    RC=1
  fi
done

[[ "$RC" -eq 0 ]] && echo "=== arm $ARM evaluation complete: ablation/results/$ARM/<task>/accuracy.csv ==="
exit "$RC"
