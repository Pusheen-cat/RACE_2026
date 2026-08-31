#!/usr/bin/env bash
# Length-generalization evaluation for all seven tasks on their canonical
# grids (see supervised/lib/eval_grids.py). Each task's grid is sharded
# across the given GPUs — one worker subprocess per GPU — and the part CSVs
# are merged into results/<task>/accuracy.csv.
#
# Run from the release root:
#
#   bash scripts/eval_all.sh <gpus> [tasks]
#
#   bash scripts/eval_all.sh 0,1,2,3               # all seven tasks
#   bash scripts/eval_all.sh 4,5 "div seq nest"    # a subset
#
# Each task is evaluated with its own stage checkpoint (checkpoints/<task>.pt),
# matching the training curriculum: the model is evaluated on a task as of the
# end of that task's training stage. Workers stream one CSV row per finished
# cell and skip already-written cells, so rerunning the same command resumes
# an interrupted evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${1:?usage: bash scripts/eval_all.sh <gpus, e.g. 0,1,2,3> [tasks]}"
TASKS="${2:-copy add mult sub div seq nest}"

run_eval() {
  local name="$1"; shift
  mkdir -p "results/$name"
  echo "=== eval $name (GPUs $GPUS) ==="
  python -u "$@" 2>&1 | tee "results/$name/eval.log"
}

for task in $TASKS; do
  case "$task" in
    copy)
      run_eval copy -m supervised.eval.eval_copy --fanout --gpus "$GPUS" \
        --ckpt checkpoints/copy.pt --out results/copy/accuracy.csv
      ;;
    add)
      run_eval add -m supervised.eval.eval_binary_op --fanout --gpus "$GPUS" \
        --op + --ckpt checkpoints/add.pt --out results/add/accuracy.csv
      ;;
    mult)
      run_eval mult -m supervised.eval.eval_binary_op --fanout --gpus "$GPUS" \
        --op '*' --ckpt checkpoints/mult.pt --out results/mult/accuracy.csv
      ;;
    sub)
      run_eval sub -m supervised.eval.eval_binary_op --fanout --gpus "$GPUS" \
        --op - --ckpt checkpoints/sub.pt --out results/sub/accuracy.csv
      ;;
    div)
      run_eval div -m supervised.eval.eval_binary_op --fanout --gpus "$GPUS" \
        --op / --ckpt checkpoints/div.pt --out results/div/accuracy.csv
      ;;
    seq)
      run_eval seq -m supervised.eval.eval_sequential --fanout --gpus "$GPUS" \
        --task_kind seq --ckpt checkpoints/seq.pt --out results/seq/accuracy.csv
      ;;
    nest)
      run_eval nest -m supervised.eval.eval_sequential --fanout --gpus "$GPUS" \
        --task_kind seq_paren --ckpt checkpoints/nest.pt --out results/nest/accuracy.csv
      ;;
    *)
      echo "unknown task: $task" >&2; exit 1
      ;;
  esac
done

echo "=== evaluation complete: results/<task>/accuracy.csv ==="
