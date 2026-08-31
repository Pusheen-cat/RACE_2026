"""Total-step estimator for cosine-LR scheduling.

Reads `teacher/stats/task_<id>.txt`, which has rows of the form

    length  accuracy  avg_steps  max_steps  visible_avg  visible_max  ...

and returns the per-length average teacher-trajectory step count plus the
total number of gradient steps that would result from a given length range
and `n_per_length`. The `task_<id>_grid.txt` variant carries the same
statistic on the (l1, l2) operand grid for the binary-op tasks.

CLI:

    python -m supervised.lib.util_steps --task copy \
        --lengths_min 1 --lengths_max 40 --n_per_length 250000 --batch_size 1000
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Iterable, Optional, Tuple

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TASK_IDS = {
    "copy": 0,
    "add": 10,
    "sub": 20,
    "mul": 30,
    "div": 40,
    "seq": 50,
    "seq_paren": 60,
}


def stats_path(task_id: int) -> str:
    return os.path.join(REPO_ROOT, "teacher", "stats", f"task_{task_id}.txt")


def avg_steps_per_length(task_id: int) -> Dict[int, float]:
    """Parse the per-task stats table and return ``{L: avg_steps}``."""
    path = stats_path(task_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"teacher stats table not found: {path}")
    out: Dict[int, float] = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            # Data rows: "<length> <acc> <avg_steps> <max_steps> ..."
            if len(parts) < 4:
                continue
            if not parts[0].isdigit():
                continue
            try:
                L = int(parts[0])
                avg = float(parts[2])
            except ValueError:
                continue
            out[L] = avg
    if not out:
        raise RuntimeError(f"no data rows parsed from {path}")
    return out


def estimate_total_pairs(
    task_id: int,
    lengths: Iterable[int],
    n_per_length: int,
) -> int:
    """Sum of `n_per_length * avg_steps[L]` over the supplied lengths.

    Any length missing from the stats table falls back to the largest table
    length's avg_steps (so an under-shoot is never silent — the pipeline
    always passes lengths inside the measured range).
    """
    table = avg_steps_per_length(task_id)
    if not table:
        return 0
    fallback_L = max(table)
    total = 0.0
    for L in lengths:
        avg = table.get(int(L), table[fallback_L])
        total += n_per_length * avg
    return int(math.ceil(total))


def estimate_total_steps(
    task_id: int,
    lengths: Iterable[int],
    n_per_length: int,
    batch_size: int,
) -> int:
    pairs = estimate_total_pairs(task_id, lengths, n_per_length)
    return max(1, math.ceil(pairs / max(1, batch_size)))


# ─── (l1, l2) grid variant — for binary ops ───────────────────────────────

def stats_grid_path(task_id: int) -> str:
    return os.path.join(REPO_ROOT, "teacher", "stats", f"task_{task_id}_grid.txt")


def avg_steps_grid(task_id: int) -> Dict[Tuple[int, int], float]:
    """Parse ``task_<id>_grid.txt`` and return ``{(l1, l2): avg_steps}``.

    The file's "# avg_steps" section is a table whose first column is
    ``l1\\l2`` followed by the integer column labels ``l2``; each
    subsequent data row starts with an integer ``l1`` and contains one
    cell value per column.
    """
    path = stats_grid_path(task_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"teacher stats grid not found: {path}")
    out: Dict[Tuple[int, int], float] = {}
    in_block = False
    cols: Optional[list] = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if stripped.startswith("# avg_steps"):
                in_block = True
                cols = None
                continue
            if not in_block:
                continue
            if stripped.startswith("#"):
                # Next section header (e.g. ``# max_steps``) — stop.
                break
            if not stripped:
                continue
            parts = stripped.split()
            if cols is None:
                # First non-blank line inside the block is the column header,
                # starting with ``l1\l2`` followed by integer column labels.
                if parts[0] in ("l1\\l2", r"l1\l2"):
                    try:
                        cols = [int(p) for p in parts[1:]]
                    except ValueError:
                        cols = None
                continue
            # Data row: l1 followed by len(cols) values.
            try:
                l1 = int(parts[0])
            except ValueError:
                continue
            for i, val_str in enumerate(parts[1:]):
                if i >= len(cols):
                    break
                try:
                    out[(l1, cols[i])] = float(val_str)
                except ValueError:
                    continue
    if not out:
        raise RuntimeError(f"no grid rows parsed from {path}")
    return out


def estimate_total_pairs_grid(
    task_id: int,
    l1_range: Iterable[int],
    l2_range: Iterable[int],
    n_per_combo: int,
    combos: Optional[Iterable[Tuple[int, int]]] = None,
) -> int:
    """Sum of ``n_per_combo * avg_steps[(l1, l2)]`` over the requested grid.

    ``combos`` overrides the Cartesian ``l1_range × l2_range`` with an
    explicit cell list (used by the pruned division grid). Cells outside
    the measured grid fall back to the largest cell's avg_steps (so we
    never silently under-estimate).
    """
    table = avg_steps_grid(task_id)
    if not table:
        return 0
    fallback = max(table.values())
    if combos is None:
        combos = [(l1, l2) for l1 in l1_range for l2 in l2_range]
    total = 0.0
    for l1, l2 in combos:
        avg = table.get((int(l1), int(l2)), fallback)
        total += n_per_combo * avg
    return int(math.ceil(total))


def estimate_total_steps_grid(
    task_id: int,
    l1_range: Iterable[int],
    l2_range: Iterable[int],
    n_per_combo: int,
    batch_size: int,
    combos: Optional[Iterable[Tuple[int, int]]] = None,
) -> int:
    pairs = estimate_total_pairs_grid(
        task_id, l1_range, l2_range, n_per_combo, combos=combos
    )
    return max(1, math.ceil(pairs / max(1, batch_size)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", type=int, default=None,
                   help="task id (0/10/20/30/40/50/60). Use --task instead "
                        "if you prefer the symbolic name.")
    p.add_argument("--task", default=None,
                   choices=list(TASK_IDS),
                   help="symbolic task name. Resolved to a task id.")
    p.add_argument("--lengths_min", type=int, default=1)
    p.add_argument("--lengths_max", type=int, default=40)
    p.add_argument("--n_per_length", type=int, default=250_000)
    p.add_argument("--batch_size", type=int, default=1000)
    args = p.parse_args()

    if args.task is not None:
        task_id = TASK_IDS[args.task]
    elif args.task_id is not None:
        task_id = args.task_id
    else:
        raise SystemExit("must pass --task or --task_id")

    table = avg_steps_per_length(task_id)
    print(f"task_id={task_id}  file={stats_path(task_id)}")
    lengths = list(range(args.lengths_min, args.lengths_max + 1))
    print(f"L,avg_steps")
    for L in lengths:
        avg = table.get(L)
        if avg is None:
            print(f"{L},MISSING")
        else:
            print(f"{L},{avg:.2f}")

    pairs = estimate_total_pairs(task_id, lengths, args.n_per_length)
    steps = estimate_total_steps(
        task_id, lengths, args.n_per_length, args.batch_size
    )
    print(f"# n_per_length={args.n_per_length}  batch_size={args.batch_size}")
    print(f"# total_pairs={pairs:,}")
    print(f"# total_grad_steps={steps:,}")


if __name__ == "__main__":
    main()
