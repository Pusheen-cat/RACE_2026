"""Canonical evaluation grids for the seven tasks.

Single source of truth: the per-task eval scripts and the multi-GPU launcher
all read their default grids from here.

Per task (training ranges in parentheses):

  * copy — lengths 1..500, 100 samples/length            (training: 1..40)
  * add / sub / mult — (l1, l2) ∈ (1..50)², 100 samples/cell
                                                          (training: {1..10}²)
  * div  — (l1, l2) ∈ (1..50)² restricted to the cells where the quotient
           can reach the smallest representable value: targets carry 3
           decimal digits, so only cases with a/b >= 0.01 are used.
           Cell-level this means l2 <= l1 + 2 (1 372 cells); sample-level
           the pair generator additionally redraws operands until
           100·a >= b, i.e. the truncated 3-decimal result is >= 0.010.
           (Training applies the same restriction on its {1..10}² grid —
           72 feasible cells; see train_binary_op.py.)
  * seq / nest — the factored-grid DIAGONAL n_numbers = max_digit_len,
           1..30, 100 samples/cell (training: full {2..10} × {1..10}
           grid). Evaluating the full (n, d) product grid is too
           expensive — trajectory length grows with both axes — so the
           diagonal scales them together. The n = 1 cell is degenerate
           (a bare number) but well-defined and kept.
"""

from __future__ import annotations

from typing import List, Tuple

# Division only uses cases whose result is >= DIV_MIN_RESULT (0.01), in
# training and evaluation alike; the constant lives with the generators.
from supervised.lib.data_gen import DIV_MIN_RESULT  # noqa: F401  (re-export)


def copy_lengths() -> List[int]:
    """Copy eval lengths: 1..500."""
    return list(range(1, 501))


COPY_N_PER = 100


def binary_op_grid(op: str) -> List[Tuple[List[Tuple[int, int]], int]]:
    """Eval grid for one binary op, as ``[(cells, n_per_cell), ...]`` groups.

    All four ops use (l1, l2) ∈ (1..50)² at 100 samples/cell; division keeps
    only the cells where the result-≥-0.01 condition is satisfiable
    (l2 <= l1 + 2).
    """
    if op in ("+", "-", "*"):
        cells = [(l1, l2) for l1 in range(1, 51) for l2 in range(1, 51)]
        return [(cells, 100)]
    if op == "/":
        cells = [
            (l1, l2)
            for l1 in range(1, 51)
            for l2 in range(1, 51)
            if l2 <= l1 + 2
        ]
        return [(cells, 100)]
    raise ValueError(f"unknown op {op!r}")


def seq_grid() -> List[Tuple[int, int]]:
    """Factored cells for seq / nest: the diagonal n_numbers =
    max_digit_len ∈ 1..30 (not a product grid — the full grid search is
    too expensive, so both difficulty axes are scaled together)."""
    return [(n, n) for n in range(1, 31)]


SEQ_N_PER = 100
SEQ_MAX_STEPS = 200_000
