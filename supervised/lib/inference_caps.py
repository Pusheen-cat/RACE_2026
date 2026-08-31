"""Per-task greedy-inference step caps.

Single source of truth for the per-trajectory `max_steps` budget used by
`greedy_inference.batched_run`. Each task has its own formula sized to
just barely cover a correct trajectory; on failure the cap fires quickly
and frees the CPU/GPU for the next sample.

Modify the formulas here — never inline them at call sites.
"""

from __future__ import annotations


def compute_max_steps(
    task_kind: str,
    *,
    total_input_length: int = 0,
    l1: int = 0,
    l2: int = 0,
) -> int:
    """Return the per-trajectory greedy-inference step cap for ``task_kind``.

    Supported kinds (and the parameters their formula reads):
      * ``"copy"``           — ``total_input_length``
      * ``"addition"``       — ``total_input_length``
      * ``"subtraction"``    — ``total_input_length``
      * ``"multiplication"`` — ``l1``, ``l2``
      * ``"division"``       — ``l1``, ``l2``

    The sequential / nested evals use a flat global cap instead (their
    trajectory length depends on the sampled operator mix, not just the
    grid cell). Other kinds raise ``NotImplementedError`` so a new task
    cannot silently fall back to an unrelated cap; add the formula here
    when that task starts being evaluated.
    """
    if task_kind == "copy":
        return total_input_length + 5
    if task_kind == "addition":
        return total_input_length * 5
    if task_kind == "subtraction":
        return total_input_length * 7
    if task_kind == "multiplication":
        return 30 * l1 * l2
    if task_kind == "division":
        return l1 * l1 * 23 + 220 * l1 + l2 * 30 + 600
    raise NotImplementedError(
        f"compute_max_steps: no formula for task_kind={task_kind!r}. "
        f"Add one to supervised/lib/inference_caps.py."
    )


OP_TO_TASK_KIND = {
    "+": "addition",
    "-": "subtraction",
    "*": "multiplication",
    "/": "division",
}
