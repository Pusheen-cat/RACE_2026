"""Ablation-study data generators.

Exp 2 — ``masked_binary_op_specs`` / ``masked_copy_specs``: same teacher
trajectories as the release generators but with control bit (b) or (c)
zeroed at the end of data prep (see ``ctl_mask``).

Exp 3 — ``nocleanup_binary_op_specs`` / ``nocleanup_copy_specs``: the
trajectory is *generated* under the normal (cleanup) semantics, but the
recorded per-step states/actions come from the augmented tape where the
done-stage auto-eq cleanup never deletes anything (see ``engine_nocleanup``).
"""

from __future__ import annotations

from typing import List

from supervised.lib.data_gen import (
    _binary_op_strs, gen_binary_op_task, gen_copy_task,
)
from supervised.lib.dataset import TaskSpec

from ablation.lib.ctl_mask import mask_steps
from ablation.lib.engine_nocleanup import collect_steps_nocleanup


# ─── Exp 2: control-element masking ─────────────────────────────────────────

def gen_masked_binary_op_task(tg, l1: int, l2: int, op: str, drop: str,
                              max_steps: int = 200_000):
    task_str, target_str, steps = gen_binary_op_task(tg, l1, l2, op,
                                                     max_steps=max_steps)
    return task_str, target_str, mask_steps(steps, drop)


def gen_masked_copy_task(tg, length: int, drop: str, max_steps: int = 4000):
    task_str, target_str, steps = gen_copy_task(tg, length, max_steps=max_steps)
    if steps is None:
        return task_str, target_str, None
    return task_str, target_str, mask_steps(steps, drop)


def masked_copy_specs(
    lengths,
    n_tasks_per_length: int,
    drop: str,
) -> List[TaskSpec]:
    specs = []
    for L in lengths:
        L_ = int(L)
        specs.append(TaskSpec(
            name=f"copy/no{drop}/L{L_}",
            n_tasks=n_tasks_per_length,
            gen_fn=lambda tg, L_=L_, drop=drop:
                gen_masked_copy_task(tg, L_, drop),
        ))
    return specs


def masked_binary_op_specs(
    op: str,
    l1_range,
    l2_range,
    n_tasks_per_combo: int,
    drop: str,
) -> List[TaskSpec]:
    specs = []
    for l1 in l1_range:
        for l2 in l2_range:
            l1_, l2_ = int(l1), int(l2)
            specs.append(TaskSpec(
                name=f"{op}/no{drop}/L{l1_}-L{l2_}",
                n_tasks=n_tasks_per_combo,
                gen_fn=lambda tg, l1_=l1_, l2_=l2_, op=op, drop=drop:
                    gen_masked_binary_op_task(tg, l1_, l2_, op, drop),
            ))
    return specs


# ─── Exp 3: no-cleanup augmented trajectories ────────────────────────────────

def gen_nocleanup_binary_op_task(tg, l1: int, l2: int, op: str = "*",
                                 max_steps: int = 200_000):
    """`a <op> b =` with explicit operand lengths; steps recorded on the
    augmented (no-cleanup) tape. Op-generic — the dual engine dispatches
    whatever teacher rule matches, so any binary op works."""
    task_str, target_str = _binary_op_strs(tg, l1, l2, op)
    return task_str, target_str, collect_steps_nocleanup(task_str, max_steps)


def nocleanup_binary_op_specs(
    l1_range,
    l2_range,
    n_tasks_per_combo: int,
    op: str = "*",
) -> List[TaskSpec]:
    specs = []
    for l1 in l1_range:
        for l2 in l2_range:
            l1_, l2_ = int(l1), int(l2)
            specs.append(TaskSpec(
                name=f"{op}/nocleanup/L{l1_}-L{l2_}",
                n_tasks=n_tasks_per_combo,
                gen_fn=lambda tg, l1_=l1_, l2_=l2_, op=op:
                    gen_nocleanup_binary_op_task(tg, l1_, l2_, op),
            ))
    return specs


def gen_nocleanup_copy_task(tg, length: int, max_steps: int = 20_000):
    """Copy task through the dual (no-cleanup) engine. NOTE: copy tasks
    carry their '=' in the task string, so auto-append — and therefore the
    done-stage cleanup — never fires; the augmented steps are verified
    bit-identical to the clean ones. Kept for protocol symmetry (the
    edit_no_cleanup copy arm is a baseline-equivalent control)."""
    task_str, target_str = tg.gen_task_1_copy(length)
    return task_str, target_str, collect_steps_nocleanup(task_str, max_steps)


def nocleanup_copy_specs(
    lengths,
    n_tasks_per_length: int,
) -> List[TaskSpec]:
    specs = []
    for L in lengths:
        L_ = int(L)
        specs.append(TaskSpec(
            name=f"copy/nocleanup/L{L_}",
            n_tasks=n_tasks_per_length,
            gen_fn=lambda tg, L_=L_: gen_nocleanup_copy_task(tg, L_),
        ))
    return specs
