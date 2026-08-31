"""Streaming supervised dataset for the teacher-imitation pipeline.

Each "training example" is one *step* of a teacher trajectory:

    (state_tokens (S, 1+ctl), state_depth (S,), action (S_eff+1, M, 1+ctl), done)

Trajectories are generated on the fly by `data_gen.py`; the corpus is never
materialised, since the largest task families (multiplication, sequential)
expand into billions of steps.

The dataset accepts a *bag* of `(gen_callable, n_tasks)` entries. A worker
shuffles its slice of the bag, iterates through it generating one task at a
time, runs the teacher rules to produce per-step tuples, and emits them via
a per-worker shuffle buffer so consecutive items are not from the same
trajectory.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info

from supervised.lib.data_gen import make_task_generator


@dataclass
class TaskSpec:
    """A self-contained recipe for generating ONE teacher task.

    `gen_fn(tg) -> (task_str, target_str, steps)`. `steps` is the result of
    `data_gen._collect_steps(task_str, ...)` — None if the trajectory failed
    to terminate (rare; that task is dropped silently).
    """
    name: str
    n_tasks: int
    gen_fn: Callable


def _split_for_worker(specs: Sequence[TaskSpec], worker_id: int, num_workers: int):
    """Split each spec's `n_tasks` count across workers.

    Each worker gets a roughly-equal share of every spec. Uneven leftover
    tasks (n_tasks % num_workers) are distributed to the first `r` workers.
    """
    out: List[TaskSpec] = []
    for spec in specs:
        base = spec.n_tasks // num_workers
        extra = 1 if worker_id < (spec.n_tasks % num_workers) else 0
        n = base + extra
        if n > 0:
            out.append(TaskSpec(name=spec.name, n_tasks=n, gen_fn=spec.gen_fn))
    return out


class TeacherStreamDataset(IterableDataset):
    def __init__(
        self,
        specs: Sequence[TaskSpec],
        shuffle_buffer: int = 4000,
        seed: int = 0,
        curriculum: bool = False,
    ):
        super().__init__()
        self.specs = list(specs)
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        # `curriculum=True`: the dataset preserves the spec list's order
        # (assumed to be ascending in difficulty / length) and yields each
        # task's steps in trajectory order, no shuffle buffer. Each worker
        # marches through the same ascending schedule, so the global stream
        # progresses short→long with only the natural cross-worker
        # interleaving that DataLoader introduces.
        self.curriculum = curriculum

    def __iter__(self) -> Iterator[Tuple[list, list, list, int]]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1

        rng = random.Random(self.seed * 1_000_003 + worker_id)
        my_specs = _split_for_worker(self.specs, worker_id, num_workers)
        schedule: List[int] = []
        for i, spec in enumerate(my_specs):
            schedule.extend([i] * spec.n_tasks)
        if not self.curriculum:
            rng.shuffle(schedule)

        tg = make_task_generator()

        if self.curriculum:
            for spec_idx in schedule:
                spec = my_specs[spec_idx]
                try:
                    _task_str, _target_str, steps = spec.gen_fn(tg)
                except Exception:
                    continue
                if steps is None:
                    continue
                for step in steps:
                    yield step
            return

        # Steps from the most recent few trajectories live in this buffer; we
        # emit a random one at each draw so consecutive yielded items come
        # from different trajectories most of the time.
        buf: List[Tuple[list, list, list, int]] = []

        for spec_idx in schedule:
            spec = my_specs[spec_idx]
            try:
                _task_str, _target_str, steps = spec.gen_fn(tg)
            except Exception:
                continue
            if steps is None:
                continue
            for step in steps:
                buf.append(step)
                if len(buf) >= self.shuffle_buffer:
                    pop = rng.randrange(len(buf))
                    yield buf[pop]
                    buf[pop] = buf[-1]
                    buf.pop()
        # Drain.
        rng.shuffle(buf)
        for item in buf:
            yield item


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def collate_steps(batch, pad_id: int):
    """Pad a list of step-tuples into batched tensors.

    state and depth pad along the S_total axis; action pads along the
    S_eff+1 axis. M and (1+ctl) are constant across teacher rules.
    """
    states_list, depths_list, actions_list, done_list = [], [], [], []
    max_S, max_Sa = 0, 0
    M = None
    C = None
    for step in batch:
        toks, dep, act, done = step
        states_list.append(toks)
        depths_list.append(dep)
        actions_list.append(act)
        done_list.append(done)
        max_S = max(max_S, len(toks))
        max_Sa = max(max_Sa, len(act))
        if M is None:
            M = len(act[0])
            C = len(act[0][0])

    B = len(batch)
    states = torch.full((B, max_S, C), 0, dtype=torch.long)
    states[..., 0] = pad_id
    depths = torch.zeros((B, max_S), dtype=torch.long)
    actions = torch.full((B, max_Sa, M, C), 0, dtype=torch.long)
    actions[..., 0] = pad_id
    dones = torch.tensor(done_list, dtype=torch.long)

    for i in range(B):
        toks = states_list[i]
        dep = depths_list[i]
        S_i = len(toks)
        states[i, :S_i] = torch.tensor(toks, dtype=torch.long)
        depths[i, :S_i] = torch.tensor(dep, dtype=torch.long)

        act = actions_list[i]
        Sa_i = len(act)
        actions[i, :Sa_i] = torch.tensor(act, dtype=torch.long)

    return states, depths, actions, dones


def collate_factory(pad_id: int):
    def _collate(batch):
        return collate_steps(batch, pad_id=pad_id)
    return _collate


# ─────────────────────────────────────────────────────────────────────────────
# Spec builders
# ─────────────────────────────────────────────────────────────────────────────

def copy_specs(lengths: Sequence[int], n_tasks_per_length: int) -> List[TaskSpec]:
    from supervised.lib.data_gen import gen_copy_task
    specs = []
    for L in lengths:
        L_ = int(L)
        specs.append(TaskSpec(
            name=f"copy/L{L_}",
            n_tasks=n_tasks_per_length,
            gen_fn=lambda tg, L_=L_: gen_copy_task(tg, L_),
        ))
    return specs


def binary_op_specs(
    op: str,
    l1_range: Sequence[int],
    l2_range: Sequence[int],
    n_tasks_per_combo: int,
    combos: Sequence[Tuple[int, int]] = None,
    min_div_result: float = None,
) -> List[TaskSpec]:
    """One spec per (l1, l2) cell.

    ``combos`` overrides the Cartesian ``l1_range × l2_range`` with an
    explicit cell list (the division stage prunes infeasible cells).
    ``min_div_result`` is forwarded to the generator — the division stage
    passes ``data_gen.DIV_MIN_RESULT`` so every drawn task has a result of
    at least 0.01.
    """
    from supervised.lib.data_gen import gen_binary_op_task
    if combos is None:
        combos = [(l1, l2) for l1 in l1_range for l2 in l2_range]
    specs = []
    for l1, l2 in combos:
        l1_, l2_ = int(l1), int(l2)
        specs.append(TaskSpec(
            name=f"{op}/L{l1_}-L{l2_}",
            n_tasks=n_tasks_per_combo,
            gen_fn=lambda tg, l1_=l1_, l2_=l2_, op=op, mdr=min_div_result:
                gen_binary_op_task(tg, l1_, l2_, op, min_div_result=mdr),
        ))
    return specs


def seq_factored_specs(
    combos: Sequence[Tuple[int, int]], n_tasks_per_combo: int,
) -> List[TaskSpec]:
    """One spec per (n_numbers, max_digit_len) cell of the seq factored grid."""
    from supervised.lib.data_gen import gen_seq_factored_task
    specs = []
    for n_, d_ in combos:
        n_, d_ = int(n_), int(d_)
        specs.append(TaskSpec(
            name=f"seq_factored/N{n_}-D{d_}",
            n_tasks=n_tasks_per_combo,
            gen_fn=lambda tg, n_=n_, d_=d_: gen_seq_factored_task(tg, n_, d_),
        ))
    return specs


def seq_paren_factored_specs(
    combos: Sequence[Tuple[int, int]], n_tasks_per_combo: int,
) -> List[TaskSpec]:
    """One spec per (n_numbers, max_digit_len) cell of the nested factored grid."""
    from supervised.lib.data_gen import gen_seq_paren_factored_task
    specs = []
    for n_, d_ in combos:
        n_, d_ = int(n_), int(d_)
        specs.append(TaskSpec(
            name=f"seq_paren_factored/N{n_}-D{d_}",
            n_tasks=n_tasks_per_combo,
            gen_fn=lambda tg, n_=n_, d_=d_: gen_seq_paren_factored_task(tg, n_, d_),
        ))
    return specs


def total_tasks(specs: Sequence[TaskSpec]) -> int:
    return sum(s.n_tasks for s in specs)
