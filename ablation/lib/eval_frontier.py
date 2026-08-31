"""Shared 100×100 binary-op frontier evaluation (all arms; mult and add).

Protocol:
  * grid: EVEN×EVEN cells (l1, l2) ∈ {2,4,..,l_max}² by default
    (``parity="all"`` restores the dense grid), evaluated in ascending
    (l1+l2, l1) order;
  * 10 samples per cell; per-item step cap from
    ``inference_caps.compute_max_steps`` (e.g. 30·l1·l2 for mult);
  * consecutive cells are packed into one merged ``batched_run_ablation``
    call of 100–200 items (5–10 cells);
  * frontier skip: cell (A, B) is recorded as 0/n_per without running when
    the set of already-decided cells {(a, b): a ≥ A−2, b ≥ B−2} is non-empty
    and every member has zero correct;
  * tasks per cell are seeded identically across arms, so every arm sees the
    same operand pairs;
  * streaming, resumable CSV: ``l1,l2,accuracy,n_total,n_correct,skipped``.

Variants:
  * ``plain``      exp 1 / baseline — success = final_done==1 AND full tape
                   == "a*b=answer" (full-tape match).
  * ``no_b``       exp 2 — (b) zeroed in the model's input states.
  * ``no_c``       exp 2 — (c) zeroed in input states and forced 0 in decoded
                   actions.
  * ``nocleanup``  exp 3 — no-cleanup engine, no can_dispatch guard, success
                   = final_done==1 AND full tape == the teacher's augmented
                   reference tape (computed per sample via ``dual_run``;
                   samples whose reference generation fails are re-drawn).
                   The answer sits in the reference tape interleaved with the
                   retained ghost segments, so plain string conventions like
                   "everything after the last '='" do NOT identify it — the
                   eval compares the whole tape verbatim.
"""

from __future__ import annotations

import csv
import os
import random
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import torch

from teacher import tokens as T
from teacher.engine import apply_action
from teacher.state import TeacherState

from supervised.lib.data_gen import _binary_op_strs, make_task_generator
from supervised.lib.inference_caps import OP_TO_TASK_KIND, compute_max_steps

from ablation.lib.engine_nocleanup import apply_action_nocleanup, dual_run
from ablation.lib.greedy_ablation import batched_run_ablation

CSV_FIELDS = ["l1", "l2", "accuracy", "n_total", "n_correct", "skipped"]

VARIANTS = {
    "plain": dict(engine_fn=apply_action, use_can_dispatch=True,
                  state_ctl_zero=(), action_ctl_zero=(), mode="full_tape"),
    "no_b": dict(engine_fn=apply_action, use_can_dispatch=True,
                 state_ctl_zero=(T.CTRL_RESULT,), action_ctl_zero=(),
                 mode="full_tape"),
    "no_c": dict(engine_fn=apply_action, use_can_dispatch=True,
                 state_ctl_zero=(T.CTRL_TAG_C,), action_ctl_zero=(T.CTRL_TAG_C,),
                 mode="full_tape"),
    "nocleanup": dict(engine_fn=apply_action_nocleanup, use_can_dispatch=False,
                      state_ctl_zero=(), action_ctl_zero=(), mode="reference"),
}


def tape_string(state: TeacherState) -> str:
    return T.decode([row[0] for row in state.tokens if row[0] != T.PAD])


def _cell_seed(seed: int, l1: int, l2: int) -> int:
    return seed * 1_000_003 + l1 * 1009 + l2


def _draw_cell_tasks(seed: int, l1: int, l2: int, n_per: int, op: str = "*"):
    """Same tasks for every arm at a given (seed, l1, l2, op)."""
    rnd_state = random.getstate()
    random.seed(_cell_seed(seed, l1, l2))
    tg = make_task_generator()
    pairs = [_binary_op_strs(tg, l1, l2, op) for _ in range(n_per)]
    random.setstate(rnd_state)
    return pairs


def _cell_cap(op: str, l1: int, l2: int) -> int:
    """Per-item step cap — same formulas as the release evals."""
    kind = OP_TO_TASK_KIND[op]
    return compute_max_steps(kind, total_input_length=l1 + l2 + 2, l1=l1, l2=l2)


def _gen_reference(job):
    """Pool worker: teacher augmented reference for one exp-3 sample."""
    task_str, max_steps = job
    _steps, final_ds = dual_run(task_str, max_steps)
    if final_ds is None:
        return None
    return tape_string(final_ds.abl)


def _load_existing(out_csv: str) -> Dict[Tuple[int, int], int]:
    decided: Dict[Tuple[int, int], int] = {}
    if os.path.isfile(out_csv):
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                decided[(int(row["l1"]), int(row["l2"]))] = int(row["n_correct"])
    return decided


def _frontier_skip(cell, decided: Dict[Tuple[int, int], int]) -> bool:
    A, B = cell
    nbrs = [v for (a, b), v in decided.items() if a >= A - 2 and b >= B - 2]
    return bool(nbrs) and all(v == 0 for v in nbrs)


def run_frontier_eval(
    agent,
    device: torch.device,
    pad_id: int,
    out_csv: str,
    variant: str,
    *,
    op: str = "*",
    l_max: int = 100,
    sum_max: int = 0,
    shard: Optional[Tuple[int, int]] = None,
    extra_decided: Tuple[str, ...] = (),
    n_per: int = 10,
    parity: str = "even",
    pack_min_items: int = 100,
    pack_max_cells: int = 20,
    seed: int = 123,
    ref_pool_workers: int = 12,
    log=print,
) -> None:
    cfg = VARIANTS[variant]
    decided = _load_existing(out_csv)
    if decided:
        log(f"resuming: {len(decided)} cells already in {out_csv}")
    # Read-only decided cells from other shards / the main CSV: excluded
    # from evaluation and fed to the skip rule, but never (re)written here.
    for p in extra_decided:
        extra = _load_existing(p)
        for k, v in extra.items():
            decided.setdefault(k, v)
        log(f"extra decided: +{len(extra)} cells from {p}")

    new_file = not os.path.isfile(out_csv)
    csv_f = open(out_csv, "a", newline="")
    writer = csv.writer(csv_f)
    if new_file:
        writer.writerow(CSV_FIELDS)
        csv_f.flush()

    def write_row(l1, l2, n_total, n_correct, skipped):
        acc = (n_correct / n_total) if n_total else 0.0
        writer.writerow([l1, l2, acc, n_total, n_correct, int(skipped)])
        csv_f.flush()
        decided[(l1, l2)] = n_correct

    if parity == "even":
        axis = range(2, l_max + 1, 2)
    elif parity == "all":
        axis = range(1, l_max + 1)
    else:
        raise ValueError(f"parity must be 'even' or 'all'; got {parity!r}")
    cells = sorted(
        ((l1, l2) for l1 in axis for l2 in axis
         if not sum_max or l1 + l2 <= sum_max),
        key=lambda c: (c[0] + c[1], c[0]),
    )
    if shard is not None:
        r, nsh = shard
        cells = [(l1, l2) for (l1, l2) in cells if (l1 * 31 + l2) % nsh == r]
        log(f"shard {r}/{nsh}: {len(cells)} cells in this partition")
    if sum_max:
        log(f"sum_max={sum_max}: cells beyond the diagonal are left "
            f"unevaluated (absent from the CSV, not recorded as 0)")

    ref_pool = Pool(ref_pool_workers) if cfg["mode"] == "reference" else None

    def flush(pack: List[Tuple[int, int]]):
        if not pack:
            return
        items: List[TeacherState] = []
        caps: List[int] = []
        tape_caps: List[int] = []
        md_caps: List[int] = []
        meta = []  # (cell, target_or_ref)
        for (l1, l2) in pack:
            pairs = _draw_cell_tasks(seed, l1, l2, n_per, op)
            cap = _cell_cap(op, l1, l2)
            if cfg["mode"] == "reference":
                refs = ref_pool.map(_gen_reference,
                                    [(t, cap) for t, _ in pairs])
                # Re-draw any sample whose reference generation failed.
                tries = 0
                while any(r is None for r in refs) and tries < 5:
                    tries += 1
                    retry_idx = [k for k, r in enumerate(refs) if r is None]
                    random.seed(_cell_seed(seed, l1, l2) + 7777 * tries)
                    tg = make_task_generator()
                    for k in retry_idx:
                        pairs[k] = _binary_op_strs(tg, l1, l2, op)
                    new_refs = ref_pool.map(
                        _gen_reference, [(pairs[k][0], cap) for k in retry_idx])
                    for k, r in zip(retry_idx, new_refs):
                        refs[k] = r
                expected = refs
            else:
                expected = [target for _, target in pairs]
            for (task_str, _target), exp in zip(pairs, expected):
                if exp is None:
                    continue  # unrecoverable reference failure: drop sample
                items.append(TeacherState.from_text(task_str, ctl_size=3))
                caps.append(cap)
                if cfg["mode"] == "reference":
                    tape_caps.append(max(20_000, 3 * len(exp)))
                    md_caps.append(tape_caps[-1])
                else:
                    tape_caps.append(max(20_000, 30 * (l1 + l2) * min(l1, l2)))
                    # Clean-engine mult keeps the visible slice ≲4·(l1+l2);
                    # a much larger slice means divergence — fail early
                    # instead of ballooning toward the encoder limit.
                    md_caps.append(max(256, 12 * (l1 + l2)))
                meta.append(((l1, l2), exp))

        final_states, success = batched_run_ablation(
            agent, items, device, pad_id, caps,
            engine_fn=cfg["engine_fn"],
            use_can_dispatch=cfg["use_can_dispatch"],
            state_ctl_zero=cfg["state_ctl_zero"],
            action_ctl_zero=cfg["action_ctl_zero"],
            tape_caps=tape_caps,
            md_caps=md_caps,
        )

        per_cell: Dict[Tuple[int, int], List[int]] = {c: [0, 0] for c in pack}
        for (cell, exp), st, ok in zip(meta, final_states, success):
            per_cell[cell][1] += 1
            if ok and tape_string(st) == exp:
                per_cell[cell][0] += 1
        for cell in pack:
            n_ok, n_tot = per_cell[cell]
            write_row(cell[0], cell[1], n_tot, n_ok, skipped=False)
            log(f"  cell {cell}: {n_ok}/{n_tot}")

    pack: List[Tuple[int, int]] = []
    try:
        for cell in cells:
            if cell in decided:
                continue
            if _frontier_skip(cell, decided):
                write_row(cell[0], cell[1], n_per, 0, skipped=True)
                continue
            pack.append(cell)
            if len(pack) * n_per >= pack_min_items or len(pack) >= pack_max_cells:
                flush(pack)
                pack = []
        flush(pack)
    finally:
        if ref_pool is not None:
            ref_pool.close()
            ref_pool.join()
        csv_f.close()
    log("frontier eval complete")
