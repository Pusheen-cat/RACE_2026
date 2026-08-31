"""Exp 3 — edit-base ablation: the done-stage auto-eq cleanup never deletes.

Design: trajectory *generation* runs under the normal (cleanup) semantics —
the teacher rules are Markov functions of the clean visible tape and would
misbehave on a tape with leftovers. In parallel we maintain an **augmented
state** in which the cleanup branch is replaced by the standard return
(clear ctrl, set (b), depth−1) for every max-depth cell, so the sub-call
operands and the auto-appended `=` ride along as *ghost* cells. The model's
training labels are the augmented states plus **augmented actions** = the
clean action rows at the mapped positions + `keep` rows at ghost positions.

Key objects:

  * ``apply_action_traced(state, action, cleanup)`` — re-implementation of
    ``teacher.engine.apply_action`` that additionally returns per-output-cell
    provenance ``(src_index_in_old_state, slot)`` and takes a ``cleanup``
    switch. With ``cleanup=True`` it is behaviourally identical to the
    original (asserted by the verification gate).
  * ``apply_action_nocleanup(state, action)`` — the inference-side engine
    (``cleanup=False``, provenance discarded).
  * ``DualState`` + ``dual_auto_append`` + ``dual_step`` — lock-step
    evolution of (clean, augmented) states with an index mapping.
  * ``collect_steps_nocleanup(task_str, max_steps)`` — drop-in analogue of
    ``supervised.lib.data_gen._collect_steps`` yielding augmented per-step
    tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from teacher import tokens as T
from teacher.dispatcher import dispatch_rule
from teacher.engine import TeacherError
from teacher.runner import _auto_append_eq
from teacher.state import TeacherState

CTL_SIZE = 3


# ---------------------------------------------------------------------------
# Traced engine (mirror of teacher/engine.py:apply_action + provenance)
# ---------------------------------------------------------------------------
def apply_action_traced(
    state: TeacherState,
    action,
    cleanup: bool,
) -> Tuple[TeacherState, int, List[Tuple[int, int]]]:
    """Apply ``action`` to ``state``; return ``(new_state, final_done, prov)``.

    ``prov[k] = (s, m)``: output cell k came from old-state cell ``s``
    (slot ``m`` of the action for visible cells; ``m=0`` pass-through for
    non-visible cells). Cells dropped by the cleanup branch simply do not
    appear. With ``cleanup=False`` the done branch applies the standard
    return treatment to every max-depth cell (auto-eq tags are ignored).
    """
    ctl_size = state.ctl_size
    if not action or not action[0] or not action[0][0]:
        raise TeacherError("empty action tensor")
    M = len(action[0])
    C = len(action[0][0])
    if C != 1 + ctl_size:
        raise TeacherError(f"action C-dim {C} != 1+ctl_size {1 + ctl_size}")

    done_indicator = action[-1][0][0]

    md_indices = state.max_depth_indices()
    md_relative = {i: rel for rel, i in enumerate(md_indices)}
    md_set = set(md_indices)

    if len(md_indices) + 1 != len(action):
        raise TeacherError(
            f"action S-dim {len(action)} != n_max_depth_tokens+1 "
            f"{len(md_indices) + 1}"
        )

    new_tokens: List[List[int]] = []
    new_depth: List[int] = []
    new_auto_eq: List[Optional[int]] = []
    prov: List[Tuple[int, int]] = []
    for s in range(len(state.tokens)):
        if state.vocab(s) == T.PAD:
            continue
        if s in md_set:
            rel = md_relative[s]
            for m in range(M):
                cell = list(action[rel][m])
                vocab = cell[0]
                if vocab == T.MASK or vocab == T.PAD:
                    continue
                cell[1 + T.CTRL_RESULT] = 0
                new_tokens.append(cell)
                new_depth.append(state.depth[s])
                new_auto_eq.append(state.auto_eq_depth[s] if m == 0 else None)
                prov.append((s, m))
        else:
            new_tokens.append(list(state.tokens[s]))
            new_depth.append(state.depth[s])
            new_auto_eq.append(state.auto_eq_depth[s])
            prov.append((s, 0))

    final_done = 0
    if done_indicator == 1:
        if not new_depth:
            final_done = 1
        else:
            new_max = max(new_depth)
            md_at_new_max = [i for i, d in enumerate(new_depth) if d == new_max]

            cleanup_idx: Optional[int] = None
            if cleanup:
                for i in md_at_new_max:
                    if new_auto_eq[i] == new_max:
                        cleanup_idx = i  # last match wins

            if cleanup_idx is not None:
                drop_set = {i for i in md_at_new_max if i <= cleanup_idx}
                for i in md_at_new_max:
                    if i > cleanup_idx:
                        for k in range(ctl_size):
                            new_tokens[i][1 + k] = 0
                        new_tokens[i][1 + T.CTRL_RESULT] = 1
                        new_depth[i] = new_max - 1
                kept_tokens, kept_depth, kept_auto_eq, kept_prov = [], [], [], []
                for i in range(len(new_tokens)):
                    if i in drop_set:
                        continue
                    kept_tokens.append(new_tokens[i])
                    kept_depth.append(new_depth[i])
                    kept_auto_eq.append(new_auto_eq[i])
                    kept_prov.append(prov[i])
                new_tokens, new_depth, new_auto_eq, prov = (
                    kept_tokens, kept_depth, kept_auto_eq, kept_prov)
            else:
                for i, d in enumerate(new_depth):
                    if d == new_max:
                        for k in range(ctl_size):
                            new_tokens[i][1 + k] = 0
                        new_tokens[i][1 + T.CTRL_RESULT] = 1
                        new_depth[i] = d - 1

            if new_max == 0:
                final_done = 1

    has_self_call = False
    for i, row in enumerate(new_tokens):
        if row[1 + T.CTRL_SELF_CALL] == 1:
            has_self_call = True
            for k in range(ctl_size):
                row[1 + k] = 0
            new_depth[i] += 1

    if done_indicator == 1 and has_self_call:
        final_done = 2

    return (
        TeacherState(tokens=new_tokens, depth=new_depth, ctl_size=ctl_size,
                     auto_eq_depth=new_auto_eq),
        final_done,
        prov,
    )


def apply_action_nocleanup(state: TeacherState, action) -> Tuple[TeacherState, int]:
    """Inference-side engine for exp 3: done never triggers auto-eq cleanup."""
    new_state, final_done, _ = apply_action_traced(state, action, cleanup=False)
    return new_state, final_done


# ---------------------------------------------------------------------------
# Dual (clean, augmented) evolution
# ---------------------------------------------------------------------------
class GhostSliceError(RuntimeError):
    """Raised when auto-append-eq would fire on a visible slice that contains
    ghost cells — the train/inference placement convention would diverge.
    Expected to never happen for the binary-op tasks (verified by the gate)."""


@dataclass
class DualState:
    clean: TeacherState
    abl: TeacherState
    map_a2c: List[Optional[int]]  # per abl cell: clean cell index, None = ghost


def dual_from_text(task_str: str, ctl_size: int = CTL_SIZE) -> DualState:
    clean = TeacherState.from_text(task_str, ctl_size=ctl_size)
    abl = TeacherState.from_text(task_str, ctl_size=ctl_size)
    return DualState(clean=clean, abl=abl, map_a2c=list(range(len(clean.tokens))))


def _check_mapped_cells(ds: DualState) -> None:
    """Invariant: every mapped abl cell is identical (row, depth) to its clean
    counterpart, and mapped cells appear in the same order as in clean."""
    seen = []
    for i, j in enumerate(ds.map_a2c):
        if j is None:
            continue
        seen.append(j)
        if ds.abl.tokens[i] != ds.clean.tokens[j] or ds.abl.depth[i] != ds.clean.depth[j]:
            raise AssertionError(
                f"dual invariant broken at abl cell {i} -> clean {j}: "
                f"{ds.abl.tokens[i]}@{ds.abl.depth[i]} vs "
                f"{ds.clean.tokens[j]}@{ds.clean.depth[j]}"
            )
    if seen != list(range(len(ds.clean.tokens))):
        raise AssertionError(
            f"mapping is not an order-preserving bijection onto clean: {seen[:20]}..."
        )


def dual_auto_append(ds: DualState) -> DualState:
    """Mirror the runner's auto-append-`=` on both states, keeping the map."""
    md_c = ds.clean.max_depth_indices()
    if not md_c or any(ds.clean.vocab(i) == T.EQ for i in md_c):
        return ds
    # Clean tape needs an '='. The abl visible slice must be ghost-free here,
    # otherwise the "append after last visible cell" convention diverges
    # between generation (clean-driven) and inference (abl-driven).
    md_a = ds.abl.max_depth_indices()
    if any(ds.map_a2c[i] is None for i in md_a):
        raise GhostSliceError(
            "auto-append-eq fired on a visible slice containing ghost cells"
        )
    p_c = md_c[-1] + 1
    p_a = md_a[-1] + 1
    if ds.map_a2c[md_a[-1]] != md_c[-1]:
        raise AssertionError("auto-append anchors disagree between clean/abl")
    new_clean = _auto_append_eq(ds.clean)
    new_abl = _auto_append_eq(ds.abl)
    assert len(new_clean.tokens) == len(ds.clean.tokens) + 1
    assert len(new_abl.tokens) == len(ds.abl.tokens) + 1
    new_map = [(m if (m is None or m < p_c) else m + 1) for m in ds.map_a2c]
    new_map.insert(p_a, p_c)
    return DualState(clean=new_clean, abl=new_abl, map_a2c=new_map)


def build_augmented_action(ds: DualState, action) -> list:
    """Extend the clean action with keep-rows at ghost positions."""
    md_c = ds.clean.max_depth_indices()
    md_a = ds.abl.max_depth_indices()
    rel_c = {j: rel for rel, j in enumerate(md_c)}
    ctl_size = ds.abl.ctl_size
    M = len(action[0])

    # Mapped members of the abl visible slice must be exactly clean's visible
    # slice, in order.
    mapped = [ds.map_a2c[i] for i in md_a if ds.map_a2c[i] is not None]
    if mapped != md_c:
        raise AssertionError(
            f"visible-slice mismatch: mapped={mapped[:10]}... md_c={md_c[:10]}..."
        )

    aug = []
    for i in md_a:
        j = ds.map_a2c[i]
        if j is not None:
            aug.append([list(cell) for cell in action[rel_c[j]]])
        else:
            ctrl = list(ds.abl.ctrl(i))
            ctrl[T.CTRL_RESULT] = 0  # engine strips it anyway; keep labels clean
            keep_row = [[ds.abl.vocab(i)] + ctrl]
            for _ in range(M - 1):
                keep_row.append([T.MASK] + [0] * ctl_size)
            aug.append(keep_row)
    aug.append([list(cell) for cell in action[-1]])  # done row
    return aug


def dual_step(ds: DualState, action) -> Tuple[DualState, list, int]:
    """One lock-step engine application. Returns
    ``(new_dual_state, augmented_action, final_done)``; the clean and
    augmented engines must agree on ``final_done``."""
    aug_action = build_augmented_action(ds, action)

    clean2, fd_c, prov_c = apply_action_traced(ds.clean, action, cleanup=True)
    abl2, fd_a, prov_a = apply_action_traced(ds.abl, aug_action, cleanup=False)
    if fd_c != fd_a:
        raise AssertionError(f"final_done diverged: clean={fd_c} abl={fd_a}")

    index_c = {p: idx for idx, p in enumerate(prov_c)}
    new_map: List[Optional[int]] = []
    for idx_a, (src_a, m) in enumerate(prov_a):
        j_old = ds.map_a2c[src_a]
        if j_old is None:
            new_map.append(None)
        else:
            new_map.append(index_c.get((j_old, m)))  # None if cleanup-dropped

    ds2 = DualState(clean=clean2, abl=abl2, map_a2c=new_map)
    return ds2, aug_action, fd_c


# ---------------------------------------------------------------------------
# Step collection (drop-in analogue of data_gen._collect_steps)
# ---------------------------------------------------------------------------
def dual_run(
    task_str: str,
    max_steps: int = 200_000,
    check_invariants: bool = False,
) -> Tuple[Optional[List[Tuple[list, list, list, int]]], Optional[DualState]]:
    """Run the teacher dispatcher under dual evolution.

    Returns ``(steps, final_dual_state)`` where each step is a tuple on the
    AUGMENTED tape:

        (abl_state_tokens, abl_state_depth, augmented_action, done)

    ``(None, None)`` if the trajectory fails (rule error / step cap /
    invariant breach), mirroring the baseline generator's task-drop
    behaviour. The final DualState exposes the teacher's augmented reference
    tape (``.abl``) and the ghost mapping (``.map_a2c``) — the exp-3 eval
    compares the model's final tape against ``final_ds.abl`` verbatim.
    """
    ds = dual_from_text(task_str)
    try:
        ds = dual_auto_append(ds)
    except (GhostSliceError, AssertionError):
        return None, None
    out: List[Tuple[list, list, list, int]] = []
    for _ in range(max_steps):
        try:
            ds = dual_auto_append(ds)
            if check_invariants:
                _check_mapped_cells(ds)
            builder = dispatch_rule(ds.clean)
            action_nested = builder.to_nested_list()
        except (NotImplementedError, RuntimeError, AssertionError):
            return None, None
        done_value = int(action_nested[-1][0][0])

        tokens_snap = [list(row) for row in ds.abl.tokens]
        depth_snap = list(ds.abl.depth)
        try:
            ds, aug_action, final_done = dual_step(ds, action_nested)
        except (TeacherError, AssertionError):
            return None, None
        out.append((tokens_snap, depth_snap, aug_action, done_value))

        if final_done == 1:
            return out, ds
        if final_done == 2:
            return None, None
    return None, None


def collect_steps_nocleanup(
    task_str: str,
    max_steps: int = 200_000,
    check_invariants: bool = False,
) -> Optional[List[Tuple[list, list, list, int]]]:
    """Training-side wrapper: steps only (drop-in for `_collect_steps`)."""
    steps, _ = dual_run(task_str, max_steps, check_invariants)
    return steps
