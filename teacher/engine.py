
"""Pure-Python mirror of `dlm_agent.reconstruct` for a single trajectory.

This is the *only* place that knows how the environment evolves under an
action; rule files (`teacher/tasks/*.py`) read state and emit actions but never
touch this logic. Keep it lock-step with `model/dlm_agent.py:289-424`, with
one teacher-side extension: the `auto_eq_depth` tag the runner attaches to an
auto-appended `=` cell (see `runner._auto_append_eq` and `state.TeacherState`).

When done=1 fires from a self-call whose input had an auto-appended `=` AT
THE CURRENT MAX DEPTH (i.e. the auto-eq's `auto_eq_depth` equals `new_max`),
the engine drops the sub-call's operands and the auto-eq itself, returning
only the digits inserted *after* it. Self-calls whose `=` was already
present (rule-supplied or carried over via (a)) get the standard return.
This mirrors the user's global rule: "if `=` was added based on the rule at
the self-call input stage, only the value after `=` is returned at the
return stage."
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from teacher import tokens as T
from teacher.state import TeacherState


class TeacherError(RuntimeError):
    """Raised when the rule produces an action that the engine considers
    invalid (e.g., self-call + done in the same step)."""


def apply_action(state: TeacherState, action) -> Tuple[TeacherState, int]:
    """Apply a per-step action (nested list, shape (S_eff+1, 1+max_insertion, 1+ctl_size))
    to `state`. Returns `(new_state, final_done)` where `final_done` is:

      0 — episode continues
      1 — episode terminated (done at depth 0)
      2 — illegal: done and self-call set in the same step
    """
    ctl_size = state.ctl_size
    if not action or not action[0] or not action[0][0]:
        raise TeacherError("empty action tensor")
    M = len(action[0])
    C = len(action[0][0])
    if C != 1 + ctl_size:
        raise TeacherError(f"action C-dim {C} != 1+ctl_size {1 + ctl_size}")

    # 1) Pull done indicator out of the trailing slot.
    done_indicator = action[-1][0][0]

    # 2) Index max-depth (visible) positions in the original tape.
    md_indices = state.max_depth_indices()
    md_relative = {i: rel for rel, i in enumerate(md_indices)}
    md_set = set(md_indices)

    if len(md_indices) + 1 != len(action):
        raise TeacherError(
            f"action S-dim {len(action)} != n_max_depth_tokens+1 "
            f"{len(md_indices) + 1}"
        )

    # 3) Expand each original row into M cells, then filter. The (b) bit in
    # action ctrl is always forced to 0 — per the spec the model cannot
    # generate (b); only the engine sets it on self-call return. This also
    # implements the "(b) auto-clears between steps" rule, since the action's
    # slot 0 overwrites the previous step's (b) on each max-depth token.
    #
    # The auto-eq tag is preserved on slot 0 of expanded max-depth cells
    # (carrying the parent cell's tag). Insertion slots (m >= 1) are always
    # tagged None — only the runner can mint an auto-eq cell.
    new_tokens: List[List[int]] = []
    new_depth: List[int] = []
    new_auto_eq: List[Optional[int]] = []
    for s in range(len(state.tokens)):
        if state.vocab(s) == T.PAD:
            continue
        if s in md_set:
            rel = md_relative[s]
            for m in range(M):
                cell = list(action[rel][m])
                vocab = cell[0]
                if vocab == T.MASK or vocab == T.PAD:
                    continue  # MASK = delete / no-insert, PAD = empty slot
                cell[1 + T.CTRL_RESULT] = 0  # strip any (b) the rule wrote
                new_tokens.append(cell)
                new_depth.append(state.depth[s])
                new_auto_eq.append(state.auto_eq_depth[s] if m == 0 else None)
        else:
            new_tokens.append(list(state.tokens[s]))
            new_depth.append(state.depth[s])
            new_auto_eq.append(state.auto_eq_depth[s])

    # 4) Done logic — applied *before* self-call, matching the engine. On
    # return from a self-call, returned tokens get *only* (b); any other ctrl
    # bits the inner rule may have set are cleared (per the spec, inner-loop
    # control elements do not propagate to the outer scope).
    final_done = 0
    if done_indicator == 1:
        if not new_depth:
            final_done = 1
        else:
            new_max = max(new_depth)

            # Auto-eq cleanup branch. If a max-depth cell carries an auto-eq
            # tag whose insertion-depth equals new_max, this self-call's
            # auto-appended `=` matches: drop max-depth cells before-and-
            # including it, keep cells after (with the standard return
            # treatment).
            md_at_new_max = [i for i, d in enumerate(new_depth) if d == new_max]
            cleanup_idx: Optional[int] = None
            for i in md_at_new_max:
                if new_auto_eq[i] == new_max:
                    cleanup_idx = i  # last match wins (rightmost auto-eq)

            if cleanup_idx is not None:
                # Indices to drop: every max-depth cell with index <= cleanup_idx.
                drop_set = {i for i in md_at_new_max if i <= cleanup_idx}
                # Cells AFTER auto-eq at max-depth: standard return.
                for i in md_at_new_max:
                    if i > cleanup_idx:
                        for k in range(ctl_size):
                            new_tokens[i][1 + k] = 0
                        new_tokens[i][1 + T.CTRL_RESULT] = 1
                        new_depth[i] = new_max - 1
                # Compact: drop the cells in drop_set.
                kept_tokens, kept_depth, kept_auto_eq = [], [], []
                for i in range(len(new_tokens)):
                    if i in drop_set:
                        continue
                    kept_tokens.append(new_tokens[i])
                    kept_depth.append(new_depth[i])
                    kept_auto_eq.append(new_auto_eq[i])
                new_tokens, new_depth, new_auto_eq = kept_tokens, kept_depth, kept_auto_eq
            else:
                # Standard return: clear ctrl, set (b), depth -=1 on every
                # max-depth cell.
                for i, d in enumerate(new_depth):
                    if d == new_max:
                        for k in range(ctl_size):
                            new_tokens[i][1 + k] = 0
                        new_tokens[i][1 + T.CTRL_RESULT] = 1
                        new_depth[i] = d - 1

            if new_max == 0:
                final_done = 1

    # 5) Self-call logic — every token whose (a) bit is set ascends by 1 with
    # all ctrl bits cleared. The auto-eq tag is independent of ctrl bits and
    # survives the bump (so an outer auto-eq carried into a deeper sub-call
    # only matches when it eventually returns to its insertion depth).
    has_self_call = False
    for i, row in enumerate(new_tokens):
        if row[1 + T.CTRL_SELF_CALL] == 1:
            has_self_call = True
            for k in range(ctl_size):
                row[1 + k] = 0
            new_depth[i] += 1

    # 6) Done + self-call together → invalid.
    if done_indicator == 1 and has_self_call:
        final_done = 2

    return (
        TeacherState(tokens=new_tokens, depth=new_depth, ctl_size=ctl_size,
                     auto_eq_depth=new_auto_eq),
        final_done,
    )
