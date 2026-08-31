"""Teacher rule for `copy_task` (task id 0).

Spec (Markov function of the visible tape — tokens at max depth):

  Rule 1 — memorization base case (RHS empty AND LHS digit count <= 3):
      Insert each LHS digit sequentially after '=' and call done.
      (e.g. `34=` -> `34=34`)

  Rule 2 — otherwise (RHS has tokens OR LHS digit count > 3):

      Rule 2-1 — some LHS digit has no control bit at all:
          Set (a) on the rightmost up-to-3 such digits and on '='. Also
          retag any digit carrying (b) as (c) in the same step. Emit a
          self-call (done=0).

      Rule 2-2 — every LHS digit carries at least one control bit, but
                 some lack (c):
          Retag every digit carrying (b) as (c). Emit no self-call (done=0).

      Rule 2-3 — every LHS digit already carries (c):
          Call done.

The split between 2-2 and 2-3 enforces the teacher's safe done timing: the rule
refuses to emit `done=1` on the same step it retags `(b)` -> `(c)`, so the
closing step never carries a freshly returned `(b)` digit alongside the
final answer.
"""

from __future__ import annotations

from teacher import tokens as T
from teacher.action import ActionBuilder
from teacher.state import TeacherState


MAX_INSERTION = 3  # matches model_max_insertion default


def copy_rule(state: TeacherState) -> ActionBuilder:
    md = state.max_depth_indices()
    builder = ActionBuilder(
        n_max_depth_tokens=len(md),
        ctl_size=state.ctl_size,
        max_insertion=MAX_INSERTION,
    )

    # Locate '=' among visible tokens.
    eq_local = None
    for s_eff, i in enumerate(md):
        if state.vocab(i) == T.EQ:
            eq_local = s_eff
            break
    if eq_local is None:
        raise RuntimeError("copy_rule: no '=' found in visible tape")

    digits_left_eff = [
        s_eff for s_eff, i in enumerate(md[:eq_local])
        if T.is_digit(state.vocab(i))
    ]
    digits_left_no_ctrl_eff = [
        s_eff for s_eff in digits_left_eff
        if all(c == 0 for c in state.ctrl(md[s_eff]))
    ]
    digits_left_no_c_eff = [
        s_eff for s_eff in digits_left_eff
        if state.ctrl(md[s_eff])[T.CTRL_TAG_C] == 0
    ]
    has_any_after_eq = (eq_local + 1) < len(md)

    # ---------------- Rule 1: RHS empty AND LHS length <= 3 ----------------
    if not has_any_after_eq and len(digits_left_eff) <= 3:
        for s_eff, i in enumerate(md):
            builder.keep(s_eff, state.vocab(i), state.ctrl(i))
        for slot, src_s_eff in enumerate(digits_left_eff, start=1):
            src_i = md[src_s_eff]
            builder.insert(eq_local, slot, state.vocab(src_i), [0] * state.ctl_size)
        builder.set_done(1)
        return builder

    # ---------------- Rule 2-1: some LHS digit has no ctrl ----------------
    if digits_left_no_ctrl_eff:
        chosen = set(digits_left_no_ctrl_eff[-3:])
        for s_eff, i in enumerate(md):
            vocab = state.vocab(i)
            ctrl = list(state.ctrl(i))
            if s_eff in chosen or s_eff == eq_local:
                ctrl = [0] * state.ctl_size
                ctrl[T.CTRL_SELF_CALL] = 1
            elif T.is_digit(vocab) and ctrl[T.CTRL_RESULT] == 1:
                ctrl[T.CTRL_RESULT] = 0
                ctrl[T.CTRL_TAG_C] = 1
            builder.replace(s_eff, vocab, ctrl)
        builder.set_done(0)
        return builder

    # ---------------- Rule 2-2: all LHS have ctrl, some lack (c) ----------------
    if digits_left_no_c_eff:
        for s_eff, i in enumerate(md):
            vocab = state.vocab(i)
            ctrl = list(state.ctrl(i))
            if T.is_digit(vocab) and ctrl[T.CTRL_RESULT] == 1:
                ctrl[T.CTRL_RESULT] = 0
                ctrl[T.CTRL_TAG_C] = 1
            builder.replace(s_eff, vocab, ctrl)
        builder.set_done(0)
        return builder

    # ---------------- Rule 2-3: all LHS digits carry (c) -> done ----------------
    for s_eff, i in enumerate(md):
        builder.keep(s_eff, state.vocab(i), state.ctrl(i))
    builder.set_done(1)
    return builder
