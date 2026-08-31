"""Teacher rule for `add_task` (task id 10).

Literal translation of the user-supplied spec. (a)/(b)/(c) refer to ctrl
bits CTRL_SELF_CALL/CTRL_RESULT/CTRL_TAG_C.

Applicability of the rule (enforced by the dispatcher):
  - exactly one `=` on the visible tape,
  - LHS of `=` has the form `<digits> + <digits>`.

Rule sections:
  Rule 1 — `a + b =` with single-digit a, b (RHS empty):
    Insert digits of `a+b` after `=` and done=1.

  Rule 2 — otherwise:
    Rule 2-3 — RHS empty:
      (a) on the `+` token and the token immediately to its left, and
      on the `=` token and the token immediately to its left.

    Rule 2-2 — `+` exists on the RHS:
      (a) on the RHS `+` and each RHS digit (0-9) that does not have (c);
      (c) on every LHS digit with (b).

    Rule 2-1 — RHS has tokens but no `+`:
      Rule 2-1-1 — every digit (LHS+RHS) has (b) or (c):
        Rule 2-1-1-2 — every digit (c):
          done=1.
        Rule 2-1-1-1 — not all (c):
          retag every digit (b)->(c). done=0.
      Rule 2-1-2 — some digit raw (no ctrl):
        Rule 2-1-2-1 — n_rhs_b in {0, >=2} (start point or carry):
          For each side of the LHS `+`, pick the rightmost digit token
          that has neither (b) nor (c) and (a) it (skip a side with no
          such token). (a) on `=`. If both sides picked, (a) on the LHS
          `+`. Retag LHS digits with (b) to (c). In the carry case
          (n_rhs_b >= 2) only: (c) the rightmost RHS (b) digit and
          insert `+` at slot 1 of `=`.
        Rule 2-1-2-2 — n_rhs_b == 1 (no-carry continuation):
          (a) on every LHS digit without (b)/(c). (a) on `=`. If both
          sides of LHS `+` have such a digit, (a) on the LHS `+`. Retag
          every digit (LHS+RHS) with (b) to (c).
"""

from __future__ import annotations

from typing import List, Optional

from teacher import tokens as T
from teacher.action import ActionBuilder
from teacher.state import TeacherState


MAX_INSERTION = 3


def _has_b(state: TeacherState, i: int) -> bool:
    return state.ctrl_at(i, T.CTRL_RESULT) == 1


def _has_c(state: TeacherState, i: int) -> bool:
    return state.ctrl_at(i, T.CTRL_TAG_C) == 1


def _no_bc(state: TeacherState, i: int) -> bool:
    return not _has_b(state, i) and not _has_c(state, i)


def _digit_value(state: TeacherState, i: int) -> int:
    return state.vocab(i) - T.DIGIT_0


def add_rule(state: TeacherState) -> ActionBuilder:
    md = state.max_depth_indices()
    n = len(md)
    builder = ActionBuilder(n, state.ctl_size, MAX_INSERTION)

    eq_eff: Optional[int] = None
    plus_left_eff: Optional[int] = None
    for s_eff, i in enumerate(md):
        v = state.vocab(i)
        if v == T.EQ and eq_eff is None:
            eq_eff = s_eff
        elif v == T.PLUS and eq_eff is None and plus_left_eff is None:
            plus_left_eff = s_eff
    if eq_eff is None:
        raise RuntimeError("add_rule: no '=' visible")
    if plus_left_eff is None:
        raise RuntimeError("add_rule: no '+' on LHS — dispatcher should not have routed here")

    lhs_eff = list(range(eq_eff))
    rhs_eff = list(range(eq_eff + 1, n))
    left_of_plus = [s for s in lhs_eff if s < plus_left_eff]
    right_of_plus = [s for s in lhs_eff if s > plus_left_eff]
    rhs_has_plus = any(state.vocab(md[s]) == T.PLUS for s in rhs_eff)

    # ---- Rule 1: `a + b =` single-digit base case ----
    is_pure_single = (
        plus_left_eff == 1 and eq_eff == 3 and not rhs_eff
        and T.is_digit(state.vocab(md[0]))
        and T.is_digit(state.vocab(md[2]))
    )
    if is_pure_single:
        a = _digit_value(state, md[0])
        b = _digit_value(state, md[2])
        c = a + b
        for s_eff, i in enumerate(md):
            builder.keep(s_eff, state.vocab(i), state.ctrl(i))
        for slot, dv in enumerate([int(ch) for ch in str(c)], start=1):
            builder.insert(eq_eff, slot, T.DIGIT_0 + dv, [0] * state.ctl_size)
        builder.set_done(1)
        return builder

    # ---- Rule 2-3: RHS empty ----
    if not rhs_eff:
        return _rule_2_3(state, md, n, eq_eff, plus_left_eff, builder)

    # ---- Rule 2-2: `+` on RHS ----
    if rhs_has_plus:
        return _rule_2_2(state, md, n, lhs_eff, rhs_eff, builder)

    # ---- Rule 2-1: RHS non-empty, no `+` on RHS ----
    digit_eff = [s for s in range(n) if T.is_digit(state.vocab(md[s]))]

    # 2-1-1: every digit has (b) or (c).
    if digit_eff and all(
        _has_b(state, md[s]) or _has_c(state, md[s]) for s in digit_eff
    ):
        # 2-1-1-2: every digit (c) → done=1.
        if all(_has_c(state, md[s]) for s in digit_eff):
            for s_eff, i in enumerate(md):
                builder.keep(s_eff, state.vocab(i), state.ctrl(i))
            builder.set_done(1)
            return builder
        # 2-1-1-1: retag (b)->(c).
        for s_eff, i in enumerate(md):
            vocab = state.vocab(i)
            ctrl = list(state.ctrl(i))
            if T.is_digit(vocab) and ctrl[T.CTRL_RESULT] == 1:
                ctrl[T.CTRL_RESULT] = 0
                ctrl[T.CTRL_TAG_C] = 1
            builder.replace(s_eff, vocab, ctrl)
        builder.set_done(0)
        return builder

    # 2-1-2: some digit raw. Count RHS digits with (b).
    rhs_b_digits = [
        s for s in rhs_eff
        if T.is_digit(state.vocab(md[s])) and _has_b(state, md[s])
    ]
    n_rhs_b = len(rhs_b_digits)
    if n_rhs_b == 1:
        return _rule_2_1_2_2(state, md, n, eq_eff, plus_left_eff,
                             left_of_plus, right_of_plus,
                             rhs_b_digits[0], builder)
    return _rule_2_1_2_1(state, md, n, eq_eff, plus_left_eff,
                         left_of_plus, right_of_plus,
                         rhs_b_digits, builder)


# ============================================================================
# Rule 2-3 — RHS empty
# ============================================================================

def _rule_2_3(state, md, n, eq_eff, plus_left_eff, builder) -> ActionBuilder:
    """Spec 2-3 — (a) on `+`, on the token immediately left of `+`, on
    `=`, and on the token immediately left of `=`.
    """
    a_set = {plus_left_eff, eq_eff}
    if plus_left_eff - 1 >= 0:
        a_set.add(plus_left_eff - 1)
    if eq_eff - 1 >= 0:
        a_set.add(eq_eff - 1)

    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        ctrl = list(state.ctrl(i))
        if s_eff in a_set:
            ctrl = [0] * state.ctl_size
            ctrl[T.CTRL_SELF_CALL] = 1
        builder.replace(s_eff, vocab, ctrl)
    builder.set_done(0)
    return builder


# ============================================================================
# Rule 2-2 — `+` on RHS
# ============================================================================

def _rule_2_2(state, md, n, lhs_eff, rhs_eff, builder) -> ActionBuilder:
    """Spec 2-2 — (a) on the RHS `+` and each RHS digit without (c); (c)
    on every LHS digit with (b).
    """
    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        ctrl = list(state.ctrl(i))
        if s_eff in rhs_eff and not _has_c(state, md[s_eff]) and (
            T.is_digit(vocab) or vocab == T.PLUS
        ):
            ctrl = [0] * state.ctl_size
            ctrl[T.CTRL_SELF_CALL] = 1
        elif s_eff in lhs_eff and T.is_digit(vocab) and ctrl[T.CTRL_RESULT] == 1:
            ctrl[T.CTRL_RESULT] = 0
            ctrl[T.CTRL_TAG_C] = 1
        builder.replace(s_eff, vocab, ctrl)
    builder.set_done(0)
    return builder


# ============================================================================
# Rule 2-1-2-1 — start point / carry (n_rhs_b in {0, >=2})
# ============================================================================

def _rule_2_1_2_1(state, md, n, eq_eff, plus_left_eff,
                  left_of_plus, right_of_plus, rhs_b_digits, builder) -> ActionBuilder:
    left_picks = [s for s in left_of_plus
                  if T.is_digit(state.vocab(md[s])) and _no_bc(state, md[s])]
    right_picks = [s for s in right_of_plus
                   if T.is_digit(state.vocab(md[s])) and _no_bc(state, md[s])]
    a_set = set()
    if left_picks:
        a_set.add(left_picks[-1])
    if right_picks:
        a_set.add(right_picks[-1])
    a_set.add(eq_eff)
    if left_picks and right_picks:
        a_set.add(plus_left_eff)

    rightmost_rhs_b = rhs_b_digits[-1] if rhs_b_digits else None
    is_carry = len(rhs_b_digits) >= 2

    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        ctrl = list(state.ctrl(i))
        if s_eff in a_set:
            ctrl = [0] * state.ctl_size
            ctrl[T.CTRL_SELF_CALL] = 1
        elif is_carry and s_eff == rightmost_rhs_b:
            ctrl = [0] * state.ctl_size
            ctrl[T.CTRL_TAG_C] = 1
        elif s_eff < eq_eff and T.is_digit(vocab) and ctrl[T.CTRL_RESULT] == 1:
            ctrl[T.CTRL_RESULT] = 0
            ctrl[T.CTRL_TAG_C] = 1
        builder.replace(s_eff, vocab, ctrl)

    if is_carry:
        builder.insert(eq_eff, 1, T.PLUS, [0] * state.ctl_size)

    builder.set_done(0)
    return builder


# ============================================================================
# Rule 2-1-2-2 — no-carry (n_rhs_b == 1)
# ============================================================================

def _rule_2_1_2_2(state, md, n, eq_eff, plus_left_eff,
                  left_of_plus, right_of_plus, rhs_b_digit_eff, builder) -> ActionBuilder:
    left_a = [s for s in left_of_plus
              if T.is_digit(state.vocab(md[s])) and _no_bc(state, md[s])]
    right_a = [s for s in right_of_plus
               if T.is_digit(state.vocab(md[s])) and _no_bc(state, md[s])]
    a_set = set(left_a) | set(right_a) | {eq_eff}
    if left_a and right_a:
        a_set.add(plus_left_eff)

    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        ctrl = list(state.ctrl(i))
        if s_eff in a_set:
            ctrl = [0] * state.ctl_size
            ctrl[T.CTRL_SELF_CALL] = 1
        elif T.is_digit(vocab) and ctrl[T.CTRL_RESULT] == 1:
            ctrl[T.CTRL_RESULT] = 0
            ctrl[T.CTRL_TAG_C] = 1
        builder.replace(s_eff, vocab, ctrl)

    builder.set_done(0)
    return builder
