"""Teacher rule for `multiply_task` (task ids 30 / 31).

Direct translation of the user-supplied multiplication spec.
Notation: (a)/(b)/(c) refer to the ctrl bits
CTRL_SELF_CALL/CTRL_RESULT/CTRL_TAG_C.

Applicability: visible LHS is of the form `<digits> * <digits>` with
exactly one `*`.

Rule 1 — memorisation (1-step):
  `d1 * d2 =` with single digits d1, d2 and RHS empty, all cells clean.
  Insert digits of d1*d2 after `=`, done=1. (100 cases: 0*0..9*9.)

Rule 2 — otherwise. Branch order (the rule list in the spec has 2-1
overlapping with 2-2; the more-specific 2-2 runs first):

  Rule 2-2 — RHS has tokens, no `+` on RHS, and every digit (LHS and
  RHS) has (c):
      done=1.

  Rule 2-1 — RHS has tokens, no `+` on RHS, and every digit (LHS and
  RHS) has (b) or (c) (but NOT all (c)):
      Retag every digit (b)->(c). Among the digits on the RHS, drop
      the consecutive leading 0s from the left except the rightmost
      digit. done=0.

  Rule 2-3 — RHS empty:
      2-3-1 (>=2 number tokens left of `*`):
          (a) on the rightmost number-token left of `*`, on `*`, on
          every number-token right of `*`, and on `=`.
      2-3-2 (exactly 1 number-token left of `*`):
          (a) on that token, on `*`, on the rightmost number-token
          right of `*`, and on `=`.

  Rule 2-4 — RHS has only digits (no `+`), and at least one digit
  (LHS or RHS) is raw:
      If 2+ RHS digits have (b): insert `+` at slot 1 of `=`.
      For each LHS digit with (b):
          assign (c) if the immediately-preceding visible token is a
          number-token without (b); otherwise leave (b) intact.
      (a) on every other LHS digit (i.e., LHS digits not newly (c)),
      on `*`, and on `=`.
      (c) on the rightmost RHS (b) digit only.

  Rule 2-5 — `+` is on the RHS:
      (a) on every RHS token that does not already have (c).
      (c) on every LHS digit with (b).

Done-timing note:
  - 2-1 and 2-2 are separate steps: 2-1 only retags/drops leading 0s
    (done=0), and 2-2 fires done=1 once the tape is all (c) — so the
    closing step never carries a freshly retagged digit.
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


def _no_ctrl(state: TeacherState, i: int) -> bool:
    return all(c == 0 for c in state.ctrl(i))


def _digit_value(state: TeacherState, i: int) -> int:
    return state.vocab(i) - T.DIGIT_0


def multiply_rule(state: TeacherState) -> ActionBuilder:
    md = state.max_depth_indices()
    n = len(md)
    builder = ActionBuilder(n, state.ctl_size, MAX_INSERTION)

    eq_eff: Optional[int] = None
    star_eff: Optional[int] = None
    for s_eff, i in enumerate(md):
        v = state.vocab(i)
        if v == T.EQ and eq_eff is None:
            eq_eff = s_eff
        elif v == T.STAR and eq_eff is None and star_eff is None:
            star_eff = s_eff
    if eq_eff is None:
        raise RuntimeError("multiply_rule: no '=' visible (auto-append should have run)")
    if star_eff is None:
        raise RuntimeError("multiply_rule: no '*' on LHS — dispatcher should not have routed here")

    lhs_eff = list(range(eq_eff))
    rhs_eff = list(range(eq_eff + 1, n))
    left_of_star = [s for s in lhs_eff if s < star_eff]
    right_of_star = [s for s in lhs_eff if s > star_eff]
    lhs_digits_eff = [s for s in lhs_eff if T.is_digit(state.vocab(md[s]))]
    rhs_digits_eff = [s for s in rhs_eff if T.is_digit(state.vocab(md[s]))]
    has_plus_rhs = any(state.vocab(md[s]) == T.PLUS for s in rhs_eff)

    # ---------------- Rule 1: base case ----------------
    is_pure_single = (
        star_eff == 1 and eq_eff == 3 and not rhs_eff
        and T.is_digit(state.vocab(md[0]))
        and T.is_digit(state.vocab(md[2]))
        and all(_no_ctrl(state, md[s]) for s in range(n))
    )
    if is_pure_single:
        a = _digit_value(state, md[0])
        b = _digit_value(state, md[2])
        c = a * b
        for s_eff, i in enumerate(md):
            builder.keep(s_eff, state.vocab(i), state.ctrl(i))
        digits = [int(ch) for ch in str(c)]
        if len(digits) > MAX_INSERTION:
            raise RuntimeError(
                f"multiply Rule 1: result {c} has {len(digits)} digits > "
                f"MAX_INSERTION={MAX_INSERTION}"
            )
        for slot, dv in enumerate(digits, start=1):
            builder.insert(eq_eff, slot, T.DIGIT_0 + dv, [0] * state.ctl_size)
        builder.set_done(1)
        return builder

    # ---------------- Rule 2-2: every digit (c) → done ----------------
    if (rhs_digits_eff
            and not has_plus_rhs
            and all(_has_c(state, md[s]) for s in lhs_digits_eff)
            and all(_has_c(state, md[s]) for s in rhs_digits_eff)):
        return _rule_2_2(state, md, n, builder)

    # ---------------- Rule 2-1: every digit (b)/(c) → retag + drop 0s ----------------
    if (rhs_digits_eff
            and not has_plus_rhs
            and all((_has_b(state, md[s]) or _has_c(state, md[s]))
                    for s in lhs_digits_eff)
            and all((_has_b(state, md[s]) or _has_c(state, md[s]))
                    for s in rhs_digits_eff)):
        return _rule_2_1(state, md, n, rhs_digits_eff, builder)

    # ---------------- Rule 2-3: RHS empty ----------------
    if not rhs_eff:
        if len(left_of_star) >= 2:
            return _rule_2_3_1(state, md, n, eq_eff, star_eff,
                               left_of_star, right_of_star, builder)
        if len(left_of_star) == 1:
            return _rule_2_3_2(state, md, n, eq_eff, star_eff,
                               left_of_star, right_of_star, builder)
        raise RuntimeError("multiply_rule: 0 number-tokens left of '*' with empty RHS")

    # ---------------- Rule 2-4: RHS only digits ----------------
    if not has_plus_rhs:
        return _rule_2_4(state, md, n, eq_eff, star_eff,
                         lhs_digits_eff, rhs_digits_eff, builder)

    # ---------------- Rule 2-5: '+' on RHS ----------------
    return _rule_2_5(state, md, n, eq_eff, lhs_digits_eff, builder)


# ============================================================================
#                             Rule 2-1 / 2-2
# ============================================================================

def _rule_2_1(state, md, n, rhs_digits_eff, builder) -> ActionBuilder:
    """Spec 2-1 — retag (b)->(c) on every digit; drop leading 0s from the
    RHS number-run except the rightmost digit. done=0.
    """
    delete_set = set()
    for s in rhs_digits_eff[:-1]:
        if _digit_value(state, md[s]) == 0:
            delete_set.add(s)
        else:
            break

    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        if s_eff in delete_set:
            builder.delete(s_eff)
            continue
        ctrl = list(state.ctrl(i))
        if T.is_digit(vocab) and ctrl[T.CTRL_RESULT] == 1:
            ctrl[T.CTRL_RESULT] = 0
            ctrl[T.CTRL_TAG_C] = 1
        builder.replace(s_eff, vocab, ctrl)
    builder.set_done(0)
    return builder


def _rule_2_2(state, md, n, builder) -> ActionBuilder:
    """Spec 2-2 — every digit already (c) → done=1."""
    for s_eff, i in enumerate(md):
        builder.keep(s_eff, state.vocab(i), state.ctrl(i))
    builder.set_done(1)
    return builder


# ============================================================================
#                             Rule 2-3
# ============================================================================

def _rule_2_3_1(state, md, n, eq_eff, star_eff,
                left_of_star, right_of_star, builder) -> ActionBuilder:
    """Spec 2-3-1 — 2+ number-tokens left of `*`."""
    a_set = {left_of_star[-1], star_eff, eq_eff}
    a_set.update(right_of_star)
    return _apply_a_set(state, md, n, a_set, builder)


def _rule_2_3_2(state, md, n, eq_eff, star_eff,
                left_of_star, right_of_star, builder) -> ActionBuilder:
    """Spec 2-3-2 — 1 number-token left of `*`."""
    a_set = {left_of_star[0], star_eff, eq_eff}
    if right_of_star:
        a_set.add(right_of_star[-1])
    return _apply_a_set(state, md, n, a_set, builder)


def _apply_a_set(state, md, n, a_set, builder) -> ActionBuilder:
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
#                             Rule 2-4
# ============================================================================

def _rule_2_4(state, md, n, eq_eff, star_eff,
              lhs_digits_eff, rhs_digits_eff, builder) -> ActionBuilder:
    """Spec 2-4 — RHS only digits, at least one digit raw.

    For LHS (b) digits: (c) iff predecessor visible token is a digit
    without (b). All other LHS digits get (a), along with `*` and `=`.
    Rightmost RHS (b) digit gets (c). 2+ RHS (b) digits triggers `+`
    insertion at slot 1 of `=`.
    """
    lhs_b_digits = [s for s in lhs_digits_eff if _has_b(state, md[s])]
    c_lhs_set = set()
    for s in lhs_b_digits:
        if s > 0:
            prev = md[s - 1]
            if T.is_digit(state.vocab(prev)) and not _has_b(state, prev):
                c_lhs_set.add(s)

    rhs_b_digits = [s for s in rhs_digits_eff if _has_b(state, md[s])]
    c_rhs_eff: Optional[int] = rhs_b_digits[-1] if rhs_b_digits else None

    a_set = set()
    for s_eff in lhs_digits_eff:
        if s_eff in c_lhs_set:
            continue
        a_set.add(s_eff)
    a_set.add(star_eff)
    a_set.add(eq_eff)

    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        ctrl = list(state.ctrl(i))
        if s_eff in a_set:
            ctrl = [0] * state.ctl_size
            ctrl[T.CTRL_SELF_CALL] = 1
        elif s_eff in c_lhs_set or s_eff == c_rhs_eff:
            ctrl = [0] * state.ctl_size
            ctrl[T.CTRL_TAG_C] = 1
        builder.replace(s_eff, vocab, ctrl)

    if len(rhs_b_digits) >= 2:
        builder.insert(eq_eff, 1, T.PLUS, [0] * state.ctl_size)
    builder.set_done(0)
    return builder


# ============================================================================
#                             Rule 2-5
# ============================================================================

def _rule_2_5(state, md, n, eq_eff, lhs_digits_eff, builder) -> ActionBuilder:
    """Spec 2-5 — `+` on RHS. (a) on every RHS token without (c); (c) on
    every LHS (b) digit.
    """
    rhs_eff = list(range(eq_eff + 1, n))
    a_set = {s for s in rhs_eff if not _has_c(state, md[s])}

    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        ctrl = list(state.ctrl(i))
        if s_eff in a_set:
            ctrl = [0] * state.ctl_size
            ctrl[T.CTRL_SELF_CALL] = 1
        elif s_eff in lhs_digits_eff and ctrl[T.CTRL_RESULT] == 1:
            ctrl[T.CTRL_RESULT] = 0
            ctrl[T.CTRL_TAG_C] = 1
        builder.replace(s_eff, vocab, ctrl)
    builder.set_done(0)
    return builder