"""Teacher rule for `subtract_task` (task ids 20 / 21).

Literal translation of the user-supplied spec. (a)/(b)/(c) refer to the
ctrl bits CTRL_SELF_CALL/CTRL_RESULT/CTRL_TAG_C. The letter `b`
(no parens) is the literal token T.B_LOWER (the borrow marker).

Applicability of the rule (enforced by the dispatcher):
  - exactly one `=` on the visible tape, AND
  - on the LHS of `=`:
      Case A — exactly one `-` and no other operators or `b` letters, OR
      Case B — exactly one `b` letter and no `-` (a `+` may coexist).

Rule sections:
  Rule 1 — memorisation (1-step):
    1-1 d1-d2=                 (single-digit subtraction)
    1-2 b<n>=  (k = 1 or 2)
    1-3 b<n>=-
    1-4 -d=
    1-5 d+b=

  Rule 2 — LHS starts with `b` (and Rule 1 didn't fire):
    2-1 No RHS:
      Clear ctrl on `=` and on LHS digit tokens. (a) on the leftmost `b`,
      on `=`, and on the two tokens immediately to the left of `=`.
    2-2 RHS non-empty. Three branches:
      2-2-A — some LHS digit raw:
        (a) on the rightmost up-to-2 raw LHS digits, on the leftmost `b`,
        on `=`, on `-` immediately right of `=` if present. Retag
        (b)->(c) on every digit that's not lifted. done=0.
      2-2-B — no raw LHS digit, some digit (LHS or RHS) still (b):
        Retag every digit (b)->(c). done=0.
      2-2-C — every digit already (c):
        done=1.

  Rule 3 — general (LHS does not start with `b`, Rule 1 didn't fire):
    3-1 — Tokens on RHS, every LHS digit (c), every RHS digit (b),
          no `b`/`+` visible:
            Retag every digit (b)->(c). Drop consecutive 0s immediately
            after `=` or after `-` (excluding the rightmost digit of each
            number slot). done=0.
    3-2 — Tokens on RHS, every digit (c), no `b`/`+`:
            done=1.
    3-3 — `-` immediately right of `=`:
            (c) on LHS digits with (b); (a) on the `-` and the next token.
    3-4 — `+` on RHS, no `-` right of `=`:
            (a) on the `+` and the tokens on both sides of it; (c) on LHS
            digits with (b).
    3-5 — Neither `-` right of `=` nor `+` on RHS. Sub-branches:
      3-5-3 — RHS empty:
        Relative to the LHS `-` sign, (a) on rightmost digit on each side
        (skip a side with no digits), on the `-` sign, and on `=`.
      3-5-1 — `b` immediately right of `=` (RHS non-empty):
        3-5-1-1 every LHS digit has (b)/(c):
          (c) on LHS (b) digits; (a) on every RHS token.
        3-5-1-2 some LHS digit raw:
          Insert `+` at slot 1 of `=`. (c) on every digit (LHS+RHS) with
          (b). Pick scheme based on raw digits on each side of LHS `-`:
            both sides have a raw digit  -> (a) on rightmost-raw of each,
                                            on LHS `-`, on `=`
            only left side has raw        -> (a) on rightmost-raw, on `=`
            only right side has raw       -> (a) on rightmost-raw, on
                                            LHS `-`, on `=`
      3-5-2 — RHS non-empty, no `b` right of `=`:
        3-5-2-1 every LHS digit has (b)/(c):
          (c) on LHS (b) digits; (a) on every RHS token.
        3-5-2-2 some LHS digit raw:
          (c) on every digit (LHS+RHS) with (b). Same pick scheme as
          3-5-1-2 (without `+` insertion).
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


def _no_ctrl(state: TeacherState, i: int) -> bool:
    return all(c == 0 for c in state.ctrl(i))


def _digit_value(state: TeacherState, i: int) -> int:
    return state.vocab(i) - T.DIGIT_0


def subtract_rule(state: TeacherState) -> ActionBuilder:
    md = state.max_depth_indices()
    n = len(md)
    builder = ActionBuilder(n, state.ctl_size, MAX_INSERTION)

    eq_eff: Optional[int] = None
    minus_left_eff: Optional[int] = None
    b_left_effs: List[int] = []
    for s_eff, i in enumerate(md):
        v = state.vocab(i)
        if v == T.EQ and eq_eff is None:
            eq_eff = s_eff
        elif eq_eff is None:
            if v == T.MINUS and minus_left_eff is None:
                minus_left_eff = s_eff
            elif v == T.B_LOWER:
                b_left_effs.append(s_eff)
    if eq_eff is None:
        raise RuntimeError("subtract_rule: no '=' visible")

    lhs_eff = list(range(eq_eff))
    rhs_eff = list(range(eq_eff + 1, n))

    minus_after_eq_eff: Optional[int] = (
        rhs_eff[0] if rhs_eff and state.vocab(md[rhs_eff[0]]) == T.MINUS else None
    )
    plus_rhs_eff: Optional[int] = next(
        (s for s in rhs_eff if state.vocab(md[s]) == T.PLUS), None
    )
    b_rhs_first_eff: Optional[int] = (
        rhs_eff[0] if rhs_eff and state.vocab(md[rhs_eff[0]]) == T.B_LOWER else None
    )

    # ---- Rule 1 ----
    r1 = _try_rule_1(state, md, n, eq_eff, lhs_eff, rhs_eff,
                     minus_after_eq_eff, builder)
    if r1 is not None:
        return _finalize(r1)

    # ---- Rule 2 ----
    if lhs_eff and state.vocab(md[lhs_eff[0]]) == T.B_LOWER:
        return _finalize(_rule_2(state, md, n, eq_eff, lhs_eff, rhs_eff,
                                 b_left_effs, minus_after_eq_eff, builder))

    # ---- Rule 3 ----
    return _finalize(_rule_3(state, md, n, eq_eff, lhs_eff, rhs_eff,
                             minus_left_eff, minus_after_eq_eff,
                             plus_rhs_eff, b_rhs_first_eff, builder))


# ============================================================================
# Rule 1 — memorisation
# ============================================================================

def _try_rule_1(state, md, n, eq_eff, lhs_eff, rhs_eff,
                minus_after_eq_eff, builder) -> Optional[ActionBuilder]:
    visible_no_ctrl = all(_no_ctrl(state, md[s]) for s in range(n))

    # 1-1: d1 - d2 =   (fires regardless of ctrl bits)
    if (n == 4 and eq_eff == 3
            and T.is_digit(state.vocab(md[0]))
            and state.vocab(md[1]) == T.MINUS
            and T.is_digit(state.vocab(md[2]))):
        d1 = _digit_value(state, md[0])
        d2 = _digit_value(state, md[2])
        result = d1 - d2
        chars = list(str(result))
        return _emit_insert_after_anchor(state, md, n, eq_eff, chars, builder)

    # 1-2: b<n>=  (k = 1 or 2)
    if (visible_no_ctrl and not rhs_eff and len(lhs_eff) >= 2
            and state.vocab(md[0]) == T.B_LOWER
            and 1 <= len(lhs_eff) - 1 <= 2
            and all(T.is_digit(state.vocab(md[s])) for s in lhs_eff[1:])):
        digits = [_digit_value(state, md[s]) for s in lhs_eff[1:]]
        k = len(digits)
        nv = sum(d * (10 ** (k - 1 - i)) for i, d in enumerate(digits))
        if nv == 0:
            chars = ['0'] * k
        else:
            chars = list('-' + str(10 ** k - nv).zfill(k))
        return _emit_insert_after_anchor(state, md, n, eq_eff, chars, builder)

    # 1-3: b<n>=-
    if (visible_no_ctrl and minus_after_eq_eff is not None
            and len(rhs_eff) == 1
            and len(lhs_eff) >= 2
            and state.vocab(md[0]) == T.B_LOWER
            and 1 <= len(lhs_eff) - 1 <= 2
            and all(T.is_digit(state.vocab(md[s])) for s in lhs_eff[1:])):
        digits = [_digit_value(state, md[s]) for s in lhs_eff[1:]]
        k = len(digits)
        nv = sum(d * (10 ** (k - 1 - i)) for i, d in enumerate(digits))
        chars = list(str(10 ** k - 1 - nv).zfill(k))
        return _emit_insert_after_anchor(state, md, n, minus_after_eq_eff,
                                         chars, builder)

    # 1-4: -d=
    if (visible_no_ctrl and n == 3 and eq_eff == 2
            and state.vocab(md[0]) == T.MINUS
            and T.is_digit(state.vocab(md[1]))):
        d = _digit_value(state, md[1])
        if d == 0:
            chars = ['0']
        else:
            chars = ['b', str(10 - d)]
        return _emit_insert_after_anchor(state, md, n, eq_eff, chars, builder)

    # 1-5: d+b=
    if (visible_no_ctrl and n == 4 and eq_eff == 3
            and T.is_digit(state.vocab(md[0]))
            and state.vocab(md[1]) == T.PLUS
            and state.vocab(md[2]) == T.B_LOWER):
        d = _digit_value(state, md[0])
        if d == 0:
            chars = ['b', '9']
        else:
            chars = [str(d - 1)]
        return _emit_insert_after_anchor(state, md, n, eq_eff, chars, builder)

    return None


_CHAR_TO_VOCAB = {
    '-': T.MINUS, '+': T.PLUS, 'b': T.B_LOWER, '=': T.EQ,
    **{str(i): T.DIGIT_0 + i for i in range(10)},
}


def _emit_insert_after_anchor(state, md, n, anchor_eff, chars, builder) -> ActionBuilder:
    if len(chars) > MAX_INSERTION:
        raise RuntimeError(
            f"subtract Rule 1: tried to insert {len(chars)} > "
            f"MAX_INSERTION={MAX_INSERTION}"
        )
    for s_eff, i in enumerate(md):
        builder.keep(s_eff, state.vocab(i), state.ctrl(i))
    for slot, ch in enumerate(chars, start=1):
        builder.insert(anchor_eff, slot,
                       _CHAR_TO_VOCAB[ch], [0] * state.ctl_size)
    builder.set_done(1)
    return builder


# ============================================================================
# Rule 2 — LHS starts with `b`
# ============================================================================

def _rule_2(state, md, n, eq_eff, lhs_eff, rhs_eff,
            b_left_effs, minus_after_eq_eff, builder) -> ActionBuilder:
    if not rhs_eff:
        return _rule_2_1(state, md, n, eq_eff, lhs_eff, b_left_effs, builder)

    raw_lhs_digits = [
        s for s in lhs_eff
        if T.is_digit(state.vocab(md[s])) and _no_bc(state, md[s])
    ]
    if raw_lhs_digits:
        return _rule_2_2_self_call(state, md, n, eq_eff, lhs_eff, rhs_eff,
                                   raw_lhs_digits, b_left_effs,
                                   minus_after_eq_eff, builder)

    any_b_digit = any(
        T.is_digit(state.vocab(md[s])) and _has_b(state, md[s])
        for s in range(n)
    )
    if any_b_digit:
        return _rule_2_2_retag_only(state, md, n, builder)

    return _rule_2_2_done(state, md, n, builder)


def _rule_2_1(state, md, n, eq_eff, lhs_eff, b_left_effs, builder) -> ActionBuilder:
    leftmost_b = b_left_effs[0] if b_left_effs else None
    a_set = {eq_eff}
    if leftmost_b is not None:
        a_set.add(leftmost_b)
    for offset in (1, 2):
        cand = eq_eff - offset
        if cand >= 0:
            a_set.add(cand)

    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        ctrl = list(state.ctrl(i))
        if s_eff in a_set:
            ctrl = [0] * state.ctl_size
            ctrl[T.CTRL_SELF_CALL] = 1
        elif s_eff == eq_eff:
            ctrl = [0] * state.ctl_size
        elif s_eff < eq_eff and T.is_digit(vocab):
            ctrl = [0] * state.ctl_size
        builder.replace(s_eff, vocab, ctrl)
    builder.set_done(0)
    return builder


def _rule_2_2_self_call(state, md, n, eq_eff, lhs_eff, rhs_eff,
                        raw_lhs_digits, b_left_effs, minus_after_eq_eff,
                        builder) -> ActionBuilder:
    a_set = set(raw_lhs_digits[-2:])
    if b_left_effs:
        a_set.add(b_left_effs[0])
    a_set.add(eq_eff)
    if minus_after_eq_eff is not None:
        a_set.add(minus_after_eq_eff)

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


def _rule_2_2_retag_only(state, md, n, builder) -> ActionBuilder:
    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        ctrl = list(state.ctrl(i))
        if T.is_digit(vocab) and ctrl[T.CTRL_RESULT] == 1:
            ctrl[T.CTRL_RESULT] = 0
            ctrl[T.CTRL_TAG_C] = 1
        builder.replace(s_eff, vocab, ctrl)
    builder.set_done(0)
    return builder


def _rule_2_2_done(state, md, n, builder) -> ActionBuilder:
    for s_eff, i in enumerate(md):
        builder.keep(s_eff, state.vocab(i), state.ctrl(i))
    builder.set_done(1)
    return builder


# ============================================================================
# Rule 3 — general
# ============================================================================

def _all_lhs_digits_c(state, md, lhs_eff) -> bool:
    saw_digit = False
    for s in lhs_eff:
        v = state.vocab(md[s])
        if T.is_digit(v):
            saw_digit = True
            if not _has_c(state, md[s]):
                return False
    return saw_digit


def _all_rhs_digits_b(state, md, rhs_eff) -> bool:
    saw_digit = False
    for s in rhs_eff:
        v = state.vocab(md[s])
        if T.is_digit(v):
            saw_digit = True
            if not _has_b(state, md[s]):
                return False
    return saw_digit


def _all_digits_c(state, md, n) -> bool:
    saw_digit = False
    for s in range(n):
        if T.is_digit(state.vocab(md[s])):
            saw_digit = True
            if not _has_c(state, md[s]):
                return False
    return saw_digit


def _all_lhs_digits_have_bc(state, md, lhs_eff) -> bool:
    for s in lhs_eff:
        if T.is_digit(state.vocab(md[s])) and _no_bc(state, md[s]):
            return False
    return True


def _rule_3(state, md, n, eq_eff, lhs_eff, rhs_eff, minus_left_eff,
            minus_after_eq_eff, plus_rhs_eff, b_rhs_first_eff,
            builder) -> ActionBuilder:
    visible_has_b = any(state.vocab(md[s]) == T.B_LOWER for s in range(n))
    visible_has_plus = any(state.vocab(md[s]) == T.PLUS for s in range(n))
    no_b_no_plus = not visible_has_b and not visible_has_plus

    # 3-1
    if (rhs_eff and no_b_no_plus
            and _all_lhs_digits_c(state, md, lhs_eff)
            and _all_rhs_digits_b(state, md, rhs_eff)):
        return _rule_3_1(state, md, n, builder)

    # 3-2
    if rhs_eff and no_b_no_plus and _all_digits_c(state, md, n):
        return _rule_3_2(state, md, n, builder)

    # 3-3
    if minus_after_eq_eff is not None:
        return _rule_3_3(state, md, n, lhs_eff, minus_after_eq_eff, builder)

    # 3-4
    if plus_rhs_eff is not None:
        return _rule_3_4(state, md, n, lhs_eff, plus_rhs_eff, builder)

    # 3-5: neither `-` right of `=` nor `+` on RHS.
    if not rhs_eff:
        return _rule_3_5_3(state, md, n, eq_eff, lhs_eff, minus_left_eff,
                           builder)

    if b_rhs_first_eff is not None:
        # 3-5-1
        if _all_lhs_digits_have_bc(state, md, lhs_eff):
            return _rule_3_5_x_1(state, md, n, lhs_eff, rhs_eff, builder)
        return _rule_3_5_x_2(state, md, n, eq_eff, lhs_eff, minus_left_eff,
                             insert_plus_at_eq=True, builder=builder)

    # 3-5-2
    if _all_lhs_digits_have_bc(state, md, lhs_eff):
        return _rule_3_5_x_1(state, md, n, lhs_eff, rhs_eff, builder)
    return _rule_3_5_x_2(state, md, n, eq_eff, lhs_eff, minus_left_eff,
                         insert_plus_at_eq=False, builder=builder)


def _rule_3_1(state, md, n, builder) -> ActionBuilder:
    delete_set = set()
    boundary_idxs: List[int] = []
    for s_eff in range(n):
        v = state.vocab(md[s_eff])
        if v == T.EQ or v == T.MINUS:
            boundary_idxs.append(s_eff)
    for b_idx in boundary_idxs:
        slot_digits: List[int] = []
        cur = b_idx + 1
        while cur < n:
            v = state.vocab(md[cur])
            if T.is_digit(v):
                slot_digits.append(cur)
                cur += 1
            else:
                break
        if len(slot_digits) <= 1:
            continue
        for s in slot_digits[:-1]:
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


def _rule_3_2(state, md, n, builder) -> ActionBuilder:
    for s_eff, i in enumerate(md):
        builder.keep(s_eff, state.vocab(i), state.ctrl(i))
    builder.set_done(1)
    return builder


def _rule_3_3(state, md, n, lhs_eff, minus_after_eq_eff, builder) -> ActionBuilder:
    a_set = {minus_after_eq_eff}
    after_minus_eff = minus_after_eq_eff + 1
    if after_minus_eff < n:
        a_set.add(after_minus_eff)

    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        ctrl = list(state.ctrl(i))
        if s_eff in a_set:
            ctrl = [0] * state.ctl_size
            ctrl[T.CTRL_SELF_CALL] = 1
        elif s_eff in lhs_eff and T.is_digit(vocab) and ctrl[T.CTRL_RESULT] == 1:
            ctrl[T.CTRL_RESULT] = 0
            ctrl[T.CTRL_TAG_C] = 1
        builder.replace(s_eff, vocab, ctrl)
    builder.set_done(0)
    return builder


def _rule_3_4(state, md, n, lhs_eff, plus_rhs_eff, builder) -> ActionBuilder:
    a_set = {plus_rhs_eff}
    if plus_rhs_eff - 1 >= 0:
        a_set.add(plus_rhs_eff - 1)
    if plus_rhs_eff + 1 < n:
        a_set.add(plus_rhs_eff + 1)

    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        ctrl = list(state.ctrl(i))
        if s_eff in a_set:
            ctrl = [0] * state.ctl_size
            ctrl[T.CTRL_SELF_CALL] = 1
        elif s_eff in lhs_eff and T.is_digit(vocab) and ctrl[T.CTRL_RESULT] == 1:
            ctrl[T.CTRL_RESULT] = 0
            ctrl[T.CTRL_TAG_C] = 1
        builder.replace(s_eff, vocab, ctrl)
    builder.set_done(0)
    return builder


def _rule_3_5_3(state, md, n, eq_eff, lhs_eff, minus_left_eff, builder) -> ActionBuilder:
    """3-5-3: RHS empty. (a) on rightmost digit on each side of LHS `-`,
    on the LHS `-` itself, and on `=`. A side with no digits contributes
    no pick.
    """
    if minus_left_eff is None:
        return _no_op(state, md, n, builder)

    left_digits = [s for s in lhs_eff
                   if s < minus_left_eff and T.is_digit(state.vocab(md[s]))]
    right_digits = [s for s in lhs_eff
                    if s > minus_left_eff and T.is_digit(state.vocab(md[s]))]
    a_set = {minus_left_eff, eq_eff}
    if left_digits:
        a_set.add(left_digits[-1])
    if right_digits:
        a_set.add(right_digits[-1])

    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        ctrl = list(state.ctrl(i))
        if s_eff in a_set:
            ctrl = [0] * state.ctl_size
            ctrl[T.CTRL_SELF_CALL] = 1
        builder.replace(s_eff, vocab, ctrl)
    builder.set_done(0)
    return builder


def _rule_3_5_x_1(state, md, n, lhs_eff, rhs_eff, builder) -> ActionBuilder:
    """3-5-1-1 / 3-5-2-1: (c) on LHS (b) digits; (a) on every RHS token."""
    a_set = set(rhs_eff)
    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        ctrl = list(state.ctrl(i))
        if s_eff in a_set:
            ctrl = [0] * state.ctl_size
            ctrl[T.CTRL_SELF_CALL] = 1
        elif s_eff in lhs_eff and T.is_digit(vocab) and ctrl[T.CTRL_RESULT] == 1:
            ctrl[T.CTRL_RESULT] = 0
            ctrl[T.CTRL_TAG_C] = 1
        builder.replace(s_eff, vocab, ctrl)
    builder.set_done(0)
    return builder


def _rule_3_5_x_2(state, md, n, eq_eff, lhs_eff, minus_left_eff,
                  insert_plus_at_eq: bool, builder) -> ActionBuilder:
    """3-5-1-2 / 3-5-2-2 — pick scheme with optional `+` insertion."""
    if minus_left_eff is None:
        return _no_op(state, md, n, builder)

    left_side = [s for s in lhs_eff
                 if s < minus_left_eff
                 and T.is_digit(state.vocab(md[s]))
                 and _no_bc(state, md[s])]
    right_side = [s for s in lhs_eff
                  if s > minus_left_eff
                  and T.is_digit(state.vocab(md[s]))
                  and _no_bc(state, md[s])]
    left_pick = left_side[-1] if left_side else None
    right_pick = right_side[-1] if right_side else None

    if left_pick is None and right_pick is None:
        return _no_op(state, md, n, builder)

    if left_pick is not None and right_pick is not None:
        a_set = {left_pick, right_pick, minus_left_eff, eq_eff}
    elif left_pick is not None:
        a_set = {left_pick, eq_eff}
    else:
        a_set = {right_pick, minus_left_eff, eq_eff}

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

    if insert_plus_at_eq:
        builder.insert(eq_eff, 1, T.PLUS, [0] * state.ctl_size)

    builder.set_done(0)
    return builder


def _no_op(state, md, n, builder) -> ActionBuilder:
    for s_eff, i in enumerate(md):
        builder.replace(s_eff, state.vocab(i), list(state.ctrl(i)))
    builder.set_done(0)
    return builder


# ============================================================================
# Sanity / finaliser
# ============================================================================

def _finalize(builder: ActionBuilder) -> ActionBuilder:
    done_val = builder._action[-1][0][0]
    if done_val == 1:
        for s in range(builder.n):
            for m in range(builder.M):
                if builder._action[s][m][1 + T.CTRL_SELF_CALL] == 1:
                    raise RuntimeError(
                        f"subtract_rule produced done=1 with (a)=1 at s={s}, m={m}"
                    )
    return builder
