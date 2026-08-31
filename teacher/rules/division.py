"""Teacher rule for `division_task` (task ids 40 / 41).

Direct translation of the user-supplied division spec for this teacher
method. Visible-tape Markov rule. (a)/(b)/(c) refer to ctrl bits
CTRL_SELF_CALL / CTRL_RESULT / CTRL_TAG_C.

The spec uses two scratchpad symbols:
  '%'  — modulo / remainder marker (T.PERCENT)
  '_'  — operand bracket marker  (T.UNDERSCORE)

Top-level routing on the visible tape:
  Section 1 — literal memorisation shapes that emit done=1 immediately.
              1-1, 1-2, 1-3, 1-4, 1-5.
  Section 2 — transitional shape `d%MULTI=` → `_d%MULTI=` (done=0).
  Section 3 — visible LHS starts with `_`.
  Section 4 — visible LHS contains `%` (no `_`, no `/`).
  Section 5 — visible LHS contains `/`.

Done-timing notes:
  - Section 4's "done" branch is split: 4-5 (all digits (c) → done=1)
    fires before 4-4 (all digits (b)/(c) → retag (b)→(c) only,
    done=0), so retagging and termination never share a step.
  - Section 5's "done" branch 5-6 (all digits (c) → done=1) fires
    before the structural rules; the 5-4-1 / 5-5-1 transitions then
    no longer emit done=1 themselves (they retag + restructure and
    let 5-6 close on the next step).
  - 3-2-1-1 is a pure transition step (retag + eq2→`%` + drop front
    `_`); the closing `done=1` is emitted later by Section 4's 4-5
    once dispatch routes to Section 4.

Targets from `_gen_two_op_task(L, '/')` are decimal-only; `%` and `_`
must never appear in the final tape.
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


def _ctrl_a() -> List[int]:
    return [1 if k == T.CTRL_SELF_CALL else 0 for k in range(3)]


def _ctrl_c() -> List[int]:
    return [1 if k == T.CTRL_TAG_C else 0 for k in range(3)]


def _ctrl_zero() -> List[int]:
    return [0] * 3


# ============================================================================
# Top-level dispatch
# ============================================================================

def division_rule(state: TeacherState) -> ActionBuilder:
    md = state.max_depth_indices()
    n = len(md)
    builder = ActionBuilder(n, state.ctl_size, MAX_INSERTION)

    vocabs = [state.vocab(i) for i in md]
    eq_positions = [s for s, v in enumerate(vocabs) if v == T.EQ]
    if not eq_positions:
        raise RuntimeError("division_rule: no '=' visible (auto-append should have run)")

    # ---- Section 1: memorisation. ----
    r1 = _try_rule_1(state, md, n, vocabs, builder)
    if r1 is not None:
        return r1

    # ---- Section 2: transitional `d%MULTI=` → `_d%MULTI=`. ----
    r2 = _try_rule_2(state, md, n, vocabs, builder)
    if r2 is not None:
        return r2

    # ---- Sections 3/4/5 dispatch by leading marker on the visible LHS. ----
    eq_eff = eq_positions[0]
    lhs = vocabs[:eq_eff]

    if T.UNDERSCORE in lhs:
        return _rule_3(state, md, n, vocabs, eq_positions, builder)

    if T.PERCENT in lhs and T.SLASH not in lhs:
        return _rule_4(state, md, n, vocabs, eq_positions, builder)

    if T.SLASH in lhs:
        return _rule_5(state, md, n, vocabs, eq_positions, builder)

    raise RuntimeError(
        f"division_rule: unrecognised LHS pattern {T.decode(lhs)!r} "
        f"(visible={T.decode(vocabs)!r})"
    )


# ============================================================================
# Section 1 — memorisation patterns (done=1).
# ============================================================================

def _try_rule_1(state, md, n, vocabs, builder) -> Optional[ActionBuilder]:
    if not all(_no_ctrl(state, i) for i in md):
        return None

    # 1-1: 'd%=%'
    if (n == 4 and T.is_digit(vocabs[0]) and vocabs[1] == T.PERCENT
            and vocabs[2] == T.EQ and vocabs[3] == T.PERCENT):
        return _rule_1_1(state, md, n, vocabs, builder)

    # 1-2: '_='
    if n == 2 and vocabs[0] == T.UNDERSCORE and vocabs[1] == T.EQ:
        return _rule_1_2(state, md, n, vocabs, builder)

    # 1-3: 'd_='
    if (n == 3 and T.is_digit(vocabs[0]) and vocabs[1] == T.UNDERSCORE
            and vocabs[2] == T.EQ):
        return _rule_1_3(state, md, n, vocabs, builder)

    # 1-4: suffix '/=_%'
    if (n >= 4 and vocabs[-4] == T.SLASH and vocabs[-3] == T.EQ
            and vocabs[-2] == T.UNDERSCORE and vocabs[-1] == T.PERCENT):
        return _rule_1_4(state, md, n, vocabs, builder)

    # 1-5: suffix '/=.%'
    if (n >= 4 and vocabs[-4] == T.SLASH and vocabs[-3] == T.EQ
            and vocabs[-2] == T.DOT and vocabs[-1] == T.PERCENT):
        return _rule_1_5(state, md, n, vocabs, builder)

    return None


def _rule_1_1(state, md, n, vocabs, builder) -> ActionBuilder:
    """1-1) d%=% → d%=d%."""
    d = _digit_value(state, md[0])
    builder.keep(0, vocabs[0], _ctrl_zero())
    builder.keep(1, T.PERCENT, _ctrl_zero())
    builder.keep(2, T.EQ, _ctrl_zero())
    builder.replace(3, T.DIGIT_0 + d, _ctrl_zero())
    builder.insert(3, 1, T.PERCENT, _ctrl_zero())
    builder.set_done(1)
    return builder


def _rule_1_2(state, md, n, vocabs, builder) -> ActionBuilder:
    """1-2) _= → _=0_."""
    builder.keep(0, T.UNDERSCORE, _ctrl_zero())
    builder.keep(1, T.EQ, _ctrl_zero())
    builder.insert(1, 1, T.DIGIT_0, _ctrl_zero())
    builder.insert(1, 2, T.UNDERSCORE, _ctrl_zero())
    builder.set_done(1)
    return builder


def _rule_1_3(state, md, n, vocabs, builder) -> ActionBuilder:
    """1-3) d_= → d_=(d+1)_."""
    d = _digit_value(state, md[0])
    builder.keep(0, vocabs[0], _ctrl_zero())
    builder.keep(1, T.UNDERSCORE, _ctrl_zero())
    builder.keep(2, T.EQ, _ctrl_zero())

    new_digits = list(str(d + 1))
    for slot, ch in enumerate(new_digits, start=1):
        builder.insert(2, slot, T.DIGIT_0 + int(ch), _ctrl_zero())
    builder.insert(2, len(new_digits) + 1, T.UNDERSCORE, _ctrl_zero())
    builder.set_done(1)
    return builder


def _rule_1_4(state, md, n, vocabs, builder) -> ActionBuilder:
    """1-4) suffix /=_% → /=._0%."""
    suffix_start = n - 4
    for s_eff, i in enumerate(md):
        if s_eff < suffix_start:
            builder.keep(s_eff, vocabs[s_eff], state.ctrl(i))
    builder.keep(suffix_start, T.SLASH, state.ctrl(md[suffix_start]))
    builder.keep(suffix_start + 1, T.EQ, state.ctrl(md[suffix_start + 1]))
    builder.replace(suffix_start + 2, T.DOT, _ctrl_zero())
    builder.insert(suffix_start + 2, 1, T.UNDERSCORE, _ctrl_zero())
    builder.replace(suffix_start + 3, T.DIGIT_0, _ctrl_zero())
    builder.insert(suffix_start + 3, 1, T.PERCENT, _ctrl_zero())
    builder.set_done(1)
    return builder


def _rule_1_5(state, md, n, vocabs, builder) -> ActionBuilder:
    """1-5) suffix /=.% → /=.0%."""
    suffix_start = n - 4
    for s_eff, i in enumerate(md):
        if s_eff < suffix_start:
            builder.keep(s_eff, vocabs[s_eff], state.ctrl(i))
    builder.keep(suffix_start, T.SLASH, state.ctrl(md[suffix_start]))
    builder.keep(suffix_start + 1, T.EQ, state.ctrl(md[suffix_start + 1]))
    builder.keep(suffix_start + 2, T.DOT, state.ctrl(md[suffix_start + 2]))
    builder.replace(suffix_start + 3, T.DIGIT_0, _ctrl_zero())
    builder.insert(suffix_start + 3, 1, T.PERCENT, _ctrl_zero())
    builder.set_done(1)
    return builder


# ============================================================================
# Section 2 — transitional `d%MULTI=` → `_d%MULTI=` (done=0).
# ============================================================================

def _try_rule_2(state, md, n, vocabs, builder) -> Optional[ActionBuilder]:
    if not all(_no_ctrl(state, i) for i in md):
        return None

    if (n >= 5 and T.is_digit(vocabs[0]) and vocabs[1] == T.PERCENT
            and vocabs[-1] == T.EQ
            and all(T.is_digit(v) for v in vocabs[2:-1])
            and (n - 3) >= 2):
        d = _digit_value(state, md[0])
        builder.replace(0, T.UNDERSCORE, _ctrl_zero())
        builder.insert(0, 1, T.DIGIT_0 + d, _ctrl_zero())
        for s_eff in range(1, n):
            builder.keep(s_eff, vocabs[s_eff], state.ctrl(md[s_eff]))
        builder.set_done(0)
        return builder

    return None


# ============================================================================
# Section 3 — visible LHS starts with `_`.
# ============================================================================

def _rule_3(state, md, n, vocabs, eq_positions, builder) -> ActionBuilder:
    if len(eq_positions) == 1:
        return _rule_3_1(state, md, n, vocabs, eq_positions, builder)
    return _rule_3_2(state, md, n, vocabs, eq_positions, builder)


def _rule_3_1(state, md, n, vocabs, eq_positions, builder) -> ActionBuilder:
    """One '=' on the visible tape. LHS = _digits%digits."""
    eq_eff = eq_positions[0]
    rhs = vocabs[eq_eff + 1:]

    underscore_eff = vocabs.index(T.UNDERSCORE)
    percent_eff = None
    for s in range(underscore_eff + 1, eq_eff):
        if vocabs[s] == T.PERCENT:
            percent_eff = s
            break
    if percent_eff is None:
        raise RuntimeError("_rule_3_1: no '%' on LHS after '_'")

    # 3-1-1: RHS empty.
    if not rhs:
        a_set = {s for s in range(underscore_eff + 1, percent_eff)
                 if T.is_digit(vocabs[s])}
        a_set.add(eq_eff)
        return _emit_a_set(state, md, n, vocabs, a_set, builder)

    # 3-1-2: RHS only digits.
    if all(T.is_digit(v) for v in rhs):
        a_set = {0, eq_eff}
        c_set = {s for s in range(eq_eff)
                 if T.is_digit(vocabs[s]) and _has_b(state, md[s])}
        return _emit_with_sets(state, md, n, vocabs, a_set, c_set, builder)

    # 3-1-3: RHS has both digits and '_'.
    if T.UNDERSCORE in rhs:
        a_set = {s for s in range(percent_eff + 1, eq_eff)
                 if T.is_digit(vocabs[s])}
        return _emit_with_sets(
            state, md, n, vocabs, a_set, set(), builder,
            extra_inserts=[(n - 1, 1, T.EQ, _ctrl_a())],
        )

    raise RuntimeError(f"_rule_3_1: unhandled RHS shape {T.decode(rhs)!r}")


def _rule_3_2(state, md, n, vocabs, eq_positions, builder) -> ActionBuilder:
    """Two '='s on the visible tape."""
    eq1, eq2 = eq_positions[0], eq_positions[1]
    between = vocabs[eq1 + 1:eq2]
    after = vocabs[eq2 + 1:]

    all_between_digits_c = all(
        _has_c(state, md[eq1 + 1 + k]) for k, v in enumerate(between)
        if T.is_digit(v)
    ) and any(T.is_digit(v) for v in between)

    # 3-2-1: RHS of eq2 only digits.
    if after and all(T.is_digit(v) for v in after):
        # 3-2-1-1: every between digit has (c).
        # Safedone: retag + restructure but do NOT emit done=1.
        if all_between_digits_c:
            first_underscore_eff = None
            for s in range(n):
                if vocabs[s] == T.UNDERSCORE:
                    first_underscore_eff = s
                    break
            for s_eff, i in enumerate(md):
                vocab = state.vocab(i)
                ctrl = list(state.ctrl(i))
                if s_eff == first_underscore_eff:
                    builder.delete(s_eff)
                    continue
                if s_eff == eq2:
                    builder.replace(s_eff, T.PERCENT, _ctrl_zero())
                    continue
                if eq1 < s_eff < eq2:
                    ctrl = _ctrl_c()
                else:
                    if ctrl[T.CTRL_RESULT] == 1:
                        ctrl[T.CTRL_RESULT] = 0
                        ctrl[T.CTRL_TAG_C] = 1
                builder.replace(s_eff, vocab, ctrl)
            builder.set_done(0)
            return builder

        # 3-2-1-2: some between digit lacks (c).
        underscore_left_of_eq2 = None
        for s in range(eq2 - 1, -1, -1):
            if vocabs[s] == T.UNDERSCORE:
                underscore_left_of_eq2 = s
                break
        a_set = set()
        if underscore_left_of_eq2 is not None:
            a_set = {s for s in range(underscore_left_of_eq2 + 1, eq2)
                     if T.is_digit(vocabs[s])}
        a_set.add(eq2)
        return _emit_with_sets(
            state, md, n, vocabs, a_set, set(), builder,
            extra_inserts=[(eq2, 1, T.MINUS, _ctrl_zero())],
        )

    # 3-2-2: RHS of eq2 is 'digits-digits'.
    if (after and after.count(T.MINUS) == 1
            and all(T.is_digit(v) or v == T.MINUS for v in after)
            and after.index(T.MINUS) >= 1
            and after.index(T.MINUS) <= len(after) - 2):
        c_set, a_set = set(), set()
        for k, v in enumerate(between):
            s = eq1 + 1 + k
            if _has_b(state, md[s]):
                c_set.add(s)
            else:
                a_set.add(s)
        a_set.add(eq2)
        return _emit_with_sets(state, md, n, vocabs, a_set, c_set, builder)

    # 3-2-3: '_' and '-' both right of eq2, NOT adjacent.
    if T.UNDERSCORE in after and T.MINUS in after:
        und_right_eff = eq2 + 1 + after.index(T.UNDERSCORE)
        min_right_eff = eq2 + 1 + after.index(T.MINUS)
        if abs(und_right_eff - min_right_eff) > 1:
            c_set = set()
            for k, _v in enumerate(between):
                s = eq1 + 1 + k
                if _has_b(state, md[s]):
                    c_set.add(s)
            a_set = {s for s in range(und_right_eff + 1, n)}
            return _emit_with_sets(state, md, n, vocabs, a_set, c_set, builder)

    # 3-2-4: exactly one '_' on RHS of eq2 with otherwise digits.
    if (after and after.count(T.UNDERSCORE) == 1
            and all(v == T.UNDERSCORE or T.is_digit(v) for v in after)):
        for s_eff, i in enumerate(md):
            vocab = state.vocab(i)
            ctrl = list(state.ctrl(i))
            if eq1 < s_eff <= eq2:
                builder.delete(s_eff)
            else:
                builder.replace(s_eff, vocab, ctrl)
        builder.set_done(0)
        return builder

    # 3-2-5: '_' and '-' adjacent on RHS of eq2.
    if (T.UNDERSCORE in after and T.MINUS in after
            and abs(after.index(T.UNDERSCORE) - after.index(T.MINUS)) == 1):
        percent_eff = None
        for s in range(eq1):
            if vocabs[s] == T.PERCENT:
                percent_eff = s
                break
        a_set = set()
        if percent_eff is not None:
            a_set = {s for s in range(percent_eff + 1, eq1)
                     if T.is_digit(vocabs[s])}
        a_set.add(eq2)
        for s_eff, i in enumerate(md):
            vocab = state.vocab(i)
            ctrl = list(state.ctrl(i))
            if s_eff > eq2:
                builder.delete(s_eff)
            elif s_eff in a_set:
                builder.replace(s_eff, vocab, _ctrl_a())
            else:
                builder.replace(s_eff, vocab, ctrl)
        builder.set_done(0)
        return builder

    raise RuntimeError(f"_rule_3_2: unhandled visible {T.decode(vocabs)!r}")


# ============================================================================
# Section 4 — visible LHS contains '%' (no '_', no '/').
# ============================================================================

def _rule_4(state, md, n, vocabs, eq_positions, builder) -> ActionBuilder:
    eq_eff = eq_positions[0]
    lhs = vocabs[:eq_eff]
    rhs = vocabs[eq_eff + 1:]
    percent_eff = lhs.index(T.PERCENT)

    digit_eff = [s for s in range(n) if T.is_digit(vocabs[s])]
    all_digits_c = digit_eff and all(_has_c(state, md[s]) for s in digit_eff)
    all_digits_settled = digit_eff and all(
        _has_b(state, md[s]) or _has_c(state, md[s]) for s in digit_eff
    )

    # 4-5: every digit (c) → done=1.
    if all_digits_c:
        for s_eff, i in enumerate(md):
            builder.keep(s_eff, state.vocab(i), state.ctrl(i))
        builder.set_done(1)
        return builder

    # 4-4: every digit (b)/(c), but not all (c) → retag (b)→(c), done=0.
    if all_digits_settled:
        for s_eff, i in enumerate(md):
            vocab = state.vocab(i)
            ctrl = list(state.ctrl(i))
            if T.is_digit(vocab) and ctrl[T.CTRL_RESULT] == 1:
                ctrl[T.CTRL_RESULT] = 0
                ctrl[T.CTRL_TAG_C] = 1
            builder.replace(s_eff, vocab, ctrl)
        builder.set_done(0)
        return builder

    # 4-1: RHS empty.
    if not rhs:
        digits_left_of_pct = [s for s in range(percent_eff)
                              if T.is_digit(vocabs[s])]
        if len(digits_left_of_pct) == 1:
            leftmost_eff = 0
            orig_vocab = vocabs[leftmost_eff]
            builder.replace(leftmost_eff, T.UNDERSCORE, _ctrl_zero())
            builder.insert(leftmost_eff, 1, orig_vocab, _ctrl_zero())
            for s_eff in range(1, n):
                builder.keep(s_eff, vocabs[s_eff], state.ctrl(md[s_eff]))
            builder.set_done(0)
            return builder

        a_set = set()
        for s in range(eq_eff):
            if s == percent_eff - 1 and T.is_digit(vocabs[s]):
                continue
            a_set.add(s)
        a_set.add(eq_eff)
        return _emit_a_set(state, md, n, vocabs, a_set, builder)

    # 4-2: raw LHS digit + RHS non-empty.
    has_raw_lhs_digit = any(
        T.is_digit(vocabs[s]) and _no_ctrl(state, md[s])
        for s in range(eq_eff)
    )
    if has_raw_lhs_digit and rhs:
        a_set = set()
        for s in range(n):
            if vocabs[s] == T.PERCENT:
                a_set.add(s)
        a_set.add(eq_eff)
        if percent_eff - 1 >= 0 and T.is_digit(vocabs[percent_eff - 1]):
            a_set.add(percent_eff - 1)
        underscore_eff = None
        for s in range(n):
            if vocabs[s] == T.UNDERSCORE:
                underscore_eff = s
                break
        if underscore_eff is None:
            c_set = set()
        else:
            c_set = {s for s in range(underscore_eff)
                     if T.is_digit(vocabs[s]) and _has_b(state, md[s])}
        return _emit_with_sets(state, md, n, vocabs, a_set, c_set, builder)

    # 4-3: all LHS digits (b)/(c), raw RHS digit exists.
    lhs_digits = [s for s in range(eq_eff) if T.is_digit(vocabs[s])]
    all_lhs_settled = all(
        _has_b(state, md[s]) or _has_c(state, md[s]) for s in lhs_digits
    )
    has_raw_rhs_digit = any(
        T.is_digit(vocabs[s]) and _no_ctrl(state, md[s])
        for s in range(eq_eff + 1, n)
    )
    if all_lhs_settled and has_raw_rhs_digit:
        und_right = None
        for s in range(eq_eff + 1, n):
            if vocabs[s] == T.UNDERSCORE:
                und_right = s
                break
        delete_zero_eff = None
        if (und_right is not None
                and und_right + 1 < n
                and vocabs[und_right + 1] == T.DIGIT_0):
            delete_zero_eff = und_right + 1
        a_set = set()
        if und_right is not None:
            a_set = {s for s in range(und_right, n) if s != delete_zero_eff}
        c_set = {s for s in lhs_digits if _has_b(state, md[s])}
        for s_eff, i in enumerate(md):
            vocab = state.vocab(i)
            ctrl = list(state.ctrl(i))
            if s_eff == delete_zero_eff:
                builder.delete(s_eff)
                continue
            if s_eff in a_set:
                ctrl = _ctrl_a()
            elif s_eff in c_set:
                ctrl = _ctrl_c()
            builder.replace(s_eff, vocab, ctrl)
        builder.set_done(0)
        return builder

    raise RuntimeError(f"_rule_4: unhandled visible {T.decode(vocabs)!r}")


# ============================================================================
# Section 5 — visible LHS contains '/'.
# ============================================================================

def _rule_5(state, md, n, vocabs, eq_positions, builder) -> ActionBuilder:
    eq_eff = eq_positions[0]
    lhs = vocabs[:eq_eff]
    rhs = vocabs[eq_eff + 1:]
    slash_eff = lhs.index(T.SLASH)

    digit_eff = [s for s in range(n) if T.is_digit(vocabs[s])]
    all_digits_c = digit_eff and all(_has_c(state, md[s]) for s in digit_eff)

    # 5-6: all digits (c) → done=1.
    if all_digits_c:
        for s_eff, i in enumerate(md):
            builder.keep(s_eff, state.vocab(i), state.ctrl(i))
        builder.set_done(1)
        return builder

    # 5-1: RHS empty.
    if not rhs:
        a_set = {s for s in range(slash_eff + 1, eq_eff)}
        a_set.add(eq_eff)
        return _emit_a_set(state, md, n, vocabs, a_set, builder)

    # 5-5: RHS has '.', '%', and '_'.
    if T.DOT in rhs and T.PERCENT in rhs and T.UNDERSCORE in rhs:
        dot_eff = eq_eff + 1 + rhs.index(T.DOT)
        und_eff = eq_eff + 1 + rhs.index(T.UNDERSCORE)

        # 5-5-1: tokens between '.' and '_' >= 3 → transition
        # (retag + drop '_' tail + drop leading 0s). NO done=1 here;
        # the next step's 5-6 closes the trajectory.
        if und_eff > dot_eff and (und_eff - dot_eff - 1) >= 3:
            digits_between_eq_dot = [
                s for s in range(eq_eff + 1, dot_eff) if T.is_digit(vocabs[s])
            ]
            delete_zeros = set()
            for s in digits_between_eq_dot[:-1]:
                if vocabs[s] == T.DIGIT_0:
                    delete_zeros.add(s)
                else:
                    break
            for s_eff, i in enumerate(md):
                vocab = state.vocab(i)
                ctrl = list(state.ctrl(i))
                if s_eff >= und_eff:
                    builder.delete(s_eff)
                    continue
                if s_eff in delete_zeros:
                    builder.delete(s_eff)
                    continue
                if (s_eff < und_eff and T.is_digit(vocab)
                        and ctrl[T.CTRL_RESULT] == 1):
                    ctrl[T.CTRL_RESULT] = 0
                    ctrl[T.CTRL_TAG_C] = 1
                builder.replace(s_eff, vocab, ctrl)
            builder.set_done(0)
            return builder

        # 5-5-2-1: at least one '.' has (b) → (a) on '_' and right.
        has_dot_b = any(
            state.vocab(md[s]) == T.DOT and _has_b(state, md[s])
            for s in range(n)
        )
        if has_dot_b:
            a_set = {s for s in range(und_eff, n)}
            return _emit_with_sets(state, md, n, vocabs, a_set, set(), builder)

        # 5-5-2-2: no '.' has (b) → (a) on '/', '=', '.', '%';
        # (c) on (b) digits left of '_'.
        a_set = set()
        for s in range(n):
            if vocabs[s] in (T.SLASH, T.EQ, T.DOT, T.PERCENT):
                a_set.add(s)
        c_set = {s for s in range(und_eff)
                 if T.is_digit(vocabs[s]) and _has_b(state, md[s])}
        return _emit_with_sets(state, md, n, vocabs, a_set, c_set, builder)

    # 5-4: RHS has '_' and '%' but NOT '.'.
    if T.UNDERSCORE in rhs and T.PERCENT in rhs and T.DOT not in rhs:
        und_eff = eq_eff + 1 + rhs.index(T.UNDERSCORE)
        pct_eff = None
        for s in range(und_eff + 1, n):
            if vocabs[s] == T.PERCENT:
                pct_eff = s
                break

        # 5-4-1: single 0 between '_' and '%' → transition.
        if pct_eff is not None:
            tokens_between_und_pct = vocabs[und_eff + 1:pct_eff]
            if (len(tokens_between_und_pct) == 1
                    and tokens_between_und_pct[0] == T.DIGIT_0):
                digits_between_eq_und = [
                    s for s in range(eq_eff + 1, und_eff)
                    if T.is_digit(vocabs[s])
                ]
                delete_zeros = set()
                for s in digits_between_eq_und[:-1]:
                    if vocabs[s] == T.DIGIT_0:
                        delete_zeros.add(s)
                    else:
                        break

                for s_eff, i in enumerate(md):
                    vocab = state.vocab(i)
                    ctrl = list(state.ctrl(i))
                    if s_eff >= und_eff:
                        builder.delete(s_eff)
                        continue
                    if s_eff in delete_zeros:
                        builder.delete(s_eff)
                        continue
                    if T.is_digit(vocab) and ctrl[T.CTRL_RESULT] == 1:
                        ctrl[T.CTRL_RESULT] = 0
                        ctrl[T.CTRL_TAG_C] = 1
                    builder.replace(s_eff, vocab, ctrl)
                builder.set_done(0)
                return builder

        # 5-4-2: (a) on '/', '=', '_', '%'; (c) on (b) digits left of '_'.
        a_set = set()
        for s in range(n):
            if vocabs[s] in (T.SLASH, T.EQ, T.UNDERSCORE, T.PERCENT):
                a_set.add(s)
        c_set = {s for s in range(und_eff)
                 if T.is_digit(vocabs[s]) and _has_b(state, md[s])}
        return _emit_with_sets(state, md, n, vocabs, a_set, c_set, builder)

    # 5-3: RHS is 'digits%digits'.
    if T.PERCENT in rhs and all(T.is_digit(v) or v == T.PERCENT for v in rhs):
        a_set = {s for s in range(eq_eff + 1, n)}
        c_set = {s for s in range(eq_eff)
                 if T.is_digit(vocabs[s]) and _has_b(state, md[s])}
        return _emit_with_sets(state, md, n, vocabs, a_set, c_set, builder)

    # 5-2: RHS only digits, exists raw digit.
    if rhs and all(T.is_digit(v) for v in rhs):
        a_set = {s for s in range(slash_eff)}
        a_set.add(eq_eff)
        c_set = {s for s in range(eq_eff)
                 if T.is_digit(vocabs[s]) and _has_b(state, md[s])}
        return _emit_with_sets(
            state, md, n, vocabs, a_set, c_set, builder,
            extra_inserts=[(eq_eff, 1, T.PERCENT, _ctrl_zero())],
        )

    raise RuntimeError(f"_rule_5: unhandled visible {T.decode(vocabs)!r}")


# ============================================================================
# Helpers
# ============================================================================

def _emit_a_set(state, md, n, vocabs, a_set, builder) -> ActionBuilder:
    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        ctrl = list(state.ctrl(i))
        if s_eff in a_set:
            ctrl = _ctrl_a()
        builder.replace(s_eff, vocab, ctrl)
    builder.set_done(0)
    return builder


def _emit_with_sets(state, md, n, vocabs, a_set, c_set, builder,
                    extra_inserts=None) -> ActionBuilder:
    for s_eff, i in enumerate(md):
        vocab = state.vocab(i)
        ctrl = list(state.ctrl(i))
        if s_eff in a_set:
            ctrl = _ctrl_a()
        elif s_eff in c_set:
            ctrl = _ctrl_c()
        builder.replace(s_eff, vocab, ctrl)
    if extra_inserts:
        for (s_eff, slot, vocab, ctrl) in extra_inserts:
            builder.insert(s_eff, slot, vocab, ctrl)
    builder.set_done(0)
    return builder