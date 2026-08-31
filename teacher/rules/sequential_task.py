"""Teacher rule for sequential_task (task ids 50 / 51).

Literal implementation of the user-supplied rule spec. No
analysis or interpretation; each branch corresponds 1:1 to a section in
the spec.

Applicability of this rule (enforced by the dispatcher):
  - exactly one `=` on the visible tape,
  - RHS of `=` contains only digit tokens,
  - LHS of `=` contains two or more operator symbols (`+`, `-`, `*`, `/`).
"""

from __future__ import annotations

from typing import List

from teacher import tokens as T
from teacher.action import ActionBuilder
from teacher.state import TeacherState


MAX_INSERTION = 3
_OPS = (T.PLUS, T.MINUS, T.STAR, T.SLASH)
_SIGN_OPS = (T.PLUS, T.MINUS)


def sequential_task_rule(state: TeacherState) -> ActionBuilder:
    md = state.max_depth_indices()
    n = len(md)
    builder = ActionBuilder(n, state.ctl_size, MAX_INSERTION)

    eq_eff = None
    for s_eff, i in enumerate(md):
        if state.vocab(i) == T.EQ:
            eq_eff = s_eff
            break
    if eq_eff is None:
        raise RuntimeError("sequential_task_rule: no '=' visible")

    lhs = list(range(eq_eff))
    rhs = list(range(eq_eff + 1, n))

    def vat(s): return state.vocab(md[s])
    def ctrl_at(s): return list(state.ctrl(md[s]))
    def has_b(s): return state.ctrl_at(md[s], T.CTRL_RESULT) == 1
    def has_c(s): return state.ctrl_at(md[s], T.CTRL_TAG_C) == 1

    # Default: keep each cell as-is. Per-rule branches override.
    for s_eff, i in enumerate(md):
        builder.keep(s_eff, state.vocab(i), state.ctrl(i))

    # ---- Rule 1: consecutive +/- on LHS ----
    consec_pairs: List[tuple] = []
    for a, b in zip(lhs, lhs[1:]):
        if vat(a) in _SIGN_OPS and vat(b) in _SIGN_OPS:
            consec_pairs.append((a, b))
    if consec_pairs:
        for (a, b) in consec_pairs:
            va, vb = vat(a), vat(b)
            if va == T.MINUS and vb == T.MINUS:
                # `--`: mask both, insert `+` after the first mask.
                builder.delete(a)
                builder.delete(b)
                builder.insert(a, 1, T.PLUS, [0] * state.ctl_size)
            elif va == T.PLUS:        # `++` or `+-`: delete the leading `+`.
                builder.delete(a)
            else:                      # `-+`: delete the `+`.
                builder.delete(b)
        builder.set_done(0)
        return builder

    # ---- Rule 2: `-0 other-op` pattern, mask the `-` ----
    to_mask: List[int] = []
    for s in lhs:
        if vat(s) == T.MINUS and (s + 1) < eq_eff and vat(s + 1) == T.DIGIT_0:
            # "other-operator" — any operator (excluding the `-` itself).
            # The spec phrasing also matches when the cell right after `0`
            # is a non-digit, non-eq token (e.g., another op). We mask the
            # `-` only when its immediate neighbours are not numeric on
            # either side (i.e., `-` is acting as a unary minus before
            # `0`, and what follows is another operator).
            left_is_numeric = (s - 1) >= 0 and T.is_digit(vat(s - 1))
            right_is_numeric = (s + 2) < eq_eff and T.is_digit(vat(s + 2))
            if not left_is_numeric and not right_is_numeric:
                to_mask.append(s)
    if to_mask:
        for s in to_mask:
            builder.delete(s)
        builder.set_done(0)
        return builder

    # ---- Rule 3: dispatch by (c)-on-op / `*` or `/` presence ----
    cond_ops = [
        s for s in lhs
        if vat(s) in _OPS and (s - 1) >= 0 and T.is_digit(vat(s - 1))
    ]
    c_ops = [s for s in lhs if vat(s) in _OPS and has_c(s)]
    has_star_or_slash = any(vat(s) in (T.STAR, T.SLASH) for s in lhs)

    # 3-1: (c)-marked operator exists on LHS.
    if c_ops:
        c_op = c_ops[0]
        # Expand left until another op-with-digit-left, else stop at start.
        l = c_op - 1
        while l >= 0:
            if vat(l) in _OPS and (l - 1) >= 0 and T.is_digit(vat(l - 1)):
                break
            l -= 1
        l_start = l + 1
        # Expand right until another op-with-digit-left, else stop just
        # before `=`.
        r = c_op + 1
        while r < eq_eff:
            if vat(r) in _OPS and (r - 1) >= 0 and T.is_digit(vat(r - 1)):
                break
            r += 1
        r_end_excl = r

        for s in range(l_start, r_end_excl):
            ctrl = ctrl_at(s)
            ctrl[T.CTRL_SELF_CALL] = 1
            builder.replace(s, vat(s), ctrl)
        builder.set_done(0)
        return builder

    # 3-2: no (c) op AND `*` or `/` exists on LHS.
    if has_star_or_slash:
        if len(cond_ops) == 1:
            if not rhs:
                # 3-2-1-1: (a) on every token except `-`.
                for s_eff, i in enumerate(md):
                    if state.vocab(i) == T.MINUS:
                        continue
                    ctrl = list(state.ctrl(i))
                    ctrl[T.CTRL_SELF_CALL] = 1
                    builder.replace(s_eff, state.vocab(i), ctrl)
                builder.set_done(0)
                return builder

            # 3-2-1-2: RHS has tokens. Insert `-` after `=` when LHS
            # `-`-count is one and RHS is not a single `0`. Retag (b)→(c)
            # on digits, done=1.
            minus_count_lhs = sum(1 for s in lhs if vat(s) == T.MINUS)
            rhs_is_single_zero = (
                len(rhs) == 1 and vat(rhs[0]) == T.DIGIT_0
            )
            if minus_count_lhs == 1 and not rhs_is_single_zero:
                builder.insert(eq_eff, 1, T.MINUS, [0] * state.ctl_size)
            for s in rhs:
                if T.is_digit(vat(s)) and has_b(s):
                    ctrl = ctrl_at(s)
                    ctrl[T.CTRL_RESULT] = 0
                    ctrl[T.CTRL_TAG_C] = 1
                    builder.replace(s, vat(s), ctrl)
            builder.set_done(1)
            return builder

        # 3-2-2: 2+ ops satisfy the condition.
        leftmost_md = next(
            (s for s in lhs if vat(s) in (T.STAR, T.SLASH)),
            None,
        )
        if leftmost_md is None:
            raise RuntimeError(
                "sequential_task_rule (3-2-2): expected * or / on LHS, found none"
            )
        ctrl = ctrl_at(leftmost_md)
        ctrl[T.CTRL_TAG_C] = 1
        builder.replace(leftmost_md, vat(leftmost_md), ctrl)
        builder.set_done(0)
        return builder

    # 3-3: no (c) op, no `*`/`/`.
    if len(cond_ops) == 1:
        if not rhs:
            # 3-3-1-1: convert rightmost LHS sign-op (+↔-); (a) on every
            # token except the first.
            sign_ops_lhs = [s for s in lhs if vat(s) in _SIGN_OPS]
            if not sign_ops_lhs:
                raise RuntimeError(
                    "sequential_task_rule (3-3-1-1): expected +/- on LHS, none"
                )
            target_op = sign_ops_lhs[-1]
            new_target_v = T.MINUS if vat(target_op) == T.PLUS else T.PLUS
            for s_eff, i in enumerate(md):
                ctrl = list(state.ctrl(i))
                v = new_target_v if s_eff == target_op else state.vocab(i)
                if s_eff == 0:
                    builder.replace(s_eff, v, ctrl)
                    continue
                ctrl[T.CTRL_SELF_CALL] = 1
                builder.replace(s_eff, v, ctrl)
            builder.set_done(0)
            return builder

        first_rhs = rhs[0]
        if T.is_digit(vat(first_rhs)):
            # 3-3-1-2: (a) on every numeric with (b); insert `-` after `=`
            # only when the post-`=` number is non-zero; done.
            for s in lhs + rhs:
                if T.is_digit(vat(s)) and has_b(s):
                    ctrl = ctrl_at(s)
                    ctrl[T.CTRL_SELF_CALL] = 1
                    builder.replace(s, vat(s), ctrl)
            number_is_nonzero = False
            for s in rhs:
                if not T.is_digit(vat(s)):
                    break
                if vat(s) != T.DIGIT_0:
                    number_is_nonzero = True
                    break
            if number_is_nonzero:
                builder.insert(eq_eff, 1, T.MINUS, [0] * state.ctl_size)
            builder.set_done(1)
            return builder

        if vat(first_rhs) == T.MINUS:
            # 3-3-1-3: (a) on every numeric with (b); delete the `-`
            # immediately right of `=`; done.
            for s in lhs + rhs:
                if T.is_digit(vat(s)) and has_b(s):
                    ctrl = ctrl_at(s)
                    ctrl[T.CTRL_SELF_CALL] = 1
                    builder.replace(s, vat(s), ctrl)
            builder.delete(first_rhs)
            builder.set_done(1)
            return builder

    # 3-3-2: not 3-3-1 — assign (c) to the leftmost op satisfying the
    # condition.
    if cond_ops:
        target = cond_ops[0]
        ctrl = ctrl_at(target)
        ctrl[T.CTRL_TAG_C] = 1
        builder.replace(target, vat(target), ctrl)
        builder.set_done(0)
        return builder

    raise RuntimeError(
        "sequential_task_rule: no branch matched. "
        f"visible='{T.decode([state.vocab(i) for i in md])}'"
    )