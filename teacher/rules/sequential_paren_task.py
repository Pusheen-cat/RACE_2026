"""Teacher rule for sequential_paren_task (task id 60).

Direct, literal implementation of the user-provided spec. No analysis.
"""

from __future__ import annotations

from teacher import tokens as T
from teacher.action import ActionBuilder
from teacher.state import TeacherState


MAX_INSERTION = 3


def sequential_paren_task_rule(state: TeacherState) -> ActionBuilder:
    md = state.max_depth_indices()
    builder = ActionBuilder(len(md), state.ctl_size, MAX_INSERTION)

    eq_local = None
    for s_eff, i in enumerate(md):
        if state.vocab(i) == T.EQ:
            eq_local = s_eff
            break
    if eq_local is None:
        raise RuntimeError("sequential_paren_task_rule: no '=' on visible tape")

    lhs = list(range(eq_local))

    def vat(s_eff): return state.vocab(md[s_eff])
    def has_c(s_eff): return state.ctrl_at(md[s_eff], T.CTRL_TAG_C) == 1

    for s_eff, i in enumerate(md):
        builder.keep(s_eff, state.vocab(i), state.ctrl(i))

    # Look for a ')' on LHS that has (c).
    c_rparen = None
    for s in lhs:
        if vat(s) == T.RPAREN and has_c(s):
            c_rparen = s
            break

    if c_rparen is not None:
        # Find the rightmost matching '(' to the left of c_rparen.
        depth = 1
        match_lparen = None
        for s in range(c_rparen - 1, -1, -1):
            v = vat(s)
            if v == T.RPAREN:
                depth += 1
            elif v == T.LPAREN:
                depth -= 1
                if depth == 0:
                    match_lparen = s
                    break
        if match_lparen is None:
            raise RuntimeError(
                "sequential_paren_task_rule: no matching '(' for the (c)-tagged ')'"
            )

        # (a) on every token strictly between match_lparen and c_rparen.
        for s in range(match_lparen + 1, c_rparen):
            ctrl = list(state.ctrl(md[s]))
            ctrl[T.CTRL_SELF_CALL] = 1
            builder.replace(s, vat(s), ctrl)

        # Remove (mask) the matching '(' and the (c)-tagged ')'.
        builder.delete(match_lparen)
        builder.delete(c_rparen)

        builder.set_done(0)
        return builder

    # No ')' with (c): tag (c) on the first ')' that appears on LHS.
    first_rparen = None
    for s in lhs:
        if vat(s) == T.RPAREN:
            first_rparen = s
            break
    if first_rparen is None:
        raise RuntimeError(
            "sequential_paren_task_rule: no ')' on LHS — dispatcher should not have routed here"
        )

    ctrl = list(state.ctrl(md[first_rparen]))
    ctrl[T.CTRL_TAG_C] = 1
    builder.replace(first_rparen, T.RPAREN, ctrl)
    builder.set_done(0)
    return builder
