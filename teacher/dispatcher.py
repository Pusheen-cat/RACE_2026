"""Inter-task dispatcher for the teacher-trajectory rules.

A Markov function of the visible tape that picks the right per-task rule
from the LHS vocab pattern alone (vocabs only, not control bits — so an
intermediate state like `1,2(b),3(b),=(b),...` still dispatches to
copy_rule).

Wired-up rules: copy (LHS is digits-only), division (LHS contains `/`,
`%`, or `_`), subtraction (one `-` with no other ops, or borrow `b`
tokens), addition (`digits + digits`), multiplication (`digits * digits`),
sequential (≥2 operators, digits/ops only), and sequential-paren
(balanced parentheses). Any other pattern raises NotImplementedError.
"""

from __future__ import annotations

from typing import List

from teacher import tokens as T
from teacher.action import ActionBuilder
from teacher.rules.add import add_rule
from teacher.rules.copy import copy_rule
from teacher.rules.division import division_rule
from teacher.rules.multiply import multiply_rule
from teacher.rules.sequential_paren_task import sequential_paren_task_rule
from teacher.rules.sequential_task import sequential_task_rule
from teacher.rules.subtract import subtract_rule
from teacher.state import TeacherState

_SEQ_OPS = (T.PLUS, T.MINUS, T.STAR, T.SLASH)


def _lhs_vocabs(state: TeacherState) -> List[int]:
    md = state.max_depth_indices()
    lhs: List[int] = []
    for i in md:
        v = state.vocab(i)
        if v == T.EQ:
            return lhs
        lhs.append(v)
    # No '=' visible → caller (runner) is responsible for auto-appending.
    return None  # type: ignore[return-value]


def _is_single_number(vocabs: List[int]) -> bool:
    if not vocabs:
        return False
    return all(T.is_digit(v) for v in vocabs)


def _is_add_pattern(vocabs: List[int]) -> bool:
    """LHS looks like 'digits + digits' with exactly one '+' and digits on
    both sides."""
    if not vocabs:
        return False
    if vocabs.count(T.PLUS) != 1:
        return False
    plus_idx = vocabs.index(T.PLUS)
    left = vocabs[:plus_idx]
    right = vocabs[plus_idx + 1:]
    if not left or not right:
        return False
    return all(T.is_digit(v) for v in left) and all(T.is_digit(v) for v in right)


def _is_sequential_paren_pattern(vocabs: List[int]) -> bool:
    """LHS belongs to sequential_paren_task: at least one '(' and one ')'
    on the LHS, and the parentheses are well-matched."""
    if not vocabs:
        return False
    if T.LPAREN not in vocabs or T.RPAREN not in vocabs:
        return False
    depth = 0
    for v in vocabs:
        if v == T.LPAREN:
            depth += 1
        elif v == T.RPAREN:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _is_sequential_pattern(vocabs: List[int]) -> bool:
    """LHS belongs to sequential_task: at least 2 operators from {+ - * /}
    on the LHS; LHS contains only digits and ops (no parens, no other
    tokens).
    """
    if not vocabs:
        return False
    if not all(v in _SEQ_OPS or T.is_digit(v) for v in vocabs):
        return False
    op_count = sum(1 for v in vocabs if v in _SEQ_OPS)
    return op_count >= 2


def _is_division_pattern(vocabs: List[int]) -> bool:
    """LHS belongs to division_task. Visible LHS contains at least one of
    `/`, `%`, `_` and is otherwise digits-only — no `+`, `-`, `*`, `b`."""
    if not vocabs:
        return False
    if not any(v in (T.SLASH, T.PERCENT, T.UNDERSCORE) for v in vocabs):
        return False
    if any(v in (T.PLUS, T.MINUS, T.STAR, T.B_LOWER) for v in vocabs):
        return False
    return all(
        v in (T.SLASH, T.PERCENT, T.UNDERSCORE) or T.is_digit(v)
        for v in vocabs
    )


def _is_multiply_pattern(vocabs: List[int]) -> bool:
    """LHS looks like 'digits * digits' with exactly one '*' and digits on
    both sides."""
    if not vocabs:
        return False
    if vocabs.count(T.STAR) != 1:
        return False
    if any(v in (T.PLUS, T.MINUS, T.SLASH, T.B_LOWER) for v in vocabs):
        return False
    star_idx = vocabs.index(T.STAR)
    left = vocabs[:star_idx]
    right = vocabs[star_idx + 1:]
    if not left or not right:
        return False
    return all(T.is_digit(v) for v in left) and all(T.is_digit(v) for v in right)


def _is_subtract_pattern(vocabs: List[int]) -> bool:
    """LHS belongs to subtract_task. Two cases:

    Case A: exactly one '-', no other operators (+, *, /), no 'b'.
    Case B: at least one 'b', no '-', '*', '/' — only digits / '+' / b allowed.
    """
    if not vocabs:
        return False
    has_minus = T.MINUS in vocabs
    has_plus = T.PLUS in vocabs
    has_star = T.STAR in vocabs
    has_slash = T.SLASH in vocabs
    has_b = T.B_LOWER in vocabs

    if (vocabs.count(T.MINUS) == 1 and not has_plus
            and not has_star and not has_slash and not has_b):
        return True

    if has_b and not has_minus and not has_star and not has_slash:
        return all(v == T.B_LOWER or v == T.PLUS or T.is_digit(v) for v in vocabs)

    return False


def can_dispatch(state: TeacherState) -> bool:
    """True iff ``dispatch_rule(state)`` would find a matching rule pattern.

    Runs only the seven `_is_*` pattern tests on the LHS of `=`; does NOT
    execute the chosen rule. Returns False for empty / `=`-less visible
    tape, or an LHS that matches no pattern.
    """
    lhs = _lhs_vocabs(state)
    if lhs is None:
        return False
    return (
        _is_single_number(lhs)
        or _is_division_pattern(lhs)
        or _is_subtract_pattern(lhs)
        or _is_add_pattern(lhs)
        or _is_multiply_pattern(lhs)
        or _is_sequential_pattern(lhs)
        or _is_sequential_paren_pattern(lhs)
    )


def dispatch_rule(state: TeacherState) -> ActionBuilder:
    lhs = _lhs_vocabs(state)
    if lhs is None:
        raise RuntimeError("dispatch_rule: no '=' visible (runner should auto-append)")

    if _is_single_number(lhs):
        return copy_rule(state)
    if _is_division_pattern(lhs):
        return division_rule(state)
    if _is_subtract_pattern(lhs):
        return subtract_rule(state)
    if _is_add_pattern(lhs):
        return add_rule(state)
    if _is_multiply_pattern(lhs):
        return multiply_rule(state)
    if _is_sequential_pattern(lhs):
        return sequential_task_rule(state)
    if _is_sequential_paren_pattern(lhs):
        return sequential_paren_task_rule(state)

    raise NotImplementedError(
        f"dispatch_rule: no rule for LHS pattern {T.decode(lhs)!r}"
    )
