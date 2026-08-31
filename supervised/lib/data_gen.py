"""Pure-CPU teacher-trajectory generators.

Each helper produces a list of per-step training tuples for a single task,
already in tensor-ready Python list form:

    (state_tokens, state_depth, action_nested, done_int)

  * state_tokens  : list[list[int]]  (S_total, 1 + ctl_size)  — full tape
  * state_depth   : list[int]        (S_total,)
  * action_nested : list[list[list[int]]]  (S_eff + 1, 1 + max_insertion,
                                             1 + ctl_size)
  * done_int      : 0 / 1

The state recorded is the state JUST BEFORE the action is taken (after the
runner's auto-eq step). Reapplying `teacher.engine.apply_action` to that
state with the recorded action reproduces the next state.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional, Tuple

from dataset.taskgenerator import TaskGenerator
from teacher import tokens as T
from teacher.dispatcher import dispatch_rule
from teacher.engine import TeacherError, apply_action
from teacher.runner import _auto_append_eq
from teacher.state import TeacherState


# Both the teacher rules and the model use 3 control bits ((a) self-call,
# (b) result, (c) tag). The reserved CTRL_TAG_D / CTRL_TAG_E indices in
# teacher.tokens are not used by any rule.
CTL_SIZE = 3

# The division task only uses cases whose (3-decimal truncated) result is at
# least this value — the smallest value the targets can express. Applied to
# BOTH training and evaluation: cells with l2 > l1 + 2 cannot satisfy it and
# are pruned from the grids; within the remaining cells the operand pairs
# are redrawn until it holds (integer-exact check: 100·a >= b).
DIV_MIN_RESULT = 0.01


def make_task_generator() -> TaskGenerator:
    args = SimpleNamespace(
        MaskToken_id=T.MASK,
        SOSToken_id=T.SOS,
        PadToken_id=T.PAD,
        EOSToken_id=T.EOS,
        DoneToken_id=T.DONE,
    )
    return TaskGenerator(args)


def _collect_steps(
    task_str: str,
    max_steps: int,
) -> Optional[List[Tuple[list, list, list, int]]]:
    """Run the teacher dispatcher to completion, returning per-step tuples.

    Returns None if the trajectory does not terminate cleanly (rule error or
    step cap hit) — caller should drop that task.
    """
    state = TeacherState.from_text(task_str, ctl_size=CTL_SIZE)
    state = _auto_append_eq(state)
    out: List[Tuple[list, list, list, int]] = []
    for _ in range(max_steps):
        state = _auto_append_eq(state)
        try:
            builder = dispatch_rule(state)
            action_nested = builder.to_nested_list()
        except (NotImplementedError, RuntimeError):
            return None
        done_value = int(action_nested[-1][0][0])

        tokens_snap = [list(row) for row in state.tokens]
        depth_snap = list(state.depth)
        out.append((tokens_snap, depth_snap, action_nested, done_value))

        try:
            state, final_done = apply_action(state, action_nested)
        except TeacherError:
            return None
        if final_done == 1:
            return out
        if final_done == 2:
            return None
    return None


def gen_copy_task(tg: TaskGenerator, length: int, max_steps: int = 4000):
    task_str, target_str = tg.gen_task_1_copy(length)
    return task_str, target_str, _collect_steps(task_str, max_steps)


def _binary_op_strs(
    tg: TaskGenerator, l1: int, l2: int, op: str,
    min_div_result: float | None = None,
) -> Tuple[str, str]:
    """Draw the task / target strings for ``a op b =`` without running the
    teacher dispatcher. Used by eval-side bucket builders that only need
    the strings (the trajectory is produced by the model's greedy loop).

    ``min_div_result`` (division only): redraw the operands until the
    truncated 3-decimal result is at least this value. The check is exact in
    integer arithmetic — result >= 0.01 ⟺ 100·a >= b — so the evaluation
    set is precisely the subset of cases satisfying the condition.
    """
    n1 = tg._get_random_num(l1)
    n2 = tg._get_random_num(l2)
    if op == "/":
        while int(n2) == 0:
            n2 = tg._get_random_num(l2)
        if min_div_result is not None:
            # a/b >= r  ⟺  (1/r)·a >= b  (integer-exact for r = 0.01).
            multiplier = round(1.0 / min_div_result)
            for _ in range(100_000):
                if multiplier * int(n1) >= int(n2):
                    break
                n1 = tg._get_random_num(l1)
                n2 = tg._get_random_num(l2)
                while int(n2) == 0:
                    n2 = tg._get_random_num(l2)
            else:
                raise RuntimeError(
                    f"result >= {min_div_result} unsatisfiable for "
                    f"(l1, l2) = ({l1}, {l2})"
                )
    expr = f"{n1}{op}{n2}"
    if op == "+":
        ans = str(int(n1) + int(n2))
    elif op == "-":
        ans = str(int(n1) - int(n2))
    elif op == "*":
        ans = str(int(n1) * int(n2))
    elif op == "/":
        a, b = int(n1), int(n2)
        if a % b == 0:
            ans = str(a // b)
        else:
            truncated = (a * 1000) // b
            ans = f"{truncated // 1000}.{truncated % 1000:03d}"
    else:
        raise ValueError(f"unknown op {op!r}")
    return f"{expr}=", f"{expr}={ans}"


def gen_binary_op_task(
    tg: TaskGenerator,
    l1: int,
    l2: int,
    op: str,
    max_steps: int = 200_000,
    min_div_result: float | None = None,
):
    """`a op b =` with explicit operand lengths (no random split).

    ``min_div_result`` is forwarded to ``_binary_op_strs`` — the division
    training specs pass ``DIV_MIN_RESULT`` so the training stream contains
    only cases whose result is at least 0.01, matching the evaluation.
    """
    task_str, target_str = _binary_op_strs(tg, l1, l2, op,
                                           min_div_result=min_div_result)
    return task_str, target_str, _collect_steps(task_str, max_steps)


# ── Factored (n_numbers × max_digit_len) generators for the sequential and
# nested (parenthesized) tasks. Each cell has exactly ``n_numbers`` operands,
# at least one of them exactly ``max_digit_len`` digits long.

def gen_seq_factored_task(
    tg: TaskGenerator, n_numbers: int, max_digit_len: int,
    max_steps: int = 200_000,
):
    """Factored sequential task — input WITHOUT '=' (auto-appended)."""
    task_str, target_str = tg.gen_seq_factored(n_numbers, max_digit_len)
    return task_str, target_str, _collect_steps(task_str, max_steps)


def gen_seq_paren_factored_task(
    tg: TaskGenerator, n_numbers: int, max_digit_len: int,
    max_steps: int = 200_000,
):
    """Factored nested (parenthesized) task — input WITHOUT '='."""
    task_str, target_str = tg.gen_seq_paren_factored(n_numbers, max_digit_len)
    return task_str, target_str, _collect_steps(task_str, max_steps)


# ─── inference-side helpers ──────────────────────────────────────────────────

def make_initial_state(task_str: str) -> TeacherState:
    """Initial TeacherState for inference; auto-eq is applied per step inside
    the inference loop, not here."""
    return TeacherState.from_text(task_str, ctl_size=CTL_SIZE)


def final_string(state: TeacherState) -> str:
    """Decode all non-PAD vocabs in tape order — used for full-expression
    targets in the copy / binary-op stages."""
    ids = [row[0] for row in state.tokens if row[0] != T.PAD]
    return T.decode(ids)


def final_answer(state: TeacherState) -> str:
    """Post-`=` portion of the final tape; used for seq / nest targets where
    the runner's auto-eq cleanup drops the operands and `=`."""
    s = final_string(state)
    return s.split("=", 1)[1] if "=" in s else s
