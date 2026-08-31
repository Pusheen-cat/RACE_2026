"""Drive a teacher rule until termination.

A teacher trajectory is a sequence of `{action, done}` dicts in the schema that
`training/expert_paths.py` already consumes. We collect them by repeatedly
calling the rule, applying it to the state, and stopping when the engine
reports `final_done == 1`.

Global rule (auto-append `=`): if the visible tape (tokens at max depth) has no
`=`, the runner appends one transparently — not as a trajectory step. The
user's spec says this rule is "automatically applied" and not learned by the
model, so it shouldn't appear in the imitation data. The same wrapper is what
the live model will eventually apply (still TODO on the env/agent side).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

from teacher import tokens as T
from teacher.action import ActionBuilder
from teacher.engine import TeacherError, apply_action
from teacher.state import TeacherState


RuleFn = Callable[[TeacherState], ActionBuilder]


@dataclass
class TeacherTrajectory:
    task_tokens: List[int]
    steps: List[dict] = field(default_factory=list)  # {"action": nested, "done": 0|1}
    final_state: TeacherState = None
    terminated: bool = False
    n_steps: int = 0


def _auto_append_eq(state: TeacherState) -> TeacherState:
    """Append '=' to the visible tape if absent. Pure state mutation — does
    NOT add a trajectory step (the spec says this rule is automatic, not
    learned). The new cell is tagged via `auto_eq_depth` with the depth at
    which it becomes visible; the engine consumes that tag on the matching
    return (drops operands+'=' and keeps only the post-'=' digits)."""
    md = state.max_depth_indices()
    if not md:
        return state
    if any(state.vocab(i) == T.EQ for i in md):
        return state
    last = md[-1]
    new_row = [T.EQ] + [0] * state.ctl_size
    insertion_depth = state.max_depth()
    new_tokens = state.tokens[:last + 1] + [new_row] + state.tokens[last + 1:]
    new_depth = state.depth[:last + 1] + [insertion_depth] + state.depth[last + 1:]
    new_auto_eq = (state.auto_eq_depth[:last + 1]
                   + [insertion_depth]
                   + state.auto_eq_depth[last + 1:])
    return TeacherState(tokens=new_tokens, depth=new_depth,
                        ctl_size=state.ctl_size, auto_eq_depth=new_auto_eq)


def run_teacher(
    initial_state: TeacherState,
    rule_fn: RuleFn,
    max_steps: int = 500,
) -> TeacherTrajectory:
    state = _auto_append_eq(initial_state)
    task_tokens = [row[0] for row, d in zip(state.tokens, state.depth)
                   if d == 0 and row[0] != T.PAD]

    traj = TeacherTrajectory(task_tokens=task_tokens)

    for step in range(max_steps):
        state = _auto_append_eq(state)
        builder = rule_fn(state)
        action_nested = builder.to_nested_list()
        done_value = action_nested[-1][0][0]
        traj.steps.append({"action": action_nested, "done": int(done_value)})

        state, final_done = apply_action(state, action_nested)
        traj.n_steps += 1

        if final_done == 1:
            traj.final_state = state
            traj.terminated = True
            return traj
        if final_done == 2:
            raise TeacherError(
                f"rule produced self-call + done in the same step at step {step}"
            )

    traj.final_state = state
    traj.terminated = False
    return traj
