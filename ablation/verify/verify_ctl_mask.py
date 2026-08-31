"""Exp 2 gate — verify the control-element-masked teacher data.

For >= 200 random teacher mult tasks (l1, l2 ~ U{1..10}):

  1. Baseline: the teacher trajectory terminates and the final tape equals
     the target string.
  2. drop='c': replaying the (c)-masked actions from the (c)-masked initial
     state through the UNMODIFIED teacher engine reproduces exactly the
     (c)-masked recorded state sequence (masking commutes with the engine),
     and terminates with the correct final answer at the same step count.
  3. drop='b': replaying the (b)-masked actions is engine-neutral (the engine
     force-zeroes (b) on action-derived cells), so the true state evolution is
     unchanged and the (b)-masked snapshots track it 1:1.

Run:  python -m ablation.verify.verify_ctl_mask [--n 200] [--seed 0]
"""

from __future__ import annotations

import argparse
import random
import sys

from teacher.engine import apply_action
from teacher.runner import _auto_append_eq
from teacher.state import TeacherState

from supervised.lib.data_gen import (
    _binary_op_strs, _collect_steps, final_string, make_task_generator,
)

from ablation.lib.ctl_mask import drop_bit_index, mask_state_tokens, mask_steps


def replay_check(task_str: str, target_str: str, steps, drop: str) -> str:
    """Replay masked actions through the real engine; compare against the
    masked recorded states. Returns '' on success, else an error string."""
    bit = drop_bit_index(drop)
    masked = mask_steps(steps, drop)

    state = TeacherState.from_text(task_str, ctl_size=3)
    # Replay the MASKED actions in both cases. For (b) this is engine-neutral
    # (the engine force-zeroes (b) on action-derived cells regardless); for
    # (c) it is the genuine masked-world evolution.
    replay_actions = [m[2] for m in masked]

    state = _auto_append_eq(state)
    for k, (m_tokens, m_depth, m_action, m_done) in enumerate(masked):
        state = _auto_append_eq(state)
        # The recorded (masked) state must equal the current engine state
        # with the dropped bit zeroed.
        cur_masked = mask_state_tokens(state.tokens, bit)
        if drop == "c":
            # in the masked world the engine state IS the masked state
            engine_view = state.tokens
        else:
            engine_view = None  # compare via cur_masked only
        if m_tokens != cur_masked or m_depth != list(state.depth):
            return f"state mismatch at step {k}"
        if drop == "c" and engine_view != cur_masked:
            return f"(c) engine state not identical to masked state at step {k}"
        state, fd = apply_action(state, replay_actions[k])
        if k < len(masked) - 1 and fd != 0:
            return f"early termination at step {k} (fd={fd})"
    if fd != 1:
        return f"did not terminate (fd={fd})"
    if final_string(state) != target_str:
        return f"final {final_string(state)!r} != target {target_str!r}"
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--op", choices=["*", "+", "copy"], default="*")
    args = p.parse_args()

    rng = random.Random(args.seed)
    tg = make_task_generator()

    n_ok = {"base": 0, "b": 0, "c": 0}
    failures = []
    for trial in range(args.n):
        if args.op == "copy":
            task_str, target_str = tg.gen_task_1_copy(rng.randint(1, 40))
        else:
            l1, l2 = rng.randint(1, 10), rng.randint(1, 10)
            task_str, target_str = _binary_op_strs(tg, l1, l2, args.op)
        steps = _collect_steps(task_str, 200_000)
        if steps is None:
            failures.append((task_str, "base", "trajectory failed"))
            continue
        n_ok["base"] += 1
        for drop in ("b", "c"):
            try:
                err = replay_check(task_str, target_str, steps, drop)
            except AssertionError as e:
                err = f"assertion: {e}"
            if err:
                failures.append((task_str, drop, err))
            else:
                n_ok[drop] += 1
        if (trial + 1) % 50 == 0:
            print(f"  {trial + 1}/{args.n} tasks checked...")

    print(f"\nbase trajectories OK: {n_ok['base']}/{args.n}")
    print(f"drop=b replay OK:     {n_ok['b']}/{args.n}")
    print(f"drop=c replay OK:     {n_ok['c']}/{args.n}")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for t, d, e in failures[:10]:
            print(f"  [{d}] {t}: {e}")
        sys.exit(1)
    print("CTL-MASK GATE PASSED")


if __name__ == "__main__":
    main()
