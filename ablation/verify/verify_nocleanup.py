"""Exp 3 gate — verify the no-cleanup augmented teacher data.

For >= 200 random teacher mult tasks (l1, l2 ~ U{1..10}):

  1. ``collect_steps_nocleanup`` (with per-step dual invariants enabled)
     produces a terminating augmented trajectory with the same step count as
     the clean baseline.
  2. **Inference-consistency replay**: starting from the raw task string and
     using ONLY what the model-side loop will use at inference —
     ``runner._auto_append_eq`` + ``apply_action_nocleanup`` — replaying the
     recorded augmented actions reproduces the recorded augmented state
     sequence EXACTLY (tokens + ctrl + depth) and terminates with
     ``final_done == 1`` at the last step.
  3. Correctness of the augmented tape: the NON-GHOST projection of the final
     augmented tape (cells that map back to clean cells) equals the clean
     target string `A*B=answer` — i.e. the answer is still carried, just
     interleaved with retained ghost segments. The inference-replay final
     tape equals the generator's reference tape, which is what the exp-3
     eval compares the model output against (exact match).
  4. Ghost/auto-append consistency is enforced inside the generator
     (``GhostSliceError`` → task would be dropped; we count those).
  5. Size stats: max augmented S_total / S_eff vs clean, steps, ghost cells —
     used to size the training batch and the eval tape caps.

Run:  python -m ablation.verify.verify_nocleanup [--n 200] [--seed 0]
"""

from __future__ import annotations

import argparse
import random
import sys

from teacher import tokens as T
from teacher.runner import _auto_append_eq
from teacher.state import TeacherState

from supervised.lib.data_gen import (
    _binary_op_strs, _collect_steps, make_task_generator,
)

from ablation.lib.engine_nocleanup import (
    apply_action_nocleanup, dual_run,
)


def tape_string(state: TeacherState) -> str:
    return T.decode([row[0] for row in state.tokens if row[0] != T.PAD])


def inference_replay(task_str: str, steps):
    """Replay recorded augmented actions with the inference-side machinery."""
    state = TeacherState.from_text(task_str, ctl_size=3)
    state = _auto_append_eq(state)
    fd = 0
    for k, (m_tokens, m_depth, m_action, m_done) in enumerate(steps):
        state = _auto_append_eq(state)
        if [list(r) for r in state.tokens] != m_tokens or list(state.depth) != m_depth:
            return None, f"state mismatch at step {k}"
        state, fd = apply_action_nocleanup(state, m_action)
        if k < len(steps) - 1 and fd != 0:
            return None, f"early termination at step {k} (fd={fd})"
    if fd != 1:
        return None, f"did not terminate (fd={fd})"
    return state, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--op", choices=["*", "+", "copy"], default="*")
    args = p.parse_args()

    rng = random.Random(args.seed)
    tg = make_task_generator()

    n_ok = 0
    failures = []
    stats = dict(max_S_total_abl=0, max_S_eff_abl=0, max_S_total_clean=0,
                 max_S_eff_clean=0, max_steps=0, total_ghost_cells=0)

    for trial in range(args.n):
        if args.op == "copy":
            task_str, target_str = tg.gen_task_1_copy(rng.randint(1, 40))
        else:
            l1, l2 = rng.randint(1, 10), rng.randint(1, 10)
            task_str, target_str = _binary_op_strs(tg, l1, l2, args.op)

        clean_steps = _collect_steps(task_str, 200_000)
        if clean_steps is None:
            failures.append((task_str, "clean trajectory failed"))
            continue
        steps, final_ds = dual_run(task_str, 200_000, check_invariants=True)
        if steps is None:
            failures.append((task_str, "augmented trajectory failed / invariant breach"))
            continue
        if len(steps) != len(clean_steps):
            failures.append((task_str,
                             f"step count {len(steps)} != clean {len(clean_steps)}"))
            continue

        final_state, err = inference_replay(task_str, steps)
        if err:
            failures.append((task_str, err))
            continue
        # Reference tape from the generator == replayed tape.
        if tape_string(final_state) != tape_string(final_ds.abl):
            failures.append((task_str, "replayed tape != generator reference tape"))
            continue
        # Non-ghost projection of the final augmented tape == clean target.
        proj = T.decode([final_ds.abl.tokens[i][0]
                         for i in range(len(final_ds.abl.tokens))
                         if final_ds.map_a2c[i] is not None
                         and final_ds.abl.tokens[i][0] != T.PAD])
        if proj != target_str:
            failures.append((task_str,
                             f"non-ghost projection {proj!r} != target {target_str!r}"))
            continue

        n_ok += 1
        stats["max_steps"] = max(stats["max_steps"], len(steps))
        for (tok, dep, act, dn), (ctok, cdep, cact, cdn) in zip(steps, clean_steps):
            stats["max_S_total_abl"] = max(stats["max_S_total_abl"], len(tok))
            stats["max_S_total_clean"] = max(stats["max_S_total_clean"], len(ctok))
            md = max(dep) if dep else 0
            cmd = max(cdep) if cdep else 0
            stats["max_S_eff_abl"] = max(stats["max_S_eff_abl"],
                                         sum(1 for d in dep if d == md))
            stats["max_S_eff_clean"] = max(stats["max_S_eff_clean"],
                                           sum(1 for d in cdep if d == cmd))
        stats["total_ghost_cells"] += len(steps[-1][0]) - len(clean_steps[-1][0])

        if (trial + 1) % 25 == 0:
            print(f"  {trial + 1}/{args.n} tasks checked... "
                  f"(max_S_total_abl={stats['max_S_total_abl']}, "
                  f"max_S_eff_abl={stats['max_S_eff_abl']})")

    print(f"\naugmented trajectories OK: {n_ok}/{args.n}")
    print("stats:", stats)
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for t, e in failures[:10]:
            print(f"  {t}: {e}")
        sys.exit(1)
    print("NOCLEANUP GATE PASSED")


if __name__ == "__main__":
    main()
