"""Batched greedy inference.

The model and teacher engine cooperate as follows:

  1. We hold inference state in `teacher.state.TeacherState` so the
     `auto_eq_depth` tag is preserved and the engine's auto-eq cleanup
     applies during inference exactly as it did during trajectory
     generation.
  2. Each step we apply `runner._auto_append_eq` to every active state, batch
     them into model tensors, run `agent.dlm_encoder` once per active set,
     greedy-decode the action (argmax for vocab, threshold for ctrl/done),
     apply the cumulative-MASK suffix-mask, then call
     `teacher.engine.apply_action` per item to advance state.
  3. An item drops out of the active set as soon as its `final_done` is 1
     (or 2 for invalid action) or it exceeds the per-item step cap.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from teacher import tokens as T
from teacher.dispatcher import can_dispatch
from teacher.engine import TeacherError, apply_action
from teacher.runner import _auto_append_eq
from teacher.state import TeacherState


def _state_to_tensors(states: Sequence[TeacherState], pad_id: int):
    """Pack a list of TeacherStates into batched (x, depth) tensors.

    The teacher rules and the model share ctl_size=3, so no narrowing is
    needed at the boundary.

    x: (B, S_total, 1+ctl_size). Positions beyond each item's tape are PAD.
    depth: (B, S_total). Padded positions are depth 0.
    """
    B = len(states)
    ctl = states[0].ctl_size
    C = 1 + ctl
    S_total = max((len(s.tokens) for s in states), default=1)
    S_total = max(S_total, 1)

    x = torch.zeros((B, S_total, C), dtype=torch.long)
    x[..., 0] = pad_id
    depth = torch.zeros((B, S_total), dtype=torch.long)

    for b, s in enumerate(states):
        n = len(s.tokens)
        if n == 0:
            continue
        x[b, :n] = torch.tensor(s.tokens, dtype=torch.long)
        depth[b, :n] = torch.tensor(s.depth, dtype=torch.long)
    return x, depth


@torch.no_grad()
def greedy_actions(
    agent,
    states: Sequence[TeacherState],
    device: torch.device,
    pad_id: int,
    mask_id: int = T.MASK,
) -> List[list]:
    """Run one greedy step for a batch of states.

    Returns a list (length B) of per-item nested action lists, each of shape
    `(l_b + 1, M, 1+ctl)` ready for `teacher.engine.apply_action`.
    """
    B = len(states)
    if B == 0:
        return []

    x, depth = _state_to_tensors(states, pad_id=pad_id)
    x = x.to(device)
    depth = depth.to(device)

    # 1. Extract max-depth visible tokens per batch (mirror DLMAgent.get_action_and_value).
    max_depth = depth.max(dim=1, keepdim=True).values
    mask = (depth == max_depth) & (x[..., 0] != pad_id)
    lengths = mask.sum(dim=1)
    S_max = int(lengths.max().item())
    if S_max == 0:
        return [[] for _ in range(B)]

    C = x.shape[2]
    x_effective = torch.zeros((B, S_max + 1, C), dtype=x.dtype, device=device)
    x_effective[..., 0] = pad_id
    b_idx = torch.where(mask)[0]
    rel_idx = mask.long().cumsum(dim=1)[mask] - 1
    x_effective[b_idx, rel_idx] = x[mask]

    vocab_ids = x_effective[..., 0]
    control_ids = x_effective[..., 1:].float()

    # 2. Forward.
    vocab_logits, control_logit, done_binary_logit = agent.dlm_encoder(
        vocab_ids, control_ids,
    )
    # vocab_logits: (B, S_max+1, M, V); control_logit: (B, S_max+1, M, ctl);
    # done_binary_logit: (B,)

    # 3. Greedy decode. Suppress special tokens that should never be produced
    #    by an action (SOS / PAD / EOS / DONE) — the agent's regularizer
    #    discourages them, but argmax can still pick one if the model has not
    #    converged. We mask their logits to -inf BEFORE argmax.
    vocab_logits = vocab_logits.clone()
    for sp in (agent.SOSToken_id, agent.PadToken_id,
               agent.EOSToken_id, agent.DoneToken_id):
        vocab_logits[..., sp] = float("-inf")
    vocab_action = vocab_logits.argmax(dim=-1)            # (B, S_max+1, M)
    control_action = (control_logit > 0).long()           # (B, S_max+1, M, ctl)
    # The (b) = CTRL_RESULT bit is engine-set on self-call return; the model
    # is not allowed to emit it. Force it to 0 in every cell so the model's
    # prediction for (b) is never observed.
    control_action[..., T.CTRL_RESULT] = 0
    done_action = (done_binary_logit > 0).long()          # (B,)

    # 4. Cumulative MASK suffix mask (slot 0 exempt). Once a MASK appears in
    #    an insertion slot, all later slots become MASK — matches the policy
    #    that DLMAgent.get_action_and_value imposes during sampling.
    is_mask = (vocab_action == mask_id)
    is_mask_mod = is_mask.clone().int()
    is_mask_mod[:, :, 0] = 0
    cum = is_mask_mod.cumsum(dim=2)
    excl = cum - is_mask_mod
    valid_slot = (excl == 0)
    vocab_action = torch.where(
        valid_slot, vocab_action, torch.full_like(vocab_action, mask_id),
    )
    # Force ctrl bits at invalid slots to 0 (safe: engine drops MASK cells).
    control_action = control_action * valid_slot.unsqueeze(-1)

    # 5. Combine and emit per-item teacher action lists.
    action = torch.cat(
        [vocab_action.unsqueeze(-1), control_action], dim=-1,
    )                                                       # (B, S_max+1, M, C)
    M = action.shape[2]

    out: List[list] = []
    action_cpu = action.cpu().tolist()
    done_cpu = done_action.cpu().tolist()
    lengths_cpu = lengths.cpu().tolist()
    for b in range(B):
        l_b = int(lengths_cpu[b])
        if l_b == 0:
            out.append([])
            continue
        per_pos = action_cpu[b][:l_b]  # length l_b
        done_slot = [[0] * C for _ in range(M)]
        done_slot[0][0] = int(done_cpu[b])
        per_pos.append(done_slot)
        out.append(per_pos)
    return out


def step_cap_for(visible_len: int, training_cap: int) -> int:
    """Per-trajectory step cap. Inside the training range respect the
    level-based ``training_cap``; for extrapolation use a generous quadratic
    bound so multiplication / sequential paths can finish."""
    fallback = max(20 * visible_len * visible_len, 5000)
    return max(training_cap, fallback)


def _state_fingerprint(s: TeacherState):
    """Hashable snapshot of a TeacherState for cycle detection.

    Two states with identical tokens, depth, and auto-eq tags are
    indistinguishable from the model's point of view, so any future step
    starting from a state we've already seen would re-trace the same
    deterministic-greedy actions forever — a true loop.

    Returns the 64-bit hash of the structural tuple rather than the tuple
    itself. At long OOD lengths the tape can grow to ~10⁴ tokens and
    trajectories can run hundreds of thousands of steps before terminating,
    so storing one tuple per step per active trajectory blows up RSS into
    the hundreds of GBs and gets the worker OOM-killed. The 8-byte hash
    is ~10 000× smaller; the 64-bit collision risk is negligible (worst
    case: one trajectory misclassified as cyclic).
    """
    return hash((
        tuple(tuple(row) for row in s.tokens),
        tuple(s.depth),
        tuple(s.auto_eq_depth),
    ))


@torch.no_grad()
def batched_run(
    agent,
    initial_states: List[TeacherState],
    device: torch.device,
    pad_id: int,
    max_steps: int = 10_000,
    over_length_factor: float = 3.0,
    step_cap_factor: float | None = None,
    per_item_max_steps: Sequence[int] | None = None,
) -> Tuple[List[TeacherState], List[bool]]:
    """Run greedy inference until every state terminates or hits the cap.

    Multiple inference "settings" can share one merged batch: each item carries
    its own step budget so a batch may span buckets/cells with different caps,
    run once, and split back out. The per-trajectory step budget is resolved
    with this priority:

      1. ``per_item_max_steps`` set → used directly as the per-item caps.
         ``max_steps`` is ignored.
      2. ``step_cap_factor`` set → cap = ``min(max_steps, factor × L_init)``.
         Bounds throughput when OOD stragglers would otherwise hold the whole
         batch alive for ``max_steps``.
      3. Neither set → every item shares the global ``max_steps``.

    Termination rules per item:
      * ``final_done == 1`` → success.
      * ``final_done == 2`` (invalid action) → failure.
      * empty / engine error → failure.
      * the per-item step cap is hit → failure.
      * **Cycle detection**: the new state's fingerprint matches any
        state seen earlier in this trajectory → failure (greedy decode
        is deterministic, so repeating a state guarantees an infinite
        loop).
      * **Over-length guard**: the **total** tape length OR the count
        of max-depth visible tokens grows past ``over_length_factor *``
        the initial input length → failure. The total-tape check is
        what catches "model keeps recursing forever" — visible
        max-depth alone misses that pattern because deeper sub-calls
        shrink the visible set.
      * **Dispatch validity** (per-step, pre-forward): after the
        per-step ``_auto_append_eq`` the visible tape must match one
        of the teacher dispatcher's seven rule patterns (single-number,
        division, subtract, add, multiply, sequential, sequential-paren).
        If ``can_dispatch`` returns False the trajectory has diverged
        into a state the teacher never produces → failure. Cheaper than
        calling ``dispatch_rule`` because the chosen rule is not
        executed; the seven ``_is_*`` pattern tests are pure functions
        of the LHS vocab list.
      * **Max-depth bound**: ``new_state.max_depth() > initial_len[i]``
        → failure. Teacher trajectories never exceed this bound (the
        worst case is division, where depth equals input_length − 1
        exactly), so the strict ``>`` cap only catches model divergence.

    Returns (final_states, success_flags).
    """
    states = list(initial_states)
    n = len(states)
    terminated = [False] * n
    success = [False] * n  # True if final_done == 1; False on cap / invalid

    # Per-item cycle-detection sets and over-length budgets, seeded from
    # the input state.
    #   md_cap   = over_length_factor × initial input length, applied to
    #              the visible max-depth token count.
    #   tape_cap = absolute soft ceiling on TOTAL tape length, applied
    #              to bound per-step CPU cost (state-tensor build /
    #              apply_action / fingerprint hashing are all O(N) in
    #              total tape, so a quadratically-growing recursion can
    #              silently burn CPU without ever growing the visible
    #              max-depth set). Default is 4 × model_max_seq_len,
    #              which sits comfortably above the largest legitimate
    #              mid-trajectory tape (model_max_seq_len is the
    #              ENCODER input limit, but total tape can include
    #              frozen lower-depth tokens too).
    seen = [{_state_fingerprint(s)} for s in states]
    initial_len = [max(1, len(s.tokens)) for s in states]
    md_cap = [int(over_length_factor * L) for L in initial_len]
    abs_tape_cap = 4 * getattr(agent, "max_seq_len", 2500)
    tape_cap = [max(int(over_length_factor * L), abs_tape_cap)
                for L in initial_len]

    # Per-trajectory step budget (priority: per_item → step_cap_factor → shared).
    if per_item_max_steps is not None:
        assert len(per_item_max_steps) == n, (
            f"per_item_max_steps length {len(per_item_max_steps)} != n {n}"
        )
        per_item_step_cap = [int(m) for m in per_item_max_steps]
    elif step_cap_factor is None:
        per_item_step_cap = [max_steps] * n
    else:
        per_item_step_cap = [
            min(max_steps, int(step_cap_factor * L)) for L in initial_len
        ]
    global_cap = max(per_item_step_cap, default=max_steps)

    for step in range(global_cap):
        # Items that just reached their own step cap fail here, so a merged
        # batch of mixed-budget settings terminates each item at its own cap.
        for i in range(n):
            if not terminated[i] and step >= per_item_step_cap[i]:
                terminated[i] = True
                success[i] = False
        active_idx = [i for i, t in enumerate(terminated) if not t]
        if not active_idx:
            break
        # auto-eq each active state at THIS step, then drop any item
        # whose visible tape no longer matches one of the teacher's
        # seven rule patterns. A state that fails can_dispatch is one
        # the teacher would never produce, so the model has diverged
        # and continuing only burns steps.
        valid_idx: List[int] = []
        for i in active_idx:
            states[i] = _auto_append_eq(states[i])
            if not can_dispatch(states[i]):
                terminated[i] = True
                success[i] = False
                continue
            valid_idx.append(i)
        if not valid_idx:
            continue

        active_states = [states[i] for i in valid_idx]
        actions = greedy_actions(agent, active_states, device, pad_id)

        for j, i in enumerate(valid_idx):
            action_nested = actions[j]
            if not action_nested:
                terminated[i] = True
                success[i] = False
                continue
            try:
                new_state, fdone = apply_action(states[i], action_nested)
            except TeacherError:
                terminated[i] = True
                success[i] = False
                continue
            states[i] = new_state
            if fdone == 1:
                terminated[i] = True
                success[i] = True
                continue
            if fdone == 2:
                terminated[i] = True
                success[i] = False
                continue
            # Max-depth bound: teacher trajectories never exceed
            # `max_depth == initial_len`. Strict `>` so the tight cases
            # (division reaches depth = input_length − 1) still complete;
            # only model divergence fires the check.
            if new_state.max_depth() > initial_len[i]:
                terminated[i] = True
                success[i] = False
                continue
            # Over-length guard: cap BOTH total tape length AND the
            # visible max-depth token count at over_length_factor ×
            # initial input length. The total-tape cap is what catches
            # "model recurses forever, deeper sub-calls keep adding
            # tokens", since the visible-max-depth count stays small
            # while buried tokens accumulate and silently blow up the
            # per-step CPU cost (state-tensor build, apply_action,
            # fingerprint hashing — all O(total tape)).
            if (len(new_state.tokens) > tape_cap[i]
                    or len(new_state.max_depth_indices()) > md_cap[i]):
                terminated[i] = True
                success[i] = False
                continue
            # Cycle detection: bail as soon as a state repeats — greedy
            # decode is deterministic so any repeat implies an infinite
            # loop. The check happens after engine.apply_action so the
            # first-encountered state is the one we record as "seen".
            fp = _state_fingerprint(new_state)
            if fp in seen[i]:
                terminated[i] = True
                success[i] = False
                continue
            seen[i].add(fp)

    return states, success
