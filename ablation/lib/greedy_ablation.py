"""Ablation-aware batched greedy inference.

Copy of ``supervised.lib.greedy_inference`` with variant hooks:

  * ``state_ctl_zero``  — ctrl-bit indices zeroed in the model's INPUT state
                          tensor (exp 2: the dropped element is invisible).
  * ``action_ctl_zero`` — ctrl-bit indices forced to 0 in the decoded action
                          (CTRL_RESULT is always forced, as in the original;
                          exp 2 no-(c) adds CTRL_TAG_C).
  * ``engine_fn``       — ``teacher.engine.apply_action`` or exp 3's
                          ``apply_action_nocleanup``.
  * ``use_can_dispatch``— per-step teacher-pattern validity guard; must be
                          disabled for exp 3 (ghost tokens make visible tapes
                          non-dispatchable by design).
  * per-item ``tape_caps`` / ``md_caps`` overrides — exp 3 tapes legitimately
    grow far beyond the original guard defaults.
  * items whose visible slice exceeds the encoder's ``max_seq_len`` fail
    individually instead of crashing the batch.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch

from teacher import tokens as T
from teacher.dispatcher import can_dispatch
from teacher.engine import TeacherError, apply_action
from teacher.runner import _auto_append_eq
from teacher.state import TeacherState

from supervised.lib.greedy_inference import _state_fingerprint, _state_to_tensors


@torch.no_grad()
def greedy_actions_ablation(
    agent,
    states: Sequence[TeacherState],
    device: torch.device,
    pad_id: int,
    mask_id: int = T.MASK,
    state_ctl_zero: Sequence[int] = (),
    action_ctl_zero: Sequence[int] = (),
) -> List[list]:
    B = len(states)
    if B == 0:
        return []

    x, depth = _state_to_tensors(states, pad_id=pad_id)
    for bit in state_ctl_zero:
        x[..., 1 + bit] = 0
    x = x.to(device)
    depth = depth.to(device)

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

    # Memory-budgeted chunking: the fp32 insert-attention tensor scales with
    # B·T², and diverging OOD items can push T toward the encoder limit
    # (~2500). Keep B_chunk·T² under a budget scaled to the device's total
    # memory (1e8 was calibrated on a 97 GB GPU; a ~46 GB card OOMs at that
    # setting once T reaches ~450 at full batch).
    if device.type == "cuda":
        total_gb = torch.cuda.get_device_properties(device).total_memory / 2**30
        FWD_BUDGET = 1.0e8 * min(1.0, total_gb / 95.0)
    else:
        FWD_BUDGET = 1.0e8
    T_cur = S_max + 1
    chunk = max(1, int(FWD_BUDGET // (T_cur * T_cur)))
    if chunk >= B:
        vocab_logits, control_logit, done_binary_logit = agent.dlm_encoder(
            vocab_ids, control_ids,
        )
    else:
        v_parts, c_parts, d_parts = [], [], []
        for k in range(0, B, chunk):
            v, c, d = agent.dlm_encoder(
                vocab_ids[k:k + chunk], control_ids[k:k + chunk],
            )
            v_parts.append(v)
            c_parts.append(c)
            d_parts.append(d)
        vocab_logits = torch.cat(v_parts, dim=0)
        control_logit = torch.cat(c_parts, dim=0)
        done_binary_logit = torch.cat(d_parts, dim=0)

    vocab_logits = vocab_logits.clone()
    for sp in (agent.SOSToken_id, agent.PadToken_id,
               agent.EOSToken_id, agent.DoneToken_id):
        vocab_logits[..., sp] = float("-inf")
    vocab_action = vocab_logits.argmax(dim=-1)
    control_action = (control_logit > 0).long()
    control_action[..., T.CTRL_RESULT] = 0
    for bit in action_ctl_zero:
        control_action[..., bit] = 0
    done_action = (done_binary_logit > 0).long()

    is_mask = (vocab_action == mask_id)
    is_mask_mod = is_mask.clone().int()
    is_mask_mod[:, :, 0] = 0
    cum = is_mask_mod.cumsum(dim=2)
    excl = cum - is_mask_mod
    valid_slot = (excl == 0)
    vocab_action = torch.where(
        valid_slot, vocab_action, torch.full_like(vocab_action, mask_id),
    )
    control_action = control_action * valid_slot.unsqueeze(-1)

    action = torch.cat([vocab_action.unsqueeze(-1), control_action], dim=-1)
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
        per_pos = action_cpu[b][:l_b]
        done_slot = [[0] * C for _ in range(M)]
        done_slot[0][0] = int(done_cpu[b])
        per_pos.append(done_slot)
        out.append(per_pos)
    return out


@torch.no_grad()
def batched_run_ablation(
    agent,
    initial_states: List[TeacherState],
    device: torch.device,
    pad_id: int,
    per_item_max_steps: Sequence[int],
    *,
    engine_fn: Callable = apply_action,
    use_can_dispatch: bool = True,
    state_ctl_zero: Sequence[int] = (),
    action_ctl_zero: Sequence[int] = (),
    tape_caps: Optional[Sequence[int]] = None,
    md_caps: Optional[Sequence[int]] = None,
) -> Tuple[List[TeacherState], List[bool]]:
    """Greedy-run each item to termination / its own step cap.

    Same termination semantics as the original ``batched_run`` (success only
    on ``final_done == 1``; cycle detection; max-depth bound; over-length
    guards), with variant hooks and per-item tape/visible caps.
    """
    states = list(initial_states)
    n = len(states)
    terminated = [False] * n
    success = [False] * n

    seen = [{_state_fingerprint(s)} for s in states]
    initial_len = [max(1, len(s.tokens)) for s in states]
    max_seq = getattr(agent, "max_seq_len", 2500)
    if tape_caps is None:
        tape_caps = [max(3 * L, 4 * max_seq) for L in initial_len]
    if md_caps is None:
        md_caps = list(tape_caps)
    per_item_step_cap = [int(m) for m in per_item_max_steps]
    assert len(per_item_step_cap) == n
    global_cap = max(per_item_step_cap, default=0)

    for step in range(global_cap):
        for i in range(n):
            if not terminated[i] and step >= per_item_step_cap[i]:
                terminated[i] = True
        active_idx = [i for i, t in enumerate(terminated) if not t]
        if not active_idx:
            break
        valid_idx: List[int] = []
        for i in active_idx:
            states[i] = _auto_append_eq(states[i])
            if use_can_dispatch and not can_dispatch(states[i]):
                terminated[i] = True
                continue
            # Encoder length guard: fail the item, not the batch.
            if len(states[i].max_depth_indices()) + 1 > max_seq:
                terminated[i] = True
                continue
            valid_idx.append(i)
        if not valid_idx:
            continue

        active_states = [states[i] for i in valid_idx]
        actions = greedy_actions_ablation(
            agent, active_states, device, pad_id,
            state_ctl_zero=state_ctl_zero, action_ctl_zero=action_ctl_zero,
        )

        for j, i in enumerate(valid_idx):
            action_nested = actions[j]
            if not action_nested:
                terminated[i] = True
                continue
            try:
                new_state, fdone = engine_fn(states[i], action_nested)
            except TeacherError:
                terminated[i] = True
                continue
            states[i] = new_state
            if fdone == 1:
                terminated[i] = True
                success[i] = True
                continue
            if fdone == 2:
                terminated[i] = True
                continue
            if new_state.max_depth() > initial_len[i]:
                terminated[i] = True
                continue
            if (len(new_state.tokens) > tape_caps[i]
                    or len(new_state.max_depth_indices()) > md_caps[i]):
                terminated[i] = True
                continue
            fp = _state_fingerprint(new_state)
            if fp in seen[i]:
                terminated[i] = True
                continue
            seen[i].add(fp)

    return states, success
