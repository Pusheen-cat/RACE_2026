"""Shared trainer: 1 epoch of teacher-action imitation, save state dict.

We reuse `DLMAgent.get_action_and_value(x, depth, action, done)` to compute
the per-step log-probability of the teacher action under the model. The
masked sum it returns IS the negative supervised NLL we want to minimise:

    loss = -masked_log_prob.mean()  # vocab CE + ctrl Bernoulli + done

A small regularisation term (anchor + special-token suppression) is
optionally added; both coefficients default to 0 in `SupArgs`.
"""

from __future__ import annotations

import math
import os
import signal
import time
from contextlib import nullcontext as _nullcontext
from typing import Optional, Sequence

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.dlm_agent import DLMAgent
from teacher import tokens as T

from supervised.lib.dataset import (
    TaskSpec, TeacherStreamDataset, collate_factory, total_tasks,
)
from supervised.lib.sup_args import SupArgs


def build_agent(args: SupArgs, device: torch.device) -> DLMAgent:
    agent = DLMAgent(args).to(device)
    return agent


def load_state_dict_safe(agent: DLMAgent, path: str) -> None:
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = agent.load_state_dict(sd, strict=False)
    if missing:
        print(f"  init_ckpt load: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  init_ckpt load: {len(unexpected)} unexpected keys "
              f"(e.g. {unexpected[:3]})")


def save_checkpoint(agent: DLMAgent, ckpt_dir: str, name: str) -> str:
    path = os.path.join(ckpt_dir, name)
    # `name` may be a nested path; make sure the full parent directory
    # exists, not just ckpt_dir.
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(agent.state_dict(), path)
    return path


def _cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup: Optional[int] = None,
    floor_frac: float = 0.01,
    warmup_frac: float = 0.01,
):
    """Linear-warmup → cosine to ``floor_frac * base_lr`` → hold at floor.

    ``warmup`` defaults to ``max(1, ceil(warmup_frac * total_steps))`` (i.e.
    1% of total steps). Cosine bottoms out at ``floor_frac * base_lr``
    (0.01·lr by default) and stays pinned there for any step past
    ``total_steps`` — so an under-estimated horizon keeps lr at the floor
    instead of returning to base.
    """
    if warmup is None:
        warmup = max(1, math.ceil(warmup_frac * max(1, total_steps)))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    floor = floor_frac * base_lr
    denom = max(1, total_steps - warmup)
    progress = min(max((step - warmup) / denom, 0.0), 1.0)
    return floor + (base_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


_AMP_DTYPES = {
    "off":  None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def train_one_epoch(
    args: SupArgs,
    specs: Sequence[TaskSpec],
    device: torch.device,
    *,
    init_ckpt: Optional[str] = None,
    out_name: str,
    seed: int = 0,
    curriculum: bool = False,
    total_steps_override: Optional[int] = None,
    amp: str = "bf16",
    save_every: int = 5000,
) -> str:
    if amp not in _AMP_DTYPES:
        raise ValueError(f"amp must be one of {list(_AMP_DTYPES)}; got {amp!r}")
    amp_dtype = _AMP_DTYPES[amp]
    use_amp = amp_dtype is not None and device.type == "cuda"
    # fp16 needs gradient scaling because of its narrow dynamic range; bf16
    # has the same exponent range as fp32 so no scaler is required.
    scaler = (
        torch.amp.GradScaler("cuda") if (use_amp and amp_dtype == torch.float16)
        else None
    )
    print(f"AMP: {amp}  (autocast={use_amp}, grad_scaler={scaler is not None})")
    agent = build_agent(args, device)
    if init_ckpt and os.path.isfile(init_ckpt):
        print(f"Loading init checkpoint: {init_ckpt}")
        load_state_dict_safe(agent, init_ckpt)
    else:
        if init_ckpt:
            print(f"No checkpoint at {init_ckpt}; starting from random init.")

    optimizer = optim.AdamW(
        agent.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    n_tasks = total_tasks(specs)
    print(f"Training {out_name}: {n_tasks:,} tasks across {len(specs)} specs.")
    # Heuristic fallback: ~8 steps per task across the mixed task families.
    # The step count only drives cosine-LR scheduling; an underestimate is
    # fine, the scheduler just clamps the tail at the LR floor.
    estimated_total_steps = max(1000, n_tasks * 8 // args.batch_size)
    if total_steps_override is not None and total_steps_override > 0:
        estimated_total_steps = total_steps_override
        print(f"  cosine LR span overridden to {estimated_total_steps} steps")

    dataset = TeacherStreamDataset(
        specs, shuffle_buffer=args.shuffle_buffer, seed=seed,
        curriculum=curriculum,
    )
    if curriculum:
        print(f"  curriculum mode: ON (specs in given order, no shuffle buffer)")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_factory(pad_id=args.PadToken_id),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Install a SIGTERM handler that raises KeyboardInterrupt, so the
    # try/finally below also fires on `kill <pid>` (not just Ctrl-C). The
    # final save then preserves the latest in-memory weights regardless of
    # how the run terminates.
    def _term_handler(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt(f"signal {signum}")

    prev_sigterm = signal.signal(signal.SIGTERM, _term_handler)

    agent.train()
    step = 0
    t0 = time.time()
    running_loss = 0.0
    running_n = 0
    last_save_path: Optional[str] = None
    interrupted = False
    try:
        for batch in loader:
            states, depths, actions, dones = batch
            states = states.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            dones = dones.to(device, non_blocking=True)

            lr = _cosine_lr(step, estimated_total_steps, args.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp else _nullcontext()
            )
            with autocast_ctx:
                _, _, reg_loss, elem_log_prob, done_log_prob, final_mask = (
                    agent.get_action_and_value(states, depths, actions, dones)
                )
                # elem_log_prob: (B, S, M, 1 + ctl); done_log_prob: (B,);
                # final_mask: (B, S, M).
                #
                # The agent's final_mask covers slot 0 and every insertion
                # slot up to AND INCLUDING the first MASK; later slots are
                # tape-equivalent to the first MASK and need no supervision.
                # The (b) = CTRL_RESULT bit is engine-set on self-call return,
                # never model-emitted — zero it out of the loss.
                ctl_keep = torch.ones(
                    elem_log_prob.shape[-1], device=elem_log_prob.device,
                    dtype=elem_log_prob.dtype,
                )
                ctl_keep[1 + T.CTRL_RESULT] = 0.0
                elem_lp_kept = elem_log_prob * ctl_keep
                mask4 = final_mask.unsqueeze(-1).to(elem_lp_kept.dtype)
                masked_log_prob = (
                    (elem_lp_kept * mask4).sum(dim=(1, 2, 3)) + done_log_prob
                )
                loss = -masked_log_prob.mean()
                if reg_loss is not None:
                    loss = loss + reg_loss.mean()

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            running_loss += float(loss.item()) * states.shape[0]
            running_n += states.shape[0]
            step += 1
            if step % args.log_every == 0:
                avg = running_loss / max(1, running_n)
                elapsed = time.time() - t0
                sps = step / max(1e-3, elapsed)
                print(f"  step {step}  loss={avg:.4f}  lr={lr:.2e}  "
                      f"steps/sec={sps:.1f}  elapsed={elapsed:.1f}s")
                running_loss = 0.0
                running_n = 0
            # Periodic checkpoint — overwrites the same on-disk path so the
            # latest weights are always recoverable, even on SIGKILL/OOM.
            if save_every > 0 and step % save_every == 0:
                last_save_path = save_checkpoint(agent, args.ckpt_dir, out_name)
                print(f"  [periodic save] step={step} → {last_save_path}")
    except KeyboardInterrupt as e:
        interrupted = True
        print(f"\n[train_one_epoch] interrupted: {e}")
    finally:
        # Restore the prior SIGTERM disposition (the parent process may rely
        # on the default) and write one last checkpoint reflecting the final
        # in-memory state, even on interrupt.
        try:
            signal.signal(signal.SIGTERM, prev_sigterm)
        except Exception:
            pass
        last_save_path = save_checkpoint(agent, args.ckpt_dir, out_name)
        kind = "interrupt-save" if interrupted else "final-save"
        print(f"[{kind}] step={step} → {last_save_path}")

    return last_save_path
