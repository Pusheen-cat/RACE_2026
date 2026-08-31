"""Ablation trainer — parameterized copy of
``supervised.lib.train_loop.train_one_epoch``.

Differences from the original:

  * ``agent_factory(args, device)`` — builds the (ablation) agent, so exp-1
    PE variants use ``DLMAgentAblation``.
  * ``ctl_drop`` — extra ctrl-bit indices (within the ctrl portion) whose
    log-prob element is zeroed in the loss, on top of the always-dropped
    (b) = CTRL_RESULT (exp 2: pass ``(T.CTRL_TAG_C,)`` for the no-(c) arm;
    the no-(b) arm needs no extra drop since (b) is already excluded).
  * ``bucket_batching`` — length-bucketed dynamic batching for exp 3, where
    ~2% of steps have visible length up to ~1200 tokens: per-bucket batch
    size is ``clamp(mem_K / T_bucket², 8, batch_size)`` so short steps train
    at full batch while rare long steps get small batches instead of OOM or
    100× padding waste. ``mem_K`` was calibrated from the measured baseline
    ceiling (mult: B=3072 at S_eff=118 within 90 GB ⇒ K≈4.3e7; default
    3.0e7 for safety — scale it down proportionally on smaller GPUs).

Everything else (AMP, cosine LR, SIGTERM-safe saving, logging) mirrors the
original.
"""

from __future__ import annotations

import os
import signal
import time
from contextlib import nullcontext as _nullcontext
from typing import Callable, Iterator, List, Optional, Sequence

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

from teacher import tokens as T

from supervised.lib.dataset import (
    TaskSpec, TeacherStreamDataset, collate_steps, collate_factory, total_tasks,
)
from supervised.lib.sup_args import SupArgs
from supervised.lib.train_loop import (
    _AMP_DTYPES, _cosine_lr, load_state_dict_safe, save_checkpoint,
)


# ---------------------------------------------------------------------------
# Length-bucketed batching (exp 3)
# ---------------------------------------------------------------------------
_BUCKET_EDGES = (64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096)


def _bucket_of(s_eff: int) -> int:
    for k, edge in enumerate(_BUCKET_EDGES):
        if s_eff <= edge:
            return k
    return len(_BUCKET_EDGES) - 1


def _step_s_eff(step) -> int:
    _tokens, depth, _action, _done = step
    if not depth:
        return 1
    md = max(depth)
    return sum(1 for d in depth if d == md)


class BucketBatchingDataset(IterableDataset):
    """Wraps a step-level IterableDataset; yields COLLATED batches grouped by
    visible length. DataLoader must use ``batch_size=None``."""

    def __init__(self, inner: TeacherStreamDataset, pad_id: int,
                 batch_cap: int, mem_K: float, batch_floor: int = 8):
        super().__init__()
        self.inner = inner
        self.pad_id = pad_id
        self.batch_cap = batch_cap
        self.mem_K = mem_K
        self.batch_floor = batch_floor

    def _bucket_batch_size(self, k: int) -> int:
        edge = _BUCKET_EDGES[k]
        b = int(self.mem_K / (edge * edge))
        return max(self.batch_floor, min(self.batch_cap, b))

    def __iter__(self) -> Iterator:
        buckets: List[list] = [[] for _ in _BUCKET_EDGES]
        for step in self.inner:
            k = _bucket_of(_step_s_eff(step))
            buckets[k].append(step)
            if len(buckets[k]) >= self._bucket_batch_size(k):
                yield collate_steps(buckets[k], pad_id=self.pad_id)
                buckets[k] = []
        for k, bucket in enumerate(buckets):
            if bucket:
                yield collate_steps(bucket, pad_id=self.pad_id)


def _identity_collate(x):
    return x


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def train_one_epoch_ablation(
    args: SupArgs,
    specs: Sequence[TaskSpec],
    device: torch.device,
    *,
    agent_factory: Callable[[SupArgs, torch.device], torch.nn.Module],
    out_name: str,
    init_ckpt: Optional[str] = None,
    ctl_drop: Sequence[int] = (),
    bucket_batching: bool = False,
    mem_K: float = 3.0e7,
    seed: int = 0,
    total_steps_override: Optional[int] = None,
    amp: str = "bf16",
    save_every: int = 5000,
) -> str:
    if amp not in _AMP_DTYPES:
        raise ValueError(f"amp must be one of {list(_AMP_DTYPES)}; got {amp!r}")
    amp_dtype = _AMP_DTYPES[amp]
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = (
        torch.amp.GradScaler("cuda") if (use_amp and amp_dtype == torch.float16)
        else None
    )
    print(f"AMP: {amp}  (autocast={use_amp}, grad_scaler={scaler is not None})")
    agent = agent_factory(args, device)
    print(f"Agent: {type(agent).__name__}  pe_kind={getattr(agent, 'model_pe_kind', '?')}  "
          f"dim_insertion={args.model_dim_insertion}  ctl_drop={tuple(ctl_drop)}  "
          f"bucket_batching={bucket_batching}")
    if init_ckpt and os.path.isfile(init_ckpt):
        print(f"Loading init checkpoint: {init_ckpt}")
        load_state_dict_safe(agent, init_ckpt)
    elif init_ckpt:
        print(f"No checkpoint at {init_ckpt}; starting from random init.")

    optimizer = optim.AdamW(
        agent.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    n_tasks = total_tasks(specs)
    print(f"Training {out_name}: {n_tasks:,} tasks across {len(specs)} specs.")
    estimated_total_steps = max(1000, n_tasks * 8 // args.batch_size)
    if total_steps_override is not None and total_steps_override > 0:
        estimated_total_steps = total_steps_override
        print(f"  cosine LR span overridden to {estimated_total_steps} steps")

    dataset = TeacherStreamDataset(specs, shuffle_buffer=args.shuffle_buffer, seed=seed)
    if bucket_batching:
        loader = DataLoader(
            BucketBatchingDataset(dataset, pad_id=args.PadToken_id,
                                  batch_cap=args.batch_size, mem_K=mem_K),
            batch_size=None,
            num_workers=args.num_workers,
            collate_fn=_identity_collate,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_factory(pad_id=args.PadToken_id),
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

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
                # Drop (b) = CTRL_RESULT (engine-set) plus any extra ablated
                # ctrl elements.
                ctl_keep = torch.ones(
                    elem_log_prob.shape[-1], device=elem_log_prob.device,
                    dtype=elem_log_prob.dtype,
                )
                ctl_keep[1 + T.CTRL_RESULT] = 0.0
                for bit in ctl_drop:
                    ctl_keep[1 + bit] = 0.0
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
                      f"B={states.shape[0]}  S={states.shape[1]}  "
                      f"steps/sec={sps:.1f}  elapsed={elapsed:.1f}s")
                running_loss = 0.0
                running_n = 0
            if save_every > 0 and step % save_every == 0:
                last_save_path = save_checkpoint(agent, args.ckpt_dir, out_name)
                print(f"  [periodic save] step={step} → {last_save_path}")
    except KeyboardInterrupt as e:
        interrupted = True
        print(f"\n[train_one_epoch_ablation] interrupted: {e}")
    finally:
        try:
            signal.signal(signal.SIGTERM, prev_sigterm)
        except Exception:
            pass
        last_save_path = save_checkpoint(agent, args.ckpt_dir, out_name)
        kind = "interrupt-save" if interrupted else "final-save"
        print(f"[{kind}] step={step} → {last_save_path}")

    return last_save_path
