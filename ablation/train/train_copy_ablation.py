"""Train one copy-task ablation arm from scratch.

Mirrors the main copy protocol (lengths 1..40, 250 000 tasks/length = 10 M
tasks, one epoch, lr 2e-4, bf16); only the ablated component differs.

Arms:
  pe_noinv          GLPE without the back-direction sign flip
  pe_nope           T-axis NoPE (insertion-axis RoPE kept)
  pe_rope           plain 2D RoPE
  ctl_no_b          ctrl (b)=CTRL_RESULT zeroed in states+actions (loss
                    already excludes (b) via ctl_keep)
  ctl_no_c          ctrl (c)=CTRL_TAG_C zeroed in states+actions and
                    excluded from the loss
  edit_no_cleanup   augmented no-cleanup trajectories. NOTE: verified to be
                    bit-identical to the clean pipeline on copy (copy tasks
                    carry '=' in the task string, so auto-append/cleanup
                    never fires) — this arm is a baseline-equivalent control
                    and can be skipped (train_arm.sh skips it).

Copy tapes stay tiny (max augmented S_eff = 81 at L=40), so the standard
fixed-batch loop is used for every arm (no length bucketing).

Run:
  CUDA_VISIBLE_DEVICES=0 python -m ablation.train.train_copy_ablation \
      --arm pe_noinv
"""

from __future__ import annotations

import argparse

import torch

from teacher import tokens as T

from supervised.lib.dataset import copy_specs
from supervised.lib.sup_args import SupArgs
from supervised.lib.util_steps import TASK_IDS, estimate_total_steps

from ablation.lib.data_gen_ablation import (
    masked_copy_specs, nocleanup_copy_specs,
)
from ablation.lib.train_loop_ablation import train_one_epoch_ablation
from ablation.model.agent_ablation import ARM_DIM_INSERTION, DLMAgentAblation

# arm -> (pe_kind, spec_kind, extra ctl bits excluded from the loss)
ARMS = {
    "pe_noinv":        ("glpe_v1_noinv", "plain",     ()),
    "pe_nope":         ("nope_t",        "plain",     ()),
    "pe_rope":         ("rope2d",        "plain",     ()),
    "ctl_no_b":        ("glpe_v1",       "mask_b",    ()),
    "ctl_no_c":        ("glpe_v1",       "mask_c",    (T.CTRL_TAG_C,)),
    "edit_no_cleanup": ("glpe_v1",       "nocleanup", ()),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--shuffle_buffer", type=int, default=4000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lengths_min", type=int, default=1)
    p.add_argument("--lengths_max", type=int, default=40)
    p.add_argument("--n_per_length", type=int, default=250_000)
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--total_steps", type=int, default=0,
                   help="cosine-LR horizon override (0 = auto from teacher stats)")
    args_cli = p.parse_args()

    pe_kind, spec_kind, ctl_drop = ARMS[args_cli.arm]

    args = SupArgs(
        seed=args_cli.seed,
        batch_size=args_cli.batch_size,
        num_workers=args_cli.num_workers,
        shuffle_buffer=args_cli.shuffle_buffer,
        learning_rate=args_cli.lr,
        model_pe_kind=pe_kind,
        model_dim_insertion=ARM_DIM_INSERTION[pe_kind],
        ckpt_dir="ablation/checkpoints",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lengths = list(range(args_cli.lengths_min, args_cli.lengths_max + 1))
    if spec_kind == "plain":
        specs = copy_specs(lengths=lengths, n_tasks_per_length=args_cli.n_per_length)
    elif spec_kind == "mask_b":
        specs = masked_copy_specs(lengths, args_cli.n_per_length, drop="b")
    elif spec_kind == "mask_c":
        specs = masked_copy_specs(lengths, args_cli.n_per_length, drop="c")
    else:
        specs = nocleanup_copy_specs(lengths, args_cli.n_per_length)

    if args_cli.total_steps > 0:
        total_steps = args_cli.total_steps
    else:
        # Step counts per trajectory are unchanged by masking / no-cleanup.
        total_steps = estimate_total_steps(
            task_id=TASK_IDS["copy"], lengths=lengths,
            n_per_length=args_cli.n_per_length, batch_size=args_cli.batch_size,
        )
    print(f"Arm {args_cli.arm}: pe_kind={pe_kind} spec={spec_kind} "
          f"ctl_drop={ctl_drop} cosine total_steps={total_steps:,}")

    train_one_epoch_ablation(
        args, specs, device,
        agent_factory=lambda a, d: DLMAgentAblation(a).to(d),
        out_name=f"{args_cli.arm}/copy.pt",
        init_ckpt=None,
        ctl_drop=ctl_drop,
        seed=args_cli.seed,
        total_steps_override=total_steps,
        amp=args_cli.amp,
        save_every=args_cli.save_every,
    )


if __name__ == "__main__":
    main()
