"""Train one binary-op ablation arm (addition or multiplication stage).

Both stages mirror the main-curriculum protocol: (l1,l2) ∈ {1..10}², one
epoch, lr 2e-4, bf16; addition uses 100 000 tasks/combo (10 M tasks) and
multiplication 10 000 tasks/combo (1 M tasks), exactly like the main
pipeline.

Stage order and initialization follow the main curriculum
(copy → add → mult), chained within each arm:

  * ``--op '+'`` (addition): initializes from the arm's own finished copy
    checkpoint ``ablation/checkpoints/<arm>/copy.pt``. Exception:
    ``edit_no_cleanup`` has no copy arm — on copy the no-cleanup pipeline
    is bit-identical to the clean one — so its addition stage initializes
    from the MAIN curriculum's ``checkpoints/copy.pt``, the equivalent
    checkpoint.
  * ``--op '*'`` (multiplication): initializes from the arm's own finished
    addition checkpoint ``ablation/checkpoints/<arm>/add.pt``.

Arms:
  pe_noinv         exp1 — GLPE without the back-direction sign flip
  pe_nope          exp1 — T-axis NoPE (insertion-axis RoPE kept)
  pe_rope          exp1 — plain 2D RoPE
  ctl_no_b         exp2 — control element (b) masked from model inputs
  ctl_no_c         exp2 — control element (c) masked from inputs+labels+loss
  edit_no_cleanup  exp3 — done-stage auto-eq cleanup disabled (augmented
                   tapes; length-bucketed batching)

Run:
  CUDA_VISIBLE_DEVICES=0 python -m ablation.train.train_binary_op_ablation \
      --op + --arm pe_noinv
"""

from __future__ import annotations

import argparse

import torch

from teacher import tokens as T

from supervised.lib.dataset import binary_op_specs
from supervised.lib.sup_args import SupArgs
from supervised.lib.util_steps import TASK_IDS, estimate_total_steps_grid

from ablation.lib.data_gen_ablation import (
    masked_binary_op_specs, nocleanup_binary_op_specs,
)
from ablation.lib.train_loop_ablation import train_one_epoch_ablation
from ablation.model.agent_ablation import ARM_DIM_INSERTION, DLMAgentAblation

# arm -> (pe_kind, spec_kind, ctl_drop, bucket_batching)
ARMS = {
    "pe_noinv":        ("glpe_v1_noinv", "plain",     (),               False),
    "pe_nope":         ("nope_t",        "plain",     (),               False),
    "pe_rope":         ("rope2d",        "plain",     (),               False),
    "ctl_no_b":        ("glpe_v1",       "mask_b",    (),               False),
    "ctl_no_c":        ("glpe_v1",       "mask_c",    (T.CTRL_TAG_C,),  False),
    "edit_no_cleanup": ("glpe_v1",       "nocleanup", (),               True),
}

_OP_SLUG = {"*": "mult", "+": "add"}
_OP_TASK_ID = {"*": TASK_IDS["mul"], "+": TASK_IDS["add"]}
# Main-curriculum task counts per (l1, l2) cell.
_OP_N_PER_COMBO = {"+": 100_000, "*": 10_000}

# Bucketed batching adds more (smaller) optimizer steps for the ghost-
# inflated tail. Measured on the original runs: ~1.5× the fixed-batch step
# count for mult (~98% of augmented steps sit in the full-batch buckets;
# the long tail runs at mem_K/T²); add trajectories inflate far less, so
# 1.4× is a safe horizon there.
_BUCKET_STEP_FACTOR = {"*": 1.5, "+": 1.4}


def default_init_ckpt(arm: str, op: str) -> str:
    if op == "+":
        if arm == "edit_no_cleanup":
            # No copy arm (bit-identical to the baseline on copy) — the main
            # curriculum's copy checkpoint is the equivalent init.
            return "checkpoints/copy.pt"
        return f"ablation/checkpoints/{arm}/copy.pt"
    return f"ablation/checkpoints/{arm}/add.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["*", "+"], required=True)
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--init_ckpt", default=None,
                   help="default: ablation/checkpoints/<arm>/copy.pt for "
                        "--op '+' (checkpoints/copy.pt for edit_no_cleanup), "
                        "ablation/checkpoints/<arm>/add.pt for --op '*'")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--shuffle_buffer", type=int, default=4000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--l1_min", type=int, default=1)
    p.add_argument("--l1_max", type=int, default=10)
    p.add_argument("--l2_min", type=int, default=1)
    p.add_argument("--l2_max", type=int, default=10)
    p.add_argument("--n_per_combo", type=int, default=0,
                   help="tasks per (l1, l2) cell; 0 = the main-curriculum "
                        "default (100 000 for '+', 10 000 for '*')")
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--total_steps", type=int, default=0,
                   help="cosine-LR horizon override (0 = auto)")
    p.add_argument("--mem_K", type=float, default=3.0e7,
                   help="bucketed-batching memory constant (per-bucket batch "
                        "<= mem_K/T²; calibrated for ~90 GB — scale it down "
                        "proportionally on smaller GPUs)")
    args_cli = p.parse_args()

    pe_kind, spec_kind, ctl_drop, bucketing = ARMS[args_cli.arm]
    init_ckpt = args_cli.init_ckpt or default_init_ckpt(args_cli.arm, args_cli.op)
    slug = _OP_SLUG[args_cli.op]
    n_per_combo = args_cli.n_per_combo or _OP_N_PER_COMBO[args_cli.op]

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
    l1_range = list(range(args_cli.l1_min, args_cli.l1_max + 1))
    l2_range = list(range(args_cli.l2_min, args_cli.l2_max + 1))
    if spec_kind == "plain":
        specs = binary_op_specs(args_cli.op, l1_range, l2_range, n_per_combo)
    elif spec_kind == "mask_b":
        specs = masked_binary_op_specs(args_cli.op, l1_range, l2_range,
                                       n_per_combo, drop="b")
    elif spec_kind == "mask_c":
        specs = masked_binary_op_specs(args_cli.op, l1_range, l2_range,
                                       n_per_combo, drop="c")
    elif spec_kind == "nocleanup":
        specs = nocleanup_binary_op_specs(l1_range, l2_range,
                                          n_per_combo, op=args_cli.op)
    else:
        raise ValueError(spec_kind)

    base_steps = estimate_total_steps_grid(
        task_id=_OP_TASK_ID[args_cli.op],
        l1_range=l1_range, l2_range=l2_range,
        n_per_combo=n_per_combo, batch_size=args_cli.batch_size,
    )
    if args_cli.total_steps > 0:
        total_steps = args_cli.total_steps
    elif bucketing:
        total_steps = int(base_steps * _BUCKET_STEP_FACTOR[args_cli.op])
    else:
        total_steps = base_steps
    print(f"Arm {args_cli.arm}: op={args_cli.op} pe_kind={pe_kind} "
          f"spec={spec_kind} init={init_ckpt} n_per_combo={n_per_combo:,} "
          f"cosine total_steps={total_steps:,}")

    train_one_epoch_ablation(
        args, specs, device,
        agent_factory=lambda a, d: DLMAgentAblation(a).to(d),
        out_name=f"{args_cli.arm}/{slug}.pt",
        init_ckpt=init_ckpt,
        ctl_drop=ctl_drop,
        bucket_batching=bucketing,
        mem_K=args_cli.mem_K,
        seed=args_cli.seed,
        total_steps_override=total_steps,
        amp=args_cli.amp,
        save_every=args_cli.save_every,
    )


if __name__ == "__main__":
    main()
