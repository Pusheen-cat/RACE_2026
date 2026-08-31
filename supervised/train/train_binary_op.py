"""Stages 2–5: train one binary op on top of the previous checkpoint.

Operand lengths l1 ∈ {1..10}, l2 ∈ {1..10} (100 combos). Paper settings:

    add  (op '+') — 100 000 tasks/combo, init from copy.pt
    mult (op '*') —  10 000 tasks/combo, init from add.pt
    sub  (op '-') —  10 000 tasks/combo, init from mult.pt
    div  (op '/') —  10 000 tasks/combo, init from sub.pt

Division only uses cases whose result is at least 0.01 (the smallest value
the 3-decimal targets can express), the same restriction the evaluation
applies: cells with l2 > l1 + 2 cannot satisfy it and are pruned from the
grid (72/100 cells remain → 720 k tasks), and within the kept cells the
operand pairs are redrawn until 100·a >= b.

One epoch, lr 2e-4. Saves `checkpoints/<out_name>`.

Run:
    CUDA_VISIBLE_DEVICES=0 python -m supervised.train.train_binary_op \
        --op + --init_ckpt checkpoints/copy.pt --out_name add.pt
"""

from __future__ import annotations

import argparse

import torch

from supervised.lib.data_gen import DIV_MIN_RESULT
from supervised.lib.dataset import binary_op_specs
from supervised.lib.sup_args import SupArgs
from supervised.lib.train_loop import train_one_epoch
from supervised.lib.util_steps import TASK_IDS, estimate_total_steps_grid


_OP_TO_TASK = {"+": TASK_IDS["add"], "-": TASK_IDS["sub"],
               "*": TASK_IDS["mul"], "/": TASK_IDS["div"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["+", "-", "*", "/"], required=True)
    p.add_argument("--init_ckpt", required=True)
    p.add_argument("--out_name", required=True,
                   help="filename under SupArgs.ckpt_dir, e.g. 'add.pt'")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--shuffle_buffer", type=int, default=4000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--l1_min", type=int, default=1)
    p.add_argument("--l1_max", type=int, default=10)
    p.add_argument("--l2_min", type=int, default=1)
    p.add_argument("--l2_max", type=int, default=10)
    p.add_argument("--n_per_combo", type=int, default=100_000)
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16",
                   help="mixed-precision mode for forward + loss")
    p.add_argument("--pe_kind", choices=["glpe_v1", "rope2d"], default="glpe_v1",
                   help="positional-encoding family; must match the init_ckpt")
    p.add_argument("--dim_insertion", type=int, default=0,
                   help="override SupArgs.model_dim_insertion (0 = keep default). "
                        "glpe_v1 requires >= 48; use 24 with rope2d.")
    p.add_argument("--save_every", type=int, default=5000,
                   help="overwrite the on-disk checkpoint every N grad steps. "
                        "0 disables periodic saves (final save still runs).")
    p.add_argument("--total_steps", type=int, default=0,
                   help="override the cosine-LR total-steps estimate. "
                        "0 = look up from teacher/stats/task_<id>_grid.txt.")
    args_cli = p.parse_args()

    extra_kwargs = {}
    if args_cli.dim_insertion > 0:
        extra_kwargs["model_dim_insertion"] = args_cli.dim_insertion
    args = SupArgs(
        seed=args_cli.seed,
        batch_size=args_cli.batch_size,
        num_workers=args_cli.num_workers,
        shuffle_buffer=args_cli.shuffle_buffer,
        learning_rate=args_cli.lr,
        model_pe_kind=args_cli.pe_kind,
        **extra_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l1_range = list(range(args_cli.l1_min, args_cli.l1_max + 1))
    l2_range = list(range(args_cli.l2_min, args_cli.l2_max + 1))
    combos = [(l1, l2) for l1 in l1_range for l2 in l2_range]
    min_div_result = None
    if args_cli.op == "/":
        # Division only uses cases with result >= DIV_MIN_RESULT: prune the
        # cells that cannot satisfy it, redraw samples inside the rest.
        n_all = len(combos)
        combos = [(l1, l2) for (l1, l2) in combos if l2 <= l1 + 2]
        min_div_result = DIV_MIN_RESULT
        print(f"div prune: keeping {len(combos)}/{n_all} cells "
              f"(l2 <= l1 + 2 for result >= {DIV_MIN_RESULT} feasibility)")
    specs = binary_op_specs(
        op=args_cli.op,
        l1_range=l1_range,
        l2_range=l2_range,
        n_tasks_per_combo=args_cli.n_per_combo,
        combos=combos,
        min_div_result=min_div_result,
    )
    if args_cli.total_steps and args_cli.total_steps > 0:
        total_steps_override = args_cli.total_steps
    else:
        total_steps_override = estimate_total_steps_grid(
            task_id=_OP_TO_TASK[args_cli.op],
            l1_range=l1_range,
            l2_range=l2_range,
            n_per_combo=args_cli.n_per_combo,
            batch_size=args_cli.batch_size,
            combos=combos,
        )
        print(f"Estimated cosine total_steps={total_steps_override:,} "
              f"(op={args_cli.op}, task_id={_OP_TO_TASK[args_cli.op]})")

    train_one_epoch(
        args, specs, device,
        init_ckpt=args_cli.init_ckpt,
        out_name=args_cli.out_name,
        seed=args_cli.seed,
        amp=args_cli.amp,
        total_steps_override=total_steps_override,
        save_every=args_cli.save_every,
    )


if __name__ == "__main__":
    main()
