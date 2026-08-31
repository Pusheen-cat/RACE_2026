"""Stage 1: train on the copy task from random init.

Paper setting: lengths 1..40, 250 000 tasks per length → 10 M tasks, one
epoch, lr 2e-4. Saves `checkpoints/copy.pt`.

Run:
    CUDA_VISIBLE_DEVICES=0 python -m supervised.train.train_copy \
        --batch_size 2000 --num_workers 16
"""

from __future__ import annotations

import argparse

import torch

from supervised.lib.dataset import copy_specs
from supervised.lib.sup_args import SupArgs
from supervised.lib.train_loop import train_one_epoch
from supervised.lib.util_steps import TASK_IDS, estimate_total_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--shuffle_buffer", type=int, default=4000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lengths_min", type=int, default=1)
    p.add_argument("--lengths_max", type=int, default=40)
    p.add_argument("--n_per_length", type=int, default=250_000)
    p.add_argument("--out_name", type=str, default="copy.pt")
    p.add_argument("--curriculum", action="store_true",
                   help="ascending-length ordering, no shuffle")
    p.add_argument("--total_steps", type=int, default=0,
                   help="override the cosine-LR total-steps estimate")
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16",
                   help="mixed-precision mode for forward + loss")
    p.add_argument("--pe_kind", choices=["glpe_v1", "rope2d"], default="glpe_v1",
                   help="positional-encoding family; see model/positional_encodings.py")
    p.add_argument("--dim_insertion", type=int, default=0,
                   help="override SupArgs.model_dim_insertion (0 = keep default). "
                        "glpe_v1 requires >= 48; use 24 with rope2d.")
    p.add_argument("--save_every", type=int, default=5000,
                   help="overwrite the on-disk checkpoint every N grad steps. "
                        "0 disables periodic saves (final save still runs).")
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
    lengths = list(range(args_cli.lengths_min, args_cli.lengths_max + 1))
    specs = copy_specs(lengths=lengths, n_tasks_per_length=args_cli.n_per_length)

    if args_cli.total_steps and args_cli.total_steps > 0:
        total_steps_override = args_cli.total_steps
    else:
        total_steps_override = estimate_total_steps(
            task_id=TASK_IDS["copy"],
            lengths=lengths,
            n_per_length=args_cli.n_per_length,
            batch_size=args_cli.batch_size,
        )
        print(f"Estimated cosine total_steps={total_steps_override:,} "
              f"(task_id={TASK_IDS['copy']})")

    train_one_epoch(
        args, specs, device,
        init_ckpt=None,
        out_name=args_cli.out_name,
        seed=args_cli.seed,
        curriculum=args_cli.curriculum,
        total_steps_override=total_steps_override,
        amp=args_cli.amp,
        save_every=args_cli.save_every,
    )


if __name__ == "__main__":
    main()
