"""Stages 6–7: train sequential / nested (parenthesized) expressions on top
of a previous checkpoint.

Both stages use the factored (n_numbers, max_digit_len) grid: each cell has
exactly ``n_numbers`` operands, at least one of them exactly
``max_digit_len`` digits long. Paper setting: (n_numbers, max_digit_len) ∈
{2..10} × {1..10} (90 cells), 10 000 tasks per cell → 900 k tasks, one
epoch, lr 2e-4.

    seq  (--task_kind seq)       — init from div.pt, saves seq.pt
    nest (--task_kind seq_paren) — init from seq.pt, saves nest.pt
                                   (each task wraps 1..max(1, (n_numbers−1)//2)
                                   random adjacent pairs in parentheses; at
                                   n_numbers = 2 that is one fully-
                                   parenthesized binary op)

Inputs are generated WITHOUT '=' so the runner's auto-eq branch fires; the
final tape therefore keeps only the answer (evaluation compares the
post-'=' substring).

Run:
    CUDA_VISIBLE_DEVICES=0 python -m supervised.train.train_sequential \
        --task_kind seq --init_ckpt checkpoints/div.pt --out_name seq.pt
"""

from __future__ import annotations

import argparse

import torch

from supervised.lib.dataset import seq_factored_specs, seq_paren_factored_specs
from supervised.lib.sup_args import SupArgs
from supervised.lib.train_loop import train_one_epoch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_kind", choices=["seq", "seq_paren"], required=True)
    p.add_argument("--init_ckpt", required=True)
    p.add_argument("--out_name", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--shuffle_buffer", type=int, default=4000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n_numbers_min", type=int, default=2,
                   help="minimum number of operands (2 = a single binary op)")
    p.add_argument("--n_numbers_max", type=int, default=10)
    p.add_argument("--max_digit_len_min", type=int, default=1)
    p.add_argument("--max_digit_len_max", type=int, default=10)
    p.add_argument("--n_per_combo", type=int, default=10_000,
                   help="tasks per (n_numbers, max_digit_len) cell")
    p.add_argument("--total_steps", type=int, default=0,
                   help="override the cosine-LR total-steps estimate "
                        "(0 = the train-loop heuristic)")
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

    combos = [
        (n, d)
        for n in range(args_cli.n_numbers_min, args_cli.n_numbers_max + 1)
        for d in range(args_cli.max_digit_len_min, args_cli.max_digit_len_max + 1)
    ]
    print(f"factored grid: {len(combos)} cells, {args_cli.n_per_combo} tasks/cell "
          f"= {len(combos) * args_cli.n_per_combo:,} total tasks")
    if args_cli.task_kind == "seq":
        specs = seq_factored_specs(
            combos=combos, n_tasks_per_combo=args_cli.n_per_combo,
        )
    else:
        specs = seq_paren_factored_specs(
            combos=combos, n_tasks_per_combo=args_cli.n_per_combo,
        )

    # No per-cell avg_steps table for the factored grid; None lets
    # train_loop use its ``n_tasks * 8 / batch_size`` heuristic.
    total_steps_override = (
        args_cli.total_steps if args_cli.total_steps > 0 else None
    )

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
