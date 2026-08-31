"""CLI for the 100×100 binary-op frontier eval — one arm per invocation.

Arms map to (checkpoint, pe_kind, variant):

  pe_noinv / pe_nope / pe_rope   ablation ckpts, variant "plain"
  ctl_no_b / ctl_no_c            ablation ckpts, variants "no_b" / "no_c"
  edit_no_cleanup                ablation ckpt, variant "nocleanup"
  baseline                       the MAIN curriculum's checkpoints/<op>.pt,
                                 variant "plain" — evaluated under the
                                 identical protocol (same seeded tasks) for
                                 direct comparability with the arms

The eval is streaming and resumable (rerun the same command to continue an
interrupted grid). To parallelize one arm across GPUs, launch one process
per GPU with ``--shard r/n`` (r = 0..n-1) writing to per-shard CSVs, then
concatenate; the frontier-skip rule can also read other shards' finished
cells via ``--extra_decided``.

Run:
  CUDA_VISIBLE_DEVICES=0 python -u -m ablation.eval.eval_binary_op_ablation \
      --arm pe_noinv --op '*'
"""

from __future__ import annotations

import argparse
import os

import torch

from supervised.lib.sup_args import SupArgs

from ablation.lib.eval_frontier import run_frontier_eval
from ablation.model.agent_ablation import ARM_DIM_INSERTION, DLMAgentAblation

# arm -> (pe_kind, variant); the default ckpt/out are derived from (arm, op)
ARMS = {
    "pe_noinv": ("glpe_v1_noinv", "plain"),
    "pe_nope": ("nope_t", "plain"),
    "pe_rope": ("rope2d", "plain"),
    "ctl_no_b": ("glpe_v1", "no_b"),
    "ctl_no_c": ("glpe_v1", "no_c"),
    "edit_no_cleanup": ("glpe_v1", "nocleanup"),
    "baseline": ("glpe_v1", "plain"),
}

_OP_SLUG = {"*": "mult", "+": "add"}


def default_ckpt(arm: str, op: str) -> str:
    slug = _OP_SLUG[op]
    if arm == "baseline":
        return f"checkpoints/{slug}.pt"
    return f"ablation/checkpoints/{arm}/{slug}.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--op", choices=["*", "+"], default="*")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--l_max", type=int, default=100)
    p.add_argument("--sum_max", type=int, default=0,
                   help="only evaluate cells with l1+l2 <= this (0 = no cap); "
                        "beyond-cap cells are left absent, not written as 0")
    p.add_argument("--n_per", type=int, default=10)
    p.add_argument("--parity", choices=["even", "all"], default="even",
                   help="'even' = the default {2,4,..}² grid; 'all' = dense "
                        "(pair with --n_per 20 for the dense protocol)")
    p.add_argument("--shard", default=None,
                   help="'r/n': evaluate only cells with (l1*31+l2)%%n==r. "
                        "Use an ODD n with the default even×even grid "
                        "(l1*31+l2 is always even there, so an even n puts "
                        "every cell into the even residues)")
    p.add_argument("--extra_decided", default=None,
                   help="comma-separated CSVs whose cells are treated as "
                        "decided (skip-rule input; never re-evaluated/rewritten)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max_seq_len", type=int, default=0,
                   help="encoder length override (0 = 2500; the nocleanup "
                        "variant defaults to 6000 for its ghost-inflated tapes)")
    args_cli = p.parse_args()

    pe_kind, variant = ARMS[args_cli.arm]
    ckpt = args_cli.ckpt or default_ckpt(args_cli.arm, args_cli.op)
    slug = _OP_SLUG[args_cli.op]
    out_csv = args_cli.out or f"ablation/results/{args_cli.arm}/{slug}/accuracy.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    max_seq = args_cli.max_seq_len
    if max_seq <= 0:
        max_seq = 6000 if variant == "nocleanup" else 2500

    args = SupArgs(
        model_pe_kind=pe_kind,
        model_dim_insertion=ARM_DIM_INSERTION[pe_kind],
        model_max_seq_len=max_seq,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DLMAgentAblation(args).to(device)
    sd = torch.load(ckpt, map_location="cpu")
    missing, unexpected = agent.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    agent.eval()
    print(f"arm={args_cli.arm} ckpt={ckpt} pe_kind={pe_kind} variant={variant} "
          f"max_seq_len={max_seq} out={out_csv}")

    run_frontier_eval(
        agent, device, pad_id=args.PadToken_id, out_csv=out_csv,
        variant=variant, op=args_cli.op, l_max=args_cli.l_max,
        sum_max=args_cli.sum_max,
        shard=(tuple(int(x) for x in args_cli.shard.split("/"))
               if args_cli.shard else None),
        extra_decided=(tuple(args_cli.extra_decided.split(","))
                       if args_cli.extra_decided else ()),
        n_per=args_cli.n_per,
        parity=args_cli.parity, seed=args_cli.seed,
    )


if __name__ == "__main__":
    main()
