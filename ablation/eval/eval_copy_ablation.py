"""Copy-task length-generalization eval for the ablation arms.

Lengths 1..500, 100 inferences/length (the main release's copy grid),
per-item step cap ``compute_max_steps("copy") = L + 5``, success = full-tape
match. Streaming, resumable CSV ``length,accuracy,n_total,n_correct``
(same schema as the release copy results). Tasks are seeded per (seed,
length), so every arm sees the same inputs.

Arms: the five copy-trained ablation arms (pe_noinv / pe_nope / pe_rope /
ctl_no_b / ctl_no_c — their variant hooks mirror the frontier eval), plus
``baseline`` = the main curriculum's ``checkpoints/copy.pt`` evaluated on
the identical seeded tasks. ``edit_no_cleanup`` also works if its copy arm
was trained, but on copy the no-cleanup pipeline is verifiably identical to
the clean one (the '=' ships inside the task string, so cleanup never
fires), so the baseline-equivalent control is normally skipped.

To parallelize one arm across GPUs, launch one process per GPU with
disjoint ``--len_min`` / ``--len_max`` windows writing to per-shard CSVs,
then concatenate.

Run:
  CUDA_VISIBLE_DEVICES=0 python -u -m ablation.eval.eval_copy_ablation \
      --arm pe_noinv
"""

from __future__ import annotations

import argparse
import csv
import os
import random

import torch

from teacher.state import TeacherState

from supervised.lib.data_gen import make_task_generator
from supervised.lib.inference_caps import compute_max_steps
from supervised.lib.sup_args import SupArgs

from ablation.lib.eval_frontier import VARIANTS, tape_string
from ablation.lib.greedy_ablation import batched_run_ablation
from ablation.model.agent_ablation import ARM_DIM_INSERTION, DLMAgentAblation

# arm -> (default ckpt, pe_kind, variant)
ARMS = {
    "pe_noinv": ("ablation/checkpoints/pe_noinv/copy.pt",
                 "glpe_v1_noinv", "plain"),
    "pe_nope": ("ablation/checkpoints/pe_nope/copy.pt",
                "nope_t", "plain"),
    "pe_rope": ("ablation/checkpoints/pe_rope/copy.pt",
                "rope2d", "plain"),
    "ctl_no_b": ("ablation/checkpoints/ctl_no_b/copy.pt",
                 "glpe_v1", "no_b"),
    "ctl_no_c": ("ablation/checkpoints/ctl_no_c/copy.pt",
                 "glpe_v1", "no_c"),
    "edit_no_cleanup": ("ablation/checkpoints/edit_no_cleanup/copy.pt",
                        "glpe_v1", "nocleanup"),
    "baseline": ("checkpoints/copy.pt",
                 "glpe_v1", "plain"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--len_min", type=int, default=1)
    p.add_argument("--len_max", type=int, default=500)
    p.add_argument("--n_per", type=int, default=100)
    p.add_argument("--batch", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    args_cli = p.parse_args()

    ckpt, pe_kind, variant = ARMS[args_cli.arm]
    if args_cli.ckpt:
        ckpt = args_cli.ckpt
    cfg = VARIANTS[variant]
    out_csv = args_cli.out or f"ablation/results/{args_cli.arm}/copy/accuracy.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    args = SupArgs(model_pe_kind=pe_kind,
                   model_dim_insertion=ARM_DIM_INSERTION[pe_kind])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DLMAgentAblation(args).to(device)
    sd = torch.load(ckpt, map_location="cpu")
    missing, unexpected = agent.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    agent.eval()
    print(f"arm={args_cli.arm} ckpt={ckpt} pe_kind={pe_kind} "
          f"variant={variant} out={out_csv}")

    done_lengths = set()
    if os.path.isfile(out_csv):
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                done_lengths.add(int(row["length"]))
        print(f"resuming: {len(done_lengths)} lengths already present")
    new_file = not os.path.isfile(out_csv)
    csv_f = open(out_csv, "a", newline="")
    writer = csv.writer(csv_f)
    if new_file:
        writer.writerow(["length", "accuracy", "n_total", "n_correct"])
        csv_f.flush()

    tg = make_task_generator()
    for L in range(args_cli.len_min, args_cli.len_max + 1):
        if L in done_lengths:
            continue
        random.seed(args_cli.seed * 1_000_003 + L)
        pairs = [tg.gen_task_1_copy(L) for _ in range(args_cli.n_per)]
        cap = compute_max_steps("copy", total_input_length=L)
        n_correct = 0
        for k in range(0, len(pairs), args_cli.batch):
            chunk = pairs[k:k + args_cli.batch]
            states = [TeacherState.from_text(t, ctl_size=3) for t, _ in chunk]
            finals, success = batched_run_ablation(
                agent, states, device, args.PadToken_id,
                per_item_max_steps=[cap] * len(chunk),
                engine_fn=cfg["engine_fn"],
                use_can_dispatch=cfg["use_can_dispatch"],
                state_ctl_zero=cfg["state_ctl_zero"],
                action_ctl_zero=cfg["action_ctl_zero"],
            )
            # Copy: the no-cleanup reference tape == the plain target
            # (cleanup never fires — '=' ships inside the task string), so
            # one success criterion serves every variant.
            for (t, target), st, ok in zip(chunk, finals, success):
                if ok and tape_string(st) == target:
                    n_correct += 1
        writer.writerow([L, n_correct / len(pairs), len(pairs), n_correct])
        csv_f.flush()
        print(f"  L={L}: {n_correct}/{len(pairs)}")
    csv_f.close()
    print("copy eval complete")


if __name__ == "__main__":
    main()
