"""Copy-task evaluation: accuracy across lengths 1..500.

100 greedy inferences per length. `--fanout --gpus 0,1,...` shards the
lengths across GPUs by spawning one subprocess per GPU and merges the part
CSVs into one `--out` file.

Per-process (worker) mode, used by the launcher:
    CUDA_VISIBLE_DEVICES=<gpu> python -m supervised.eval.eval_copy \
        --ckpt checkpoints/copy.pt --lengths_csv 1,2,...,500 \
        --out results/copy/accuracy_part0.csv

Top-level fanout:
    python -m supervised.eval.eval_copy --fanout \
        --ckpt checkpoints/copy.pt --gpus 0,1,2,3 \
        --out results/copy/accuracy.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List

import torch

from supervised.lib import eval_grids
from supervised.lib.eval_loop import (
    build_copy_buckets, evaluate, make_tg, write_csv,
)
from supervised.lib.inference_caps import compute_max_steps
from supervised.lib.sup_args import SupArgs


def _run_worker(
    ckpt: str, lengths: List[int], n_per: int, out_csv: str, batch: int,
    pe_kind: str = "glpe_v1", dim_insertion: int = 0,
):
    extra = {}
    if dim_insertion > 0:
        extra["model_dim_insertion"] = dim_insertion
    args = SupArgs(model_pe_kind=pe_kind, **extra)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tg = make_tg()
    buckets = build_copy_buckets(tg, lengths, n_per)
    for bucket in buckets:
        total_input_length = len(bucket.pairs[0][0]) if bucket.pairs else 0
        bucket.max_steps = compute_max_steps(
            "copy", total_input_length=total_input_length,
        )
    results = evaluate(
        args, ckpt, buckets,
        device=device,
        target_mode="full_tape",
        inference_batch=batch,
    )
    rows = [(label.replace("L", ""), acc, n_total, n_correct)
            for (label, acc, n_total, n_correct) in results]
    write_csv(rows, out_csv, header=["length", "accuracy", "n_total", "n_correct"])


def fanout(ckpt: str, gpus: List[int], lengths: List[int], n_per: int,
           batch: int, out_csv: str,
           pe_kind: str = "glpe_v1", dim_insertion: int = 0):
    out_dir = os.path.dirname(out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    chunks: List[List[int]] = [[] for _ in gpus]
    for i, L in enumerate(lengths):
        chunks[i % len(gpus)].append(L)
    procs = []
    part_files = []
    for i, gpu in enumerate(gpus):
        if not chunks[i]:
            continue
        part_csv = os.path.join(out_dir, f"copy_part{i}.csv")
        part_files.append(part_csv)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            sys.executable, "-m", "supervised.eval.eval_copy",
            "--ckpt", ckpt,
            "--lengths_csv", ",".join(str(L) for L in chunks[i]),
            "--n_per", str(n_per),
            "--batch", str(batch),
            "--out", part_csv,
            "--pe_kind", pe_kind,
            "--dim_insertion", str(dim_insertion),
        ]
        print(f"GPU {gpu}: lengths={chunks[i][:5]}... ({len(chunks[i])} total)")
        procs.append(subprocess.Popen(cmd, env=env))
    rc = 0
    for p in procs:
        rc |= p.wait()
    if rc != 0:
        print("at least one worker failed", file=sys.stderr)
        return rc

    # Merge.
    rows = []
    for pf in part_files:
        with open(pf) as f:
            next(f)
            for line in f:
                rows.append(line.strip())
    rows.sort(key=lambda r: int(r.split(",")[0]))
    with open(out_csv, "w") as f:
        f.write("length,accuracy,n_total,n_correct\n")
        for r in rows:
            f.write(r + "\n")
    print(f"Merged → {out_csv}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--fanout", action="store_true")
    p.add_argument("--gpus", default="", help="comma-separated GPU ids for fanout")
    p.add_argument("--n_per", type=int, default=eval_grids.COPY_N_PER)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out", default="results/copy/accuracy.csv")
    p.add_argument("--lengths_csv", default="",
                   help="explicit length list; default = the canonical 1..500 grid")
    p.add_argument("--pe_kind", choices=["glpe_v1", "rope2d"], default="glpe_v1",
                   help="positional-encoding family; must match the trained checkpoint")
    p.add_argument("--dim_insertion", type=int, default=0,
                   help="override SupArgs.model_dim_insertion (0 = keep default); "
                        "must match the trained checkpoint")
    args = p.parse_args()

    if args.lengths_csv:
        lengths = [int(x) for x in args.lengths_csv.split(",") if x.strip()]
    else:
        lengths = eval_grids.copy_lengths()

    if args.fanout:
        gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
        if not gpus:
            print("--fanout needs --gpus", file=sys.stderr)
            return 1
        return fanout(args.ckpt, gpus, lengths, args.n_per, args.batch,
                      args.out, pe_kind=args.pe_kind,
                      dim_insertion=args.dim_insertion)

    _run_worker(args.ckpt, lengths, args.n_per, args.out, args.batch,
                pe_kind=args.pe_kind, dim_insertion=args.dim_insertion)


if __name__ == "__main__":
    sys.exit(main() or 0)
