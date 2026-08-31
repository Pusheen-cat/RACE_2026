"""Binary-op evaluation (add / sub / mult / div) on the canonical grids.

Default grids come from `supervised.lib.eval_grids.binary_op_grid`:

  * add / sub / mult — (l1, l2) ∈ (1..50)², 100 samples/cell
  * div — (1..50)² with l2 <= l1 + 2, 100 samples/cell; only cases whose
    result is >= 0.01 are evaluated (operands are redrawn until the
    condition holds — see eval_loop.build_binary_op_buckets)

Per-trajectory step caps come from `inference_caps.compute_max_steps` per
cell. Output CSV columns: l1, l2, accuracy, n_total, n_correct. Workers
stream one row per finished cell and skip cells already present in `--out`,
so a killed worker resumes when relaunched with the same arguments.

Top-level fanout (shards cells across GPUs, one subprocess per GPU):
    python -m supervised.eval.eval_binary_op --fanout \
        --op + --ckpt checkpoints/add.pt --gpus 0,1,2,3 \
        --out results/add/accuracy.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Optional, Tuple

import torch

from supervised.lib import eval_grids
from supervised.lib.eval_loop import build_binary_op_buckets, evaluate, make_tg
from supervised.lib.inference_caps import OP_TO_TASK_KIND, compute_max_steps
from supervised.lib.sup_args import SupArgs


def _read_done_pairs(out_csv: str):
    """(l1, l2) pairs already present in ``out_csv`` (empty if missing/header-only)."""
    done = set()
    if not os.path.isfile(out_csv):
        return done
    with open(out_csv) as f:
        next(f, None)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            try:
                done.add((int(parts[0]), int(parts[1])))
            except (ValueError, IndexError):
                continue
    return done


def _eval_combos(ckpt: str, op: str, combos: List[Tuple[int, int]],
                 n_per: int, batch: int, out_csv: str,
                 pe_kind: str = "glpe_v1", dim_insertion: int = 0,
                 max_steps: int = 200_000,
                 step_cap_factor: Optional[float] = None):
    """Evaluate ``combos`` on ``ckpt``, streaming each cell's row to ``out_csv``.

    Per-trajectory step cap: ``step_cap_factor`` when given (the
    ``min(max_steps, factor × L)`` heuristic), otherwise the per-cell
    ``inference_caps.compute_max_steps`` analytical budget.

    Rows are appended as each cell finishes, so a killed worker can be
    relaunched with the same args and resumes (skips cells already in
    ``out_csv``).
    """
    done = _read_done_pairs(out_csv)
    remaining = [c for c in combos if c not in done]
    if done:
        print(f"[resume] {len(done)}/{len(combos)} combos already in {out_csv}; "
              f"running {len(remaining)} remaining.")
    if not remaining:
        return

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if not os.path.isfile(out_csv) or os.path.getsize(out_csv) == 0:
        with open(out_csv, "w") as f:
            f.write("l1,l2,accuracy,n_total,n_correct\n")

    extra = {}
    if dim_insertion > 0:
        extra["model_dim_insertion"] = dim_insertion
    args = SupArgs(model_pe_kind=pe_kind, **extra)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tg = make_tg()
    buckets = build_binary_op_buckets(tg, op, remaining, n_per_combo=n_per)
    task_kind = OP_TO_TASK_KIND[op]

    for (l1, l2), bucket in zip(remaining, buckets):
        if step_cap_factor is not None:
            # Leave None → evaluate uses the global max_steps + step_cap_factor.
            bucket.max_steps = None
        else:
            total_input_length = len(bucket.pairs[0][0]) if bucket.pairs else 0
            bucket.max_steps = compute_max_steps(
                task_kind, total_input_length=total_input_length, l1=l1, l2=l2,
            )

    label_to_pair = {f"L{l1}-L{l2}": (l1, l2) for (l1, l2) in remaining}

    def _on_bucket(label, acc, n_total, n_correct):
        l1, l2 = label_to_pair[label]
        with open(out_csv, "a") as f:
            f.write(f"{l1},{l2},{acc},{n_total},{n_correct}\n")
            f.flush()

    evaluate(
        args, ckpt, buckets,
        device=device,
        target_mode="full_tape",
        inference_batch=batch,
        max_steps=max_steps,
        step_cap_factor=step_cap_factor,
        on_bucket_complete=_on_bucket,
    )


def fanout(ckpt: str, op: str, gpus: List[int], out_csv: str, batch: int,
           pe_kind: str = "glpe_v1", dim_insertion: int = 0,
           max_steps: int = 200_000,
           step_cap_factor: Optional[float] = None):
    out_dir = os.path.dirname(out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    op_safe = {"+": "add", "-": "sub", "*": "mult", "/": "div"}[op]

    def _shard(name: str, combos: List[Tuple[int, int]], n_per: int):
        if not combos:
            return []
        chunks: List[List[Tuple[int, int]]] = [[] for _ in gpus]
        for i, c in enumerate(combos):
            chunks[i % len(gpus)].append(c)
        procs = []
        part_files = []
        for i, gpu in enumerate(gpus):
            if not chunks[i]:
                continue
            part_csv = os.path.join(out_dir, f"{op_safe}_{name}_part{i}.csv")
            part_files.append(part_csv)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            combos_csv = ";".join(f"{l1}-{l2}" for (l1, l2) in chunks[i])
            cmd = [
                sys.executable, "-m", "supervised.eval.eval_binary_op",
                "--op", op,
                "--ckpt", ckpt,
                "--combos_csv", combos_csv,
                "--n_per", str(n_per),
                "--batch", str(batch),
                "--max_steps", str(max_steps),
                "--out", part_csv,
                "--pe_kind", pe_kind,
                "--dim_insertion", str(dim_insertion),
            ]
            if step_cap_factor is not None:
                cmd += ["--step_cap_factor", str(step_cap_factor)]
            print(f"GPU {gpu} {name}: {len(chunks[i])} cells")
            procs.append(subprocess.Popen(cmd, env=env))
        rc = 0
        for p in procs:
            rc |= p.wait()
        if rc != 0:
            print(f"some {name} workers failed", file=sys.stderr)
        return part_files

    part_files = []
    for gi, (cells, n_per) in enumerate(eval_grids.binary_op_grid(op)):
        part_files += _shard(f"g{gi}", cells, n_per)

    rows = []
    for pf in part_files:
        if not os.path.isfile(pf):
            continue
        with open(pf) as f:
            next(f)
            for line in f:
                rows.append(line.strip())
    rows.sort(key=lambda r: tuple(int(x) for x in r.split(",")[:2]))
    with open(out_csv, "w") as f:
        f.write("l1,l2,accuracy,n_total,n_correct\n")
        for r in rows:
            f.write(r + "\n")
    print(f"Merged → {out_csv}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["+", "-", "*", "/"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--fanout", action="store_true")
    p.add_argument("--gpus", default="", help="comma-separated GPU ids for fanout")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--combos_csv", default="",
                   help="worker-mode cells: 'l1-l2;l1-l2;...'")
    p.add_argument("--n_per", type=int, default=100,
                   help="worker-mode inferences per cell")
    p.add_argument("--max_steps", type=int, default=200_000,
                   help="global step ceiling; the per-cell compute_max_steps "
                        "budget applies below it")
    p.add_argument("--step_cap_factor", type=float, default=None,
                   help="per-trajectory cap = min(max_steps, factor × "
                        "input_len); bounds throughput when OOD stragglers "
                        "would hold the whole batch alive")
    p.add_argument("--out", default="",
                   help="output CSV (default results/<op>/accuracy.csv)")
    p.add_argument("--pe_kind", choices=["glpe_v1", "rope2d"], default="glpe_v1",
                   help="positional-encoding family; must match the trained checkpoint")
    p.add_argument("--dim_insertion", type=int, default=0,
                   help="override SupArgs.model_dim_insertion (0 = keep default); "
                        "must match the trained checkpoint")
    args = p.parse_args()

    op_safe = {"+": "add", "-": "sub", "*": "mult", "/": "div"}[args.op]
    out_csv = args.out or os.path.join("results", op_safe, "accuracy.csv")

    if args.fanout:
        gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
        if not gpus:
            print("--fanout needs --gpus", file=sys.stderr)
            return 1
        return fanout(
            ckpt=args.ckpt, op=args.op, gpus=gpus, out_csv=out_csv,
            batch=args.batch,
            pe_kind=args.pe_kind, dim_insertion=args.dim_insertion,
            max_steps=args.max_steps, step_cap_factor=args.step_cap_factor,
        )

    if args.combos_csv:
        combos = []
        for piece in args.combos_csv.split(";"):
            if not piece.strip():
                continue
            a, b = piece.split("-")
            combos.append((int(a), int(b)))
        _eval_combos(
            ckpt=args.ckpt, op=args.op, combos=combos,
            n_per=args.n_per, batch=args.batch, out_csv=out_csv,
            pe_kind=args.pe_kind, dim_insertion=args.dim_insertion,
            max_steps=args.max_steps, step_cap_factor=args.step_cap_factor,
        )
        return 0

    # No --combos_csv: run the full canonical grid in this process.
    for gi, (cells, n_per) in enumerate(eval_grids.binary_op_grid(args.op)):
        _eval_combos(
            ckpt=args.ckpt, op=args.op, combos=cells,
            n_per=n_per, batch=args.batch, out_csv=out_csv,
            pe_kind=args.pe_kind, dim_insertion=args.dim_insertion,
            max_steps=args.max_steps, step_cap_factor=args.step_cap_factor,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
