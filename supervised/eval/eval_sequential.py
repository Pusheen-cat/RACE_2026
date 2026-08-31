"""Sequential / nested evaluation on the factored grid.

Default grid: the diagonal n_numbers = max_digit_len ∈ 1..30 (not a
product grid — the full grid search is too expensive, so both difficulty
axes are scaled together), 100 greedy inferences per cell, per-trajectory
cap 200 000 steps. Targets compare the post-`=` substring only (the
training inputs lack '=', so the auto-eq cleanup drops the operands from
the final tape). Off-diagonal cells can still be evaluated explicitly via
``--combos_csv "n-d;n-d;..."``.

Output CSV columns: n_numbers, max_digit_len, accuracy, n_total, n_correct.
Workers stream one row per finished cell and skip cells already present in
`--out`, so a killed worker resumes when relaunched with the same arguments.

Top-level fanout (shards cells across GPUs, one subprocess per GPU):
    python -m supervised.eval.eval_sequential --fanout \
        --task_kind seq --ckpt checkpoints/seq.pt --gpus 0,1,2,3 \
        --out results/seq/accuracy.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Tuple

import torch

from model.dlm_agent import DLMAgent
from supervised.lib import eval_grids
from supervised.lib.data_gen import final_answer, make_initial_state
from supervised.lib.eval_loop import make_tg
from supervised.lib.greedy_inference import batched_run
from supervised.lib.sup_args import SupArgs


def _read_done_combos(out_csv: str):
    """(n_numbers, max_digit_len) pairs already written to ``out_csv``."""
    done = set()
    if not os.path.isfile(out_csv):
        return done
    with open(out_csv) as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.strip().split(",")
            try:
                done.add((int(parts[0]), int(parts[1])))
            except (ValueError, IndexError):
                continue
    return done


def _eval_combos(task_kind: str, ckpt: str, combos: List[Tuple[int, int]],
                 n_per: int, batch: int, out_csv: str, max_steps: int,
                 pe_kind: str = "glpe_v1", dim_insertion: int = 0):
    """Factored (n_numbers × max_digit_len) eval, one chunk of cells per
    forward pass (lazy per-cell build to keep the GPU busy from the first
    cell). Resumable: skips cells already in ``out_csv`` and appends one row
    per cell as it finishes. ``task_kind`` ∈ {seq, seq_paren}."""
    done = _read_done_combos(out_csv)
    remaining = [c for c in combos if c not in done]
    print(f"[resume] {len(done)}/{len(combos)} combos already in {out_csv}; "
          f"running {len(remaining)} remaining.")
    if not remaining:
        return

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if not os.path.isfile(out_csv):
        with open(out_csv, "w") as f:
            f.write("n_numbers,max_digit_len,accuracy,n_total,n_correct\n")

    sup_kwargs = {"model_pe_kind": pe_kind}
    if dim_insertion > 0:
        sup_kwargs["model_dim_insertion"] = dim_insertion
    args = SupArgs(**sup_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DLMAgent(args).to(device).eval()
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    agent.load_state_dict(sd)
    print(f"[ckpt] loaded {ckpt}")

    decode_fn = final_answer  # compare post-`=` only
    tg = make_tg()
    if task_kind == "seq":
        gen_one = lambda n_, d_: tg.gen_seq_factored(int(n_), int(d_))
    elif task_kind == "seq_paren":
        gen_one = lambda n_, d_: tg.gen_seq_paren_factored(int(n_), int(d_))
    else:
        raise ValueError(f"unsupported task_kind: {task_kind}")

    cells_per_chunk = max(1, batch // n_per)

    def _eval_one_chunk(cells):
        flat = []  # (bucket_idx, task, target)
        totals = []
        for bi, (n_, d_) in enumerate(cells):
            cell_pairs = [gen_one(n_, d_) for _ in range(n_per)]
            for t, g in cell_pairs:
                flat.append((bi, t, g))
            totals.append(len(cell_pairs))
        correct = [0] * len(cells)
        for start in range(0, len(flat), batch):
            sub = flat[start:start + batch]
            initials = [make_initial_state(t) for _, t, _ in sub]
            finals, success = batched_run(
                agent, initials, device,
                pad_id=args.PadToken_id, max_steps=max_steps,
            )
            for (bi, _, g), state, ok in zip(sub, finals, success):
                if ok and decode_fn(state) == g:
                    correct[bi] += 1
        return correct, totals

    for chunk_start in range(0, len(remaining), cells_per_chunk):
        cells = remaining[chunk_start:chunk_start + cells_per_chunk]
        t0 = time.time()
        correct, totals = _eval_one_chunk(cells)
        dt = time.time() - t0
        with open(out_csv, "a") as f:
            for (n_, d_), n_correct, n_total in zip(cells, correct, totals):
                acc = n_correct / max(1, n_total)
                f.write(f"{n_},{d_},{acc},{n_total},{n_correct}\n")
            f.flush()
        names = [f"N{n_}-D{d_}" for (n_, d_) in cells]
        accs = [f"{c}/{t}" for c, t in zip(correct, totals)]
        print(f"  [{chunk_start + len(cells)}/{len(remaining)}] "
              f"{'  '.join(f'{nm}:{a}' for nm, a in zip(names, accs))}  ({dt:.1f}s)")


def fanout(task_kind: str, ckpt: str, gpus: List[int],
           combos: List[Tuple[int, int]], n_per: int, batch: int,
           out_csv: str, max_steps: int,
           pe_kind: str = "glpe_v1", dim_insertion: int = 0):
    out_dir = os.path.dirname(out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    chunks: List[List[Tuple[int, int]]] = [[] for _ in gpus]
    for i, c in enumerate(combos):
        chunks[i % len(gpus)].append(c)
    procs = []
    part_files = []
    for i, gpu in enumerate(gpus):
        if not chunks[i]:
            continue
        part_csv = os.path.join(out_dir, f"{task_kind}_part{i}.csv")
        part_files.append(part_csv)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        combos_csv = ";".join(f"{n_}-{d_}" for (n_, d_) in chunks[i])
        cmd = [
            sys.executable, "-m", "supervised.eval.eval_sequential",
            "--task_kind", task_kind,
            "--ckpt", ckpt,
            "--combos_csv", combos_csv,
            "--n_per", str(n_per),
            "--batch", str(batch),
            "--max_steps", str(max_steps),
            "--out", part_csv,
            "--pe_kind", pe_kind,
            "--dim_insertion", str(dim_insertion),
        ]
        print(f"GPU {gpu}: {len(chunks[i])} cells")
        procs.append(subprocess.Popen(cmd, env=env))
    rc = 0
    for p in procs:
        rc |= p.wait()
    if rc != 0:
        print("some workers failed", file=sys.stderr)

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
        f.write("n_numbers,max_digit_len,accuracy,n_total,n_correct\n")
        for r in rows:
            f.write(r + "\n")
    print(f"Merged → {out_csv}")
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_kind", choices=["seq", "seq_paren"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--fanout", action="store_true")
    p.add_argument("--gpus", default="", help="comma-separated GPU ids for fanout")
    p.add_argument("--batch", type=int, default=200)
    p.add_argument("--combos_csv", default="",
                   help="explicit cells 'n-d;n-d;...'; default = the "
                        "canonical diagonal grid (n = d, 1..30)")
    p.add_argument("--n_per", type=int, default=eval_grids.SEQ_N_PER)
    p.add_argument("--max_steps", type=int, default=eval_grids.SEQ_MAX_STEPS,
                   help="per-trajectory greedy-inference step cap")
    p.add_argument("--out", default="",
                   help="output CSV (default results/<task>/accuracy.csv)")
    p.add_argument("--pe_kind", choices=["glpe_v1", "rope2d"], default="glpe_v1",
                   help="positional-encoding family; must match the checkpoint")
    p.add_argument("--dim_insertion", type=int, default=0,
                   help="override SupArgs.model_dim_insertion (0 = default); "
                        "must match the checkpoint")
    args = p.parse_args()

    if args.combos_csv:
        combos = []
        for tok in args.combos_csv.split(";"):
            tok = tok.strip()
            if not tok:
                continue
            n_str, d_str = tok.split("-")
            combos.append((int(n_str), int(d_str)))
    else:
        combos = eval_grids.seq_grid()

    task_name = {"seq": "seq", "seq_paren": "nest"}[args.task_kind]
    out_csv = args.out or os.path.join("results", task_name, "accuracy.csv")

    if args.fanout:
        gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
        if not gpus:
            print("--fanout needs --gpus", file=sys.stderr)
            return 1
        return fanout(
            task_kind=args.task_kind, ckpt=args.ckpt, gpus=gpus,
            combos=combos, n_per=args.n_per, batch=args.batch,
            out_csv=out_csv, max_steps=args.max_steps,
            pe_kind=args.pe_kind, dim_insertion=args.dim_insertion,
        )

    _eval_combos(
        task_kind=args.task_kind, ckpt=args.ckpt,
        combos=combos, n_per=args.n_per, batch=args.batch,
        out_csv=out_csv, max_steps=args.max_steps,
        pe_kind=args.pe_kind, dim_insertion=args.dim_insertion,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
