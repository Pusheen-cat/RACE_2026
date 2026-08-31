"""Bucketed greedy-inference evaluator.

Each bucket: a list of `(task_str, target_str)` pairs. We run the model with
`greedy_inference.batched_run`, decode the final tape, and compare to the
target string. Mode `full_tape` compares the full tape (`123=123`); mode
`answer_only` compares only the post-`=` substring (the sequential / nested
stages, whose auto-eq cleanup keeps just the answer).

Returns per-bucket accuracy plus counts so multi-GPU launchers can write
detailed CSVs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from model.dlm_agent import DLMAgent

from supervised.lib.data_gen import (
    final_answer, final_string, make_initial_state, make_task_generator,
)
from supervised.lib.greedy_inference import batched_run
from supervised.lib.sup_args import SupArgs


@dataclass
class Bucket:
    """One evaluation bucket: a label + a list of (task, target) pairs.

    ``max_steps`` (optional) is the per-trajectory greedy-inference step cap
    applied to every pair in this bucket; populate it via
    ``inference_caps.compute_max_steps`` at the call site so the formula stays
    in one place. ``None`` falls back to the global ``max_steps`` / the
    ``step_cap_factor`` heuristic passed to :func:`evaluate`.
    """
    label: str
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    max_steps: Optional[int] = None


def build_copy_buckets(tg, lengths: Sequence[int], n_per: int):
    out = []
    for L in lengths:
        b = Bucket(label=f"L{L}")
        for _ in range(n_per):
            t, g = tg.gen_task_1_copy(int(L))
            b.pairs.append((t, g))
        out.append(b)
    return out


def build_binary_op_buckets(
    tg,
    op: str,
    combos: Sequence[Tuple[int, int]],
    n_per_combo,  # int or callable(l1, l2) -> int
):
    from supervised.lib.data_gen import _binary_op_strs
    from supervised.lib.eval_grids import DIV_MIN_RESULT
    # Division eval only considers cases whose result is >= DIV_MIN_RESULT
    # (0.01, the smallest value the 3-decimal targets can express); the
    # pair generator redraws operands until the condition holds.
    min_div_result = DIV_MIN_RESULT if op == "/" else None
    out = []
    for l1, l2 in combos:
        b = Bucket(label=f"L{l1}-L{l2}")
        n = n_per_combo if isinstance(n_per_combo, int) else n_per_combo(l1, l2)
        for _ in range(n):
            # Eval only needs (task_str, target_str); the trajectory the
            # model is graded against is produced inside batched_run.
            # Skipping the no-op teacher dispatch makes bucket build O(n)
            # instead of O(n * trajectory_init).
            t, g = _binary_op_strs(tg, int(l1), int(l2), op,
                                   min_div_result=min_div_result)
            b.pairs.append((t, g))
        out.append(b)
    return out


def build_seq_factored_buckets(
    tg, combos: Sequence[Tuple[int, int]], n_per: int,
):
    """One bucket per (n_numbers, max_digit_len) cell of the seq factored grid."""
    out = []
    for n_, d_ in combos:
        b = Bucket(label=f"N{n_}-D{d_}")
        for _ in range(n_per):
            t, g = tg.gen_seq_factored(int(n_), int(d_))
            b.pairs.append((t, g))
        out.append(b)
    return out


def build_seq_paren_factored_buckets(
    tg, combos: Sequence[Tuple[int, int]], n_per: int,
):
    """One bucket per (n_numbers, max_digit_len) cell of the nested factored grid."""
    out = []
    for n_, d_ in combos:
        b = Bucket(label=f"N{n_}-D{d_}")
        for _ in range(n_per):
            t, g = tg.gen_seq_paren_factored(int(n_), int(d_))
            b.pairs.append((t, g))
        out.append(b)
    return out


def evaluate(
    args: SupArgs,
    ckpt_path: str,
    buckets: Sequence[Bucket],
    *,
    device: torch.device,
    target_mode: str,            # "full_tape" or "answer_only"
    inference_batch: int = 64,
    max_steps: int = 5_000,
    step_cap_factor: Optional[float] = None,
    on_bucket_complete: Optional[
        Callable[[str, float, int, int], None]
    ] = None,
):
    """Returns list[(bucket_label, accuracy_float, n_total, n_correct)].

    Multiple eval "settings" (buckets) are flattened into a single stream and
    chunked by ``inference_batch`` **across bucket boundaries**, so one merged
    ``batched_run`` call can span several buckets (a bucket with fewer pairs
    than ``inference_batch`` no longer caps the batch at its own size). Each
    item carries its own per-trajectory step cap, and outputs are split back to
    the originating bucket; ``on_bucket_complete(label, acc, n_total, n_correct)``
    fires as each bucket's last pair returns (streaming to disk for resumable
    workers).

    Per-trajectory ``max_steps`` is each bucket's ``max_steps`` field when set
    (populate via ``inference_caps.compute_max_steps``); buckets that leave it
    ``None`` fall back to the global ``max_steps`` / ``step_cap_factor``.
    """
    assert target_mode in {"full_tape", "answer_only"}
    agent = DLMAgent(args).to(device).eval()
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    agent.load_state_dict(sd)

    decode_fn = final_string if target_mode == "full_tape" else final_answer

    # Flatten pairs across buckets so a single inference batch can span
    # consecutive buckets. Buckets are processed in order, so a bucket
    # "completes" as soon as the last of its pairs returns.
    flat: List[Tuple[int, str, str]] = []  # (bucket_idx, task, target)
    flat_max_steps: List[Optional[int]] = []
    for bi, bucket in enumerate(buckets):
        for t, g in bucket.pairs:
            flat.append((bi, t, g))
            flat_max_steps.append(bucket.max_steps)
    # Per-item caps apply whenever any bucket set its own max_steps; missing
    # caps fall back to the global max_steps in that branch.
    use_per_item_cap = any(b.max_steps is not None for b in buckets)

    n_buckets = len(buckets)
    totals = [len(b.pairs) for b in buckets]
    remaining = list(totals)
    correct = [0] * n_buckets
    results: List[Optional[Tuple[str, float, int, int]]] = [None] * n_buckets

    for start in range(0, len(flat), inference_batch):
        chunk = flat[start:start + inference_batch]
        initials = [make_initial_state(t) for _, t, _ in chunk]
        if use_per_item_cap:
            chunk_caps = [
                ms if ms is not None else max_steps
                for ms in flat_max_steps[start:start + inference_batch]
            ]
            finals, success = batched_run(
                agent, initials, device,
                pad_id=args.PadToken_id,
                max_steps=max(chunk_caps) if chunk_caps else max_steps,
                per_item_max_steps=chunk_caps,
            )
        else:
            finals, success = batched_run(
                agent, initials, device,
                pad_id=args.PadToken_id, max_steps=max_steps,
                step_cap_factor=step_cap_factor,
            )
        for (bi, t, g), state, ok in zip(chunk, finals, success):
            produced = decode_fn(state)
            if ok and produced == g:
                correct[bi] += 1
            remaining[bi] -= 1
            if remaining[bi] == 0:
                n_total = totals[bi]
                n_correct = correct[bi]
                acc = n_correct / max(1, n_total)
                label = buckets[bi].label
                results[bi] = (label, acc, n_total, n_correct)
                print(f"  {label}: acc={acc:.3f} ({n_correct}/{n_total})")
                if on_bucket_complete is not None:
                    on_bucket_complete(label, acc, n_total, n_correct)
    return results


def write_csv(results: Sequence, csv_path: str, header: Sequence[str]):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in results:
            f.write(",".join(str(c) for c in row) + "\n")
    print(f"Wrote {csv_path}")


def make_tg():
    return make_task_generator()
