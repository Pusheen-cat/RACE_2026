# Ablation study

Ablation of the model's three signature features — the **GLPE-v1
positional encoding**, the **control elements** (b)/(c), and the
**done-stage auto-eq cleanup** of the tape-editing semantics — on the
supervised-imitation pipeline. Six experimental settings ("arms"), each a
single targeted change to an otherwise-identical pipeline:

| Arm | Exp | What is ablated | pe_kind | dim_ins |
|---|---|---|---|---|
| `pe_noinv` | 1 PE | back-direction sign flip ("position inversion") removed | `glpe_v1_noinv` | 48 |
| `pe_nope`  | 1 PE | T-axis NoPE — no sequence-axis rotation, no window; insertion-axis RoPE kept; sign flip kept | `nope_t` | 24 |
| `pe_rope`  | 1 PE | GLPE-v1 replaced by plain 2D RoPE | `rope2d` | 24 |
| `ctl_no_b` | 2 ctrl | control element (b) = CTRL_RESULT hidden from model inputs | `glpe_v1` | 48 |
| `ctl_no_c` | 2 ctrl | control element (c) = CTRL_TAG_C removed from inputs + labels + loss, forced 0 at inference | `glpe_v1` | 48 |
| `edit_no_cleanup` | 3 edit | done-stage auto-eq cleanup disabled — the pre-`=` content of auto-eq self-calls stays on the tape as "ghost" cells | `glpe_v1` | 48 |

The **baseline** for every comparison is the main curriculum's model
(`checkpoints/{copy,add,mult}.pt`); the eval CLIs accept `--arm baseline`
so it can be re-evaluated under the identical protocol (same seeded tasks).

This package only *imports* the release's `model/`, `teacher/`, `dataset/`
and `supervised/` packages — it never modifies them. Everything below runs
**from the release root**.

## Training protocol

Three stages per arm, chained in the **main-curriculum order** with the
main-curriculum settings; only the ablated component differs:

| Stage | Grid | Tasks | LR | Init |
|---|---|---|---|---|
| copy | lengths 1–40 | 250 k/length (10 M) | 2e-4 | from scratch |
| add (`+`) | (l1, l2) ∈ {1..10}² | 100 k/cell (10 M) | 2e-4 | the arm's own `copy.pt` |
| mult (`*`) | (l1, l2) ∈ {1..10}² | 10 k/cell (1 M) | 2e-4 | the arm's own `add.pt` |

One epoch, bf16, AdamW, cosine LR sized from `teacher/stats/`. The copy
stage is **skipped for `edit_no_cleanup`**: copy tasks carry `=` inside the
task string, so the auto-append / cleanup path never fires and the arm is
bit-identical to the baseline on copy (verified by the gate below). Its add
stage therefore initializes from the main curriculum's `checkpoints/copy.pt`
— the equivalent checkpoint — which is the only training-side dependency on
the main pipeline (train the main copy stage first for that arm; all other
arms are fully self-contained). `--arm baseline` evaluations additionally
need the main `checkpoints/{copy,add,mult}.pt`.

`edit_no_cleanup`'s binary-op stages use **length-bucketed dynamic
batching** (`ablation/lib/train_loop_ablation.py`): the augmented tapes put
~2 % of steps at visible lengths up to ~1200 tokens, so the per-bucket
batch is `clamp(mem_K / T², 8, batch_size)` — short steps train at full
batch, rare long steps get small batches instead of OOM or 100× padding
waste. `--mem_K` defaults to 3e7 (calibrated for ~90 GB GPUs); scale it
down proportionally on smaller cards.

```bash
# one arm, all stages, one GPU:
CUDA_VISIBLE_DEVICES=0 bash ablation/scripts/train_arm.sh pe_noinv
# arms are independent — run them on separate GPUs in parallel:
CUDA_VISIBLE_DEVICES=1 bash ablation/scripts/train_arm.sh ctl_no_b &
CUDA_VISIBLE_DEVICES=2 bash ablation/scripts/train_arm.sh edit_no_cleanup &
# resume one arm at a stage (copy | add | mult):
CUDA_VISIBLE_DEVICES=0 bash ablation/scripts/train_arm.sh pe_noinv 2000 16 mult
```

Individual stages: `python -m ablation.train.train_copy_ablation --arm <arm>`
and `python -m ablation.train.train_binary_op_ablation --op +|'*' --arm <arm>`
(see their docstrings for every flag). Checkpoints land in
`ablation/checkpoints/<arm>/{copy,add,mult}.pt`, logs in
`ablation/results/<arm>/<stage>/train.log`.

## Evaluation protocol

* **copy** (`ablation.eval.eval_copy_ablation`) — lengths 1..500,
  100 greedy inferences/length, per-item step cap `L + 5`, success =
  self-terminated **and** full-tape match. Streaming, resumable CSV
  `length,accuracy,n_total,n_correct`.
* **add / mult frontier** (`ablation.eval.eval_binary_op_ablation`) —
  (l1, l2) over the **even×even** grid {2, 4, .., 100}² (`--parity all
  --n_per 20` restores the dense version), evaluated in ascending
  (l1+l2, l1) order, 10 samples/cell, per-item step cap from
  `inference_caps.compute_max_steps`. Consecutive cells are packed into
  merged inference batches of 100–200 items. **Frontier skip**: a cell
  (A, B) is recorded as 0/n without running when every already-decided cell
  with a ≥ A−2, b ≥ B−2 (non-empty set) has zero correct — the accuracy
  frontier is mapped without spending GPU time deep inside the dead zone.
  Streaming, resumable CSV `l1,l2,accuracy,n_total,n_correct,skipped`.

Tasks are seeded per (seed, cell), so **every arm is evaluated on the same
inputs**. Success criteria per arm:

* `pe_*`, `baseline` — plain: `final_done == 1` and the full tape equals
  `a·b=answer`.
* `ctl_no_b` — the (b) bit is zeroed in the model's input states (the
  engine still sets it internally; the model just never sees it).
* `ctl_no_c` — (c) zeroed in input states **and** forced to 0 in decoded
  actions.
* `edit_no_cleanup` — the no-cleanup engine replaces
  `teacher.engine.apply_action`, the per-step `can_dispatch` guard is
  disabled (ghost tokens make visible tapes non-dispatchable by design),
  and success compares the full tape against the **teacher's augmented
  reference tape**, computed per sample by the dual engine
  (`engine_nocleanup.dual_run`). The answer sits in that reference
  interleaved with the retained ghost segments, so string conventions like
  "everything after the last `=`" do not identify it — the whole tape must
  match. The encoder length limit is raised to 6000 for this arm
  (augmented visible slices reach ~1154 at (10,10) during training and
  grow further out-of-distribution).

```bash
# one arm, all tasks, one GPU per task:
bash ablation/scripts/eval_arm.sh pe_noinv 0,1,2
# the baseline under the identical protocol:
bash ablation/scripts/eval_arm.sh baseline 3
```

Every eval is resumable — rerun the same command to continue. To
parallelize a single frontier eval across GPUs, run one worker per GPU with
`--shard r/n` (cells are partitioned by `(31·l1+l2) mod n`) into per-shard
CSVs and concatenate; `--extra_decided a.csv,b.csv` feeds other shards'
finished cells to the skip rule without re-evaluating them. Results land in
`ablation/results/<arm>/<task>/accuracy.csv`.

## Verification gates

Three gates check the ablation mechanisms themselves (CPU-only, no
checkpoint needed):

```bash
python -m ablation.verify.verify_pe_variants   # ablation encoder == release encoder
                                               # for the shared pe_kinds; noinv/nope
                                               # table + flag properties; forward pass
python -m ablation.verify.verify_ctl_mask      # masked-action replay through the real
                                               # engine reproduces the masked states
                                               # exactly and still solves the task
python -m ablation.verify.verify_nocleanup     # augmented trajectories replay exactly
                                               # under the inference-side no-cleanup
                                               # engine; non-ghost projection == target
```

`verify_pe_variants` additionally checks that the main-curriculum
checkpoint loads into every arm with zero missing/unexpected keys when
`checkpoints/copy.pt` exists.

## Layout

```
ablation/
  model/    backbone_ablation.py (PE variants + sign-flip switch, otherwise a
            verified copy of model/dlm_backbone.py), agent_ablation.py
  lib/      ctl_mask.py (exp 2), engine_nocleanup.py (exp 3 dual engine),
            data_gen_ablation.py, greedy_ablation.py (variant-hooked greedy
            loop), train_loop_ablation.py (+ bucketed batching),
            eval_frontier.py (shared frontier evaluator)
  train/    train_copy_ablation.py, train_binary_op_ablation.py
  eval/     eval_copy_ablation.py, eval_binary_op_ablation.py
  verify/   the three gates
  scripts/  train_arm.sh, eval_arm.sh
  checkpoints/, results/   created at runtime (not shipped)
```
