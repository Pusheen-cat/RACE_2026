# RDLM — supervised curriculum code release (EMNLP 2026)

# Under code cleaning; expect full release on early October

Code release for the EMNLP 2026 paper. A single small Transformer learns to
solve math-expression tasks by **iteratively editing a token tape** — replace
/ delete / insert tokens, recurse into sub-problems via self-calls, and
decide when to stop — instead of generating the answer left-to-right. The
model is trained purely by **supervised imitation of rule-based teacher
trajectories**, then evaluated on length generalization far beyond its
training range.

## Layout

```
teacher/            rule-based teacher: tape state, action format, engine,
                    per-task rules, dispatcher, trajectory-length stats;
                    teacher/rules/*.md documents every rule with executed
                    example trajectories (start at teacher/rules/README.md)
model/              DLM encoder backbone, positional encodings, agent wrapper
dataset/            task-string generation + character tokenizer
supervised/
  lib/              config, streaming dataset, train loop, greedy inference,
                    eval loop, canonical eval grids, step caps
  train/            per-stage training entry points
  eval/             per-task evaluation entry points (multi-GPU fanout)
scripts/
  train_curriculum.sh   sequential 7-stage training of one model
  eval_all.sh           evaluation of all 7 tasks, sharded across GPUs
ablation/           ablation study: six experimental settings ablating the
                    positional encoding, the control elements, and the
                    edit-base cleanup (self-contained; see ablation/README.md)
```

Install: Python ≥ 3.10 and `pip install -r requirements.txt` (PyTorch with
CUDA). Everything below runs **from this directory**.

## Tasks and curriculum

One model is trained continuously through seven stages; each stage loads the
previous stage's checkpoint:

| # | Task | Form | Training range | Tasks | LR |
|---|------|------|----------------|-------|----|
| 1 | copy | `a=` | digit length 1–40 | 250 k / length (10 M) | 2e-4 |
| 2 | addition | `a+b=` | (l1, l2) ∈ {1..10}² | 100 k / cell (10 M) | 2e-4 |
| 3 | multiplication | `a*b=` | (l1, l2) ∈ {1..10}² | 10 k / cell (1 M) | 2e-4 |
| 4 | subtraction | `a-b=` | (l1, l2) ∈ {1..10}² | 10 k / cell (1 M) | 2e-4 |
| 5 | division | `a/b=` | (l1, l2) ∈ {1..10}², l2 ≤ l1+2, result ≥ 0.01 | 10 k / cell (720 k) | 2e-4 |
| 6 | sequential | `a+b*c-…` (no `=`) | (n_numbers, max_digit_len) ∈ {2..10}×{1..10} | 10 k / cell (900 k) | 2e-4 |
| 7 | nested | same with parentheses | same factored grid | 10 k / cell (900 k) | 2e-4 |

Each stage is one epoch over a stream of freshly generated teacher
trajectories (per-step imitation NLL: vocab cross-entropy + control-bit
Bernoulli + done bit), cosine LR with 1 % linear warmup, bf16 autocast,
AdamW, grad-clip 1.0. The division stage uses only cases whose result is at
least 0.01, the same restriction the evaluation applies: infeasible cells
(l2 > l1+2) are pruned from its grid (72/100 remain) and operand pairs are
redrawn until the condition holds.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_curriculum.sh            # full run
CUDA_VISIBLE_DEVICES=0 bash scripts/train_curriculum.sh 2000 16 sub  # resume at a stage
```

Checkpoints land in `checkpoints/<task>.pt`, logs in
`results/<task>/train.log`. Batch size 2000 @ bf16 fits every stage in
< 60 GB; reduce it on smaller GPUs (the cosine schedule adapts via the
step-count estimate). Training throughput is bounded by CPU trajectory
generation, so keep `num_workers` high.

## Evaluation

Greedy decoding only (argmax vocab with special tokens suppressed, threshold
0 for control / done bits). A sample counts as correct iff the model
terminates by itself (`done` at depth 0) **and** the decoded tape exactly
matches the target — for copy and the binary ops the full `expr=answer`
tape, for sequential / nested the post-`=` answer (their inputs carry no
`=`, so the engine's auto-eq cleanup leaves only the answer). Per-trajectory
step budgets come from per-task analytic formulas
(`supervised/lib/inference_caps.py`); running out of budget, producing a
tape the teacher dispatcher cannot parse, exceeding depth/length bounds, or
revisiting a state (a loop, since greedy decoding is deterministic) all
count as failures.

Canonical evaluation grids (`supervised/lib/eval_grids.py`):

| Task | Grid | Samples/cell |
|------|------|--------------|
| copy | lengths 1–500 | 100 |
| addition | (l1, l2) ∈ (1..50)² | 100 |
| multiplication | (l1, l2) ∈ (1..50)² | 100 |
| subtraction | (l1, l2) ∈ (1..50)² | 100 |
| division | (1..50)² with l2 ≤ l1+2 (1 372 cells) | 100 |
| sequential | diagonal n_numbers = max_digit_len ∈ 1..30 | 100 |
| nested | diagonal n_numbers = max_digit_len ∈ 1..30 | 100 |

Division only evaluates cases whose result is at least 0.01 (the smallest
value the 3-decimal targets can express): infeasible cells (l2 > l1+2) are
excluded, and within the remaining cells the operand pairs are redrawn
until the condition holds. The sequential / nested tasks are evaluated on
the diagonal of their factored grid rather than the full (n_numbers,
max_digit_len) product — trajectory length grows with both axes, making
the full grid search prohibitively expensive — so both difficulty axes
scale together; off-diagonal cells can still be requested explicitly via
`--combos_csv`.

Each task is evaluated with its own stage checkpoint. Evaluation runs in
parallel on separate GPUs: the grid is sharded round-robin across the GPU
list, one worker subprocess per GPU, and part CSVs are merged into
`results/<task>/accuracy.csv`:

```bash
bash scripts/eval_all.sh 0,1,2,3                # all seven tasks
bash scripts/eval_all.sh 4,5 "div seq nest"     # subset
```

Workers stream one CSV row per finished cell and skip cells already present
in their output file, so rerunning the same command resumes an interrupted
evaluation. Single-task invocations (`python -m supervised.eval.eval_binary_op
--fanout --op + ...`) and custom cells (`--combos_csv "3-7;12-9"`) are also
supported; see each module's docstring.

## Model

`model/dlm_backbone.py` — a 2-layer, 4-head, 384-dim bidirectional
Transformer (**3 560 520 parameters**) over the visible tape plus
`max_insertion = 3` insertion slots per position, with:

* **directional attention** — forward/backward RoPE halves whose backward
  scores are sign-flipped for j > i;
* **GLPE-v1 positional encoding** (default) — per head, dedicated global
  (NoPE) channels plus window-masked local channels on the token and
  insertion axes; plain 2D RoPE (`--pe_kind rope2d --dim_insertion 24`) is
  also implemented;
* a weight-tied LM head for the vocab logits, a control-embedding readout
  for the control bits, and a done logit read off the tape's end.

The architecture descends from a time-conditioned discrete diffusion model.
The timestep was fixed to 0 throughout this work, so the released model
removes the time input entirely: the timestep-embedding MLP is deleted, and
each block's AdaLN head — which computed a constant at t = 0 — is replaced
by directly learnable modulation vectors (6 × 384 per block, 2 × 384 at the
final norm). This preserves the encoder computation exactly while removing
about 48 % of the original parameter count.

## Teacher trajectories

`teacher/` holds the rule-based teacher that produces the imitation targets:
a deterministic, per-step Markov function of the visible tape that solves
every task by the same tape-editing action space the model uses (digit-wise
arithmetic with carries/borrows, long multiplication and division via
recursive self-calls, operator-by-operator reduction of sequential and
parenthesized expressions). The rules defer `done = 1` until every tape
digit carries the persistent "clean" tag, so the closing step never carries
an in-flight sub-call result. `teacher/stats/` contains measured
trajectory-length tables per task; training uses them only to size the
cosine LR schedule.

Trajectories are generated on the fly on CPU (`supervised/lib/data_gen.py`)
and streamed through a shuffle buffer — no dataset is ever materialised.

Each rule is documented in detail in `teacher/rules/<task>.md`, including
executed small-digit example trajectories rendered step by step; start
with `teacher/rules/README.md` for the tape / control-bit conventions and
the trace format.

## Ablation study

`ablation/` reproduces the paper's ablation experiments: six settings that
each remove one component — the back-direction sign flip (`pe_noinv`), the
T-axis positional signal (`pe_nope`), the whole GLPE-v1 encoding
(`pe_rope`), control element (b) (`ctl_no_b`), control element (c)
(`ctl_no_c`), and the done-stage auto-eq cleanup (`edit_no_cleanup`) — and
retrain/evaluate the copy, addition, and multiplication stages under an
otherwise-identical protocol, chained per arm in the default curriculum
order (copy from scratch → add from the arm's copy → mult from the arm's
add). The package is self-contained (it imports the release code, never
modifies it); only the `edit_no_cleanup` arm borrows this curriculum's
`checkpoints/copy.pt` (its own copy stage would be bit-identical to the
baseline), and `--arm baseline` evaluations use this curriculum's
checkpoints. One command per arm:

```bash
CUDA_VISIBLE_DEVICES=0 bash ablation/scripts/train_arm.sh pe_noinv
bash ablation/scripts/eval_arm.sh pe_noinv 0,1,2
```

See `ablation/README.md` for the arm definitions, protocol, and the
verification gates.

## Citation

```bibtex
@inproceedings{rdlm2026,
  title     = {TODO — paper title},
  author    = {TODO},
  booktitle = {Proceedings of EMNLP},
  year      = {2026}
}
```
