# Teacher rules

Every task is solved by a hand-written **teacher rule**: a deterministic
function of the currently *visible* tape that emits one edit action per
step. Running a rule to completion produces the teacher trajectory that the
model imitates — the model learns exactly this action space, one step at a
time.

This directory contains one Python module and one companion `.md` document
per task:

| Task | Rule module | Document | Dispatched when the visible LHS … |
|------|-------------|----------|-----------------------------------|
| copy | `copy.py` | [copy.md](copy.md) | is digits only |
| addition | `add.py` | [add.md](add.md) | is `digits + digits` (one `+`) |
| subtraction | `subtract.py` | [subtract.md](subtract.md) | has one `-` and no other op, **or** contains the borrow letter `b` |
| multiplication | `multiply.py` | [multiply.md](multiply.md) | is `digits * digits` (one `*`) |
| division | `division.py` | [division.md](division.md) | contains `/`, `%`, or `_` (digits otherwise) |
| sequential | `sequential_task.py` | [sequential_task.md](sequential_task.md) | has ≥ 2 operators, digits/ops only |
| nested | `sequential_paren_task.py` | [sequential_paren_task.md](sequential_paren_task.md) | contains balanced `(` `)` |

`../dispatcher.py` re-evaluates this routing **every step** on the current
visible tape (checking the patterns in the table's order, top to bottom).
That is what makes the rules compositional: when the sequential rule lifts
`3*4` into a sub-call, the inner tape `3*4=` simply dispatches to the
multiplication rule; when a multi-digit addition reduces to a bare number,
the copy rule finishes it. No rule ever calls another rule directly.

## The tape, control bits, and self-calls

The state (`../state.py`) is a flat tape. Every token carries:

* a **vocab id** — digits `0-9`, operators `= + - * /`, `.`, `(`, `)`, and
  two scratchpad symbols used only inside division trajectories: `%`
  (remainder marker) and `_` (operand bracket), plus the borrow letter `b`
  used inside subtraction;
* three **control bits**, written `(a)`, `(b)`, `(c)`:

| Bit | Name | Who writes it | Meaning |
|-----|------|---------------|---------|
| `(a)` | self-call | the rule | The engine moves this token one depth level down (`depth+1`) and clears all its bits. All `(a)`-tagged tokens of one step become the input of a recursive sub-call. |
| `(b)` | result | the engine only | Set on every visible token at the moment a sub-call returns (`done=1` at depth > 0). It is ephemeral: the engine forces `(b)=0` in every incoming action, so a rule that wants to remember "this token just returned" must retag it to `(c)` in the very next step. |
| `(c)` | tag | the rule | Free persistent marker. The rules use it to mean "this token is finished / verified". |

* a **depth**. Tokens at the current maximum depth are the *visible* tape —
  the only tokens a rule (or the model) sees and edits. Lower-depth tokens
  are frozen until the sub-call above them returns.

**Actions** (`../action.py`) have shape `(S_visible + 1, 4, 1 + 3)`: for
each visible position, slot 0 replaces the token (writing MASK deletes it)
and slots 1–3 insert new tokens after it; the extra last row carries the
**done** indicator. `done=1` at depth 0 ends the trajectory; `done=1` at
depth d > 0 returns the visible tokens to depth d−1 with `(b)` set.
Emitting `done=1` and an `(a)` bit in the same step is invalid and aborts
the trajectory.

**Auto-appended `=`** (`../runner.py`): whenever the visible tape has no
`=`, the runner appends one transparently before the rule fires — this is a
state mutation, not a trajectory step. Its origin matters at return time:

* if the `=` was **auto-appended** for this sub-call, `done=1` keeps only
  the tokens *after* it (operands and the `=` itself are dropped) — the
  sub-call returns just its result;
* if the `=` was already present (original input, or lifted from the outer
  tape via `(a)`), everything returns as-is.

All rules share a **safe done timing** discipline: `done=1` is only emitted
once every digit on the tape carries the persistent `(c)` tag, and the
retag step `(b)→(c)` is never combined with the closing step — so a freshly
returned sub-call result can never ride along unverified on the final
action.

## How to read the execution traces

Each task document ends with real executed trajectories, rendered by
`../visualize.py` (`format_trajectory`). One step looks like:

```
step 2  max_depth=0
  d=0  "1"                     ||  "1"    (a)        |  [M]               |  [M]               |  [M]
  d=0  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]
  d=0  "="       (b)     orig  ||  "="    (a)        |  [M]               |  [M]               |  [M]
  done = 0
```

* One row per tape token, in tape order. `d=<n>` is the token's depth;
  rows whose depth is below the current maximum are frozen and shown
  without action cells.
* The first cell is the token (`"1"`, `[M]` = MASK) followed by its current
  control bits — `(a)`, `(b)`, `(c)` each keep a fixed column, blank when
  unset.
* An `=` token is annotated `orig` (came with the input, or was placed by a
  rule action) or `rule` (auto-appended by the runner — dropped again at
  return time, see above).
* After `||` come the four action cells for that position: slot 0 is the
  replacement (token + new control bits), slots 1–3 are insertions after
  the position; `[M]` means delete (slot 0) / no insertion (slots 1–3).
* `done = 0|1` is the step's done indicator.

So in the row `"2" (b)  ||  "2" (c)`, the token `2` currently carries `(b)`
and the action rewrites it with `(c)` — the standard retag. In the row
`"=" (b) orig  ||  "=" (a)`, the `=` is tagged `(a)`: it will be lifted
into the next sub-call.

For longer trajectories the documents additionally show a compact per-step
summary — full tape, visible slice, and done flag on one line per step —
before the full row-level trace.
