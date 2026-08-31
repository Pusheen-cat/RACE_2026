# Sequential rule (`sequential_task.py`)

**Task.** A flat multi-operator expression, e.g. `2+3*4` → `14`. Inputs
carry **no `=`** — the runner auto-appends one, so at `done=1` the auto-eq
cleanup drops the operands and only the answer remains on the tape (the
model is likewise evaluated on the post-`=` answer). The rule reduces the
expression one operator at a time, `*`/`/` before `+`/`-`, delegating each
binary reduction to the corresponding binary rule through the dispatcher.

**Dispatch condition.** The visible LHS contains only digits and operators
`+ - * /`, with at least two operators.

See [README.md](README.md) for conventions and the trace format.

## Rule

* **Rule 1 — sign-pair collapse**: consecutive `+/-` pairs on the LHS are
  normalised first — `--` → `+`, `++` → `+`, `+-`/`-+` → `-` (implemented
  as the corresponding delete/insert edits). Such pairs appear when a
  reduced segment returns a negative number.
* **Rule 2 — spurious unary minus**: a `-` directly before a `0` whose
  neighbourhood is non-numeric on both sides (a `-0` left over from a
  returning segment) is deleted.
* **Rule 3 — main reduction** (an operator "qualifies" when a digit stands
  immediately to its left, i.e. its left operand has fully returned):
  * **3-1** — an operator already tagged `(c)` exists: expand left and
    right from it up to (not including) the neighbouring qualifying
    operators, and `(a)` the whole segment — the lifted
    `number op number` sub-tape dispatches to the matching binary rule,
    and only its result returns (the sub-call's `=` is auto-appended).
  * **3-2** — no `(c)` operator, but a `*` or `/` exists:
    * one qualifying operator left and RHS empty → **3-2-1-1** `(a)` on
      every token except any `-` (final reduction of the whole LHS);
    * one qualifying operator left and RHS non-empty → **3-2-1-2**
      finalisation: retag the RHS digits `(b)`→`(c)`, insert `-` after `=`
      when the LHS had exactly one `-` and the result is non-zero, and
      `done=1`;
    * otherwise → **3-2-2** tag `(c)` on the **leftmost `*` or `/`** (the
      next segment to reduce; Rule 3-1 lifts it on the following step).
  * **3-3** — no `(c)` operator and no `*`/`/` (only `+`/`-` remain):
    * one qualifying operator and RHS empty → **3-3-1-1** flip the
      rightmost sign operator (`+`↔`-`) and `(a)` everything except the
      first token — this propagates a running sign while the final
      addition/subtraction is lifted;
    * one qualifying operator, RHS starts with a digit → **3-3-1-2**
      finalisation with the same conditional `-` insertion as 3-2-1-2;
    * one qualifying operator, RHS starts with `-` → **3-3-1-3** delete
      that `-` (double negation) and finalise;
    * otherwise → **3-3-2** tag `(c)` on the **leftmost qualifying
      operator**.

## Example — `2+3*4`

```
step  depth  tape             visible        done
  0   d=0    2+3*4=           2+3*4=         0    3-2-2: tag (c) on the "*"
  1   d=0    2+3*4=           2+3*4=         0    3-1: lift the "3*4" segment
  2   d=1    2+3*4==          3*4=           1    multiplication rule: 3*4 → 12
  3   d=0    2+12=            2+12=          0    addition rule takes over: lift ones column
  4   d=1    2+12=            2+2=           1    2+2 → 4
  5   d=0    2+12=4           2+12=4         0    lift tens digit
  6   d=1    2+12=4           1=             1    copy: 1
  7   d=0    2+12=14          2+12=14        0    addition retag
  8   d=0    2+12=14          2+12=14        1    done → auto-eq cleanup leaves "14"
```

Steps 0–2 are the sequential rule proper: precedence is enforced by
tagging the leftmost `*` (step 0) and lifting its segment (step 1); the
inner `3*4=` is the multiplication rule's memorized case, and because the
segment's `=` was auto-appended, only `12` returns, splicing the tape to
`2+12=`. From step 3 on the visible tape matches the *addition* pattern,
so the dispatcher routes every remaining step to the addition rule —
the sequential rule never runs again. At the final `done=1` the top-level
auto-appended `=` triggers the same cleanup, so the finished tape is just
the answer `14`.

Full trace:

```
step 0  max_depth=0
  d=0  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=0  "+"                     ||  "+"               |  [M]               |  [M]               |  [M]             
  d=0  "3"                     ||  "3"               |  [M]               |  [M]               |  [M]             
  d=0  "*"                     ||  "*"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=0  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  done = 0

step 1  max_depth=0
  d=0  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=0  "+"                     ||  "+"               |  [M]               |  [M]               |  [M]             
  d=0  "3"                     ||  "3"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "*"          (c)        ||  "*"    (a)   (c)  |  [M]               |  [M]               |  [M]             
  d=0  "4"                     ||  "4"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  done = 0

step 2  max_depth=1
  d=0  "2"                   
  d=0  "+"                   
  d=1  "3"                     ||  "3"               |  [M]               |  [M]               |  [M]             
  d=1  "*"                     ||  "*"               |  [M]               |  [M]               |  [M]             
  d=1  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  "1"               |  "2"               |  [M]             
  d=0  "="               rule
  done = 1

step 3  max_depth=0
  d=0  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "+"                     ||  "+"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "1"       (b)           ||  "1"       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "2"       (b)           ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "="               rule  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  done = 0

step 4  max_depth=1
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "+"                     ||  "+"               |  [M]               |  [M]               |  [M]             
  d=0  "1"                   
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  "4"               |  [M]               |  [M]             
  done = 1

step 5  max_depth=0
  d=0  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "+"       (b)           ||  "+"       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "1"                     ||  "1"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="       (b)     rule  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "4"       (b)           ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  done = 0

step 6  max_depth=1
  d=0  "2"          (c)      
  d=0  "+"                   
  d=1  "1"                     ||  "1"               |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)      
  d=1  "="               rule  ||  "="               |  "1"               |  [M]               |  [M]             
  d=0  "4"          (c)      
  done = 1

step 7  max_depth=0
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "+"                     ||  "+"               |  [M]               |  [M]               |  [M]             
  d=0  "1"       (b)           ||  "1"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="       (b)     rule  ||  "="       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "1"       (b)           ||  "1"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  done = 0

step 8  max_depth=0
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "+"                     ||  "+"               |  [M]               |  [M]               |  [M]             
  d=0  "1"          (c)        ||  "1"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=0  "1"          (c)        ||  "1"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  done = 1
```
