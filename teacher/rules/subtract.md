# Subtraction rule (`subtract.py`)

**Task.** `a-b=` → `a-b=<a-b>`, e.g. `52-8=` → `52-8=44` (results may be
negative: `3-7=` → `3-7=-4`). Column-wise subtraction from the least
significant digit; a column that goes negative is rewritten with the
literal borrow letter **`b`**, which later sub-calls consume.

**Dispatch condition.** Either the visible LHS has exactly one `-` and no
other operator and no `b` (case A — a plain subtraction), or it contains at
least one `b` letter with no `-` / `*` / `/` (case B — a borrow-resolution
tape; a `+` may coexist).

See [README.md](README.md) for conventions and the trace format.

## Rule

* **Rule 1 — memorized base cases (1 step each):**
  * **1-1** `d1-d2=` — single-digit subtraction. A negative result is
    written with a leading `-` (e.g. `2-8=` → `-6`).
  * **1-2** `b<n>=` — a `b` followed by 1–2 digits: resolve the borrow
    against `<n>` (subtract 1).
  * **1-3** `b<n>=-` — the same with a pending `-` on the RHS.
  * **1-4** `-d=` — a negative single digit: rewrite as borrow form
    `b<10-d>` (e.g. `-6=` → `b4`).
  * **1-5** `d+b=` — a digit plus a borrow: resolve to `d-1`.

* **Rule 2 — the LHS starts with `b`** (borrow-resolution tape):
  * **2-1** — no RHS: clear stale bits, `(a)` on the leftmost `b`, on `=`,
    and on the two tokens left of `=`.
  * **2-2** — RHS non-empty: **2-2-A** some LHS digit raw → `(a)` on the
    rightmost up-to-2 raw LHS digits, the leftmost `b`, `=`, and a `-`
    right of `=` if present; retag other `(b)` digits to `(c)`.
    **2-2-B** no raw LHS digit but some `(b)` remains → retag `(b)`→`(c)`.
    **2-2-C** everything `(c)` → `done=1`.

* **Rule 3 — general** (plain `a-b=` tape):
  * **3-1 / 3-2** — closing steps: when every digit is verified, retag
    `(b)`→`(c)` and drop leading zeros of the result (3-1), then `done=1`
    (3-2).
  * **3-3** — a `-` sits immediately right of `=` (a negative column
    result just returned): `(a)` on that `-` and the next token — this
    lifts `-d=`, whose memorized answer converts it to borrow form.
  * **3-4** — a `+` on the RHS: `(a)` on the `+` and its two neighbours
    (lifts `d+b=`, the borrow resolution).
  * **3-5** — otherwise, drive the column loop:
    * **3-5-3** RHS empty (start): lift the ones column `d1-d2=`.
    * **3-5-1** a `b` sits right of `=`: if raw LHS digits remain, insert
      `+` after `=` and lift the next column digit(s) (**3-5-1-2**);
      otherwise retag and lift the whole RHS (**3-5-1-1**).
    * **3-5-2** no `b` right of `=`: same two branches without the `+`
      insertion (**3-5-2-2** / **3-5-2-1**).

## Example 1 — memorized: `7-3=`

```
step 0  max_depth=0
  d=0  "7"                     ||  "7"               |  [M]               |  [M]               |  [M]             
  d=0  "-"                     ||  "-"               |  [M]               |  [M]               |  [M]             
  d=0  "3"                     ||  "3"               |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="               |  "4"               |  [M]               |  [M]             
  done = 1
```

## Example 2 — borrow: `52-8=`

```
step  depth  tape             visible        done
  0   d=0    52-8=            52-8=          0    3-5-3: lift the ones column
  1   d=1    52-8=            2-8=           1    1-1: 2-8 → -6 (negative!)
  2   d=0    52-8=-6          52-8=-6        0    3-3: lift the "-6"
  3   d=1    52-8=-6=         -6=            1    1-4: -6 → borrow form b4
  4   d=0    52-8=b4          52-8=b4        0    3-5-1-2: insert "+", lift next column digit "5"
  5   d=1    52-8=+b4         5=             1    copy rule: 5 → 5
  6   d=0    52-8=5+b4        52-8=5+b4      0    3-4: lift "5+b"
  7   d=1    52-8=5+b=4       5+b=           1    1-5: 5+b → 4 (borrow consumed)
  8   d=0    52-8=44          52-8=44        0    3-5-2-1: retag, lift the RHS for verification
  9   d=1    52-8=44=         44=            1    copy rule: verified copy of "44"
 10   d=0    52-8=44          52-8=44        0    3-1: retag (b)→(c)
 11   d=0    52-8=44          52-8=44        1    3-2: all (c) → done
```

The borrow pipeline is fully explicit on the tape: the ones column
`2-8` returns `-6` (step 1); the memorized `-6=` sub-call rewrites it as
`b4` — "digit 4, borrow one" (step 3); the next column digit `5` is
fetched by a copy sub-call (step 5); and `5+b=` resolves the borrow to `4`
(step 7), leaving the result `44`. Steps 8–9 re-lift the finished result
through a copy sub-call whose auto-appended `=` drops the operands on
return, so the tape is unchanged but every result digit now carries the
engine's `(b)` mark; two closing steps convert those to `(c)` and
terminate. Sub-calls in steps 5 and 9 are handled by the **copy rule** —
the dispatcher routes each visible tape independently.

Full trace:

```
step 0  max_depth=0
  d=0  "5"                     ||  "5"               |  [M]               |  [M]               |  [M]             
  d=0  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "-"                     ||  "-"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "8"                     ||  "8"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  done = 0

step 1  max_depth=1
  d=0  "5"                   
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "-"                     ||  "-"               |  [M]               |  [M]               |  [M]             
  d=1  "8"                     ||  "8"               |  [M]               |  [M]               |  [M]             
  d=1  "="               orig  ||  "="               |  "-"               |  "6"               |  [M]             
  done = 1

step 2  max_depth=0
  d=0  "5"                     ||  "5"               |  [M]               |  [M]               |  [M]             
  d=0  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "-"       (b)           ||  "-"       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "8"       (b)           ||  "8"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="       (b)     orig  ||  "="       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "-"       (b)           ||  "-"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "6"       (b)           ||  "6"    (a)        |  [M]               |  [M]               |  [M]             
  done = 0

step 3  max_depth=1
  d=0  "5"                   
  d=0  "2"          (c)      
  d=0  "-"                   
  d=0  "8"          (c)      
  d=0  "="               orig
  d=1  "-"                     ||  "-"               |  [M]               |  [M]               |  [M]             
  d=1  "6"                     ||  "6"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  "b"               |  "4"               |  [M]             
  done = 1

step 4  max_depth=0
  d=0  "5"                     ||  "5"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "-"                     ||  "-"               |  [M]               |  [M]               |  [M]             
  d=0  "8"          (c)        ||  "8"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="    (a)        |  "+"               |  [M]               |  [M]             
  d=0  "b"       (b)           ||  "b"       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "4"       (b)           ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  done = 0

step 5  max_depth=1
  d=1  "5"                     ||  "5"               |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)      
  d=0  "-"                   
  d=0  "8"          (c)      
  d=1  "="               orig  ||  "="               |  "5"               |  [M]               |  [M]             
  d=0  "+"                   
  d=0  "b"                   
  d=0  "4"          (c)      
  done = 1

step 6  max_depth=0
  d=0  "5"       (b)           ||  "5"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "-"                     ||  "-"               |  [M]               |  [M]               |  [M]             
  d=0  "8"          (c)        ||  "8"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="       (b)     orig  ||  "="       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "5"       (b)           ||  "5"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "+"                     ||  "+"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "b"                     ||  "b"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  done = 0

step 7  max_depth=1
  d=0  "5"          (c)      
  d=0  "2"          (c)      
  d=0  "-"                   
  d=0  "8"          (c)      
  d=0  "="               orig
  d=1  "5"                     ||  "5"               |  [M]               |  [M]               |  [M]             
  d=1  "+"                     ||  "+"               |  [M]               |  [M]               |  [M]             
  d=1  "b"                     ||  "b"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  "4"               |  [M]               |  [M]             
  d=0  "4"          (c)      
  done = 1

step 8  max_depth=0
  d=0  "5"          (c)        ||  "5"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "-"                     ||  "-"               |  [M]               |  [M]               |  [M]             
  d=0  "8"          (c)        ||  "8"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="               |  [M]               |  [M]               |  [M]             
  d=0  "4"       (b)           ||  "4"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "4"          (c)        ||  "4"    (a)        |  [M]               |  [M]               |  [M]             
  done = 0

step 9  max_depth=1
  d=0  "5"          (c)      
  d=0  "2"          (c)      
  d=0  "-"                   
  d=0  "8"          (c)      
  d=0  "="               orig
  d=1  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=1  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  "4"               |  "4"               |  [M]             
  done = 1

step 10  max_depth=0
  d=0  "5"          (c)        ||  "5"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "-"                     ||  "-"               |  [M]               |  [M]               |  [M]             
  d=0  "8"          (c)        ||  "8"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="               |  [M]               |  [M]               |  [M]             
  d=0  "4"       (b)           ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "4"       (b)           ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  done = 0

step 11  max_depth=0
  d=0  "5"          (c)        ||  "5"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "-"                     ||  "-"               |  [M]               |  [M]               |  [M]             
  d=0  "8"          (c)        ||  "8"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="               |  [M]               |  [M]               |  [M]             
  d=0  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  done = 1
```
