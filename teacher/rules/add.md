# Addition rule (`add.py`)

**Task.** `a+b=` → `a+b=<a+b>`, e.g. `27+58=` → `27+58=85`. The rule
performs column-wise addition from the least significant digit, delegating
each single-digit column sum to a memorized sub-call and threading carries
through an explicit `+` written on the RHS.

**Dispatch condition.** The visible LHS is `digits + digits` — exactly one
`+`, digits on both sides.

See [README.md](README.md) for conventions and the trace format.

## Rule

* **Rule 1 — memorized base case**: `a+b=` with single-digit `a`, `b` and
  empty RHS. Insert the digit(s) of `a+b` after `=`; `done=1`.
  (100 memorized cases: `0+0` … `9+9`; two digits are inserted when the
  sum carries, e.g. `7+8=` → `15`.)

* **Rule 2 — otherwise:**

  * **2-3** — RHS empty (start): `(a)` on the `+` and the token
    immediately left of it, and on `=` and the token immediately left of
    it — i.e. lift the ones column `d1+d2=` as a sub-call.
  * **2-2** — a `+` exists on the RHS (a pending carry sub-expression):
    `(a)` on the RHS `+` and every RHS digit without `(c)` — lift the
    carry sum; `(c)` on every LHS digit with `(b)`.
  * **2-1** — RHS has tokens but no `+`:
    * **2-1-1** — every digit (LHS and RHS) has `(b)` or `(c)`:
      * **2-1-1-2** — all `(c)`: `done=1`.
      * **2-1-1-1** — not all `(c)`: retag `(b)`→`(c)`; `done=0`.
    * **2-1-2** — some digit is still raw (no control bit):
      * **2-1-2-1** — the RHS carries 0 or ≥ 2 `(b)` digits (start of the
        next column, or a column that just produced a carry): on each side
        of the LHS `+`, `(a)` the rightmost raw digit (skip a side with
        none); `(a)` on `=` (and on the LHS `+` if both sides
        contributed); retag LHS `(b)` digits to `(c)`. In the carry case
        (≥ 2 RHS `(b)` digits) additionally `(c)` the rightmost RHS `(b)`
        digit and **insert `+` right after `=`** — the returned two-digit
        column sum `15` becomes the pending expression `+1·5`, whose `1`
        must still be added to the next column.
      * **2-1-2-2** — exactly one RHS `(b)` digit (previous column closed
        without carry): `(a)` on every raw LHS digit and on `=` (and the
        LHS `+` when both sides have one); retag all `(b)` digits to
        `(c)`.

## Example 1 — memorized: `3+4=`

```
step 0  max_depth=0
  d=0  "3"                     ||  "3"               |  [M]               |  [M]               |  [M]             
  d=0  "+"                     ||  "+"               |  [M]               |  [M]               |  [M]             
  d=0  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="               |  "7"               |  [M]               |  [M]             
  done = 1
```

## Example 2 — carry: `27+58=`

```
step  depth  tape             visible        done
  0   d=0    27+58=           27+58=         0    2-3: lift the ones column
  1   d=1    27+58=           7+8=           1    Rule 1: 7+8 → 15 (carry!)
  2   d=0    27+58=15         27+58=15       0    2-1-2-1 carry case: insert "+", (c) on "5", lift tens column
  3   d=1    27+58=+15        2+5=           1    Rule 1: 2+5 → 7
  4   d=0    27+58=7+15       27+58=7+15     0    2-2: lift the carry sum "7+1"
  5   d=1    27+58=7+1=5      7+1=           1    Rule 1: 7+1 → 8
  6   d=0    27+58=85         27+58=85       0    2-1-1-1: retag (b)→(c)
  7   d=0    27+58=85         27+58=85       1    2-1-1-2: all (c) → done
```

Step 1's column sum returns **two** `(b)` digits (`15`), which is how the
rule detects a carry: step 2 keeps the ones digit `5` (tagged `(c)`),
inserts a `+` after `=`, and lifts the tens column. After step 3 the RHS
reads `7+15`; step 4 lifts `7+1` — tens digit plus carry — as its own
addition sub-call (the auto-appended `=` of that sub-call means only its
result `8` returns, replacing the lifted tokens). The final tape `85`
assembles from the carry-resolved tens digit and the kept ones digit.

Full trace:

```
step 0  max_depth=0
  d=0  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=0  "7"                     ||  "7"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "+"                     ||  "+"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "5"                     ||  "5"               |  [M]               |  [M]               |  [M]             
  d=0  "8"                     ||  "8"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  done = 0

step 1  max_depth=1
  d=0  "2"                   
  d=1  "7"                     ||  "7"               |  [M]               |  [M]               |  [M]             
  d=1  "+"                     ||  "+"               |  [M]               |  [M]               |  [M]             
  d=0  "5"                   
  d=1  "8"                     ||  "8"               |  [M]               |  [M]               |  [M]             
  d=1  "="               orig  ||  "="               |  "1"               |  "5"               |  [M]             
  done = 1

step 2  max_depth=0
  d=0  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "7"       (b)           ||  "7"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "+"       (b)           ||  "+"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "5"                     ||  "5"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "8"       (b)           ||  "8"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="       (b)     orig  ||  "="    (a)        |  "+"               |  [M]               |  [M]             
  d=0  "1"       (b)           ||  "1"       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "5"       (b)           ||  "5"          (c)  |  [M]               |  [M]               |  [M]             
  done = 0

step 3  max_depth=1
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=0  "7"          (c)      
  d=1  "+"                     ||  "+"               |  [M]               |  [M]               |  [M]             
  d=1  "5"                     ||  "5"               |  [M]               |  [M]               |  [M]             
  d=0  "8"          (c)      
  d=1  "="               orig  ||  "="               |  "7"               |  [M]               |  [M]             
  d=0  "+"                   
  d=0  "1"                   
  d=0  "5"          (c)      
  done = 1

step 4  max_depth=0
  d=0  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "7"          (c)        ||  "7"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "+"       (b)           ||  "+"       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "5"       (b)           ||  "5"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "8"          (c)        ||  "8"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="       (b)     orig  ||  "="       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "7"       (b)           ||  "7"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "+"                     ||  "+"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "1"                     ||  "1"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "5"          (c)        ||  "5"          (c)  |  [M]               |  [M]               |  [M]             
  done = 0

step 5  max_depth=1
  d=0  "2"          (c)      
  d=0  "7"          (c)      
  d=0  "+"                   
  d=0  "5"          (c)      
  d=0  "8"          (c)      
  d=0  "="               orig
  d=1  "7"                     ||  "7"               |  [M]               |  [M]               |  [M]             
  d=1  "+"                     ||  "+"               |  [M]               |  [M]               |  [M]             
  d=1  "1"                     ||  "1"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  "8"               |  [M]               |  [M]             
  d=0  "5"          (c)      
  done = 1

step 6  max_depth=0
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "7"          (c)        ||  "7"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "+"                     ||  "+"               |  [M]               |  [M]               |  [M]             
  d=0  "5"          (c)        ||  "5"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "8"          (c)        ||  "8"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="               |  [M]               |  [M]               |  [M]             
  d=0  "8"       (b)           ||  "8"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "5"          (c)        ||  "5"          (c)  |  [M]               |  [M]               |  [M]             
  done = 0

step 7  max_depth=0
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "7"          (c)        ||  "7"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "+"                     ||  "+"               |  [M]               |  [M]               |  [M]             
  d=0  "5"          (c)        ||  "5"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "8"          (c)        ||  "8"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="               |  [M]               |  [M]               |  [M]             
  d=0  "8"          (c)        ||  "8"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "5"          (c)        ||  "5"          (c)  |  [M]               |  [M]               |  [M]             
  done = 1
```
