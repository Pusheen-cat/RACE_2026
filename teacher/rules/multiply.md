# Multiplication rule (`multiply.py`)

**Task.** `a*b=` → `a*b=<a·b>`, e.g. `12*3=` → `12*3=36`. Long
multiplication: the multiplicand is consumed digit by digit from the right,
each partial product is produced by a smaller multiplication sub-call, and
partial products are combined through addition sub-expressions written on
the RHS.

**Dispatch condition.** The visible LHS is `digits * digits` — exactly one
`*`, digits on both sides, no other operators and no `b`.

See [README.md](README.md) for conventions and the trace format.

## Rule

* **Rule 1 — memorized base case**: `d1*d2=` with single digits and empty
  RHS, all cells clean. Insert the digit(s) of `d1·d2` after `=`;
  `done=1`. (100 memorized cases: `0*0` … `9*9`.)

* **Rule 2 — otherwise** (branch order as listed; the closing check 2-2
  runs before the retag step 2-1):

  * **2-2** — RHS non-empty, no `+` on the RHS, every digit `(c)`:
    `done=1`.
  * **2-1** — RHS non-empty, no `+` on the RHS, every digit `(b)` or `(c)`
    but not all `(c)`: retag `(b)`→`(c)` and drop the consecutive leading
    zeros of the RHS number (keeping at least its rightmost digit);
    `done=0`.
  * **2-3** — RHS empty (start):
    * **2-3-1** (≥ 2 digits left of `*`): `(a)` on the rightmost digit
      left of `*`, on `*`, on every digit right of `*`, and on `=` — lift
      `d * b =`, the partial product of the lowest multiplicand digit.
    * **2-3-2** (single digit left of `*`): the same with that one digit.
  * **2-4** — RHS is digits only and at least one digit is raw: if ≥ 2 RHS
    digits carry `(b)` insert `+` after `=`; keep `(c)` on LHS `(b)`
    digits whose left neighbour is a clean digit, and `(a)` on the other
    LHS digits together with `*` and `=` — lift the next partial-product
    column; `(c)` the rightmost RHS `(b)` digit.
  * **2-5** — a `+` on the RHS: `(a)` on every RHS token without `(c)` —
    lift the pending addition; `(c)` on LHS `(b)` digits.

## Example 1 — memorized: `3*4=`

```
step 0  max_depth=0
  d=0  "3"                     ||  "3"               |  [M]               |  [M]               |  [M]             
  d=0  "*"                     ||  "*"               |  [M]               |  [M]               |  [M]             
  d=0  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="               |  "1"               |  "2"               |  [M]             
  done = 1
```

## Example 2 — two-digit multiplicand: `12*3=`

```
step  depth  tape             visible        done
  0   d=0    12*3=            12*3=          0    2-3-1: lift ones partial product
  1   d=1    12*3=            2*3=           1    Rule 1: 2*3 → 6
  2   d=0    12*3=6           12*3=6         0    2-4: (c) on "6", lift tens partial product
  3   d=1    12*3=6           1*3=           1    Rule 1: 1*3 → 3
  4   d=0    12*3=36          12*3=36        0    2-1: retag (b)→(c)
  5   d=0    12*3=36          12*3=36        1    2-2: all (c) → done
```

The ones digit's partial product `2*3 → 6` returns first and is tagged
`(c)`; the tens digit's product `1*3 → 3` returns next and lands in front
of it, assembling `36` place by place. In this example every column
product is a single digit, so no `+` ever appears; with larger digits
(e.g. `7*8`) a two-digit partial product triggers branch 2-4's `+`
insertion and branch 2-5 then lifts the resulting addition — reusing the
addition rule through the dispatcher, exactly like the carry pipeline in
[add.md](add.md).

Full trace:

```
step 0  max_depth=0
  d=0  "1"                     ||  "1"               |  [M]               |  [M]               |  [M]             
  d=0  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "*"                     ||  "*"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "3"                     ||  "3"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  done = 0

step 1  max_depth=1
  d=0  "1"                   
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "*"                     ||  "*"               |  [M]               |  [M]               |  [M]             
  d=1  "3"                     ||  "3"               |  [M]               |  [M]               |  [M]             
  d=1  "="               orig  ||  "="               |  "6"               |  [M]               |  [M]             
  done = 1

step 2  max_depth=0
  d=0  "1"                     ||  "1"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "*"       (b)           ||  "*"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "3"       (b)           ||  "3"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "="       (b)     orig  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "6"       (b)           ||  "6"          (c)  |  [M]               |  [M]               |  [M]             
  done = 0

step 3  max_depth=1
  d=1  "1"                     ||  "1"               |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)      
  d=1  "*"                     ||  "*"               |  [M]               |  [M]               |  [M]             
  d=1  "3"                     ||  "3"               |  [M]               |  [M]               |  [M]             
  d=1  "="               orig  ||  "="               |  "3"               |  [M]               |  [M]             
  d=0  "6"          (c)      
  done = 1

step 4  max_depth=0
  d=0  "1"       (b)           ||  "1"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "*"       (b)           ||  "*"       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "3"       (b)           ||  "3"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="       (b)     orig  ||  "="       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "3"       (b)           ||  "3"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "6"          (c)        ||  "6"          (c)  |  [M]               |  [M]               |  [M]             
  done = 0

step 5  max_depth=0
  d=0  "1"          (c)        ||  "1"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "*"                     ||  "*"               |  [M]               |  [M]               |  [M]             
  d=0  "3"          (c)        ||  "3"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="               |  [M]               |  [M]               |  [M]             
  d=0  "3"          (c)        ||  "3"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "6"          (c)        ||  "6"          (c)  |  [M]               |  [M]               |  [M]             
  done = 1
```
