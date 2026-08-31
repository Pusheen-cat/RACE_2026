# Copy rule (`copy.py`)

**Task.** Reproduce a number after the equals sign: `1234=` → `1234=1234`.
Copy is stage 1 of the curriculum and also the base case every other rule
falls back on — any sub-call whose visible tape is a bare number (e.g. a
lifted digit) is routed here by the dispatcher.

**Dispatch condition.** The visible LHS of `=` consists of digit tokens
only (control bits are ignored by the dispatcher, so a mid-trajectory state
like `1 2(b) 3(b) =` still routes here).

See [README.md](README.md) for the tape / control-bit conventions and how
to read the traces below.

## Rule

* **Rule 1 — memorized base case** (RHS empty **and** LHS has ≤ 3 digits):
  insert the LHS digits one by one after `=` and emit `done=1`.
  Example: `34=` → `34=34` in a single step.

* **Rule 2 — otherwise** (RHS non-empty or LHS longer than 3 digits):

  * **2-1** — some LHS digit carries no control bit at all:
    set `(a)` on the rightmost up-to-3 such digits and on `=`; in the same
    step, retag any digit carrying `(b)` to `(c)`. `done=0` — the `(a)`
    tokens become a ≤3-digit copy sub-call that Rule 1 memorizes.
  * **2-2** — every LHS digit has some control bit, but not all have `(c)`:
    retag every `(b)` digit to `(c)`. `done=0`.
  * **2-3** — every LHS digit already has `(c)`: `done=1`.

The 2-2 / 2-3 split is the safe done timing: the rule never emits `done=1`
on the same step it converts `(b)` to `(c)`.

## Example 1 — memorized: `34=`

One step (Rule 1): both digits are inserted after `=` and the trajectory
terminates.

```
step 0  max_depth=0
  d=0  "3"                     ||  "3"               |  [M]               |  [M]               |  [M]             
  d=0  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="               |  "3"               |  "4"               |  [M]             
  done = 1
```

## Example 2 — recursive: `1234=`

Four digits exceed the memorization width, so the rule copies in chunks of
three through self-calls:

```
step  depth  tape             visible        done
  0   d=0    1234=            1234=          0    2-1: (a) on "234" and "="
  1   d=1    1234=            234=           1    Rule 1: memorized copy of the chunk
  2   d=0    1234=234         1234=234       0    2-1: retag (b)→(c); (a) on "1" and "="
  3   d=1    1234=234         1=             1    Rule 1: memorized copy
  4   d=0    1234=1234        1234=1234      0    2-2: retag remaining (b)→(c)
  5   d=0    1234=1234        1234=1234      1    2-3: everything (c) → done
```

Step 0 lifts the rightmost three raw digits plus the original `=` into a
sub-call; since the lifted `=` is the *original* one, the sub-call's whole
tape (chunk + `=` + copied digits) returns to depth 0, where the copies
land after `=` carrying `(b)`. Step 2 retags them and lifts the remaining
digit `1` the same way — the returned `1` is inserted directly after `=`,
in front of the earlier chunk, so the copy assembles right-to-left in
chunks. Two bookkeeping steps close the trajectory.

Full trace:

```
step 0  max_depth=0
  d=0  "1"                     ||  "1"               |  [M]               |  [M]               |  [M]             
  d=0  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "3"                     ||  "3"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "4"                     ||  "4"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  done = 0

step 1  max_depth=1
  d=0  "1"                   
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "3"                     ||  "3"               |  [M]               |  [M]               |  [M]             
  d=1  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=1  "="               orig  ||  "="               |  "2"               |  "3"               |  "4"             
  done = 1

step 2  max_depth=0
  d=0  "1"                     ||  "1"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "3"       (b)           ||  "3"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "4"       (b)           ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="       (b)     orig  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "3"       (b)           ||  "3"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "4"       (b)           ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  done = 0

step 3  max_depth=1
  d=1  "1"                     ||  "1"               |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)      
  d=0  "3"          (c)      
  d=0  "4"          (c)      
  d=1  "="               orig  ||  "="               |  "1"               |  [M]               |  [M]             
  d=0  "2"          (c)      
  d=0  "3"          (c)      
  d=0  "4"          (c)      
  done = 1

step 4  max_depth=0
  d=0  "1"       (b)           ||  "1"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "3"          (c)        ||  "3"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="       (b)     orig  ||  "="       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "1"       (b)           ||  "1"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "3"          (c)        ||  "3"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  done = 0

step 5  max_depth=0
  d=0  "1"          (c)        ||  "1"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "3"          (c)        ||  "3"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="               |  [M]               |  [M]               |  [M]             
  d=0  "1"          (c)        ||  "1"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "3"          (c)        ||  "3"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  done = 1
```
