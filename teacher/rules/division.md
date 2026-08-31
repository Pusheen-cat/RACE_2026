# Division rule (`division.py`)

**Task.** `a/b=` → `a/b=<a÷b>`; an exact quotient is written as an integer
(`4/2=` → `4/2=2`), otherwise the result carries exactly three decimal
digits, truncated (`7/2=` → `7/2=3.500`, `1/8=` → `1/8=0.125`). Division
is by far the longest trajectory family: it computes the quotient by
**repeated subtraction**, then continues the same machinery on the
remainder for the three decimal places.

**Dispatch condition.** The visible LHS contains at least one of `/`, `%`,
`_` and is otherwise digits only (no `+`, `-`, `*`, `b`). `%` and `_` are
scratchpad symbols that exist *only inside* the trajectory:

* `%` — remainder marker: `x%y` means "x modulo-divide by y";
* `_` — operand bracket: `q_r` pairs the quotient-so-far `q` with the
  running remainder `r`.

Neither may survive to the final tape.

**Note on evaluation.** The 3-decimal format can only express results
≥ 0.01, so the evaluation set considers exactly the cases satisfying
`a/b ≥ 0.01` (see `supervised/lib/eval_grids.py`).

See [README.md](README.md) for conventions and the trace format.

## Rule (top-level routing)

The rule routes on the *shape* of the visible tape:

* **Section 1 — memorized shapes** that emit `done=1` immediately:
  `d%=%` (1-1), `_=` (1-2), `d_=` (1-3), a tape ending in `/=_%` (1-4) or
  `/=.%` (1-5). These are the terminal cleanup patterns of the loops
  below.
* **Section 2 — transition** `d%MULTI=` → `_d%MULTI=`: bracket the tape
  with a fresh `_` to enter the repeated-subtraction loop (`done=0`).
* **Section 3 — visible LHS starts with `_`** — the repeated-subtraction
  loop itself. Its sub-branches (3-1-x on the first `=`, 3-2-x on the
  second `=`) fetch the divisor, run one `remainder − divisor` step via a
  subtraction sub-call, increment the quotient counter via an addition
  sub-call, and detect the stopping point (remainder smaller than the
  divisor). The loop's state is the `q_r` pair on the RHS.
* **Section 4 — visible LHS contains `%` (no `_`, no `/`)** — the modulo
  layer: `x%y=` resolves to the bracketed loop of Section 3 and finally to
  the pair `q_r`. Closing branches: 4-5 (all digits `(c)` → `done=1`)
  fires before the retag step 4-4 (`(b)`→`(c)`, `done=0`); 4-1/4-2/4-3
  drive the lifting of sub-tapes.
* **Section 5 — visible LHS contains `/`** — the outermost layer. It first
  copies the divisor to the RHS, rewrites the RHS into the modulo form
  `a%b`, lifts it as a Section-4 sub-call, and finally post-processes the
  returned `q_r%b`: remainder 0 → keep the integer quotient (5-4-1);
  remainder ≠ 0 → append `.` and continue the same machinery for the three
  decimal digits (5-5-x). 5-6 closes when every digit is `(c)`.

## Example — exact quotient: `4/2=`

42 steps. Compact summary (full row-level trace below):

```
step  depth  tape                    visible          done
  0   d=0    4/2=                    4/2=             0    copy the divisor to the RHS
  1   d=1    4/2=                    2=               1
  2   d=0    4/2=2                   4/2=2            0    rewrite RHS into modulo form "4%2"
  3   d=1    4/2=%2                  4=               1
  4   d=0    4/2=4%2                 4/2=4%2          0    lift the modulo sub-tape "4%2"
  5   d=1    4/2=4%2=                4%2=             0    Section 2: bracket → "_4%2="
  6   d=1    4/2=_4%2=               _4%2=            0    Section 3 loop begins
  7   d=2    4/2=_4%2=               4=               1      fetch the dividend
  8   d=1    4/2=_4%2=4              _4%2=4           0
  9   d=2    4/2=_4%2=4              _=               1      initialise the counter: q=0
 10   d=1    4/2=_4%2=0_4            _4%2=0_4         0    loop state q_r = 0_4
 11   d=2    4/2=_4%2=0_4=           2=               1      fetch the divisor
 12   d=1    4/2=_4%2=0_4=2          _4%2=0_4=2       0
 13   d=2    4/2=_4%2=0_4=-2         4=               1      fetch the remainder
 14   d=1    4/2=_4%2=0_4=4-2        _4%2=0_4=4-2     0
 15   d=2    4/2=_4%2=0_4=4-2        0_=              1      increment the counter: 0 → 1
 16   d=1    4/2=_4%2=0_4=1_4-2      _4%2=0_4=1_4-2   0
 17   d=2    4/2=_4%2=0_4=1_4-2=     4-2=             1      subtract: 4-2 → 2
 18   d=1    4/2=_4%2=0_4=1_2        _4%2=0_4=1_2     0
 19   d=1    4/2=_4%2=1_2            _4%2=1_2         0    loop state q_r = 1_2
 20…27                                                     same round again: 2-2 → 0, q → 2
 28   d=1    4/2=_4%2=2_0            _4%2=2_0         0    loop state q_r = 2_0
 29…37                                                     stopping round: 0-2 goes negative
 38   d=1    4/2=_4%2=2_0=2          _4%2=2_0=2       0    stop: unbracket (drop "_", "=" → "%")
 39   d=1    4/2=4%2=2_0%2           4%2=2_0%2        1    return "2_0%2" to the outer tape
 40   d=0    4/2=2_0%2               4/2=2_0%2        0    remainder 0 → keep integer quotient
 41   d=0    4/2=2                   4/2=2            1    final tape
```

Reading the loop state: after each round the RHS pair `q_r` holds the
quotient counter and the running remainder — `0_4` → `1_2` → `2_0`. Each
round performs one `r − divisor` subtraction through a **subtraction-rule
sub-call** (steps 13–18) and one `+1` counter increment through an
**addition/copy sub-call** (step 15), all routed by the dispatcher. The
stopping round (29–37) detects that the remainder went below the divisor,
after which the bracket `_` is removed, the pair `2_0%2` returns through
the modulo layer, and the outermost step keeps the quotient `2` because
the remainder is `0` — a non-zero remainder would instead append `.` and
rerun the same loop three more times for the decimal digits (which is why
e.g. `5/4=` takes ~200 steps).

Full trace:

```
step 0  max_depth=0
  d=0  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=0  "/"                     ||  "/"               |  [M]               |  [M]               |  [M]             
  d=0  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  done = 0

step 1  max_depth=1
  d=0  "4"                   
  d=0  "/"                   
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               orig  ||  "="               |  "2"               |  [M]               |  [M]             
  done = 1

step 2  max_depth=0
  d=0  "4"                     ||  "4"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "/"                     ||  "/"               |  [M]               |  [M]               |  [M]             
  d=0  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="       (b)     orig  ||  "="    (a)        |  "%"               |  [M]               |  [M]             
  d=0  "2"       (b)           ||  "2"       (b)     |  [M]               |  [M]               |  [M]             
  done = 0

step 3  max_depth=1
  d=1  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=0  "/"                   
  d=0  "2"          (c)      
  d=1  "="               orig  ||  "="               |  "4"               |  [M]               |  [M]             
  d=0  "%"                   
  d=0  "2"                   
  done = 1

step 4  max_depth=0
  d=0  "4"       (b)           ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "/"                     ||  "/"               |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="       (b)     orig  ||  "="       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "4"       (b)           ||  "4"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "%"                     ||  "%"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  done = 0

step 5  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "4"                     ||  "_"               |  "4"               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  done = 0

step 6  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"                     ||  "4"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  done = 0

step 7  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                   
  d=2  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=1  "%"                   
  d=1  "2"                   
  d=2  "="               rule  ||  "="               |  "4"               |  [M]               |  [M]             
  done = 1

step 8  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "4"       (b)           ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="       (b)     rule  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "4"       (b)           ||  "4"       (b)     |  [M]               |  [M]               |  [M]             
  done = 0

step 9  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=2  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)      
  d=1  "%"                   
  d=1  "2"                   
  d=2  "="               rule  ||  "="               |  "0"               |  "_"               |  [M]             
  d=1  "4"                   
  done = 1

step 10  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"       (b)           ||  "_"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "="       (b)     rule  ||  "="       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "0"       (b)           ||  "0"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "_"       (b)           ||  "_"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "4"                     ||  "4"               |  "="    (a)        |  [M]               |  [M]             
  done = 0

step 11  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                   
  d=1  "4"          (c)      
  d=1  "%"                   
  d=2  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule
  d=1  "0"                   
  d=1  "_"                   
  d=1  "4"                   
  d=2  "="               orig  ||  "="               |  "2"               |  [M]               |  [M]             
  done = 1

step 12  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"       (b)           ||  "2"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "0"                     ||  "0"               |  [M]               |  [M]               |  [M]             
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"                     ||  "4"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "="       (b)     orig  ||  "="    (a)        |  "-"               |  [M]               |  [M]             
  d=1  "2"       (b)           ||  "2"       (b)     |  [M]               |  [M]               |  [M]             
  done = 0

step 13  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                   
  d=1  "4"          (c)      
  d=1  "%"                   
  d=1  "2"                   
  d=1  "="               rule
  d=1  "0"                   
  d=1  "_"                   
  d=2  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=2  "="               orig  ||  "="               |  "4"               |  [M]               |  [M]             
  d=1  "-"                   
  d=1  "2"                   
  done = 1

step 14  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "0"                     ||  "0"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "_"                     ||  "_"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "4"       (b)           ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "="       (b)     orig  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "4"       (b)           ||  "4"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "-"                     ||  "-"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  done = 0

step 15  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                   
  d=1  "4"          (c)      
  d=1  "%"                   
  d=1  "2"                   
  d=1  "="               rule
  d=2  "0"                     ||  "0"               |  [M]               |  [M]               |  [M]             
  d=2  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)      
  d=2  "="               orig  ||  "="               |  "1"               |  "_"               |  [M]             
  d=1  "4"                   
  d=1  "-"                   
  d=1  "2"                   
  done = 1

step 16  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "0"       (b)           ||  "0"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "_"       (b)           ||  "_"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "="       (b)     orig  ||  "="       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "1"       (b)           ||  "1"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "_"       (b)           ||  "_"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "4"                     ||  "4"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "-"                     ||  "-"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  done = 0

step 17  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                   
  d=1  "4"          (c)      
  d=1  "%"                   
  d=1  "2"                   
  d=1  "="               rule
  d=1  "0"          (c)      
  d=1  "_"          (c)      
  d=1  "4"          (c)      
  d=1  "="               orig
  d=1  "1"                   
  d=1  "_"                   
  d=2  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=2  "-"                     ||  "-"               |  [M]               |  [M]               |  [M]             
  d=2  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=2  "="               rule  ||  "="               |  "2"               |  [M]               |  [M]             
  done = 1

step 18  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "0"          (c)        ||  [M]               |  [M]               |  [M]               |  [M]             
  d=1  "_"          (c)        ||  [M]               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  [M]               |  [M]               |  [M]               |  [M]             
  d=1  "="               orig  ||  [M]               |  [M]               |  [M]               |  [M]             
  d=1  "1"                     ||  "1"               |  [M]               |  [M]               |  [M]             
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "2"       (b)           ||  "2"       (b)     |  [M]               |  [M]               |  [M]             
  done = 0

step 19  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "1"                     ||  "1"               |  [M]               |  [M]               |  [M]             
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  "="    (a)        |  [M]               |  [M]             
  done = 0

step 20  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                   
  d=1  "4"          (c)      
  d=1  "%"                   
  d=2  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule
  d=1  "1"                   
  d=1  "_"                   
  d=1  "2"                   
  d=2  "="               orig  ||  "="               |  "2"               |  [M]               |  [M]             
  done = 1

step 21  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"       (b)           ||  "2"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "1"                     ||  "1"               |  [M]               |  [M]               |  [M]             
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "="       (b)     orig  ||  "="    (a)        |  "-"               |  [M]               |  [M]             
  d=1  "2"       (b)           ||  "2"       (b)     |  [M]               |  [M]               |  [M]             
  done = 0

step 22  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                   
  d=1  "4"          (c)      
  d=1  "%"                   
  d=1  "2"                   
  d=1  "="               rule
  d=1  "1"                   
  d=1  "_"                   
  d=2  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=2  "="               orig  ||  "="               |  "2"               |  [M]               |  [M]             
  d=1  "-"                   
  d=1  "2"                   
  done = 1

step 23  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "1"                     ||  "1"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "_"                     ||  "_"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "="       (b)     orig  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "2"       (b)           ||  "2"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "-"                     ||  "-"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  done = 0

step 24  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                   
  d=1  "4"          (c)      
  d=1  "%"                   
  d=1  "2"                   
  d=1  "="               rule
  d=2  "1"                     ||  "1"               |  [M]               |  [M]               |  [M]             
  d=2  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "2"          (c)      
  d=2  "="               orig  ||  "="               |  "2"               |  "_"               |  [M]             
  d=1  "2"                   
  d=1  "-"                   
  d=1  "2"                   
  done = 1

step 25  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "1"       (b)           ||  "1"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "_"       (b)           ||  "_"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "="       (b)     orig  ||  "="       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "2"       (b)           ||  "2"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "_"       (b)           ||  "_"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "-"                     ||  "-"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  done = 0

step 26  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                   
  d=1  "4"          (c)      
  d=1  "%"                   
  d=1  "2"                   
  d=1  "="               rule
  d=1  "1"          (c)      
  d=1  "_"          (c)      
  d=1  "2"          (c)      
  d=1  "="               orig
  d=1  "2"                   
  d=1  "_"                   
  d=2  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=2  "-"                     ||  "-"               |  [M]               |  [M]               |  [M]             
  d=2  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=2  "="               rule  ||  "="               |  "0"               |  [M]               |  [M]             
  done = 1

step 27  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "1"          (c)        ||  [M]               |  [M]               |  [M]               |  [M]             
  d=1  "_"          (c)        ||  [M]               |  [M]               |  [M]               |  [M]             
  d=1  "2"          (c)        ||  [M]               |  [M]               |  [M]               |  [M]             
  d=1  "="               orig  ||  [M]               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "0"       (b)           ||  "0"       (b)     |  [M]               |  [M]               |  [M]             
  done = 0

step 28  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "0"                     ||  "0"               |  "="    (a)        |  [M]               |  [M]             
  done = 0

step 29  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                   
  d=1  "4"          (c)      
  d=1  "%"                   
  d=2  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule
  d=1  "2"                   
  d=1  "_"                   
  d=1  "0"                   
  d=2  "="               orig  ||  "="               |  "2"               |  [M]               |  [M]             
  done = 1

step 30  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"       (b)           ||  "2"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "0"                     ||  "0"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "="       (b)     orig  ||  "="    (a)        |  "-"               |  [M]               |  [M]             
  d=1  "2"       (b)           ||  "2"       (b)     |  [M]               |  [M]               |  [M]             
  done = 0

step 31  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                   
  d=1  "4"          (c)      
  d=1  "%"                   
  d=1  "2"                   
  d=1  "="               rule
  d=1  "2"                   
  d=1  "_"                   
  d=2  "0"                     ||  "0"               |  [M]               |  [M]               |  [M]             
  d=2  "="               orig  ||  "="               |  "0"               |  [M]               |  [M]             
  d=1  "-"                   
  d=1  "2"                   
  done = 1

step 32  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "_"                     ||  "_"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "0"       (b)           ||  "0"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "="       (b)     orig  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "0"       (b)           ||  "0"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "-"                     ||  "-"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  done = 0

step 33  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                   
  d=1  "4"          (c)      
  d=1  "%"                   
  d=1  "2"                   
  d=1  "="               rule
  d=2  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=2  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "0"          (c)      
  d=2  "="               orig  ||  "="               |  "3"               |  "_"               |  [M]             
  d=1  "0"                   
  d=1  "-"                   
  d=1  "2"                   
  done = 1

step 34  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "_"       (b)           ||  "_"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "0"          (c)        ||  "0"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "="       (b)     orig  ||  "="       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "3"       (b)           ||  "3"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "_"       (b)           ||  "_"       (b)     |  [M]               |  [M]               |  [M]             
  d=1  "0"                     ||  "0"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "-"                     ||  "-"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  done = 0

step 35  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                   
  d=1  "4"          (c)      
  d=1  "%"                   
  d=1  "2"                   
  d=1  "="               rule
  d=1  "2"          (c)      
  d=1  "_"          (c)      
  d=1  "0"          (c)      
  d=1  "="               orig
  d=1  "3"                   
  d=1  "_"                   
  d=2  "0"                     ||  "0"               |  [M]               |  [M]               |  [M]             
  d=2  "-"                     ||  "-"               |  [M]               |  [M]               |  [M]             
  d=2  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=2  "="               rule  ||  "="               |  "-"               |  "2"               |  [M]             
  done = 1

step 36  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  "_"               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "_"          (c)        ||  "_"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "0"          (c)        ||  "0"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "="               orig  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  d=1  "3"                     ||  [M]               |  [M]               |  [M]               |  [M]             
  d=1  "_"                     ||  [M]               |  [M]               |  [M]               |  [M]             
  d=1  "-"       (b)           ||  [M]               |  [M]               |  [M]               |  [M]             
  d=1  "2"       (b)           ||  [M]               |  [M]               |  [M]               |  [M]             
  done = 0

step 37  max_depth=2
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                   
  d=1  "4"          (c)      
  d=1  "%"                   
  d=2  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule
  d=1  "2"          (c)      
  d=1  "_"          (c)      
  d=1  "0"          (c)      
  d=2  "="               orig  ||  "="               |  "2"               |  [M]               |  [M]             
  done = 1

step 38  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "_"                     ||  [M]               |  [M]               |  [M]               |  [M]             
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "_"          (c)        ||  "_"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "0"          (c)        ||  "0"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "="       (b)     orig  ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  done = 0

step 39  max_depth=1
  d=0  "4"          (c)      
  d=0  "/"                   
  d=0  "2"          (c)      
  d=0  "="               orig
  d=1  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=1  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "_"          (c)        ||  "_"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "0"          (c)        ||  "0"          (c)  |  [M]               |  [M]               |  [M]             
  d=1  "%"                     ||  "%"               |  [M]               |  [M]               |  [M]             
  d=1  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  done = 1

step 40  max_depth=0
  d=0  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "/"                     ||  "/"               |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="               |  [M]               |  [M]               |  [M]             
  d=0  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "_"       (b)           ||  [M]               |  [M]               |  [M]               |  [M]             
  d=0  "0"       (b)           ||  [M]               |  [M]               |  [M]               |  [M]             
  d=0  "%"       (b)           ||  [M]               |  [M]               |  [M]               |  [M]             
  d=0  "2"       (b)           ||  [M]               |  [M]               |  [M]               |  [M]             
  done = 0

step 41  max_depth=0
  d=0  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "/"                     ||  "/"               |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="               orig  ||  "="               |  [M]               |  [M]               |  [M]             
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  done = 1
```
