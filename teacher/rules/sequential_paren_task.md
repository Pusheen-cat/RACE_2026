# Nested (parenthesized) rule (`sequential_paren_task.py`)

**Task.** A multi-operator expression with parentheses, e.g. `2*(3+4)` →
`14`. Like the sequential task, inputs carry **no `=`** (the runner
auto-appends one) and only the answer survives on the final tape. The rule
does exactly one thing: eliminate innermost parenthesized groups by
lifting their contents into sub-calls. Everything else — reducing the
lifted group, and finishing the flat expression that remains once all
parentheses are gone — is handled by the sequential and binary rules via
the dispatcher.

**Dispatch condition.** The visible LHS contains at least one `(` and one
`)`, well matched. (The sequential pattern rejects parentheses, so a mixed
tape always lands here first.)

See [README.md](README.md) for conventions and the trace format.

## Rule

A two-step loop over the visible LHS:

1. **Mark**: if no `)` on the LHS carries `(c)`, tag `(c)` on the **first
   `)`** (reading left to right — the closer of an innermost group).
   `done=0`.
2. **Eliminate**: if a `(c)`-tagged `)` exists, find its matching `(`,
   set `(a)` on every token strictly between them, and delete both
   parentheses. `done=0`. The lifted contents get an auto-appended `=`
   and dispatch to whichever rule matches (addition, multiplication,
   sequential, …); only the group's result returns, landing where the
   group stood.

Once no parentheses remain, the dispatcher stops routing to this rule and
the now-flat expression is finished by the sequential (or binary) rule.

## Example — `2*(3+4)`

```
step  depth  tape             visible        done
  0   d=0    2*(3+4)=         2*(3+4)=       0    mark: (c) on ")"
  1   d=0    2*(3+4)=         2*(3+4)=       0    eliminate: lift "3+4", delete "(" ")"
  2   d=1    2*3+4==          3+4=           1    addition rule: 3+4 → 7
  3   d=0    2*7=             2*7=           0    multiplication rule takes over
  4   d=1    2*7=             2*7=           1    memorized: 2*7 → 14
  5   d=0    2*7=14           2*7=14         0    multiplication retag
  6   d=0    2*7=14           2*7=14         1    done → auto-eq cleanup leaves "14"
```

Steps 0–1 are this rule's entire contribution: mark the innermost `)`,
then lift `3+4` and delete the parentheses. The lifted group's `=` is
auto-appended, so only `7` returns and the tape becomes `2*7=` — a plain
multiplication, which the dispatcher hands to the multiplication rule for
the remaining steps. With nested groups (e.g. `2*((3+4)*5)`) the same
two-step loop simply fires once per group, innermost first, because the
first `)` always closes an innermost group.

Full trace:

```
step 0  max_depth=0
  d=0  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=0  "*"                     ||  "*"               |  [M]               |  [M]               |  [M]             
  d=0  "("                     ||  "("               |  [M]               |  [M]               |  [M]             
  d=0  "3"                     ||  "3"               |  [M]               |  [M]               |  [M]             
  d=0  "+"                     ||  "+"               |  [M]               |  [M]               |  [M]             
  d=0  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=0  ")"                     ||  ")"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  done = 0

step 1  max_depth=0
  d=0  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=0  "*"                     ||  "*"               |  [M]               |  [M]               |  [M]             
  d=0  "("                     ||  [M]               |  [M]               |  [M]               |  [M]             
  d=0  "3"                     ||  "3"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "+"                     ||  "+"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "4"                     ||  "4"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  ")"          (c)        ||  [M]               |  [M]               |  [M]               |  [M]             
  d=0  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  done = 0

step 2  max_depth=1
  d=0  "2"                   
  d=0  "*"                   
  d=1  "3"                     ||  "3"               |  [M]               |  [M]               |  [M]             
  d=1  "+"                     ||  "+"               |  [M]               |  [M]               |  [M]             
  d=1  "4"                     ||  "4"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  "7"               |  [M]               |  [M]             
  d=0  "="               rule
  done = 1

step 3  max_depth=0
  d=0  "2"                     ||  "2"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "*"                     ||  "*"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "7"       (b)           ||  "7"    (a)        |  [M]               |  [M]               |  [M]             
  d=0  "="               rule  ||  "="    (a)        |  [M]               |  [M]               |  [M]             
  done = 0

step 4  max_depth=1
  d=1  "2"                     ||  "2"               |  [M]               |  [M]               |  [M]             
  d=1  "*"                     ||  "*"               |  [M]               |  [M]               |  [M]             
  d=1  "7"                     ||  "7"               |  [M]               |  [M]               |  [M]             
  d=1  "="               rule  ||  "="               |  "1"               |  "4"               |  [M]             
  done = 1

step 5  max_depth=0
  d=0  "2"       (b)           ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "*"       (b)           ||  "*"       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "7"       (b)           ||  "7"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="       (b)     rule  ||  "="       (b)     |  [M]               |  [M]               |  [M]             
  d=0  "1"       (b)           ||  "1"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "4"       (b)           ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  done = 0

step 6  max_depth=0
  d=0  "2"          (c)        ||  "2"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "*"                     ||  "*"               |  [M]               |  [M]               |  [M]             
  d=0  "7"          (c)        ||  "7"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "="               rule  ||  "="               |  [M]               |  [M]               |  [M]             
  d=0  "1"          (c)        ||  "1"          (c)  |  [M]               |  [M]               |  [M]             
  d=0  "4"          (c)        ||  "4"          (c)  |  [M]               |  [M]               |  [M]             
  done = 1
```
