"""Rule-based teacher that produces the imitation-learning trajectories.

A teacher trajectory is the `(state, action, done)` sequence a hand-written
deterministic rule set emits while solving a task by iterative tape editing —
the same action space the model uses. The rules defer `done=1` until every
LHS digit on the visible tape carries the persistent `(c)` tag, so no
in-flight sub-call result rides along on the closing step.

Layout:
  tokens.py      vocab IDs + control-bit indices
  state.py       TeacherState (flat tape: tokens, depth, auto-eq tags)
  action.py      ActionBuilder (the only way rules construct actions)
  engine.py      apply_action — single source of truth for state evolution
  runner.py      drives rule -> action -> engine; auto-appends '='
  dispatcher.py  per-step routing from the visible-tape pattern to a rule
  rules/         one module per task rule, each with a companion .md
                 document (rules/README.md explains the conventions and
                 the trace format; each task doc walks through executed
                 small-digit example trajectories)
  stats/         per-task trajectory-length tables (drives LR scheduling)
  visualize.py   trajectory pretty-printer
"""
