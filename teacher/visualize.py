"""Pretty-printer for teacher trajectories — one row per token.

Each step is rendered as a block:

    step <i>  max_depth=<d>
      d=<dep>  "<tok>" (a)( )(c)  [orig|rule]   ||  "<r0>" (a)( )( )  |  "<i1>" ( )( )( )  |  ...
      ...
      done = 0|1

Conventions:
- The input column shows the token in `"..."` (printable) or `[NAME]` (special).
- Control bits are rendered as fixed-width slots `(a)`, `(b)`, ... — one slot
  per ctrl index, with three spaces in place of an unset bit so the same
  letter always lands in the same column.
- An `=` cell is annotated `orig` if it came from the original input or `rule`
  if the runner auto-appended it (`state.auto_eq_depth[i] is not None`).
- For tokens at the current max depth, four action cells are rendered after
  the `||` separator: slot 0 = replacement, slots 1..3 = insertions.
- Tokens NOT at max depth (frozen under an outer self-call) are dimmed and
  their action cells are omitted entirely.
- MASK output cells (and their ctrl slots) are dimmed so the active edits
  stand out.
- `done` is printed on its own trailing line.
"""

from __future__ import annotations

import sys
from typing import Callable, List, Optional, Tuple

from teacher import tokens as T
from teacher.action import ActionBuilder
from teacher.engine import TeacherError, apply_action
from teacher.runner import _auto_append_eq
from teacher.state import TeacherState


RuleFn = Callable[[TeacherState], ActionBuilder]


# ANSI escape codes. We only emit them when the output is a TTY (or the caller
# explicitly opts in); plain-text dumps to files stay color-free.
_GRAY = "\033[90m"
_RESET = "\033[0m"


# Letters used to label control bits. Index k -> _TAG_LETTERS[k].
_TAG_LETTERS = ("a", "b", "c", "d", "e")


# Width of one control slot like "(a)".
_CTL_SLOT_W = 3
_CTL_BLANK = " " * _CTL_SLOT_W

# Token cell padding. The widest token we render is "[DONE]" = 6 chars;
# everything else fits in 6.
_TOK_W = 6

# Width reserved for the "=" annotation column (`orig` / `rule` / blank).
_EQ_MARK_W = 4


def _vocab_repr(vocab_id: int) -> str:
    """Render a vocab id: special tokens go in `[...]`, the rest in `"..."`.

    `[M]` is a deliberately short alias for MASK because action rows are
    dominated by MASK insert slots.
    """
    if vocab_id == T.MASK:
        return "[M]"
    if vocab_id == T.SOS:
        return "[SOS]"
    if vocab_id == T.PAD:
        return "[PAD]"
    if vocab_id == T.EOS:
        return "[EOS]"
    if vocab_id == T.DONE:
        return "[DONE]"
    return f'"{T.decode([vocab_id])}"'


def _ctl_columns(ctrl: List[int], ctl_size: int) -> str:
    """Render ctrl bits as fixed-width slots, one per ctrl index. Unset bits
    leave 3 spaces so identical labels stay in the same column."""
    parts = []
    for k in range(ctl_size):
        if k < len(ctrl) and ctrl[k]:
            letter = _TAG_LETTERS[k] if k < len(_TAG_LETTERS) else f"{k}"
            parts.append(f"({letter})")
        else:
            parts.append(_CTL_BLANK)
    return "".join(parts)


def _wrap_gray(s: str, dim: bool, use_color: bool) -> str:
    if dim and use_color:
        return f"{_GRAY}{s}{_RESET}"
    return s


def _format_cell(vocab_id: int, ctrl: List[int], ctl_size: int,
                 dim: bool, use_color: bool) -> str:
    """One token+ctrl cell, dimmed in its entirety if `dim` is set."""
    tok = _vocab_repr(vocab_id).ljust(_TOK_W)
    ctl = _ctl_columns(ctrl, ctl_size)
    return _wrap_gray(f"{tok} {ctl}", dim, use_color)


def _eq_marker_text(state: TeacherState, idx: int) -> str:
    """`orig` / `rule` / empty depending on whether the `=` was original."""
    if state.vocab(idx) != T.EQ:
        return ""
    return "rule" if state.auto_eq_depth[idx] is not None else "orig"


def format_step(
    state: TeacherState,
    action: list,
    step_idx: Optional[int] = None,
    use_color: bool = True,
    indent: str = "  ",
) -> str:
    """Format a single (state, action) pair into a multi-line string.

    `action` is the nested list shape `(S_eff + 1, M, 1+ctl_size)` produced by
    `ActionBuilder.to_nested_list()` — i.e. one row per max-depth token plus a
    trailing "done slot" row.
    """
    ctl_size = state.ctl_size
    md_indices = state.max_depth_indices()
    md_relative = {i: rel for rel, i in enumerate(md_indices)}
    max_d = state.max_depth()

    M = len(action[0]) if action and action[0] else 0

    out_lines = []
    if step_idx is not None:
        out_lines.append(f"step {step_idx}  max_depth={max_d}")

    for i in range(len(state.tokens)):
        if state.vocab(i) == T.PAD:
            continue
        d = state.depth[i]
        inactive = d < max_d
        vocab = state.vocab(i)
        ctrl = state.ctrl(i)

        in_cell = _format_cell(vocab, ctrl, ctl_size,
                               dim=inactive, use_color=use_color)
        eq_mark = _eq_marker_text(state, i)
        eq_text = eq_mark.ljust(_EQ_MARK_W)
        eq_text = _wrap_gray(eq_text, inactive, use_color)

        prefix = f"{indent}d={d}  {in_cell}  {eq_text}"

        if inactive:
            out_lines.append(prefix)
            continue

        rel = md_relative[i]
        slot_strs = []
        for m in range(M):
            cell = action[rel][m]
            cv = cell[0]
            cc = cell[1:1 + ctl_size]
            # Dim MASK output cells (including their ctrl columns) so the
            # active edits / inserts pop visually.
            dim_slot = (cv == T.MASK)
            slot_strs.append(_format_cell(cv, cc, ctl_size,
                                          dim=dim_slot,
                                          use_color=use_color))
        sep = "  |  "
        out_lines.append(f"{prefix}  ||  " + sep.join(slot_strs))

    done_val = int(action[-1][0][0]) if action else 0
    out_lines.append(f"{indent}done = {done_val}")
    return "\n".join(out_lines)


def format_trajectory(
    states: List[TeacherState],
    actions: List[list],
    use_color: Optional[bool] = None,
    indent: str = "  ",
) -> str:
    """Format every captured (state, action) pair, separated by blank lines.

    `use_color=None` means autodetect from `sys.stdout.isatty()`."""
    if use_color is None:
        use_color = sys.stdout.isatty()
    blocks = []
    for i, (st, act) in enumerate(zip(states, actions)):
        blocks.append(format_step(st, act, step_idx=i,
                                  use_color=use_color, indent=indent))
    return "\n\n".join(blocks)


def walk_trajectory(
    initial: TeacherState,
    rule_fn: RuleFn,
    max_steps: int = 2000,
) -> Tuple[List[TeacherState], List[list], TeacherState, bool, Optional[str]]:
    """Run `rule_fn` on `initial` while capturing the (state, action) pair at
    every step. Mirrors `runner.run_teacher` but exposes the per-step state
    snapshots that the visualizer needs.

    Returns `(states, actions, final_state, terminated, error)`. `states[i]`
    is the state the rule saw at step i (after auto-eq insertion), and
    `actions[i]` is the nested-list action the rule emitted in response.
    """
    state = _auto_append_eq(initial)
    states: List[TeacherState] = []
    actions: List[list] = []
    error: Optional[str] = None
    for step_i in range(max_steps):
        state = _auto_append_eq(state)
        try:
            builder = rule_fn(state)
        except Exception as e:
            error = f"<dispatch error step={step_i}: {type(e).__name__}: {e}>"
            return states, actions, state, False, error
        try:
            action = builder.to_nested_list()
        except Exception as e:
            error = f"<action build error step={step_i}: {type(e).__name__}: {e}>"
            return states, actions, state, False, error

        states.append(state)
        actions.append(action)

        try:
            state, fdone = apply_action(state, action)
        except TeacherError as e:
            error = f"<engine error step={step_i}: {e}>"
            return states, actions, state, False, error

        if fdone == 1:
            return states, actions, state, True, None
        if fdone == 2:
            error = f"<self-call+done same step at {step_i}>"
            return states, actions, state, False, error

    error = f"<not terminated after {max_steps} steps>"
    return states, actions, state, False, error


__all__ = ["format_step", "format_trajectory", "walk_trajectory"]
