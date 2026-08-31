"""Exp 2 — control-element removal transforms.

The same teacher trajectory is used; the dropped bit is zeroed at the very
end of data prep ("masked at the end as the label"), and the training loop
additionally flows no loss into that control element.

  drop="b"  (CTRL_RESULT, ctrl idx 1): zeroed in STATE tokens AND in the
            action labels. NOTE: teacher rules copy the current ctrl through
            ``keep``/``replace``, so recorded actions frequently carry
            (b)=1 — but the engine force-zeroes (b) on every action-derived
            cell, and the training loss already excludes the (b) element, so
            zeroing it in labels changes nothing observable.
  drop="c"  (CTRL_TAG_C, ctrl idx 2): zeroed in STATE tokens AND in the
            action labels.

Masking commutes with the teacher engine for (c) (the engine never reads
(c); it only reads (a) and writes (b)), so replaying the masked actions
reproduces exactly the masked state sequence — verified by
``ablation/verify/verify_ctl_mask.py``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from teacher import tokens as T

_DROP_TO_BIT = {"b": T.CTRL_RESULT, "c": T.CTRL_TAG_C}


def drop_bit_index(drop: str) -> int:
    if drop not in _DROP_TO_BIT:
        raise ValueError(f"drop must be 'b' or 'c'; got {drop!r}")
    return _DROP_TO_BIT[drop]


def mask_state_tokens(tokens: List[List[int]], bit: int) -> List[List[int]]:
    """Return a copy of state rows with ctrl bit ``bit`` zeroed."""
    out = []
    for row in tokens:
        r = list(row)
        r[1 + bit] = 0
        out.append(r)
    return out


def mask_action(action, bit: int):
    """Return a copy of a nested action with ctrl bit ``bit`` zeroed."""
    out = []
    for pos in action:
        new_pos = []
        for cell in pos:
            c = list(cell)
            c[1 + bit] = 0
            new_pos.append(c)
        out.append(new_pos)
    return out


def mask_steps(
    steps: Optional[List[Tuple[list, list, list, int]]],
    drop: str,
) -> Optional[List[Tuple[list, list, list, int]]]:
    """Apply the drop transform to a `_collect_steps` result (None passes
    through — the caller drops failed tasks the same way as the baseline).
    """
    if steps is None:
        return None
    bit = drop_bit_index(drop)
    out = []
    for tokens, depth, action, done in steps:
        out.append((
            mask_state_tokens(tokens, bit),
            list(depth),
            mask_action(action, bit),
            done,
        ))
    return out
