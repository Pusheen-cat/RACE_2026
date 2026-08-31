"""Pure-Python state representation for the teacher path generator.

A `TeacherState` is a flat tape of tokens plus a parallel depth list. Each token
is `[vocab, c_0, c_1, ..., c_{ctl_size-1}]`. Pads (vocab=PAD) are dropped on
construction — the tape is always tightly packed.

A second parallel list `auto_eq_depth` tags `=` cells the runner auto-appended
(see `runner._auto_append_eq` and `engine.apply_action`'s done branch). For any
non-auto-appended cell the entry is `None`; for an auto-appended `=` it holds
the depth at which the cell became visible (i.e. `state.max_depth()` at the
moment of insertion). The engine consumes that tag on the matching return.

Rule code in `teacher/tasks/*.py` only ever talks to this object; raw tensors
live in `action.py` and `engine.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from teacher import tokens as T


@dataclass
class TeacherState:
    tokens: List[List[int]]   # rows of length 1+ctl_size: [vocab, c0, c1, ...]
    depth: List[int]          # one per row, same length as tokens
    ctl_size: int
    auto_eq_depth: List[Optional[int]] = field(default_factory=list)

    def __post_init__(self):
        if not self.auto_eq_depth:
            self.auto_eq_depth = [None] * len(self.tokens)
        elif len(self.auto_eq_depth) != len(self.tokens):
            raise ValueError(
                f"auto_eq_depth length {len(self.auto_eq_depth)} != tokens "
                f"length {len(self.tokens)}"
            )

    # ---------- constructors ----------
    @classmethod
    def from_text(cls, text: str, ctl_size: int) -> "TeacherState":
        ids = T.encode(text)
        rows = [[v] + [0] * ctl_size for v in ids]
        return cls(tokens=rows, depth=[0] * len(rows), ctl_size=ctl_size,
                   auto_eq_depth=[None] * len(rows))

    @classmethod
    def from_token_ids(cls, ids: List[int], ctl_size: int) -> "TeacherState":
        rows = [[v] + [0] * ctl_size for v in ids]
        return cls(tokens=rows, depth=[0] * len(rows), ctl_size=ctl_size,
                   auto_eq_depth=[None] * len(rows))

    # ---------- queries ----------
    def __len__(self) -> int:
        return len(self.tokens)

    def vocab(self, i: int) -> int:
        return self.tokens[i][0]

    def ctrl(self, i: int) -> List[int]:
        return self.tokens[i][1:]

    def ctrl_at(self, i: int, k: int) -> int:
        return self.tokens[i][1 + k]

    def max_depth(self) -> int:
        return max(self.depth) if self.depth else 0

    def max_depth_indices(self) -> List[int]:
        d = self.max_depth()
        return [i for i, di in enumerate(self.depth) if di == d
                and self.tokens[i][0] != T.PAD]

    def depth0_vocab_string(self) -> str:
        ids = [row[0] for row, d in zip(self.tokens, self.depth)
               if d == 0 and row[0] != T.PAD]
        return T.decode(ids)

    def vocabs_at_depth(self, d: int) -> List[int]:
        return [row[0] for row, di in zip(self.tokens, self.depth)
                if di == d and row[0] != T.PAD]

    # ---------- debug ----------
    def pretty(self) -> str:
        lines = []
        for i, (row, d, ae) in enumerate(zip(self.tokens, self.depth, self.auto_eq_depth)):
            ctrl = row[1:]
            tags = []
            n_ctrl = len(ctrl)
            if T.CTRL_SELF_CALL < n_ctrl and ctrl[T.CTRL_SELF_CALL]: tags.append("a")
            if T.CTRL_RESULT    < n_ctrl and ctrl[T.CTRL_RESULT]:    tags.append("b")
            if T.CTRL_TAG_C     < n_ctrl and ctrl[T.CTRL_TAG_C]:     tags.append("c")
            if T.CTRL_TAG_D     < n_ctrl and ctrl[T.CTRL_TAG_D]:     tags.append("d")
            if T.CTRL_TAG_E     < n_ctrl and ctrl[T.CTRL_TAG_E]:     tags.append("e")
            tag_s = f"({','.join(tags)})" if tags else ""
            ae_s = f" auto_eq@{ae}" if ae is not None else ""
            lines.append(f"  [{i}] d={d} {T.decode([row[0]])}{tag_s}{ae_s}")
        return "\n".join(lines)
