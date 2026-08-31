"""Action-tensor builder.

The model's per-step action is shaped `(S_eff + 1, 1 + max_insertion, 1 + ctl_size)`:
  - `action[s, 0, :]`   replaces the s-th max-depth token (vocab + ctrl bits)
  - `action[s, m, :]`   for m >= 1 inserts a new token after position s
  - `action[-1, 0, 0]`  holds the done indicator (0 / 1)

A "no-op insertion" is encoded as vocab=MASK; the engine drops MASK cells.

Rule code stays declarative:
    builder.keep(s, vocab, ctrl)            # leave the token alone
    builder.replace(s, vocab, new_ctrl)     # change vocab and/or ctrl bits
    builder.insert(s, slot, vocab, ctrl)    # insert into slot 1..max_insertion
    builder.set_done(1)                     # mark trajectory complete
"""

from __future__ import annotations

from typing import List

from teacher import tokens as T


class ActionBuilder:
    def __init__(self, n_max_depth_tokens: int, ctl_size: int, max_insertion: int):
        self.n = n_max_depth_tokens
        self.ctl_size = ctl_size
        self.max_insertion = max_insertion
        self.M = 1 + max_insertion
        self.C = 1 + ctl_size

        # (S_eff + 1) positions: last one is the done slot.
        # Default slot 0 = PAD (rule must call keep/replace), slots 1..M-1 = MASK.
        self._action = [
            [[T.MASK] + [0] * ctl_size for _ in range(self.M)]
            for _ in range(self.n + 1)
        ]
        for s in range(self.n):
            self._action[s][0] = [T.PAD] + [0] * ctl_size  # sentinel
        # done slot vocab cell starts at 0 (= done indicator value 0)
        self._action[-1][0] = [0] + [0] * ctl_size

    # ---- per-position edits (s_eff is the index within max_depth_indices) ----
    def keep(self, s_eff: int, vocab: int, ctrl: List[int]) -> None:
        self._action[s_eff][0] = [vocab] + list(ctrl)

    def replace(self, s_eff: int, vocab: int, ctrl: List[int]) -> None:
        self._action[s_eff][0] = [vocab] + list(ctrl)

    def delete(self, s_eff: int) -> None:
        self._action[s_eff][0] = [T.MASK] + [0] * self.ctl_size

    def insert(self, s_eff: int, slot: int, vocab: int, ctrl: List[int]) -> None:
        if not (1 <= slot <= self.max_insertion):
            raise ValueError(f"slot must be in 1..{self.max_insertion}, got {slot}")
        self._action[s_eff][slot] = [vocab] + list(ctrl)

    def set_done(self, value: int) -> None:
        if value not in (0, 1):
            raise ValueError(f"done must be 0 or 1, got {value}")
        self._action[-1][0][0] = value

    # ---- export ----
    def to_nested_list(self):
        # validate that every visible position got something
        for s in range(self.n):
            if self._action[s][0][0] == T.PAD:
                raise RuntimeError(
                    f"action position {s} was never assigned (still PAD); "
                    f"rule must call keep/replace/delete for every max-depth token"
                )
        return [[list(slot) for slot in row] for row in self._action]

    def to_tensor(self):
        import torch  # local import: rule code can run without torch
        return torch.tensor(self.to_nested_list(), dtype=torch.long)
