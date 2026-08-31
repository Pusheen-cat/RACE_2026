"""Vocab IDs and control-bit indices used by teacher rules.

The vocab IDs match `dataset.taskgenerator.Tokenizer`; we duplicate them here so
rule modules don't need to import the dataset package.

Control-bit indices below are counted *within the control portion* of a token —
i.e. excluding the leading vocab cell. So in the action tensor the offset is +1
(action[s, m, 0] is vocab; action[s, m, 1+CTRL_*] is the control bit).

  CTRL_SELF_CALL (a) — set by rule, engine bumps depth and clears all bits
  CTRL_RESULT    (b) — set automatically by engine on self-call return,
                       auto-cleared by the next action (rules cannot write it)
  CTRL_TAG_C     (c) — first user-controllable tag
  CTRL_TAG_D, _E (d, e) — reserved for future tasks (NOT used by add or copy)
"""

# --- vocab IDs ---
MASK = 0
SOS = 1
PAD = 2
EOS = 3
DONE = 4

EQ = 5
PLUS = 6
MINUS = 7
STAR = 8
SLASH = 9

DIGIT_0 = 10
DIGIT_9 = 19

DOT = 20
LPAREN = 21
RPAREN = 22

# Division-task scratchpad symbols. They live in this teacher tokenizer only;
# the dataset Tokenizer never sees them because division targets are decimal
# (e.g. "12/3=4.000"). They're introduced and removed inside the trajectory.
PERCENT = 23     # '%' modulo / remainder marker
UNDERSCORE = 24  # '_' operand bracket marker

# 'b' (lowercase, used for the 'borrow' marker in subtract_task) lives in the
# letter range from `Tokenizer`: 'a'..'j' = 30..39, so 'b' = 31.
B_LOWER = 31

# --- control-bit indices (within the ctrl portion) ---
CTRL_SELF_CALL = 0  # (a)
CTRL_RESULT = 1     # (b) — engine-set only
CTRL_TAG_C = 2      # (c)
CTRL_TAG_D = 3      # (d) — reserved
CTRL_TAG_E = 4      # (e) — reserved


def is_digit(token_id: int) -> bool:
    return DIGIT_0 <= token_id <= DIGIT_9


_ID_TO_CHAR = {
    MASK: "[MASK]", SOS: "[SOS]", PAD: "[PAD]", EOS: "[EOS]", DONE: "[DONE]",
    EQ: "=", PLUS: "+", MINUS: "-", STAR: "*", SLASH: "/",
    DOT: ".", LPAREN: "(", RPAREN: ")",
    PERCENT: "%", UNDERSCORE: "_",
    B_LOWER: "b",
}
for _i in range(10):
    _ID_TO_CHAR[DIGIT_0 + _i] = str(_i)

_CHAR_TO_ID = {v: k for k, v in _ID_TO_CHAR.items() if not v.startswith("[")}


def encode(text: str) -> list:
    return [_CHAR_TO_ID[ch] for ch in text]


def decode(ids) -> str:
    """Decode a sequence of vocab IDs back to text.

    Unmapped IDs (e.g., reserved-but-not-named entries within the model's
    vocab_size budget) decode to ``[?<id>]`` placeholders so that an eval
    pass can score the model's output as a string mismatch rather than
    crashing with KeyError mid-grid.
    """
    return "".join(_ID_TO_CHAR.get(i, f"[?{i}]") for i in ids)
