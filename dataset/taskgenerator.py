"""Math-expression task generation and the character tokenizer.

Task strings are plain expressions over digits, the four operators, '=',
'.', and parentheses. The teacher rules and the model both work on the
integer-ID form produced by `Tokenizer`.

Generators used by the pipeline:

  * `gen_task_1_copy(L)` — `number=` with an L-digit number.
  * `_get_random_num(l)` — one l-digit number (no leading zero for l > 1);
    the binary-op stages draw each operand with this directly.
  * `gen_seq_factored(n, d)` / `gen_seq_paren_factored(n, d)` — factored
    sequential / nested tasks: exactly ``n`` operands with digit-lengths in
    [1, d], at least one operand exactly ``d`` digits, operators drawn from
    {+, -, *} (nested additionally wraps random adjacent pairs in
    parentheses). Both return the NO-equation ``(expr, answer)`` form — the
    '=' is auto-appended by the teacher runner.
"""

import random


class Tokenizer:
    def __init__(self, args):
        self.MaskToken_id = args.MaskToken_id  # 0
        self.SOSToken_id = args.SOSToken_id    # 1
        self.PadToken_id = args.PadToken_id    # 2
        self.EOSToken_id = args.EOSToken_id    # 3
        self.DoneToken_id = args.DoneToken_id  # 4

        self.char2id = {}
        self.id2char = {}

        # 5-9: operators
        ops = ['=', '+', '-', '*', '/']
        for i, char in enumerate(ops, 5):
            self._add_token(char, i)

        # 10-19: digits
        for i in range(10):
            self._add_token(str(i), i + 10)

        # 20-29: symbols
        symbols = ['.', '(', ')', '{', '}', '|', '&', '~', 'XOR', ',']
        for i, char in enumerate(symbols, 20):
            self._add_token(char, i)

        # 30-39: lowercase letters
        for i, char in enumerate("abcdefghij", 30):
            self._add_token(char, i)

        # 40-49: uppercase letters
        for i, char in enumerate("ABCDEFGHIJ", 40):
            self._add_token(char, i)

        # Special token names for decoding purposes
        self.id2char[0] = "[MASK]"
        self.id2char[1] = "[SOS]"
        self.id2char[2] = "[PAD]"
        self.id2char[3] = "[EOS]"
        self.id2char[4] = "[DONE]"

    def _add_token(self, char, token_id):
        self.char2id[char] = token_id
        self.id2char[token_id] = char

    def encode(self, text):
        return [self.char2id[char] for char in text]

    def decode(self, ids):
        return "".join([self.id2char[i] for i in ids])


class TaskGenerator:
    def __init__(self, args):
        self.args = args
        self.tokenizer = Tokenizer(args)

    def _get_random_num(self, length):
        if length == 1:
            return str(random.randint(0, 9))
        first = str(random.randint(1, 9))
        rest = "".join([str(random.randint(0, 9)) for _ in range(length - 1)])
        return first + rest

    def _distribute_digits(self, total_digits, num_bins):
        lengths = [1] * num_bins
        for _ in range(total_digits - num_bins):
            lengths[random.randint(0, num_bins - 1)] += 1
        return lengths

    # ==========================================
    # Copy
    # ==========================================
    def gen_task_1_copy(self, length):
        # `length` counts the digits (the '=' is extra).
        num = self._get_random_num(length)
        task = f"{num}="
        target = f"{num}={num}"
        return task, target

    # ==========================================
    # Factored sequential / nested tasks: parameterized by
    # (n_numbers, max_digit_len) instead of total length. Each task has
    # exactly ``n_numbers`` operands; operand digit-lengths are drawn from
    # [1, max_digit_len] with **at least one operand exactly max_digit_len
    # digits** (so the cell's difficulty is pinned to max_digit_len). Both
    # return the NO-equation ``(expr, answer)`` form.
    # ==========================================
    def _factored_num_lengths(self, n_numbers, max_digit_len):
        """``n_numbers`` digit-lengths in [1, max_digit_len], at least one
        equal to ``max_digit_len``."""
        n = max(1, int(n_numbers))
        m = max(1, int(max_digit_len))
        lengths = [random.randint(1, m) for _ in range(n)]
        lengths[random.randint(0, n - 1)] = m  # pin >=1 operand to max_digit_len
        return lengths

    def _gen_multi_no_paren_factored_core(self, n_numbers, max_digit_len):
        ops_choices = ['+', '-', '*']
        lengths = self._factored_num_lengths(n_numbers, max_digit_len)
        nums = [self._get_random_num(l) for l in lengths]
        ops = [random.choice(ops_choices) for _ in range(len(nums) - 1)]
        expr = nums[0]
        for i in range(len(ops)):
            expr += ops[i] + nums[i + 1]
        return expr, str(eval(expr))

    def _gen_multi_paren_factored_core(self, n_numbers, max_digit_len):
        ops_choices = ['+', '-', '*']
        lengths = self._factored_num_lengths(n_numbers, max_digit_len)
        nums = [self._get_random_num(l) for l in lengths]
        ops = [random.choice(ops_choices) for _ in range(len(nums) - 1)]
        # Wrap up to half of the operators in parentheses (>=1 when possible)
        # by randomly grouping adjacent operand pairs; groups may nest.
        if ops:
            num_parens = random.randint(1, max(1, len(ops) // 2))
            for _ in range(num_parens):
                if not ops:
                    break
                idx = random.randint(0, len(ops) - 1)
                nums[idx] = f"({nums[idx]}{ops[idx]}{nums[idx + 1]})"
                nums.pop(idx + 1)
                ops.pop(idx)
        expr = nums[0]
        for i in range(len(ops)):
            expr += ops[i] + nums[i + 1]
        return expr, str(eval(expr))

    def gen_seq_factored(self, n_numbers, max_digit_len):
        """Multi-op (no paren) over ``n_numbers`` operands; ops ∈ {+, -, *}."""
        return self._gen_multi_no_paren_factored_core(n_numbers, max_digit_len)

    def gen_seq_paren_factored(self, n_numbers, max_digit_len):
        """Multi-op with parentheses over ``n_numbers`` operands; ops ∈ {+, -, *}."""
        return self._gen_multi_paren_factored_core(n_numbers, max_digit_len)
