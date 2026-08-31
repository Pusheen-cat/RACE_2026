"""Shared configuration dataclass for the supervised pipeline.

Defaults reproduce the paper configuration: a 2-layer, 4-head, 384-dim
directional encoder with the GLPE-v1 positional encoding
(`model_dim_insertion=48`), vocab 32 and 3 control bits.

Notes on the sizes:

  * `model_vocab_size = 32` covers every token the teacher rules can
    introduce mid-trajectory: '%' (23), '_' (24), and 'b' (31). Embedding
    rows for tokens not yet observed at a given stage are random-init at
    stage 1 and start receiving gradient when a later stage first emits
    them.
  * `model_ctl_size = 3` — only (a) self-call, (b) result, and (c) tag are
    used by the task rules (`teacher.tokens` reserves two more indices that
    no rule writes).
  * `model_max_seq_len = 2500` so length-500+ inference (the tape grows to
    roughly 2x the input length) does not overflow the encoder input.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SupArgs:
    seed: int = 1

    model_layer: int = 2
    model_head: int = 4
    model_hidden: int = 384
    model_emb_control: int = 24
    model_dropout: float = 0.1
    model_vocab_size: int = 32
    model_ctl_size: int = 3
    model_max_insertion: int = 3
    model_dim_insertion: int = 48
    model_directional: bool = True
    model_max_seq_len: int = 2500
    # Positional-encoding family: "glpe_v1" (paper default, global/local PE —
    # requires model_hidden // model_head >= 96 and model_dim_insertion >= 48)
    # or "rope2d" (plain 2D RoPE; pair it with model_dim_insertion=24).
    # See model/positional_encodings.py.
    model_pe_kind: str = "glpe_v1"

    MaskToken_id: int = 0
    SOSToken_id: int = 1
    PadToken_id: int = 2
    EOSToken_id: int = 3
    DoneToken_id: int = 4

    loss_regularization_stable_vocab: float = 0.0
    loss_regularization_NoSpecialToken: float = 0.0

    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    batch_size: int = 1000
    num_workers: int = 16
    shuffle_buffer: int = 4000

    log_every: int = 50
    ckpt_dir: str = "checkpoints"
    results_dir: str = "results"
