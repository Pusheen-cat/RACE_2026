"""Ablation variants of the DLM backbone (exp 1 — positional-encoding ablation).

This is a modified copy of ``model/dlm_backbone.py``. It extends the
positional-encoding family with two ablation arms and a sign-flip switch,
while keeping the module tree (attribute names, parameter shapes) exactly
identical to the original so any release checkpoint loads unchanged
(PE tables are non-persistent buffers). ``ablation.verify.verify_pe_variants``
asserts numerical identity with the original encoder for the two shared
pe_kinds, guarding against copy drift.

``pe_kind`` accepted here:

  * ``"rope2d"``       — unchanged original 2D RoPE path.
  * ``"glpe_v1"``      — unchanged original GLPE path (used for the
                         equivalence gate and the exp-2 / exp-3 arms).
  * ``"glpe_v1_noinv"``— GLPE with the back-direction *position inversion*
                         removed: identical freqs/window, but the
                         ``att_back * sign`` multiplication (j > i → −1) is
                         skipped in both the main and the insert→main paths.
  * ``"nope_t"``       — T-axis NoPE: no rotation along the sequence axis
                         anywhere (main table = identity), no window mask.
                         The insertion-slot axis keeps the rope2d
                         ``theta_insert`` rotation so slots 1..3 stay
                         distinguishable. The directional sign flip is kept
                         (it is ablated separately by ``glpe_v1_noinv``).

The per-kind attention behaviour is resolved by ``resolve_pe(pe_kind)`` into
``(attn_kind ∈ {"rope2d", "glpe_v1"}, sign_flip)``; freq tables are resolved
in ``DLMEncoderAblation.__init__``.
"""
import math

import torch
import torch.nn as nn

from model.positional_encodings import (
    apply_rotary_emb,
    apply_rotary_emb_2d,
    glpe_v1_d_global_per_dir,
    precompute_freqs_cis_glpe_v1_insert,
    precompute_freqs_cis_glpe_v1_main,
    precompute_freqs_cis_rope2d,
    precompute_window_mask_glpe_v1,
)

ABLATION_PE_KINDS = ("rope2d", "glpe_v1", "glpe_v1_noinv", "nope_t")

# pe_kind -> (attn_kind, sign_flip)
_PE_RESOLVE = {
    "rope2d":        ("rope2d",  True),
    "glpe_v1":       ("glpe_v1", True),
    "glpe_v1_noinv": ("glpe_v1", False),
    "nope_t":        ("rope2d",  True),
}


def resolve_pe(pe_kind: str):
    if pe_kind not in _PE_RESOLVE:
        raise ValueError(
            f"Unknown ablation pe_kind={pe_kind!r}; expected one of {ABLATION_PE_KINDS}."
        )
    return _PE_RESOLVE[pe_kind]


# ---------------------------------------------------------------------------
# NoPE-T tables: identity rotation on the T axis, rope2d rotation on the
# insertion-slot axis.
# ---------------------------------------------------------------------------
def precompute_freqs_cis_nope_t_main(full_dim: int, dim_insert: int, end: int,
                                     directional: bool = True):
    """Main-stream table: all-identity rotations, shape ``(end, full_dim/2)``
    (post-halving when directional)."""
    if directional:
        full_dim = full_dim // 2
    assert full_dim % 2 == 0
    ones = torch.ones((end, full_dim // 2), dtype=torch.float32)
    return torch.polar(ones, torch.zeros_like(ones))


def precompute_freqs_cis_nope_t_insert(full_dim: int, dim_insert: int, end: int,
                                       max_insertion: int, directional: bool = True):
    """Insert-stream table: T-axis part identity; insertion-axis part is the
    rope2d ``theta_insert`` rotation (verbatim from
    ``precompute_freqs_cis_rope2d``'s insert branch)."""
    if directional:
        dim_insert = dim_insert // 2
        full_dim = full_dim // 2
        assert dim_insert % 4 == 0
    assert dim_insert % 2 == 0
    assert full_dim % 2 == 0

    dim = full_dim - dim_insert

    assert max_insertion in [1, 3, 7, 15]
    theta_insert = 2 * (max_insertion + 1)
    end_low_freq = max_insertion + 1
    assert dim_insert % 4 == 0
    assert 2 ** (dim_insert // 4) == theta_insert

    freqs_low = math.pi / (
        theta_insert ** (torch.arange(0, dim_insert, 2)[: (dim_insert // 2)].float() / dim_insert)
    )
    t_low = torch.arange(1, end_low_freq, device=freqs_low.device)
    freqs_low = torch.outer(t_low, freqs_low).float()
    freqs_cis_low = torch.polar(torch.ones_like(freqs_low), freqs_low)  # (M, dim_insert/2)

    ones_t = torch.ones((end, max_insertion, dim // 2), dtype=freqs_cis_low.dtype)
    freqs_cis_low = freqs_cis_low.unsqueeze(0).expand(end, max_insertion, -1)
    return torch.cat([ones_t, freqs_cis_low], dim=-1)


# ---------------------------------------------------------------------------
# Attention — copy of AttentionWithRoPE with (attn_kind, sign_flip)
# ---------------------------------------------------------------------------
class AttentionAblation(nn.Module):
    def __init__(self, hidden_size, num_heads, directional=False, max_insertion=None,
                 dropout=0.0, attn_kind: str = "rope2d", sign_flip: bool = True,
                 dim_insert: int = None):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.directional = directional
        self.max_insertion = max_insertion

        assert attn_kind in ("rope2d", "glpe_v1")
        self.attn_kind = attn_kind
        self.sign_flip = sign_flip
        if attn_kind == "glpe_v1":
            if dim_insert is None:
                raise ValueError("AttentionAblation requires dim_insert when attn_kind='glpe_v1'.")
            self.glpe_v1_D_global = glpe_v1_d_global_per_dir(
                self.head_dim, dim_insert, directional
            )
        else:
            self.glpe_v1_D_global = None

    def forward(self, x, x_insert, freqs_cis, freqs_cis_insert, pad_mask=None, window_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        qkv_insert = self.qkv(x_insert).reshape(B, T, self.max_insertion, 3, self.num_heads, self.head_dim)
        q_insert, k_insert, v_insert = qkv_insert.unbind(3)

        if not self.directional:
            raise NotImplementedError("ablation backbone only supports directional=True")

        q_front, q_back = torch.chunk(q, 2, dim=-1)
        k_front, k_back = torch.chunk(k, 2, dim=-1)
        q_front, k_front = apply_rotary_emb(q_front, k_front, freqs_cis)
        q_back, k_back = apply_rotary_emb(q_back, k_back, freqs_cis)

        q_front = q_front.transpose(1, 2)
        q_back = q_back.transpose(1, 2)
        k_front = k_front.transpose(1, 2)
        k_back = k_back.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        if self.attn_kind == "glpe_v1":
            D_g = self.glpe_v1_D_global
            wm = window_mask.to(q_front.dtype) if window_mask is not None else None
            s_f_g = q_front[..., :D_g] @ k_front[..., :D_g].transpose(-2, -1)
            s_f_l = q_front[..., D_g:] @ k_front[..., D_g:].transpose(-2, -1)
            if wm is not None:
                s_f_l = s_f_l * wm
            att_front = (s_f_g + s_f_l) / scale
            s_b_g = q_back[..., :D_g] @ k_back[..., :D_g].transpose(-2, -1)
            s_b_l = q_back[..., D_g:] @ k_back[..., D_g:].transpose(-2, -1)
            if wm is not None:
                s_b_l = s_b_l * wm
            att_back = (s_b_g + s_b_l) / scale
        else:
            att_front = (q_front @ k_front.transpose(-2, -1)) / scale
            att_back = (q_back @ k_back.transpose(-2, -1)) / scale

        # Position inversion (back-direction sign flip on j > i) — the
        # glpe_v1_noinv arm skips this multiplication entirely.
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        sign = torch.ones(T, T, device=x.device)
        sign[causal] = -1
        if self.sign_flip:
            att_back = att_back * sign

        att = att_front + att_back

        if pad_mask is not None:
            att = att.masked_fill(~pad_mask.view(B, 1, 1, T), float('-inf'))

        att = att.softmax(dim=-1)
        att = self.dropout(att)
        attn_out = att @ v

        # Insertion-slot attention.
        q_insert_front, q_insert_back = torch.chunk(q_insert, 2, dim=-1)
        k_insert_front, k_insert_back = torch.chunk(k_insert, 2, dim=-1)
        q_insert_front, k_insert_front = apply_rotary_emb_2d(q_insert_front, k_insert_front, freqs_cis_insert)
        q_insert_back, k_insert_back = apply_rotary_emb_2d(q_insert_back, k_insert_back, freqs_cis_insert)

        q_insert_front = q_insert_front.transpose(1, 3)
        q_insert_back = q_insert_back.transpose(1, 3)
        k_insert_front = k_insert_front.transpose(1, 3)
        k_insert_back = k_insert_back.transpose(1, 3)
        v_insert = v_insert.transpose(1, 3)

        if self.attn_kind == "glpe_v1":
            D_g = self.glpe_v1_D_global
            wm = window_mask.to(q_insert_front.dtype) if window_mask is not None else None
            s_if_g = q_insert_front[..., :D_g] @ k_front[..., :D_g].unsqueeze(2).transpose(-2, -1)
            s_if_l = q_insert_front[..., D_g:] @ k_front[..., D_g:].unsqueeze(2).transpose(-2, -1)
            if wm is not None:
                s_if_l = s_if_l * wm
            att_insert_front = (s_if_g + s_if_l) / scale
            s_ib_g = q_insert_back[..., :D_g] @ k_back[..., :D_g].unsqueeze(2).transpose(-2, -1)
            s_ib_l = q_insert_back[..., D_g:] @ k_back[..., D_g:].unsqueeze(2).transpose(-2, -1)
            if wm is not None:
                s_ib_l = s_ib_l * wm
            att_insert_back = (s_ib_g + s_ib_l) / scale
        else:
            att_insert_front = (q_insert_front @ k_front.unsqueeze(2).transpose(-2, -1)) / scale
            att_insert_back = (q_insert_back @ k_back.unsqueeze(2).transpose(-2, -1)) / scale

        score_front = (q_insert_front * k_insert_front).sum(dim=-1) / math.sqrt(self.head_dim)
        score_back = (q_insert_back * k_insert_back).sum(dim=-1) / math.sqrt(self.head_dim)
        score = score_front + score_back

        if self.sign_flip:
            att_insert_back = att_insert_back * sign

        att_insert = att_insert_front + att_insert_back

        score_expanded = score.unsqueeze(-1)
        att_insert = torch.cat([att_insert, score_expanded], dim=-1)

        if pad_mask is not None:
            mask_t = pad_mask.view(B, 1, 1, 1, T)
            mask_self = torch.ones((B, 1, 1, 1, 1), dtype=torch.bool, device=x.device)
            mask_full = torch.cat([mask_t, mask_self], dim=-1)
            att_insert = att_insert.masked_fill(~mask_full, float('-inf'))

        att_insert = att_insert.softmax(dim=-1)
        att_insert = self.dropout(att_insert)
        score = att_insert[..., -1]
        att_insert = att_insert[..., :-1]

        attn_insert_out = att_insert @ v.unsqueeze(2) + score.unsqueeze(-1) * v_insert

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_insert_out = attn_insert_out.transpose(1, 3).contiguous().view(B, T, self.max_insertion, C)
        return self.proj(attn_out), self.proj(attn_insert_out)


class TransformerBlockAblation(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, directional=False,
                 max_insertion=None, dropout=0.0, attn_kind: str = "rope2d",
                 sign_flip: bool = True, dim_insert: int = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = AttentionAblation(
            hidden_size, num_heads,
            directional=directional, max_insertion=max_insertion,
            dropout=dropout, attn_kind=attn_kind, sign_flip=sign_flip,
            dim_insert=dim_insert,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

        # Constant AdaLN modulation — same parameterization and init as the
        # release TransformerBlock (rows: shift_msa, scale_msa, gate_msa,
        # shift_mlp, scale_mlp, gate_mlp; shifts/scales 0, gates 1).
        modulation = torch.zeros(6, hidden_size)
        modulation[2] = 1.0  # gate_msa
        modulation[5] = 1.0  # gate_mlp
        self.adaLN_modulation = nn.Parameter(modulation)

    def forward(self, x, x_insert, freqs_cis, freqs_cis_insert, pad_mask=None, window_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation

        norm_x = self.norm1(x) * (1 + scale_msa) + shift_msa
        norm_x_insert = self.norm1(x_insert) * (1 + scale_msa) + shift_msa
        norm_x, norm_x_insert = self.attn(norm_x, norm_x_insert, freqs_cis, freqs_cis_insert, pad_mask, window_mask)
        x = x + gate_msa * norm_x
        x_insert = x_insert + gate_msa * norm_x_insert

        norm_x = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(norm_x)

        norm_x_insert = self.norm2(x_insert) * (1 + scale_mlp) + shift_mlp
        x_insert = x_insert + gate_mlp * self.mlp(norm_x_insert)

        return x, x_insert


class DLMEncoderAblation(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            control_vocab_size: int = 5,
            hidden_size: int = 384,
            control_emb_size: int = 24,
            num_layers: int = 2,
            num_heads: int = 4,
            mlp_ratio: float = 4.0,
            max_seq_len: int = 1024,
            dropout: float = 0.1,

            max_insertion: int = 3,
            dim_insert: int = 24,
            directional: bool = True,

            pe_kind: str = "rope2d",

            MaskToken_id: int = 0,
            SOSToken_id: int = 1,
            PadToken_id: int = 2,
            EOSToken_id: int = 3,
            DoneToken_id: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.control_emb_size = control_emb_size
        self.max_seq_len = max_seq_len

        self.max_insertion = max_insertion

        attn_kind, sign_flip = resolve_pe(pe_kind)
        self.pe_kind = pe_kind
        self.attn_kind = attn_kind
        self.sign_flip = sign_flip

        self.token_emb = nn.Embedding(vocab_size, hidden_size - control_emb_size)
        self.control_emb = nn.init.normal_(nn.Parameter(torch.empty(control_vocab_size, control_emb_size)), std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlockAblation(
                hidden_size, num_heads, mlp_ratio,
                directional=directional, max_insertion=max_insertion,
                dropout=dropout, attn_kind=attn_kind, sign_flip=sign_flip,
                dim_insert=dim_insert,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_adaLN_modulation = nn.Parameter(torch.zeros(2, hidden_size))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # Positional-encoding buffers (non-persistent; not in state_dict).
        head_dim = hidden_size // num_heads
        if pe_kind == "rope2d":
            freqs_cis = precompute_freqs_cis_rope2d(
                head_dim, dim_insert, max_seq_len * 2, directional=directional
            )
            freqs_cis_insert = precompute_freqs_cis_rope2d(
                head_dim, dim_insert, max_seq_len * 2,
                directional=directional, max_insertion=max_insertion,
            )
            window_mask = None
        elif pe_kind in ("glpe_v1", "glpe_v1_noinv"):
            freqs_cis = precompute_freqs_cis_glpe_v1_main(
                head_dim, dim_insert, max_seq_len * 2,
                max_insertion=max_insertion, directional=directional,
            )
            freqs_cis_insert = precompute_freqs_cis_glpe_v1_insert(
                head_dim, dim_insert, max_seq_len * 2,
                max_insertion=max_insertion, directional=directional,
            )
            window_mask = precompute_window_mask_glpe_v1(
                max_seq_len * 2, max_insertion
            )
        elif pe_kind == "nope_t":
            freqs_cis = precompute_freqs_cis_nope_t_main(
                head_dim, dim_insert, max_seq_len * 2, directional=directional
            )
            freqs_cis_insert = precompute_freqs_cis_nope_t_insert(
                head_dim, dim_insert, max_seq_len * 2,
                max_insertion=max_insertion, directional=directional,
            )
            window_mask = None
        else:
            raise ValueError(f"Unknown pe_kind={pe_kind!r}")

        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.register_buffer("freqs_cis_insert", freqs_cis_insert, persistent=False)
        if window_mask is not None:
            self.register_buffer("window_mask", window_mask, persistent=False)
        else:
            self.window_mask = None

        self.MaskToken_id = MaskToken_id
        self.SOSToken_id = SOSToken_id
        self.PadToken_id = PadToken_id
        self.EOSToken_id = EOSToken_id
        self.DoneToken_id = DoneToken_id

    def forward(self, input_ids: torch.Tensor, control_ids: torch.Tensor):
        B, S = input_ids.shape
        assert S <= self.max_seq_len, f"Sequence length {S} exceeds max {self.max_seq_len}"

        sos = torch.full((B, 1), self.SOSToken_id, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([sos, input_ids], dim=1)
        control_ids = torch.cat([torch.zeros_like(control_ids[:, :1]), control_ids], dim=1)

        pad_mask = (input_ids != self.PadToken_id)

        eos_pos = pad_mask.sum(dim=1)
        input_ids[torch.arange(B, device=input_ids.device), eos_pos] = self.EOSToken_id
        pad_mask = (input_ids != self.PadToken_id)

        x_vocab = self.token_emb(input_ids)
        x_control = control_ids.float() @ self.control_emb
        x = torch.cat([x_vocab, x_control], dim=-1)

        x_insert_vocab = self.token_emb(self.MaskToken_id * torch.ones_like(input_ids))
        x_insert_control = torch.zeros_like(control_ids).float() @ self.control_emb
        x_insert = torch.cat([x_insert_vocab, x_insert_control], dim=-1)
        x_insert = x_insert.unsqueeze(2).repeat(1, 1, self.max_insertion, 1)

        freqs_cis = self.freqs_cis[:S + 1]
        freqs_cis_insert = self.freqs_cis_insert[:S + 1]
        if self.window_mask is not None:
            window_mask = self.window_mask[:S + 1, :S + 1]
        else:
            window_mask = None

        for block in self.blocks:
            x, x_insert = block(x, x_insert, freqs_cis, freqs_cis_insert, pad_mask, window_mask)

        x = torch.cat([x.unsqueeze(2), x_insert], dim=2)

        shift, scale = self.final_adaLN_modulation
        x = self.final_norm(x) * (1 + scale) + shift

        x = x[:, 1:]

        vocab_logits = self.lm_head(x[..., :-self.control_emb_size])
        control_logit = x[..., -self.control_emb_size:] @ self.control_emb.T

        eos_pos = eos_pos - 1
        done_logit = vocab_logits[:, :, 0][torch.arange(B, device=input_ids.device), eos_pos]

        done_binary_logit = done_logit[:, self.DoneToken_id] - done_logit[:, self.EOSToken_id]

        return vocab_logits, control_logit, done_binary_logit
