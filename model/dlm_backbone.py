"""DLM encoder backbone.

A bidirectional Transformer over a token tape plus, at every position,
`max_insertion` insertion slots that are processed in parallel through every
block. Attention is directional: the head dim is split into a forward and a
backward half, each RoPE-rotated, and the backward half's scores are sign-
flipped for j > i before the two halves are summed.

The architecture descends from a time-conditioned discrete diffusion model
whose AdaLN modulation was driven by a timestep embedding. The timestep was
fixed to 0 throughout this work, which makes the whole time branch compute a
constant; it is therefore removed here. Each block's six AdaLN modulation
vectors (shift/scale/gate for attention and MLP) and the final norm's
shift/scale are kept as directly learnable parameters — exactly the values
the original network could express at t = 0 — so the encoder computation is
preserved while the auxiliary time branch's parameters are eliminated.
"""

import math
import torch
import torch.nn as nn

from model.positional_encodings import (
    PE_DEFAULT,
    PE_KINDS,
    apply_rotary_emb,
    apply_rotary_emb_2d,
    glpe_v1_d_global_per_dir,
    precompute_freqs_cis_glpe_v1_insert,
    precompute_freqs_cis_glpe_v1_main,
    precompute_freqs_cis_rope2d,
    precompute_window_mask_glpe_v1,
)


# ---------------------------------------------------------------------------
# Transformer modules
# ---------------------------------------------------------------------------
class AttentionWithRoPE(nn.Module):
    def __init__(self, hidden_size, num_heads, directional=False, max_insertion=None, dropout=0.0,
                 pe_kind: str = PE_DEFAULT, dim_insert: int = None):
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

        if pe_kind not in PE_KINDS:
            raise ValueError(f"Unknown pe_kind={pe_kind!r}; expected one of {PE_KINDS}.")
        self.pe_kind = pe_kind
        # Per-direction global block width (in dims) for GLPE_v1's score split.
        # Computed only when needed; left at None for RoPE2d so the existing path
        # is unaffected.
        if pe_kind == "glpe_v1":
            if dim_insert is None:
                raise ValueError("AttentionWithRoPE requires dim_insert when pe_kind='glpe_v1'.")
            self.glpe_v1_D_global = glpe_v1_d_global_per_dir(
                self.head_dim, dim_insert, directional
            )
        else:
            self.glpe_v1_D_global = None

    def forward(self, x, x_insert, freqs_cis, freqs_cis_insert, pad_mask=None, window_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each (B, T, num_heads, head_dim)

        # x_insert: (B, T, max_insertion, C)
        qkv_insert = self.qkv(x_insert).reshape(B, T, self.max_insertion, 3, self.num_heads, self.head_dim)
        q_insert, k_insert, v_insert = qkv_insert.unbind(3)  # each (B, T, max_insertion, num_heads, head_dim)

        if self.directional:
            q_front, q_back = torch.chunk(q, 2, dim=-1)
            k_front, k_back = torch.chunk(k, 2, dim=-1)
            q_front, k_front = apply_rotary_emb(q_front, k_front, freqs_cis)
            q_back, k_back = apply_rotary_emb(q_back, k_back, freqs_cis)

            q_front = q_front.transpose(1, 2)  # (B, H, T, D)
            q_back = q_back.transpose(1, 2)
            k_front = k_front.transpose(1, 2)
            k_back = k_back.transpose(1, 2)
            v = v.transpose(1, 2)

            scale = math.sqrt(self.head_dim)
            if self.pe_kind == "glpe_v1":
                # Split-and-mask score: global Q·K (regions A+B, unrotated on T)
                # plus a window-masked local Q·K (regions C+D, RoPE applied).
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

            # Sign-flip the backward-direction scores at causal positions (j > i).
            causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            sign = torch.ones(T, T, device=x.device)
            sign[causal] = -1
            att_back = att_back * sign

            att = att_front + att_back

            # Padding mask on the main attention: key positions that are PAD
            # get a -inf score.
            if pad_mask is not None:
                # pad_mask: (B, T) -> (B, 1, 1, T) masks key positions
                att = att.masked_fill(~pad_mask.view(B, 1, 1, T), float('-inf'))

            att = att.softmax(dim=-1)
            att = self.dropout(att)

            # att: (B, H, T, T); v: (B, H, T, D) -> attn_out: (B, H, T, D)
            attn_out = att @ v

            # Insertion-slot attention.
            q_insert_front, q_insert_back = torch.chunk(q_insert, 2, dim=-1)
            k_insert_front, k_insert_back = torch.chunk(k_insert, 2, dim=-1)
            q_insert_front, k_insert_front = apply_rotary_emb_2d(q_insert_front, k_insert_front, freqs_cis_insert)
            q_insert_back, k_insert_back = apply_rotary_emb_2d(q_insert_back, k_insert_back, freqs_cis_insert)

            q_insert_front = q_insert_front.transpose(1, 3)  # (B, T, max_insertion, H, D) -> (B, H, max_insertion, T, D)
            q_insert_back = q_insert_back.transpose(1, 3)
            k_insert_front = k_insert_front.transpose(1, 3)
            k_insert_back = k_insert_back.transpose(1, 3)
            v_insert = v_insert.transpose(1, 3)

            if self.pe_kind == "glpe_v1":
                # Same split-and-mask, applied to insert→main cross attention.
                # The self-score path below stays unsplit because |Δt|=0 always
                # sits inside the window.
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
                # Keys come from the MAIN tokens, not the insertion slots.
                att_insert_front = (q_insert_front @ k_front.unsqueeze(2).transpose(-2, -1)) / scale
                att_insert_back = (q_insert_back @ k_back.unsqueeze(2).transpose(-2, -1)) / scale  # (B, H, max_insertion, T, T)

            # Self-attention of each insertion slot to its own position.
            # q_insert_*, k_insert_*: (B, H, max_insertion, T, D)
            score_front = (q_insert_front * k_insert_front).sum(dim=-1) / math.sqrt(self.head_dim)  # (B, H, max_insertion, T)
            score_back = (q_insert_back * k_insert_back).sum(dim=-1) / math.sqrt(self.head_dim)
            score = score_front + score_back  # (B, H, max_insertion, T)

            att_insert_back = att_insert_back * sign

            att_insert = att_insert_front + att_insert_back  # (B, H, max_insertion, T, T)

            score_expanded = score.unsqueeze(-1)  # (B, H, max_insertion, T, 1)
            att_insert = torch.cat([att_insert, score_expanded], dim=-1)  # (B, H, max_insertion, T, T+1)

            # Padding mask on the insert attention.
            if pad_mask is not None:
                # Mask over the T context keys.
                mask_t = pad_mask.view(B, 1, 1, 1, T)
                # The appended self-score column is always valid.
                mask_self = torch.ones((B, 1, 1, 1, 1), dtype=torch.bool, device=x.device)
                mask_full = torch.cat([mask_t, mask_self], dim=-1)  # (B, 1, 1, 1, T+1)
                att_insert = att_insert.masked_fill(~mask_full, float('-inf'))

            att_insert = att_insert.softmax(dim=-1)  # (B, H, max_insertion, T, T+1)
            att_insert = self.dropout(att_insert)
            score = att_insert[..., -1]       # (B, H, max_insertion, T)
            att_insert = att_insert[..., :-1]  # (B, H, max_insertion, T, T)

            # Context part attends to main v; self part uses v_insert.
            attn_insert_out = att_insert @ v.unsqueeze(2) + score.unsqueeze(-1) * v_insert  # (B, H, max_insertion, T, D)

        else:
            raise NotImplementedError("directional=False is not implemented")

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_insert_out = attn_insert_out.transpose(1, 3).contiguous().view(B, T, self.max_insertion, C)
        return self.proj(attn_out), self.proj(attn_insert_out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, directional=False, max_insertion=None, dropout=0.0,
                 pe_kind: str = PE_DEFAULT, dim_insert: int = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = AttentionWithRoPE(
            hidden_size, num_heads,
            directional=directional, max_insertion=max_insertion,
            dropout=dropout, pe_kind=pe_kind, dim_insert=dim_insert,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

        # Constant AdaLN modulation. Row order: shift_msa, scale_msa, gate_msa,
        # shift_mlp, scale_mlp, gate_mlp. In the time-conditioned ancestor these
        # six vectors were the output of a SiLU+Linear head over the timestep
        # embedding; with the timestep fixed to 0 that head computes a constant,
        # so the vectors are learned directly. Init: shifts/scales 0, gates 1
        # (identity modulation — a plain pre-LN Transformer block).
        modulation = torch.zeros(6, hidden_size)
        modulation[2] = 1.0  # gate_msa
        modulation[5] = 1.0  # gate_mlp
        self.adaLN_modulation = nn.Parameter(modulation)

    def forward(self, x, x_insert, freqs_cis, freqs_cis_insert, pad_mask=None, window_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation

        # Attention with AdaLN. The (hidden,) vectors broadcast over both the
        # (B, T, C) main path and the (B, T, M, C) insertion path.
        norm_x = self.norm1(x) * (1 + scale_msa) + shift_msa
        norm_x_insert = self.norm1(x_insert) * (1 + scale_msa) + shift_msa
        norm_x, norm_x_insert = self.attn(norm_x, norm_x_insert, freqs_cis, freqs_cis_insert, pad_mask, window_mask)
        x = x + gate_msa * norm_x
        x_insert = x_insert + gate_msa * norm_x_insert

        # MLP with AdaLN.
        norm_x = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(norm_x)

        norm_x_insert = self.norm2(x_insert) * (1 + scale_mlp) + shift_mlp
        x_insert = x_insert + gate_mlp * self.mlp(norm_x_insert)

        return x, x_insert


# ---------------------------------------------------------------------------
# Main model: DLM backbone
# ---------------------------------------------------------------------------
class DLMEncoder(nn.Module):
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

            pe_kind: str = PE_DEFAULT,

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

        if pe_kind not in PE_KINDS:
            raise ValueError(f"Unknown pe_kind={pe_kind!r}; expected one of {PE_KINDS}.")
        self.pe_kind = pe_kind

        # Token embedding: the hidden vector is the concatenation of a vocab
        # embedding and a (multi-hot) control-bit embedding.
        self.token_emb = nn.Embedding(vocab_size, hidden_size - control_emb_size)
        self.control_emb = nn.init.normal_(nn.Parameter(torch.empty(control_vocab_size, control_emb_size)), std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size, num_heads, mlp_ratio,
                directional=directional, max_insertion=max_insertion,
                dropout=dropout, pe_kind=pe_kind, dim_insert=dim_insert,
            )
            for _ in range(num_layers)
        ])

        # Final LayerNorm with constant AdaLN modulation (rows: shift, scale),
        # then the LM head. See TransformerBlock.adaLN_modulation for why the
        # modulation is a direct parameter.
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_adaLN_modulation = nn.Parameter(torch.zeros(2, hidden_size))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying between the token embedding and the LM head.
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
        elif pe_kind == "glpe_v1":
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
        else:
            raise ValueError(f"Unknown pe_kind={pe_kind!r}; expected one of {PE_KINDS}.")

        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.register_buffer("freqs_cis_insert", freqs_cis_insert, persistent=False)
        if window_mask is not None:
            self.register_buffer("window_mask", window_mask, persistent=False)
        else:
            # Keep the attribute defined so forward() can read it uniformly.
            self.window_mask = None

        # Token ids
        self.MaskToken_id = MaskToken_id
        self.SOSToken_id = SOSToken_id
        self.PadToken_id = PadToken_id
        self.EOSToken_id = EOSToken_id
        self.DoneToken_id = DoneToken_id

    def forward(self, input_ids: torch.Tensor, control_ids: torch.Tensor):
        """
        Args:
            input_ids: (Batch, Seq_Len) integer token IDs
            control_ids: (Batch, Seq_Len, control_vocab_size) bool tensor
        Returns:
            vocab_logits: (Batch, Seq_Len, 1 + max_insertion, vocab)
            control_logit: (Batch, Seq_Len, 1 + max_insertion, control_vocab)
            done_binary_logit: (Batch,)
        """
        B, S = input_ids.shape
        assert S <= self.max_seq_len, f"Sequence length {S} exceeds max {self.max_seq_len}"

        # Insert SOS at the front.
        sos = torch.full((B, 1), self.SOSToken_id, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([sos, input_ids], dim=1)  # (B, S+1)
        control_ids = torch.cat([torch.zeros_like(control_ids[:, :1]), control_ids], dim=1)

        # Padding mask: True where the position participates in attention.
        pad_mask = (input_ids != self.PadToken_id)  # (B, S+1)

        # Write EOS into the first PAD position of every row (end of tape).
        eos_pos = pad_mask.sum(dim=1)  # (B,) first PAD position
        input_ids[torch.arange(B, device=input_ids.device), eos_pos] = self.EOSToken_id  # (B, S+1)
        pad_mask = (input_ids != self.PadToken_id)  # (B, S+1)

        # 1. Embedding
        x_vocab = self.token_emb(input_ids)  # (B, S+1, vocab_embed_dim)
        x_control = control_ids.float() @ self.control_emb  # (B, S+1, control_embed_dim)
        x = torch.cat([x_vocab, x_control], dim=-1)

        # Insertion slots start as MASK tokens with no control bits.
        x_insert_vocab = self.token_emb(self.MaskToken_id * torch.ones_like(input_ids))  # (B, S+1, vocab_embed_dim)
        x_insert_control = torch.zeros_like(control_ids).float() @ self.control_emb  # (B, S+1, control_embed_dim)
        x_insert = torch.cat([x_insert_vocab, x_insert_control], dim=-1)  # (B, S+1, C)
        x_insert = x_insert.unsqueeze(2).repeat(1, 1, self.max_insertion, 1)  # (B, S+1, max_insertion, C)

        # 2. Slice the PE tables to the current sequence length.
        freqs_cis = self.freqs_cis[:S + 1]
        freqs_cis_insert = self.freqs_cis_insert[:S + 1]
        if self.window_mask is not None:
            window_mask = self.window_mask[:S + 1, :S + 1]
        else:
            window_mask = None

        # 3. Encoder blocks
        for block in self.blocks:
            x, x_insert = block(x, x_insert, freqs_cis, freqs_cis_insert, pad_mask, window_mask)

        # Stack the main path and the insertion slots along a new axis.
        x = torch.cat([x.unsqueeze(2), x_insert], dim=2)  # (B, S+1, 1+max_insertion, C)

        # 4. Final AdaLN
        shift, scale = self.final_adaLN_modulation
        x = self.final_norm(x) * (1 + scale) + shift

        # Drop the SOS position.
        x = x[:, 1:]

        # LM head: vocab logits from the vocab part of the hidden vector,
        # control logits from the control part against the control embedding.
        vocab_logits = self.lm_head(x[..., :-self.control_emb_size])  # (B, S, 1+max_insert, vocab)
        control_logit = x[..., -self.control_emb_size:] @ self.control_emb.T  # (B, S, 1+max_insert, control_vocab)

        # Done probability is read off the EOS position: the binary done logit
        # is logit(DONE) - logit(EOS) in slot 0 at the tape's end.
        eos_pos = eos_pos - 1  # account for the dropped SOS position
        done_logit = vocab_logits[:, :, 0][torch.arange(B, device=input_ids.device), eos_pos]  # (B, vocab)
        done_binary_logit = done_logit[:, self.DoneToken_id] - done_logit[:, self.EOSToken_id]

        return vocab_logits, control_logit, done_binary_logit
