"""DLMAgent wrapper that swaps in the ablation encoder.

``DLMAgent.__init__`` hard-constructs ``model.dlm_backbone.DLMEncoder``, which
rejects the ablation pe_kinds. We call it with a *base* pe_kind so the parent
initialises normally, then rebuild ``self.dlm_encoder`` as a
``DLMEncoderAblation`` with the requested ablation pe_kind. Every downstream
consumer (``get_action_and_value``, ``.dlm_encoder`` in the greedy loop) is
unchanged; state-dict keys/shapes are identical to the original model, so
checkpoints from the main curriculum load directly.
"""

from __future__ import annotations

from dataclasses import replace

from model.dlm_agent import DLMAgent

from ablation.model.backbone_ablation import (
    ABLATION_PE_KINDS,
    DLMEncoderAblation,
    resolve_pe,
)

# Base (release-code) pe_kind used only for the parent constructor.
_BASE_KIND = {
    "rope2d": "rope2d",
    "glpe_v1": "glpe_v1",
    "glpe_v1_noinv": "glpe_v1",
    "nope_t": "rope2d",
}

# dim_insertion each arm expects (glpe partition needs 48; rope2d asserts 24).
ARM_DIM_INSERTION = {
    "rope2d": 24,
    "glpe_v1": 48,
    "glpe_v1_noinv": 48,
    "nope_t": 24,
}


class DLMAgentAblation(DLMAgent):
    def __init__(self, args):
        pe_kind = getattr(args, "model_pe_kind", "rope2d")
        if pe_kind not in ABLATION_PE_KINDS:
            raise ValueError(
                f"model_pe_kind={pe_kind!r} not in {ABLATION_PE_KINDS}"
            )
        resolve_pe(pe_kind)  # validate early
        base_args = replace(args, model_pe_kind=_BASE_KIND[pe_kind])
        super().__init__(base_args)
        self.args = args
        self.model_pe_kind = pe_kind

        self.dlm_encoder = DLMEncoderAblation(
            vocab_size=args.model_vocab_size,
            control_vocab_size=args.model_ctl_size,
            hidden_size=args.model_hidden,
            control_emb_size=args.model_emb_control,
            num_layers=args.model_layer,
            num_heads=args.model_head,
            mlp_ratio=4.0,
            max_seq_len=args.model_max_seq_len,
            dropout=args.model_dropout,
            max_insertion=args.model_max_insertion,
            dim_insert=args.model_dim_insertion,
            directional=args.model_directional,
            pe_kind=pe_kind,
            MaskToken_id=self.MaskToken_id,
            SOSToken_id=self.SOSToken_id,
            PadToken_id=self.PadToken_id,
            EOSToken_id=self.EOSToken_id,
            DoneToken_id=self.DoneToken_id,
        )
