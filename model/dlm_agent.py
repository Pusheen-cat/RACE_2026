"""Agent wrapper around the DLM encoder.

`get_action_and_value` extracts the max-depth (visible) tokens from the full
tape, runs the encoder once, and scores a per-step edit action:

  * slot 0 of every visible position — replace the token (MASK = delete),
  * slots 1..max_insertion — insert new tokens after the position,
  * per-position control bits (self-call / result / tag), and
  * one trailing done bit read off the tape's end.

With `action=None` it samples an action from the model's distribution;
with a stored action it evaluates the action's log-probability (this is the
supervised-imitation path: the NLL of the teacher action).
"""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from model.dlm_backbone import DLMEncoder
from model.positional_encodings import PE_DEFAULT


class DLMAgent(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Model shape
        self.model_layer = args.model_layer
        self.model_head = args.model_head
        self.model_hidden = args.model_hidden
        self.model_emb_control = args.model_emb_control
        self.model_dropout = args.model_dropout
        # Vocab / control details
        self.model_vocab = args.model_vocab_size
        self.model_ctl_size = args.model_ctl_size
        self.model_max_insertion = args.model_max_insertion
        self.model_dim_insertion = args.model_dim_insertion
        self.model_directional = args.model_directional

        self.max_seq_len = args.model_max_seq_len

        # Special-token ids
        self.MaskToken_id = args.MaskToken_id  # 0
        self.SOSToken_id = args.SOSToken_id    # 1
        self.PadToken_id = args.PadToken_id    # 2
        self.EOSToken_id = args.EOSToken_id    # 3
        self.DoneToken_id = args.DoneToken_id  # 4

        self.model_pe_kind = getattr(args, "model_pe_kind", PE_DEFAULT)

        self.dlm_encoder = DLMEncoder(
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

            pe_kind=self.model_pe_kind,

            MaskToken_id=self.MaskToken_id,
            SOSToken_id=self.SOSToken_id,
            PadToken_id=self.PadToken_id,
            EOSToken_id=self.EOSToken_id,
            DoneToken_id=self.DoneToken_id,
        )

        self.loss_regularization_stable_vocab = args.loss_regularization_stable_vocab
        self.loss_regularization_NoSpecialToken = args.loss_regularization_NoSpecialToken

    def get_action_and_value(self, x, depth, action=None, done=None):
        """
        x:      (B, S_total, 1 + control_elements) — full tape
        depth:  (B, S_total)
        action: (B, S_eff + 1, 1 + max_insertion, 1 + control_elements) or None
        done:   (B,) or None

        Returns (action, masked_log_prob, reg_loss, elem_log_prob,
                 done_log_prob, final_mask).
        """
        B, S_total, C = x.shape  # C = 1 + control_elements

        # 1. Per-row max depth and the visibility mask (max-depth, non-PAD).
        max_depth = depth.max(dim=1, keepdim=True).values  # (B, 1)
        mask = (depth == max_depth) & (x[..., 0] != self.PadToken_id)  # (B, S_total)
        # 2. Longest visible slice across the batch.
        lengths = mask.sum(dim=1)  # (B,)
        S_max = lengths.max().item()
        # 3. Compact the visible tokens left-aligned into (B, S_max + 1, C).
        x_effective = torch.zeros((B, S_max + 1, C), dtype=x.dtype, device=x.device)
        x_effective[:, :, 0] = self.PadToken_id
        # 4-5. Scatter each row's visible tokens to positions 0..len-1.
        b_indices = torch.where(mask)[0]
        relative_indices = mask.long().cumsum(dim=1)[mask] - 1
        x_effective[b_indices, relative_indices] = x[mask]

        vocab_ids, control_ids = x_effective[..., 0], x_effective[..., 1:]
        padding_mask = (vocab_ids != self.PadToken_id)  # (B, S_max+1)

        vocab_logits, control_logit, done_binary_logit = self.dlm_encoder(vocab_ids, control_ids)
        # vocab_logits:  (B, S, 1 + max_insertion, vocab)
        # control_logit: (B, S, 1 + max_insertion, control_elements)

        vocab_dist = Categorical(logits=vocab_logits)
        control_dist = Bernoulli(logits=control_logit)
        done_dist = Bernoulli(logits=done_binary_logit)

        # Sample an action, or unpack the stored one for evaluation.
        is_sampling = (action is None)
        if is_sampling:
            vocab_action = vocab_dist.sample()      # (B, S, 1 + max_insertion)
            control_action = control_dist.sample()  # (B, S, 1 + max_insertion, control_elements)
            done_action = done_dist.sample()        # (B,)

            vocab_action = torch.where(
                padding_mask.unsqueeze(-1),
                vocab_action,
                torch.full_like(vocab_action, self.PadToken_id)
            )
            action = torch.cat([vocab_action.unsqueeze(-1), control_action], dim=-1).long()
        else:
            vocab_action = action[..., 0].long()      # Categorical wants long
            control_action = action[..., 1:].float()  # Bernoulli wants float
            done_action = done.float()

        # Dynamic selection mask over insertion slots: everything at or after
        # the SECOND MASK in the insertion vector is redundant (tape-equivalent
        # to the first MASK), so only slot 0 plus slots up to and including the
        # first MASK are scored.
        is_mask_token = (vocab_action == self.MaskToken_id)
        is_mask_token_modified = is_mask_token.clone()
        is_mask_token_modified[:, :, 0] = False  # slot 0 (replace) is always valid
        cum_mask = torch.cumsum(is_mask_token_modified.int(), dim=2)
        exclusive_cum_mask = cum_mask - is_mask_token_modified.int()
        valid_insertion_mask = (exclusive_cum_mask == 0)
        final_mask = valid_insertion_mask & padding_mask.unsqueeze(-1).bool()  # (B, S, 1 + max_insertion)

        # Log-probabilities.
        vocab_log_prob = vocab_dist.log_prob(vocab_action)  # (B, S, 1 + max_insertion)
        control_log_prob_per_element = control_dist.log_prob(control_action)  # (B, S, 1+M, ctl)
        control_log_prob = control_log_prob_per_element.sum(dim=-1)  # (B, S, 1 + max_insertion)
        done_log_prob = done_dist.log_prob(done_action)  # (B,)

        # Per-action-element logprob: vocab in index 0, then the ctrl
        # Bernoullis — callers that weight individual bits (e.g. dropping the
        # engine-set (b) bit from the loss) need the unsummed form.
        elem_log_prob = torch.cat(
            [vocab_log_prob.unsqueeze(-1), control_log_prob_per_element], dim=-1
        )  # (B, S, 1 + max_insertion, 1 + control_elements)

        step_log_prob = vocab_log_prob + control_log_prob
        masked_log_prob = (step_log_prob * final_mask).sum(dim=(1, 2)) + done_log_prob  # (B,)

        # After sampling, blank out the unscored positions so the action tensor
        # round-trips through the tape-reconstruction step cleanly.
        if is_sampling:
            action[..., 0, 0].masked_fill_(~final_mask[..., 0], self.PadToken_id)
            action[..., 1:, 0].masked_fill_(~final_mask[..., 1:], self.MaskToken_id)
            action[:, -1, 0, 0] = done_action  # done indicator rides in the last position

        if is_sampling:
            reg_loss = None
        else:
            reg_loss = self.action_regularization(vocab_ids, control_ids, vocab_logits, control_logit, final_mask)

        return (
            action, masked_log_prob, reg_loss,
            elem_log_prob, done_log_prob, final_mask,
        )

    def action_regularization(self, vocab_ids, control_ids, vocab_logits, control_logit, final_mask):
        """Two optional auxiliary losses (both default to weight 0):

        * anchor loss — encourage slot 0 to predict the existing token and
          insertion slots to predict MASK;
        * special-token suppression — push down SOS / PAD / EOS / DONE logits
          at every scored position.
        """
        B, S, I_plus_1, V = vocab_logits.shape
        max_insertion = I_plus_1 - 1
        # Target ids: the current token in slot 0, MASK in the insertion slots.
        mask_tokens = torch.full((B, S, max_insertion), self.MaskToken_id,
                                 dtype=vocab_ids.dtype, device=vocab_ids.device)
        full_target_ids = torch.cat([vocab_ids.unsqueeze(-1), mask_tokens], dim=-1)  # (B, S, I+1)
        # Center the logits over the vocab axis for numerical stability.
        centered_logits = vocab_logits - vocab_logits.mean(dim=-1, keepdim=True)  # (B, S, I+1, V)
        target_logits = torch.gather(centered_logits, dim=-1, index=full_target_ids.unsqueeze(-1)).squeeze(-1)
        reg_loss = -target_logits * final_mask
        total_reg_loss = reg_loss.sum(dim=(1, 2))
        anchor_loss = self.loss_regularization_stable_vocab * total_reg_loss

        # Special-token suppression.
        special_ids = [self.SOSToken_id, self.PadToken_id, self.EOSToken_id, self.DoneToken_id]
        special_logits = centered_logits[..., special_ids]
        special_penalty_map = special_logits.sum(dim=-1) * final_mask
        special_loss = special_penalty_map.sum(dim=(1, 2)) * self.loss_regularization_NoSpecialToken

        return anchor_loss + special_loss  # (B,)
