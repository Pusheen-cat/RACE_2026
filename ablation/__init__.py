"""Ablation study of the model's three signature features.

Self-contained analysis package: it imports the release's `model/`,
`teacher/`, `dataset/` and `supervised/` packages but never modifies them.
Six experimental settings (arms) ablate the positional encoding, the
control elements, and the tape-editing semantics; see README.md in this
directory for the full protocol.

Layout:
  model/    encoder/agent variants with the extra PE arms + sign-flip switch
  lib/      arm-specific data generation, engines, trainer, greedy loop,
            frontier evaluator
  train/    per-stage training entry points (copy / add / mult per arm)
  eval/     per-task evaluation entry points (copy / add / mult per arm)
  verify/   correctness gates for the three ablation mechanisms
  scripts/  train_arm.sh / eval_arm.sh — one-command run per arm
"""
