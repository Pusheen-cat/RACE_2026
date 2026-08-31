"""Gate for the ablation PE variants (exp 1).

Checks:
  1. DLMEncoderAblation(pe_kind="glpe_v1") is numerically IDENTICAL to the
     release DLMEncoder(pe_kind="glpe_v1") given the same weights (guards
     against copy drift). Same for "rope2d".
  2. "glpe_v1_noinv" differs from "glpe_v1" (the flip matters) while its PE
     buffers are identical to glpe_v1's.
  3. "nope_t" tables: main table identity everywhere; insert table identity on
     the T-part, equal to rope2d's insertion-axis part on the low dims; the
     insert table is constant in T.
  4. If a curriculum checkpoint exists (default checkpoints/copy.pt, override
     with --ckpt), DLMAgentAblation loads it with zero missing/unexpected
     keys for every arm.
  5. One real mult training batch forwards through get_action_and_value for
     every arm (finite loss).

Run:  python -m ablation.verify.verify_pe_variants
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from model.dlm_backbone import DLMEncoder

from ablation.model.agent_ablation import ARM_DIM_INSERTION, DLMAgentAblation
from ablation.model.backbone_ablation import DLMEncoderAblation

FAILED = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not ok:
        FAILED.append(name)


def _mk_kwargs(dim_insert):
    return dict(
        vocab_size=32, control_vocab_size=3, hidden_size=384,
        control_emb_size=24, num_layers=2, num_heads=4, mlp_ratio=4.0,
        max_seq_len=256, dropout=0.0, max_insertion=3,
        dim_insert=dim_insert, directional=True,
    )


def _rand_inputs(B=4, S=37, ctl=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    vocab = torch.randint(5, 20, (B, S), generator=g)
    vocab[:, -1] = 2  # every row needs >=1 PAD (EOS is written at first PAD)
    vocab[0, -5:] = 2  # one row with a longer PAD tail
    control = (torch.rand(B, S, ctl, generator=g) < 0.15).float()
    return vocab, control


def equivalence_check(pe_kind):
    torch.manual_seed(0)
    dim_ins = 48 if pe_kind == "glpe_v1" else 24
    ref = DLMEncoder(pe_kind=pe_kind, **_mk_kwargs(dim_ins)).eval()
    abl = DLMEncoderAblation(pe_kind=pe_kind, **_mk_kwargs(dim_ins)).eval()
    abl.load_state_dict(ref.state_dict(), strict=True)
    vocab, control = _rand_inputs()
    with torch.no_grad():
        v1, c1, d1 = ref(vocab.clone(), control.clone())
        v2, c2, d2 = abl(vocab.clone(), control.clone())
    same = (torch.equal(v1, v2) and torch.equal(c1, c2) and torch.equal(d1, d2))
    check(f"equivalence[{pe_kind}]", same,
          f"max|dv|={float((v1 - v2).abs().max()):.2e}")


def noinv_check():
    torch.manual_seed(0)
    base = DLMEncoderAblation(pe_kind="glpe_v1", **_mk_kwargs(48)).eval()
    noinv = DLMEncoderAblation(pe_kind="glpe_v1_noinv", **_mk_kwargs(48)).eval()
    noinv.load_state_dict(base.state_dict(), strict=True)
    check("noinv buffers == glpe_v1 buffers",
          torch.equal(base.freqs_cis, noinv.freqs_cis)
          and torch.equal(base.freqs_cis_insert, noinv.freqs_cis_insert)
          and torch.equal(base.window_mask, noinv.window_mask))
    vocab, control = _rand_inputs()
    with torch.no_grad():
        v1, _, _ = base(vocab.clone(), control.clone())
        v2, _, _ = noinv(vocab.clone(), control.clone())
    check("noinv output differs from glpe_v1", not torch.equal(v1, v2),
          f"max|dv|={float((v1 - v2).abs().max()):.2e}")
    check("noinv sign_flip flags", noinv.sign_flip is False and base.sign_flip is True)


def nope_check():
    enc = DLMEncoderAblation(pe_kind="nope_t", **_mk_kwargs(24)).eval()
    rope = DLMEncoderAblation(pe_kind="rope2d", **_mk_kwargs(24)).eval()
    ones = torch.ones_like(enc.freqs_cis.real)
    check("nope_t main table = identity",
          torch.allclose(enc.freqs_cis.real, ones)
          and torch.allclose(enc.freqs_cis.imag, torch.zeros_like(ones)))
    # insert table: T-part identity, low part == rope2d's low part
    dim_low = 24 // 2 // 2  # per-dir dim_insert / 2 pairs = 6
    ins = enc.freqs_cis_insert
    check("nope_t insert T-part = identity",
          torch.allclose(ins[..., :-dim_low].real, torch.ones_like(ins[..., :-dim_low].real))
          and torch.allclose(ins[..., :-dim_low].imag, torch.zeros_like(ins[..., :-dim_low].imag)))
    check("nope_t insert low-part == rope2d low-part",
          torch.allclose(torch.view_as_real(ins[..., -dim_low:]),
                         torch.view_as_real(rope.freqs_cis_insert[..., -dim_low:])))
    check("nope_t insert constant in T",
          torch.equal(ins[0], ins[100]) and torch.equal(ins[1], ins[200]))
    check("nope_t window mask is None", enc.window_mask is None)
    check("nope_t sign_flip kept", enc.sign_flip is True)


def ckpt_and_forward_check(ckpt_path: str):
    from supervised.lib import data_gen
    from supervised.lib.dataset import collate_steps
    from supervised.lib.sup_args import SupArgs

    tg = data_gen.make_task_generator()
    _, _, steps = data_gen.gen_binary_op_task(tg, 3, 3, "*")
    batch = steps[:16]
    states, depths, actions, dones = collate_steps(batch, pad_id=2)

    sd = None
    if os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
    else:
        print(f"[skip] checkpoint-load check: no checkpoint at {ckpt_path} "
              f"(run the main curriculum's copy stage first, or pass --ckpt)")

    for arm in ("glpe_v1", "glpe_v1_noinv", "nope_t", "rope2d"):
        args = SupArgs(model_pe_kind=arm,
                       model_dim_insertion=ARM_DIM_INSERTION[arm])
        agent = DLMAgentAblation(args).eval()
        if sd is not None:
            missing, unexpected = agent.load_state_dict(sd, strict=False)
            check(f"ckpt load[{arm}] no missing/unexpected",
                  not missing and not unexpected,
                  f"missing={missing[:2]} unexpected={unexpected[:2]}"
                  if (missing or unexpected) else "")
        with torch.no_grad():
            result = agent.get_action_and_value(states, depths, actions, dones)
        mlp = result[1]
        check(f"forward[{arm}] finite log_prob", bool(torch.isfinite(mlp).all()),
              f"mean_lp={float(mlp.mean()):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/copy.pt",
                   help="curriculum checkpoint for the load check "
                        "(skipped with a note if the file is missing)")
    args = p.parse_args()

    equivalence_check("glpe_v1")
    equivalence_check("rope2d")
    noinv_check()
    nope_check()
    ckpt_and_forward_check(args.ckpt)
    print()
    if FAILED:
        print(f"FAILED: {FAILED}")
        sys.exit(1)
    print("ALL PE-VARIANT CHECKS PASSED")


if __name__ == "__main__":
    main()
