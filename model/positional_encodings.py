"""Positional-encoding strategies for ``DLMEncoder``.

Two PE families are exposed:

- ``rope2d`` — the original 2D RoPE used since the project started.
  1D RoPE on the T-axis (``theta=2000``) for the first ``full_dim -
  dim_insert`` pairs per direction, plus an insertion-axis RoPE
  (``theta_insert = 2*(max_insertion+1)``) on the remaining
  ``dim_insert`` pairs. ``apply_rotary_emb`` / ``apply_rotary_emb_2d``
  are the runtime kernels.

- ``glpe_v1`` — Global/Local PE with NoPE on long-range "global"
  channels and a local RoPE plus an explicit T-axis attention window
  on "local" channels. The per-direction head is partitioned into
  four regions::

      | region | width                    | T-rot               | I-rot               | window  |
      | A      | full - 3*i_per/2         | none                | none                | none    |
      | B      | i_per/2                  | none                | RoPE on i           | none    |
      | C      | i_per/2                  | local RoPE on t     | none                | |Δt|≤L  |
      | D      | i_per/2                  | local RoPE on t+i*  | local RoPE on t+i*  | |Δt|≤L  |

  ``i_per`` is the post-directional-halving ``dim_insert`` and ``L =
  max(4, max_insertion+1)``. Region D applies a single composed
  rotation ``R((t+i)·f)`` to one dim pair (the (t, i) diagonal
  collision is intentional — regions B and C provide independent Δt
  and Δi handles).

  Periods are the bottom ``i_per/4`` entries of the doubling sequence
  ``{1, 2, 4, …, 8·L}``. Period 1 yields rotation by ``2π`` per step,
  which is identity for integer offsets — kept intentionally as a
  content-only channel inside the local window (it differs from
  region A only by being window-masked).

  When ``pe_kind="glpe_v1"`` the attention forward splits the score
  into a global Q·K (regions A+B, unmasked) and a local Q·K (regions
  C+D, multiplied by the ``|Δt| ≤ L`` window mask), sums them, then
  softmaxes. The back-direction ``sign`` flip on ``j > i`` applies to
  the summed score, exactly as in ``rope2d``.

Both families share the same ``apply_rotary_emb`` /
``apply_rotary_emb_2d`` runtime kernels — only the precomputed
``freqs_cis`` content differs. GLPE_v1 additionally requires a
``(T_max, T_max)`` boolean window-mask buffer, surfaced by
``precompute_window_mask_glpe_v1``.

Constraints (asserted at construction time for GLPE_v1, pre-halving
values):

- ``directional=True`` (the only branch implemented anywhere in this
  codebase).
- ``head_dim >= 96`` and ``dim_insert >= 48`` so region A is
  non-negative — region A width per direction is
  ``head_dim/2 - 3·dim_insert/4``.
- ``dim_insert % 8 == 0``.
"""
import math
from typing import Tuple

import torch


PE_KINDS = ("rope2d", "glpe_v1")
PE_DEFAULT = "rope2d"


# ---------------------------------------------------------------------------
# Runtime rotation kernels (shared by RoPE2d and GLPE_v1)
# ---------------------------------------------------------------------------
def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(xq, xk, freqs_cis):
    """1D RoPE.

    ``xq``/``xk``: ``(B, T, H, D)``. ``freqs_cis``: ``(T, rope_dim/2)``
    complex. If ``rope_dim/2 < D/2`` the missing pairs are padded with
    identity rotations.
    """
    B, T, H, D = xq.shape
    half_dim = D // 2
    if freqs_cis.shape[-1] < half_dim:
        pad = torch.ones(
            freqs_cis.shape[0],
            half_dim - freqs_cis.shape[-1],
            dtype=freqs_cis.dtype,
            device=freqs_cis.device,
        )
        freqs_cis = torch.cat([freqs_cis, pad], dim=-1)
    cos = freqs_cis.real.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D/2)
    sin = freqs_cis.imag.unsqueeze(0).unsqueeze(2)
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out


def apply_rotary_emb_2d(xq, xk, freqs_cis):
    """2D RoPE over (T, max_insertion).

    ``xq``/``xk``: ``(B, T, M, H, D)``. ``freqs_cis``:
    ``(T, M, rope_dim/2)`` complex; ``rope_dim/2`` must equal ``D/2``.
    """
    cos = freqs_cis.real.unsqueeze(0).unsqueeze(3)  # (1, T, M, 1, D/2)
    sin = freqs_cis.imag.unsqueeze(0).unsqueeze(3)
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out


# ---------------------------------------------------------------------------
# RoPE2d — original 2D RoPE (kept behaviorally identical).
# ---------------------------------------------------------------------------
def precompute_freqs_cis_rope2d(
    full_dim: int,
    dim_insert: int,
    end: int,
    theta: float = 2000.0,
    directional: bool = False,
    max_insertion: int = None,
):
    """Original 2D RoPE precompute.

    ``full_dim`` = head_dim (pre-halving); ``dim_insert`` is also
    pre-halving. When ``directional=True`` both are halved internally
    so the returned table covers one direction.

    If ``max_insertion`` is ``None`` returns ``(end, full_dim/2)``
    complex for the main stream; otherwise returns
    ``(end, max_insertion, full_dim/2)`` complex for the insert stream.
    """
    if directional:
        dim_insert = dim_insert // 2
        full_dim = full_dim // 2
        assert dim_insert % 4 == 0
    assert dim_insert % 2 == 0
    assert full_dim % 2 == 0

    dim = full_dim - dim_insert

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # (T, dim/2) complex

    if max_insertion:
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
        freqs_cis_low = torch.polar(torch.ones_like(freqs_low), freqs_low)

        freqs_cis = freqs_cis.unsqueeze(1).expand(end, max_insertion, -1)
        freqs_cis_low = freqs_cis_low.unsqueeze(0).expand(end, max_insertion, -1)
        freqs_cis = torch.cat([freqs_cis, freqs_cis_low], dim=-1)
    else:
        pad_len = dim_insert // 2
        pad = torch.ones((end, pad_len), dtype=freqs_cis.dtype, device=freqs_cis.device)
        freqs_cis = torch.cat([freqs_cis, pad], dim=-1)

    return freqs_cis


# Backward-compatible alias for the historical name.
precompute_freqs_cis = precompute_freqs_cis_rope2d


# ---------------------------------------------------------------------------
# GLPE_v1 — Global/Local PE with explicit T-axis window mask.
# ---------------------------------------------------------------------------
def glpe_v1_local_len(max_insertion: int) -> int:
    return max(4, max_insertion + 1)


def _glpe_v1_period_list(local_len: int, n_pairs: int):
    """Bottom ``n_pairs`` periods from the doubling sequence
    ``{1, 2, 4, …, 8·local_len}``."""
    max_period = 8 * local_len
    periods = []
    p = 1
    while p <= max_period:
        periods.append(p)
        p *= 2
    if len(periods) < n_pairs:
        raise ValueError(
            f"GLPE_v1: only {len(periods)} candidate periods up to "
            f"8*local_len={max_period}, but {n_pairs} pairs need a frequency."
        )
    return periods[:n_pairs]


def _glpe_v1_local_freqs(local_len: int, n_pairs: int) -> torch.Tensor:
    periods = _glpe_v1_period_list(local_len, n_pairs)
    return torch.tensor(
        [2.0 * math.pi / p for p in periods], dtype=torch.float32
    )


def _glpe_v1_partition_per_dir(
    full_dim_per_dir: int, dim_insert_per_dir: int
) -> Tuple[int, int, int, int]:
    """Region widths (in dims) per direction: ``(A, B, C, D)``."""
    n_B = dim_insert_per_dir // 2
    n_C = dim_insert_per_dir // 2
    n_D = dim_insert_per_dir // 2
    n_A = full_dim_per_dir - n_B - n_C - n_D
    if n_A < 0:
        raise ValueError(
            f"GLPE_v1: region A would be negative — got "
            f"full_dim_per_dir={full_dim_per_dir}, "
            f"dim_insert_per_dir={dim_insert_per_dir}. Need "
            f"full_dim_per_dir >= 3·dim_insert_per_dir/2."
        )
    return n_A, n_B, n_C, n_D


def glpe_v1_d_global_per_dir(
    head_dim: int, dim_insert: int, directional: bool
) -> int:
    """Width (in dims) of the global block per direction in q/k front
    and back tensors. Used by the attention forward to split scores."""
    _assert_glpe_v1_sizes(head_dim, dim_insert)
    if directional:
        full_dim_per_dir = head_dim // 2
        dim_insert_per_dir = dim_insert // 2
    else:
        full_dim_per_dir = head_dim
        dim_insert_per_dir = dim_insert
    n_A, n_B, _, _ = _glpe_v1_partition_per_dir(
        full_dim_per_dir, dim_insert_per_dir
    )
    return n_A + n_B  # = full_dim_per_dir - dim_insert_per_dir


def _assert_glpe_v1_sizes(head_dim: int, dim_insert: int):
    """Pre-halving size checks."""
    if head_dim < 96:
        raise ValueError(
            f"GLPE_v1 requires head_dim >= 96 (got {head_dim}). With "
            f"hidden_size=384 and num_heads=4, head_dim=96 — keep that "
            f"configuration or scale up."
        )
    if dim_insert < 48:
        raise ValueError(
            f"GLPE_v1 requires dim_insert >= 48 (got {dim_insert}). "
            f"Bump model_dim_insertion."
        )
    if dim_insert % 8 != 0:
        raise ValueError(
            f"GLPE_v1 requires dim_insert % 8 == 0 (got {dim_insert})."
        )


def precompute_freqs_cis_glpe_v1_main(
    head_dim: int,
    dim_insert: int,
    end: int,
    max_insertion: int,
    directional: bool,
) -> torch.Tensor:
    """Main-stream rotation table for GLPE_v1.

    Returns ``(end, full_dim_per_dir / 2)`` complex. Per-pair angles
    (main tokens are at insertion index 0, so regions B and D collapse
    to identity / pure-t rotation)::

        region A: 0       region B: 0
        region C: t·f     region D: t·f
    """
    if not directional:
        raise NotImplementedError("GLPE_v1 only supports directional=True")
    _assert_glpe_v1_sizes(head_dim, dim_insert)

    full_dim_per_dir = head_dim // 2
    dim_insert_per_dir = dim_insert // 2
    n_A, n_B, n_C, n_D = _glpe_v1_partition_per_dir(
        full_dim_per_dir, dim_insert_per_dir
    )
    n_p_A, n_p_B = n_A // 2, n_B // 2
    n_p_C, n_p_D = n_C // 2, n_D // 2
    assert n_p_C == n_p_D

    local_len = glpe_v1_local_len(max_insertion)
    freqs_local = _glpe_v1_local_freqs(local_len, n_p_C)  # (n_p_C,)

    t = torch.arange(end, dtype=torch.float32)

    angles_AB = torch.zeros((end, n_p_A + n_p_B), dtype=torch.float32)
    angles_C = torch.outer(t, freqs_local)               # (T, n_p_C)
    angles_D = torch.outer(t, freqs_local)               # (T, n_p_D)  same as C for i=0
    angles = torch.cat([angles_AB, angles_C, angles_D], dim=-1)
    return torch.polar(torch.ones_like(angles), angles)


def precompute_freqs_cis_glpe_v1_insert(
    head_dim: int,
    dim_insert: int,
    end: int,
    max_insertion: int,
    directional: bool,
) -> torch.Tensor:
    """Insert-stream rotation table for GLPE_v1.

    Returns ``(end, max_insertion, full_dim_per_dir / 2)`` complex.
    Per-pair angles indexed by insert slot ``i ∈ {1..max_insertion}``::

        region A: 0          region B: i·f
        region C: t·f        region D: (t+i)·f
    """
    if not directional:
        raise NotImplementedError("GLPE_v1 only supports directional=True")
    _assert_glpe_v1_sizes(head_dim, dim_insert)

    full_dim_per_dir = head_dim // 2
    dim_insert_per_dir = dim_insert // 2
    n_A, n_B, n_C, n_D = _glpe_v1_partition_per_dir(
        full_dim_per_dir, dim_insert_per_dir
    )
    n_p_A, n_p_B = n_A // 2, n_B // 2
    n_p_C, n_p_D = n_C // 2, n_D // 2
    assert n_p_B == n_p_C == n_p_D

    local_len = glpe_v1_local_len(max_insertion)
    freqs_local = _glpe_v1_local_freqs(local_len, n_p_C)  # (n_p_C,)

    t = torch.arange(end, dtype=torch.float32)
    i_idx = torch.arange(1, max_insertion + 1, dtype=torch.float32)  # (M,)

    angles_A = torch.zeros(
        (end, max_insertion, n_p_A), dtype=torch.float32
    )
    angles_B = torch.outer(i_idx, freqs_local)                  # (M, n_p_B)
    angles_B = angles_B.unsqueeze(0).expand(end, -1, -1)        # (T, M, n_p_B)
    angles_C = torch.outer(t, freqs_local)                      # (T, n_p_C)
    angles_C = angles_C.unsqueeze(1).expand(-1, max_insertion, -1)
    t_plus_i = t.unsqueeze(1) + i_idx.unsqueeze(0)              # (T, M)
    angles_D = torch.einsum("ti,p->tip", t_plus_i, freqs_local)  # (T, M, n_p_D)

    angles = torch.cat([angles_A, angles_B, angles_C, angles_D], dim=-1)
    return torch.polar(torch.ones_like(angles), angles)


def precompute_window_mask_glpe_v1(end: int, max_insertion: int) -> torch.Tensor:
    """``(end, end)`` bool window mask: ``True`` if ``|i-j| <= local_len``."""
    local_len = glpe_v1_local_len(max_insertion)
    idx = torch.arange(end)
    diff = idx.unsqueeze(0) - idx.unsqueeze(1)
    return diff.abs() <= local_len
