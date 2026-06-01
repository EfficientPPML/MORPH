"""
★ KERNEL: MSM (Multi-Scalar Multiplication) ★

Computes  Σ scalars[i] · points[i]  for G1 or G2 points.

Public API
----------
    msm_g1(points, scalars) -> G1Projective
    msm_g2(points, scalars) -> G2Projective

Algorithm: Pippenger's bucket method.
  - Window size w = max(1, floor(log2(n)) - 1)
  - Scalars split into ceil(256/w) windows of w bits each
  - Each window: accumulate into 2^w buckets, then sum buckets

For the tiny cubic circuit (n ≤ 5), this reduces to simple double-and-add,
but the code is written generically for any n (e.g. n = 2^20).

This kernel is a PURE FUNCTION with no Groth16 logic inside.
Call sites in the protocol are explicitly marked with:
    # ═══ KERNEL: MSM (G1) ═══  /  # ═══ KERNEL: MSM (G2) ═══
"""

import math
from bls12_377.g1 import G1Affine, G1Projective
from bls12_377.g2 import G2Affine, G2Projective
from bls12_377.field import FP2_ONE, FP2_ZERO


# ── Scalar bit extraction ──────────────────────────────────────────────────────

def _scalar_bits(s: int, start: int, w: int) -> int:
    """Extract w bits of s starting at bit position `start`."""
    return (s >> start) & ((1 << w) - 1)


# ── Generic Pippenger ─────────────────────────────────────────────────────────

def _pippenger(points, scalars, zero, add_fn, dbl_fn):
    """
    Pippenger's bucket method.

    points  : list of curve points (affine or projective)
    scalars : list of ints (field elements in Fr, 0 ≤ s < r)
    zero    : identity element (projective)
    add_fn  : (proj, point) -> proj   (mixed or projective add)
    dbl_fn  : proj -> proj
    """
    n = len(scalars)
    if n == 0:
        return zero

    # Window size
    w = max(1, int(math.log2(n)) - 1) if n > 1 else 1
    num_windows = (256 + w - 1) // w   # ceil(256/w)
    num_buckets = 1 << w               # 2^w

    result = zero

    for win in range(num_windows - 1, -1, -1):
        # Double result w times (shift window)
        if win < num_windows - 1:
            for _ in range(w):
                result = dbl_fn(result)

        # Accumulate into buckets for this window
        buckets = [zero] * num_buckets
        for i, (pt, s) in enumerate(zip(points, scalars)):
            idx = _scalar_bits(s, win * w, w)
            if idx != 0:
                buckets[idx] = add_fn(buckets[idx], pt)

        # Sum buckets: running sum from high to low
        running = zero
        for b in range(num_buckets - 1, 0, -1):
            running = add_fn(running, buckets[b])
            result  = add_fn(result, running)

    return result


# ── G1 MSM ────────────────────────────────────────────────────────────────────

def msm_g1(points: list, scalars: list) -> G1Projective:
    """
    ★ MSM KERNEL — G1 ★

    Compute  Σ scalars[i] · points[i]  over G1.

    Parameters
    ----------
    points  : list of G1Affine or G1Projective elements
    scalars : list of ints in Fr

    Returns
    -------
    G1Projective — the accumulated sum.
    """
    assert len(points) == len(scalars), "msm_g1: length mismatch"
    if not points:
        return G1Projective(1, 1, 0)

    zero = G1Projective(1, 1, 0)

    def add_fn(acc: G1Projective, pt) -> G1Projective:
        if isinstance(pt, G1Affine):
            return acc.add_affine(pt)
        return acc.add(pt)

    def dbl_fn(acc: G1Projective) -> G1Projective:
        return acc.double()

    return _pippenger(points, scalars, zero, add_fn, dbl_fn)


# ── G2 MSM ────────────────────────────────────────────────────────────────────

def msm_g2(points: list, scalars: list) -> G2Projective:
    """
    ★ MSM KERNEL — G2 ★

    Compute  Σ scalars[i] · points[i]  over G2.

    Parameters
    ----------
    points  : list of G2Affine or G2Projective elements
    scalars : list of ints in Fr

    Returns
    -------
    G2Projective — the accumulated sum.
    """
    assert len(points) == len(scalars), "msm_g2: length mismatch"
    if not points:
        return G2Projective(FP2_ONE, FP2_ONE, FP2_ZERO)

    zero = G2Projective(FP2_ONE, FP2_ONE, FP2_ZERO)

    def add_fn(acc: G2Projective, pt) -> G2Projective:
        if isinstance(pt, G2Affine):
            return acc.add_affine(pt)
        return acc.add(pt)

    def dbl_fn(acc: G2Projective) -> G2Projective:
        return acc.double()

    return _pippenger(points, scalars, zero, add_fn, dbl_fn)
