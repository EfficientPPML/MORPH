"""
TPU-native G1 elliptic curve ops over BLS12-377, in Extended Twisted
Edwards 4-coord DRNS form.

Used by the prover so that point additions and scalar multiplications
stay on the TPU without round-tripping through Weierstrass affine
between every step.

Public API
----------
    encode_g1(point)                  -> (4, 1, M_fp)  DRNS-encoded
    decode_g1(tpu_pt)                 -> G1Affine     (one Weierstrass decode)
    identity()                        -> (4, 1, M_fp)  zero point (Edwards 0)
    point_add_tpu(P, Q)               -> (4, 1, M_fp)
    scalar_mul_tpu(P, k_int)          -> (4, 1, M_fp)  binary ladder

Notes
-----
* ``ec_ctx.point_double`` is disabled (logic bug).  ``scalar_mul_tpu``
  uses ``point_add(P, P)`` for doubling — slow but correct.
* All ops share the same ``ec_ctx`` obtained from the cached MSM context
  so the JIT'd kernels are reused.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
import path_setup  # noqa: F401

import functools

import jax
import jax.numpy as jnp

import tpu_contexts
from bls12_377.g1 import G1Affine, G1Projective, G1_GEN_X, G1_GEN_Y
from kernels.msm import _TPU_MSM_LENGTH


# ── EC context handle (shared with MSM) ──────────────────────────────────────


def _ec_ctx():
    """Edwards EC context, lazily fetched from the cached MSM ctx."""
    return tpu_contexts.get_msm_context(_TPU_MSM_LENGTH).ec_ctx


@functools.cache
def _point_add_jit():
    """A jit-compiled wrapper around ``ec_ctx._point_add`` so that point
    additions on a stable (4, 1, M) shape pay the trace+compile cost
    exactly once per process."""
    ec_ctx = _ec_ctx()
    return jax.jit(ec_ctx._point_add)


# ── identity / encode / decode ───────────────────────────────────────────────


def identity():
    """Return the Edwards neutral element in DRNS form, shape (4, 1, M_fp)."""
    ec_ctx = _ec_ctx()
    # ec_ctx.zero_point = [0, 1, 1, 0] in plain ints (Edwards form).
    ff = ec_ctx.get_finite_field_context()
    zero = ff.to_computational_format(ec_ctx.zero_point)   # (4, M_fp)
    return zero.reshape(4, 1, zero.shape[-1])


def encode_g1(pt) -> jnp.ndarray:
    """Encode a G1Affine / G1Projective into Edwards DRNS, shape (4, 1, M_fp).

    The curve identity (point at infinity) is represented as the Edwards
    neutral [0, 1, 1, 0]; ``encode_g1`` rejects the infinity case so the
    caller can substitute ``identity()`` explicitly.
    """
    if isinstance(pt, G1Projective):
        pt = pt.to_affine()
    if pt.is_infinity():
        return identity()
    ec_ctx = _ec_ctx()
    arr = ec_ctx.to_computational_format([int(pt.x), int(pt.y)])  # (4, 1, M_fp)
    return arr


def encode_g1_batch(pts) -> jnp.ndarray:
    """Encode a list of points into (4, N, M_fp) Edwards DRNS in one pass."""
    ec_ctx = _ec_ctx()
    xy = []
    for pt in pts:
        if isinstance(pt, G1Projective):
            pt = pt.to_affine()
        if pt.is_infinity():
            xy.append([G1_GEN_X, G1_GEN_Y])  # placeholder, caller must mask
        else:
            xy.append([int(pt.x), int(pt.y)])
    return ec_ctx.to_computational_format(xy)


def decode_g1(tpu_pt: jnp.ndarray) -> G1Affine:
    """Decode an Edwards DRNS array back to ``G1Affine`` (one Weierstrass
    untwist).  Accepts shape (4, M_fp) or (4, 1, M_fp).

    Returns the projective identity if the decoded point is off-curve
    (which is how the Edwards neutral round-trips to junk in Weierstrass)."""
    ec_ctx = _ec_ctx()
    if tpu_pt.ndim == 3 and tpu_pt.shape[1] == 1:
        tpu_pt = tpu_pt.reshape(tpu_pt.shape[0], tpu_pt.shape[-1])  # (4, M)
    out = ec_ctx.to_original_format(tpu_pt)
    if out is None or len(out) != 2:
        raise RuntimeError(f"decode_g1: unexpected output shape {out!r}")
    x, y = int(out[0]), int(out[1])
    aff = G1Affine(x, y)
    if not aff.is_on_curve():
        # Edwards neutral → Weierstrass returns garbage; map back to identity.
        return G1Affine(0, 0, infinity=True)
    return aff


# ── core ops ────────────────────────────────────────────────────────────────


def _as_4_1_M(arr: jnp.ndarray) -> jnp.ndarray:
    """Normalize a point to shape (4, 1, M_fp)."""
    if arr.ndim == 2:
        return arr.reshape(arr.shape[0], 1, arr.shape[-1])
    if arr.ndim == 3:
        return arr
    raise ValueError(f"unexpected EC point shape {arr.shape}")


def point_add_tpu(P: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """Edwards point addition on TPU.  Inputs/output: (4, 1, M_fp)."""
    P = _as_4_1_M(P)
    Q = _as_4_1_M(Q)
    return _point_add_jit()(P, Q)


def scalar_mul_tpu(P: jnp.ndarray, k: int) -> jnp.ndarray:
    """Binary-ladder scalar multiplication k·P on TPU.

    k is a Python int (0 <= k < r).  Uses ``point_add(acc, acc)`` for
    doubling since ``point_double`` is disabled in the EC context.
    """
    k = int(k)
    if k == 0:
        return identity()
    P = _as_4_1_M(P)
    if k == 1:
        return P

    # Walk bits MSB→LSB.
    bits = []
    kk = k
    while kk:
        bits.append(kk & 1)
        kk >>= 1
    bits.reverse()                  # MSB first

    acc = P                         # start at the MSB (always 1)
    for bit in bits[1:]:
        acc = point_add_tpu(acc, acc)        # double
        if bit:
            acc = point_add_tpu(acc, P)      # add
    return acc


def warmup() -> None:
    """Force JIT compilation of ``point_add`` on the (4, 1, M_fp) shape."""
    P = identity()
    point_add_tpu(P, P).block_until_ready()


# ── JIT'd composite kernels for the optimised prover ────────────────────────


# BLS12-377 Fr is 253 bits.  We pad scalars to a fixed bit-width inside
# the JIT'd binary ladder so the kernel shape stays stable across calls.
_G1_SCALAR_BITS = 253


def _identity_g1_cached():
    """Cached (4, 1, M_fp) identity point.  Compile-time constant."""
    return identity()


def int_to_bits_msb(k: int, num_bits: int = _G1_SCALAR_BITS) -> jnp.ndarray:
    """Decompose a non-negative Python int into MSB-first bits.

    Returns a length-``num_bits`` jnp.int8 array; bits beyond ``k.bit_length()``
    are zero.  Routed through numpy to keep the conversion cost off-TPU.
    """
    import numpy as _np
    k = int(k)
    out = _np.zeros(num_bits, dtype=_np.int8)
    # MSB-first: bit at index (num_bits-1) is LSB.
    for i in range(num_bits):
        out[num_bits - 1 - i] = (k >> i) & 1
    return jnp.asarray(out)


def _scalar_mul_g1_traced(P: jnp.ndarray, k_bits: jnp.ndarray) -> jnp.ndarray:
    """Binary-ladder scalar-mul body, intended to be inlined inside larger
    JIT kernels.  Uses ETE point_add for doubling (unified, identity-safe).

    P      : (4, 1, M_fp)
    k_bits : (S,) int8, MSB first
    """
    ec = _ec_ctx()
    identity_pt = _identity_g1_cached()

    def step(carry, bit):
        acc, p = carry
        acc2       = ec._point_add(acc, acc)
        acc2_plus  = ec._point_add(acc2, p)
        acc_new    = jnp.where(bit != 0, acc2_plus, acc2)
        return (acc_new, p), None

    (final, _), _ = jax.lax.scan(step, (identity_pt, P), k_bits)
    return final


def _reshape_to_4_1_M(arr):
    """Add the unit batch axis the EC point ops expect.  Works for both
    ``(4, M_fp)`` (raw MSM kernel output) and ``(4, 1, M_fp)`` (already
    in the right shape) inputs.

    Called from inside JIT'd kernels so the reshape becomes part of the
    XLA graph and gets fused with the surrounding EC math instead of
    being a separate dispatch.
    """
    return arr.reshape(arr.shape[0], 1, arr.shape[-1])


def _pi_a_or_b_g1_inner(fixed_g1, sum_g1, delta_g1, k_bits):
    """Compute  fixed + sum + k·delta  on G1 ETE (used for π_A and π_B_g1).

    ``sum_g1`` arrives as the raw MSM kernel output ``(4, M_fp)``; the
    reshape to ``(4, 1, M_fp)`` is fused in this JIT scope so XLA can
    fold it into the first ``point_add``'s input layout.
    """
    ec = _ec_ctx()
    sum_g1 = _reshape_to_4_1_M(sum_g1)
    k_delta = _scalar_mul_g1_traced(delta_g1, k_bits)
    return ec._point_add(ec._point_add(fixed_g1, sum_g1), k_delta)


def _pi_c_inner(private_sum, h_sum, pi_A, pi_B_g1, delta_g1,
                s_bits, r_bits, rs_bits):
    """Compute  private + h + s·pi_A + r·pi_B_g1 + rs·delta  on G1 ETE.

    ``private_sum`` and ``h_sum`` arrive as raw MSM kernel outputs
    ``(4, M_fp)``; reshape into ``(4, 1, M_fp)`` is fused in this JIT
    scope.
    """
    ec = _ec_ctx()
    private_sum = _reshape_to_4_1_M(private_sum)
    h_sum       = _reshape_to_4_1_M(h_sum)
    s_A   = _scalar_mul_g1_traced(pi_A,    s_bits)
    r_Bg1 = _scalar_mul_g1_traced(pi_B_g1, r_bits)
    rs_D  = _scalar_mul_g1_traced(delta_g1, rs_bits)
    sum_ph  = ec._point_add(private_sum, h_sum)
    sum_phA = ec._point_add(sum_ph, s_A)
    sum_BD  = ec._point_add(r_Bg1, rs_D)
    return ec._point_add(sum_phA, sum_BD)


@functools.cache
def _affine_combo_g1_jit():
    """JIT'd fixed + sum + k·delta on G1 ETE (used for π_A, π_B_g1)."""
    return jax.jit(_pi_a_or_b_g1_inner)


@functools.cache
def _pi_c_jit():
    """JIT'd π_C composition kernel on G1 ETE."""
    return jax.jit(_pi_c_inner)


def affine_combo_g1_tpu(fixed_g1, sum_g1, delta_g1, k_bits):
    """fixed + sum + k·delta on G1 ETE.

    Shape contract:
      * ``fixed_g1``, ``delta_g1`` : ``(4, 1, M_fp)`` — pre-encoded EC
        constants from :mod:`tpu_precompute`.
      * ``sum_g1``                 : ``(4, M_fp)``   — raw MSM output;
        the unit-axis reshape is fused inside the JIT'd kernel.
      * ``k_bits``                 : ``(S,)`` int8 MSB-first scalar bits.
    """
    return _affine_combo_g1_jit()(fixed_g1, sum_g1, delta_g1, k_bits)


def pi_c_tpu(private_sum, h_sum, pi_A, pi_B_g1, delta_g1,
             s_bits, r_bits, rs_bits):
    """π_C kernel:  private + h + s·pi_A + r·pi_B_g1 + rs·delta.

    Shape contract:
      * ``private_sum``, ``h_sum`` : ``(4, M_fp)``   — raw MSM outputs;
        the unit-axis reshape is fused inside the JIT'd kernel.
      * ``pi_A``, ``pi_B_g1``, ``delta_g1`` : ``(4, 1, M_fp)``.
      * bit arrays : ``(S,)`` int8 MSB-first.
    """
    return _pi_c_jit()(private_sum, h_sum, pi_A, pi_B_g1, delta_g1,
                       s_bits, r_bits, rs_bits)
