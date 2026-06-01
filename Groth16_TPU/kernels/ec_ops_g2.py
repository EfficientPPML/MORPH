"""TPU-native G2 elliptic curve ops over BLS12-377, in XYZZ-Weierstrass
4-coord DRNS form over Fp2.

Used by the optimised prover so that the post-MSM G2 composition
(``β_g2 + B_sum_g2 + s_blind·δ_g2``) stays on the TPU as a single
JIT'd kernel instead of round-tripping through Weierstrass-projective
CPU code.

Public API
----------
    identity_g2()                                    -> (4, 1, 2, M_fp)
    point_add_g2_tpu(P, Q)                           -> (4, 1, 2, M_fp)
    point_double_g2_tpu(P)                           -> (4, 1, 2, M_fp)
    scalar_mul_g2_tpu(P, k_bits)                     -> (4, 1, 2, M_fp)
    affine_combo_g2_tpu(beta, sum, delta, k_bits)    -> (4, 1, 2, M_fp)

Notes
-----
* XYZZ add formulas are NOT unified — they degenerate when ``P == Q`` or
  either operand is the identity.  The binary ladder here uses
  ``point_double`` for doublings and starts from ``P`` (assumes the input
  scalar is non-zero, true for the Groth16 blinders).
* The composite kernel is JIT-compiled once per shape and reused across
  proofs.
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


# ── EC context handle ───────────────────────────────────────────────────────


def _ec_ctx_g2():
    """XYZZ-Weierstrass Fq2 EC context, fetched from the cached G2 MSM ctx."""
    return tpu_contexts.get_g2_msm_context().ec_ctx


# BLS12-377 Fr is 253 bits; same fixed-bit ladder length as G1.
_G2_SCALAR_BITS = 253


# ── identity / encode / decode ───────────────────────────────────────────────


@functools.cache
def _identity_g2_cached():
    """XYZZ identity (1, 1, 0, 0) in DRNS Fp2 form, shape (4, 1, 2, M_fp)."""
    ec = _ec_ctx_g2()
    fe = ec.fe_ctx
    cf = fe.to_computational_format(ec.zero_point)  # (4, 2, M_fp)
    return jnp.expand_dims(cf, axis=1)              # (4, 1, 2, M_fp)


def identity_g2() -> jnp.ndarray:
    return _identity_g2_cached()


# ── JIT'd core ops ──────────────────────────────────────────────────────────


@functools.cache
def _point_add_g2_jit():
    """jit-wrapped XYZZ Fq2 add on the stable (4, 1, 2, M_fp) shape."""
    return jax.jit(_ec_ctx_g2()._point_add_jax)


@functools.cache
def _point_double_g2_jit():
    """jit-wrapped XYZZ Fq2 double on the stable (4, 1, 2, M_fp) shape."""
    return jax.jit(_ec_ctx_g2()._point_double_jax)


def point_add_g2_tpu(P: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """XYZZ point addition on TPU.  Inputs/output: (4, 1, 2, M_fp).

    XYZZ add is NOT identity- or self-add-safe; callers must avoid those.
    """
    return _point_add_g2_jit()(P, Q)


def point_double_g2_tpu(P: jnp.ndarray) -> jnp.ndarray:
    """XYZZ point doubling on TPU.  Input/output: (4, 1, 2, M_fp)."""
    return _point_double_g2_jit()(P)


# ── JIT'd binary-ladder scalar-mul (XYZZ) ───────────────────────────────────


def _scalar_mul_g2_traced(P: jnp.ndarray, k_bits: jnp.ndarray) -> jnp.ndarray:
    """Binary-ladder scalar mul over G2 XYZZ.  Inlined inside larger JIT
    kernels (composite ones).

    P      : (4, 1, 2, M_fp)
    k_bits : (S,) int8, MSB first  (S = 253 for BLS12-377 Fr)

    XYZZ add/double DON'T handle identity (the formula degenerates),
    so we can't naively seed the ladder with identity and process all
    bits.  Instead we track a "pristine" flag in the scan carry: while
    pristine, the accumulator is conceptually the identity and we
    bypass the EC formula entirely (doubling identity stays identity;
    adding P leaves either P or identity depending on the bit).  As
    soon as the first 1-bit is consumed, pristine flips to False and
    the standard double-and-add path takes over.

    For Groth16 blinders ``1 ≤ k < r`` (Fr is 253 bits), the leading
    1-bit position varies — this generalised ladder handles all values.
    """
    ec = _ec_ctx_g2()
    identity_pt = _identity_g2_cached()

    def step(carry, bit):
        acc, pristine, p = carry
        # Non-pristine path: real XYZZ double-and-add.
        acc2 = ec._point_double_jax(acc)
        acc2_plus = ec._point_add_jax(acc2, p)
        acc_nonpristine = jnp.where(bit != 0, acc2_plus, acc2)
        # Pristine path: doubling identity stays identity; if this is the
        # first 1-bit, transition to P.  Otherwise remain identity.
        acc_pristine = jnp.where(bit != 0, p, identity_pt)
        acc_new = jnp.where(pristine, acc_pristine, acc_nonpristine)
        # Pristine stays True only while every bit so far has been 0.
        pristine_new = jnp.logical_and(pristine, bit == 0)
        return (acc_new, pristine_new, p), None

    init_carry = (identity_pt, jnp.array(True), P)
    (final, _, _), _ = jax.lax.scan(step, init_carry, k_bits)
    return final


def _pi_b_g2_inner(beta_g2, sum_g2, delta_g2, k_bits):
    """β_g2 + sum_g2 + k·δ_g2  on G2 XYZZ.

    All point ops are direct XYZZ kernel calls — no identity handling
    is required because none of the operands is the identity in normal
    Groth16 flow (β_g2 and δ_g2 are fixed non-identity generators, and
    sum_g2 is the MSM result over non-identity bases).
    """
    ec = _ec_ctx_g2()
    k_delta = _scalar_mul_g2_traced(delta_g2, k_bits)
    sum1 = ec._point_add_jax(beta_g2, sum_g2)
    return ec._point_add_jax(sum1, k_delta)


@functools.cache
def _pi_b_g2_jit():
    """JIT'd π_B_g2 composition kernel."""
    return jax.jit(_pi_b_g2_inner)


def affine_combo_g2_tpu(beta_g2, sum_g2, delta_g2, k_bits):
    """β_g2 + sum_g2 + k·δ_g2 on G2 XYZZ.  All points (4, 1, 2, M_fp);
    k_bits (S,) int8 MSB-first.
    """
    return _pi_b_g2_jit()(beta_g2, sum_g2, delta_g2, k_bits)


# ── warmup ──────────────────────────────────────────────────────────────────


def warmup() -> None:
    """Force JIT compilation of point_add / point_double on (4, 1, 2, M_fp)."""
    P = identity_g2()
    # Add/double the identity — values don't matter, only the shape and
    # the JIT trace.
    point_add_g2_tpu(P, P).block_until_ready()
    point_double_g2_tpu(P).block_until_ready()