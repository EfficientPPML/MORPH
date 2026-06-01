"""
TPU-backed MSM adapter — drop-in for the upstream
``Groth16/kernels/msm.py`` API.

Public API matches the upstream module:
    msm_g1(points, scalars) -> G1Projective    (TPU)
    msm_g2(points, scalars) -> G2Projective    (CPU fallback — Fp2 not
                                                supported by the TPU stack)

Implementation notes
--------------------
* The TPU MSM context (``FusionMSMContext``) is built/compiled per
  ``msm_length``.  We pad every G1 call to a single fixed length
  (``_TPU_MSM_LENGTH``) so we share a single compiled kernel across all
  call sites in the protocol.  Padding entries pair the generator with
  scalar 0, which contributes nothing to the sum.
* Points are converted via ``ec_ctx.to_computational_format`` which does
  the full Weierstrass(Fp) → twist → Edwards 4-coord → DRNS pipeline.
  The result of ``multiscalar_multiply`` (an Edwards 4-coord DRNS array)
  is decoded by ``ec_ctx.to_original_format`` back to ``[x, y]``
  Weierstrass affine.
"""

import functools
import math
import os
import sys

import jax
import numpy as np

# Ensure path_setup has applied.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
import path_setup  # noqa: F401

import jax.numpy as jnp

from cpu_kernels import msm as _cpu_msm
import tpu_contexts

from bls12_377.g1 import G1Affine, G1Projective, G1_GEN_X, G1_GEN_Y
from bls12_377.g2 import G2Affine, G2Projective
from bls12_377.params import r as _FR, G2_GEN_X, G2_GEN_Y


# Single fixed length for all G1 MSM calls.  Must satisfy:
#   * power of two (FusionMSMContext requirement),
#   * >= max len of any pk vector the protocol will pass us,
#   * large enough that ``slice_bits = log2(msm_length)`` ≥ 8.
#
# Small msm_length combined with slice_bits ≤ 4 hits a numerical bug in
# the FusionMSMContext last-window duplication path (bucket_num_last_window
# is 2 for 253-bit scalars at slice_bits=4, and the duplication-ratio
# path produces wrong results once ≥ 3 input scalars actually have
# distinct large values).  msm_length = 1024 (slice_bits = 10) works
# correctly across all our tests and the compile is one-time.
#
# All Groth16 MSM call sites for the cubic-sized circuit fit in 1024
# easily (the largest is len(h_g1) = n - 1, well under 1024 for any
# small n we'd run).
_TPU_MSM_LENGTH = 1024


def encode_points_for_msm(points):
    """Encode a list of G1 points into the tiled MSM layout once.

    Returns ``(tiled_points, identity_mask)`` where

      tiled_points  : (tile_num, tile_length, coord_dim, M_fp) ready for
                      ``ctx.points``.  Identity points and padding both
                      use the G1 generator as a placeholder.

      identity_mask : tuple of bools, length == len(points), True for
                      positions where the input was the curve identity.
                      ``msm_g1_tpu`` uses this to zero out the
                      corresponding scalars so the placeholder generator
                      contributes nothing.
    """
    affine_xy = []
    identity_mask = []
    for pt in points:
        if isinstance(pt, G1Projective):
            pt = pt.to_affine()
        if pt.is_infinity():
            affine_xy.append([G1_GEN_X, G1_GEN_Y])
            identity_mask.append(True)
        else:
            affine_xy.append([int(pt.x), int(pt.y)])
            identity_mask.append(False)

    while len(affine_xy) < _TPU_MSM_LENGTH:
        affine_xy.append([G1_GEN_X, G1_GEN_Y])

    if len(affine_xy) > _TPU_MSM_LENGTH:
        raise NotImplementedError(
            f"encode_points_for_msm: up to {_TPU_MSM_LENGTH} points; got {len(affine_xy)}."
        )

    ctx = tpu_contexts.get_msm_context(_TPU_MSM_LENGTH)
    ec_ctx = ctx.ec_ctx
    pts_comp = ec_ctx.to_computational_format(affine_xy)        # (4, N, M)
    pts_comp = jnp.asarray(pts_comp).transpose(1, 0, 2)          # (N, 4, M)
    tiled = pts_comp.reshape(
        ctx.tile_num, ctx.tile_length, ctx.coordinate_dim, ctx.moduli_num
    )
    return tiled, tuple(identity_mask)


def _pad_scalars(scalars, identity_mask=None):
    """Reduce scalars mod r, mask out positions where the matching point
    was the curve identity, and pad to _TPU_MSM_LENGTH with zeros."""
    out = []
    for i, s in enumerate(scalars):
        if identity_mask is not None and i < len(identity_mask) and identity_mask[i]:
            out.append(0)
        else:
            out.append(int(s) % _FR)
    while len(out) < _TPU_MSM_LENGTH:
        out.append(0)
    if len(out) > _TPU_MSM_LENGTH:
        raise NotImplementedError(
            f"msm_g1: up to {_TPU_MSM_LENGTH} scalars; got {len(out)}."
        )
    return out


def encode_scalars_for_msm_g1(scalars, identity_mask=None):
    """One-shot pre-encoding of an Fr scalar list into the tiled MSM format.

    Combines ``_pad_scalars`` and ``ctx.to_computational_format`` so the
    caller can hand the result back to :func:`msm_g1_tpu` without
    re-paying the Python loop and DRNS encoding on every MSM call.
    Particularly useful when the same witness scalars feed multiple MSMs
    (A_g1, B_g1, B_g2) — encode once, reuse.

    Returns a JAX ``jnp.ndarray`` of whatever shape the MSM kernel's
    ``multiscalar_multiply`` expects (opaque to the caller — just pass
    it back into ``msm_g1_tpu``).
    """
    padded = _pad_scalars(scalars, identity_mask=identity_mask)
    ctx = tpu_contexts.get_msm_context(_TPU_MSM_LENGTH)
    return ctx.to_computational_format(padded)


def encode_scalars_for_msm_g1_packed(scalars, identity_mask=None):
    """T3 fast path: encode + slice via the packed-uint32 representation.

    Goes through :func:`encode_scalars_packed` (CPU, one-time per witness)
    + :func:`slice_scalars_jit_g1` (TPU JIT'd), bypassing the CPU
    ``utils.slice_scalars`` Python double-loop entirely.  The returned
    array is the same shape that ``FusionMSMContext.multiscalar_multiply``
    expects — drop it into :func:`msm_g1_tpu` like any other pre-encoded
    scalar array.
    """
    # Mask + reduce mod r (still a CPU pass, but only N ops — no per-window
    # inner loop).
    cleaned = _pad_scalars(scalars, identity_mask=identity_mask)
    packed  = encode_scalars_packed(cleaned)
    return slice_scalars_jit_g1(packed)


# ── T3: TPU-side window-slice extraction from packed uint32 ─────────────────

# 253-bit Fr fits in 8 uint32 chunks (256 bits) with 3 unused high bits.
_FR_CHUNK_BITS = 32
_FR_NUM_CHUNKS = 8


def encode_scalars_packed(scalars, identity_mask=None,
                          target_length: int = _TPU_MSM_LENGTH):
    """Pack a list of Fr scalars into a ``(target_length, _FR_NUM_CHUNKS)``
    uint32 array — one row per scalar, low chunk first.

    This is the "packed" parallel representation used by
    :func:`slice_scalars_jit_g1` for TPU-resident window-slice
    extraction (T3, approach C).  Identity-masked positions and
    padding positions both encode as all-zero rows, which downstream
    slice as all-zero windows — the MSM kernel skips ``slice == 0``,
    so the contribution is zero either way.
    """
    if len(scalars) > target_length:
        raise NotImplementedError(
            f"encode_scalars_packed: up to {target_length} scalars; got {len(scalars)}."
        )
    out = np.zeros((target_length, _FR_NUM_CHUNKS), dtype=np.uint32)
    chunk_mask = (1 << _FR_CHUNK_BITS) - 1
    for i, s in enumerate(scalars):
        if identity_mask is not None and i < len(identity_mask) and identity_mask[i]:
            continue
        v = int(s) % _FR
        for j in range(_FR_NUM_CHUNKS):
            out[i, j] = v & chunk_mask
            v >>= _FR_CHUNK_BITS
    return jnp.asarray(out)


@functools.lru_cache(maxsize=None)
def _slice_scalars_jit_factory(scalar_bits: int, slice_bits: int,
                               tile_length: int, num_chunks: int):
    """Build a JIT'd slicer for a fixed (scalar_bits, slice_bits, tile_length,
    num_chunks) shape.  Output shape: ``(1, window_num, tile_length)`` int32.

    The window-touching-chunk layout is computed at trace time (Python
    side), so the resulting JAX graph is a flat list of bit-shifts and
    bitwise-ANDs — no dynamic indexing.
    """
    window_num = math.ceil(scalar_bits / slice_bits)
    slice_mask = (1 << slice_bits) - 1
    chunk_bits = _FR_CHUNK_BITS
    chunk_mask = (1 << chunk_bits) - 1

    def inner(packed):  # (tile_length, num_chunks) uint32
        windows = []
        for i in range(window_num):
            start = i * slice_bits
            chunk_lo = start // chunk_bits
            offset   = start % chunk_bits
            bits_lo  = min(slice_bits, chunk_bits - offset)
            # Low-chunk contribution.
            low = jnp.right_shift(packed[:, chunk_lo].astype(jnp.uint32),
                                  jnp.uint32(offset))
            if bits_lo < slice_bits:
                bits_hi = slice_bits - bits_lo
                if chunk_lo + 1 < num_chunks:
                    high = packed[:, chunk_lo + 1].astype(jnp.uint32)
                    high_part = jnp.bitwise_and(high, jnp.uint32((1 << bits_hi) - 1))
                    slice_val = jnp.bitwise_or(
                        jnp.bitwise_and(low, jnp.uint32((1 << bits_lo) - 1)),
                        jnp.left_shift(high_part, jnp.uint32(bits_lo)),
                    )
                else:
                    # Off the end — high bits are implicit zero.
                    slice_val = jnp.bitwise_and(low, jnp.uint32((1 << bits_lo) - 1))
            else:
                slice_val = jnp.bitwise_and(low, jnp.uint32(slice_mask))
            windows.append(slice_val.astype(jnp.int32))
        stacked = jnp.stack(windows, axis=0)  # (window_num, tile_length)
        return jnp.expand_dims(stacked, axis=0)  # (1, window_num, tile_length)

    return jax.jit(inner)


def slice_scalars_jit_g1(packed: jnp.ndarray) -> jnp.ndarray:
    """TPU-side window-slice extraction for the G1 MSM context.

    Input  : ``(tile_length, _FR_NUM_CHUNKS)`` uint32 from
             :func:`encode_scalars_packed`.
    Output : ``(1, window_num, tile_length)`` int32, the layout
             ``FusionMSMContext.multiscalar_multiply`` consumes.
    """
    ctx = tpu_contexts.get_msm_context(_TPU_MSM_LENGTH)
    fn = _slice_scalars_jit_factory(
        scalar_bits=ctx.scalar_bits,
        slice_bits=ctx.slice_bits,
        tile_length=ctx.tile_length,
        num_chunks=_FR_NUM_CHUNKS,
    )
    return fn(packed)


def _msm_dispatch(points_tiled, tiled_slices):
    """Run the MSM kernel against pre-tiled points + pre-encoded scalars.

    Returns the raw Edwards 4-coord DRNS accumulator (shape (4, M_fp)).
    """
    ctx = tpu_contexts.get_msm_context(_TPU_MSM_LENGTH)
    ctx.points = points_tiled
    ctx.reset()
    result = ctx.multiscalar_multiply(tiled_slices)
    if isinstance(result, (tuple, list)):
        result = result[0]
    result.block_until_ready()
    return result


def msm_g1_tpu(points_tiled_or_list, scalars):
    """MSM-G1 that **stays in TPU-native form**.

    Parameters
    ----------
    points_tiled_or_list
        Either a list of G1 points (will be encoded on the fly), OR a
        ``(tiled_points, identity_mask)`` tuple produced by
        ``encode_points_for_msm`` (preferred — re-uses the encoding
        across proofs and tracks which positions were the curve
        identity so the scalar at that index gets masked to 0).
    scalars
        Either an iterable of Fr scalars (Python ints) — gets padded to
        ``_TPU_MSM_LENGTH`` and DRNS-encoded inline, OR a pre-encoded
        ``jnp.ndarray`` produced by :func:`encode_scalars_for_msm_g1` —
        skipped through directly to the MSM kernel.  Pre-encoding wins
        when the same witness drives multiple MSM call sites.

    Returns
    -------
    jnp.ndarray
        Edwards 4-coord DRNS, shape ``(4, M_fp)``.  Pair with
        ``kernels.ec_ops.point_add_tpu`` etc. for further on-TPU work.
    """
    if isinstance(points_tiled_or_list, tuple) and len(points_tiled_or_list) == 2 \
            and isinstance(points_tiled_or_list[0], jnp.ndarray):
        points_tiled, identity_mask = points_tiled_or_list
    else:
        points_tiled, identity_mask = encode_points_for_msm(points_tiled_or_list)

    if isinstance(scalars, jnp.ndarray):
        # Pre-encoded fast path — caller already paid the padding+DRNS cost.
        tiled_slices = scalars
    else:
        tiled_slices = encode_scalars_for_msm_g1(
            scalars, identity_mask=identity_mask
        )
    return _msm_dispatch(points_tiled, tiled_slices)


def msm_g1_pure_tpu(points_tiled: jnp.ndarray, tiled_slices: jnp.ndarray) -> jnp.ndarray:
    """No-branch MSM-G1 for ``prove_pure_tpu``.

    Both inputs must already be JAX arrays in MSM-kernel layout — the
    caller is responsible for any encoding.  This deliberately omits:

      * the ``points`` Python-list path (``encode_points_for_msm`` is
        what the framework's ``precompute_pk`` already did),
      * the ``scalars`` Python-list path (use
        :func:`encode_scalars_for_msm_g1_packed` to pre-encode and pass
        the resulting JAX array straight in),
      * the result-shape ``isinstance`` check (the kernel returns a
        single jnp.ndarray on this code path),
      * the per-call ``block_until_ready`` (the caller blocks once at
        the end of the pure-TPU body, so we don't force premature sync
        between successive MSMs).

    Output shape is the raw kernel return, ``(4, M_fp)`` — no unit-axis
    reshape here.  The downstream JIT'd EC composites
    (``ec_ops.affine_combo_g1_tpu``, ``ec_ops.pi_c_tpu``) accept the
    flat ``(4, M_fp)`` form directly and reshape inside their own JIT
    scope, so XLA fuses the layout change with the subsequent EC math.

    Returns
    -------
    jnp.ndarray
        Edwards 4-coord DRNS, shape ``(4, M_fp)``.
    """
    ctx = tpu_contexts.get_msm_context(_TPU_MSM_LENGTH)
    ctx.points = points_tiled
    ctx.reset()
    return ctx.multiscalar_multiply(tiled_slices)


def msm_g1(points, scalars) -> G1Projective:
    """TPU MSM over G1, returning a CPU ``G1Projective`` (legacy API).

    Kept as a drop-in replacement for the CPU kernel.  New code that
    plans to combine the result with further EC ops should call
    :func:`msm_g1_tpu` instead and stay on the TPU.
    """
    assert len(points) == len(scalars), "msm_g1: length mismatch"
    if not points:
        return G1Projective(1, 1, 0)

    from kernels import ec_ops  # local import to avoid circular path

    # Mask scalars where the corresponding point is the identity.
    masked_scalars = []
    for pt, s in zip(points, scalars):
        aff = pt.to_affine() if isinstance(pt, G1Projective) else pt
        masked_scalars.append(0 if aff.is_infinity() else int(s) % _FR)

    result = msm_g1_tpu(points, masked_scalars)
    return ec_ops.decode_g1(result).to_projective()


def encode_points_g2_for_msm(points):
    """Encode a list of G2 points into the tiled MSM layout once.

    Identity bases are encoded as projective ``(0, 1, 0)`` (Z = 0).  RCB-
    2015's complete add treats Z = 0 as the identity, so masked positions
    contribute nothing to any bucket regardless of the scalar they're
    paired with — no separate identity_mask channel needed.

    Returns:
        ``tiled_points`` : ``(tile_num, tile_length, coord=3, 2, M_fp)``
        ready to be assigned to ``ctx.points`` on the G2 MSM driver.
    """
    affine_xy = []
    identity_mask_positions = []
    for i, pt in enumerate(points):
        if isinstance(pt, G2Projective):
            pt = pt.to_affine()
        if pt.is_infinity():
            # Placeholder; we'll rewrite X = 0, Z = 0 below so the
            # encoded triple becomes (0, 1, 0) — the unique Z = 0 point
            # on the projective curve.
            affine_xy.append([G2_GEN_X, G2_GEN_Y])
            identity_mask_positions.append(i)
        else:
            affine_xy.append([pt.x, pt.y])
    while len(affine_xy) < _TPU_MSM_LENGTH:
        affine_xy.append([G2_GEN_X, G2_GEN_Y])
        identity_mask_positions.append(len(affine_xy) - 1)
    if len(affine_xy) > _TPU_MSM_LENGTH:
        raise NotImplementedError(
            f"encode_points_g2_for_msm: up to {_TPU_MSM_LENGTH} points; "
            f"got {len(affine_xy)}."
        )

    ctx = tpu_contexts.get_g2_msm_context()
    fe = ctx.ec_ctx.fe_ctx
    pts_comp = ctx.ec_ctx.to_computational_format(affine_xy)  # (3, N, 2, M)
    pts_comp = jnp.asarray(pts_comp).transpose(1, 0, 2, 3)    # (N, 3, 2, M)

    # Encode identity bases as projective (0, 1, 0).  The projective curve
    # ``Y²Z = X³ + b·Z³`` has exactly one point with Z = 0, namely
    # ``(0 : 1 : 0)``; any other (X, Y, 0) is off-curve and corrupts the
    # RCB-2015 add formula.  We rewrite both X (→ 0) and Z (→ 0) at the
    # identity slots; the placeholder Y from the generator stays (any
    # non-zero Y is fine since (0 : k : 0) = (0 : 1 : 0) in P²).
    if identity_mask_positions:
        idx = jnp.array(identity_mask_positions, dtype=jnp.int32)
        zero_fq2_x = fe.to_computational_format((0, 0))[0]   # (2, M)
        zero_fq2_z = fe.to_computational_format((0, 0))[0]   # (2, M)
        pts_comp = pts_comp.at[idx, 0].set(zero_fq2_x)
        pts_comp = pts_comp.at[idx, 2].set(zero_fq2_z)

    tiled = pts_comp.reshape(
        ctx.tile_num, ctx.tile_length, ctx.coordinate_dim,
        ctx.fe_dim, ctx.moduli_num,
    )
    return tiled


def msm_g2_tpu(points_or_tiled, scalars):
    """G2 MSM on TPU using :class:`Fq2FusionMSMContext`.

    Parameters
    ----------
    points_or_tiled
        Either a list of G2 points (``G2Affine`` / ``G2Projective``), or
        a pre-tiled ``jnp.ndarray`` of shape ``(tile_num, tile_length,
        coord, 2, M)`` produced by :func:`encode_points_g2_for_msm`.
    scalars
        Iterable of Fr scalars (Python ints).  Length must be exactly
        ``_TPU_MSM_LENGTH``; pad with zeros if fewer are supplied.

    Returns
    -------
    jnp.ndarray
        Fq2 projective computational form, shape ``(coord=3, 2, M)``.
        Decode with :func:`decode_g2` to get a ``G2Projective``.
    """
    if isinstance(points_or_tiled, jnp.ndarray):
        tiled_points = points_or_tiled
    else:
        tiled_points = encode_points_g2_for_msm(points_or_tiled)

    scalars_padded = [int(s) % _FR for s in scalars]
    while len(scalars_padded) < _TPU_MSM_LENGTH:
        scalars_padded.append(0)
    if len(scalars_padded) > _TPU_MSM_LENGTH:
        raise NotImplementedError(
            f"msm_g2_tpu: up to {_TPU_MSM_LENGTH} scalars; "
            f"got {len(scalars_padded)}."
        )

    ctx = tpu_contexts.get_g2_msm_context()
    ctx.points = tiled_points
    ctx.reset()
    tiled_slices = ctx.to_computational_format(scalars_padded)
    result = ctx.multiscalar_multiply(tiled_slices)
    result.block_until_ready()
    return result


def decode_g2(tpu_pt: jnp.ndarray) -> G2Projective:
    """Decode a projective-Fp2 DRNS array back to a ``G2Projective``.

    Accepts shapes ``(coord=3, 1, 2, M_fp)`` or ``(coord=3, 2, M_fp)``.
    The identity (``Z = 0``) decodes to the point at infinity.
    """
    ctx = tpu_contexts.get_g2_msm_context()
    if tpu_pt.ndim == 4:
        # (3, 1, 2, M) → (3, 2, M)
        tpu_pt = tpu_pt[:, 0, :, :]
    out = ctx.to_original_format(tpu_pt)
    # to_original_format returns [x_aff, y_aff] for a single 3D point;
    # identity decodes to [(0, 0), (0, 0)].
    (x_c0, x_c1), (y_c0, y_c1) = out
    x_int_pair = (int(x_c0), int(x_c1))
    y_int_pair = (int(y_c0), int(y_c1))
    if (x_int_pair == (0, 0)) and (y_int_pair == (0, 0)):
        return G2Affine(G2_GEN_X, G2_GEN_Y, infinity=True).to_projective()
    return G2Affine(x_int_pair, y_int_pair).to_projective()


def msm_g2(points, scalars) -> G2Projective:
    """TPU MSM over G2 returning a CPU ``G2Projective`` (drop-in replacement
    for the upstream CPU kernel).

    Empty inputs yield the curve identity.  All-identity inputs yield
    the identity too (the underlying TPU kernel sees masked-zero scalars).
    """
    assert len(points) == len(scalars), "msm_g2: length mismatch"
    if not points:
        return G2Affine(G2_GEN_X, G2_GEN_Y, infinity=True).to_projective()
    result = msm_g2_tpu(points, scalars)
    return decode_g2(result)


# ── helpers ───────────────────────────────────────────────────────────────────


def _zero_buckets_like(ctx, like_arr):
    """Reset a bucket-like array to the EC identity in DRNS form.

    All bucket tensors place ``coordinate_dim`` at axis 0 and
    ``moduli_num`` at axis -1, with arbitrary intermediate axes."""
    ec_ctx = ctx.ec_ctx
    zero = ec_ctx.get_finite_field_context().to_computational_format(
        ec_ctx.zero_point
    )                                              # shape (coord, mod)
    target_shape = like_arr.shape
    coord, mod = zero.shape
    assert target_shape[0] == coord and target_shape[-1] == mod, (
        f"shape mismatch: zero={zero.shape} target={target_shape}"
    )
    middle = (1,) * (len(target_shape) - 2)
    return jnp.broadcast_to(zero.reshape(coord, *middle, mod), target_shape)
