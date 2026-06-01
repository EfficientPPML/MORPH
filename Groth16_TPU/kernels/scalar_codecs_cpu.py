"""High-performance CPU-side scalar codecs used by ``prove_pure_tpu_wrapper``.

Two standalone helpers, both designed to replace slower paths the wrapper
previously used:

* ``drns_encode_fr_fast(a_ints, rns_moduli, radix_bits)``
    Replaces ``ff.to_computational_format(a_ints)`` for the witness's
    DRNS encoding (the H-pipeline input).  Uses ``gmpy2.mpz`` to do
    the per-modulus big-int reduction at ~2× CPython's PyLong speed,
    and a ``ProcessPoolExecutor`` to spread the 24 moduli across CPU
    cores.

* ``slice_scalars_cpu_fast(a_ints, identity_mask=...)``
    Replaces ``encode_scalars_for_msm_g1_packed`` (T3's packed path)
    for the G1 MSM scalars.  Goes through ``int.to_bytes(32, 'little')``
    + ``np.frombuffer`` to pack each scalar in one C call, then does
    the 26 × 10-bit window extraction via SIMD-vectorised NumPy ops.
    No multiprocessing — the work is sub-millisecond and pool dispatch
    would dominate.

Both return ``jnp.ndarray`` on the default TPU device so the caller
can drop the result straight into a TPU op without an extra device
transfer.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
import path_setup  # noqa: F401

import functools
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# gmpy2 is already a dependency elsewhere in the repo (see
# elliptic_curve_context.py); pull it lazily so this module still
# imports if gmpy2 is missing — we fall back to CPython int.
try:
    import gmpy2
    _HAVE_GMPY2 = True
except Exception:
    _HAVE_GMPY2 = False

from bls12_377.params import r as _FR


# ─────────────────────────────────────────────────────────────────────────────
# DRNS encode — replaces ff.to_computational_format(a_ints)
# ─────────────────────────────────────────────────────────────────────────────


# Match the forkserver context used elsewhere in the codebase (avoids
# JAX + fork deadlock).
_MP_CONTEXT = multiprocessing.get_context("forkserver")


def _drns_column_worker(args):
    """Compute one DRNS column = residues of all scalars mod a single m_i.

    Returns a length-N list of uint32 (Python ints fitting in 32 bits).
    Uses gmpy2 if available — saves ~50 % per-op vs CPython big-int %.
    """
    scalars_mpz, m_i, radix_bits, identity_mask = args
    m_mpz = gmpy2.mpz(m_i) if _HAVE_GMPY2 else m_i
    shift = 1 << radix_bits
    if _HAVE_GMPY2:
        # gmpy2 path: keeps everything in mpz which has faster mod than
        # CPython PyLong for ~250-bit inputs and ~28-bit moduli.
        if identity_mask is None:
            out = [int(((s % m_mpz) * shift) % m_mpz) for s in scalars_mpz]
        else:
            out = [
                0 if (i < len(identity_mask) and identity_mask[i])
                else int(((s % m_mpz) * shift) % m_mpz)
                for i, s in enumerate(scalars_mpz)
            ]
    else:
        if identity_mask is None:
            out = [int(((int(s) % m_i) * shift) % m_i) for s in scalars_mpz]
        else:
            out = [
                0 if (i < len(identity_mask) and identity_mask[i])
                else int(((int(s) % m_i) * shift) % m_i)
                for i, s in enumerate(scalars_mpz)
            ]
    return out


# Total scalar·modulus mod ops below which the single-threaded gmpy2
# path beats spawning a ProcessPoolExecutor (forkserver startup ~200ms
# vs ~5ms of actual mod work for N·M ≈ 25K).  Picked empirically; the
# pool wins clearly once total ops > ~50K.
_PARALLEL_OPS_THRESHOLD = 50_000


# ── NumPy-vectorised DRNS encode via 4 × uint64 limbs ───────────────────────


_FR_LIMB_BYTES   = 8                    # 64-bit limbs
_FR_NUM_LIMBS    = 4                    # 4 × 64 = 256 bits, fits a 253-bit Fr
_FR_TOTAL_BYTES  = _FR_LIMB_BYTES * _FR_NUM_LIMBS


@functools.lru_cache(maxsize=None)
def _pow2_modm_table(rns_moduli_t):
    """Precomputed table ``T[j, k] = (2^{64·k}) mod m_j``.

    Keyed on a tuple of moduli (so :func:`functools.lru_cache` can hash it).
    For 24 moduli × 4 limbs this is 96 small ints — built once per (moduli)
    set and reused across every encode call.
    """
    import numpy as _np
    M = len(rns_moduli_t)
    out = _np.empty((M, _FR_NUM_LIMBS), dtype=_np.uint64)
    for j, m in enumerate(rns_moduli_t):
        cur = 1
        for k in range(_FR_NUM_LIMBS):
            out[j, k] = cur
            cur = (cur * (1 << 64)) % m
    return out


def drns_encode_fr_numpy(
    a_ints: list,
    rns_moduli: list,
    radix_bits: int,
    target_length: int = None,
    identity_mask: tuple = None,
) -> jnp.ndarray:
    """Pure-NumPy vectorised DRNS encode.

    Pipeline:

      1. ``int.to_bytes(32, "little")`` per scalar (1 C call each)
         packs 1024 × 253-bit scalars into a single 32 KB bytearray.
      2. ``np.frombuffer`` views the buffer as a ``(N, 4)`` uint64
         limb array — zero-copy.
      3. ``limbs[:, None, :] % m[None, :, None]`` reduces every
         (scalar, modulus, limb) triple — single broadcast op.
      4. Multiply by ``pow2_modm[j, k] = 2^{64·k} mod m_j`` and sum
         across limbs to get ``v % m_j`` for every (scalar, modulus).
      5. Apply the Montgomery shift: ``(v * 2^{radix_bits}) % m``.

    All array shapes are bounded by 2⁵⁶, so uint64 NumPy arithmetic
    can't overflow.  Total work for ``N=1024, M=24`` is ~5 vectorised
    ops on arrays of ≤ ``1024 × 24 × 4`` elements.
    """
    import numpy as _np
    M = len(rns_moduli)
    N = len(a_ints)
    if target_length is None:
        target_length = N

    # ── 1. & 2. Pack 1024 × 253-bit ints into (target_length, 4) uint64
    raw = bytearray(target_length * _FR_TOTAL_BYTES)
    for i in range(N):
        if identity_mask is not None and i < len(identity_mask) and identity_mask[i]:
            continue
        v = int(a_ints[i]) % _FR
        raw[i * _FR_TOTAL_BYTES:(i + 1) * _FR_TOTAL_BYTES] = v.to_bytes(
            _FR_TOTAL_BYTES, "little"
        )
    limbs = _np.frombuffer(raw, dtype=_np.uint64).reshape(
        target_length, _FR_NUM_LIMBS
    )                                                     # (N, 4)

    m_arr = _np.asarray(rns_moduli, dtype=_np.uint64)     # (M,)
    pow2_modm = _pow2_modm_table(tuple(rns_moduli))       # (M, 4)

    # ── 3. limb % m for every (scalar, modulus, limb)
    reduced = limbs[:, None, :] % m_arr[None, :, None]    # (N, M, 4) uint64, ≤ m_j ≤ 2^28

    # ── 4. weighted sum across limbs:  Σ_k (limb_k % m_j) · pow2_modm[j, k]
    weighted = reduced * pow2_modm[None, :, :]            # (N, M, 4) ≤ 2^56
    summed   = weighted.sum(axis=2) % m_arr[None, :]      # (N, M) uint64

    # ── 5. Montgomery shift:  (v * 2^{radix_bits}) % m
    shifted = (summed << _np.uint64(radix_bits)) % m_arr[None, :]   # (N, M) uint64 ≤ m_j

    return jnp.asarray(shifted.astype(_np.uint32)).to_device(
        jax.devices("tpu")[0]
    )


def drns_encode_fr_fast(
    a_ints: list,
    rns_moduli: list,
    radix_bits: int,
    target_length: int = None,
    identity_mask: tuple = None,
    num_workers: int = None,
) -> jnp.ndarray:
    """Encode a list of Fr scalars into DRNS computational form.

    Parameters
    ----------
    a_ints
        Python list of Fr scalars (will be reduced mod ``_FR``).
    rns_moduli
        The DRNS moduli ``m_0 … m_{M-1}`` (each ~28-bit prime).  Read
        from ``ff_ctx.rns_moduli``.
    radix_bits
        Montgomery radix bit-width (typically 32).  Read from
        ``ff_ctx.radix_bits``.
    target_length
        If set, the output is right-padded with zero rows up to this
        length.  Used to make the result match the upstream MSM /
        H-pipeline shape expectations.  Default = ``len(a_ints)``.
    identity_mask
        Optional length-N iterable of bools; positions where the mask
        is True are encoded as the zero residue (used by the MSM path
        to neutralise identity-point placeholders).
    num_workers
        Pool size for the column-parallel pass.  Default picks
        ``min(cpu_count, 4, M)``.

    Returns
    -------
    jnp.ndarray of shape ``(target_length, len(rns_moduli))`` uint32,
    on the default TPU device.
    """
    M = len(rns_moduli)
    N = len(a_ints)
    if target_length is None:
        target_length = N
    if target_length < N:
        raise ValueError(
            f"target_length ({target_length}) < len(a_ints) ({N})"
        )

    total_ops = N * M
    if num_workers is None:
        # Skip the pool entirely for small inputs — forkserver pool
        # startup is ~200 ms and the math at N=1024 is ~5 ms.
        if total_ops < _PARALLEL_OPS_THRESHOLD:
            num_workers = 1
        else:
            num_workers = max(1, min(os.cpu_count() or 1, 4, M))

    if num_workers <= 1:
        # Inline-fast path — straight CPython big-int, no gmpy2 mpz
        # conversion (which costs more than it saves at N ≈ 1024 with
        # small ~28-bit moduli; gmpy2 wins are bigger on much wider
        # operands).  Direct rewrite of
        # ``DRNSlazyContext.to_computational_format``'s individual_convert
        # in a flat row-major loop.
        shift = 1 << radix_bits
        out = np.zeros((target_length, M), dtype=np.uint32)
        if identity_mask is None:
            for i, a in enumerate(a_ints):
                v = int(a) % _FR
                row = out[i]
                for j, m in enumerate(rns_moduli):
                    row[j] = ((v % m) * shift) % m
        else:
            for i, a in enumerate(a_ints):
                if i < len(identity_mask) and identity_mask[i]:
                    continue
                v = int(a) % _FR
                row = out[i]
                for j, m in enumerate(rns_moduli):
                    row[j] = ((v % m) * shift) % m
        return jnp.asarray(out).to_device(jax.devices("tpu")[0])

    # Pool path — gmpy2 conversion pays off because workers do many
    # passes over the same scalars before returning.
    if _HAVE_GMPY2:
        scalars_mpz = [gmpy2.mpz(int(a) % _FR) for a in a_ints]
    else:
        scalars_mpz = [int(a) % _FR for a in a_ints]
    work_items = [
        (scalars_mpz, m_i, radix_bits, identity_mask)
        for m_i in rns_moduli
    ]
    with ProcessPoolExecutor(max_workers=num_workers,
                             mp_context=_MP_CONTEXT) as pool:
        columns = list(pool.map(_drns_column_worker, work_items))

    out = np.zeros((target_length, M), dtype=np.uint32)
    for j, col in enumerate(columns):
        out[:N, j] = col

    return jnp.asarray(out).to_device(jax.devices("tpu")[0])


# ─────────────────────────────────────────────────────────────────────────────
# Scalar slicer — replaces encode_scalars_for_msm_g1_packed
# ─────────────────────────────────────────────────────────────────────────────


_FR_CHUNK_BITS = 32
_FR_NUM_CHUNKS = 8                # 8 × 32 = 256, plenty for 253-bit Fr
_FR_PACKED_BYTES = _FR_NUM_CHUNKS * (_FR_CHUNK_BITS // 8)   # 32 bytes / row


def slice_scalars_cpu_fast(
    a_ints: list,
    identity_mask: tuple = None,
    scalar_bits: int = 253,
    slice_bits: int = 10,
    target_length: int = 1024,
) -> jnp.ndarray:
    """Pure-CPU window-slice extraction for the G1 MSM kernel.

    Replaces the three-step
    ``_pad_scalars → encode_scalars_packed → slice_scalars_jit_g1`` chain
    used by ``encode_scalars_for_msm_g1_packed``.  Same output shape
    (``(1, ceil(scalar_bits/slice_bits), target_length)`` int32), but
    runs entirely on CPU using:

      1. ``(int(s) % r).to_bytes(32, 'little')`` to pack each scalar's
         bits into 32 bytes in one C call,
      2. ``np.frombuffer(buf, dtype=uint32)`` to view those bytes as
         8 × uint32 limbs (zero-copy),
      3. NumPy SIMD-vectorised ``>>`` / ``&`` / ``|`` per window to
         extract the 26 × 10-bit slices.

    No multiprocessing — the entire pipeline runs in <2 ms on 1024
    scalars and pool dispatch would dominate.

    Returns
    -------
    jnp.ndarray on TPU device, shape ``(1, window_num, target_length)``
    int32, ready for ``msm_g1_pure_tpu``.
    """
    window_num = math.ceil(scalar_bits / slice_bits)
    N = len(a_ints)
    if N > target_length:
        raise ValueError(
            f"slice_scalars_cpu_fast: up to {target_length} scalars; got {N}."
        )

    # ── Step 1+2: pack into (target_length, 8) uint32 via to_bytes
    # `to_bytes` is a single C call per scalar.  For 1024 scalars
    # that's ~1024 C calls → low-µs territory.
    packed_bytes = bytearray(target_length * _FR_PACKED_BYTES)
    for i in range(N):
        if identity_mask is not None and i < len(identity_mask) and identity_mask[i]:
            continue
        v = int(a_ints[i]) % _FR
        # to_bytes is C-implemented; little-endian matches NumPy uint32 view.
        packed_bytes[i * _FR_PACKED_BYTES:(i + 1) * _FR_PACKED_BYTES] = \
            v.to_bytes(_FR_PACKED_BYTES, "little")
    packed = np.frombuffer(packed_bytes, dtype=np.uint32).reshape(
        target_length, _FR_NUM_CHUNKS
    )

    # ── Step 3: 26 × 10-bit window extraction, fully vectorised
    windows = np.empty((window_num, target_length), dtype=np.int32)
    for w in range(window_num):
        start    = w * slice_bits
        chunk_lo = start // _FR_CHUNK_BITS
        offset   = start %  _FR_CHUNK_BITS
        bits_lo  = min(slice_bits, _FR_CHUNK_BITS - offset)
        low      = (packed[:, chunk_lo] >> np.uint32(offset))
        if bits_lo < slice_bits:
            bits_hi = slice_bits - bits_lo
            if chunk_lo + 1 < _FR_NUM_CHUNKS:
                high = packed[:, chunk_lo + 1] & np.uint32((1 << bits_hi) - 1)
                slice_val = (low & np.uint32((1 << bits_lo) - 1)) \
                            | (high << np.uint32(bits_lo))
            else:
                slice_val = low & np.uint32((1 << bits_lo) - 1)
        else:
            slice_val = low & np.uint32((1 << slice_bits) - 1)
        windows[w] = slice_val.astype(np.int32)

    return jnp.asarray(windows[np.newaxis, :, :])
