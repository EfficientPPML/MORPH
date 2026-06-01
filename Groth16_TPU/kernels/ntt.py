"""
TPU-backed NTT/INTT adapter — drop-in for the upstream
``Groth16/kernels/ntt.py`` API.

Public API matches the upstream module:
    get_omega(n: int) -> int
    ntt(a: list[int], omega: int) -> list[int]
    intt(a: list[int], omega: int) -> list[int]

Implementation routes through ``tpu_contexts.get_ntt_context(n)``,
which builds a JIT-compiled ``NTT3Step`` over BLS12-377 Fr keyed on n.
For very small n where the TPU context is overkill (or unsupported),
we fall back to the upstream CPU NTT in ``cpu_kernels.ntt``.

Nothing in this module is performance-critical at n ≤ 32; the goal is
correctness parity with the CPU NTT so the upstream protocol code is
unchanged.
"""

import os
import sys

# Ensure path_setup has been applied so that ``cpu_kernels`` and the
# MAPLE-root TPU contexts are importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
import path_setup  # noqa: F401

from cpu_kernels import ntt as _cpu_ntt
import tpu_contexts


# Sizes below this fall back to the CPU NTT.  n=1 has no nontrivial
# transform; n=2 is too small for the 3-step factoring (would require
# r=c=√2, not a power of two on each side).  n=4 and up are fine.
_MIN_TPU_N = 4


# ── public API ────────────────────────────────────────────────────────────────


def get_omega(n: int) -> int:
    """Primitive n-th root of unity in Fr.  Delegates to the upstream
    CPU implementation so the bytes are identical to what the protocol
    code computes elsewhere."""
    return _cpu_ntt.get_omega(n)


def ntt(a: list, omega: int) -> list:
    """Forward NTT.  ``omega`` is required for API parity with the CPU
    kernel; the TPU context derives its own twiddles from a cached
    primitive 2n-th root, but we sanity-check the omegas match."""
    n = len(a)
    if n < _MIN_TPU_N or (n & (n - 1)) != 0:
        return _cpu_ntt.ntt(a, omega)

    _check_omega(n, omega)
    ctx = tpu_contexts.get_ntt_context(n)
    arr = ctx.to_computational_format([int(x) for x in a])
    out = ctx.ntt(arr)
    out.block_until_ready()
    flat = [int(x) for x in ctx.to_original_format(out)]
    # NTT3Step emits results in bit-reversed order; the upstream
    # protocol expects natural order.
    return _bit_reverse(flat)


def intt(a: list, omega: int) -> list:
    """Inverse NTT.  Same omega as ntt() (the inverse is internal)."""
    n = len(a)
    if n < _MIN_TPU_N or (n & (n - 1)) != 0:
        return _cpu_ntt.intt(a, omega)

    _check_omega(n, omega)
    # Inverse direction of the forward bit-reversal: feed the upstream
    # input through a bit-reversal so NTT3Step's expected (bit-reversed
    # input) layout receives natural-ordered data.
    ctx = tpu_contexts.get_ntt_context(n)
    a_br = _bit_reverse([int(x) for x in a])
    arr = ctx.to_computational_format(a_br)
    out = ctx.intt(arr)
    out.block_until_ready()
    return [int(x) for x in ctx.to_original_format(out)]


# ── DRNS-native fast path used by the prover ─────────────────────────────────


def ntt_raw(arr, n: int):
    """Forward NTT on an already DRNS-Fr-encoded array.

    Input shape: ``(1, r, c, M_fr)`` (use ``to_computational_format`` from
    ``get_ntt_context(n)`` to produce it).  Output shape same.

    Bit-reversal is NOT applied — the result is in NTT3Step's natural
    internal order (bit-reversed wrt the natural-coeff input).  Callers
    that only do pointwise ops between :func:`ntt_raw` and :func:`intt_raw`
    don't care, since the order is consistent across A/B/C."""
    ctx = tpu_contexts.get_ntt_context(n)
    return ctx.ntt(arr)


def intt_raw(arr, n: int):
    """Inverse NTT on an array that was produced by :func:`ntt_raw`
    (or pointwise-combined siblings of one).  Returns natural-order
    coefficients in DRNS Fr, shape ``(1, r, c, M_fr)``."""
    ctx = tpu_contexts.get_ntt_context(n)
    return ctx.intt(arr)


def get_ff_ctx(n: int):
    """Return the DRNS-lazy Fr context paired with the NTT of size ``n``.

    The prover uses this for ``modular_multiply`` / ``modular_add`` /
    ``modular_subtract`` on the DRNS-Fr arrays that flow through the H
    pipeline."""
    return tpu_contexts.get_ntt_context(n)._raw.ff_ctx


# ── helpers ───────────────────────────────────────────────────────────────────


def _bit_reverse(xs: list) -> list:
    n = len(xs)
    bits = (n - 1).bit_length() if n > 1 else 0
    out = [0] * n
    for i in range(n):
        rev = 0
        v = i
        for _ in range(bits):
            rev = (rev << 1) | (v & 1)
            v >>= 1
        out[rev] = xs[i]
    return out


def _check_omega(n: int, omega: int) -> None:
    """The TPU context derives twiddles from a cached primitive 2n-th
    root psi (so its internal omega = psi^2).  Both that omega and the
    one passed in are computed from the same QNR g via
    ``g^((r-1)/n)``, so they must match bit-for-bit.  We assert this
    early so the failure mode is obvious if the caller passes a custom
    omega from a different generator."""
    expected = _cpu_ntt.get_omega(n)
    if int(omega) != int(expected):
        raise ValueError(
            f"TPU NTT adapter: omega for n={n} must equal get_omega(n).\n"
            f"  got      : {omega}\n"
            f"  expected : {expected}"
        )
