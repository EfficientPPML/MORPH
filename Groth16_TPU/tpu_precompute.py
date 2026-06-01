"""
One-time TPU encodings of the Groth16 proving key and QAP, cached so
they're paid once per process instead of per proof.

Two caches keyed by Python ``id``:

    _PK_CACHE [id(pk)]  → encoded pk-side tensors
    _QAP_CACHE[id(qap)] → encoded qap-side tensors

The caches are intentionally weak via ``id`` (no need to be smart about
GC — there's typically one pk + one qap per process for Groth16).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import path_setup  # noqa: F401

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from kernels import ec_ops as _ec_ops
from kernels.msm import encode_points_for_msm, encode_points_g2_for_msm
from kernels.ntt import get_ff_ctx, _MIN_TPU_N


# ── data classes ──────────────────────────────────────────────────────────────


@dataclass
class PKTPU:
    """Pre-encoded TPU views of the proving key.

    G1 fixed points (α, β, δ) live in Edwards 4-coord DRNS Fp,
    shape ``(4, 1, M_fp)``.

    Per-wire G1 vectors (A_g1, B_g1, private_g1, h_g1) live in the
    tiled layout expected by ``msm_g1_tpu``, shape
    ``(tile_num, tile_length, 4, M_fp)``.
    """
    alpha_g1: Any        # (4, 1, M_fp)
    beta_g1:  Any        # (4, 1, M_fp)
    delta_g1: Any        # (4, 1, M_fp)
    A_g1:        Any     # tiled (tile_num, tile_length, 4, M_fp)
    B_g1:        Any     # tiled
    private_g1:  Any     # tiled
    h_g1:        Any     # tiled
    B_g2:        Any     # (tile_num, tile_length, 3, 2, M_fp) tiled G2 bases
    beta_g2:  Any        # (3, 1, 2, M_fp)  projective-encoded G2 point
    delta_g2: Any        # (3, 1, 2, M_fp)  projective-encoded G2 point


@dataclass
class QAPTPU:
    """Pre-encoded TPU views of the QAP.

    All arrays live in DRNS-lazy Fr.

    U, V, W : shape ``(num_wires, n, M_fr)``  — qap basis polynomials
    coset_shift   : shape ``(n, M_fr)``        — [g^0, g^1, …, g^{n-1}]
    coset_unshift : shape ``(n, M_fr)``        — [g^{-0}, …, g^{-(n-1)}]
    inv_neg2      : shape ``(M_fr,)``         — (−2)⁻¹ mod r
    """
    U: Any
    V: Any
    W: Any
    coset_shift:   Any
    coset_unshift: Any
    inv_neg2:      Any
    n:             int


# ── caches ────────────────────────────────────────────────────────────────────


_PK_CACHE: dict[int, PKTPU] = {}
_QAP_CACHE: dict[int, QAPTPU] = {}


@dataclass
class CircuitTPU:
    """All one-time, witness-independent TPU artefacts for a single circuit.

    Owned by :func:`precompute_circuit`; consumed by :class:`TPUProver`
    and by the function-based provers in ``prover.py``.  Holding the
    structural shape fields (``n``, ``num_wires``, ``num_public``) here
    means no consumer needs to keep a reference to the original ``pk`` /
    ``qap`` Python objects past this point.
    """
    pk_tpu:     PKTPU
    qap_tpu:    QAPTPU
    n:          int
    num_wires:  int
    num_public: int


def precompute_circuit(pk, qap) -> CircuitTPU:
    """Encode all circuit-only TPU artefacts in one call.

    Both ``precompute_pk(pk)`` and ``precompute_qap(qap)`` depend solely
    on the circuit (proving key + QAP polys); neither touches the
    witness.  This is the one-time-per-circuit step — call it once per
    ``(pk, qap)``, hand the returned :class:`CircuitTPU` to
    :class:`TPUProver` (or to the function-based provers) and reuse for
    every subsequent prove.

    Both halves are id-cached so repeat calls are free.
    """
    return CircuitTPU(
        pk_tpu=precompute_pk(pk),
        qap_tpu=precompute_qap(qap),
        n=qap.n,
        num_wires=qap.num_wires,
        num_public=qap.num_public,
    )


def precompute_pk(pk) -> PKTPU:
    """Encode ``pk`` into TPU formats; cached on ``id(pk)``."""
    key = id(pk)
    cached = _PK_CACHE.get(key)
    if cached is not None:
        return cached

    # Encode the two fixed G2 generators as a single ``(C, 2, 2, M_fp)`` batch
    # then split — the Fq2 EC context's format conversion only fires once.
    # ``C`` is the coord dim of the G2 EC context (3 for projective RCB).
    import tpu_contexts as _tc
    _g2_ctx = _tc.get_g2_msm_context()
    _g2_pair_cf = _g2_ctx.ec_ctx.to_computational_format(
        [[pk.beta_g2.x, pk.beta_g2.y], [pk.delta_g2.x, pk.delta_g2.y]]
    )  # (C, 2, 2, M_fp)
    beta_g2_cf  = _g2_pair_cf[:, 0:1, :, :]   # (C, 1, 2, M_fp)
    delta_g2_cf = _g2_pair_cf[:, 1:2, :, :]   # (C, 1, 2, M_fp)

    enc = PKTPU(
        alpha_g1   = _ec_ops.encode_g1(pk.alpha_g1),
        beta_g1    = _ec_ops.encode_g1(pk.beta_g1),
        delta_g1   = _ec_ops.encode_g1(pk.delta_g1),
        A_g1       = encode_points_for_msm(pk.A_g1),
        B_g1       = encode_points_for_msm(pk.B_g1),
        private_g1 = encode_points_for_msm(pk.private_g1),
        h_g1       = encode_points_for_msm(pk.h_g1),
        B_g2       = encode_points_g2_for_msm(pk.B_g2),
        beta_g2    = beta_g2_cf,
        delta_g2   = delta_g2_cf,
    )
    _PK_CACHE[key] = enc
    return enc


def precompute_qap(qap) -> QAPTPU:
    """Encode ``qap`` into TPU formats; cached on ``id(qap)``."""
    key = id(qap)
    cached = _QAP_CACHE.get(key)
    if cached is not None:
        return cached

    from bls12_377.params import r as _R

    n = qap.n
    if n < _MIN_TPU_N or (n & (n - 1)) != 0:
        # Fall back: small n would route NTT to CPU anyway.  We still
        # encode the qap polys for the matvec step below so the precompute
        # is uniform, but we use the n-sized Fr context bound to n=_MIN_TPU_N
        # is not safe.  For consistency, just skip TPU precompute and let
        # the prover use the CPU path for very small n.
        raise NotImplementedError(
            f"precompute_qap: n={n} too small / not power-of-two for the TPU path"
        )

    ff = get_ff_ctx(n)

    # Encode qap.U[i][j] etc. straight as nested lists → DRNS Fr.
    # Shape after encoding: (num_wires, n, M_fr).
    U_drns = ff.to_computational_format(qap.U)
    V_drns = ff.to_computational_format(qap.V)
    W_drns = ff.to_computational_format(qap.W)

    # Coset shift vectors as plain ints, then DRNS-encode.
    g = qap.omega2n
    g_inv = pow(g, _R - 2, _R)
    shift_vec   = [pow(g, j, _R)     for j in range(n)]
    unshift_vec = [pow(g_inv, j, _R) for j in range(n)]
    coset_shift   = ff.to_computational_format(shift_vec)    # (n, M_fr)
    coset_unshift = ff.to_computational_format(unshift_vec)  # (n, M_fr)

    inv_neg2_int = pow(_R - 2, _R - 2, _R)
    inv_neg2 = ff.to_computational_format(inv_neg2_int)      # (M_fr,)

    enc = QAPTPU(
        U=U_drns, V=V_drns, W=W_drns,
        coset_shift=coset_shift,
        coset_unshift=coset_unshift,
        inv_neg2=inv_neg2,
        n=n,
    )
    _QAP_CACHE[key] = enc
    return enc


def reset_caches():
    """Drop all cached encodings (useful for tests)."""
    _PK_CACHE.clear()
    _QAP_CACHE.clear()
