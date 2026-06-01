"""
Groth16 Full-ZK Prover — TPU-native variant.

Compared to the upstream pure-Python reference, this prover:

  * Pre-encodes the proving key and QAP into TPU-native formats once
    (cached on id(pk)/id(qap)) — see tpu_precompute.precompute_pk and
    precompute_qap.

  * Computes H(x) entirely in DRNS-lazy Fr on the TPU.  combine_polys
    is a single DRNS matvec; the coset shift, NTT, pointwise
    (A·B − C)·(−2)⁻¹, INTT, and un-shift all run on DRNS Fr arrays.

  * Combines G1 EC points in Extended Twisted Edwards 4-coord DRNS via
    ``ec_ops.point_add_tpu`` and ``ec_ops.scalar_mul_tpu`` — no
    intermediate Weierstrass affine decode/encode.

  * G2 stays on CPU (the TPU MSM stack is single-base-field only), and
    the verifier's pairing is unchanged.

Timing
------
``prove()`` prints two numbers:

  * TPU init+compile (≈ JIT warmup + pk/qap precompute on first call).
  * proof generation  (steady-state per-proof cost).

Proof equations (Groth16, full ZK):
    π_A = [α]₁  +  Σᵢ aᵢ·[Uᵢ(τ)]₁            +  r_blind·[δ]₁
    π_B = [β]₂  +  Σᵢ aᵢ·[Vᵢ(τ)]₂            +  s_blind·[δ]₂
    π_C = (Σᵢ∈priv aᵢ·[(β·Uᵢ+α·Vᵢ+Wᵢ)(τ)/δ]₁
          +  Σⱼ hⱼ·[τʲ·t(τ)/δ]₁)
          +  s_blind·π_A  +  r_blind·[π_B]₁  −  r_blind·s_blind·[δ]₁
"""

import functools
import secrets
import time
from dataclasses import dataclass
from typing      import List

import jax
import jax.numpy as jnp

from bls12_377.params import r
from bls12_377.g1     import G1Affine
from bls12_377.g2     import G2Affine
from groth16.qap      import QAP
from groth16.setup    import ProvingKey
from kernels.ntt      import _MIN_TPU_N, ntt_raw, intt_raw, get_ff_ctx
from kernels.msm      import (msm_g1_tpu, msm_g1_pure_tpu,
                              msm_g2_tpu, decode_g2,
                              encode_scalars_for_msm_g1,
                              encode_scalars_for_msm_g1_packed,
                              _TPU_MSM_LENGTH)
from kernels.scalar_codecs_cpu import (drns_encode_fr_numpy,
                                       slice_scalars_cpu_fast)
from kernels          import ec_ops
from kernels          import ec_ops_g2
import tpu_contexts
import tpu_precompute


@dataclass
class Proof:
    """A Groth16 proof: three elliptic curve points."""
    A: G1Affine   # π_A ∈ G1
    B: G2Affine   # π_B ∈ G2
    C: G1Affine   # π_C ∈ G1


# ── warmup ───────────────────────────────────────────────────────────────────


def _warmup_tpu_kernels(n: int) -> None:
    """Build + JIT-compile the TPU MSM-G1, NTT, EC, and Fr DRNS kernels."""
    tpu_contexts.get_msm_context(_TPU_MSM_LENGTH)

    if n >= _MIN_TPU_N and (n & (n - 1)) == 0:
        ntt_ctx = tpu_contexts.get_ntt_context(n)
        dummy = ntt_ctx.to_computational_format([0] * n)
        ntt_ctx.ntt(dummy).block_until_ready()
        ntt_ctx.intt(dummy).block_until_ready()

        # Warm the Fr DRNS ops we use in the H pipeline.
        ff = get_ff_ctx(n)
        a = ff.to_computational_format([0] * n)
        b = ff.to_computational_format([1] * n)
        ff.modular_multiply(a, b).block_until_ready()
        ff.modular_add(a, b).block_until_ready()
        ff.modular_subtract(a, b).block_until_ready()

    # Warm point_add on (4, 1, M_fp).
    ec_ops.warmup()


# ── DRNS-Fr H computation ────────────────────────────────────────────────────


def _combine_polys_drns(polys_drns, a_drns, ff):
    """Σᵢ aᵢ · polys[i] as DRNS Fr coefficient vectors.

    polys_drns : (m, n, M_fr) — qap.U/V/W pre-encoded
    a_drns     : (m, M_fr)    — witness pre-encoded
    Returns      (n, M_fr)
    """
    # Broadcast a to (m, 1, M_fr) and multiply against polys.
    a_b = a_drns.reshape(a_drns.shape[0], 1, a_drns.shape[-1])
    prod = ff.modular_multiply(polys_drns, a_b)             # (m, n, M_fr)

    # Sum-reduce along axis 0 using a balanced tree of modular_add.
    # Each DRNS slot is uint32 with per-modulus value ~m_i (≈ 2^28); a
    # naive lazy-add lets the slot grow as #-of-additions · m_i, which
    # overflows uint32 once we accumulate more than ~16 values per slot
    # (10-level reduce for m = 1024 wires goes to 2^10 · 2^28 = 2^38).
    # Re-canonicalise after every level so each slot stays < m_i.
    arr = prod
    while arr.shape[0] > 1:
        half = arr.shape[0] // 2
        a, b = arr[:half], arr[half:2 * half]
        rest = arr[2 * half:] if arr.shape[0] > 2 * half else None
        merged = ff.modular_add(a, b)
        merged = ff.modular_reduce(merged)
        arr = merged if rest is None else jnp.concatenate([merged, rest], axis=0)
    return arr[0]                                            # (n, M_fr)


def _ntt_input_shape(arr_n_m, n: int):
    """Reshape an ``(n, M_fr)`` DRNS Fr vector into ``(1, r, c, M_fr)``
    for direct NTT3Step entry."""
    ctx = tpu_contexts.get_ntt_context(n)
    return arr_n_m.reshape(1, ctx.r, ctx.c, arr_n_m.shape[-1])


def _ntt_output_to_flat(arr_1rcM, n: int):
    """Flatten a ``(1, r, c, M_fr)`` NTT result back to ``(n, M_fr)``."""
    return arr_1rcM.reshape(n, arr_1rcM.shape[-1])


def compute_h_coset_ntt_drns(a_drns, qap_tpu, ff):
    """H(x) = (A·B − C)/t on the coset, entirely in DRNS Fr.

    a_drns  : witness, shape (num_wires, M_fr)
    qap_tpu : QAPTPU with U/V/W/coset_shift/coset_unshift/inv_neg2

    Returns the H coefficient vector as a Python int list of length n-1
    (decode happens at the boundary since the next consumer is the MSM
    over h_g1 which wants plain ints).
    """
    n = qap_tpu.n

    # ── 1. A, B, C in coefficient domain (DRNS Fr).
    A_co = _combine_polys_drns(qap_tpu.U, a_drns, ff)        # (n, M_fr)
    B_co = _combine_polys_drns(qap_tpu.V, a_drns, ff)
    C_co = _combine_polys_drns(qap_tpu.W, a_drns, ff)

    # ── 2. Coset shift: multiply coeff[j] by g^j (pointwise).
    A_sh = ff.modular_multiply(A_co, qap_tpu.coset_shift)
    B_sh = ff.modular_multiply(B_co, qap_tpu.coset_shift)
    C_sh = ff.modular_multiply(C_co, qap_tpu.coset_shift)

    # ── 3. NTT (skip bit-reversal — order is consistent across A,B,C
    #         and INTT will undo it).
    A_ev = ntt_raw(_ntt_input_shape(A_sh, n), n)
    B_ev = ntt_raw(_ntt_input_shape(B_sh, n), n)
    C_ev = ntt_raw(_ntt_input_shape(C_sh, n), n)

    # ── 4. Pointwise H_eval = (A·B − C) · inv_neg2 on the coset.
    AB = ff.modular_multiply(A_ev, B_ev)
    AB_minus_C = ff.modular_subtract(AB, C_ev)
    # inv_neg2 has shape (M_fr,); broadcast it across (1, r, c, M_fr).
    H_ev = ff.modular_multiply(AB_minus_C, qap_tpu.inv_neg2)

    # ── 5. INTT back to coefficient domain (natural order).
    H_co_rc = intt_raw(H_ev, n)                              # (1, r, c, M_fr)
    H_co = _ntt_output_to_flat(H_co_rc, n)                   # (n, M_fr)

    # ── 6. Un-shift: divide coeff[j] by g^j.
    H_co = ff.modular_multiply(H_co, qap_tpu.coset_unshift)

    H_co.block_until_ready()

    # ── 7. Decode to plain ints (length n-1; h has degree ≤ n-2).
    h_list = ff.to_original_format(H_co)
    return [int(x) for x in h_list[:n - 1]]


# ── DRNS-Fr H computation, JIT-fused (used by prove_tpu_opt) ─────────────────


def _combine_polys_drns_static(polys_drns, a_drns, ff):
    """Same Σᵢ aᵢ·polys[i] reduction as ``_combine_polys_drns`` but using
    the *raw* ``_modular_*`` methods so it traces cleanly inside a JIT.

    Re-canonicalises after every reduce level — without it, summing many
    wires (e.g. 1024 at n=1024 padded) overflows the uint32 DRNS slots.
    See ``_combine_polys_drns`` for the math.
    """
    a_b = a_drns.reshape(a_drns.shape[0], 1, a_drns.shape[-1])
    prod = ff._modular_multiply(polys_drns, a_b)
    arr = prod
    while arr.shape[0] > 1:
        half = arr.shape[0] // 2
        a, b = arr[:half], arr[half:2 * half]
        rest = arr[2 * half:] if arr.shape[0] > 2 * half else None
        merged = ff._modular_add(a, b)
        merged = ff._modular_reduce(merged)
        arr = merged if rest is None else jnp.concatenate([merged, rest], axis=0)
    return arr[0]


@functools.lru_cache(maxsize=None)
def _h_pre_ntt_kernel(n: int, ntt_r: int, ntt_c: int):
    """JIT'd kernel: combine_polys + coset_shift + reshape for NTT entry.

    Output: (A_in_rc, B_in_rc, C_in_rc) each of shape (1, r, c, M_fr).
    """
    ff = get_ff_ctx(n)

    def inner(U, V, W, a_drns, coset_shift):
        A_co = _combine_polys_drns_static(U, a_drns, ff)
        B_co = _combine_polys_drns_static(V, a_drns, ff)
        C_co = _combine_polys_drns_static(W, a_drns, ff)
        A_sh = ff._modular_multiply(A_co, coset_shift)
        B_sh = ff._modular_multiply(B_co, coset_shift)
        C_sh = ff._modular_multiply(C_co, coset_shift)
        m = A_sh.shape[-1]
        return (A_sh.reshape(1, ntt_r, ntt_c, m),
                B_sh.reshape(1, ntt_r, ntt_c, m),
                C_sh.reshape(1, ntt_r, ntt_c, m))

    return jax.jit(inner)


@functools.lru_cache(maxsize=None)
def _h_pointwise_kernel(n: int):
    """JIT'd kernel: (A·B − C) · inv_neg2 on the coset (post-NTT)."""
    ff = get_ff_ctx(n)

    def inner(A_ev, B_ev, C_ev, inv_neg2):
        AB = ff._modular_multiply(A_ev, B_ev)
        AB_minus_C = ff._modular_subtract(AB, C_ev)
        return ff._modular_multiply(AB_minus_C, inv_neg2)

    return jax.jit(inner)


@functools.lru_cache(maxsize=None)
def _h_post_intt_kernel(n: int):
    """JIT'd kernel: flatten the (1, r, c, M) INTT result then un-shift."""
    ff = get_ff_ctx(n)

    def inner(H_co_rc, coset_unshift):
        m = H_co_rc.shape[-1]
        H_co = H_co_rc.reshape(n, m)
        return ff._modular_multiply(H_co, coset_unshift)

    return jax.jit(inner)


def compute_h_coset_ntt_drns_opt(a_drns, qap_tpu):
    """Like ``compute_h_coset_ntt_drns`` but only the NTT/INTT remain
    as separate JAX calls — everything else is fused into three JIT'd
    kernels and the final DRNS→int conversion is deferred to the caller.

    Returns the DRNS-encoded H coefficients of shape ``(n, M_fr)``.
    """
    n = qap_tpu.n
    ntt_ctx = tpu_contexts.get_ntt_context(n)

    pre   = _h_pre_ntt_kernel(n, ntt_ctx.r, ntt_ctx.c)
    point = _h_pointwise_kernel(n)
    post  = _h_post_intt_kernel(n)

    A_in_rc, B_in_rc, C_in_rc = pre(
        qap_tpu.U, qap_tpu.V, qap_tpu.W, a_drns, qap_tpu.coset_shift,
    )
    A_ev = ntt_raw(A_in_rc, n)
    B_ev = ntt_raw(B_in_rc, n)
    C_ev = ntt_raw(C_in_rc, n)
    H_ev = point(A_ev, B_ev, C_ev, qap_tpu.inv_neg2)
    H_co_rc = intt_raw(H_ev, n)
    return post(H_co_rc, qap_tpu.coset_unshift)


def h_drns_to_int_list(H_co_drns, ff, n: int) -> List[int]:
    """Decode the DRNS-Fr H polynomial back to a Python int list of length
    ``n - 1`` (h has degree ≤ n - 2).  Kept explicit so the caller can
    overlap this with other CPU work if desired.
    """
    H_co_drns.block_until_ready()
    h_list = ff.to_original_format(H_co_drns)
    return [int(x) for x in h_list[:n - 1]]


# ── prove ────────────────────────────────────────────────────────────────────


def prove(pk: ProvingKey, qap: QAP, witness: List[int]) -> Proof:
    """Generate a full-ZK Groth16 proof using TPU-native kernels."""
    n          = qap.n
    num_public = qap.num_public

    # ── TPU init + JIT compile (one-time per process) ───────────────────
    init_t0 = time.time()
    _warmup_tpu_kernels(n)
    pk_tpu  = tpu_precompute.precompute_pk(pk)
    qap_tpu = tpu_precompute.precompute_qap(qap)
    init_time = time.time() - init_t0
    print(f"  [prove] TPU init+compile : {init_time:.2f}s")

    # Witness reduced mod r, both as ints (for MSM) and DRNS Fr (for H).
    a_ints = [int(w) % r for w in witness]
    ff = get_ff_ctx(n)
    a_drns = ff.to_computational_format(a_ints)              # (m, M_fr)


    # ── Actual proof generation timer ───────────────────────────────────
    run_t0 = time.time()


    # ZK blinders.
    r_blind = secrets.randbelow(r - 1) + 1
    s_blind = secrets.randbelow(r - 1) + 1

    # ── π_A = α + Σ aᵢ·A_g1[i] + r_blind·δ ──────────────────────────────
    A_sum = msm_g1_tpu(pk_tpu.A_g1, a_ints)                  # (4, M_fp)
    pi_A_tpu = ec_ops.point_add_tpu(
        ec_ops.point_add_tpu(pk_tpu.alpha_g1, A_sum),
        ec_ops.scalar_mul_tpu(pk_tpu.delta_g1, r_blind),
    )

    # ── [π_B]₁ (G1 counterpart of π_B, needed for π_C blinding) ─────────
    B_sum_g1 = msm_g1_tpu(pk_tpu.B_g1, a_ints)
    # NB: pk doesn't carry a separate β_g1 in the TPU cache because the
    # current ProvingKey stores it; encode it lazily here.
    # (TPU cache could carry it; small enough not to matter.)
    pi_B_g1_tpu = ec_ops.point_add_tpu(
        ec_ops.point_add_tpu(ec_ops.encode_g1(pk.beta_g1), B_sum_g1),
        ec_ops.scalar_mul_tpu(pk_tpu.delta_g1, s_blind),
    )

    # ── π_B on G2 (TPU XYZZ-Weierstrass Fp2 path) ───────────────────────
    B_sum_g2_tpu = msm_g2_tpu(pk_tpu.B_g2, a_ints)
    B_sum_g2 = decode_g2(B_sum_g2_tpu)
    pi_B_proj = (pk.beta_g2.to_projective()
                 .add(B_sum_g2)
                 .add(pk.delta_g2.to_projective().scalar_mul(s_blind)))

    # ── H(x) on TPU; result decoded to ints for the h MSM ───────────────
    h_coeffs = compute_h_coset_ntt_drns(a_drns, qap_tpu, ff)
    h_sum = msm_g1_tpu(pk_tpu.h_g1, h_coeffs)

    # ── private wire MSM ────────────────────────────────────────────────
    private_idxs = list(range(1, qap.num_wires - num_public))
    private_scalars = [a_ints[i] for i in private_idxs]
    private_sum = msm_g1_tpu(pk_tpu.private_g1, private_scalars)

    # ── π_C = private_sum + h_sum + s_blind·π_A + r_blind·[π_B]₁
    #         − r_blind·s_blind·δ                                ──────────
    rs = (r - r_blind * s_blind % r) % r
    pi_C_tpu = ec_ops.point_add_tpu(
        ec_ops.point_add_tpu(
            ec_ops.point_add_tpu(private_sum, h_sum),
            ec_ops.scalar_mul_tpu(pi_A_tpu, s_blind),
        ),
        ec_ops.point_add_tpu(
            ec_ops.scalar_mul_tpu(pi_B_g1_tpu, r_blind),
            ec_ops.scalar_mul_tpu(pk_tpu.delta_g1, rs),
        ),
    )

    run_time = time.time() - run_t0
    # ── Decode the two G1 outputs at the boundary ───────────────────────
    pi_A = ec_ops.decode_g1(pi_A_tpu)
    pi_C = ec_ops.decode_g1(pi_C_tpu)
    pi_B = pi_B_proj.to_affine()

    proof = Proof(A=pi_A, B=pi_B, C=pi_C)

    print(f"  [prove] proof generation : {run_time:.2f}s")
    return proof


# ── prove_tpu_opt ────────────────────────────────────────────────────────────


def prove_tpu_opt(pk: ProvingKey, qap: QAP, witness: List[int]) -> Proof:
    """Optimised Groth16 prover that keeps every per-proof EC composition
    on the TPU as a single JIT-compiled kernel.

    Differences vs :func:`prove`:

      * **β_g1 / β_g2 / δ_g2** are pre-encoded in :class:`PKTPU` rather
        than re-encoded each call.
      * **π_A**, **π_B_g1**, **π_C** (G1 ETE) and **π_B_g2** (G2 XYZZ) are
        each a single fused JIT kernel containing the binary-ladder
        scalar mul plus the point adds.  This collapses ~2500 per-proof
        JAX dispatches into 4 kernel launches.
      * **H(x)** computation around the NTT/INTT is JIT-fused into three
        kernels (pre-NTT, pointwise, post-INTT); the DRNS→int conversion
        that feeds the h MSM is moved out to an explicit step.
      * **G2 π_B** stays in XYZZ DRNS form through to the final decode
        — no intermediate CPU Fp2-projective round-trip.
    """
    n          = qap.n
    num_public = qap.num_public

    # ── Circuit-only precompute (one-time per (pk, qap)) ────────────────
    # No witness dependency.  Both halves are cached on ``id(pk)`` /
    # ``id(qap)`` so repeat calls are free.  JIT caches for the per-prove
    # kernels populate lazily on first call.
    init_t0 = time.time()
    pk_tpu, qap_tpu = tpu_precompute.precompute_circuit(pk, qap)
    init_time = time.time() - init_t0
    print(f"  [prove_opt] circuit precompute : {init_time:.2f}s")

    # Witness reduced mod r, both as ints (for MSM) and DRNS Fr (for H).
    a_ints = [int(w) % r for w in witness]
    ff = get_ff_ctx(n)
    a_drns = ff.to_computational_format(a_ints)              # (m, M_fr)

    # ── Actual proof generation timer ───────────────────────────────────
    run_t0 = time.time()

    # ZK blinders + the rs constant used in the π_C kernel.
    r_blind = secrets.randbelow(r - 1) + 1
    s_blind = secrets.randbelow(r - 1) + 1
    rs      = (r - r_blind * s_blind % r) % r

    # Decompose to MSB-first bit arrays so each JIT'd scalar mul has a
    # stable (S,) int8 input shape.
    r_bits  = ec_ops.int_to_bits_msb(r_blind)
    s_bits  = ec_ops.int_to_bits_msb(s_blind)
    rs_bits = ec_ops.int_to_bits_msb(rs)

    # ── G1 MSMs.  Pre-encode the witness scalars once per identity mask
    #    so the Python pad+DRNS step doesn't repeat across A and B.  The
    #    A_g1 and B_g1 pk vectors share the same identity mask (both cover
    #    every wire and have no Groth16-trusted-setup-induced infinity
    #    points by construction), so a single encoding feeds both.
    #    T3 path: skip the Python ``utils.slice_scalars`` loop by going
    #    through the packed-uint32 representation and a JIT'd slicer. ──
    a_ints_g1_A = encode_scalars_for_msm_g1_packed(
        a_ints, identity_mask=pk_tpu.A_g1[1]
    )
    a_ints_g1_B = encode_scalars_for_msm_g1_packed(
        a_ints, identity_mask=pk_tpu.B_g1[1]
    )
    A_sum    = ec_ops._as_4_1_M(msm_g1_tpu(pk_tpu.A_g1, a_ints_g1_A))
    B_sum_g1 = ec_ops._as_4_1_M(msm_g1_tpu(pk_tpu.B_g1, a_ints_g1_B))

    private_idxs   = list(range(1, qap.num_wires - num_public))
    private_sum    = ec_ops._as_4_1_M(
        msm_g1_tpu(pk_tpu.private_g1, [a_ints[i] for i in private_idxs])
    )

    # ── H(x) computation — JIT'd pre/post-NTT, NTT/INTT separate ───────
    h_co_drns = compute_h_coset_ntt_drns_opt(a_drns, qap_tpu)
    # Explicit DRNS→int conversion between the H kernel and the h MSM.
    h_coeffs  = h_drns_to_int_list(h_co_drns, ff, n)
    h_sum     = ec_ops._as_4_1_M(msm_g1_tpu(pk_tpu.h_g1, h_coeffs))

    # ── G2 MSM (stays as XYZZ DRNS array) ──────────────────────────────
    B_sum_g2_tpu = msm_g2_tpu(pk_tpu.B_g2, a_ints)     # (4, 1, 2, M_fp)

    # ── π_A (one JIT kernel: α + A_sum + r·δ) ──────────────────────────
    pi_A_tpu = ec_ops.affine_combo_g1_tpu(
        pk_tpu.alpha_g1, A_sum, pk_tpu.delta_g1, r_bits,
    )

    # ── π_B_g1 (one JIT kernel: β + B_sum + s·δ) ───────────────────────
    pi_B_g1_tpu = ec_ops.affine_combo_g1_tpu(
        pk_tpu.beta_g1, B_sum_g1, pk_tpu.delta_g1, s_bits,
    )

    # ── π_B_g2 (one JIT kernel: β_g2 + B_sum_g2 + s·δ_g2, on XYZZ) ─────
    pi_B_g2_tpu = ec_ops_g2.affine_combo_g2_tpu(
        pk_tpu.beta_g2, B_sum_g2_tpu, pk_tpu.delta_g2, s_bits,
    )

    # ── π_C (one JIT kernel: private + h + s·A + r·B + rs·δ) ───────────
    pi_C_tpu = ec_ops.pi_c_tpu(
        private_sum, h_sum, pi_A_tpu, pi_B_g1_tpu, pk_tpu.delta_g1,
        s_bits, r_bits, rs_bits,
    )

    # Ensure all TPU work is in-flight before stopping the timer.
    pi_A_tpu.block_until_ready()
    pi_C_tpu.block_until_ready()
    pi_B_g2_tpu.block_until_ready()
    run_time = time.time() - run_t0

    # ── Decode the three EC outputs at the boundary (CPU work) ─────────
    pi_A = ec_ops.decode_g1(pi_A_tpu)
    pi_C = ec_ops.decode_g1(pi_C_tpu)
    pi_B = decode_g2(pi_B_g2_tpu).to_affine()

    print(f"  [prove_opt] proof generation : {run_time:.2f}s")
    return Proof(A=pi_A, B=pi_B, C=pi_C)



# ── prove_pure_tpu — wrapper / inner split for iterative optimisation ───────
#
# Public entry is :func:`prove_pure_tpu_wrapper`.  It owns the CPU work at
# the top (witness encoding, blinder generation, MSM-scalar pre-encoding)
# and at the bottom (decoding the three EC outputs back to G1Affine /
# G2Affine).  In between it hands off to :func:`prove_pure_tpu`, which is
# the "syntactically one TPU kernel" body — the target you're meant to
# iterate on without touching anything in the wrapper.
#
# Current state of :func:`prove_pure_tpu` still has *one* unavoidable
# CPU step in the middle: ``h_drns_to_int_list`` between the H pipeline
# (DRNS Fr) and the h-MSM (which consumes Python int scalars).  Removing
# that requires a TPU-side DRNS → window-slice path; tracked as T10b /
# T3-bis.  Marked clearly below so it's the obvious next thing to attack.


def prove_pure_tpu_wrapper(pk: ProvingKey, qap: QAP, witness: List[int]) -> Proof:
    """Wrapper around :func:`prove_pure_tpu` that owns every CPU step.

    Top of function:  encode witness + scalars + blinder bits.
    Bottom:           decode the three EC outputs into ``G1Affine`` /
                      ``G2Affine`` (and pack them into a :class:`Proof`).
    Middle:           one call into :func:`prove_pure_tpu`, which is
                      pure TPU (modulo the labelled h-boundary).
    """
    n          = qap.n
    num_public = qap.num_public

    # ── Circuit-only precompute (one-time per (pk, qap)) ────────────────
    init_t0 = time.time()
    pk_tpu, qap_tpu = tpu_precompute.precompute_circuit(pk, qap)
    init_time = time.time() - init_t0
    print(f"  [prove_pure] circuit precompute : {init_time:.2f}s")

    # ─────────────────────────────────────────────────────────────────────
    # CPU encode (top of wrapper)
    # ─────────────────────────────────────────────────────────────────────

    # Witness reduced mod r; both as ints (for the G2 MSM) and DRNS Fr
    # (for the H pipeline).
    a_ints = [int(w) % r for w in witness]
    ff = get_ff_ctx(n)
    # T13: high-perf CPU DRNS encode via NumPy uint64 limb arithmetic
    # (~2.1× faster than ``ff.to_computational_format`` at N=1024).
    # Bit-equal to the upstream path; see scalar_codecs_cpu for the
    # broadcast pipeline.
    a_drns = drns_encode_fr_numpy(
        a_ints, rns_moduli=ff.rns_moduli, radix_bits=ff.radix_bits,
        target_length=len(a_ints),
    )                                                        # (m, M_fr)

    # G1 MSM scalars in pre-encoded packed-uint32 form.
    # T14: pure-CPU NumPy slicer (int.to_bytes + vectorised shift+mask).
    # Replaces ``encode_scalars_for_msm_g1_packed`` for all three pk
    # vectors.  A_g1 and B_g1 share the witness; each gets its own
    # identity-masked encode since the masks differ in general.
    a_g1_A = slice_scalars_cpu_fast(a_ints, identity_mask=pk_tpu.A_g1[1])
    a_g1_B = slice_scalars_cpu_fast(a_ints, identity_mask=pk_tpu.B_g1[1])
    # Private wires: indices 1 … num_wires − num_public − 1.
    private_idxs    = list(range(1, qap.num_wires - num_public))
    private_scalars = [a_ints[i] for i in private_idxs]
    private_g1_packed = slice_scalars_cpu_fast(
        private_scalars, identity_mask=pk_tpu.private_g1[1]
    )

    # ZK blinders + the rs constant used by the π_C kernel.  MSB-first
    # bit arrays so each JIT'd scalar mul sees a stable (S,) int8 shape.
    r_blind = secrets.randbelow(r - 1) + 1
    s_blind = secrets.randbelow(r - 1) + 1
    rs      = (r - r_blind * s_blind % r) % r
    r_bits  = ec_ops.int_to_bits_msb(r_blind)
    s_bits  = ec_ops.int_to_bits_msb(s_blind)
    rs_bits = ec_ops.int_to_bits_msb(rs)

    # ─────────────────────────────────────────────────────────────────────
    # Pure TPU body
    # ─────────────────────────────────────────────────────────────────────
    run_t0 = time.time()
    pi_A_tpu, pi_B_g2_tpu, pi_C_tpu = prove_pure_tpu(
        pk_tpu          = pk_tpu,
        qap_tpu         = qap_tpu,
        a_drns          = a_drns,
        a_ints          = a_ints,                # G2 MSM still takes Python list
        a_g1_A_packed   = a_g1_A,
        a_g1_B_packed   = a_g1_B,
        private_packed  = private_g1_packed,
        ff              = ff,
        n               = n,
        r_bits          = r_bits,
        s_bits          = s_bits,
        rs_bits         = rs_bits,
    )
    # All three outputs are TPU-resident.  Block here so the timer below
    # reflects the actual TPU compute (JAX is async by default).
    pi_A_tpu.block_until_ready()
    pi_C_tpu.block_until_ready()
    pi_B_g2_tpu.block_until_ready()
    run_time = time.time() - run_t0
    print(f"  [prove_pure] proof generation : {run_time:.2f}s")

    # ─────────────────────────────────────────────────────────────────────
    # CPU decode (bottom of wrapper)
    # ─────────────────────────────────────────────────────────────────────
    pi_A = ec_ops.decode_g1(pi_A_tpu)
    pi_C = ec_ops.decode_g1(pi_C_tpu)
    pi_B = decode_g2(pi_B_g2_tpu).to_affine()
    return Proof(A=pi_A, B=pi_B, C=pi_C)


def prove_pure_tpu(
    *,
    pk_tpu,
    qap_tpu,
    a_drns,
    a_ints,
    a_g1_A_packed,
    a_g1_B_packed,
    private_packed,
    ff,
    n,
    r_bits,
    s_bits,
    rs_bits,
):
    """Pure-TPU prove body.  Returns ``(pi_A_tpu, pi_B_g2_tpu, pi_C_tpu)``
    — three EC point arrays in TPU-native DRNS form.

    Every line is a JAX op (or a JIT'd kernel call) — no mid-flow CPU
    steps, except the one host-side h_drns → int → slice round-trip
    (the TPU-side fast path for that step never produced correct slices
    and has been dropped).

    Inputs are already encoded by the wrapper, outputs are decoded by
    the wrapper.  This function should *never* call ``int(...)`` on a
    JAX array, never iterate a Python list of TPU outputs, never write
    a `for j in range(n)` loop over witness data, etc.
    """
    # ── G1 MSMs (TPU) ───────────────────────────────────────────────────
    # ``pk_tpu.A_g1`` is a ``(tiled_points, identity_mask)`` tuple from
    # ``precompute_pk``; we want just the JAX-array half for the
    # no-branch dispatch.  ``a_*_packed`` are already JAX arrays from
    # ``encode_scalars_for_msm_g1_packed`` in the wrapper.
    # ``msm_g1_pure_tpu`` returns (4, 1, M_fp) directly — the reshape
    # the EC ops need is fused into the MSM output (metadata-only
    # under XLA), so no ``_as_4_1_M`` wrapping required here.
    A_sum       = msm_g1_pure_tpu(pk_tpu.A_g1[0],       a_g1_A_packed)
    B_sum_g1    = msm_g1_pure_tpu(pk_tpu.B_g1[0],       a_g1_B_packed)
    private_sum = msm_g1_pure_tpu(pk_tpu.private_g1[0], private_packed)

    # ── H(x) pipeline (TPU): combine_polys + coset + NTT + pointwise +
    #    INTT + un-shift.  Returns h_co in DRNS Fr form.
    h_co_drns = compute_h_coset_ntt_drns_opt(a_drns, qap_tpu)

    # ── Host-side h DRNS → int → packed slice (one CPU boundary).
    # The TPU-side fused variant was attempted but never produced correct
    # slices; the host round-trip is fast enough (~33 ms at N=1024) and
    # correct.
    h_coeffs_cpu = h_drns_to_int_list(h_co_drns, ff, n)
    h_packed     = slice_scalars_cpu_fast(
        h_coeffs_cpu, identity_mask=pk_tpu.h_g1[1]
    )

    # ── h MSM (TPU, no-branch dispatch) ────────────────────────────────
    h_sum = msm_g1_pure_tpu(pk_tpu.h_g1[0], h_packed)

    # ── G2 MSM (TPU, XYZZ Fp2 Pippenger) ───────────────────────────────
    B_sum_g2_tpu = msm_g2_tpu(pk_tpu.B_g2, a_ints)            # (4, 1, 2, M_fp)

    # ── Composite kernels (each = one JIT'd JAX kernel) ────────────────
    pi_A_tpu = ec_ops.affine_combo_g1_tpu(
        pk_tpu.alpha_g1, A_sum, pk_tpu.delta_g1, r_bits,
    )
    pi_B_g1_tpu = ec_ops.affine_combo_g1_tpu(
        pk_tpu.beta_g1, B_sum_g1, pk_tpu.delta_g1, s_bits,
    )
    pi_B_g2_tpu = ec_ops_g2.affine_combo_g2_tpu(
        pk_tpu.beta_g2, B_sum_g2_tpu, pk_tpu.delta_g2, s_bits,
    )
    pi_C_tpu = ec_ops.pi_c_tpu(
        private_sum, h_sum, pi_A_tpu, pi_B_g1_tpu, pk_tpu.delta_g1,
        s_bits, r_bits, rs_bits,
    )

    return pi_A_tpu, pi_B_g2_tpu, pi_C_tpu