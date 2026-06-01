"""
Groth16 Verifier — high-performance, pure-CPython.

Checks the pairing equation:

    e(π_A, π_B) == e([α]₁, [β]₂) · e(vk_pub, [γ]₂) · e(π_C, [δ]₂)

where:
    vk_pub = Σᵢ (public_input[i]) · ic[i+1]  +  ic[0]
             (ic[0] is for the implicit constant wire = 1)

The verifier knows only:
    - The verification key (vk)
    - The public inputs
    - The proof (π_A, π_B, π_C)

It learns nothing about the private witness.

Three verify functions are exposed; pick the one that matches your
deployment.

  ============  ===========================================================
  function       implementation
  ============  ===========================================================
  ``verify``     fan out 4 Miller loops to a forkserver process pool, then
                 fold them and apply ONE final exponentiation.  Fastest;
                 ~4-5× over the textbook version on a 4-core CPU.
  ``verify_serial``
                 single multi-Miller (shared per-iteration squaring) +
                 single final exponentiation.  ~3× over the textbook.
                 Use when you can't fan out across processes.
  ``verify_naive``
                 four independent ate_pairings, multiplied in Fp12.
                 The textbook expression — slow, kept for reference and
                 benchmarking.
  ============  ===========================================================

The optimisation underneath everything: the Groth16 verification
equation is symbolically rearranged so the verifier needs only **one**
final exponentiation rather than four.

  Textbook:    e(A, B)  ==  e(α, β) · e(vk_pub, γ) · e(C, δ)
  Optimised:   e(A, B) · e(-α, β) · e(-vk_pub, γ) · e(-C, δ)  ==  1

Negating a G1 point is free (flip y); the heavy cost (Miller loops +
final exponentiation) is what we batch.
"""

import atexit
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor

from bls12_377.pairing import (
    ate_pairing, miller_loop, final_exp, multi_pairing,
)
from bls12_377.field   import fp12_mul, fp12_eq, FP12_ONE
from bls12_377.g1      import G1Affine, G1Projective
from bls12_377.g2      import G2Affine
from groth16.setup     import VerificationKey
from groth16.prover    import Proof


def _validate_inputs(vk: VerificationKey, public_inputs: list,
                     proof: Proof) -> bool:
    """Return True if the structural / on-curve checks pass.

    Separates the cheap "is this proof obviously malformed" checks from
    the expensive pairing equation.
    """
    if len(public_inputs) != len(vk.ic) - 1:
        raise ValueError(
            f"Expected {len(vk.ic) - 1} public inputs, got {len(public_inputs)}"
        )

    # Reject identity points (trivial forgery if A or B is infinity).
    if proof.A.is_infinity() or proof.B.is_infinity() or proof.C.is_infinity():
        return False

    # On-curve checks (soundness requirement).
    if not (proof.A.is_on_curve() and proof.C.is_on_curve()):
        return False
    if not proof.B.is_on_curve():
        return False

    return True


def _compute_vk_pub(vk: VerificationKey, public_inputs: list) -> G1Affine:
    """Aggregate the public-input commitment via a small CPU MSM.

    The verifier computes ``vk_pub = ic[0] + Σ public_inputs[i] · ic[i+1]``
    where ``len(public_inputs)`` equals the circuit's `num_public` —
    typically 1 for our demos, ≤ a handful for richer circuits.  A
    plain CPython EC accumulator is well under a millisecond at this
    size and keeps the verifier entirely off the TPU.
    """
    scalars = [1] + list(public_inputs)
    points  = vk.ic[:len(scalars)]
    acc = G1Projective(1, 1, 0)    # point at infinity (Z=0)
    for s, P in zip(scalars, points):
        s = int(s)
        if s == 0:
            continue
        contrib = P.to_projective() if s == 1 else P.to_projective().scalar_mul(s)
        acc = acc.add(contrib)
    return acc.to_affine()


# ══════════════════════════════════════════════════════════════════════
# Parallel verifier — process pool fans out the 4 Miller loops
# ══════════════════════════════════════════════════════════════════════
#
# A persistent ProcessPoolExecutor (forkserver context) is the right
# trade-off for the daemon: pay the ~1-2 s pool startup once on first
# verify, then every subsequent verify hands off four Miller loops to
# 4 workers in parallel.  Per-call overhead is just pickle+IPC of four
# G1/G2 points (~32 KB total) and four Fp12 results — sub-millisecond.
#
# Pool is sized to the number of pairs we typically fold (4).  Reuse
# across the whole process lifetime.

_POOL = None
_POOL_WORKERS = 4


def _get_pool():
    """Lazy-init the parallel-Miller worker pool."""
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(
            max_workers = _POOL_WORKERS,
            mp_context  = _mp.get_context("forkserver"),
        )
        atexit.register(_POOL.shutdown, wait=False)
    return _POOL


def _miller_worker(P_and_Q):
    """Worker entry-point: compute one Miller loop, return the Fp12.

    Pickled across the IPC boundary.  Re-imports
    :func:`bls12_377.pairing.miller_loop` inside the worker.
    """
    from bls12_377.pairing import miller_loop
    P, Q = P_and_Q
    return miller_loop(P, Q)


def verify(vk: VerificationKey, public_inputs: list, proof: Proof) -> bool:
    """High-performance Groth16 verify.

    Strategy:

    1. Reduce the equation ``e(A, B) = e(α, β) · e(vk_pub, γ) · e(C, δ)``
       to ``∏ e(P_i, -Q_i) == 1`` (negate three G1 points; identity in
       Fp12 on the RHS).
    2. Fan out the four Miller loops to a forkserver process pool (one
       Miller per worker).
    3. Multiply the four resulting Fp12 elements on the main thread.
    4. Apply a single final exponentiation.

    On a 4-core CPU this is roughly 4-5× the textbook verifier:

    *   Miller loops parallelised across cores: ~200 ms wall (vs ~600 ms
        serial).
    *   Final exponentiation: ~430 ms, applied **once** instead of four
        times.
    *   Total: ~700 ms, vs ~2.5-3 s for the naive verifier.

    The worker pool is process-global; pay the ~1-2 s startup once.
    """
    if not _validate_inputs(vk, public_inputs, proof):
        return False

    vk_pub = _compute_vk_pub(vk, public_inputs)

    # Four pairs whose product is the identity iff the proof is valid.
    # G1 negation is cheap (flip y); G2 stays as-is.
    pairs = [
        (proof.A,           proof.B    ),
        (vk.alpha_g1.neg(), vk.beta_g2 ),
        (vk_pub.neg(),      vk.gamma_g2),
        (proof.C.neg(),     vk.delta_g2),
    ]

    # ── Parallel: 4 Miller loops × workers ──
    pool = _get_pool()
    ml_results = list(pool.map(_miller_worker, pairs))

    # ── Serial: multiply Fp12 results, then ONE final exponentiation ──
    f = FP12_ONE
    for ml in ml_results:
        f = fp12_mul(f, ml)
    result = final_exp(f)

    return fp12_eq(result, FP12_ONE)


# ══════════════════════════════════════════════════════════════════════
# Serial multi-pairing verifier (no IPC overhead)
# ══════════════════════════════════════════════════════════════════════


def verify_serial(vk: VerificationKey, public_inputs: list,
                   proof: Proof) -> bool:
    """Verify using a single shared-squaring Miller loop + one final exp.

    Same correctness as :func:`verify`; ~3× the textbook verifier.
    Use when you can't fan out across processes (e.g. single-core
    deployment) or want to avoid the pool startup cost.
    """
    if not _validate_inputs(vk, public_inputs, proof):
        return False

    vk_pub = _compute_vk_pub(vk, public_inputs)

    pairs = [
        (proof.A,           proof.B    ),
        (vk.alpha_g1.neg(), vk.beta_g2 ),
        (vk_pub.neg(),      vk.gamma_g2),
        (proof.C.neg(),     vk.delta_g2),
    ]
    result = multi_pairing(pairs)
    return fp12_eq(result, FP12_ONE)


# ══════════════════════════════════════════════════════════════════════
# Naive textbook verifier (kept for benchmarking + correctness reference)
# ══════════════════════════════════════════════════════════════════════


def verify_naive(vk: VerificationKey, public_inputs: list, proof: Proof) -> bool:
    """Same semantics as :func:`verify`, but uses four independent
    ate_pairings and three Fp12 multiplications.

    Slower (~3× the cost) — kept for benchmarking and as a reference
    against which the optimized path can be diffed during debugging.
    """
    if not _validate_inputs(vk, public_inputs, proof):
        return False

    vk_pub = _compute_vk_pub(vk, public_inputs)

    print("      [Pairing] e(A, B)        ...")
    lhs = ate_pairing(proof.A, proof.B)

    print("      [Pairing] e(α, β)        ...")
    p1  = ate_pairing(vk.alpha_g1, vk.beta_g2)

    print("      [Pairing] e(vk_pub, γ)   ...")
    p2  = ate_pairing(vk_pub, vk.gamma_g2)

    print("      [Pairing] e(C, δ)        ...")
    p3  = ate_pairing(proof.C, vk.delta_g2)

    rhs = fp12_mul(fp12_mul(p1, p2), p3)
    return fp12_eq(lhs, rhs)
