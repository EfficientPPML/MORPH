"""
Groth16 Full-ZK Prover.

Given the proving key (pk) and a witness vector, computes the proof (π_A, π_B, π_C).

Full-ZK: random blinding scalars r_blind, s_blind ∈ Fr are sampled per proof.
These ensure the proof reveals nothing about the witness beyond what the
public inputs already imply.

Proof equations (Groth16, full ZK):
    π_A = [α]₁  +  Σᵢ aᵢ·[Uᵢ(τ)]₁            +  r_blind·[δ]₁
    π_B = [β]₂  +  Σᵢ aᵢ·[Vᵢ(τ)]₂            +  s_blind·[δ]₂
    π_C = (Σᵢ∈priv aᵢ·[(β·Uᵢ+α·Vᵢ+Wᵢ)(τ)/δ]₁
          +  Σⱼ hⱼ·[τʲ·t(τ)/δ]₁)
          +  s_blind·π_A  +  r_blind·[π_B]₁  −  r_blind·s_blind·[δ]₁

where [π_B]₁ is the G1 version of π_B (using B_g1 from pk).

Kernel call sites are explicitly marked below.
"""

import secrets
from dataclasses import dataclass
from typing      import List

from bls12_377.params import r
from bls12_377.g1     import G1Affine, G1Projective
from bls12_377.g2     import G2Affine, G2Projective
from groth16.qap      import QAP
from groth16.setup    import ProvingKey
from kernels.ntt      import get_omega, ntt, intt
from kernels.msm      import msm_g1, msm_g2


@dataclass
class Proof:
    """A Groth16 proof: three elliptic curve points."""
    A: G1Affine   # π_A ∈ G1
    B: G2Affine   # π_B ∈ G2
    C: G1Affine   # π_C ∈ G1


def prove(pk: ProvingKey, qap: QAP, witness: List[int]) -> Proof:
    """
    Generate a full-ZK Groth16 proof.

    Parameters
    ----------
    pk      : ProvingKey from trusted_setup()
    qap     : QAP from r1cs_to_qap()
    witness : full witness vector [a₀, a₁, ..., aₘ] in Fr

    Returns
    -------
    Proof(A, B, C)
    """
    n          = qap.n
    num_public = qap.num_public

    # Reduce witness mod r
    a = [w % r for w in witness]

    # ── ZK blinding scalars ─────────────────────────────────────────────────
    r_blind = secrets.randbelow(r - 1) + 1
    s_blind = secrets.randbelow(r - 1) + 1

    # ── Compute π_A ─────────────────────────────────────────────────────────
    # π_A = [α]₁ + Σᵢ aᵢ·[Uᵢ(τ)]₁ + r_blind·[δ]₁
    # ═══ KERNEL: MSM (G1) — compute π_A ════════════════════════════════════
    A_sum = msm_g1(pk.A_g1, a)
    # ════════════════════════════════════════════════════════════════════════
    pi_A_proj = (pk.alpha_g1.to_projective()
                 .add(A_sum)
                 .add(pk.delta_g1.to_projective().scalar_mul(r_blind)))

    # ── Compute π_B (G2) ────────────────────────────────────────────────────
    # π_B = [β]₂ + Σᵢ aᵢ·[Vᵢ(τ)]₂ + s_blind·[δ]₂
    # ═══ KERNEL: MSM (G2) — compute π_B ════════════════════════════════════
    B_sum_g2 = msm_g2(pk.B_g2, a)
    # ════════════════════════════════════════════════════════════════════════
    pi_B_proj = (pk.beta_g2.to_projective()
                 .add(B_sum_g2)
                 .add(pk.delta_g2.to_projective().scalar_mul(s_blind)))

    # G1 counterpart of π_B (needed for ZK correction term in π_C):
    # ═══ KERNEL: MSM (G1) — compute [π_B]₁ ════════════════════════════════
    B_sum_g1 = msm_g1(pk.B_g1, a)
    # ════════════════════════════════════════════════════════════════════════
    pi_B_g1_proj = (pk.beta_g1.to_projective()
                    .add(B_sum_g1)
                    .add(pk.delta_g1.to_projective().scalar_mul(s_blind)))

    # ── Compute H(x) — the quotient polynomial ──────────────────────────────
    # A(x) = Σ aᵢ·Uᵢ(x),  B(x) = Σ aᵢ·Vᵢ(x),  C(x) = Σ aᵢ·Wᵢ(x)
    # H(x) = (A(x)·B(x) − C(x)) / t(x)   in Fr[x]

    # Build A(x), B(x), C(x) coefficient vectors (length n)
    def combine_polys(polys, weights):
        """Compute Σ weights[i]·polys[i] as coefficient vectors."""
        result = [0] * n
        for i, (p, w) in enumerate(zip(polys, weights)):
            for j in range(n):
                result[j] = (result[j] + w * p[j]) % r
        return result

    A_coeffs = combine_polys(qap.U, a)
    B_coeffs = combine_polys(qap.V, a)
    C_coeffs = combine_polys(qap.W, a)

    # ═══ KERNEL: NTT — compute H via coset evaluation (no poly division) ════
    h_coeffs = compute_h_coset_ntt(A_coeffs, B_coeffs, C_coeffs, n, qap.omega, qap.omega2n)
    # ════════════════════════════════════════════════════════════════════════

    # ═══ KERNEL: MSM (G1) — compute H(τ)·t(τ)/δ contribution to π_C ════════
    h_g1_contribution = msm_g1(pk.h_g1, h_coeffs)
    # ════════════════════════════════════════════════════════════════════════

    # ── Private wire sum ────────────────────────────────────────────────────
    # Private wires: indices 1..num_wires-1-num_public
    private_idxs  = list(range(1, qap.num_wires - num_public))
    private_scalars = [a[i] for i in private_idxs]

    # ═══ KERNEL: MSM (G1) — private wire terms in π_C ═══════════════════════
    private_sum = msm_g1(pk.private_g1, private_scalars)
    # ════════════════════════════════════════════════════════════════════════

    # ── Assemble π_C with ZK blinding ──────────────────────────────────────
    # π_C = private_sum + H·t/δ
    #       + s_blind·π_A + r_blind·[π_B]₁ − r_blind·s_blind·[δ]₁
    pi_C_proj = (
        private_sum
        .add(h_g1_contribution)
        .add(pi_A_proj.scalar_mul(s_blind))
        .add(pi_B_g1_proj.scalar_mul(r_blind))
        .add(pk.delta_g1.to_projective().scalar_mul((r - r_blind * s_blind % r) % r))
    )

    return Proof(
        A = pi_A_proj.to_affine(),
        B = pi_B_proj.to_affine(),
        C = pi_C_proj.to_affine(),
    )


def compute_h_coset_ntt(A_coeffs, B_coeffs, C_coeffs, n, omega_n, omega2n):
    """
    Compute H(x) = (A(x)·B(x) − C(x)) / t(x) using the coset NTT trick.

    Instead of explicit polynomial multiplication + division, we:
      1. Evaluate A, B, C on a coset {g·ωⁿ^k} where g = ω_{2n}
      2. Divide pointwise by t(g·ωⁿ^k) = −2 (constant on the coset!)
      3. INTT back to get H coefficients

    This avoids polynomial division entirely.  O(n log n) vs O(n²).

    Why the coset?
      t(x) = xⁿ − 1 vanishes at the n-th roots of unity, so we can't divide
      there.  Shifting by g = ω_{2n} (a primitive 2n-th root) moves us to
      points where t is uniformly −2:
          t(g·ωⁿ^k) = (g·ωⁿ^k)ⁿ − 1 = gⁿ·1 − 1 = (−1) − 1 = −2

    Parameters
    ----------
    A_coeffs, B_coeffs, C_coeffs : coefficient vectors (length n)
    n        : number of constraints (power of 2)
    omega_n  : primitive n-th root of unity in Fr
    omega2n  : primitive 2n-th root of unity in Fr  (used as coset shift g)

    Returns
    -------
    h_coeffs : list of length n−1 (degree ≤ n−2)
    """
    g = omega2n   # coset generator;  g² = omega_n,  gⁿ = −1

    # ── Step 1: shift coefficients to evaluate on coset {g·ωⁿ^k} ───────
    # Multiplying coeff[j] by g^j is equivalent to evaluating p(g·x)
    # via NTT instead of p(x).
    g_pow = 1
    A_shifted = [0] * n
    B_shifted = [0] * n
    C_shifted = [0] * n
    for j in range(n):
        A_shifted[j] = A_coeffs[j] * g_pow % r
        B_shifted[j] = B_coeffs[j] * g_pow % r
        C_shifted[j] = C_coeffs[j] * g_pow % r
        g_pow = g_pow * g % r

    # ── Step 2: NTT to get evaluations on the coset ─────────────────────
    # ═══ KERNEL: NTT ═══
    A_evals = ntt(A_shifted, omega_n)
    B_evals = ntt(B_shifted, omega_n)
    C_evals = ntt(C_shifted, omega_n)
    # ═══════════════════

    # ── Step 3: pointwise (A·B − C) / t, where t = −2 on the coset ─────
    inv_neg2 = pow(r - 2, r - 2, r)   # (−2)⁻¹ mod r
    H_evals = [
        (A_evals[k] * B_evals[k] - C_evals[k]) * inv_neg2 % r
        for k in range(n)
    ]

    # ── Step 4: INTT back to coefficient domain (still shifted) ─────────
    # ═══ KERNEL: INTT ═══
    H_shifted = intt(H_evals, omega_n)
    # ════════════════════

    # ── Step 5: un-shift by dividing coeff[j] by g^j ───────────────────
    g_inv = pow(g, r - 2, r)
    g_inv_pow = 1
    h_coeffs = [0] * (n - 1)
    for j in range(n - 1):
        h_coeffs[j] = H_shifted[j] * g_inv_pow % r
        g_inv_pow = g_inv_pow * g_inv % r

    return h_coeffs


def poly_divmod(f: list, n: int) -> list:
    """
    Divide polynomial f (length 2n-1) by t(x) = xⁿ − 1.
    Returns quotient h of degree n-2 (length n-1).

    For t(x) = xⁿ − 1, the quotient coefficients are simply h[k] = f[n+k].
    The remainder is f[k] + h[k] for k = 0..n-2 (must be zero for valid witness).
    """
    h = [f[n + k] % r for k in range(n - 1)]
    return h
