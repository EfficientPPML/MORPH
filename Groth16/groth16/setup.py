"""
Groth16 Trusted Setup.

Samples toxic waste (τ, α, β, γ, δ) ∈ Fr and encodes the QAP
polynomials into G1/G2 curve points to produce:

    ProvingKey  (pk) — used by the prover
    VerificationKey (vk) — used by the verifier

The MSM kernel is used extensively here to encode polynomial
evaluations at the secret point τ into curve points.

Security note: In a real deployment, the toxic waste must be
destroyed after setup (MPC ceremony). Here we return it for
demonstration purposes only — do NOT do this in production.
"""

import multiprocessing as _mp
import os
import secrets
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing      import List

from bls12_377.params import r
from bls12_377.g1     import G1Affine, G1Projective, G1_GENERATOR
from bls12_377.g2     import G2Affine, G2Projective, G2_GENERATOR
from groth16.qap      import QAP, eval_poly
from kernels.msm      import msm_g1, msm_g2


# ── Optional acceleration deps ────────────────────────────────────────────────


try:
    from flint import fmpz, fmpz_mod_poly_ctx, fmpz_mod_poly
    _HAVE_FLINT = True
except ImportError:
    _HAVE_FLINT = False

try:
    import gmpy2
    _HAVE_GMPY2 = True
except ImportError:
    _HAVE_GMPY2 = False


# Match the forkserver convention used elsewhere in the repo (avoids
# JAX + fork deadlocks).  Children must NOT re-execute the daemon's
# main module — callers are expected to drive this from a clean entry
# point (e.g. ``prep_circuit.py``) or to not be holding the TPU.
_MP_CONTEXT = _mp.get_context("forkserver")


# ── Polynomial evaluation at τ — fast paths ───────────────────────────────────


def _eval_polys_flint(polys: List[List[int]], tau: int, modulus: int) -> List[int]:
    """Batch-evaluate a list of polynomials at a single point τ using
    python-flint.  ~5-10× faster than the CPython ``eval_poly`` loop
    on 253-bit Fr."""
    ctx    = fmpz_mod_poly_ctx(fmpz(modulus))
    tau_fz = fmpz(tau)
    out    = []
    for coeffs in polys:
        if not coeffs:
            out.append(0)
            continue
        p = fmpz_mod_poly([fmpz(c) for c in coeffs], ctx)
        out.append(int(p(tau_fz)))
    return out


def _eval_polys_gmpy2(polys: List[List[int]], tau: int, modulus: int) -> List[int]:
    """Batch poly eval using gmpy2 mpz.  ~2-3× over CPython; fallback
    when python-flint is unavailable."""
    g_r   = gmpy2.mpz(modulus)
    g_tau = gmpy2.mpz(tau)
    out   = []
    for coeffs in polys:
        # Horner — ((c[-1]·τ + c[-2])·τ + … + c[0])
        acc = gmpy2.mpz(0)
        for c in reversed(coeffs):
            acc = (acc * g_tau + gmpy2.mpz(c)) % g_r
        out.append(int(acc))
    return out


def _eval_polys_baseline(polys: List[List[int]], tau: int, _modulus: int) -> List[int]:
    """Slowest, pure-Python fallback — exactly the original code."""
    return [eval_poly(p, tau) for p in polys]


def _eval_polys(polys, tau, modulus):
    """Pick the best available batch polynomial-evaluation backend."""
    if _HAVE_FLINT:
        return _eval_polys_flint(polys, tau, modulus)
    if _HAVE_GMPY2:
        return _eval_polys_gmpy2(polys, tau, modulus)
    return _eval_polys_baseline(polys, tau, modulus)


# ── Parallel EC scalar mul of the fixed generator ─────────────────────────────


def _g1_worker(scalar_int: int):
    """Forkserver worker: ``[s]·G1_GENERATOR`` → ``G1Affine``."""
    return G1_GENERATOR.to_projective().scalar_mul(scalar_int % r).to_affine()


def _g2_worker(scalar_int: int):
    """Forkserver worker: ``[s]·G2_GENERATOR`` → ``G2Affine``."""
    return G2_GENERATOR.to_projective().scalar_mul(scalar_int % r).to_affine()


def _parallel_g1_list(scalars: List[int], num_workers: int) -> List[G1Affine]:
    if num_workers <= 1 or len(scalars) < 64:
        return [_g1_worker(s) for s in scalars]
    with ProcessPoolExecutor(max_workers=num_workers,
                             mp_context=_MP_CONTEXT) as pool:
        return list(pool.map(_g1_worker, scalars))


def _parallel_g2_list(scalars: List[int], num_workers: int) -> List[G2Affine]:
    if num_workers <= 1 or len(scalars) < 64:
        return [_g2_worker(s) for s in scalars]
    with ProcessPoolExecutor(max_workers=num_workers,
                             mp_context=_MP_CONTEXT) as pool:
        return list(pool.map(_g2_worker, scalars))


# ── Bulk per-wire scalar arithmetic with gmpy2 ────────────────────────────────


def _affine_combo_bulk(beta: int, alpha: int, U_tau, V_tau, W_tau,
                        idxs, inv_factor: int, modulus: int) -> List[int]:
    """Compute ``[(β·Uᵢ + α·Vᵢ + Wᵢ)(τ) · inv_factor] mod r`` for each
    ``i`` in ``idxs``.  Used for both the private-wire pk entries
    (``inv_factor = δ⁻¹``) and the public-input vk entries
    (``inv_factor = γ⁻¹``).  gmpy2 mpz is ~2-3× faster than CPython
    int on 253-bit operands."""
    if _HAVE_GMPY2:
        g_r     = gmpy2.mpz(modulus)
        g_alpha = gmpy2.mpz(alpha)
        g_beta  = gmpy2.mpz(beta)
        g_inv   = gmpy2.mpz(inv_factor)
        out = []
        for i in idxs:
            v = (g_beta * U_tau[i] + g_alpha * V_tau[i] + W_tau[i]) % g_r
            out.append(int(v * g_inv % g_r))
        return out
    return [
        (beta * U_tau[i] + alpha * V_tau[i] + W_tau[i]) % modulus
        * inv_factor % modulus
        for i in idxs
    ]


def _tau_powers(tau: int, n: int, modulus: int) -> List[int]:
    """``[τ⁰, τ¹, …, τⁿ⁻¹] mod r``.  gmpy2 ``powmod`` shaves ~30 %."""
    if _HAVE_GMPY2:
        g_tau = gmpy2.mpz(tau)
        g_r   = gmpy2.mpz(modulus)
        out   = [gmpy2.mpz(1)]
        for _ in range(1, n):
            out.append(out[-1] * g_tau % g_r)
        return [int(v) for v in out]
    return [pow(tau, j, modulus) for j in range(n)]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ProvingKey:
    """
    Groth16 proving key (pk).

    G1 elements
    -----------
    alpha_g1      : [α]₁
    beta_g1       : [β]₁
    delta_g1      : [δ]₁
    tau_g1        : list of [τʲ]₁  for j = 0..n-1
    A_g1          : list of [Uᵢ(τ)]₁  for all wires i
    B_g1          : list of [Vᵢ(τ)]₁  for all wires i  (needed for ZK blinding)
    private_g1    : list of [(β·Uᵢ+α·Vᵢ+Wᵢ)(τ)/δ]₁  for private wires
    h_g1          : list of [τʲ·t(τ)/δ]₁  for j = 0..n-2  (H polynomial CRS)

    G2 elements
    -----------
    beta_g2       : [β]₂
    gamma_g2      : [γ]₂
    delta_g2      : [δ]₂
    B_g2          : list of [Vᵢ(τ)]₂  for all wires i
    """
    alpha_g1:   G1Affine
    beta_g1:    G1Affine
    delta_g1:   G1Affine
    tau_g1:     List[G1Affine]
    A_g1:       List[G1Affine]
    B_g1:       List[G1Affine]
    private_g1: List[G1Affine]
    h_g1:       List[G1Affine]
    beta_g2:    G2Affine
    gamma_g2:   G2Affine
    delta_g2:   G2Affine
    B_g2:       List[G2Affine]


@dataclass
class VerificationKey:
    """
    Groth16 verification key (vk).

    alpha_g1  : [α]₁
    beta_g2   : [β]₂
    gamma_g2  : [γ]₂
    delta_g2  : [δ]₂
    ic        : list of [(β·Uᵢ+α·Vᵢ+Wᵢ)(τ)/γ]₁  for public wires (length = num_public+1)
                ic[0] corresponds to the constant wire (always 1).
                ic[1..] correspond to each public input wire.
    """
    alpha_g1: G1Affine
    beta_g2:  G2Affine
    gamma_g2: G2Affine
    delta_g2: G2Affine
    ic:       List[G1Affine]   # "input commitments"


# ── Helper: scalar-multiply generator ────────────────────────────────────────

def _g1(s: int) -> G1Affine:
    """Return [s]·G1_GENERATOR as affine point."""
    return G1_GENERATOR.to_projective().scalar_mul(s % r).to_affine()

def _g2(s: int) -> G2Affine:
    """Return [s]·G2_GENERATOR as affine point."""
    return G2_GENERATOR.to_projective().scalar_mul(s % r).to_affine()


# ── Trusted setup ─────────────────────────────────────────────────────────────

def trusted_setup(qap: QAP, toxic_waste: dict = None):
    """
    Run the Groth16 trusted setup for the given QAP.

    Parameters
    ----------
    qap          : QAP instance from r1cs_to_qap()
    toxic_waste  : optional dict with keys τ,α,β,γ,δ (for testing).
                   If None, random values are sampled.

    Returns
    -------
    (ProvingKey, VerificationKey)
    """
    n = qap.n

    # ── Sample toxic waste ──────────────────────────────────────────────────
    if toxic_waste is None:
        tau   = secrets.randbelow(r - 2) + 2
        alpha = secrets.randbelow(r - 2) + 2
        beta  = secrets.randbelow(r - 2) + 2
        gamma = secrets.randbelow(r - 2) + 2
        delta = secrets.randbelow(r - 2) + 2
    else:
        tau   = toxic_waste["tau"]   % r
        alpha = toxic_waste["alpha"] % r
        beta  = toxic_waste["beta"]  % r
        gamma = toxic_waste["gamma"] % r
        delta = toxic_waste["delta"] % r

    delta_inv = pow(delta, r - 2, r)
    gamma_inv = pow(gamma, r - 2, r)

    # ── Evaluate QAP polynomials at τ ───────────────────────────────────────
    # python-flint / gmpy2 fast paths inside ``_eval_polys``; CPython
    # fallback if neither is installed.
    U_tau = _eval_polys(qap.U, tau, r)
    V_tau = _eval_polys(qap.V, tau, r)
    W_tau = _eval_polys(qap.W, tau, r)
    t_tau = _eval_polys([qap.t_coeffs], tau, r)[0]

    # ── Fixed-base single-point scalar mults (no parallelism — 6 points) ────
    alpha_g1 = _g1(alpha)
    beta_g1  = _g1(beta)
    delta_g1 = _g1(delta)
    beta_g2  = _g2(beta)
    gamma_g2 = _g2(gamma)
    delta_g2 = _g2(delta)

    # ── Powers of τ ─────────────────────────────────────────────────────────
    tau_powers = _tau_powers(tau, n, r)

    # ── Wire-vector EC scalar muls — parallel ────────────────────────────────
    # All of these are embarrassingly parallel: one EC scalar mul of
    # the fixed generator per scalar in the input list.  The ProcessPool
    # spawns ``num_workers`` forkserver children once and reuses them
    # across maps.
    num_workers = min(os.cpu_count() or 1, 4)

    tau_g1 = _parallel_g1_list(tau_powers,                       num_workers)
    A_g1   = _parallel_g1_list(U_tau,                            num_workers)
    B_g1 = _parallel_g1_list(V_tau,                              num_workers)
    B_g2 = _parallel_g2_list(V_tau,                              num_workers)

    # Private wire contributions: (β·Uᵢ + α·Vᵢ + Wᵢ)(τ) · δ⁻¹
    num_public   = qap.num_public
    private_idxs = list(range(1, qap.num_wires - num_public))
    private_vals = _affine_combo_bulk(
        beta, alpha, U_tau, V_tau, W_tau,
        private_idxs, delta_inv, r,
    )
    private_g1 = _parallel_g1_list(private_vals, num_workers)

    # H polynomial CRS: [τʲ · t(τ) / δ]₁  for j = 0..n-2
    if _HAVE_GMPY2:
        g_r       = gmpy2.mpz(r)
        g_t_tau   = gmpy2.mpz(t_tau)
        g_dinv    = gmpy2.mpz(delta_inv)
        h_scalars = [int(gmpy2.mpz(tau_powers[j]) * g_t_tau % g_r * g_dinv % g_r)
                     for j in range(n - 1)]
    else:
        h_scalars = [tau_powers[j] * t_tau % r * delta_inv % r for j in range(n - 1)]
    h_g1 = _parallel_g1_list(h_scalars, num_workers)

    # Public wire IC: (β·Uᵢ + α·Vᵢ + Wᵢ)(τ) · γ⁻¹
    public_idxs = [0] + list(range(qap.num_wires - num_public, qap.num_wires))
    ic_vals = _affine_combo_bulk(
        beta, alpha, U_tau, V_tau, W_tau,
        public_idxs, gamma_inv, r,
    )
    ic = _parallel_g1_list(ic_vals, num_workers)
    # ════════════════════════════════════════════════════════════════════════

    pk = ProvingKey(
        alpha_g1   = alpha_g1,
        beta_g1    = beta_g1,
        delta_g1   = delta_g1,
        tau_g1     = tau_g1,
        A_g1       = A_g1,
        B_g1       = B_g1,
        private_g1 = private_g1,
        h_g1       = h_g1,
        beta_g2    = beta_g2,
        gamma_g2   = gamma_g2,
        delta_g2   = delta_g2,
        B_g2       = B_g2,
    )

    vk = VerificationKey(
        alpha_g1 = alpha_g1,
        beta_g2  = beta_g2,
        gamma_g2 = gamma_g2,
        delta_g2 = delta_g2,
        ic       = ic,
    )

    return pk, vk
