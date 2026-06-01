"""
Primitive 2n-th roots of unity (psi) for the BLS12-377 scalar field Fr.

`number_theory_transform_context.NTT3Step` accepts a precomputed `psi`
parameter; if absent, it falls back to `utils.root_of_unity` which trial-
divides r-1 (infeasible for the 253-bit r).  We compute psi here using
the same fast path as Groth16/kernels/ntt.py: any quadratic non-residue
in Fr generates the 2-Sylow subgroup, so for any m = 2^k dividing r-1
(true for m ≤ 2^47 since r-1 = 2^47 · t),
    psi = g_qnr ^ ((r - 1) / m)
is a primitive m-th root of unity.

We expose:
    primitive_root_of_unity(m)   -> int
    psi_for_ntt_size(n)          -> int        (m = 2*n, primitive 2n-th root)
    omega_for_ntt_size(n)        -> int        (= psi^2, primitive n-th root)
"""

import functools

# Avoid pulling in path_setup just for this module — keep it dependency-free
# so it can be imported by both adapter and test code.
_R_BLS = 8444461749428370424248824938781546531375899335154063827935233455917409239041


@functools.cache
def _quadratic_non_residue() -> int:
    """Return the smallest QNR in F_r (i.e. some g with g^((r-1)/2) ≢ 1)."""
    for g in range(2, 1000):
        if pow(g, (_R_BLS - 1) // 2, _R_BLS) != 1:
            return g
    raise RuntimeError("no QNR found in [2, 1000) — r should be a prime")


@functools.cache
def primitive_root_of_unity(m: int) -> int:
    """Primitive m-th root of unity in Fr.  Requires m | (r-1) (which
    holds for any power of two ≤ 2^47, since r-1 = 2^47 · t)."""
    if m <= 0 or (m & (m - 1)) != 0:
        raise ValueError(f"m must be a positive power of two, got {m}")
    if (_R_BLS - 1) % m != 0:
        raise ValueError(f"m={m} does not divide r-1")
    g = _quadratic_non_residue()
    psi = pow(g, (_R_BLS - 1) // m, _R_BLS)
    # Sanity: psi^m == 1, psi^(m/2) == -1.
    assert pow(psi, m, _R_BLS) == 1
    assert m == 1 or pow(psi, m // 2, _R_BLS) == _R_BLS - 1
    return psi


def psi_for_ntt_size(n: int) -> int:
    """Primitive 2n-th root of unity, suitable as ``psi`` for NTT3Step."""
    return primitive_root_of_unity(2 * n)


def omega_for_ntt_size(n: int) -> int:
    """Primitive n-th root of unity in Fr.  Equals ``psi^2`` for NTT3Step."""
    return primitive_root_of_unity(n)


if __name__ == "__main__":
    # Quick self-check.
    for n in (4, 8, 16, 32, 64):
        psi = psi_for_ntt_size(n)
        omega = omega_for_ntt_size(n)
        assert pow(omega, n, _R_BLS) == 1
        assert pow(omega, n // 2, _R_BLS) == _R_BLS - 1
        assert (psi * psi) % _R_BLS == omega
        print(f"  n={n}: psi={psi}, omega=psi^2 (verified)")
    print("✓ ntt_params self-check passed")
