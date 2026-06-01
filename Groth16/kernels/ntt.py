"""
★ KERNEL: NTT / INTT ★

Number Theoretic Transform over the BLS12-377 scalar field Fr.

Public API
----------
    get_omega(n)          -> primitive n-th root of unity in Fr
    ntt(a, omega)         -> evaluation form  (length-n list)
    intt(a, omega)        -> coefficient form (length-n list)

Algorithm: Cooley-Tukey radix-2 iterative FFT with bit-reversal permutation.

BLS12-377 Fr supports NTT sizes up to 2^47 because 2^47 | (r − 1).

This kernel is a PURE FUNCTION with no Groth16 logic inside.
Call sites in the protocol are explicitly marked with:
    # ═══ KERNEL: NTT ═══  /  # ═══ KERNEL: INTT ═══
"""

from bls12_377.params import r

# ── Primitive root of Fr* ──────────────────────────────────────────────────────
# r − 1 = 2^47 · t  (t odd).  We find a generator g of Fr* and derive ω from it.

def _find_generator() -> int:
    """Return a generator of (Fr)* by checking small candidates."""
    # Precompute the set of (r-1)/q for each prime q | r-1.
    # For our purpose we only need g^((r-1)/2) ≠ 1 (not a square)
    # and g^((r-1)/large_factor) ≠ 1 for the large odd part.
    #
    # Known: r-1 = 2^47 · t where t is the odd cofactor.
    # A generator g satisfies g^((r-1)/q) ≢ 1 (mod r) for all primes q | r-1.
    # We test small primes and g^((r-1)/2) ≠ 1 is the key check.
    for g in range(2, 1000):
        if pow(g, (r - 1) // 2, r) != 1:
            return g
    raise RuntimeError("No generator found in first 1000 candidates")

_GENERATOR = _find_generator()   # generator of Fr*


def get_omega(n: int) -> int:
    """
    Return a primitive n-th root of unity in Fr.

    Requires n to be a power of 2 and n | (r − 1).
    The maximum supported n is 2^47 for BLS12-377.
    """
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
    assert (r - 1) % n == 0, f"n={n} does not divide r-1"
    return pow(_GENERATOR, (r - 1) // n, r)


# ── Bit-reversal permutation ───────────────────────────────────────────────────

def _bit_reverse_copy(a: list) -> list:
    n = len(a)
    bits = n.bit_length() - 1
    result = [0] * n
    for i in range(n):
        rev = int(f"{i:0{bits}b}"[::-1], 2)
        result[rev] = a[i]
    return result


# ── Core NTT ──────────────────────────────────────────────────────────────────

def ntt(a: list, omega: int) -> list:
    """
    ★ NTT KERNEL ★

    Compute the Number Theoretic Transform of `a` over Fr.

        â[k] = Σ_{j=0}^{n-1} a[j] · ω^{jk}  mod r

    Parameters
    ----------
    a     : coefficient vector, length must be a power of 2
    omega : primitive n-th root of unity in Fr (use get_omega(n))

    Returns
    -------
    Evaluation vector â of the same length.
    """
    n = len(a)
    assert n > 0 and (n & (n - 1)) == 0, "length must be a power of 2"

    A = _bit_reverse_copy(a)

    length = 2
    while length <= n:
        w_len = pow(omega, n // length, r)  # primitive length-th root
        for i in range(0, n, length):
            w = 1
            for j in range(length // 2):
                u = A[i + j]
                v = A[i + j + length // 2] * w % r
                A[i + j]              = (u + v) % r
                A[i + j + length // 2] = (u - v) % r
                w = w * w_len % r
        length <<= 1

    return A


def intt(a: list, omega: int) -> list:
    """
    ★ INTT KERNEL ★

    Compute the Inverse NTT of `a` over Fr.

    Parameters
    ----------
    a     : evaluation vector, length must be a power of 2
    omega : the SAME primitive root used in ntt() (we compute its inverse here)

    Returns
    -------
    Coefficient vector of the same length.
    """
    n      = len(a)
    omega_inv = pow(omega, r - 2, r)   # ω⁻¹ in Fr
    result    = ntt(a, omega_inv)
    n_inv     = pow(n, r - 2, r)       # n⁻¹ in Fr
    return [x * n_inv % r for x in result]
