"""Tests for the NTT / INTT kernel."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bls12_377.params import r
from kernels.ntt import get_omega, ntt, intt


def fr_mul(a, b):
    return a * b % r


# ── get_omega ─────────────────────────────────────────────────────────────────

def test_omega_is_primitive_root():
    """ω^n == 1 and ω^(n/2) != 1 for several sizes."""
    for log_n in [1, 2, 3, 4, 8, 16]:
        n = 1 << log_n
        omega = get_omega(n)
        assert pow(omega, n, r) == 1, f"ω^{n} != 1 for n={n}"
        if n > 1:
            assert pow(omega, n // 2, r) != 1, f"ω is not primitive for n={n}"


def test_omega_max_size():
    """BLS12-377 Fr supports NTT up to 2^47."""
    n = 1 << 47
    omega = get_omega(n)
    # Only check ω^n == 1 (computing ω^(n/2) is fine with pow)
    assert pow(omega, n, r) == 1


# ── NTT / INTT round-trip ────────────────────────────────────────────────────

def test_ntt_intt_roundtrip_small():
    """NTT then INTT recovers original coefficients."""
    n = 8
    omega = get_omega(n)
    a = [i + 1 for i in range(n)]  # [1, 2, 3, ..., 8]
    a_hat = ntt(a, omega)
    a_back = intt(a_hat, omega)
    assert a_back == [x % r for x in a]


def test_ntt_intt_roundtrip_random():
    """Round-trip with larger random-ish coefficients."""
    import hashlib
    n = 16
    omega = get_omega(n)
    # Deterministic pseudo-random coefficients
    a = [int(hashlib.sha256(i.to_bytes(4, "big")).hexdigest(), 16) % r for i in range(n)]
    assert intt(ntt(a, omega), omega) == a


# ── NTT correctness vs naive DFT ─────────────────────────────────────────────

def test_ntt_matches_naive_dft():
    """NTT output matches the definition: â[k] = Σ a[j]·ω^{jk}."""
    n = 8
    omega = get_omega(n)
    a = [3, 1, 4, 1, 5, 9, 2, 6]

    a_hat = ntt(a, omega)

    for k in range(n):
        expected = 0
        for j in range(n):
            expected = (expected + a[j] * pow(omega, j * k, r)) % r
        assert a_hat[k] == expected, f"mismatch at k={k}"


# ── Convolution property ─────────────────────────────────────────────────────

def test_ntt_convolution():
    """Pointwise multiply in evaluation domain == polynomial multiplication."""
    n = 8
    omega = get_omega(n)

    # Two polynomials of degree 3 (padded to length 8 for no aliasing)
    a = [1, 2, 3, 4, 0, 0, 0, 0]
    b = [5, 6, 7, 8, 0, 0, 0, 0]

    # Naive polynomial multiplication
    c_naive = [0] * (7)  # degree 6
    for i in range(4):
        for j in range(4):
            c_naive[i + j] = (c_naive[i + j] + a[i] * b[j]) % r
    c_naive += [0]  # pad to length 8

    # NTT-based multiplication
    a_hat = ntt(a, omega)
    b_hat = ntt(b, omega)
    c_hat = [fr_mul(a_hat[k], b_hat[k]) for k in range(n)]
    c_ntt = intt(c_hat, omega)

    assert c_ntt == c_naive


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_ntt_all_zeros():
    n = 4
    omega = get_omega(n)
    assert ntt([0] * n, omega) == [0] * n
    assert intt([0] * n, omega) == [0] * n


def test_ntt_size_one():
    omega = get_omega(1)
    assert ntt([42], omega) == [42]
    assert intt([42], omega) == [42]


def test_ntt_size_two():
    omega = get_omega(2)
    a = [3, 7]
    a_hat = ntt(a, omega)
    assert a_hat[0] == (3 + 7) % r
    assert a_hat[1] == (3 + 7 * omega) % r
    assert intt(a_hat, omega) == a


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\nAll NTT tests passed.")
