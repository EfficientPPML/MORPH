"""Tests for the MSM (Multi-Scalar Multiplication) kernel."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bls12_377.params import r
from bls12_377.g1 import G1Affine, G1Projective, G1_GENERATOR
from bls12_377.g2 import G2Affine, G2Projective, G2_GENERATOR
from kernels.msm import msm_g1, msm_g2


G1 = G1_GENERATOR
G2 = G2_GENERATOR


# ── G1 MSM ────────────────────────────────────────────────────────────────────

def test_msm_g1_single_scalar():
    """MSM with one point equals scalar multiplication."""
    s = 12345
    expected = G1.to_projective().scalar_mul(s).to_affine()
    result = msm_g1([G1], [s]).to_affine()
    assert result == expected


def test_msm_g1_two_points():
    """MSM([G, G], [a, b]) == [a+b]*G."""
    a, b = 111, 222
    expected = G1.to_projective().scalar_mul(a + b).to_affine()
    result = msm_g1([G1, G1], [a, b]).to_affine()
    assert result == expected


def test_msm_g1_distinct_points():
    """MSM([P, Q], [a, b]) == a*P + b*Q."""
    a, b = 7, 13
    P = G1
    Q = G1.to_projective().scalar_mul(5).to_affine()
    expected = (P.to_projective().scalar_mul(a)
                 .add(Q.to_projective().scalar_mul(b))
                 .to_affine())
    result = msm_g1([P, Q], [a, b]).to_affine()
    assert result == expected


def test_msm_g1_zero_scalar():
    """Zero scalar contributes nothing."""
    P = G1
    Q = G1.to_projective().scalar_mul(99).to_affine()
    expected = Q.to_projective().scalar_mul(3).to_affine()
    result = msm_g1([P, Q], [0, 3]).to_affine()
    assert result == expected


def test_msm_g1_empty():
    """Empty MSM returns identity."""
    result = msm_g1([], [])
    assert result.is_infinity()


def test_msm_g1_identity_point():
    """MSM with identity point and nonzero scalar gives identity contribution."""
    inf = G1Affine(0, 0, infinity=True)
    result = msm_g1([inf, G1], [999, 7]).to_affine()
    expected = G1.to_projective().scalar_mul(7).to_affine()
    assert result == expected


def test_msm_g1_large_scalar():
    """Scalar larger than r is handled correctly (reduced mod r)."""
    s = r + 5
    expected = G1.to_projective().scalar_mul(5).to_affine()
    result = msm_g1([G1], [s]).to_affine()
    assert result == expected


def test_msm_g1_many_points():
    """MSM with several points matches naive sum."""
    n = 10
    points = [G1.to_projective().scalar_mul(i + 1).to_affine() for i in range(n)]
    scalars = [(i * 7 + 3) % r for i in range(n)]

    # Naive
    acc = G1Projective(1, 1, 0)
    for p, s in zip(points, scalars):
        acc = acc.add(p.to_projective().scalar_mul(s))
    expected = acc.to_affine()

    result = msm_g1(points, scalars).to_affine()
    assert result == expected


# ── G2 MSM ────────────────────────────────────────────────────────────────────

def test_msm_g2_single_scalar():
    """MSM with one G2 point equals scalar multiplication."""
    s = 42
    expected = G2.to_projective().scalar_mul(s).to_affine()
    result = msm_g2([G2], [s]).to_affine()
    assert result == expected


def test_msm_g2_two_points():
    """MSM([G2, G2], [a, b]) == [a+b]*G2."""
    a, b = 100, 200
    expected = G2.to_projective().scalar_mul(a + b).to_affine()
    result = msm_g2([G2, G2], [a, b]).to_affine()
    assert result == expected


def test_msm_g2_empty():
    """Empty G2 MSM returns identity."""
    result = msm_g2([], [])
    assert result.is_infinity()


# ── Consistency between G1 and G2 ────────────────────────────────────────────

def test_msm_g1_g2_same_structure():
    """Both kernels handle the same scalar pattern identically (up to group)."""
    scalars = [3, 5, 7]
    g1_points = [G1.to_projective().scalar_mul(i + 1).to_affine() for i in range(3)]
    g2_points = [G2.to_projective().scalar_mul(i + 1).to_affine() for i in range(3)]

    r1 = msm_g1(g1_points, scalars)
    r2 = msm_g2(g2_points, scalars)

    # Both should be non-trivial
    assert not r1.is_infinity()
    assert not r2.is_infinity()

    # Result should have subgroup order r
    assert r1.scalar_mul(r).is_infinity()
    assert r2.scalar_mul(r).is_infinity()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\nAll MSM tests passed.")
