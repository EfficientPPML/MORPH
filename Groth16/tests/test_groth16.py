"""End-to-end Groth16 protocol tests."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bls12_377.params import r
from bls12_377.g1 import G1Affine, G1_GENERATOR
from groth16.r1cs import build_cubic_r1cs
from groth16.qap import r1cs_to_qap
from groth16.setup import trusted_setup
from groth16.prover import prove, Proof
from groth16.verifier import verify


def _setup():
    """Shared setup for all tests."""
    r1cs = build_cubic_r1cs()
    qap = r1cs_to_qap(r1cs)
    pk, vk = trusted_setup(qap)
    return qap, pk, vk


# ── Completeness ──────────────────────────────────────────────────────────────

def test_completeness_x3():
    """x=3, out=35 verifies."""
    qap, pk, vk = _setup()
    proof = prove(pk, qap, [1, 3, 9, 27, 35])
    assert verify(vk, [35], proof)


def test_completeness_x1():
    """x=1, out=7 verifies."""
    qap, pk, vk = _setup()
    proof = prove(pk, qap, [1, 1, 1, 1, 7])
    assert verify(vk, [7], proof)


def test_completeness_x0():
    """x=0, out=5 verifies."""
    qap, pk, vk = _setup()
    proof = prove(pk, qap, [1, 0, 0, 0, 5])
    assert verify(vk, [5], proof)


# ── Soundness ─────────────────────────────────────────────────────────────────

def test_soundness_wrong_public_input():
    """Correct witness but wrong claimed public input rejects."""
    qap, pk, vk = _setup()
    proof = prove(pk, qap, [1, 3, 9, 27, 35])
    assert not verify(vk, [99], proof)


def test_soundness_tampered_pi_A():
    """Random π_A rejects."""
    qap, pk, vk = _setup()
    proof = prove(pk, qap, [1, 3, 9, 27, 35])
    fake_A = G1_GENERATOR.to_projective().scalar_mul(12345).to_affine()
    assert not verify(vk, [35], Proof(A=fake_A, B=proof.B, C=proof.C))


def test_soundness_tampered_pi_C():
    """Random π_C rejects."""
    qap, pk, vk = _setup()
    proof = prove(pk, qap, [1, 3, 9, 27, 35])
    fake_C = G1_GENERATOR.to_projective().scalar_mul(67890).to_affine()
    assert not verify(vk, [35], Proof(A=proof.A, B=proof.B, C=fake_C))


def test_soundness_identity_A():
    """Proof with A = infinity is rejected."""
    qap, pk, vk = _setup()
    proof = prove(pk, qap, [1, 3, 9, 27, 35])
    inf = G1Affine(0, 0, infinity=True)
    assert not verify(vk, [35], Proof(A=inf, B=proof.B, C=proof.C))


# ── Zero-knowledge ────────────────────────────────────────────────────────────

def test_zk_proofs_differ():
    """Two proofs for the same statement have different components."""
    qap, pk, vk = _setup()
    p1 = prove(pk, qap, [1, 3, 9, 27, 35])
    p2 = prove(pk, qap, [1, 3, 9, 27, 35])
    assert p1.A != p2.A
    assert p1.B != p2.B
    assert p1.C != p2.C
    assert verify(vk, [35], p1)
    assert verify(vk, [35], p2)


# ── Input validation ─────────────────────────────────────────────────────────

def test_verifier_rejects_wrong_input_count():
    """Verifier raises on wrong number of public inputs."""
    qap, pk, vk = _setup()
    proof = prove(pk, qap, [1, 3, 9, 27, 35])
    try:
        verify(vk, [35, 99], proof)  # too many
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        verify(vk, [], proof)  # too few
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ── Fixed toxic waste (deterministic) ────────────────────────────────────────

def test_deterministic_setup():
    """With fixed toxic waste, the same proof inputs give a valid proof."""
    r1cs = build_cubic_r1cs()
    qap = r1cs_to_qap(r1cs)
    tw = {"tau": 7, "alpha": 11, "beta": 13, "gamma": 17, "delta": 19}
    pk, vk = trusted_setup(qap, toxic_waste=tw)
    proof = prove(pk, qap, [1, 3, 9, 27, 35])
    assert verify(vk, [35], proof)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\nAll Groth16 tests passed.")
