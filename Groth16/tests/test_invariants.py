"""
Layered mathematical invariant verification for the Groth16 implementation.

This script verifies 27 invariants across 7 layers of the stack, from
field arithmetic up through the full Groth16 protocol.  Each layer's
invariants can only hold if every layer below it is correct — a bug at
any level cascades upward and is caught.

Layer dependency graph:

    Layer 7: Groth16 protocol      ← uses layers 3-6
    Layer 6: QAP divisibility       ← uses layer 4 (NTT)
    Layer 5: MSM linearity          ← uses layer 2 (curve groups)
    Layer 4: NTT convolution        ← uses layer 1 (Fr field)
    Layer 3: Pairing bilinearity    ← uses layers 1+2 (fields + curves)
    Layer 2: Curve group order      ← uses layer 1 (Fp field)
    Layer 1: Field primality        ← foundation

Run:  python tests/test_invariants.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bls12_377.params import p, r
from bls12_377.field import (
    FP2_ONE, FP12_ONE,
    fp2_inv, fp2_mul, fp2_eq,
    fp12_mul, fp12_sq, fp12_pow, fp12_eq, fp12_frob, fp12_conj,
)
from bls12_377.g1 import G1_GENERATOR, G1Projective
from bls12_377.g2 import G2_GENERATOR, G2Projective
from bls12_377.pairing import ate_pairing
from kernels.ntt import get_omega, ntt, intt
from kernels.msm import msm_g1, msm_g2
from groth16.r1cs import build_cubic_r1cs
from groth16.qap import r1cs_to_qap, eval_poly
from groth16.setup import trusted_setup
from groth16.prover import prove, compute_h_coset_ntt, poly_divmod
from groth16.verifier import verify


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: FIELD ARITHMETIC — Fp, Fr, Fp2, Fp6, Fp12
# ═══════════════════════════════════════════════════════════════════════════════

def test_1a_p_is_prime():
    """Fermat test: 2^(p-1) ≡ 1 (mod p)."""
    assert pow(2, p - 1, p) == 1

def test_1b_r_is_prime():
    """Fermat test: 2^(r-1) ≡ 1 (mod r)."""
    assert pow(2, r - 1, r) == 1

def test_1c_fp2_inverse():
    """a · a⁻¹ = 1 in Fp2."""
    a = (17, 23)
    assert fp2_eq(fp2_mul(a, fp2_inv(a)), FP2_ONE)

def test_1d_fp12_frobenius_12_is_identity():
    """Frob^12 = id (Galois theory: [Fp12 : Fp] = 12)."""
    x = fp12_sq(
        (((3, 5), (7, 11), (13, 17)), ((19, 23), (29, 31), (37, 41)))
    )
    assert fp12_eq(fp12_frob(x, 12), x)

def test_1e_fp12_frobenius_6_is_conjugation():
    """Frob^6 = conjugation on Fp12 = Fp6[w]/(w²-v)."""
    x = fp12_sq(
        (((3, 5), (7, 11), (13, 17)), ((19, 23), (29, 31), (37, 41)))
    )
    assert fp12_eq(fp12_frob(x, 6), fp12_conj(x))

def test_1f_fp12_mul_associative():
    """(a·b)·c = a·(b·c) in Fp12."""
    x = fp12_sq(
        (((3, 5), (7, 11), (13, 17)), ((19, 23), (29, 31), (37, 41)))
    )
    y = fp12_sq(x)
    z = fp12_mul(x, y)
    assert fp12_eq(fp12_mul(fp12_mul(x, y), z), fp12_mul(x, fp12_mul(y, z)))


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: ELLIPTIC CURVE GROUPS — G1, G2
# ═══════════════════════════════════════════════════════════════════════════════

def test_2a_g1_on_curve():
    assert G1_GENERATOR.is_on_curve()

def test_2b_g2_on_curve():
    assert G2_GENERATOR.is_on_curve()

def test_2c_g1_subgroup_order():
    """[r]·G1 = O (generator has prime order r)."""
    assert G1_GENERATOR.to_projective().scalar_mul(r).is_infinity()

def test_2d_g2_subgroup_order():
    """[r]·G2 = O (generator has prime order r)."""
    assert G2_GENERATOR.to_projective().scalar_mul(r).is_infinity()

def test_2e_g1_scalar_mul_homomorphism():
    """[a+b]·G = [a]·G + [b]·G."""
    G = G1_GENERATOR.to_projective()
    a, b = 1234567, 7654321
    lhs = G.scalar_mul(a + b).to_affine()
    rhs = G.scalar_mul(a).add(G.scalar_mul(b)).to_affine()
    assert lhs == rhs


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: PAIRING — bilinearity, non-degeneracy, order
# ═══════════════════════════════════════════════════════════════════════════════

def _e_base():
    return ate_pairing(G1_GENERATOR, G2_GENERATOR)

def test_3a_pairing_non_degenerate():
    """e(G1, G2) ≠ 1."""
    assert not fp12_eq(_e_base(), FP12_ONE)

def test_3b_pairing_gt_order():
    """e(G1, G2)^r = 1 (GT has order r)."""
    assert fp12_eq(fp12_pow(_e_base(), r), FP12_ONE)

def test_3c_pairing_left_linear():
    """e([a]·G1, G2) = e(G1, G2)^a."""
    a = 42
    G1p = G1_GENERATOR.to_projective()
    lhs = ate_pairing(G1p.scalar_mul(a).to_affine(), G2_GENERATOR)
    rhs = fp12_pow(_e_base(), a)
    assert fp12_eq(lhs, rhs)

def test_3d_pairing_right_linear():
    """e(G1, [a]·G2) = e(G1, G2)^a."""
    a = 42
    G2p = G2_GENERATOR.to_projective()
    lhs = ate_pairing(G1_GENERATOR, G2p.scalar_mul(a).to_affine())
    rhs = fp12_pow(_e_base(), a)
    assert fp12_eq(lhs, rhs)

def test_3e_pairing_cross_linear():
    """e([a]·G1, [b]·G2) = e(G1, G2)^{a·b}."""
    a, b = 42, 7
    G1p = G1_GENERATOR.to_projective()
    G2p = G2_GENERATOR.to_projective()
    lhs = ate_pairing(G1p.scalar_mul(a).to_affine(), G2p.scalar_mul(b).to_affine())
    rhs = fp12_pow(_e_base(), a * b)
    assert fp12_eq(lhs, rhs)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4: NTT KERNEL — roots of unity, DFT definition, convolution
# ═══════════════════════════════════════════════════════════════════════════════

def test_4a_omega_primitive():
    """ω^n = 1 and ω^(n/2) ≠ 1 for several sizes."""
    for k in [4, 8, 16]:
        omega = get_omega(k)
        assert pow(omega, k, r) == 1
        assert pow(omega, k // 2, r) != 1

def test_4b_ntt_matches_naive_dft():
    """NTT output matches the definition: â[k] = Σ a[j]·ω^{jk}."""
    n = 8
    omega = get_omega(n)
    a = [3, 1, 4, 1, 5, 9, 2, 6]
    a_hat = ntt(a, omega)
    for k in range(n):
        naive = sum(a[j] * pow(omega, j * k, r) for j in range(n)) % r
        assert a_hat[k] == naive

def test_4c_convolution_theorem():
    """INTT(NTT(a) ⊙ NTT(b)) = a * b  (polynomial multiplication)."""
    n = 8
    omega = get_omega(n)
    a = [1, 2, 3, 0, 0, 0, 0, 0]
    b = [4, 5, 6, 0, 0, 0, 0, 0]
    c_ntt = intt([ntt(a, omega)[i] * ntt(b, omega)[i] % r for i in range(n)], omega)
    c_naive = [0] * 5
    for i in range(3):
        for j in range(3):
            c_naive[i + j] = (c_naive[i + j] + a[i] * b[j]) % r
    assert c_ntt[:5] == c_naive and c_ntt[5:] == [0, 0, 0]


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 5: MSM KERNEL — linearity
# ═══════════════════════════════════════════════════════════════════════════════

def test_5a_msm_sum():
    """MSM([G,G,G],[a,b,c]) = [a+b+c]·G."""
    G = G1_GENERATOR.to_projective()
    abc = [111, 222, 333]
    assert msm_g1([G1_GENERATOR] * 3, abc).to_affine() == G.scalar_mul(sum(abc)).to_affine()

def test_5b_msm_scalar_linearity():
    """MSM(P, a·s) = [a]·MSM(P, s)."""
    G = G1_GENERATOR.to_projective()
    pts = [G.scalar_mul(i + 1).to_affine() for i in range(5)]
    s = [3, 1, 4, 1, 5]
    a = 17
    lhs = msm_g1(pts, [x * a % r for x in s]).to_affine()
    rhs = msm_g1(pts, s).scalar_mul(a).to_affine()
    assert lhs == rhs


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 6: QAP — R1CS satisfaction and polynomial divisibility
# ═══════════════════════════════════════════════════════════════════════════════

def _qap_fixture():
    r1cs = build_cubic_r1cs()
    qap = r1cs_to_qap(r1cs)
    return r1cs, qap

def test_6a_r1cs_satisfied():
    """A[j]·w ⊙ B[j]·w = C[j]·w for each constraint."""
    r1cs, _ = _qap_fixture()
    witness = [1, 3, 9, 27, 35]
    for j in range(r1cs.num_constraints):
        aw = sum(r1cs.A[j].get(i, 0) * witness[i] for i in range(r1cs.num_wires)) % r
        bw = sum(r1cs.B[j].get(i, 0) * witness[i] for i in range(r1cs.num_wires)) % r
        cw = sum(r1cs.C[j].get(i, 0) * witness[i] for i in range(r1cs.num_wires)) % r
        assert aw * bw % r == cw

def test_6b_qap_divisibility():
    """A(τ)·B(τ) - C(τ) = H(τ)·t(τ) at a random evaluation point."""
    _, qap = _qap_fixture()
    n = qap.n
    witness = [1, 3, 9, 27, 35]
    tau = 9999999937

    def combine_at(polys, w, x):
        return sum(w[i] * eval_poly(polys[i], x) for i in range(len(w))) % r

    def combine_coeffs(polys, w):
        result = [0] * n
        for i in range(len(w)):
            for j in range(n):
                result[j] = (result[j] + w[i] * polys[i][j]) % r
        return result

    A_c = combine_coeffs(qap.U, witness)
    B_c = combine_coeffs(qap.V, witness)
    C_c = combine_coeffs(qap.W, witness)
    h = compute_h_coset_ntt(A_c, B_c, C_c, n, qap.omega, qap.omega2n)

    lhs = (combine_at(qap.U, witness, tau) * combine_at(qap.V, witness, tau)
           - combine_at(qap.W, witness, tau)) % r
    rhs = eval_poly(h, tau) * eval_poly(qap.t_coeffs, tau) % r
    assert lhs == rhs

def test_6c_coset_ntt_matches_poly_divmod():
    """Two independent H(x) computations produce identical results."""
    _, qap = _qap_fixture()
    n = qap.n
    witness = [1, 3, 9, 27, 35]

    def combine_coeffs(polys, w):
        result = [0] * n
        for i in range(len(w)):
            for j in range(n):
                result[j] = (result[j] + w[i] * polys[i][j]) % r
        return result

    A_c = combine_coeffs(qap.U, witness)
    B_c = combine_coeffs(qap.V, witness)
    C_c = combine_coeffs(qap.W, witness)

    h_ntt = compute_h_coset_ntt(A_c, B_c, C_c, n, qap.omega, qap.omega2n)

    AB = [0] * (2 * n - 1)
    for i in range(n):
        for j in range(n):
            AB[i + j] = (AB[i + j] + A_c[i] * B_c[j]) % r
    for i in range(n):
        AB[i] = (AB[i] - C_c[i]) % r
    h_div = poly_divmod(AB, n)

    assert h_ntt == h_div


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 7: GROTH16 PROTOCOL — completeness, soundness, zero-knowledge
# ═══════════════════════════════════════════════════════════════════════════════

def _protocol_fixture():
    r1cs = build_cubic_r1cs()
    qap = r1cs_to_qap(r1cs)
    tw = {"tau": 7, "alpha": 11, "beta": 13, "gamma": 17, "delta": 19}
    pk, vk = trusted_setup(qap, toxic_waste=tw)
    return qap, pk, vk

def test_7a_completeness():
    """Valid proofs verify for x = 0, 1, 2, 3, 5, 10."""
    qap, pk, vk = _protocol_fixture()
    for x in [0, 1, 2, 3, 5, 10]:
        sym1 = x * x % r
        y = sym1 * x % r
        out = (y + x + 5) % r
        proof = prove(pk, qap, [1, x, sym1, y, out])
        assert verify(vk, [out], proof), f"completeness fails for x={x}"

def test_7b_soundness():
    """Wrong public inputs are rejected."""
    qap, pk, vk = _protocol_fixture()
    proof = prove(pk, qap, [1, 3, 9, 27, 35])
    for bad in [0, 1, 34, 36]:
        assert not verify(vk, [bad], proof)

def test_7c_zero_knowledge():
    """Two proofs for the same statement differ but both verify."""
    qap, pk, vk = _protocol_fixture()
    p1 = prove(pk, qap, [1, 3, 9, 27, 35])
    p2 = prove(pk, qap, [1, 3, 9, 27, 35])
    assert p1.A != p2.A and p1.B != p2.B and p1.C != p2.C
    assert verify(vk, [35], p1) and verify(vk, [35], p2)

def test_7d_pairing_equation_manual():
    """Independently recompute e(A,B) = e(α,β)·e(vk_pub,γ)·e(C,δ)."""
    qap, pk, vk = _protocol_fixture()
    proof = prove(pk, qap, [1, 3, 9, 27, 35])
    vk_pub = msm_g1(vk.ic, [1, 35]).to_affine()
    e_AB = ate_pairing(proof.A, proof.B)
    e_ab = ate_pairing(vk.alpha_g1, vk.beta_g2)
    e_vk = ate_pairing(vk_pub, vk.gamma_g2)
    e_cd = ate_pairing(proof.C, vk.delta_g2)
    assert fp12_eq(e_AB, fp12_mul(fp12_mul(e_ab, e_vk), e_cd))


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    layers = {
        "1": "FIELD ARITHMETIC",
        "2": "ELLIPTIC CURVE GROUPS",
        "3": "PAIRING",
        "4": "NTT KERNEL",
        "5": "MSM KERNEL",
        "6": "QAP",
        "7": "GROTH16 PROTOCOL",
    }

    tests = sorted(
        [(name, fn) for name, fn in globals().items()
         if name.startswith("test_") and callable(fn)],
        key=lambda x: x[0],
    )

    current_layer = None
    passed = 0
    for name, fn in tests:
        layer = name.split("_")[1][0]
        if layer != current_layer:
            current_layer = layer
            print(f"\n{'=' * 60}")
            print(f"LAYER {layer}: {layers.get(layer, '???')}")
            print(f"{'=' * 60}")
        fn()
        label = fn.__doc__ or name
        print(f"  {label:50s} PASS")
        passed += 1

    print(f"\n{'=' * 60}")
    print(f"  ALL {passed} INVARIANTS VERIFIED ACROSS {len(layers)} LAYERS")
    print(f"{'=' * 60}")
