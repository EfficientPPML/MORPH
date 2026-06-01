"""
Groth16 Full-ZK Demo — BLS12-377
=================================
Circuit:  x³ + x + 5 == out
Witness:  x = 3  =>  out = 35
Curve:    BLS12-377
Kernels:  NTT / INTT  (polynomial arithmetic)
          MSM G1 / G2  (Pippenger's algorithm)
"""

import sys
import time

from groth16.r1cs     import build_cubic_r1cs, generate_witness, verify_witness
from groth16.qap      import r1cs_to_qap
from groth16.setup    import trusted_setup
from groth16.prover   import prove
from groth16.verifier import verify

SEP = "=" * 62


def main():
    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 0 — CIRCUIT SETUP
    # ──────────────────────────────────────────────────────────────────────────
    print(SEP)
    print("PHASE 0: CIRCUIT SETUP")
    print(SEP)
    t0 = time.time()

    print("  Circuit  : x³ + x + 5 == out")
    print("  Wires    : [1, x, sym1, y, out]  (5 wires)")
    print("  R1CS     : 3 constraints  (padded to n=4)")

    r1cs = build_cubic_r1cs()

    # ─── KERNEL: NTT used inside r1cs_to_qap for polynomial interpolation ───
    qap = r1cs_to_qap(r1cs)
    # ─────────────────────────────────────────────────────────────────────────

    print(f"  QAP degree      : {qap.degree}")
    print(f"  NTT domain size : {qap.ntt_size}  (= 2n, for H-poly multiplication)")
    print(f"  Elapsed         : {time.time()-t0:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1 — TRUSTED SETUP
    # ──────────────────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("PHASE 1: TRUSTED SETUP")
    print(SEP)
    t1 = time.time()

    print("  Sampling toxic waste (τ, α, β, γ, δ)  [random] ...")
    print("  [MSM G1] encoding proving key points ...")
    print("  [MSM G2] encoding verification key points ...")

    pk, vk = trusted_setup(qap)

    print(f"  pk.A_g1      : {len(pk.A_g1)} G1 points  (one per wire)")
    print(f"  pk.B_g2      : {len(pk.B_g2)} G2 points  (one per wire)")
    print(f"  pk.private_g1: {len(pk.private_g1)} G1 points  (private wires)")
    print(f"  pk.h_g1      : {len(pk.h_g1)} G1 points  (H-poly CRS, degree n-2)")
    print(f"  vk.ic        : {len(vk.ic)} G1 points  (constant + public inputs)")
    print("  Toxic waste discarded  (not returned to caller in production)")
    print(f"  Elapsed : {time.time()-t1:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2 — GENERATE PROOF
    # ──────────────────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("PHASE 2: GENERATE PROOF")
    print(SEP)
    t2 = time.time()

    x_secret      = 3
    witness       = generate_witness(x_secret)
    public_inputs = [witness[4]]   # out = 35

    assert verify_witness(r1cs, witness), "Witness does not satisfy R1CS!"

    print(f"  Witness       : {witness}")
    print(f"  Public inputs : {public_inputs}  (out = x³+x+5 = {public_inputs[0]})")
    print(f"  Private       : x={x_secret} is hidden from the verifier")
    print()
    print("  [MSM G1] computing π_A  ...")
    print("  [MSM G2] computing π_B  ...")
    print("  [NTT]    evaluating A(x), B(x), C(x) at 2n points ...")
    print("  [INTT]   this step uses polynomial division (direct for tiny n) ...")
    print("  [MSM G1] computing H(τ)·t(τ)/δ contribution to π_C ...")
    print("  [MSM G1] computing private-wire sum for π_C ...")
    print("  Sampling ZK blinding scalars r, s ∈ Fr ...")

    proof = prove(pk, qap, witness)

    print()
    print(f"  π_A : {proof.A}")
    print(f"  π_B : {proof.B}")
    print(f"  π_C : {proof.C}")
    print(f"  Elapsed : {time.time()-t2:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3 — VERIFY
    # ──────────────────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("PHASE 3: VERIFY PROOF")
    print(SEP)
    t3 = time.time()

    print("  [MSM G1] aggregating public inputs into vk_pub ...")
    print("  Running pairing checks ...")
    print("  Checking: e(A,B) == e(α,β) · e(vk_pub,γ) · e(C,δ)")
    print()

    valid = verify(vk, public_inputs, proof)

    print()
    print(f"  Elapsed : {time.time()-t3:.2f}s")
    print()
    print(SEP)
    if valid:
        print("  RESULT: Proof VALID  ✓")
        print(f"  Verifier accepts: prover knows x s.t. x³+x+5={public_inputs[0]}")
    else:
        print("  RESULT: Proof INVALID  ✗")
        sys.exit(1)
    print(SEP)
    print(f"\n  Total elapsed: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
