"""
QAP (Quadratic Arithmetic Program) — R1CS → QAP conversion.

For each wire i, we interpolate three polynomials:
    U_i(x), V_i(x), W_i(x)  of degree n−1

such that the R1CS constraint at evaluation point h_j (the j-th root of unity)
corresponds to the j-th constraint:
    U_i(h_j) = A[j][i],  V_i(h_j) = B[j][i],  W_i(h_j) = C[j][i]

Target polynomial: t(x) = Π_{j=0}^{n-1} (x − h_j)  (vanishing polynomial)

The Groth16 prover uses these to check:
    A(x) · B(x) − C(x) = H(x) · t(x)    in Fr[x]

where A(x) = Σ a_i · U_i(x),  B(x) = Σ a_i · V_i(x),  C(x) = Σ a_i · W_i(x).

═══ KERNEL: NTT used here for polynomial interpolation ═══
"""

from dataclasses import dataclass
from typing      import List
from bls12_377.params import r
from groth16.r1cs     import R1CS
from kernels.ntt      import get_omega, ntt, intt


@dataclass
class QAP:
    """
    Quadratic Arithmetic Program derived from an R1CS.

    Fields
    ------
    n            : number of constraints (power of 2)
    ntt_size     : size used for NTT in the prover (= 2n, to hold H(x))
    degree       : polynomial degree = n − 1
    num_wires    : m + 1
    num_public   : number of public wires
    omega        : primitive n-th root of unity in Fr  (for setup/evaluation)
    omega2n      : primitive 2n-th root of unity       (for prover)
    t_coeffs     : coefficients of t(x)  (length n+1)
    U            : list of n coefficient vectors, one per wire, length n
    V            : same for V
    W            : same for W
    """
    n:          int
    ntt_size:   int
    degree:     int
    num_wires:  int
    num_public: int
    omega:      int
    omega2n:    int
    t_coeffs:   List[int]
    U:          List[List[int]]   # U[wire][coeff_index]
    V:          List[List[int]]
    W:          List[List[int]]


def _interpolate(evals: List[int], omega: int) -> List[int]:
    """
    Given evaluation vector evals[0..n-1] at the n-th roots of unity,
    return the coefficient vector of the interpolating polynomial via INTT.

    ═══ KERNEL: INTT ═══════════════════════════════════════════════════
    """
    return intt(evals, omega)
    # ════════════════════════════════════════════════════════════════════


def _vanishing_poly(n: int) -> List[int]:
    """
    Return coefficients of t(x) = x^n − 1 = −1 + x^n
    (the product Π(x − ω^j) over the n-th roots of unity equals x^n−1).

    Returns coefficient list of length n+1 (index = degree).
    """
    coeffs = [0] * (n + 1)
    coeffs[0] = r - 1   # −1 mod r
    coeffs[n] = 1
    return coeffs


def r1cs_to_qap(r1cs: R1CS) -> QAP:
    """
    Convert an R1CS into a QAP by interpolating each wire's constraint columns.

    Uses the NTT domain {ω^0, ω^1, ..., ω^{n-1}} as evaluation points.
    """
    n = r1cs.num_constraints
    assert n > 0 and (n & (n - 1)) == 0, "num_constraints must be power of 2"

    omega   = get_omega(n)       # primitive n-th root of unity
    omega2n = get_omega(2 * n)   # primitive 2n-th root (for prover's H computation)

    # For each wire i, build evaluation vectors from the R1CS matrices
    # eval_U[i][j] = A[j].get(i, 0)
    U_coeffs = []
    V_coeffs = []
    W_coeffs = []

    for i in range(r1cs.num_wires):
        eval_U = [r1cs.A[j].get(i, 0) % r for j in range(n)]
        eval_V = [r1cs.B[j].get(i, 0) % r for j in range(n)]
        eval_W = [r1cs.C[j].get(i, 0) % r for j in range(n)]

        # ═══ KERNEL: INTT (polynomial interpolation) ════════════════════════
        U_coeffs.append(intt(eval_U, omega))
        V_coeffs.append(intt(eval_V, omega))
        W_coeffs.append(intt(eval_W, omega))
        # ════════════════════════════════════════════════════════════════════

    t_coeffs = _vanishing_poly(n)

    return QAP(
        n          = n,
        ntt_size   = 2 * n,
        degree     = n - 1,
        num_wires  = r1cs.num_wires,
        num_public = r1cs.num_public,
        omega      = omega,
        omega2n    = omega2n,
        t_coeffs   = t_coeffs,
        U          = U_coeffs,
        V          = V_coeffs,
        W          = W_coeffs,
    )


def eval_poly(coeffs: List[int], x: int) -> int:
    """Evaluate a polynomial (coefficient list) at x in Fr."""
    result = 0
    xi = 1
    for c in coeffs:
        result = (result + c * xi) % r
        xi = xi * x % r
    return result


def poly_mul(a: List[int], b: List[int]) -> List[int]:
    """Multiply two polynomials in Fr[x] (coefficient lists)."""
    if not a or not b:
        return [0]
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] = (result[i + j] + ai * bj) % r
    return result
