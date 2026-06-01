"""
R1CS (Rank-1 Constraint System) for the cubic circuit.

Circuit: x³ + x + 5 = out
  Witness vector: w = [1, x, sym1, y, out]
                       0   1   2    3   4

Constraints (each row is a tuple (A_row, B_row, C_row) of coefficient dicts):
  #0:  sym1 = x · x       =>  A=[x], B=[x],    C=[sym1]
  #1:  y    = sym1 · x    =>  A=[sym1], B=[x],  C=[y]
  #2:  out  = y + x + 5   =>  A=[1·y + 1·x + 5·1], B=[1], C=[out]
  #3:  (padding to n=4)   =>  A=[1], B=[1],    C=[1]   (dummy: 1·1=1)

n = 4 constraints (padded to power of 2).
m+1 = 5 wires (indices 0..4).
"""

from dataclasses import dataclass, field
from typing      import List, Dict
from bls12_377.params import r


@dataclass
class R1CS:
    """
    R1CS instance with n constraints and m+1 wires.

    Each constraint: (Σ A_i · w_i) · (Σ B_i · w_i) = (Σ C_i · w_i)

    A, B, C are lists of n dicts {wire_index: coefficient_mod_r}.
    num_constraints  = n  (padded to power of 2)
    num_public       = number of public inputs (excluding the constant wire 0)
    num_private      = number of private witnesses
    """
    num_constraints: int
    num_wires:       int      # = m + 1  (includes constant wire at index 0)
    num_public:      int      # public inputs occupy wire indices 1..num_public
    A: List[Dict[int, int]]
    B: List[Dict[int, int]]
    C: List[Dict[int, int]]


def build_cubic_r1cs() -> R1CS:
    """
    Build the R1CS for x³ + x + 5 == out.

    Wires:
        0 : constant 1
        1 : x        (private)
        2 : sym1     (private, = x²)
        3 : y        (private, = x³)
        4 : out      (public,  = 35 when x=3)

    Returns R1CS with n=4 constraints.
    """
    # Constraint 0:  x · x = sym1
    A0 = {1: 1}
    B0 = {1: 1}
    C0 = {2: 1}

    # Constraint 1:  sym1 · x = y
    A1 = {2: 1}
    B1 = {1: 1}
    C1 = {3: 1}

    # Constraint 2:  (y + x + 5) · 1 = out
    #   A = y + x + 5·const,  B = const,  C = out
    A2 = {3: 1, 1: 1, 0: 5}
    B2 = {0: 1}
    C2 = {4: 1}

    # Constraint 3:  (padding)  1 · 1 = 1
    A3 = {0: 1}
    B3 = {0: 1}
    C3 = {0: 1}

    return R1CS(
        num_constraints = 4,
        num_wires       = 5,
        num_public      = 1,   # wire 4 = out
        A = [A0, A1, A2, A3],
        B = [B0, B1, B2, B3],
        C = [C0, C1, C2, C3],
    )


def generate_witness(x_val: int) -> List[int]:
    """
    Given private input x, compute the full witness vector.

    Returns [1, x, x², x³, x³+x+5]  (all values mod r).
    """
    x    = x_val % r
    sym1 = x * x % r
    y    = sym1 * x % r
    out  = (y + x + 5) % r
    return [1, x, sym1, y, out]


def verify_witness(r1cs: R1CS, witness: List[int]) -> bool:
    """Check that witness satisfies every R1CS constraint."""
    def dot(row: Dict[int, int], w: List[int]) -> int:
        return sum(coeff * w[idx] for idx, coeff in row.items()) % r

    for i in range(r1cs.num_constraints):
        a = dot(r1cs.A[i], witness)
        b = dot(r1cs.B[i], witness)
        c = dot(r1cs.C[i], witness)
        if a * b % r != c:
            return False
    return True
