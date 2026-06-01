"""
G2 group for BLS12-377.

Curve: y² = x³ + (0,1)  over Fp2   (D-type sextic twist of G1).
The B coefficient (0,1) = u in Fp2 comes from twisting y²=x³+1 by w⁶=u.

Points in projective (Jacobian) coordinates (X : Y : Z), X,Y,Z ∈ Fp2.
Identity: (1 : 1 : 0).
"""

from .params import p, G2_B, G2_GEN_X, G2_GEN_Y
from .field  import (
    FP2_ZERO, FP2_ONE,
    fp2_add, fp2_sub, fp2_mul, fp2_sq, fp2_neg, fp2_inv, fp2_scale, fp2_eq,
)

# ── Affine point ──────────────────────────────────────────────────────────────

class G2Affine:
    __slots__ = ("x", "y", "infinity")

    def __init__(self, x: tuple, y: tuple, infinity: bool = False):
        self.x        = (x[0] % p, x[1] % p)
        self.y        = (y[0] % p, y[1] % p)
        self.infinity = infinity

    def is_infinity(self) -> bool:
        return self.infinity

    def is_on_curve(self) -> bool:
        if self.infinity:
            return True
        # y² = x³ + B_G2  in Fp2
        lhs = fp2_sq(self.y)
        x2  = fp2_sq(self.x)
        x3  = fp2_mul(x2, self.x)
        rhs = fp2_add(x3, G2_B)
        return fp2_eq(lhs, rhs)

    def __eq__(self, other) -> bool:
        if self.infinity and other.infinity:
            return True
        if self.infinity or other.infinity:
            return False
        return fp2_eq(self.x, other.x) and fp2_eq(self.y, other.y)

    def __repr__(self):
        if self.infinity:
            return "G2Affine(∞)"
        return f"G2Affine(x={self.x}, y={self.y})"

    def to_projective(self) -> "G2Projective":
        if self.infinity:
            return G2Projective(FP2_ONE, FP2_ONE, FP2_ZERO)
        return G2Projective(self.x, self.y, FP2_ONE)

    def neg(self) -> "G2Affine":
        if self.infinity:
            return self
        return G2Affine(self.x, fp2_neg(self.y))


# ── Projective (Jacobian) point ───────────────────────────────────────────────

class G2Projective:
    __slots__ = ("X", "Y", "Z")

    def __init__(self, X: tuple, Y: tuple, Z: tuple):
        self.X = (X[0] % p, X[1] % p)
        self.Y = (Y[0] % p, Y[1] % p)
        self.Z = (Z[0] % p, Z[1] % p)

    def is_infinity(self) -> bool:
        return self.Z == FP2_ZERO or (self.Z[0] % p == 0 and self.Z[1] % p == 0)

    def to_affine(self) -> G2Affine:
        if self.is_infinity():
            return G2Affine(FP2_ZERO, FP2_ZERO, infinity=True)
        z_inv  = fp2_inv(self.Z)
        z_inv2 = fp2_sq(z_inv)
        z_inv3 = fp2_mul(z_inv2, z_inv)
        return G2Affine(fp2_mul(self.X, z_inv2), fp2_mul(self.Y, z_inv3))

    def double(self) -> "G2Projective":
        if self.is_infinity():
            return self
        X, Y, Z = self.X, self.Y, self.Z
        if Y == FP2_ZERO or (Y[0] % p == 0 and Y[1] % p == 0):
            return G2Projective(FP2_ONE, FP2_ONE, FP2_ZERO)
        XX  = fp2_sq(X)
        YY  = fp2_sq(Y)
        ZZ  = fp2_sq(Z)
        S   = fp2_scale(fp2_mul(X, YY), 4)
        M   = fp2_scale(XX, 3)   # a=0 curve
        X3  = fp2_sub(fp2_sq(M), fp2_scale(S, 2))
        Y3  = fp2_sub(fp2_mul(M, fp2_sub(S, X3)), fp2_scale(fp2_sq(YY), 8))
        Z3  = fp2_scale(fp2_mul(Y, Z), 2)
        return G2Projective(X3, Y3, Z3)

    def add(self, other: "G2Projective") -> "G2Projective":
        if self.is_infinity():
            return other
        if other.is_infinity():
            return self
        X1, Y1, Z1 = self.X,  self.Y,  self.Z
        X2, Y2, Z2 = other.X, other.Y, other.Z
        Z1Z1 = fp2_sq(Z1)
        Z2Z2 = fp2_sq(Z2)
        U1   = fp2_mul(X1, Z2Z2)
        U2   = fp2_mul(X2, Z1Z1)
        S1   = fp2_mul(Y1, fp2_mul(Z2, Z2Z2))
        S2   = fp2_mul(Y2, fp2_mul(Z1, Z1Z1))
        H    = fp2_sub(U2, U1)
        R    = fp2_sub(S2, S1)
        if fp2_eq(H, FP2_ZERO):
            if fp2_eq(R, FP2_ZERO):
                return self.double()
            return G2Projective(FP2_ONE, FP2_ONE, FP2_ZERO)
        HH  = fp2_sq(H)
        HHH = fp2_mul(H, HH)
        X3  = fp2_sub(fp2_sq(R), fp2_add(HHH, fp2_scale(fp2_mul(U1, HH), 2)))
        Y3  = fp2_sub(fp2_mul(R, fp2_sub(fp2_mul(U1, HH), X3)), fp2_mul(S1, HHH))
        Z3  = fp2_mul(H, fp2_mul(Z1, Z2))
        return G2Projective(X3, Y3, Z3)

    def add_affine(self, other: G2Affine) -> "G2Projective":
        if other.is_infinity():
            return self
        if self.is_infinity():
            return other.to_projective()
        X1, Y1, Z1 = self.X, self.Y, self.Z
        X2, Y2     = other.x, other.y
        Z1Z1 = fp2_sq(Z1)
        U2   = fp2_mul(X2, Z1Z1)
        S2   = fp2_mul(Y2, fp2_mul(Z1, Z1Z1))
        H    = fp2_sub(U2, X1)
        R    = fp2_sub(S2, Y1)
        if fp2_eq(H, FP2_ZERO):
            if fp2_eq(R, FP2_ZERO):
                return self.double()
            return G2Projective(FP2_ONE, FP2_ONE, FP2_ZERO)
        HH  = fp2_sq(H)
        HHH = fp2_mul(H, HH)
        X3  = fp2_sub(fp2_sq(R), fp2_add(HHH, fp2_scale(fp2_mul(X1, HH), 2)))
        Y3  = fp2_sub(fp2_mul(R, fp2_sub(fp2_mul(X1, HH), X3)), fp2_mul(Y1, HHH))
        Z3  = fp2_mul(H, Z1)
        return G2Projective(X3, Y3, Z3)

    def neg(self) -> "G2Projective":
        return G2Projective(self.X, fp2_neg(self.Y), self.Z)

    def scalar_mul(self, k: int) -> "G2Projective":
        result = G2Projective(FP2_ONE, FP2_ONE, FP2_ZERO)
        addend = self
        while k:
            if k & 1:
                result = result.add(addend)
            addend = addend.double()
            k >>= 1
        return result

    def __eq__(self, other) -> bool:
        return self.to_affine() == other.to_affine()

    def __repr__(self):
        return f"G2Proj({self.to_affine()})"


# ── Generator ─────────────────────────────────────────────────────────────────

G2_GENERATOR = G2Affine(G2_GEN_X, G2_GEN_Y)
assert G2_GENERATOR.is_on_curve(), "BLS12-377 G2 generator not on curve — check params.py"
