"""
G1 group for BLS12-377.

Curve: y² = x³ + 1  over Fp.

Points are stored in projective (Jacobian) coordinates (X : Y : Z)
representing the affine point (X/Z², Y/Z³).

The identity (point at infinity) is represented as (1 : 1 : 0).
"""

from .params import p, r, G1_B, G1_GEN_X, G1_GEN_Y
from .field  import fp_add, fp_sub, fp_mul, fp_inv, fp_neg


# ── Affine point ──────────────────────────────────────────────────────────────

class G1Affine:
    __slots__ = ("x", "y", "infinity")

    def __init__(self, x: int, y: int, infinity: bool = False):
        self.x        = x % p
        self.y        = y % p
        self.infinity = infinity

    def is_infinity(self) -> bool:
        return self.infinity

    def is_on_curve(self) -> bool:
        if self.infinity:
            return True
        lhs = fp_mul(self.y, self.y)
        rhs = (fp_mul(fp_mul(self.x, self.x), self.x) + G1_B) % p
        return lhs == rhs

    def __eq__(self, other) -> bool:
        if self.infinity and other.infinity:
            return True
        if self.infinity or other.infinity:
            return False
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        if self.infinity:
            return "G1Affine(∞)"
        return f"G1Affine(x=0x{self.x:x}, y=0x{self.y:x})"

    def to_projective(self) -> "G1Projective":
        if self.infinity:
            return G1Projective(1, 1, 0)
        return G1Projective(self.x, self.y, 1)

    def neg(self) -> "G1Affine":
        if self.infinity:
            return self
        return G1Affine(self.x, fp_neg(self.y))


# ── Projective (Jacobian) point ───────────────────────────────────────────────

class G1Projective:
    __slots__ = ("X", "Y", "Z")

    def __init__(self, X: int, Y: int, Z: int):
        self.X = X % p
        self.Y = Y % p
        self.Z = Z % p

    def is_infinity(self) -> bool:
        return self.Z == 0

    def to_affine(self) -> G1Affine:
        if self.is_infinity():
            return G1Affine(0, 0, infinity=True)
        z_inv  = fp_inv(self.Z)
        z_inv2 = fp_mul(z_inv, z_inv)
        z_inv3 = fp_mul(z_inv2, z_inv)
        return G1Affine(fp_mul(self.X, z_inv2), fp_mul(self.Y, z_inv3))

    def double(self) -> "G1Projective":
        """Point doubling in Jacobian coordinates (a=0 curve)."""
        if self.is_infinity():
            return self
        X, Y, Z = self.X, self.Y, self.Z
        if Y == 0:
            return G1Projective(1, 1, 0)
        # Formulas from https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
        XX  = fp_mul(X, X)
        YY  = fp_mul(Y, Y)
        ZZ  = fp_mul(Z, Z)
        S   = fp_mul(4, fp_mul(X, YY))
        M   = fp_mul(3, XX)   # a=0, so no ZZ⁴ term
        X3  = fp_sub(fp_mul(M, M), fp_mul(2, S))
        Y3  = fp_sub(fp_mul(M, fp_sub(S, X3)), fp_mul(8, fp_mul(YY, YY)))
        Z3  = fp_mul(2, fp_mul(Y, Z))
        return G1Projective(X3, Y3, Z3)

    def add(self, other: "G1Projective") -> "G1Projective":
        """Point addition in Jacobian coordinates."""
        if self.is_infinity():
            return other
        if other.is_infinity():
            return self
        X1, Y1, Z1 = self.X,  self.Y,  self.Z
        X2, Y2, Z2 = other.X, other.Y, other.Z
        Z1Z1 = fp_mul(Z1, Z1)
        Z2Z2 = fp_mul(Z2, Z2)
        U1   = fp_mul(X1, Z2Z2)
        U2   = fp_mul(X2, Z1Z1)
        S1   = fp_mul(Y1, fp_mul(Z2, Z2Z2))
        S2   = fp_mul(Y2, fp_mul(Z1, Z1Z1))
        H    = fp_sub(U2, U1)
        R    = fp_sub(S2, S1)
        if H == 0:
            if R == 0:
                return self.double()
            return G1Projective(1, 1, 0)  # point at infinity
        HH  = fp_mul(H, H)
        HHH = fp_mul(H, HH)
        X3  = fp_sub(fp_mul(R, R), fp_add(HHH, fp_mul(2, fp_mul(U1, HH))))
        Y3  = fp_sub(fp_mul(R, fp_sub(fp_mul(U1, HH), X3)), fp_mul(S1, HHH))
        Z3  = fp_mul(H, fp_mul(Z1, Z2))
        return G1Projective(X3, Y3, Z3)

    def add_affine(self, other: G1Affine) -> "G1Projective":
        """Mixed addition: projective + affine (saves a squaring)."""
        if other.is_infinity():
            return self
        if self.is_infinity():
            return other.to_projective()
        X1, Y1, Z1 = self.X, self.Y, self.Z
        X2, Y2     = other.x, other.y
        Z1Z1 = fp_mul(Z1, Z1)
        U2   = fp_mul(X2, Z1Z1)
        S2   = fp_mul(Y2, fp_mul(Z1, Z1Z1))
        H    = fp_sub(U2, X1)
        R    = fp_sub(S2, Y1)
        if H == 0:
            if R == 0:
                return self.double()
            return G1Projective(1, 1, 0)
        HH  = fp_mul(H, H)
        HHH = fp_mul(H, HH)
        X3  = fp_sub(fp_mul(R, R), fp_add(HHH, fp_mul(2, fp_mul(X1, HH))))
        Y3  = fp_sub(fp_mul(R, fp_sub(fp_mul(X1, HH), X3)), fp_mul(Y1, HHH))
        Z3  = fp_mul(H, Z1)
        return G1Projective(X3, Y3, Z3)

    def neg(self) -> "G1Projective":
        return G1Projective(self.X, fp_neg(self.Y), self.Z)

    def scalar_mul(self, k: int) -> "G1Projective":
        """Double-and-add scalar multiplication."""
        result = G1Projective(1, 1, 0)   # identity
        addend = self
        k = k % r   # reduce by subgroup order
        while k:
            if k & 1:
                result = result.add(addend)
            addend = addend.double()
            k >>= 1
        return result

    def __eq__(self, other) -> bool:
        # Compare in affine
        return self.to_affine() == other.to_affine()

    def __repr__(self):
        return f"G1Proj({self.to_affine()})"


# ── Generator ─────────────────────────────────────────────────────────────────

G1_GENERATOR = G1Affine(G1_GEN_X, G1_GEN_Y)
assert G1_GENERATOR.is_on_curve(), "BLS12-377 G1 generator not on curve — check params.py"
