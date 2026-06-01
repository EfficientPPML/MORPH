"""
BLS12-377 field tower arithmetic.

Hierarchy:
    Fp   — base field,  modulus p
    Fr   — scalar field, modulus r  (NTT / witness domain)
    Fp2  = Fp[u] / (u² + 5)
    Fp6  = Fp2[v] / (v³ − u)
    Fp12 = Fp6[w] / (w² − v)

All elements are plain Python ints (for Fp/Fr) or tuples of ints (for towers).
No classes — just module-level functions for clarity and speed.

Fp2  element: (c0, c1)           where value = c0 + c1·u
Fp6  element: (c0, c1, c2)       where value = c0 + c1·v + c2·v²  (ci ∈ Fp2)
Fp12 element: (c0, c1)           where value = c0 + c1·w           (ci ∈ Fp6)
"""

from .params import p, r, FRB6_V, FRB6_V2, FRB12_W

# ══════════════════════════════════════════════════════════════════════════════
# Fp  (base field, modulus p)
# ══════════════════════════════════════════════════════════════════════════════

def fp_add(a: int, b: int) -> int:
    return (a + b) % p

def fp_sub(a: int, b: int) -> int:
    return (a - b) % p

def fp_mul(a: int, b: int) -> int:
    return (a * b) % p

def fp_neg(a: int) -> int:
    return (-a) % p

def fp_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("fp_inv(0)")
    return pow(a, p - 2, p)

def fp_div(a: int, b: int) -> int:
    return fp_mul(a, fp_inv(b))

def fp_pow(a: int, e: int) -> int:
    return pow(a, e, p)


# ══════════════════════════════════════════════════════════════════════════════
# Fr  (scalar field, modulus r)
# ══════════════════════════════════════════════════════════════════════════════

def fr_add(a: int, b: int) -> int:
    return (a + b) % r

def fr_sub(a: int, b: int) -> int:
    return (a - b) % r

def fr_mul(a: int, b: int) -> int:
    return (a * b) % r

def fr_neg(a: int) -> int:
    return (-a) % r

def fr_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("fr_inv(0)")
    return pow(a, r - 2, r)

def fr_div(a: int, b: int) -> int:
    return fr_mul(a, fr_inv(b))

def fr_pow(a: int, e: int) -> int:
    return pow(a, e % (r - 1), r)


# ══════════════════════════════════════════════════════════════════════════════
# Fp2 = Fp[u] / (u² + 5)
#   element: (c0, c1) where value = c0 + c1·u,  u² = −5
# ══════════════════════════════════════════════════════════════════════════════

FP2_ZERO = (0, 0)
FP2_ONE  = (1, 0)

def fp2_add(a, b):
    return (fp_add(a[0], b[0]), fp_add(a[1], b[1]))

def fp2_sub(a, b):
    return (fp_sub(a[0], b[0]), fp_sub(a[1], b[1]))

def fp2_neg(a):
    return (fp_neg(a[0]), fp_neg(a[1]))

def fp2_mul(a, b):
    # (a0 + a1·u)(b0 + b1·u) = (a0·b0 − 5·a1·b1) + (a0·b1 + a1·b0)·u
    a0, a1 = a;  b0, b1 = b
    return (
        fp_sub(fp_mul(a0, b0), fp_mul(5, fp_mul(a1, b1))),
        fp_add(fp_mul(a0, b1), fp_mul(a1, b0)),
    )

def fp2_scale(a, s: int):
    """Multiply Fp2 element a by Fp scalar s."""
    return (fp_mul(a[0], s), fp_mul(a[1], s))

def fp2_conj(a):
    """Frobenius on Fp2: (a0 + a1·u) → (a0 − a1·u)."""
    return (a[0], fp_neg(a[1]))

def fp2_sq(a):
    # (a0 + a1·u)² = (a0²−5·a1²) + (2·a0·a1)·u
    a0, a1 = a
    return (
        fp_sub(fp_mul(a0, a0), fp_mul(5, fp_mul(a1, a1))),
        fp_mul(2, fp_mul(a0, a1)),
    )

def fp2_inv(a):
    # 1/(a0+a1·u) = (a0−a1·u) / (a0²+5·a1²)
    a0, a1 = a
    denom = fp_inv(fp_add(fp_mul(a0, a0), fp_mul(5, fp_mul(a1, a1))))
    return (fp_mul(a0, denom), fp_mul(fp_neg(a1), denom))

def fp2_div(a, b):
    return fp2_mul(a, fp2_inv(b))

def fp2_pow(a, e: int):
    result = FP2_ONE
    base   = (a[0] % p, a[1] % p)
    e = e % (p * p - 1)   # order of Fp2*
    while e:
        if e & 1:
            result = fp2_mul(result, base)
        base = fp2_sq(base)
        e >>= 1
    return result

def fp2_frob(a, k: int):
    """Frobenius^k on Fp2: identity if k even, conjugate if k odd."""
    return a if k % 2 == 0 else fp2_conj(a)

def fp2_eq(a, b) -> bool:
    return a[0] % p == b[0] % p and a[1] % p == b[1] % p

# u⁻¹ in Fp2: since u·u⁻¹=1 and u=(0,1), we get u⁻¹=(0,−1/5)
FP2_U_INV = (0, fp_neg(fp_inv(5)))   # = (0, −5⁻¹ mod p)


# ══════════════════════════════════════════════════════════════════════════════
# Fp6 = Fp2[v] / (v³ − u)
#   element: (c0, c1, c2) where value = c0 + c1·v + c2·v²,  v³ = u
# ══════════════════════════════════════════════════════════════════════════════

FP6_ZERO = (FP2_ZERO, FP2_ZERO, FP2_ZERO)
FP6_ONE  = (FP2_ONE,  FP2_ZERO, FP2_ZERO)

def fp6_add(a, b):
    return (fp2_add(a[0], b[0]), fp2_add(a[1], b[1]), fp2_add(a[2], b[2]))

def fp6_sub(a, b):
    return (fp2_sub(a[0], b[0]), fp2_sub(a[1], b[1]), fp2_sub(a[2], b[2]))

def fp6_neg(a):
    return (fp2_neg(a[0]), fp2_neg(a[1]), fp2_neg(a[2]))

def _fp6_mul_by_nonres(x):
    """Multiply Fp6 element by v (the Fp6 generator).
       v·(c0 + c1·v + c2·v²) = c2·v³ + c0·v + c1·v² = u·c2 + c0·v + c1·v²
       where v³=u means multiplying c2 by u in Fp2.
    """
    # u·c2 in Fp2: u·(c2_0 + c2_1·u) = c2_0·u + c2_1·u² = −5·c2_1 + c2_0·u
    c2 = x[2]
    uc2 = (fp_mul(p - 5, c2[1]) % p, c2[0])  # = (−5·c2_1, c2_0)
    return (uc2, x[0], x[1])

def fp6_mul(a, b):
    a0, a1, a2 = a
    b0, b1, b2 = b
    # Karatsuba-style:
    t0 = fp2_mul(a0, b0)
    t1 = fp2_mul(a1, b1)
    t2 = fp2_mul(a2, b2)

    c0 = fp2_add(t0, _fp2_mul_u(fp2_add(fp2_mul(fp2_add(a1, a2), fp2_add(b1, b2)), fp2_neg(fp2_add(t1, t2)))))
    c1 = fp2_add(fp2_mul(fp2_add(a0, a1), fp2_add(b0, b1)), fp2_neg(fp2_add(t0, t1)))
    c1 = fp2_add(c1, _fp2_mul_u(t2))
    c2 = fp2_add(fp2_mul(fp2_add(a0, a2), fp2_add(b0, b2)), fp2_neg(fp2_add(t0, t2)))
    c2 = fp2_add(c2, t1)
    return (c0, c1, c2)

def fp6_sq(a):
    return fp6_mul(a, a)

def fp6_scale(a, s):
    """Scale Fp6 by Fp2 element s."""
    return (fp2_mul(a[0], s), fp2_mul(a[1], s), fp2_mul(a[2], s))

def fp6_scale_fp(a, s: int):
    """Scale Fp6 by Fp scalar s."""
    return (fp2_scale(a[0], s), fp2_scale(a[1], s), fp2_scale(a[2], s))

def fp6_inv(a):
    a0, a1, a2 = a
    # Inversion formula for Fp6 = Fp2[v]/(v³−u):
    t0 = fp2_sub(fp2_sq(a0), _fp2_mul_u(fp2_mul(a1, a2)))
    t1 = fp2_sub(_fp2_mul_u(fp2_sq(a2)), fp2_mul(a0, a1))
    t2 = fp2_sub(fp2_sq(a1), fp2_mul(a0, a2))
    factor = fp2_inv(fp2_add(fp2_mul(a0, t0), _fp2_mul_u(fp2_add(fp2_mul(a2, t1), fp2_mul(a1, t2)))))
    return (fp2_mul(t0, factor), fp2_mul(t1, factor), fp2_mul(t2, factor))

def _fp2_mul_u(x):
    """Multiply Fp2 element by u (the abstract generator, u²=−5).
       u·(c0 + c1·u) = c0·u + c1·u² = −5·c1 + c0·u
    """
    return (fp_mul(p - 5, x[1]) % p, x[0])

def fp6_frob(a, k: int):
    """Frobenius^k on Fp6.
       Frob^k(c0 + c1·v + c2·v²)
         = frob_fp2^k(c0)
         + frob_fp2^k(c1) · FRB6_V[k]  · v
         + frob_fp2^k(c2) · FRB6_V2[k] · v²
    """
    k6 = k % 6
    fc0 = fp2_frob(a[0], k)
    fc1 = fp2_scale(fp2_frob(a[1], k), FRB6_V[k6])
    fc2 = fp2_scale(fp2_frob(a[2], k), FRB6_V2[k6])
    return (fc0, fc1, fc2)

def fp6_mul_by_v(a):
    """Multiply Fp6 element by v (shift coefficients, wrap with non-residue)."""
    return _fp6_mul_by_nonres(a)


# ══════════════════════════════════════════════════════════════════════════════
# Fp12 = Fp6[w] / (w² − v)
#   element: (c0, c1) where value = c0 + c1·w,  w² = v  (v ∈ Fp6)
# ══════════════════════════════════════════════════════════════════════════════

FP12_ZERO = (FP6_ZERO, FP6_ZERO)
FP12_ONE  = (FP6_ONE,  FP6_ZERO)

def fp12_add(a, b):
    return (fp6_add(a[0], b[0]), fp6_add(a[1], b[1]))

def fp12_sub(a, b):
    return (fp6_sub(a[0], b[0]), fp6_sub(a[1], b[1]))

def fp12_neg(a):
    return (fp6_neg(a[0]), fp6_neg(a[1]))

def fp12_mul(a, b):
    # (A0 + A1·w)(B0 + B1·w) = (A0·B0 + v·A1·B1) + (A0·B1 + A1·B0)·w
    # where w² = v, so v·X in Fp6 means fp6_mul_by_v(X).
    A0, A1 = a;  B0, B1 = b
    t0 = fp6_mul(A0, B0)
    t1 = fp6_mul(A1, B1)
    c0 = fp6_add(t0, fp6_mul_by_v(t1))
    c1 = fp6_add(fp6_mul(A0, B1), fp6_mul(A1, B0))
    return (c0, c1)

def fp12_sq(a):
    # Squaring: (A0+A1·w)² = (A0²+v·A1²) + 2·A0·A1·w
    A0, A1 = a
    t0 = fp6_sq(A0)
    t1 = fp6_sq(A1)
    c0 = fp6_add(t0, fp6_mul_by_v(t1))
    c1 = fp6_mul(fp6_add(A0, A1), fp6_add(A0, A1))
    c1 = fp6_sub(fp6_sub(c1, t0), t1)
    return (c0, c1)

def fp12_inv(a):
    A0, A1 = a
    # (A0+A1·w)⁻¹ = (A0−A1·w) / (A0²−v·A1²)
    denom = fp6_inv(fp6_sub(fp6_sq(A0), fp6_mul_by_v(fp6_sq(A1))))
    return (fp6_mul(A0, denom), fp6_neg(fp6_mul(A1, denom)))

def fp12_conj(a):
    """Frobenius^6 on Fp12: (A0 + A1·w) → (A0 − A1·w)."""
    return (a[0], fp6_neg(a[1]))

def fp12_frob(a, k: int):
    """Frobenius^k on Fp12.
       Frob^k(A0 + A1·w)
         = frob_fp6^k(A0) + frob_fp6^k(A1) · FRB12_W[k] · w
    """
    k12 = k % 12
    fA0 = fp6_frob(a[0], k)
    fA1 = fp6_scale_fp(fp6_frob(a[1], k), FRB12_W[k12])
    return (fA0, fA1)

def fp12_pow(a, e: int):
    """Square-and-multiply exponentiation in Fp12."""
    result = FP12_ONE
    base   = a
    while e:
        if e & 1:
            result = fp12_mul(result, base)
        base = fp12_sq(base)
        e >>= 1
    return result

def fp12_eq(a, b) -> bool:
    def fp6_eq(x, y):
        return fp2_eq(x[0], y[0]) and fp2_eq(x[1], y[1]) and fp2_eq(x[2], y[2])
    return fp6_eq(a[0], b[0]) and fp6_eq(a[1], b[1])
