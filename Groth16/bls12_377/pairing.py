"""
Optimal Ate Pairing for BLS12-377.

    ate_pairing(P: G1Affine, Q: G2Affine) -> Fp12

Implementation
--------------
1. Miller loop over the bits of X_SEED = 0x8508c00000000001
   using sparse line evaluations (D-type twist, affine G2 coordinates).
2. Final exponentiation:
   - Easy part:  f^(p^6−1) · f^(p^2+1)     (Frobenius + inverse)
   - Hard part:  f^((p^4−p^2+1)/r)          (naive square-and-multiply)

Line evaluation (D-twist, multiplication convention, P ∈ G1, T ∈ G2):
   The untwist map ψ(X,Y) = (X·v, Y·v·w) embeds G2 into E(Fp12).
   The tangent/chord line at ψ(T), evaluated at P = (xP, yP), gives:

       l(P) = yP  +  (λ·X_T − Y_T) · v · w
                  +  (−λ · xP)      · w

   In Fp12 = (A, B·w):
       A = (yP, 0, 0)       ∈ Fp6
       B = (−λ·xP, m, 0)    ∈ Fp6
   where λ is the slope and m = λ·X_T − Y_T.

   This matches the gnark-crypto "034" sparse pattern.
"""

from .params import p, r, X_SEED_BITS
from .field  import (
    FP12_ONE, FP6_ZERO,
    fp2_add, fp2_sub, fp2_mul, fp2_sq, fp2_neg, fp2_inv, fp2_scale,
    fp6_add, fp6_sub, fp6_mul, fp6_neg, fp6_scale_fp,
    fp12_mul, fp12_sq, fp12_inv, fp12_conj, fp12_frob, fp12_pow,
)
from .g1 import G1Affine
from .g2 import G2Affine


# ── Line evaluation helpers ───────────────────────────────────────────────────

def _line_double(T_x, T_y, xP: int, yP: int):
    """
    Compute the tangent line at T = (T_x, T_y) ∈ G2(Fp2),
    evaluated at P = (xP, yP) ∈ G1(Fp).

    Returns (new_T_x, new_T_y, l) where l ∈ Fp12 is the sparse line element.
    """
    # Slope: λ = 3·X²/(2·Y)  in Fp2
    X2   = fp2_sq(T_x)
    lam  = fp2_mul(fp2_scale(X2, 3), fp2_inv(fp2_scale(T_y, 2)))

    # New point: X' = λ²−2X,  Y' = λ(X−X')−Y
    lam2 = fp2_sq(lam)
    X_new = fp2_sub(lam2, fp2_scale(T_x, 2))
    Y_new = fp2_sub(fp2_mul(lam, fp2_sub(T_x, X_new)), T_y)

    # Line coefficients in Fp12 = (A, B·w)
    # Multiplication convention: l = yP + m·v·w + (−λ·xP)·w
    m     = fp2_sub(fp2_mul(lam, T_x), T_y)           # = λ·X − Y ∈ Fp2
    neg_lxp = fp2_neg(fp2_scale(lam, xP % p))         # −λ·xP ∈ Fp2

    A = ((yP % p, 0), (0, 0), (0, 0))                 # (yP, 0, 0) ∈ Fp6
    B = (neg_lxp, m, (0, 0))                           # (−λ·xP, m, 0) ∈ Fp6

    l = (A, B)   # Fp12 element
    return X_new, Y_new, l


def _line_add(T_x, T_y, Q_x, Q_y, xP: int, yP: int):
    """
    Compute the chord through T and Q (both ∈ G2(Fp2)),
    evaluated at P = (xP, yP) ∈ G1(Fp).

    Returns (new_T_x, new_T_y, l).
    """
    # Slope: λ = (Y_Q − Y_T)/(X_Q − X_T)  in Fp2
    lam  = fp2_mul(fp2_sub(Q_y, T_y), fp2_inv(fp2_sub(Q_x, T_x)))

    lam2  = fp2_sq(lam)
    X_new = fp2_sub(fp2_sub(lam2, T_x), Q_x)
    Y_new = fp2_sub(fp2_mul(lam, fp2_sub(T_x, X_new)), T_y)

    m     = fp2_sub(fp2_mul(lam, T_x), T_y)
    neg_lxp = fp2_neg(fp2_scale(lam, xP % p))

    A = ((yP % p, 0), (0, 0), (0, 0))
    B = (neg_lxp, m, (0, 0))

    l = (A, B)
    return X_new, Y_new, l


# ── Miller loop ───────────────────────────────────────────────────────────────

def miller_loop(P: G1Affine, Q: G2Affine):
    """
    Compute the Miller function f_{X_SEED, Q}(P) in Fp12.
    Uses the optimal ate pairing loop parameter X_SEED for BLS12-377.
    """
    if P.is_infinity() or Q.is_infinity():
        return FP12_ONE

    xP, yP = P.x, P.y
    Q_x, Q_y = Q.x, Q.y     # affine G2 coordinates, Fp2 elements

    T_x, T_y = Q_x, Q_y     # running G2 point (affine)
    f = FP12_ONE

    for bit in X_SEED_BITS:  # bits of X_SEED from bit 62 down to bit 0
        # ── Doubling step ──
        f = fp12_sq(f)
        T_x, T_y, l_dbl = _line_double(T_x, T_y, xP, yP)
        f = fp12_mul(f, l_dbl)

        # ── Addition step (if bit == 1) ──
        if bit == 1:
            T_x, T_y, l_add = _line_add(T_x, T_y, Q_x, Q_y, xP, yP)
            f = fp12_mul(f, l_add)

    # X_SEED is positive, no conjugation needed
    return f


# ── Final exponentiation ──────────────────────────────────────────────────────

def _easy_part(f):
    """
    Raise f to (p^6−1)(p^2+1).

    f^(p^6−1) = conj(f) · f⁻¹         (Frobenius^6 = conjugation)
    f^(p^2+1) = Frob^2(f) · f
    """
    # f^(p^6−1)
    f1 = fp12_mul(fp12_conj(f), fp12_inv(f))
    # f^(p^6−1)(p^2+1)
    f2 = fp12_mul(fp12_frob(f1, 2), f1)
    return f2


_HARD_EXP = (p**4 - p**2 + 1) // r          # module-level constant, was recomputed


def _hard_part(f):
    """
    Raise f to (p^4−p^2+1)/r via square-and-multiply in Fp12.
    f is assumed to be in the cyclotomic subgroup after the easy part.
    """
    return fp12_pow(f, _HARD_EXP)


def final_exp(f):
    """Full final exponentiation: easy part then hard part."""
    return _hard_part(_easy_part(f))


# ── Multi-pairing (interleaved Miller loop) ──────────────────────────────────

def multi_miller_loop(pairs):
    """Product of Miller loops for a list of ``(P, Q)`` pairs, sharing the
    one ``fp12_sq(f)`` per loop iteration across all pairs.

    For ``N`` pairs, this saves ``(N - 1)`` squarings per Miller-loop step
    compared with computing the loops independently — i.e. roughly
    ``62 · (N − 1)`` Fp12 squarings for the 62-bit ate seed.  The combined
    Fp12 element is returned BEFORE final exponentiation; the caller is
    expected to apply :func:`final_exp` once on the full product (which
    saves ``N − 1`` final-exponentiations, the dominant cost).

    Identity pairs (``P`` or ``Q`` at infinity) contribute the multiplicative
    identity and are silently dropped.

    Returns ``FP12_ONE`` if the list is empty after filtering identities.
    """
    # Filter identities — they contribute the multiplicative identity.
    active = [(P, Q) for (P, Q) in pairs
              if not P.is_infinity() and not Q.is_infinity()]
    if not active:
        return FP12_ONE

    # Initialise running G2 points (one per active pair).
    Ts = [(Q.x, Q.y) for (_, Q) in active]
    f = FP12_ONE

    for bit in X_SEED_BITS:
        # ── Shared doubling-of-accumulator step ──
        f = fp12_sq(f)

        # ── Per-pair tangent line + multiply into f ──
        for i, (P, Q) in enumerate(active):
            T_x, T_y = Ts[i]
            T_x, T_y, l_dbl = _line_double(T_x, T_y, P.x, P.y)
            f = fp12_mul(f, l_dbl)
            Ts[i] = (T_x, T_y)

        # ── Per-pair addition step (bit == 1) ──
        if bit == 1:
            for i, (P, Q) in enumerate(active):
                T_x, T_y = Ts[i]
                T_x, T_y, l_add = _line_add(T_x, T_y, Q.x, Q.y, P.x, P.y)
                f = fp12_mul(f, l_add)
                Ts[i] = (T_x, T_y)

    return f


# ── Public API ────────────────────────────────────────────────────────────────

def ate_pairing(P: G1Affine, Q: G2Affine):
    """
    Compute the optimal ate pairing e(P, Q) ∈ Fp12 (GT subgroup).

    e: G1 × G2 → GT
    """
    return final_exp(miller_loop(P, Q))


def multi_pairing(pairs):
    """Compute ``∏ e(P_i, Q_i)`` for a list of ``(G1Affine, G2Affine)``
    pairs, using a single shared final exponentiation.

    For ``N`` pairs this is roughly ``N×`` faster than computing
    ``ate_pairing(P_i, Q_i)`` in a loop and multiplying — the final
    exponentiation is the dominant cost (~70% of one pairing) and is
    paid once here.
    """
    return final_exp(multi_miller_loop(pairs))
