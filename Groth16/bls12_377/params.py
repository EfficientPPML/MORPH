"""
BLS12-377 curve parameters.

References:
  Zexe paper (BCGMMW 2020): https://eprint.iacr.org/2018/962
  arkworks-rs/curves bls12_377
"""

# ── Scalar field Fr ────────────────────────────────────────────────────────────
# Order of G1 and G2 subgroups.
# r - 1 = 2^47 * t  =>  NTT domain supports sizes up to 2^47
r = 8444461749428370424248824938781546531375899335154063827935233455917409239041

# ── Base field Fp ──────────────────────────────────────────────────────────────
# Coordinate field for G1.  G2 is over Fp2.
p = 258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177

# ── Curve seed ────────────────────────────────────────────────────────────────
# Used as the Miller-loop parameter for the optimal ate pairing.
X_SEED = 0x8508c00000000001  # = 9586122913090633729  (positive)

# ── G1 curve: y² = x³ + 1 over Fp ────────────────────────────────────────────
G1_B = 1

G1_GEN_X = 0x008848defe740a67c8fc6225bf87ff5485951e2caa9d41bb188282c8bd37cb5cd5481512ffcd394eeab9b16eb21be9ef
G1_GEN_Y = 0x01914a69c5102eff1f674f5d30afeec4bd7fb348ca3e52d96d182ad44fb82305c2fe3d3634a9591afd82de55559c8ea6

# ── Fp2 tower: Fp2 = Fp[u] / (u² + 5) ───────────────────────────────────────
# Non-residue α = -5  (u² = -5)
FP2_NONRES = p - 5   # = -5 mod p

# ── G2 curve: y² = x³ + (0,1) over Fp2 ──────────────────────────────────────
# D-type sextic twist of G1 with twist factor w where w^6 = u.
# Derivation: the twist maps (x,y) → (x·w⁻², y·w⁻³); substituting into
# y² = x³ + 1 gives the twisted curve y² = x³ + u  i.e. B_G2 = (0, 1).
G2_B = (0, 155198655607781456406391640216936120121836107652948796323930557600032281009004493664981332883744016074664192874906)  # Fp2 element (c0, c1) = u^{-1}

# G2 generator (affine, over Fp2). Each coordinate is an Fp2 element (c0, c1).
G2_GEN_X = (
    233578398248691099356572568220835526895379068987715365179118596935057653620464273615301663571204657964920925606294,
    140913150380207355837477652521042157274541796891053068589147167627541651775299824604154852141315666357241556069118,
)
G2_GEN_Y = (
    63160294768292073209381361943935198908131692476676907196754037919244929611450776219210369229519898517858833747423,
    149157405641012693445398062341192467754805999074082136895788947234480009303640899064710353187729182149407503257491,
)

# ── Frobenius precomputed constants ───────────────────────────────────────────
#
# Tower:  Fp2 = Fp[u]/(u²+5)
#         Fp6 = Fp2[v]/(v³−u)
#         Fp12 = Fp6[w]/(w²−v)
#
# Key Frobenius facts (all exponents computed mod p):
#   Frob(u)    = u^p  = −u          (u is a QNR, so u^p = −u)
#   Frob(v)    = v^p  = γ₁ · v      where γ₁ = 5^((p−1)/6) mod p  ∈ Fp
#   Frob(w)    = w^p  = δ₁ · w      where δ₁ = 5^((p−1)/12) mod p ∈ Fp
#
# Note: (−5)^((p−1)/6) = 5^((p−1)/6) because (p−1)/6 is even for BLS12-377.
# Note: δ₁² = γ₁  (verified: 5^((p−1)/6) = (5^((p−1)/12))²)
#
# Frobenius coefficients for Fp6 at power k:
#   FRB6_V[k]  = γ₁^k  mod p   (coefficient of v  under Frob^k)
#   FRB6_V2[k] = γ₁^(2k) mod p (coefficient of v² under Frob^k)
#
# Frobenius coefficients for Fp12 at power k:
#   FRB12_W[k] = δ₁^k mod p    (coefficient of w  under Frob^k)

GAMMA1  = pow(5, (p - 1) // 6,  p)   # γ₁
DELTA1  = pow(5, (p - 1) // 12, p)   # δ₁  (square root of γ₁ in Fp)

# Precomputed tables (indices 0..5 for Fp6, 0..11 for Fp12)
FRB6_V  = [pow(GAMMA1, k, p) for k in range(6)]
FRB6_V2 = [pow(GAMMA1, 2 * k, p) for k in range(6)]
FRB12_W = [pow(DELTA1, k, p) for k in range(12)]

# ── Miller-loop bit decomposition of X_SEED ──────────────────────────────────
# Precompute bits of X_SEED for the ate pairing Miller loop.
# Format: list of (0|1) from bit (len-2) down to bit 0, skipping leading 1.
def _bits_of(n: int):
    bits = []
    while n > 1:
        bits.append(n & 1)
        n >>= 1
    return list(reversed(bits))   # MSB-1 … LSB

X_SEED_BITS = _bits_of(X_SEED)  # 63 bits for X_SEED = 0x8508c00000000001
