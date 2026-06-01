"""
Cached factories for the TPU NTT and MSM contexts.

Each unique configuration is built and JIT-compiled once per process.
Configs are deliberately small/correctness-focused: BLS12-377 Fr for NTT,
BLS12-377 Fp (Twisted Edwards form) for MSM, no sharding.
"""

import functools

import jax

# path_setup must be imported by the application before this module is
# used; it adds the right paths so the imports below resolve.
import number_theory_transform_context as _ntt_ctx_mod  # noqa: E402
import multiscalar_multiplication_context as _msm_ctx_mod  # noqa: E402
import elliptic_curve_context as _ec_ctx_mod  # noqa: E402
import finite_field_context as _ff_ctx_mod  # noqa: E402
import utils as _utils  # noqa: E402

from adapters.ntt_params import psi_for_ntt_size

# ── BLS12-377 parameters lifted from multiscalar_multiplication_perf_test.py
# These describe the Twisted Edwards model used by the TPU MSM stack.
_BLS12_377_FP_PRIME_TWIST = (
    0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
)
_FR_BLS = 8444461749428370424248824938781546531375899335154063827935233455917409239041

# Curve params for ExtendedTwistedEdwardsNDContext.
_EC_PARAMS = {
    "prime": 258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177,
    "order": _FR_BLS,
    "a": -1,
    "twist_d": 122268283598675559488486339158635529096981886914877139579534153582033676785385790730042363341236035746924960903179,
    "alpha": -1,
    "b": 1,
    "s": 10189023633222963290707194929886294091415157242906428298294512798502806398782149227503530278436336312243746741931,
    "MA": 228097355113300204138531148905234651262148041026195375645000724271212049151994375092458297304264351187709081232384,
    "MB": 10189023633222963290707194929886294091415157242906428298294512798502806398782149227503530278436336312243746741931,
    "t": 23560188534917577818843641916571445935985386319233886518929971599490231428764380923487987729215299304184915158756,
    "generator": [
        71222569531709137229370268896323705690285216175189308202338047559628438110820800641278662592954630774340654489393,
        6177051365529633638563236407038680211609544222665285371549726196884440490905471891908272386851767077598415378235,
    ],
}

# DRNS sizing.
NUM_MODULI_NTT = 24        # for the 253-bit Fr.  21 (the perf-test 256-bit
                            # preset) is borderline — at small NTT sizes (n≤16)
                            # we observed a sporadic single-position error with
                            # 21 moduli that disappears with extra headroom.
                            # 22 also works; 24 keeps a safe margin.
NUM_MODULI_MSM = 32        # for the 377-bit Fp (matches perf-test preset)
PRECISION_BITS = 28
RADIX_BITS = 32

# Default MSM scalar bit-width — matches BLS12-377 Fr (~253 bits).
MSM_SCALAR_BITS = 253


# ──────────────────────────────────────────────────────────────────────
# NTT factory
# ──────────────────────────────────────────────────────────────────────


def _factor_n_for_ntt3step(n: int) -> tuple[int, int]:
    """Return ``(r, c)`` such that ``r * c == n`` and both are powers of 2,
    biased toward a square split.  ``n`` itself must be a power of two."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a positive power of two, got {n}")
    # Pick r = 2^ceil(log2(n)/2), c = n // r.  For n=16 → r=4, c=4.
    log_n = n.bit_length() - 1
    r = 1 << ((log_n + 1) // 2)
    c = n // r
    return r, c


@functools.cache
def get_ntt_context(n: int):
    """Return a JIT-compiled NTT3Step context for transform length ``n``
    over ``Fr``.  The returned object exposes:

      * ``.ntt(arr)``           — JIT-wrapped, expects ``(1, r, c, M)`` uint32.
      * ``.intt(arr)``          — JIT-wrapped, expects same shape.
      * ``.to_computational_format(plain_int_list)`` → ``(n, M)`` uint32.
      * ``.to_original_format(arr)`` → list of plain ints.
      * ``.r``, ``.c``, ``.n``, ``.psi``, ``.omega`` — config metadata.
    """
    r_dim, c_dim = _factor_n_for_ntt3step(n)
    rns_moduli = _utils.find_moduli_specified_number(NUM_MODULI_NTT, PRECISION_BITS)
    ff_ctx = _ntt_ctx_mod.DRNSLazyExtensionContext({
        "prime": _FR_BLS,
        "rns_moduli": rns_moduli,
        "precision_bits": PRECISION_BITS,
        "radix_bits": RADIX_BITS,
    })
    psi = psi_for_ntt_size(n)
    raw = _ntt_ctx_mod.NTT3Step({
        "prime": _FR_BLS,
        "r": r_dim,
        "c": c_dim,
        "psi": psi,
        "finite_field_context": ff_ctx,
    })

    class _CompiledNTT:
        """Caches jit-wrapped ntt / intt and exposes input/output shaping."""

        def __init__(self, raw_ctx):
            self._raw = raw_ctx
            self.r = raw_ctx.r
            self.c = raw_ctx.c
            self.n = raw_ctx.r * raw_ctx.c
            self.psi = raw_ctx.psi
            self.omega = raw_ctx.omega
            self.num_moduli = NUM_MODULI_NTT
            self._ntt_jit = jax.jit(raw_ctx.ntt)
            self._intt_jit = jax.jit(raw_ctx.intt)

        def to_computational_format(self, a):
            """Plain int list (length n) → DRNS array shaped (1, r, c, M)."""
            arr = self._raw.to_computational_format(a)
            # arr.shape is (n, M); reshape to (1, r, c, M) for the NTT.
            return arr.reshape(1, self.r, self.c, -1)

        def to_original_format(self, arr):
            """DRNS array (1, r, c, M) → plain int list of length n."""
            flat = arr.reshape(-1, arr.shape[-1])
            decoded = self._raw.to_original_format(flat)
            # ff_ctx.to_original_format returns a flat list for a 2-D array.
            return list(decoded)

        def ntt(self, arr):
            return self._ntt_jit(arr)

        def intt(self, arr):
            return self._intt_jit(arr)

    return _CompiledNTT(raw)


# ──────────────────────────────────────────────────────────────────────
# MSM factory
# ──────────────────────────────────────────────────────────────────────


def _slice_bits_for(msm_length: int, max_slice_bits: int = 15) -> int:
    """Choose ``slice_bits`` so that ``2^slice_bits <= tile_length`` and
    we still get a manageable number of windows.

    NOTE: very small slice_bits (≤ 4) combined with very small msm_length
    (≤ 16) trip a numerical bug in the FusionMSMContext bucketization
    (bucket_num_last_window = 2 + duplication path).  We sidestep this by
    requiring slice_bits ≥ 8 in practice — the adapter pads msm inputs
    up to whichever ``msm_length`` makes that fit."""
    if msm_length <= 0 or (msm_length & (msm_length - 1)) != 0:
        raise ValueError(f"msm_length must be a positive power of two, got {msm_length}")
    log_n = msm_length.bit_length() - 1
    return min(max_slice_bits, log_n)


@functools.cache
def get_msm_context(msm_length: int, scalar_bits: int = MSM_SCALAR_BITS):
    """Return a JIT-compiled FusionMSMContext for the given length.

    Compiles with ``use_fused=True`` and ``use_sharding=False``.
    The returned context still exposes the upstream ``multiscalar_multiply`` API.
    """
    rns_moduli = _utils.find_moduli_specified_number(NUM_MODULI_MSM, PRECISION_BITS)
    ec_parameters = {
        "finite_field_context_class": _ff_ctx_mod.DRNSlazyContext,
        "finite_field_parameters": {
            "prime": _BLS12_377_FP_PRIME_TWIST,
            "rns_moduli": rns_moduli,
            "precision_bits": PRECISION_BITS,
            "radix_bits": RADIX_BITS,
        },
        **_EC_PARAMS,
    }
    msm_parameters = {
        "elliptic_curve_context_class": _ec_ctx_mod.ExtendedTwistedEdwardsNDContext,
        "elliptic_curve_parameters": ec_parameters,
        "coordinate_dim": 4,
        "msm_length": msm_length,
        "tile_length": msm_length,
        "slice_bits": _slice_bits_for(msm_length),
        "scalar_bits": scalar_bits,
        "order": ec_parameters["order"],
        # Per-bucket slot capacity = ceil(tile_length / bucket_num_per_window) *
        # ret_space_ratio.  For msm_length=1024/slice_bits=10 the expected
        # bucket occupancy is 1.  Pathological witnesses (many wires
        # sharing the same scalar value) stuff multiple positions into
        # the same bucket and silently overflow.  Bumped 2 → 8 (T6/T6b
        # — "four wires sharing scalar=1").  T33: sudoku originally
        # had 22 wires with scalar=1 (bit decomposition).  Rather than
        # bumping the ratio higher (which costs HBM and compile time),
        # we restructured sudoku to use (v-1)(v-2)(v-3)(v-4)=0 range
        # checks, dropping the count to 6 — fits comfortably at 8.
        # Memory cost at 8: ~26 MB; fine on TPU HBM.
        "c_kernel_ret_space_ratio": 8,
    }
    ctx = _msm_ctx_mod.FusionMSMContext(msm_parameters)
    ctx.set_use_sharding(False)
    ctx.set_use_compiled_kernels(True)
    ctx.compile(parameters={"use_fused": True})
    return ctx


# ──────────────────────────────────────────────────────────────────────
# G2 MSM factory  (XYZZ-Weierstrass over Fp2)
# ──────────────────────────────────────────────────────────────────────


# u² + 5 = 0 in BLS12-377's Fp2 tower.
_QNR_BLS12_377 = 5

# G2 curve y² = x³ + B over Fp2 (B = u/anything; matches Groth16/bls12_377/params.py).
_G2_B = (
    0,
    155198655607781456406391640216936120121836107652948796323930557600032281009004493664981332883744016074664192874906,
)


@functools.cache
def get_g2_msm_context(msm_length: int = 1024,
                       scalar_bits: int = MSM_SCALAR_BITS):
  """Return an Fq2 Pippenger MSM context for BLS12-377 G2.

  Backend is :class:`ProjectiveCompleteFq2NDContext` — projective short-
  Weierstrass with RCB-2015 complete add/double.  G2 has no rational
  2-torsion in Fp2 so the ETE form isn't constructible, but RCB's
  unified formulas are identity-safe under every input pair.  The
  driver and EC kernels are JIT-compiled on first use; the cache key
  is ``(msm_length, scalar_bits)``.
  """
  rns_moduli = _utils.find_moduli_specified_number(NUM_MODULI_MSM, PRECISION_BITS)
  ff_parameters = {
      "prime": _BLS12_377_FP_PRIME_TWIST,
      "rns_moduli": rns_moduli,
      "precision_bits": PRECISION_BITS,
      "radix_bits": RADIX_BITS,
  }
  ec_parameters = {
      "prime": _BLS12_377_FP_PRIME_TWIST,
      "order": _FR_BLS,
      "quadratic_non_residue": _QNR_BLS12_377,
      "finite_field_context_class": _ff_ctx_mod.DRNSlazyContext,
      "finite_field_parameters": ff_parameters,
      "field_extension_parameters": {
          "prime": _BLS12_377_FP_PRIME_TWIST,
          "quadratic_non_residue": _QNR_BLS12_377,
          "finite_field_context_class": _ff_ctx_mod.DRNSlazyContext,
          "finite_field_parameters": ff_parameters,
      },
      "a": 0,
      "b": _G2_B,
  }
  msm_parameters = {
      "elliptic_curve_context_class": _ec_ctx_mod.ProjectiveCompleteFq2NDContext,
      "elliptic_curve_parameters": ec_parameters,
      "coordinate_dim": 3,
      "msm_length": msm_length,
      "tile_length": msm_length,
      # G2 deliberately uses slice_bits=8 instead of log2(msm_length)=10.
      # T20 sweep — slice_bits=8 was Pareto optimal across compile and
      # warm-prove time:
      #   sbits=10: first 1393s, warm 0.83s  (baseline)
      #   sbits=8 : first  608s, warm 0.69s  ← shipped
      #   sbits=6 : first  476s, warm 0.93s  (faster compile, slower warm)
      # Shrinking the bucket tensor from 20 MB → 6.5 MB makes XLA's layout
      # / memory passes much cheaper without hurting steady-state.
      "slice_bits": min(8, _slice_bits_for(msm_length)),
      "scalar_bits": scalar_bits,
      "order": _FR_BLS,
      "c_kernel_ret_space_ratio": 8,
  }
  ctx = _msm_ctx_mod.Fq2FusionMSMContext(msm_parameters)
  ctx.compile(parameters={"use_fused": True})
  return ctx


def reset_caches():
    """Clear all factory caches (useful for tests)."""
    get_ntt_context.cache_clear()
    get_msm_context.cache_clear()
    get_g2_msm_context.cache_clear()
