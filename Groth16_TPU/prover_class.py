"""TPU Groth16 prover — phases 2-4 of the demo pipeline.

The five-phase Groth16 demo pipeline splits responsibilities like this:

    # Phase 1 — circuit setup (CPU + TPU encode, fully cacheable).
    setup = compile_circuit_cached(Circuit(r1cs, max_size), "setup.pkl")

    # Phase 2 — TPU kernel setup (this class' __init__).
    prover = Prover(setup)

    # Phase 3 — per-witness encode (this class' encode_witness method).
    encoded = prover.encode_witness(witness, seed=None)

    # Phase 4 — per-witness prove (this class' prove method).
    proof = prover.prove(encoded)

    # Phase 5 — verify (CPU pairings, on the Setup).
    ok = setup.verify(setup.public_inputs(witness), proof)

This module owns phases 2-4.  Phase 1 lives in :mod:`framework`;
phase 5 is :meth:`framework.Setup.verify`.

Per-instance ownership
----------------------
Every TPU context the prove path needs lives on the instance:

  * ``self.msm_ctx_g1``  — Fused G1 Pippenger MSM context.
  * ``self.msm_ctx_g2``  — Fq2 fused MSM context (RCB-2015 projective).
  * ``self.ntt_ctx``     — n-point NTT3Step context.
  * ``self.ff``          — Fr DRNS context paired with the NTT.
  * ``self.ec_ctx_g1``   — Edwards 4-coord Fq1 EC context.
  * ``self.ec_ctx_g2``   — Projective Fq2 EC context.
  * ``self.setup``       — the bound :class:`framework.Setup`.
  * ``self.pk_tpu``, ``self.qap_tpu``, ``self.n``, ``self.num_wires``,
    ``self.num_public`` — convenience aliases into ``self.setup.*``.

The TPU MSM/NTT/EC contexts come from cached singleton factories
(``tpu_contexts.get_*_context``) so multiple ``Prover`` instances
share the same underlying compiled-kernel state.
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import path_setup  # noqa: F401

import random
import secrets
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TYPE_CHECKING

import jax
import jax.numpy as jnp

from bls12_377.params import r
from utils           import jax_jit_lower_compile
from groth16.r1cs    import verify_witness

from kernels.ntt import get_ff_ctx
from kernels.msm import decode_g2
from kernels.scalar_codecs_cpu import (drns_encode_fr_numpy,
                                       slice_scalars_cpu_fast)
from kernels       import ec_ops
from kernels       import ec_ops_g2
from circuit_pad import pad_witness as _pad_witness
from circuit_pad import pad_witness_v2 as _pad_witness_v2
import tpu_contexts

from prover import (Proof,
                    _h_pre_ntt_kernel,
                    _h_pointwise_kernel,
                    _h_post_intt_kernel,
                    h_drns_to_int_list)

if TYPE_CHECKING:
    from framework import Setup


@dataclass
class InitEvent:
    """One progress event emitted during :class:`Prover` boot.

    The four phases are:

      * ``"tpu_contexts"``  — building MSM / NTT / Fr DRNS contexts
      * ``"aot_compile"``   — :func:`jax_jit_lower_compile` of prover kernels
      * ``"warmup"``        — throwaway prove to populate the JIT trace cache
      * (callers may add their own phases — e.g. ``"setup_load"`` upstream)

    Within each phase, ``(step, total)`` indicates sub-progress.
    """
    phase:   str
    step:    int
    total:   int
    elapsed: float
    detail:  str = ""


ProgressCB = Callable[["InitEvent"], None]


@dataclass
class EncodedWitness:
    """All per-witness derived state, ready to feed to :meth:`Prover.prove`.

    Output of :meth:`Prover.encode_witness`.  Holding this as an explicit
    dataclass:

      * makes phase-3 cost (CPU encode) visible separately from phase-4
        (TPU prove);
      * lets callers cache / batch / time the encoding step;
      * guarantees ``Prover.prove`` is a pure TPU call — no Python work
        on witness data.
    """
    a_ints:         List[int]   # witness reduced mod r
    a_drns:         Any         # DRNS-Fr encoded witness for the H pipeline
    a_g1_A_packed:  Any         # G1 A_g1 MSM scalars (sliced + packed)
    a_g1_B_packed:  Any         # G1 B_g1 MSM scalars
    private_packed: Any         # G1 private_g1 MSM scalars
    r_bits:         Any         # MSB-first int8 ladder bits for blinder r
    s_bits:         Any         # ditto for s
    rs_bits:        Any         # ditto for (-r*s) mod r


@dataclass
class ProveTelemetry:
    """Per-call timing breakdown for :meth:`Prover.prove`.

    Populated only when ``record_telemetry=True``.  Each entry is wall
    time in **milliseconds** for the labelled TPU op (with an explicit
    ``block_until_ready()`` between ops so the timings reflect actual
    completion order, not async dispatch).
    """
    total_ms:           float = 0.0
    msm_g1_A_ms:        float = 0.0
    msm_g1_B_ms:        float = 0.0
    msm_g1_private_ms:  float = 0.0
    h_pipeline_ms:      float = 0.0   # H(x) coset/NTT/INTT pipeline
    h_decode_ms:        float = 0.0   # host-side DRNS → int → packed slices
    msm_g1_h_ms:        float = 0.0
    msm_g2_B_ms:        float = 0.0
    composite_piA_ms:   float = 0.0
    composite_piB_g1_ms: float = 0.0
    composite_piB_g2_ms: float = 0.0
    composite_piC_ms:   float = 0.0
    decode_ms:          float = 0.0   # CPU EC decode at the bottom

    def to_dict(self) -> dict:
        return {
            "total_ms":            self.total_ms,
            "msm_g1_A_ms":         self.msm_g1_A_ms,
            "msm_g1_B_ms":         self.msm_g1_B_ms,
            "msm_g1_private_ms":   self.msm_g1_private_ms,
            "h_pipeline_ms":       self.h_pipeline_ms,
            "h_decode_ms":         self.h_decode_ms,
            "msm_g1_h_ms":         self.msm_g1_h_ms,
            "msm_g2_B_ms":         self.msm_g2_B_ms,
            "composite_piA_ms":    self.composite_piA_ms,
            "composite_piB_g1_ms": self.composite_piB_g1_ms,
            "composite_piB_g2_ms": self.composite_piB_g2_ms,
            "composite_piC_ms":    self.composite_piC_ms,
            "decode_ms":           self.decode_ms,
        }


def make_warmup_setup(max_size: int = 1024):
    """Construct a synthetic :class:`framework.Setup`-shaped object with
    all-zero ``pk_tpu`` / ``qap_tpu`` tensors at the canonical shapes
    for ``max_size``.

    The point isn't to produce a valid setup — it's to give
    :class:`Prover` an object with the right shapes so that
    ``__init__`` can AOT-compile and ``warmup()`` can run.  All output
    of the warmup prove is discarded.

    The synthetic tensors live in HBM (≈ 300 MB of zeros at max_size=1024).
    Drop them as soon as warmup is done by calling
    :meth:`Prover.bind_setup` with a real circuit, or
    :meth:`Prover.unbind_setup` to free HBM with no replacement.
    """
    from framework import Setup
    from tpu_precompute import PKTPU, QAPTPU

    n  = max_size
    nw = max_size
    M_fp = tpu_contexts.NUM_MODULI_MSM   # 32 for BLS12-377 Fp at 28-bit moduli
    M_fr = tpu_contexts.NUM_MODULI_NTT   # 24 for BLS12-377 Fr
    msm_ctx = tpu_contexts.get_msm_context(max_size)
    tile_num    = msm_ctx.tile_num
    tile_length = msm_ctx.tile_length

    zeros = jnp.zeros

    # G1 PK shapes — fixed bases and tiled per-wire vectors.
    alpha_g1 = zeros((4, 1, M_fp), jnp.uint32)
    beta_g1  = zeros((4, 1, M_fp), jnp.uint32)
    delta_g1 = zeros((4, 1, M_fp), jnp.uint32)

    mask_all_identity = tuple([True] * max_size)
    A_g1_tiled  = zeros((tile_num, tile_length, 4, M_fp), jnp.uint32)
    A_g1        = (A_g1_tiled, mask_all_identity)
    B_g1        = (zeros((tile_num, tile_length, 4, M_fp), jnp.uint32),
                   mask_all_identity)
    private_g1  = (zeros((tile_num, tile_length, 4, M_fp), jnp.uint32),
                   mask_all_identity)
    h_g1        = (zeros((tile_num, tile_length, 4, M_fp), jnp.uint32),
                   mask_all_identity)

    # G2 PK — projective (X, Y, Z) over Fp2 → coord 3, extension 2.
    # G2 has no identity_mask (RCB encodes identity as projective (0, 1, 0));
    # B_g2 is a plain jnp.ndarray, NOT a (points, mask) tuple.
    B_g2       = zeros((tile_num, tile_length, 3, 2, M_fp), jnp.uint32)
    beta_g2    = zeros((3, 1, 2, M_fp), jnp.uint32)
    delta_g2   = zeros((3, 1, 2, M_fp), jnp.uint32)

    pk_tpu = PKTPU(
        alpha_g1   = alpha_g1,
        beta_g1    = beta_g1,
        delta_g1   = delta_g1,
        A_g1       = A_g1,
        B_g1       = B_g1,
        private_g1 = private_g1,
        h_g1       = h_g1,
        B_g2       = B_g2,
        beta_g2    = beta_g2,
        delta_g2   = delta_g2,
    )

    qap_tpu = QAPTPU(
        U             = zeros((nw, n, M_fr), jnp.uint32),
        V             = zeros((nw, n, M_fr), jnp.uint32),
        W             = zeros((nw, n, M_fr), jnp.uint32),
        coset_shift   = zeros((n, M_fr),     jnp.uint32),
        coset_unshift = zeros((n, M_fr),     jnp.uint32),
        inv_neg2      = zeros((M_fr,),       jnp.uint32),
        n             = n,
    )

    # Stub padded R1CS so the .verify pathway (unused in warmup) doesn't NPE.
    from groth16.r1cs import R1CS
    stub_r1cs = R1CS(
        num_constraints = max_size,
        num_wires       = max_size,
        num_public      = 1,
        A = [{0: 1}] * max_size,
        B = [{0: 1}] * max_size,
        C = [{0: 1}] * max_size,
    )
    # Stub QAP exposing only ``.n`` — the only attribute Prover.__init__ reads.
    from types import SimpleNamespace
    stub_qap = SimpleNamespace(n=n, num_wires=nw, num_public=1)

    return Setup(
        padded_r1cs     = stub_r1cs,
        qap             = stub_qap,
        pk              = None,
        vk              = None,
        pk_tpu          = pk_tpu,
        qap_tpu         = qap_tpu,
        max_size        = max_size,
        num_wires_orig  = max_size,
        num_public_orig = 1,
    )


class Prover:
    """Stateful TPU Groth16 prover bound to a :class:`framework.Setup`.

    Owns TPU contexts + AOT-compiled kernels.  Public surface:

      * :meth:`encode_witness(witness, seed=None)` — phase 3 (per witness,
        CPU side: validate + DRNS-encode + slice + blind).
      * :meth:`prove(encoded)` — phase 4 (per witness, TPU side).
      * :meth:`bind_setup(new_setup)` — swap the bound circuit without
        rebuilding TPU contexts or re-AOT-compiling.  Old ``pk_tpu`` /
        ``qap_tpu`` references go to GC, freeing HBM.

    The :class:`Setup` is held by reference — ``prover.setup`` exposes
    ``pk_tpu`` / ``qap_tpu`` / ``num_public`` / etc. for callers that
    need them.
    """

    def __init__(self, setup: "Setup",
                 max_size: int | None = None, verbose: bool = True,
                 on_init_progress: Optional[ProgressCB] = None):
        """Stand up the TPU kernels for ``setup``.

        ``setup`` is a :class:`framework.Setup` returned by
        :func:`framework.compile_circuit_cached`.  It owns the
        circuit-only TPU artefacts; this class owns everything else
        (contexts, AOT-compiled kernels, the prove path).

        ``max_size`` defaults to ``setup.max_size``.

        ``on_init_progress`` is an optional callback invoked at every
        major init phase boundary — used by the demo GUI for progress
        bars.  Receives :class:`InitEvent`.  ``verbose=True`` continues
        to print the same wall-clock summaries to stdout.
        """
        self.setup      = setup
        self.pk_tpu     = setup.pk_tpu
        self.qap_tpu    = setup.qap_tpu
        self.n          = setup.n
        self.num_wires  = setup.num_wires
        self.num_public = setup.num_public
        self.max_size   = max_size if max_size is not None else setup.max_size
        self.verbose    = verbose

        if self.n > self.max_size:
            raise ValueError(
                f"circuit.n={self.n} exceeds max_size={self.max_size}"
            )

        cb = on_init_progress or (lambda _e: None)
        init_t0 = time.time()

        # ── Phase 2b — TPU contexts (4 sub-steps) ─────────────────────
        # Currently sourced from the global tpu_contexts cache (one
        # compiled instance per (msm_length, n) per process).  Binding
        # them here makes ownership explicit and keeps the rest of the
        # class free of ``tpu_contexts.get_*`` calls.
        ctx_t0 = time.time()
        cb(InitEvent("tpu_contexts", 0, 4, time.time()-init_t0,
                     "Building G1 fused MSM context"))
        self.msm_ctx_g1 = tpu_contexts.get_msm_context(self.max_size)
        cb(InitEvent("tpu_contexts", 1, 4, time.time()-init_t0,
                     "Building G2 fused MSM context (RCB projective)"))
        self.msm_ctx_g2 = tpu_contexts.get_g2_msm_context(self.max_size)
        cb(InitEvent("tpu_contexts", 2, 4, time.time()-init_t0,
                     f"Building NTT3Step (n={self.n})"))
        self.ntt_ctx    = tpu_contexts.get_ntt_context(self.n)
        cb(InitEvent("tpu_contexts", 3, 4, time.time()-init_t0,
                     "Building Fr DRNS context"))
        self.ff         = get_ff_ctx(self.n)
        # The EC contexts come bundled with their MSM contexts.
        self.ec_ctx_g1  = self.msm_ctx_g1.ec_ctx
        self.ec_ctx_g2  = self.msm_ctx_g2.ec_ctx
        if verbose:
            print(f"  [TPUProver init] contexts          : {time.time()-ctx_t0:.2f}s",
                  flush=True)

        # Identity-mask indices for each pk vector.
        self._A_g1_points,       self._A_g1_mask       = self.pk_tpu.A_g1
        self._B_g1_points,       self._B_g1_mask       = self.pk_tpu.B_g1
        self._private_g1_points, self._private_g1_mask = self.pk_tpu.private_g1
        self._h_g1_points,       self._h_g1_mask       = self.pk_tpu.h_g1
        self._private_idxs = list(range(1, self.num_wires - self.num_public))

        # ── Prime any ``@functools.cache``d helpers that allocate JAX
        # arrays BEFORE we kick off any AOT trace.  Otherwise the FIRST
        # call inside the trace caches a Tracer and leaks it into all
        # subsequent calls.  The known offender is
        # ``ec_ops_g2._identity_g2_cached()`` — call it (via
        # ``warmup()``) here so the cached identity is concrete.
        ec_ops_g2.warmup()

        # ── Phase 2c — AOT-compile every kernel we own ────────────────
        # ``jax_jit_lower_compile`` returns a ``Compiled`` artifact
        # bound to specific input shapes/dtypes.  Calling them with
        # matching real arrays runs the compiled HLO directly — no JIT
        # trigger, no abstract trace, no implicit cache lookup.
        aot_t0 = time.time()
        self._aot_compile_kernels(
            on_step=lambda step, total, detail: cb(
                InitEvent("aot_compile", step, total,
                          time.time()-init_t0, detail)),
        )
        if verbose:
            print(f"  [TPUProver init] AOT compile       : {time.time()-aot_t0:.2f}s",
                  flush=True)

    # ──────────────────────────────────────────────────────────────────
    # Setup swap (so warmup-then-real, or circuit-A-then-circuit-B,
    # both work without paying the AOT / JIT cache cost again)
    # ──────────────────────────────────────────────────────────────────

    def bind_setup(self, new_setup: "Setup") -> None:
        """Replace the bound circuit on this prover.

        Validates that the new setup's shape parameters match what the
        AOT-compiled kernels expect (``max_size``, ``n``, ``M_fp``,
        ``M_fr``); refuses to bind if they don't.  On success, the
        previous ``pk_tpu`` / ``qap_tpu`` references are dropped (HBM
        freed on the next JAX GC pass) and per-wire identity masks +
        the private-wire index list are recomputed for the new setup.
        """
        if new_setup.max_size != self.max_size:
            raise ValueError(
                f"Prover.bind_setup: bound max_size={self.max_size}, "
                f"new setup has max_size={new_setup.max_size} — these "
                f"prover kernels were AOT-compiled for the bound size"
            )
        if new_setup.n != self.n:
            raise ValueError(
                f"Prover.bind_setup: bound n={self.n}, new setup n={new_setup.n}"
            )
        new_M_fp = new_setup.pk_tpu.alpha_g1.shape[-1]
        new_M_fr = new_setup.qap_tpu.U.shape[-1]
        # ``_M_fp`` / ``_M_fr`` are captured at AOT compile time and never
        # change for the life of this prover (the kernels were compiled
        # against those shapes).  Compare against them rather than the
        # currently-bound setup, so ``bind_setup`` works after a previous
        # ``unbind_setup`` left ``self.pk_tpu`` as None.
        if new_M_fp != self._M_fp or new_M_fr != self._M_fr:
            raise ValueError(
                f"Prover.bind_setup: DRNS shapes differ — kernels compiled "
                f"for (M_fp={self._M_fp}, M_fr={self._M_fr}) vs new setup "
                f"(M_fp={new_M_fp}, M_fr={new_M_fr})"
            )

        self.setup      = new_setup
        self.pk_tpu     = new_setup.pk_tpu
        self.qap_tpu    = new_setup.qap_tpu
        self.num_public = new_setup.num_public
        # num_wires is padded to max_size; stays constant across circuits.

        self._A_g1_points,       self._A_g1_mask       = self.pk_tpu.A_g1
        self._B_g1_points,       self._B_g1_mask       = self.pk_tpu.B_g1
        self._private_g1_points, self._private_g1_mask = self.pk_tpu.private_g1
        self._h_g1_points,       self._h_g1_mask       = self.pk_tpu.h_g1
        self._private_idxs = list(range(1, self.num_wires - self.num_public))

    def unbind_setup(self) -> None:
        """Drop the bound setup's ``pk_tpu`` / ``qap_tpu`` references.

        Frees HBM (~300 MB at max_size=1024) without tearing down the
        TPU contexts or AOT-compiled kernels.  After ``unbind_setup``
        the prover is no longer usable for :meth:`encode_witness` /
        :meth:`prove` — call :meth:`bind_setup` to rebind.
        """
        self.setup      = None
        self.pk_tpu     = None
        self.qap_tpu    = None
        self.num_public = 0
        self._A_g1_points       = self._A_g1_mask       = None
        self._B_g1_points       = self._B_g1_mask       = None
        self._private_g1_points = self._private_g1_mask = None
        self._h_g1_points       = self._h_g1_mask       = None
        self._private_idxs      = []

    # ──────────────────────────────────────────────────────────────────
    # AOT compile
    # ──────────────────────────────────────────────────────────────────

    def _aot_compile_kernels(self,
                              on_step: Optional[Callable[[int, int, str], None]] = None,
                              ) -> None:
        """``jax.jit(F).lower(*shape_structs).compile()`` for every kernel
        we own.  Bound to ``self.*``; called with real arrays at prove
        time without re-tracing.

        ``on_step(step, total, detail)`` fires before each sub-compile —
        used to drive a progress bar.
        """
        step_cb = on_step or (lambda *_a, **_kw: None)
        n    = self.n
        M_fp = self.pk_tpu.alpha_g1.shape[-1]
        M_fr = self.qap_tpu.U.shape[-1]
        nw   = self.num_wires
        # Stash so bind_setup() can compare against these without needing
        # a bound pk_tpu / qap_tpu (which goes to None after unbind_setup).
        self._M_fp = M_fp
        self._M_fr = M_fr

        ss = jax.ShapeDtypeStruct
        pt_g1_3d  = ss((4, 1, M_fp),                            jnp.uint32)
        pt_g1_2d  = ss((4,    M_fp),                            jnp.uint32)
        # G2 uses projective RCB-2015 coords (3 coords, identity-safe).
        pt_g2_4d  = ss((3, 1, 2, M_fp),                         jnp.uint32)
        bits_ss   = ss((ec_ops._G1_SCALAR_BITS,),               jnp.int8)
        U_ss      = ss((nw, n, M_fr),                           jnp.uint32)
        adrns_ss  = ss((nw,    M_fr),                           jnp.uint32)
        coset_ss  = ss((n,     M_fr),                           jnp.uint32)
        invneg_ss = ss((M_fr,),                                 jnp.uint32)
        ntt_ev_ss = ss((1, self.ntt_ctx.r, self.ntt_ctx.c, M_fr), jnp.uint32)

        TOTAL = 6

        step_cb(0, TOTAL, "G1 composite: π_A / π_B (α + Σ + k·δ)")
        self._affine_combo_g1 = jax_jit_lower_compile(
            ec_ops._pi_a_or_b_g1_inner,
            pt_g1_3d, pt_g1_2d, pt_g1_3d, bits_ss,
        )
        step_cb(1, TOTAL, "G1 composite: π_C (full combination)")
        self._pi_c = jax_jit_lower_compile(
            ec_ops._pi_c_inner,
            pt_g1_2d, pt_g1_2d,                              # private, h
            pt_g1_3d, pt_g1_3d, pt_g1_3d,                     # pi_A, pi_B_g1, delta
            bits_ss, bits_ss, bits_ss,
        )

        step_cb(2, TOTAL, "G2 composite: π_B over Fp2")
        self._affine_combo_g2 = jax_jit_lower_compile(
            ec_ops_g2._pi_b_g2_inner,
            pt_g2_4d, pt_g2_4d, pt_g2_4d, bits_ss,
        )

        step_cb(3, TOTAL, "H pipeline: pre-NTT (U·a + coset shift)")
        self._h_pre = jax_jit_lower_compile(
            _h_pre_ntt_kernel(self.n, self.ntt_ctx.r, self.ntt_ctx.c),
            U_ss, U_ss, U_ss, adrns_ss, coset_ss,
        )
        step_cb(4, TOTAL, "H pipeline: pointwise (A·B − C) / (−2)")
        self._h_point = jax_jit_lower_compile(
            _h_pointwise_kernel(self.n),
            ntt_ev_ss, ntt_ev_ss, ntt_ev_ss, invneg_ss,
        )
        step_cb(5, TOTAL, "H pipeline: post-INTT (un-shift coset)")
        self._h_post = jax_jit_lower_compile(
            _h_post_intt_kernel(self.n),
            ntt_ev_ss, coset_ss,
        )

    # ──────────────────────────────────────────────────────────────────
    # TPU dispatch helpers — go through instance contexts
    # ──────────────────────────────────────────────────────────────────

    def _msm_g1(self, points_tiled, tiled_slices):
        """G1 Pippenger MSM on this prover's MSM context.

        Equivalent to ``msm_g1_pure_tpu`` but uses ``self.msm_ctx_g1``
        directly instead of the global factory lookup.
        """
        ctx = self.msm_ctx_g1
        ctx.points = points_tiled
        ctx.reset()
        return ctx.multiscalar_multiply(tiled_slices)

    def _msm_g2(self, points_tiled, scalars_ints):
        """G2 MSM on this prover's fused MSM context.

        ``points_tiled`` has shape ``(tile_num, tile_length, 3, 2, M_fp)``
        with identity bases pre-encoded as projective ``(X, Y, 0)``.  RCB
        complete add makes those slots contribute nothing, so no scalar
        masking is needed here.

        Output shape: ``(3, 1, 2, M_fp)`` — matches the shape the G2
        composite kernel (``_affine_combo_g2``) was AOT-compiled for.
        """
        ctx = self.msm_ctx_g2
        ctx.points = points_tiled
        ctx.reset()
        cleaned = [int(s) % r for s in scalars_ints]
        tiled_slices = ctx.to_computational_format(cleaned)
        result = ctx.multiscalar_multiply(tiled_slices)  # (3, 2, M)
        return jnp.expand_dims(result, axis=1)            # (3, 1, 2, M)

    # ──────────────────────────────────────────────────────────────────
    # H pipeline (bound to this instance's NTT + Fr contexts)
    # ──────────────────────────────────────────────────────────────────

    def _compute_h_drns(self, a_drns):
        """H(x) on the coset, in DRNS Fr form — uses ``self.*`` kernels."""
        A_in, B_in, C_in = self._h_pre(
            self.qap_tpu.U, self.qap_tpu.V, self.qap_tpu.W,
            a_drns, self.qap_tpu.coset_shift,
        )
        A_ev = self.ntt_ctx.ntt(A_in)
        B_ev = self.ntt_ctx.ntt(B_in)
        C_ev = self.ntt_ctx.ntt(C_in)
        H_ev = self._h_point(A_ev, B_ev, C_ev, self.qap_tpu.inv_neg2)
        H_co_rc = self.ntt_ctx.intt(H_ev)
        return self._h_post(H_co_rc, self.qap_tpu.coset_unshift)

    # ──────────────────────────────────────────────────────────────────
    # Pure TPU body
    # ──────────────────────────────────────────────────────────────────

    def _prove_pure_tpu(
        self,
        *,
        a_drns,
        a_ints,
        a_g1_A_packed,
        a_g1_B_packed,
        private_packed,
        r_bits, s_bits, rs_bits,
    ):
        """Pure-TPU body — uses bound kernels on self.  One unavoidable
        host-side step remains (the h_drns decode; T15 workaround)."""
        # G1 MSMs
        A_sum       = self._msm_g1(self._A_g1_points,       a_g1_A_packed)
        B_sum_g1    = self._msm_g1(self._B_g1_points,       a_g1_B_packed)
        private_sum = self._msm_g1(self._private_g1_points, private_packed)

        # H pipeline (TPU)
        h_co_drns = self._compute_h_drns(a_drns)

        # ⚠ CPU BOUNDARY — host-side h decode (T15 workaround).
        h_coeffs = h_drns_to_int_list(h_co_drns, self.ff, self.n)
        h_packed = slice_scalars_cpu_fast(h_coeffs, identity_mask=self._h_g1_mask)

        # h MSM
        h_sum = self._msm_g1(self._h_g1_points, h_packed)

        # G2 MSM
        B_sum_g2_tpu = self._msm_g2(self.pk_tpu.B_g2, a_ints)

        # Composite kernels
        pi_A_tpu = self._affine_combo_g1(
            self.pk_tpu.alpha_g1, A_sum, self.pk_tpu.delta_g1, r_bits,
        )
        pi_B_g1_tpu = self._affine_combo_g1(
            self.pk_tpu.beta_g1, B_sum_g1, self.pk_tpu.delta_g1, s_bits,
        )
        pi_B_g2_tpu = self._affine_combo_g2(
            self.pk_tpu.beta_g2, B_sum_g2_tpu, self.pk_tpu.delta_g2, s_bits,
        )
        pi_C_tpu = self._pi_c(
            private_sum, h_sum, pi_A_tpu, pi_B_g1_tpu, self.pk_tpu.delta_g1,
            s_bits, r_bits, rs_bits,
        )
        return pi_A_tpu, pi_B_g2_tpu, pi_C_tpu

    # ──────────────────────────────────────────────────────────────────
    # Phase 3 — witness encode (per witness, CPU side)
    # ──────────────────────────────────────────────────────────────────

    def encode_witness(
        self,
        witness: List[int],
        seed: Optional[int] = None,
        skip_r1cs_check: bool = False,
    ) -> EncodedWitness:
        """Validate + encode a witness for :meth:`prove`.

        Args:
          witness: Python ints, length ≤ ``max_size``.  Shorter inputs are
            zero-padded automatically.
          seed: When set, derive the Groth16 blinders ``r, s`` from
            ``random.Random(seed)`` so the output proof is deterministic.
            Required for reproducible demos / tests.  Leave ``None`` in
            production — the default uses ``secrets.randbelow`` which is
            cryptographically random.
          skip_r1cs_check: When True, encode an *invalid* witness anyway.
            Used by the GUI demo to show the verifier correctly rejecting
            a wrong witness — the prover still produces a syntactically
            well-formed proof, but the verifier's pairing check returns
            False.  Default False (raise on invalid witness).

        Raises:
          ValueError: witness is longer than ``max_size``, or — unless
            ``skip_r1cs_check=True`` — doesn't satisfy the padded R1CS.

        Returns:
          :class:`EncodedWitness` — feed directly to :meth:`prove`.
        """
        if len(witness) > self.max_size:
            raise ValueError(
                f"witness has {len(witness)} entries; max_size is "
                f"{self.max_size}"
            )
        # Pad using whichever strategy the bound Setup was compiled with.
        # v1 (default): zero-pad at the end, original wire indices kept.
        # v2 (T32 fix): permuting pad — public wires move to the END of
        #               the padded space so Groth16's IC[1] is non-trivial.
        padding_version = getattr(self.setup, "padding_version", 1)
        if padding_version == 2:
            w_padded = _pad_witness_v2(
                witness,
                num_wires_orig  = self.setup.num_wires_orig,
                num_public_orig = self.setup.num_public_orig,
                target_num_wires = self.max_size,
            )
        else:
            w_padded = _pad_witness(witness, target_num_wires=self.max_size)
        if (not skip_r1cs_check) and not verify_witness(self.setup.padded_r1cs, w_padded):
            raise ValueError(
                "witness does not satisfy the (padded) R1CS — check the "
                "original circuit + witness consistency"
            )

        a_ints = [int(x) % r for x in w_padded]

        a_drns = drns_encode_fr_numpy(
            a_ints, rns_moduli=self.ff.rns_moduli,
            radix_bits=self.ff.radix_bits,
            target_length=len(a_ints),
        )

        a_g1_A_packed   = slice_scalars_cpu_fast(
            a_ints, identity_mask=self._A_g1_mask)
        a_g1_B_packed   = slice_scalars_cpu_fast(
            a_ints, identity_mask=self._B_g1_mask)
        private_scalars = [a_ints[i] for i in self._private_idxs]
        private_packed  = slice_scalars_cpu_fast(
            private_scalars, identity_mask=self._private_g1_mask)

        if seed is None:
            r_blind = secrets.randbelow(r - 1) + 1
            s_blind = secrets.randbelow(r - 1) + 1
        else:
            warnings.warn(
                "Prover.encode_witness called with deterministic seed — "
                "ZK property is NOT guaranteed.  Use seed=None in production.",
                stacklevel=2,
            )
            rng = random.Random(seed)
            r_blind = rng.randrange(1, r)
            s_blind = rng.randrange(1, r)
        rs = (r - r_blind * s_blind % r) % r

        return EncodedWitness(
            a_ints         = a_ints,
            a_drns         = a_drns,
            a_g1_A_packed  = a_g1_A_packed,
            a_g1_B_packed  = a_g1_B_packed,
            private_packed = private_packed,
            r_bits         = ec_ops.int_to_bits_msb(r_blind),
            s_bits         = ec_ops.int_to_bits_msb(s_blind),
            rs_bits        = ec_ops.int_to_bits_msb(rs),
        )

    # ──────────────────────────────────────────────────────────────────
    # Phase 4 — prove (per witness, TPU side)
    # ──────────────────────────────────────────────────────────────────

    def prove(self, encoded: EncodedWitness,
              record_telemetry: bool = False):
        """Run the TPU prove kernels on a pre-encoded witness.

        Pure TPU dispatch.  Validation, encoding, and blinder generation
        all happen in :meth:`encode_witness`; this method makes no Python
        decisions based on witness contents.

        Returns:
          * ``record_telemetry=False`` (default): just the :class:`Proof`
            (back-compat).
          * ``record_telemetry=True``:  ``(Proof, ProveTelemetry)``.  Each
            sub-step is forced to completion via ``block_until_ready()``
            before timing the next, so the telemetry reflects true wall
            time of each TPU op rather than async dispatch.
        """
        if not record_telemetry:
            pi_A_tpu, pi_B_g2_tpu, pi_C_tpu = self._prove_pure_tpu(
                a_drns          = encoded.a_drns,
                a_ints          = encoded.a_ints,
                a_g1_A_packed   = encoded.a_g1_A_packed,
                a_g1_B_packed   = encoded.a_g1_B_packed,
                private_packed  = encoded.private_packed,
                r_bits          = encoded.r_bits,
                s_bits          = encoded.s_bits,
                rs_bits         = encoded.rs_bits,
            )
            pi_A_tpu.block_until_ready()
            pi_C_tpu.block_until_ready()
            pi_B_g2_tpu.block_until_ready()

            pi_A = ec_ops.decode_g1(pi_A_tpu)
            pi_C = ec_ops.decode_g1(pi_C_tpu)
            pi_B = decode_g2(pi_B_g2_tpu).to_affine()
            return Proof(A=pi_A, B=pi_B, C=pi_C)

        # ── Telemetry path — same algorithm, instrumented ────────────
        t0 = time.time()
        T = ProveTelemetry()

        # G1 MSMs
        ts = time.time()
        A_sum = self._msm_g1(self._A_g1_points, encoded.a_g1_A_packed)
        A_sum.block_until_ready()
        T.msm_g1_A_ms = (time.time() - ts) * 1000

        ts = time.time()
        B_sum_g1 = self._msm_g1(self._B_g1_points, encoded.a_g1_B_packed)
        B_sum_g1.block_until_ready()
        T.msm_g1_B_ms = (time.time() - ts) * 1000

        ts = time.time()
        private_sum = self._msm_g1(self._private_g1_points, encoded.private_packed)
        private_sum.block_until_ready()
        T.msm_g1_private_ms = (time.time() - ts) * 1000

        # H pipeline (TPU)
        ts = time.time()
        h_co_drns = self._compute_h_drns(encoded.a_drns)
        h_co_drns.block_until_ready()
        T.h_pipeline_ms = (time.time() - ts) * 1000

        # H DRNS → int → packed slices (host)
        ts = time.time()
        h_coeffs = h_drns_to_int_list(h_co_drns, self.ff, self.n)
        h_packed = slice_scalars_cpu_fast(h_coeffs, identity_mask=self._h_g1_mask)
        T.h_decode_ms = (time.time() - ts) * 1000

        # h MSM
        ts = time.time()
        h_sum = self._msm_g1(self._h_g1_points, h_packed)
        h_sum.block_until_ready()
        T.msm_g1_h_ms = (time.time() - ts) * 1000

        # G2 MSM
        ts = time.time()
        B_sum_g2_tpu = self._msm_g2(self.pk_tpu.B_g2, encoded.a_ints)
        B_sum_g2_tpu.block_until_ready()
        T.msm_g2_B_ms = (time.time() - ts) * 1000

        # Composite kernels
        ts = time.time()
        pi_A_tpu = self._affine_combo_g1(
            self.pk_tpu.alpha_g1, A_sum, self.pk_tpu.delta_g1, encoded.r_bits)
        pi_A_tpu.block_until_ready()
        T.composite_piA_ms = (time.time() - ts) * 1000

        ts = time.time()
        pi_B_g1_tpu = self._affine_combo_g1(
            self.pk_tpu.beta_g1, B_sum_g1, self.pk_tpu.delta_g1, encoded.s_bits)
        pi_B_g1_tpu.block_until_ready()
        T.composite_piB_g1_ms = (time.time() - ts) * 1000

        ts = time.time()
        pi_B_g2_tpu = self._affine_combo_g2(
            self.pk_tpu.beta_g2, B_sum_g2_tpu, self.pk_tpu.delta_g2, encoded.s_bits)
        pi_B_g2_tpu.block_until_ready()
        T.composite_piB_g2_ms = (time.time() - ts) * 1000

        ts = time.time()
        pi_C_tpu = self._pi_c(
            private_sum, h_sum, pi_A_tpu, pi_B_g1_tpu, self.pk_tpu.delta_g1,
            encoded.s_bits, encoded.r_bits, encoded.rs_bits)
        pi_C_tpu.block_until_ready()
        T.composite_piC_ms = (time.time() - ts) * 1000

        # CPU decode
        ts = time.time()
        pi_A = ec_ops.decode_g1(pi_A_tpu)
        pi_C = ec_ops.decode_g1(pi_C_tpu)
        pi_B = decode_g2(pi_B_g2_tpu).to_affine()
        T.decode_ms = (time.time() - ts) * 1000

        T.total_ms = (time.time() - t0) * 1000
        return Proof(A=pi_A, B=pi_B, C=pi_C), T

    # ──────────────────────────────────────────────────────────────────
    # Phase 2d — warmup (throwaway prove to populate JIT trace cache)
    # ──────────────────────────────────────────────────────────────────

    def warmup(self, on_progress: Optional[ProgressCB] = None) -> None:
        """Run one throwaway prove to populate the JIT trace cache.

        Critical on the first ever process boot: the G2 fused MSM kernel
        traces lazily on first call (~10 min compile).  After this call
        the trace is cached and every subsequent ``prove`` is the fast
        ~0.7 s path.

        The dummy witness is all-zero (definitely NOT a valid witness for
        any real circuit) — we go around :meth:`encode_witness` so the
        R1CS check doesn't reject it.  We only care about populating the
        JIT cache; the resulting proof is discarded.
        """
        cb = on_progress or (lambda _e: None)
        t0 = time.time()
        cb(InitEvent("warmup", 0, 1, 0.0,
                     "Running first prove (~10 min cold compile, ~1 s warm)"))

        a_ints = [0] * self.max_size
        a_drns = drns_encode_fr_numpy(
            a_ints, rns_moduli=self.ff.rns_moduli,
            radix_bits=self.ff.radix_bits,
            target_length=len(a_ints),
        )
        a_g1_A = slice_scalars_cpu_fast(a_ints, identity_mask=self._A_g1_mask)
        a_g1_B = slice_scalars_cpu_fast(a_ints, identity_mask=self._B_g1_mask)
        priv_packed = slice_scalars_cpu_fast(
            [0] * len(self._private_idxs),
            identity_mask=self._private_g1_mask,
        )
        bits = ec_ops.int_to_bits_msb(1)
        # Call the pure-TPU body directly — skip the host-side EC decode
        # at the bottom of ``prove()``, which would hit divide-by-zero
        # on the degenerate (all-identity) output of an all-zero witness.
        # The JIT trace cache only needs the TPU dispatches populated;
        # ``ec_ops.decode_g1`` is CPU-only and has nothing to warm.
        pi_A, pi_B_g2, pi_C = self._prove_pure_tpu(
            a_drns         = a_drns,
            a_ints         = a_ints,
            a_g1_A_packed  = a_g1_A,
            a_g1_B_packed  = a_g1_B,
            private_packed = priv_packed,
            r_bits  = bits,
            s_bits  = bits,
            rs_bits = bits,
        )
        # Force completion so we time the actual TPU work, not async dispatch.
        pi_A.block_until_ready()
        pi_C.block_until_ready()
        pi_B_g2.block_until_ready()
        cb(InitEvent("warmup", 1, 1, time.time() - t0,
                     "Warmup complete"))


# Back-compat alias — ``TPUProver`` is the historical name.
TPUProver = Prover
