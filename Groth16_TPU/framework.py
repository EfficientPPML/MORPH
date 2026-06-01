"""Setup-side wrapper around the TPU Groth16 prover.

Build an R1CS for your circuit, hand it to :func:`compile_circuit` (or
:func:`compile_circuit_cached`), and get back a :class:`Setup` bundle
that owns *all circuit-only state* — both CPU and TPU form.  The setup
is fully cacheable to disk: one trusted-setup run lasts forever.

For the per-witness prove path, hand the :class:`Setup` to a
:class:`prover_class.Prover` instance.

Example
-------

>>> from Groth16_TPU.framework    import Circuit, compile_circuit_cached
>>> from Groth16_TPU.prover_class import Prover
>>> r1cs  = build_my_r1cs()
>>> setup = compile_circuit_cached(Circuit(r1cs, max_size=1024), "circ.pkl")
>>> prover = Prover(setup)
>>> encoded = prover.encode_witness(my_witness, seed=42)
>>> proof   = prover.prove(encoded)
>>> ok      = setup.verify(setup.public_inputs(my_witness), proof)
"""

import os
import pickle
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import path_setup  # noqa: F401

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from groth16.r1cs import R1CS, verify_witness
from groth16.qap import r1cs_to_qap, QAP, _vanishing_poly
from groth16.setup import trusted_setup
from groth16.verifier import verify as _verify
from bls12_377.params import r as _FR
from prover import Proof
from kernels.ntt import get_ff_ctx, intt_raw, _MIN_TPU_N, get_omega
from circuit_pad import pad_r1cs, pad_r1cs_v2, pad_witness, pad_witness_v2
import tpu_contexts
import tpu_precompute
import jax.numpy as _jnp


# Cache-file format version — bump if the pickled layout ever changes.
# v2 adds TPU-side precompute caches (pk_tpu, qap_tpu) so on-load we can
# skip ~83s of `precompute_pk` / `precompute_qap` redo on each restart.
_CACHE_FORMAT_VERSION = 7  # bumped: Setup now records padding_version (T32)


# ─────────────────────────────────────────────────────────────────────────────
# Batched r1cs_to_qap (T10)
# ─────────────────────────────────────────────────────────────────────────────


def _bit_reverse(a):
    """Permute ``a`` (len = power of 2) so the i-th element lands at the
    bit-reversed index.  Mirrors ``kernels.ntt._bit_reverse``.
    """
    n = len(a)
    bits = n.bit_length() - 1
    out = [0] * n
    for i in range(n):
        j = 0
        x = i
        for _ in range(bits):
            j = (j << 1) | (x & 1)
            x >>= 1
        out[j] = a[i]
    return out


def _r1cs_to_qap_batched(r1cs: R1CS) -> QAP:
    """Drop-in replacement for ``groth16.qap.r1cs_to_qap`` that batches the
    INTT calls.  Equivalent output (a ``QAP`` instance), produced in ~one
    pass of host-side encoding / decoding instead of 3·num_wires individual
    `intt()` calls.

    For n=1024, num_wires=1024 this cuts r1cs_to_qap from ~418s to ~20-40s
    by amortising the per-call CPU encode/decode dispatch.
    """
    n = r1cs.num_constraints
    assert n > 0 and (n & (n - 1)) == 0, "num_constraints must be power of 2"

    omega   = get_omega(n)
    omega2n = get_omega(2 * n)

    # Build flat eval-table list for all 3·num_wires polys.  Each row is a
    # pre-bit-reversed length-n int vector — bit-reversal matches what
    # ``kernels.ntt.intt`` does internally before feeding NTT3Step.
    rows = []
    A = r1cs.A
    B = r1cs.B
    C = r1cs.C
    nw = r1cs.num_wires
    for i in range(nw):
        eval_U = [A[j].get(i, 0) % _FR for j in range(n)]
        eval_V = [B[j].get(i, 0) % _FR for j in range(n)]
        eval_W = [C[j].get(i, 0) % _FR for j in range(n)]
        rows.append(_bit_reverse(eval_U))
        rows.append(_bit_reverse(eval_V))
        rows.append(_bit_reverse(eval_W))
    # ``rows`` shape (3·num_wires, n).  rows[3i+0/1/2] = U/V/W eval i.

    if n < _MIN_TPU_N:
        # Fall back to the upstream path for tiny n where TPU NTT isn't set up.
        return r1cs_to_qap(r1cs)

    ff = get_ff_ctx(n)
    ntt_ctx = tpu_contexts.get_ntt_context(n)

    # One big encode pass: (3·nw, n) Python ints → (3·nw, n, M_fr) JAX uint32.
    arr = ff.to_computational_format(rows)
    # NTT3Step takes (B, r, c, M_fr).  Reshape (3·nw, n, M_fr) → (3·nw, r, c, M_fr).
    arr = arr.reshape(3 * nw, ntt_ctx.r, ntt_ctx.c, arr.shape[-1])

    # Run intt for each row.  The JIT'd kernel is shape-keyed on the
    # full input shape; we feed one row at a time so it sees the same
    # shape as the per-call path (and reuses any existing JIT cache).
    out_chunks = []
    for i in range(3 * nw):
        slice_i = arr[i:i + 1]                    # (1, r, c, M_fr)
        out_chunks.append(intt_raw(slice_i, n))
    out = _jnp.concatenate(out_chunks, axis=0)    # (3·nw, r, c, M_fr)
    out = out.reshape(3 * nw, n, arr.shape[-1])
    out.block_until_ready()

    # One big decode pass → nested Python int lists of shape (3·nw, n).
    decoded = ff.to_original_format(out)

    U_coeffs = [decoded[3 * i + 0] for i in range(nw)]
    V_coeffs = [decoded[3 * i + 1] for i in range(nw)]
    W_coeffs = [decoded[3 * i + 2] for i in range(nw)]
    t_coeffs = _vanishing_poly(n)

    return QAP(
        n          = n,
        ntt_size   = 2 * n,
        degree     = n - 1,
        num_wires  = nw,
        num_public = r1cs.num_public,
        omega      = omega,
        omega2n    = omega2n,
        t_coeffs   = t_coeffs,
        U          = U_coeffs,
        V          = V_coeffs,
        W          = W_coeffs,
    )


_DEFAULT_MAX_SIZE = 1024


@dataclass
class Circuit:
    """User-facing spec of an R1CS-based circuit.

    Attributes
    ----------
    r1cs
        The R1CS as built by upstream Groth16 tools (`groth16.r1cs.R1CS`).
        Must satisfy ``r1cs.num_constraints <= max_size`` and
        ``r1cs.num_wires <= max_size``.
    max_size
        Power-of-two upper bound for both `num_constraints` and
        `num_wires` after padding.  Defaults to 1024 — the largest
        circuit the current TPU prover JIT-shape is tuned for.
    padding_version
        ``1`` (default) keeps the original wire indices through padding
        — current behaviour, has a known soundness gap (the public-input
        value isn't actually bound to the proof; see
        ``PUBLIC_INPUT_BINDING.md``).  ``2`` permutes the original public
        wires to the END of the padded space so the verifier's IC[1]
        becomes non-trivial and the binding is restored.  v2 is opt-in
        while we collect operational evidence.
    """
    r1cs: R1CS
    max_size: int = _DEFAULT_MAX_SIZE
    padding_version: int = 1


@dataclass
class Setup:
    """A circuit compiled for the TPU Groth16 prover.

    Owns **all circuit-only state** — both CPU-form (``padded_r1cs``,
    ``qap``, ``pk``, ``vk``) and TPU-form (``pk_tpu``, ``qap_tpu``).  All
    cacheable to disk via :meth:`save` / :meth:`load`; per-process or
    per-witness work lives elsewhere (see :class:`prover_class.Prover`).

    Phase-1 of the demo pipeline.
    """
    padded_r1cs:    R1CS
    qap:            Any
    pk:             Any
    vk:             Any
    pk_tpu:         Any    # tpu_precompute.PKTPU
    qap_tpu:        Any    # tpu_precompute.QAPTPU
    max_size:       int
    # Pre-padding shape — needed by :meth:`public_inputs` to find the
    # original public-wire positions in the (now padded) witness.  In
    # the unpadded R1CS, public wires are
    # ``[0] + [num_wires_orig - num_public .. num_wires_orig - 1]``;
    # padding appends private-zero wires at the end so the original
    # public wires keep their indices.
    num_wires_orig:  int
    num_public_orig: int
    # Which padding strategy was used to produce ``padded_r1cs``.  See
    # :class:`Circuit.padding_version`.  Defaults to ``1`` for back-compat
    # with pkls saved before T32 was wired up.
    padding_version: int = 1

    # ── derived properties ────────────────────────────────────────────
    @property
    def num_public(self) -> int:
        """Number of public-input values (excluding the constant-1 wire 0)."""
        return self.num_public_orig

    @property
    def num_wires(self) -> int:
        return self.padded_r1cs.num_wires

    @property
    def n(self) -> int:
        """QAP domain size (power of 2)."""
        return self.qap.n

    # ── ergonomic helpers ─────────────────────────────────────────────
    def public_inputs(self, witness: List[int]) -> List[int]:
        """Extract the public-input slice from a witness.

        Returns the values at the original-R1CS public wire positions —
        i.e. ``witness[num_wires_orig - num_public : num_wires_orig]``.
        Padding appends private-zero wires at the end so these indices
        are stable across padded / unpadded witnesses.
        """
        start = self.num_wires_orig - self.num_public_orig
        end   = self.num_wires_orig
        return list(witness[start:end])

    def verify(self, public_inputs: List[int], proof: Proof) -> bool:
        """Verify a proof (CPU pairings).  Stateless wrt this setup."""
        return _verify(self.vk, public_inputs, proof)

    # ── disk I/O ──────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str) -> "Setup":
        """Restore a previously-saved :class:`Setup`."""
        return _load_setup_from_disk(path)

    def save(self, path: str) -> None:
        """Persist this :class:`Setup` for fast reload in future processes."""
        _save_setup_to_disk(self, path)


# Back-compat alias — older code (and tests) import ``CompiledCircuit``.
CompiledCircuit = Setup


SetupProgressCB = Callable[[str, float, str], None] if False else "Callable[[str, float, str], None]"


def _validate_circuit(r1cs, max_size: int) -> None:
    """Step 0 — shape validation.  Pure CPU, microseconds."""
    if (max_size <= 0) or (max_size & (max_size - 1)) != 0:
        raise ValueError(
            f"max_size must be a positive power of 2; got {max_size}")
    if r1cs.num_constraints > max_size:
        raise ValueError(
            f"r1cs has {r1cs.num_constraints} constraints; max_size is {max_size}")
    if r1cs.num_wires > max_size:
        raise ValueError(
            f"r1cs has {r1cs.num_wires} wires; max_size is {max_size}")


def compile_circuit(circ: Circuit,
                    progress: "Optional[Callable[[str, float, str], None]]" = None
                    ) -> "Setup":
    """One-time setup: validate, pad, build QAP, run trusted setup,
    precompute TPU artefacts.

    The work is split into named phases so callers (e.g. the demo
    daemon) can render progress.  ``progress(phase, frac, detail)``
    is invoked at every step boundary:

      * ``r1cs_pad``       — pad R1CS to ``max_size`` constraints/wires (CPU, ms)
      * ``qap_build``      — batched ``r1cs_to_qap`` (TPU INTT, ~20-40 s)
      * ``trusted_setup``  — sample α/β/γ/δ/τ + commit pk/vk EC points (CPU, ~30-90 s)
      * ``encode_pk``      — DRNS-encode pk for the TPU (CPU+TPU, parallel)
      * ``encode_qap``     — DRNS-encode qap U/V/W (CPU+TPU, parallel)

    ``progress`` may be ``None`` (silent).
    """
    cb = progress or (lambda *_a, **_kw: None)

    r1cs     = circ.r1cs
    max_size = circ.max_size

    # Step 0 — validate
    _validate_circuit(r1cs, max_size)

    # Step 1 — pad R1CS.  v1 (default) keeps original wire indices;
    # v2 permutes public wires to the END of the padded space so
    # Groth16's IC[1] becomes non-trivial (see PUBLIC_INPUT_BINDING.md).
    cb("r1cs_pad", 0.00, f"Padding R1CS to {max_size} constraints / wires (v{circ.padding_version})")
    if circ.padding_version == 2:
        padded = pad_r1cs_v2(
            r1cs,
            target_num_constraints=max_size,
            target_num_wires=max_size,
        )
    else:
        padded = pad_r1cs(
            r1cs,
            target_num_constraints=max_size,
            target_num_wires=max_size,
        )

    # Step 2 — build QAP (batched TPU INTT amortises per-wire encode cost).
    cb("qap_build", 0.10, "Building QAP via batched INTT")
    qap = _r1cs_to_qap_batched(padded)

    # Step 3 — sample toxic waste, commit pk / vk EC points.
    cb("trusted_setup", 0.30,
       "Sampling α/β/γ/δ/τ; committing pk and vk to EC points")
    pk, vk = trusted_setup(qap)

    # Step 4 — DRNS-encode pk for the TPU (heavy: uses MP pool for the
    # per-wire scalar reductions; see ``finite_field_context.py:517``).
    cb("encode_pk", 0.70, "DRNS-encoding pk_tpu")
    pk_tpu = tpu_precompute.precompute_pk(pk)

    # Step 5 — DRNS-encode QAP U/V/W (same MP-pool path).
    cb("encode_qap", 0.85, "DRNS-encoding qap_tpu (U/V/W polynomials)")
    qap_tpu = tpu_precompute.precompute_qap(qap)

    cb("done", 1.00, "Setup ready")

    return Setup(
        padded_r1cs     = padded,
        qap             = qap,
        pk              = pk,
        vk              = vk,
        pk_tpu          = pk_tpu,
        qap_tpu         = qap_tpu,
        max_size        = max_size,
        num_wires_orig  = r1cs.num_wires,
        num_public_orig = r1cs.num_public,
        padding_version = circ.padding_version,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Disk cache for the (slow) CPU side of compile_circuit
# ─────────────────────────────────────────────────────────────────────────────


def _jax_to_npbundle(obj):
    """Recursively convert JAX arrays in a structure to numpy for pickling.
    Tuples / lists / dicts are walked; everything else is passed through.
    """
    import numpy as _np
    if hasattr(obj, "shape") and hasattr(obj, "dtype") and hasattr(obj, "device"):
        # Likely a jax.Array — pull it host-side as a numpy array.
        return ("__jax__", _np.asarray(obj))
    if isinstance(obj, tuple):
        return tuple(_jax_to_npbundle(x) for x in obj)
    if isinstance(obj, list):
        return [_jax_to_npbundle(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jax_to_npbundle(v) for k, v in obj.items()}
    return obj


def _npbundle_to_jax(obj):
    """Inverse of :func:`_jax_to_npbundle` — turn the ``("__jax__", np_array)``
    tuples back into device-resident JAX arrays.
    """
    import jax
    import jax.numpy as _jnp
    if isinstance(obj, tuple):
        if len(obj) == 2 and obj[0] == "__jax__":
            return _jnp.asarray(obj[1]).to_device(jax.devices("tpu")[0])
        return tuple(_npbundle_to_jax(x) for x in obj)
    if isinstance(obj, list):
        return [_npbundle_to_jax(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _npbundle_to_jax(v) for k, v in obj.items()}
    return obj


def _pktpu_to_npbundle(pk_tpu):
    """Serialise a PKTPU dataclass (TPU-resident JAX arrays + identity-mask
    tuples) into a plain dict of numpy bundles.  Pickle-safe.
    """
    return {
        "alpha_g1":   _jax_to_npbundle(pk_tpu.alpha_g1),
        "beta_g1":    _jax_to_npbundle(pk_tpu.beta_g1),
        "delta_g1":   _jax_to_npbundle(pk_tpu.delta_g1),
        "A_g1":       _jax_to_npbundle(pk_tpu.A_g1),
        "B_g1":       _jax_to_npbundle(pk_tpu.B_g1),
        "private_g1": _jax_to_npbundle(pk_tpu.private_g1),
        "h_g1":       _jax_to_npbundle(pk_tpu.h_g1),
        "B_g2":       _jax_to_npbundle(pk_tpu.B_g2),
        "beta_g2":    _jax_to_npbundle(pk_tpu.beta_g2),
        "delta_g2":   _jax_to_npbundle(pk_tpu.delta_g2),
    }


def _npbundle_to_pktpu(bundle):
    from tpu_precompute import PKTPU
    return PKTPU(
        alpha_g1   = _npbundle_to_jax(bundle["alpha_g1"]),
        beta_g1    = _npbundle_to_jax(bundle["beta_g1"]),
        delta_g1   = _npbundle_to_jax(bundle["delta_g1"]),
        A_g1       = _npbundle_to_jax(bundle["A_g1"]),
        B_g1       = _npbundle_to_jax(bundle["B_g1"]),
        private_g1 = _npbundle_to_jax(bundle["private_g1"]),
        h_g1       = _npbundle_to_jax(bundle["h_g1"]),
        B_g2       = _npbundle_to_jax(bundle["B_g2"]),
        beta_g2    = _npbundle_to_jax(bundle["beta_g2"]),
        delta_g2   = _npbundle_to_jax(bundle["delta_g2"]),
    )


def _qaptpu_to_npbundle(qap_tpu):
    return {
        "U":             _jax_to_npbundle(qap_tpu.U),
        "V":             _jax_to_npbundle(qap_tpu.V),
        "W":             _jax_to_npbundle(qap_tpu.W),
        "coset_shift":   _jax_to_npbundle(qap_tpu.coset_shift),
        "coset_unshift": _jax_to_npbundle(qap_tpu.coset_unshift),
        "inv_neg2":      _jax_to_npbundle(qap_tpu.inv_neg2),
        "n":             qap_tpu.n,
    }


def _npbundle_to_qaptpu(bundle):
    from tpu_precompute import QAPTPU
    return QAPTPU(
        U             = _npbundle_to_jax(bundle["U"]),
        V             = _npbundle_to_jax(bundle["V"]),
        W             = _npbundle_to_jax(bundle["W"]),
        coset_shift   = _npbundle_to_jax(bundle["coset_shift"]),
        coset_unshift = _npbundle_to_jax(bundle["coset_unshift"]),
        inv_neg2      = _npbundle_to_jax(bundle["inv_neg2"]),
        n             = bundle["n"],
    )


def _save_setup_to_disk(setup: Setup, path: str) -> None:
    """Pickle CPU-side + TPU-side setup artefacts so a later run skips
    both trusted-setup (~minutes) and TPU encoding (~80 s).

    TPU-resident JAX arrays are pulled host-side as numpy via
    :func:`_jax_to_npbundle`; reuploaded to the device on load.
    """
    payload = {
        "version":          _CACHE_FORMAT_VERSION,
        "max_size":         setup.max_size,
        "padded_r1cs":      setup.padded_r1cs,
        "qap":              setup.qap,
        "pk":               setup.pk,
        "vk":               setup.vk,
        "pk_tpu":           _pktpu_to_npbundle(setup.pk_tpu),
        "qap_tpu":          _qaptpu_to_npbundle(setup.qap_tpu),
        "num_wires_orig":   setup.num_wires_orig,
        "num_public_orig":  setup.num_public_orig,
        "padding_version":  setup.padding_version,
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_setup_from_disk(path: str) -> Setup:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    # Accept v6 (pre-T32) pkls and treat them as padding_version=1 — they
    # were all built before T32 introduced v2 padding, so v1 is correct.
    # Reject anything older than v6 (the layout was incompatible before
    # num_wires_orig was tracked).
    file_version = payload.get("version")
    if file_version not in (6, _CACHE_FORMAT_VERSION):
        raise ValueError(
            f"setup cache at {path} has version "
            f"{file_version!r}; expected {_CACHE_FORMAT_VERSION} (or 6 for "
            f"pre-T32 pkls).  Delete and re-run compile_circuit_cached to refresh."
        )
    pk_tpu  = _npbundle_to_pktpu(payload["pk_tpu"])
    qap_tpu = _npbundle_to_qaptpu(payload["qap_tpu"])
    setup = Setup(
        padded_r1cs     = payload["padded_r1cs"],
        qap             = payload["qap"],
        pk              = payload["pk"],
        vk              = payload["vk"],
        pk_tpu          = pk_tpu,
        qap_tpu         = qap_tpu,
        max_size        = payload["max_size"],
        num_wires_orig  = payload["num_wires_orig"],
        num_public_orig = payload["num_public_orig"],
        padding_version = payload.get("padding_version", 1),
    )
    # Also prime the id-keyed caches in ``tpu_precompute`` so any
    # downstream ``precompute_pk(setup.pk)`` / ``precompute_qap(setup.qap)``
    # callers return the same objects (back-compat with code paths that
    # still go through the function-based API).
    tpu_precompute._PK_CACHE[id(setup.pk)]   = pk_tpu
    tpu_precompute._QAP_CACHE[id(setup.qap)] = qap_tpu
    return setup


def compile_circuit_cached(
    circ: Circuit,
    cache_path: str,
    force_recompile: bool = False,
    progress: Optional[Callable[[str, float, str], None]] = None,
) -> CompiledCircuit:
    """Like :func:`compile_circuit`, but memoise the slow CPU-side setup
    artefacts to ``cache_path`` on disk.

    Useful during iterative development: run once to populate the cache
    (paying the ~minutes-long trusted-setup cost), then every subsequent
    run loads from disk in seconds and only pays the (much cheaper)
    TPU precompute on each invocation.

    Cache-invalidation is the *caller's responsibility*: if the R1CS or
    ``max_size`` change, the cache file at ``cache_path`` must be deleted
    (or ``force_recompile=True`` passed) — we don't hash the inputs
    automatically.  A cheap sanity check (max_size + num_constraints +
    num_wires) catches obvious mismatches and raises.

    Args:
        circ: User circuit spec.
        cache_path: Filesystem path for the pickle.  Created on miss,
            read on hit.
        force_recompile: If True, ignore any existing cache file and
            run the full setup (overwrites the cache file on success).
    """
    if (not force_recompile) and os.path.exists(cache_path):
        t = time.time()
        ctx = _load_setup_from_disk(cache_path)
        # Sanity check against the supplied Circuit spec.
        mismatch = (
            ctx.max_size != circ.max_size
            or ctx.padded_r1cs.num_public != circ.r1cs.num_public
        )
        if mismatch:
            raise ValueError(
                f"setup cache at {cache_path} disagrees with the supplied "
                f"Circuit (max_size {ctx.max_size} vs {circ.max_size}, or "
                f"num_public {ctx.padded_r1cs.num_public} vs "
                f"{circ.r1cs.num_public}).  Delete the cache file and retry "
                f"with force_recompile=True if intentional."
            )
        print(f"  [compile_cached] loaded from {cache_path} in {time.time()-t:.2f}s")
        return ctx

    # Cache miss → full path, then save.
    t = time.time()
    ctx = compile_circuit(circ, progress=progress)
    print(f"  [compile_cached] compiled from scratch in {time.time()-t:.2f}s")
    if progress is not None:
        progress("save_pkl", 0.98, f"Writing {os.path.basename(cache_path)}")
    _save_setup_to_disk(ctx, cache_path)
    if progress is not None:
        progress("done", 1.00, "Setup cached to disk")
    print(f"  [compile_cached] saved cache → {cache_path}")
    return ctx
