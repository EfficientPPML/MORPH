"""Multi-circuit boot worker for the GUI demo.

Two state machines run in the daemon process:

1. **TPU env**:  one ``Prover`` instance with TPU contexts + AOT-compiled
   kernels + a warm JIT cache.  Warmed up at daemon start against a
   synthetic all-zero :class:`Setup` (see :func:`prover_class.make_warmup_setup`)
   so the long compile / warmup happens BEFORE the user picks a circuit.
   After warmup, the synthetic setup's tensors are dropped via
   :meth:`Prover.unbind_setup`, freeing the ~300 MB of warmup HBM.

2. **Per-circuit registry**:  for each entry in ``demo_circuits.CIRCUITS``,
   a :class:`CircuitState` tracks whether the on-disk pkl exists,
   whether a ``prep_circuit`` subprocess is currently building it,
   and whether it is the active circuit (bound into the prover).

Concurrency invariants:

  * **One** prover in the process — switching active drops the previous
    setup's tensors via :meth:`Prover.bind_setup`.
  * **One** ``prep_circuit`` subprocess per circuit at a time (per-key
    lock).  Multiple circuits may be building in parallel.
  * Active-circuit swaps and ``/prove`` are serialised via
    ``_prove_lock`` — TPU is single-tenant.  Build subprocesses do
    NOT take this lock; they run on separate CPU cores and don't touch
    the TPU until activation.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


# ── State dataclasses ──────────────────────────────────────────────────


@dataclass
class TPUEnvState:
    """Process-singleton TPU env state.  Polled by the UI ~1× / sec."""
    phase:      str   = "idle"
    progress:   float = 0.0
    detail:     str   = ""
    elapsed:    float = 0.0
    done:       bool  = False        # True once warmup completes
    error:      Optional[str] = None
    start_time: float = 0.0
    # Prover instance kept alive across circuit swaps.  After warmup
    # this is bound to a synthetic Setup until the first user-driven
    # circuit ``activate``; from then on it's bound to whatever the
    # user selected.
    prover:      object = field(default=None, repr=False)
    active_key:  Optional[str] = None   # circuit currently bound to ``prover``


@dataclass
class BuildSubstate:
    """Live state of a ``prep_circuit`` subprocess for one circuit."""
    phase:      str   = "queued"     # "queued" | "running" | "done" | "error" | "cancelled"
    progress:   float = 0.0          # in [0, 1]
    detail:     str   = ""
    elapsed:    float = 0.0
    start_time: float = 0.0
    error:      Optional[str] = None


@dataclass
class CircuitState:
    """Per-circuit state in the registry."""
    key:         str
    pkl_path:    str
    pkl_exists:  bool                = False
    is_active:   bool                = False
    build:       Optional[BuildSubstate] = None    # only set if build is/was running


# ── Module-level singletons ───────────────────────────────────────────


_env       = TPUEnvState()
_env_lock  = threading.Lock()
_circuits: Dict[str, CircuitState] = {}
_circuits_lock = threading.Lock()
# Per-circuit build subprocess locks (lazy-init in _build_lock).
_build_locks: Dict[str, threading.Lock] = {}
_build_locks_meta = threading.Lock()
# Per-circuit subprocess Popen handles (for cancel) — reserved for the
# legacy CLI path; current daemon builds run in-thread under ``_tpu_lock``.
_build_procs: Dict[str, subprocess.Popen] = {}
# Serialises {prove, activate, deactivate} — TPU is single-tenant.
_prove_lock = threading.Lock()
# Serialises every TPU-using operation: env warmup, build's qap_build /
# encode_pk / encode_qap, and prove.  Builds and proves interleave at
# the granularity of whatever holds this lock; neither truly runs in
# parallel, but both make progress.
_tpu_lock = threading.RLock()
# Per-build cancel signal — checked between phases by the build worker.
_cancel_flags: Dict[str, bool] = {}
# Idempotency for start_env.
_env_started: bool = False

# ── Persistent circuit categories ─────────────────────────────────────
# Source-of-truth config file at a well-known location.  Lists EVERY
# registered circuit and its category ("practical" or "archive"), so
# the file alone tells you the full state — no need to consult the
# CircuitSpec defaults to interpret it.  Users may hand-edit the file
# and restart the daemon; the UI's "Move to archive / practical"
# button rewrites the same file.

_circuit_categories: Dict[str, str] = {}
_categories_path: Optional[str] = None
_category_lock = threading.Lock()
_VALID_CATEGORIES = ("practical", "archive")


def init_categories(path: str, defaults: Dict[str, str]) -> None:
    """Wire up the categories config file.

    Called once at daemon startup with ``defaults`` mapping every
    registered circuit key to its CircuitSpec-default category.
    Behaviour:

      * If the file exists, load it.  Any registered circuit missing
        from the file is filled in with its spec default; any extra
        key not in ``defaults`` is dropped.  The file is then re-written
        so it's always self-consistent.
      * If the file doesn't exist, write a fresh one containing every
        registered circuit at its spec default.

    Result: ``circuit_categories.json`` always lists every active
    circuit explicitly.  Hand-editing is just changing the value.
    """
    global _categories_path, _circuit_categories
    _categories_path = path
    loaded: Dict[str, str] = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                loaded = {
                    str(k): str(v) for k, v in raw.items()
                    if v in _VALID_CATEGORIES
                }
        except Exception:
            loaded = {}
    # Merge: prefer stored value, fall back to default for missing keys.
    merged = {
        k: loaded.get(k, defaults[k])
        for k in defaults
    }
    with _category_lock:
        _circuit_categories = merged
        _persist_categories_locked()


def _persist_categories_locked() -> None:
    """Caller MUST hold ``_category_lock``.  Writes the full mapping
    atomically (write to .tmp, then rename)."""
    if _categories_path is None:
        return
    tmp = _categories_path + ".tmp"
    os.makedirs(os.path.dirname(_categories_path) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(_circuit_categories, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, _categories_path)


def get_category(key: str) -> Optional[str]:
    """Return the stored category for ``key``, or ``None`` if unknown."""
    with _category_lock:
        return _circuit_categories.get(key)


def set_category(key: str, category: str) -> None:
    """Persist ``key → category``.  Rewrites the full config file."""
    if category not in _VALID_CATEGORIES:
        raise ValueError(
            f"category must be one of {_VALID_CATEGORIES}; got {category!r}"
        )
    with _category_lock:
        if key not in _circuit_categories:
            raise KeyError(f"circuit {key!r} is not registered")
        _circuit_categories[key] = category
        _persist_categories_locked()


def _build_lock(key: str) -> threading.Lock:
    with _build_locks_meta:
        lock = _build_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _build_locks[key] = lock
        return lock


# ── Public read-only accessors ─────────────────────────────────────────


def env_state() -> TPUEnvState:
    """Snapshot of the TPU env state.  Cheap; UI polls ~1× / sec."""
    with _env_lock:
        return replace(_env)


def list_circuit_states() -> List[CircuitState]:
    """Snapshot of every registered circuit's state.  Stable order."""
    with _circuits_lock:
        return [replace(cs, build=(replace(cs.build) if cs.build else None))
                for cs in _circuits.values()]


def get_circuit_state(key: str) -> Optional[CircuitState]:
    with _circuits_lock:
        cs = _circuits.get(key)
        if cs is None:
            return None
        return replace(cs, build=(replace(cs.build) if cs.build else None))


def register_circuit(key: str, pkl_path: str) -> None:
    """Make ``key`` available in the registry.  Idempotent."""
    with _circuits_lock:
        if key in _circuits:
            return
        _circuits[key] = CircuitState(
            key        = key,
            pkl_path   = pkl_path,
            pkl_exists = os.path.exists(pkl_path),
        )


def _env_update(**kwargs) -> None:
    with _env_lock:
        for k, v in kwargs.items():
            setattr(_env, k, v)


def _circuit_update(key: str, **kwargs) -> None:
    with _circuits_lock:
        cs = _circuits.get(key)
        if cs is None:
            return
        for k, v in kwargs.items():
            setattr(cs, k, v)


def _build_update(key: str, **kwargs) -> None:
    with _circuits_lock:
        cs = _circuits.get(key)
        if cs is None or cs.build is None:
            return
        for k, v in kwargs.items():
            setattr(cs.build, k, v)


# ── TPU env warmup (one-time at daemon start) ─────────────────────────


def start_env(max_size: int = 1024) -> None:
    """Kick off TPU env warmup in a background thread.  Idempotent.

    Builds a process-wide :class:`Prover` against a synthetic Setup,
    runs warmup, then drops the synthetic setup's tensors.  After this
    completes (``env_state().done == True``), per-circuit activation
    via :func:`activate_circuit` is fast (~5-10 s; reuses contexts +
    AOT kernels + warm JIT cache).
    """
    global _env_started
    with _env_lock:
        if _env_started:
            return
        _env_started = True
        _env.phase      = "starting"
        _env.detail     = "Loading modules"
        _env.progress   = 0.0
        _env.elapsed    = 0.0
        _env.done       = False
        _env.error      = None
        _env.start_time = time.time()

    def _worker():
        t0 = _env.start_time
        try:
            from prover_class import Prover, InitEvent, make_warmup_setup

            # Phase-range table for the env warmup path (no setup_compile).
            phase_range = {
                "tpu_contexts": (0.00, 0.45),
                "aot_compile":  (0.45, 0.85),
                "warmup":       (0.85, 1.00),
            }

            def cb(e: "InitEvent"):
                lo, hi = phase_range.get(e.phase, (0.0, 1.0))
                frac = (e.step + 1) / max(1, e.total)
                _env_update(
                    phase    = e.phase,
                    detail   = e.detail,
                    progress = lo + frac * (hi - lo),
                    elapsed  = time.time() - t0,
                )

            with _tpu_lock:
                synthetic = make_warmup_setup(max_size)
                prover    = Prover(synthetic, verbose=False, on_init_progress=cb)
                prover.warmup(on_progress=cb)
                # Drop synthetic setup's tensors to free HBM (≈ 300 MB at max_size=1024).
                prover.unbind_setup()
                del synthetic

            _env_update(
                phase      = "ready",
                detail     = "TPU env warm — select a circuit to build/use",
                progress   = 1.0,
                elapsed    = time.time() - t0,
                prover     = prover,
                active_key = None,
                done       = True,
            )
        except Exception:
            _env_update(
                error   = traceback.format_exc(),
                done    = True,
                elapsed = time.time() - t0,
            )

    threading.Thread(target=_worker, daemon=True, name="tpu-env-warmup").start()


# ── Per-circuit build (prep_circuit subprocess) ───────────────────────


def start_build(key: str, max_size: int = 1024, force: bool = False) -> str:
    """Spawn ``prep_circuit`` for ``key`` in a background thread.

    Returns a status string: ``"started"``, ``"already_running"``, or
    ``"already_built"``.  The subprocess streams JSON progress events
    which are folded into :attr:`CircuitState.build`.

    ``force=True`` rebuilds even when a pickle already exists on disk —
    the existing (possibly stale / version-incompatible) cache is ignored
    and overwritten with a fresh trusted setup.
    """
    cs = get_circuit_state(key)
    if cs is None:
        raise KeyError(f"circuit {key!r} not registered")
    if cs.pkl_exists and not force:
        return "already_built"
    lock = _build_lock(key)
    if not lock.acquire(blocking=False):
        return "already_running"
    # Initialize the build substate; we'll release the lock in the worker.
    _circuit_update(key, build=BuildSubstate(
        phase      = "queued",
        progress   = 0.0,
        detail     = "Spawning prep_circuit subprocess",
        elapsed    = 0.0,
        start_time = time.time(),
    ))

    _cancel_flags[key] = False

    def _worker():
        t0 = time.time()
        try:
            _run_inprocess_build(
                key       = key,
                pkl_path  = cs.pkl_path,
                max_size  = max_size,
                t_start   = t0,
                force     = force,
            )
            _build_update(key,
                phase    = "done",
                detail   = "Setup pickle ready on disk",
                progress = 1.0,
                elapsed  = time.time() - t0,
            )
            _circuit_update(key, pkl_exists=True)
        except _Cancelled:
            _build_update(key,
                phase   = "cancelled",
                detail  = "Build cancelled by user",
                elapsed = time.time() - t0,
            )
        except Exception:
            _build_update(key,
                phase   = "error",
                error   = traceback.format_exc(),
                elapsed = time.time() - t0,
            )
        finally:
            _cancel_flags.pop(key, None)
            lock.release()

    threading.Thread(target=_worker, daemon=True,
                     name=f"build-{key}").start()
    return "started"


class _Cancelled(RuntimeError):
    pass


def _check_cancel(key: str) -> None:
    if _cancel_flags.get(key):
        raise _Cancelled()


def _run_inprocess_build(*, key: str, pkl_path: str, max_size: int,
                          t_start: float, force: bool = False) -> None:
    """Run ``framework.compile_circuit_cached`` inside the daemon process.

    Acquires ``_tpu_lock`` for the duration so the TPU isn't shared with
    the env warmup or any other build / prove.  The user-visible
    behaviour: build phases that don't touch the TPU still emit progress
    (the lock is reentrant); only TPU-bound phases actually serialise.

    A ``prep_circuit`` subprocess approach would be cleaner from a
    sandboxing standpoint, but on a TPU VM the device is single-tenant
    per process — the subprocess can't acquire it while the daemon
    holds it.  Running in-process sidesteps that entirely.
    """
    _build_update(key,
        phase    = "running",
        detail   = "Acquiring TPU (queued behind warmup / active prove)",
        progress = 0.0,
        elapsed  = time.time() - t_start,
    )
    _check_cancel(key)

    from framework     import Circuit, compile_circuit_cached
    from demo_circuits import CIRCUITS as _CIRCUITS

    spec = _CIRCUITS[key]
    circ = Circuit(r1cs=spec.r1cs_builder(), max_size=max_size)

    def on_progress(phase: str, frac: float, detail: str) -> None:
        _check_cancel(key)
        _build_update(key,
            phase    = "running",
            detail   = f"{phase}: {detail}",
            progress = float(frac),
            elapsed  = time.time() - t_start,
        )

    with _tpu_lock:
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        compile_circuit_cached(
            circ,
            cache_path      = pkl_path,
            force_recompile = force,
            progress        = on_progress,
        )

    if not os.path.exists(pkl_path):
        raise RuntimeError(
            f"compile_circuit_cached returned but pickle isn't on disk at {pkl_path}"
        )


def cancel_build(key: str) -> str:
    """Signal the in-flight build for ``key`` to stop at the next progress
    callback boundary.  Returns ``"cancelling"`` if a build was running,
    else ``"not_running"``.

    Because builds run in-process under the TPU lock, we can't preempt a
    running TPU kernel — cancel takes effect at the next ``progress``
    callback (phase boundary).
    """
    if key not in _cancel_flags:
        return "not_running"
    _cancel_flags[key] = True
    return "cancelling"


# ── Activation (bind/unbind a circuit's Setup onto the warm prover) ───


def activate_circuit(key: str) -> None:
    """Load ``key``'s Setup pkl and bind it to the warm prover.

    Raises:
      RuntimeError: TPU env not yet warm (``env_state().done == False``).
      FileNotFoundError: no pkl on disk for ``key``.
    """
    env = env_state()
    if not env.done:
        raise RuntimeError(
            f"TPU env not ready (phase={env.phase}, progress={env.progress:.0%})"
        )
    if env.error:
        raise RuntimeError(f"TPU env failed to warm up: {env.error[:300]}")
    cs = get_circuit_state(key)
    if cs is None:
        raise KeyError(f"circuit {key!r} not registered")
    if not cs.pkl_exists:
        raise FileNotFoundError(
            f"circuit {key!r} hasn't been built yet — call /circuits/{key}/build first"
        )

    # Serialise with /prove, any other activate calls, AND any build's
    # TPU phases (Setup.load uploads pk_tpu / qap_tpu tensors to HBM).
    with _prove_lock, _tpu_lock:
        from framework import Setup
        new_setup = Setup.load(cs.pkl_path)
        # Mark old active inactive before swapping (UI badge consistency).
        old_key = env.active_key
        if old_key is not None and old_key != key:
            _circuit_update(old_key, is_active=False)
        # Atomic swap on the prover.
        with _env_lock:
            if _env.prover is None:
                raise RuntimeError("prover instance is None — env warmup may have failed")
            _env.prover.bind_setup(new_setup)
            _env.active_key = key
        _circuit_update(key, is_active=True)


def deactivate_circuit() -> None:
    """Free the currently-bound circuit's HBM tensors.  Prover stays warm."""
    with _prove_lock, _tpu_lock:
        with _env_lock:
            if _env.prover is None or _env.active_key is None:
                return
            old_key = _env.active_key
            _env.prover.unbind_setup()
            _env.active_key = None
        _circuit_update(old_key, is_active=False)


# ── Prove (serialised against activation) ─────────────────────────────


def run_prove(circuit_key: str, *, witness: List[int], seed: Optional[int],
              verify: bool, skip_r1cs_check: bool = False):
    """Encode + prove + optionally verify against the currently-active circuit.

    ``skip_r1cs_check`` lets the caller prove a deliberately-invalid
    witness — used by the demo to show the verifier rejecting wrong
    witnesses.  See :meth:`prover_class.Prover.encode_witness`.

    Returns ``(proof, encode_ms, telemetry, verify_ms, verified)``.
    """
    env = env_state()
    if not env.done:
        raise RuntimeError(
            f"TPU env not ready (phase={env.phase}, progress={env.progress:.0%})"
        )
    with _prove_lock, _tpu_lock:
        with _env_lock:
            if _env.active_key != circuit_key:
                raise RuntimeError(
                    f"circuit {circuit_key!r} is not active "
                    f"(active={_env.active_key!r}).  POST /circuits/{circuit_key}/activate first."
                )
            prover = _env.prover
            setup  = prover.setup
        t_enc = time.time()
        encoded = prover.encode_witness(
            witness, seed=seed, skip_r1cs_check=skip_r1cs_check,
        )
        encode_ms = (time.time() - t_enc) * 1000

        proof, tele = prover.prove(encoded, record_telemetry=True)

        verify_ms = 0.0
        verified: Optional[bool] = None
        if verify:
            t_v = time.time()
            verified = setup.verify(setup.public_inputs(witness), proof)
            verify_ms = (time.time() - t_v) * 1000
    return proof, setup, encode_ms, tele, verify_ms, verified
