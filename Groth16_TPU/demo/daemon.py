"""TPU-prover daemon — owns the TPU, exposes a multi-circuit HTTP API.

Run from MAPLE root::

    python -m Groth16_TPU.demo.daemon --host 127.0.0.1 --port 8000

Architecture:

  * Daemon boots the TPU env immediately on startup (background thread)
    by AOT-compiling kernels + warming the JIT cache against a synthetic
    all-zero :class:`Setup` (see :func:`prover_class.make_warmup_setup`).
    After warmup the synthetic tensors are dropped from HBM.

  * Per circuit (cubic, power_11, fibonacci_20, range_32, …) the daemon
    tracks whether the on-disk setup pkl exists, whether a build is in
    flight, and whether it is currently active.  Only ONE circuit is
    active (bound to the prover) at a time.

  * Build subprocesses run independently of the TPU — multiple builds
    may proceed concurrently while ``/prove`` continues to serve the
    active circuit.

API
---

``GET  /healthz``                       liveness ping
``GET  /status``                        TPU env state
``GET  /circuits``                      list every circuit + per-circuit state
``POST /circuits/{key}/build``          start prep_circuit subprocess (``?force=true`` to rebuild)
``POST /circuits/{key}/cancel``         SIGTERM the running build
``POST /circuits/{key}/activate``       load pkl into the warm prover
``POST /circuits/deactivate``           free the active circuit's HBM
``POST /prove``                         body {circuit, …} — runs on active only
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import path_setup  # noqa: F401

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import init_worker
from demo_circuits import CIRCUITS, list_circuits
from protocol import proof_to_dict


_MAX_SIZE = 1024


# ── Helpers ────────────────────────────────────────────────────────────


def _cache_dir() -> str:
    d = os.path.join(_PARENT, ".dev_loop_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _circuit_info(key: str) -> Dict[str, Any]:
    """Compose static + dynamic info for one circuit."""
    spec = CIRCUITS[key]
    state = init_worker.get_circuit_state(key)
    # Defaults if registry is missing the key (shouldn't happen — lifespan registers all).
    if state is None:
        pkl_exists = os.path.exists(spec.cache_path(_cache_dir(), max_size=_MAX_SIZE))
        is_active  = False
        build      = None
    else:
        pkl_exists = state.pkl_exists
        is_active  = state.is_active
        build      = state.build

    if is_active:
        status = "active"
    elif build is not None and build.phase == "running":
        status = "building"
    elif build is not None and build.phase == "error":
        status = "error"
    elif pkl_exists:
        status = "ready"
    else:
        status = "available"

    return {
        "key":             spec.key,
        "name":            spec.name,
        "description":     spec.description,
        "public_label":    spec.public_label,
        "num_wires_orig":  spec.num_wires_orig,
        "num_public_orig": spec.r1cs_builder().num_public,
        "wire_labels":     list(spec.wire_labels),
        "wire_helps":      list(spec.wire_helps) if spec.wire_helps else
                           [""] * spec.num_wires_orig,
        # Stored category from the config file.  Falls back to the
        # CircuitSpec default if (somehow) the file lacks this key.
        "category":        init_worker.get_category(spec.key) or spec.category,
        "try_it":          spec.try_it,
        "long_info":       spec.long_info,
        "input_schema":    [
            {
                "name":    f.name,
                "label":   f.label,
                "kind":    f.kind,
                "default": f.default,
                "min":     f.min,
                "max":     f.max,
                "help":    f.help,
            }
            for f in spec.input_schema
        ],
        "status":      status,
        "pkl_exists":  pkl_exists,
        "is_active":   is_active,
        "build":       (None if build is None else {
            "phase":      build.phase,
            "progress":   build.progress,
            "detail":     build.detail,
            "elapsed":    build.elapsed,
            "start_time": build.start_time,
            "error":      build.error,
        }),
    }


# ── FastAPI app ────────────────────────────────────────────────────────


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Register every circuit, then kick off TPU env warmup."""
    cache = _cache_dir()
    # Categories config — lives next to demo/ for visibility (not in the
    # opaque .dev_loop_cache).  Lists EVERY registered circuit + its
    # current archive/practical state; hand-editable.
    categories_path = os.path.join(_HERE, "circuit_categories.json")
    defaults = {spec.key: spec.category for spec in list_circuits()}
    init_worker.init_categories(categories_path, defaults)
    for spec in list_circuits():
        init_worker.register_circuit(
            spec.key,
            spec.cache_path(cache, max_size=_MAX_SIZE),
        )
    init_worker.start_env(max_size=_MAX_SIZE)
    yield


app = FastAPI(
    title       = "MAPLE — TPU Groth16 Prover Daemon",
    description = "Multi-circuit HTTP wrapper over the TPU prover.",
    version     = "0.2",
    lifespan    = _lifespan,
)

# Permissive CORS so the Streamlit dev server can call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request bodies ─────────────────────────────────────────────────────


class ProveBody(BaseModel):
    circuit:  str
    inputs:   Dict[str, int]  = None
    witness:  list[int]       = None
    seed:     int | None      = None
    verify:   bool            = True
    # Demo-mode: prove an *invalid* witness anyway, expecting the
    # verifier to reject it.  Default False — encode_witness raises
    # ValueError → 400 if the witness doesn't satisfy the R1CS.
    skip_r1cs_check: bool     = False


# ── Handlers ───────────────────────────────────────────────────────────


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> Dict[str, Any]:
    s = init_worker.env_state()
    return {
        "phase":      s.phase,
        "progress":   s.progress,
        "detail":     s.detail,
        "elapsed":    s.elapsed,
        "done":       s.done,
        "error":      s.error,
        "start_time": s.start_time,
        "active":     s.active_key,
    }


@app.get("/circuits")
def circuits() -> Dict[str, Any]:
    env = init_worker.env_state()
    return {
        "circuits":   [_circuit_info(spec.key) for spec in list_circuits()],
        "active":     env.active_key,
        "env_ready":  env.done and (env.error is None),
    }


@app.post("/circuits/{key}/build")
def build_circuit(key: str, force: bool = False) -> Dict[str, Any]:
    if key not in CIRCUITS:
        raise HTTPException(status_code=404, detail=f"unknown circuit {key!r}")
    try:
        outcome = init_worker.start_build(key, max_size=_MAX_SIZE, force=force)
    except KeyError as ex:
        raise HTTPException(status_code=404, detail=str(ex)) from ex
    return {"key": key, "outcome": outcome}


@app.post("/circuits/{key}/cancel")
def cancel_circuit_build(key: str) -> Dict[str, Any]:
    if key not in CIRCUITS:
        raise HTTPException(status_code=404, detail=f"unknown circuit {key!r}")
    return {"key": key, "outcome": init_worker.cancel_build(key)}


@app.post("/circuits/{key}/activate")
def activate_circuit(key: str) -> Dict[str, Any]:
    if key not in CIRCUITS:
        raise HTTPException(status_code=404, detail=f"unknown circuit {key!r}")
    try:
        init_worker.activate_circuit(key)
    except FileNotFoundError as ex:
        raise HTTPException(status_code=409, detail=str(ex)) from ex
    except RuntimeError as ex:
        raise HTTPException(status_code=503, detail=str(ex)) from ex
    return {"key": key, "outcome": "active"}


@app.post("/circuits/deactivate")
def deactivate_active() -> Dict[str, Any]:
    init_worker.deactivate_circuit()
    return {"outcome": "deactivated"}


class _CategoryBody(BaseModel):
    category: str


@app.post("/circuits/{key}/category")
def set_circuit_category(key: str, body: _CategoryBody) -> Dict[str, Any]:
    """Move a circuit between ``practical`` and ``archive``.  Rewrites
    the full ``demo/circuit_categories.json`` config file."""
    if key not in CIRCUITS:
        raise HTTPException(status_code=404, detail=f"unknown circuit {key!r}")
    try:
        init_worker.set_category(key, body.category)
    except (ValueError, KeyError) as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    return {"key": key, "category": body.category}


@app.post("/prove")
def prove(req: ProveBody) -> Dict[str, Any]:
    if req.circuit not in CIRCUITS:
        raise HTTPException(status_code=404,
                            detail=f"unknown circuit {req.circuit!r}")
    spec = CIRCUITS[req.circuit]
    try:
        # Build the witness either from inputs or the explicit vector.
        if req.witness is not None:
            witness = [int(w) for w in req.witness]
        else:
            inputs = req.inputs or {}
            witness = list(spec.witness_builder(**inputs))

        proof, setup, encode_ms, tele, verify_ms, verified = init_worker.run_prove(
            req.circuit, witness=witness, seed=req.seed, verify=req.verify,
            skip_r1cs_check=req.skip_r1cs_check,
        )

        proof_dict = proof_to_dict(proof)
        proof_bytes = 192          # 48 + 96 + 48 — affine-compressed G1/G2/G1.
        witness_bytes = len(witness) * 32

        return {
            "proof":         proof_dict,
            "witness":       [str(w) for w in witness],
            "public_inputs": [str(p) for p in setup.public_inputs(witness)],
            "public_output": str(spec.public_output(witness)),
            "encode_ms":     encode_ms,
            "prove_ms":      tele.total_ms,
            "phase_ms":      tele.to_dict(),
            "verify_ms":     verify_ms,
            "verified":      verified,
            "proof_bytes":   proof_bytes,
            "witness_bytes": witness_bytes,
            "seed_used":     req.seed,
        }
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    except RuntimeError as ex:
        # Active-circuit mismatch / env not ready / etc.
        raise HTTPException(status_code=503, detail=str(ex)) from ex
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── CLI ────────────────────────────────────────────────────────────────


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default 127.0.0.1; use 0.0.0.0 to expose)")
    parser.add_argument("--port", type=int, default=8000,
                        help="port (default 8000)")
    args = parser.parse_args()

    print(f"[daemon] starting on http://{args.host}:{args.port}", flush=True)
    print(f"[daemon] registered circuits: {list(CIRCUITS)}",       flush=True)
    print(f"[daemon] try: curl http://{args.host}:{args.port}/healthz",
          flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    _main()
