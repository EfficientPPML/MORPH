"""Build one circuit's Setup pickle from scratch — designed to be
spawned as a subprocess by the demo daemon.

Why a separate process: the heavy CPU phases of ``compile_circuit``
(in particular ``precompute_pk`` / ``precompute_qap``) fan out via
``ProcessPoolExecutor(mp_context=forkserver)``.  Forkserver subprocesses
import the parent's ``__main__`` module before running their assigned
worker.  If the parent is the daemon (which holds the TPU), those
subprocesses crash trying to acquire the TPU again.

By running setup in its own fresh Python process — which has
``__name__ == "__main__"`` set to *this* script and a proper main-block
guard — the forkserver children import a small module that does
nothing at top level, and just run their DRNS worker.  No TPU
contention.

Usage::

    python -m Groth16_TPU.demo.prep_circuit \\
        --circuit cubic \\
        --cache-path /home/user/work/MAPLE/Groth16_TPU/.dev_loop_cache/cubic_1024_setup.pkl \\
        --max-size 1024

Phase events go to stdout, one per line, JSON-encoded, so a parent
process can stream them as progress::

    {"phase": "r1cs_pad",      "frac": 0.00, "detail": "Padding ..."}
    {"phase": "qap_build",     "frac": 0.10, "detail": "Building QAP via batched INTT"}
    {"phase": "trusted_setup", "frac": 0.30, "detail": "Sampling α/β/γ/δ/τ; ..."}
    {"phase": "encode_pk",     "frac": 0.70, "detail": "DRNS-encoding pk_tpu"}
    {"phase": "encode_qap",    "frac": 0.85, "detail": "DRNS-encoding qap_tpu (U/V/W polynomials)"}
    {"phase": "save_pkl",      "frac": 0.98, "detail": "Writing cubic_1024_setup.pkl"}
    {"phase": "done",          "frac": 1.00, "detail": "Setup cached to disk"}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _setup_paths() -> None:
    here   = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    for p in (here, parent):
        if p not in sys.path:
            sys.path.insert(0, p)


def _main() -> int:
    _setup_paths()
    import path_setup  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit",    required=True,
                        help="circuit key from demo_circuits.CIRCUITS")
    parser.add_argument("--cache-path", required=True,
                        help="output .pkl path")
    parser.add_argument("--max-size",   type=int, default=1024,
                        help="padded size (default 1024)")
    parser.add_argument("--force",      action="store_true",
                        help="rebuild even if the cache file already exists")
    args = parser.parse_args()

    from framework      import Circuit, compile_circuit_cached
    from demo_circuits  import CIRCUITS

    if args.circuit not in CIRCUITS:
        print(f"unknown circuit {args.circuit!r}; "
              f"have {list(CIRCUITS)}", file=sys.stderr)
        return 2

    spec = CIRCUITS[args.circuit]
    circ = Circuit(r1cs=spec.r1cs_builder(), max_size=args.max_size)

    if (not args.force) and os.path.exists(args.cache_path):
        print(json.dumps({
            "phase":  "cache_hit",
            "frac":   1.0,
            "detail": f"Cache file already exists at {args.cache_path}",
        }), flush=True)
        return 0

    os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)

    def on_progress(phase: str, frac: float, detail: str) -> None:
        print(json.dumps({
            "phase":  phase,
            "frac":   frac,
            "detail": detail,
        }), flush=True)

    t0 = time.time()
    compile_circuit_cached(
        circ,
        cache_path      = args.cache_path,
        force_recompile = bool(args.force),
        progress        = on_progress,
    )
    print(json.dumps({
        "phase":   "done",
        "frac":    1.0,
        "detail":  f"Setup ready ({time.time() - t0:.1f}s)",
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
