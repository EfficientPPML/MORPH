"""Wire-protocol dataclasses shared by the daemon and the client.

JSON-serialisable forms of:

  * :class:`StatusReply`  — output of ``GET /status``
  * :class:`CircuitInfo`  — output of ``GET /circuits``
  * :class:`ProveRequest` — input  of ``POST /prove``
  * :class:`ProveReply`   — output of ``POST /prove``

All ints (incl. proof coordinates and witness wires) are encoded as
**strings** when they cross the wire — Groth16 EC coordinates are
~377 bits, well beyond JSON's safe-integer range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── /status ────────────────────────────────────────────────────────────


@dataclass
class StatusReply:
    """Snapshot of the TPU env state.  Polled by the UI ~1× / sec.

    ``done`` is True once warmup completes — the prover then sits idle
    with no circuit bound until the user activates one via
    ``POST /circuits/{key}/activate``.  ``active`` carries the key of
    the currently-active circuit, or ``None`` when idle.
    """
    phase:      str
    progress:   float
    detail:     str
    elapsed:    float
    done:       bool
    error:      Optional[str]
    start_time: float
    active:     Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StatusReply":
        return cls(
            phase      = d["phase"],
            progress   = float(d["progress"]),
            detail     = d.get("detail", ""),
            elapsed    = float(d.get("elapsed", 0.0)),
            done       = bool(d["done"]),
            error      = d.get("error"),
            start_time = float(d.get("start_time", 0.0)),
            active     = d.get("active"),
        )


# ── /circuits ──────────────────────────────────────────────────────────


@dataclass
class BuildInfo:
    """Per-circuit build subprocess state, when present."""
    phase:      str    # "queued" | "running" | "done" | "error" | "cancelled"
    progress:   float
    detail:     str
    elapsed:    float
    start_time: float
    error:      Optional[str] = None


@dataclass
class CircuitInfo:
    """Server-side metadata + state for a registered circuit.

    Status combines disk + memory state:

      * ``available`` — no pkl on disk, no build running
      * ``building``  — prep_circuit subprocess in flight
      * ``ready``     — pkl on disk, not bound to prover
      * ``active``    — pkl on disk and currently bound to the warm prover
      * ``error``     — last build failed; pkl still missing
    """
    key:             str
    name:            str
    description:     str
    public_label:    str
    num_wires_orig:  int
    num_public_orig: int
    wire_labels:     List[str]
    # Per-wire help strings (parallel to ``wire_labels``).  Empty string
    # for wires without specific help.
    wire_helps:      List[str] = field(default_factory=list)
    # Input schema rendered as plain dicts for cross-language friendliness.
    input_schema:    List[Dict[str, Any]] = field(default_factory=list)
    # ``"practical"`` (real-world ZK use case) or ``"archive"`` (toy /
    # teaching circuit).  UI uses this to put archived circuits behind
    # an opt-in expander in the sidebar.
    category:        str = "practical"
    # One-line suggestion shown above the witness editor.
    try_it:          str = ""
    # Long-form educational markdown shown in a collapsible expander above
    # the witness editor.  Empty → no expander rendered.
    long_info:       str = ""
    # Live state — derived from pkl_exists / is_active / build on the daemon.
    status:          str = "available"
    pkl_exists:      bool = False
    is_active:       bool = False
    build:           Optional[BuildInfo] = None


# ── /prove ─────────────────────────────────────────────────────────────


@dataclass
class ProveRequest:
    """Body of ``POST /prove``."""
    circuit:  str             # circuit key
    # Either the high-level inputs ({"x": 3, ...}) — daemon runs the
    # witness_builder — OR a complete witness vector.  Exactly one of
    # ``inputs`` / ``witness`` must be present.
    inputs:   Optional[Dict[str, int]] = None
    witness:  Optional[List[int]]      = None
    seed:     Optional[int]            = None
    verify:   bool                     = True


def _ec_point_to_dict(point) -> Dict[str, str]:
    """Serialise a G1Affine / G2Affine to JSON-safe dict of int strings."""
    if point.is_infinity():
        return {"infinity": True}
    if isinstance(point.x, tuple):  # G2 Fq2
        return {
            "x0": str(point.x[0]),
            "x1": str(point.x[1]),
            "y0": str(point.y[0]),
            "y1": str(point.y[1]),
        }
    return {"x": str(point.x), "y": str(point.y)}


def proof_to_dict(proof) -> Dict[str, Any]:
    return {
        "A": _ec_point_to_dict(proof.A),
        "B": _ec_point_to_dict(proof.B),
        "C": _ec_point_to_dict(proof.C),
    }


@dataclass
class ProveReply:
    """Body of ``POST /prove`` reply."""
    proof:           Dict[str, Any]     # serialised Proof
    witness:         List[str]          # full witness as string-encoded ints
    public_inputs:   List[str]          # extracted public-input slice
    public_output:   str                # human-readable y value
    encode_ms:       float
    prove_ms:        float              # total TPU prove wall-time
    phase_ms:        Dict[str, float]   # per-stage breakdown
    verify_ms:       float              # 0 if verify=False on request
    verified:        Optional[bool]     # None if verify=False on request
    proof_bytes:     int                # serialised proof size
    witness_bytes:   int                # naive "send the whole witness" size
    seed_used:       Optional[int]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProveReply":
        return cls(
            proof         = d["proof"],
            witness       = list(d["witness"]),
            public_inputs = list(d["public_inputs"]),
            public_output = str(d["public_output"]),
            encode_ms     = float(d["encode_ms"]),
            prove_ms      = float(d["prove_ms"]),
            phase_ms      = dict(d.get("phase_ms", {})),
            verify_ms     = float(d.get("verify_ms", 0.0)),
            verified      = d.get("verified"),
            proof_bytes   = int(d["proof_bytes"]),
            witness_bytes = int(d["witness_bytes"]),
            seed_used     = d.get("seed_used"),
        )
