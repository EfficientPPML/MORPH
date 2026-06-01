"""Thin HTTP client for the TPU prover daemon.

UIs (Streamlit, CLI, etc.) talk to the daemon through this module
instead of importing :mod:`prover_class` directly.  No JAX, no TPU,
no multiprocessing-vs-Streamlit conflicts — pure ``requests``.

Usage::

    from client import DaemonClient

    cli = DaemonClient("http://127.0.0.1:8000")
    while not cli.status().done:
        time.sleep(1)

    reply = cli.prove(circuit="cubic", inputs={"x": 3}, seed=42)
    print(reply.public_output, reply.prove_ms, reply.verified)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import requests

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from protocol import StatusReply, ProveReply, CircuitInfo, BuildInfo


class DaemonError(RuntimeError):
    """Raised when the daemon returns a non-2xx status code."""


class DaemonClient:
    """HTTP wrapper around the prover daemon.

    All methods are synchronous.  Connection failures and 4xx/5xx
    replies surface as :class:`DaemonError`.
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    # ── plumbing ──────────────────────────────────────────────────────

    def _get(self, path: str) -> Any:
        try:
            r = requests.get(self.base_url + path, timeout=self.timeout)
        except requests.RequestException as ex:
            raise DaemonError(f"GET {path}: {ex}") from ex
        if not r.ok:
            raise DaemonError(f"GET {path}: {r.status_code} {r.text[:300]}")
        return r.json()

    def _post(self, path: str, body: Dict[str, Any]) -> Any:
        try:
            r = requests.post(self.base_url + path, json=body, timeout=self.timeout)
        except requests.RequestException as ex:
            raise DaemonError(f"POST {path}: {ex}") from ex
        if not r.ok:
            try:
                err_detail = r.json().get("detail", r.text)
            except Exception:
                err_detail = r.text
            raise DaemonError(
                f"POST {path}: {r.status_code} — {str(err_detail)[:300]}"
            )
        return r.json()

    # ── public API ────────────────────────────────────────────────────

    def healthz(self) -> bool:
        """Quick liveness probe — returns True if the daemon is reachable."""
        try:
            r = requests.get(self.base_url + "/healthz", timeout=2.0)
            return r.ok
        except requests.RequestException:
            return False

    def status(self) -> StatusReply:
        """Current :class:`StatusReply` (boot phase, progress, etc.)."""
        return StatusReply.from_dict(self._get("/status"))

    def circuits(self) -> tuple[List[CircuitInfo], Optional[str]]:
        """Return (registered_circuits, active_circuit_key).

        Each :class:`CircuitInfo` is enriched with live state — see
        :attr:`CircuitInfo.status` for ``available | building | ready |
        active | error`` and :attr:`CircuitInfo.build` for the
        prep_circuit subprocess progress when in flight.
        """
        data = self._get("/circuits")
        circuits = [self._decode_circuit(c) for c in data["circuits"]]
        return circuits, data.get("active")

    @staticmethod
    def _decode_circuit(c: Dict[str, Any]) -> CircuitInfo:
        b = c.get("build")
        return CircuitInfo(
            key             = c["key"],
            name            = c["name"],
            description     = c["description"],
            public_label    = c["public_label"],
            num_wires_orig  = c["num_wires_orig"],
            num_public_orig = c["num_public_orig"],
            wire_labels     = c["wire_labels"],
            wire_helps      = c.get("wire_helps", [""] * c["num_wires_orig"]),
            input_schema    = c["input_schema"],
            category        = c.get("category", "practical"),
            try_it          = c.get("try_it", ""),
            long_info       = c.get("long_info", ""),
            status          = c.get("status", "available"),
            pkl_exists      = bool(c.get("pkl_exists", False)),
            is_active       = bool(c.get("is_active", False)),
            build           = (None if b is None else BuildInfo(
                phase      = b["phase"],
                progress   = float(b.get("progress", 0.0)),
                detail     = b.get("detail", ""),
                elapsed    = float(b.get("elapsed", 0.0)),
                start_time = float(b.get("start_time", 0.0)),
                error      = b.get("error"),
            )),
        )

    # ── circuit lifecycle ────────────────────────────────────────────

    def build(self, key: str, force: bool = False) -> Dict[str, Any]:
        """Start ``prep_circuit`` for ``key``.  Idempotent.

        ``force=True`` rebuilds even if a pickle already exists, overwriting
        the cached setup (used by the UI's *Rebuild* button).
        """
        suffix = "?force=true" if force else ""
        return self._post(f"/circuits/{key}/build{suffix}", {})

    def cancel_build(self, key: str) -> Dict[str, Any]:
        """SIGTERM the in-flight build for ``key`` (if any)."""
        return self._post(f"/circuits/{key}/cancel", {})

    def activate(self, key: str) -> Dict[str, Any]:
        """Bind ``key``'s setup into the warm prover (drops the prior active)."""
        return self._post(f"/circuits/{key}/activate", {})

    def deactivate(self) -> Dict[str, Any]:
        """Free the active circuit's HBM tensors.  Prover stays warm."""
        return self._post("/circuits/deactivate", {})

    def set_category(self, key: str, category: str) -> Dict[str, Any]:
        """Move ``key`` between ``"practical"`` and ``"archive"``.  Persisted."""
        return self._post(f"/circuits/{key}/category", {"category": category})

    def prove(
        self,
        circuit:  str,
        inputs:   Optional[Dict[str, int]] = None,
        witness:  Optional[List[int]]      = None,
        seed:     Optional[int]            = None,
        verify:   bool                     = True,
        timeout:  Optional[float]          = None,
        skip_r1cs_check: bool              = False,
    ) -> ProveReply:
        """Run a proof on the daemon.

        Exactly one of ``inputs`` / ``witness`` must be provided.  The
        daemon's witness_builder fills in intermediate wires from
        ``inputs``; ``witness`` is for the advanced path where the UI
        already has a hand-edited witness vector.

        ``skip_r1cs_check=True`` lets the prover encode an invalid
        witness anyway — used by the demo to show the verifier rejecting
        wrong witnesses.  Default ``False`` (400 on invalid witness).
        """
        if (inputs is None) == (witness is None):
            raise ValueError("provide exactly one of inputs or witness")
        body = {
            "circuit": circuit,
            "seed":    seed,
            "verify":  verify,
            "skip_r1cs_check": skip_r1cs_check,
        }
        if inputs is not None:
            body["inputs"] = inputs
        else:
            body["witness"] = list(witness)
        # Allow a per-call override of the connection timeout — prove
        # warm is ~700 ms but a cold first-prove can be ~12 min.
        save_timeout, self.timeout = self.timeout, (timeout or self.timeout)
        try:
            data = self._post("/prove", body)
        finally:
            self.timeout = save_timeout
        return ProveReply.from_dict(data)
