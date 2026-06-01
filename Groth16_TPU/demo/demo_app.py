"""Streamlit GUI for the TPU Groth16 demo — talks to the prover daemon.

UI process only — no JAX, no TPU.  All heavy work lives in the daemon
(see :mod:`daemon`), which the UI calls over HTTP.  See ``./demo.sh
start`` for the one-shot wrapper that boots both processes.

Layout:

  * **Sidebar** — list every circuit with a live status badge + per-row
    Build / Stop / Use buttons.  TPU env status sits above the list.
  * **Main**    — for the selected circuit: witness editor (works during
    boot, no daemon round-trip), Check button, and a Generate Proof
    button that's enabled only when the selected circuit is also the
    active one (bound to the prover on the daemon side).
"""

from __future__ import annotations

import os
import sys
import time as _time

import streamlit as st

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
import path_setup  # noqa: F401

from client import DaemonClient, DaemonError
from demo_circuits import CIRCUITS as _LOCAL_CIRCUITS
from groth16.r1cs  import verify_witness
from circuit_pad   import pad_r1cs, pad_witness


# ── Page setup ─────────────────────────────────────────────────────────


_DAEMON_URL_DEFAULT = os.environ.get("MAPLE_DAEMON_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title = "ZKP on TPU",
    page_icon  = "🔐",
    layout     = "wide",
)
st.title("🔐 ZKP on TPU — Groth16 Demo")
st.caption("BLS12-377 · Fused-MSM · Projective-RCB G2 · TPU v4+")

# Keep the per-circuit action buttons (Build / Rebuild / Stop / Use) on a
# single line.  In the 3-column sidebar layout the longer "Rebuild" label
# wraps to two lines by default; nowrap + tighter padding makes it fit.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] .stButton button {
        white-space: nowrap;
        padding-left: 0.25rem;
        padding-right: 0.25rem;
    }
    section[data-testid="stSidebar"] .stButton button p {
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Daemon connection ──────────────────────────────────────────────────


with st.sidebar:
    st.header("Daemon")
    daemon_url = st.text_input(
        "URL",
        value = _DAEMON_URL_DEFAULT,
        help  = "Where the TPU prover daemon listens.  Default: localhost:8000",
    )
client = DaemonClient(daemon_url)

if not client.healthz():
    st.error(
        f"### Can't reach prover daemon at `{daemon_url}`\n\n"
        "Start it with:\n\n"
        "```bash\n./Groth16_TPU/demo/demo.sh start\n```\n"
    )
    st.stop()


# ── Helpers ────────────────────────────────────────────────────────────


_STATUS_ICON = {
    "available": "⚪",
    "building":  "🔵",
    "ready":     "🟢",
    "active":    "⭐",
    "error":     "🔴",
}

_STATUS_LABEL = {
    "available": "available",
    "building":  "building",
    "ready":     "ready",
    "active":    "active",
    "error":     "error",
}


def _do_build(key: str, force: bool = False) -> None:
    try:
        client.build(key, force=force)
    except DaemonError as ex:
        verb = "rebuild" if force else "build"
        st.session_state["_circuit_op_error"] = f"{verb}({key}): {ex}"


def _do_cancel(key: str) -> None:
    try:
        client.cancel_build(key)
    except DaemonError as ex:
        st.session_state["_circuit_op_error"] = f"cancel({key}): {ex}"


def _do_activate(key: str) -> None:
    try:
        client.activate(key)
        st.session_state["selected_key"] = key
    except DaemonError as ex:
        st.session_state["_circuit_op_error"] = f"activate({key}): {ex}"


def _do_select(key: str) -> None:
    st.session_state["selected_key"] = key


def _do_set_category(key: str, category: str) -> None:
    try:
        client.set_category(key, category)
    except DaemonError as ex:
        st.session_state["_circuit_op_error"] = f"category({key}): {ex}"


# ── Sidebar: env status + circuit list (auto-refreshing) ──────────────


def _render_env_panel(env):
    """Top-of-sidebar TPU env status (warmup progress / ready badge)."""
    if env.error:
        st.error(f"❌ TPU env failed:\n```\n{env.error[:1200]}\n```")
    elif env.done:
        st.success(
            f"✓ TPU env warm  ({env.elapsed:.1f}s)" +
            (f"\n\nActive: **{env.active}**" if env.active else
             "\n\nNo circuit active — Use one below."),
            icon = "🟢",
        )
    else:
        live = (_time.time() - env.start_time) if env.start_time else env.elapsed
        st.progress(
            env.progress,
            text = f"TPU env: {env.phase} · {live:.1f}s",
        )
        st.caption(f"↳ {env.detail or '...'}")
        st.caption(
            "**While you wait** — pick a circuit below and click **Build** "
            "to queue its trusted-setup pickle.  Builds run on CPU, "
            "independently of the TPU warmup."
        )


def _render_circuit_card(ci, env_done: bool) -> None:
    """One row in the circuit list.  Buttons live in normal script flow
    (NOT inside a fragment) so clicks trigger full-app reruns — that's
    what makes the main panel re-render when the user clicks Use."""
    with st.container(border=True):
        icon  = _STATUS_ICON.get(ci.status, "•")
        label = _STATUS_LABEL.get(ci.status, ci.status)
        st.markdown(f"{icon} **{ci.name}**  \n*{label}*")

        # Live build progress (snapshot at this app rerun; heartbeat
        # fragment below will trigger reruns while a build runs).
        if ci.build and ci.build.phase == "running":
            st.progress(
                max(0.0, min(1.0, ci.build.progress)),
                text = f"{ci.build.detail or 'building'}  ({ci.build.elapsed:.1f}s)",
            )
        elif ci.build and ci.build.phase == "error":
            st.caption(f"⚠ build failed: `{(ci.build.error or '')[:160]}`")

        btn_cols = st.columns(3, gap="small")
        with btn_cols[0]:
            if ci.status == "available":
                if st.button("Build", key=f"btn_build_{ci.key}",
                             use_container_width=True):
                    _do_build(ci.key)
                    st.rerun()
            elif ci.status in ("ready", "active", "error"):
                # Already built (or a prior build errored) — let the user
                # force a fresh trusted setup, overwriting the cached pkl.
                if st.button("Rebuild", key=f"btn_rebuild_{ci.key}",
                             use_container_width=True):
                    _do_build(ci.key, force=True)
                    st.rerun()
            else:  # building
                st.button("Build", key=f"btn_build_{ci.key}",
                          disabled=True, use_container_width=True)

        with btn_cols[1]:
            if ci.status == "building":
                if st.button("Stop", key=f"btn_stop_{ci.key}",
                             use_container_width=True):
                    _do_cancel(ci.key)
                    st.rerun()
            else:
                st.button("Stop", key=f"btn_stop_{ci.key}",
                          disabled=True, use_container_width=True)

        with btn_cols[2]:
            if ci.status == "active":
                st.button("Active", key=f"btn_use_{ci.key}",
                          use_container_width=True, disabled=True,
                          type="primary")
            elif ci.status == "ready":
                if st.button("Use", key=f"btn_use_{ci.key}",
                             use_container_width=True,
                             disabled=not env_done):
                    _do_activate(ci.key)
                    st.rerun()
            else:
                st.button("Use", key=f"btn_use_{ci.key}",
                          disabled=True, use_container_width=True)

        # "Show in main panel" — local UI navigation, independent of
        # activation.  Lets the user explore witnesses for unbuilt circuits.
        if st.session_state.get("selected_key") != ci.key:
            if st.button("Show in main panel ⤴", key=f"btn_show_{ci.key}",
                         use_container_width=True):
                _do_select(ci.key)
                st.rerun()
        else:
            st.caption("📌 showing in main panel")

        # Category toggle — small text-style button at the bottom right.
        # Move circuits in/out of the Archived section without restart.
        if ci.category == "archive":
            move_label = "↑ Move to practical"
            move_to    = "practical"
        else:
            move_label = "↓ Move to archive"
            move_to    = "archive"
        if st.button(move_label, key=f"btn_cat_{ci.key}",
                     use_container_width=True):
            _do_set_category(ci.key, move_to)
            st.rerun()


@st.fragment(run_every=1.0)
def _heartbeat():
    """Tiny invisible polling fragment.

    The buttons + circuit cards live in NORMAL script flow above so
    clicks naturally trigger full-app reruns (the canonical Streamlit
    pattern).  This fragment only exists to refresh the display while
    a long-running operation (env warmup, circuit build) is in flight.

    Strategy: snapshot env + per-circuit build state on each tick.  If
    the snapshot differs from the previous tick — phase changed,
    progress moved, a build finished, env warmup completed — trigger
    a full-app rerun via ``st.rerun(scope="app")`` so the sidebar
    re-renders.  When nothing is happening, no rerun fires.
    """
    try:
        env = client.status()
        circuits, _active = client.circuits()
    except DaemonError:
        return

    sig_parts = [
        env.phase, f"{env.progress:.2f}", str(env.done),
        env.active or "",
    ]
    for c in circuits:
        sig_parts.append(c.key)
        sig_parts.append(c.status)
        b = c.build
        if b:
            sig_parts.append(b.phase)
            sig_parts.append(f"{b.progress:.2f}")
        else:
            sig_parts.append("nobuild")
    snapshot = "|".join(sig_parts)

    last = st.session_state.get("_sidebar_snapshot")
    st.session_state["_sidebar_snapshot"] = snapshot
    # Trigger a full-app rerun if anything changed AND something is
    # actively in flight (env warming / build running).  Avoids the
    # rerun storm when the system is idle.
    something_changing = (not env.done) or any(
        c.build and c.build.phase == "running" for c in circuits
    )
    if last is not None and last != snapshot and something_changing:
        st.rerun(scope="app")


with st.sidebar:
    # Snapshot env + circuits ONCE per full app rerun.  Buttons rendered
    # below run in normal script flow → clicks trigger full reruns.
    try:
        env = client.status()
        circuits, active_key = client.circuits()
    except DaemonError as ex:
        st.error(f"Lost connection to daemon: {ex}")
        st.stop()

    _render_env_panel(env)
    st.divider()
    st.header("Circuits")

    err = st.session_state.pop("_circuit_op_error", None)
    if err:
        st.error(err)

    practical = [c for c in circuits if c.category == "practical"]
    archived  = [c for c in circuits if c.category == "archive"]

    for ci in practical:
        _render_circuit_card(ci, env_done=env.done)

    # Archived (toy / teaching) circuits live behind an expander.  User
    # ticks individual ones to surface them in the live list above —
    # but for simplicity v1 just renders every archived circuit when
    # the expander is open.
    if archived:
        with st.expander(f"▸ Archived demos ({len(archived)})", expanded=False):
            st.caption(
                "Toy circuits kept for reference — useful for sanity checks "
                "but not real-world ZK patterns."
            )
            for ci in archived:
                _render_circuit_card(ci, env_done=env.done)

    # Polling heartbeat — keeps the UI fresh during builds / warmup,
    # but doesn't own any of the interactive widgets above.
    _heartbeat()

    st.divider()
    st.header("Options")
    deterministic = st.checkbox(
        "Deterministic blinders (seed = 42)",
        value = True,
        help  = "When on, the proof is reproducible byte-for-byte.  Turn off "
                "in production — Groth16's ZK property needs cryptographic "
                "randomness.",
    )
seed = 42 if deterministic else None


# ── Pick the circuit shown in the main panel ──────────────────────────
# ``env``, ``circuits``, ``active_key`` are already in scope from the
# sidebar block above (Python ``with`` doesn't open a new scope), so we
# reuse the same snapshot — no second round-trip to the daemon.

# Default selection: active circuit > first ready > first registered.
if "selected_key" not in st.session_state:
    if active_key:
        st.session_state["selected_key"] = active_key
    else:
        ready = [c for c in circuits if c.pkl_exists]
        st.session_state["selected_key"] = (
            ready[0].key if ready else (circuits[0].key if circuits else None)
        )

selected_key = st.session_state["selected_key"]
spec = next((c for c in circuits if c.key == selected_key), None)
if spec is None:
    st.warning("No circuit selected.  Pick one in the sidebar.")
    st.stop()


# ── Main: header banner ────────────────────────────────────────────────


st.header(f"📐 {spec.name}")
st.markdown(spec.description)
st.caption(
    f"max wires (padded): 1024  ·  num_public: {spec.num_public_orig}  ·  "
    f"unpadded wires: {spec.num_wires_orig}"
)

if spec.key != active_key:
    if spec.status == "ready":
        st.info(
            f"This circuit is **ready on disk** but not active.  "
            f"Click **Use** in the sidebar to bind it to the TPU prover "
            f"(takes ~5–10 s).  Until then you can still play with the "
            f"witness — `Generate proof` will activate this circuit before running.",
            icon = "ℹ️",
        )
    elif spec.status == "building":
        st.warning(
            f"Setup pickle is **still being built**.  You can edit the "
            f"witness below; once the build finishes, click **Use** in the "
            f"sidebar to activate.",
            icon = "⏳",
        )
    elif spec.status == "available":
        st.warning(
            f"This circuit hasn't been built yet.  Click **Build** in the "
            f"sidebar to compile its trusted-setup pickle (~135 s CPU).",
            icon = "⚠️",
        )
    elif spec.status == "error":
        st.error(
            f"Last build failed for this circuit.  Click **Rebuild** in the sidebar.",
            icon = "🔴",
        )


# ── Per-circuit "how this works" panel ────────────────────────────────


if spec.long_info:
    with st.expander("🧠 How this circuit works", expanded=False):
        st.markdown(spec.long_info)


# ── ZKP intro ──────────────────────────────────────────────────────────


with st.expander("ℹ️ What is a zero-knowledge proof?"):
    st.markdown("""
**Groth16** lets a *prover* convince a *verifier* of a statement
("I know an `x` such that `f(x) = y`") **without revealing `x`**.

* The prover does all the work — heavy MSMs, NTTs, EC math — on the
  TPU here.
* The verifier just runs three pairings on the CPU (~2.4 s, always).
* The proof itself is **192 bytes**, *regardless of how big the
  computation was*.

That's the magic.  A computation involving thousands of constraints
collapses to a constant-size proof that the verifier can check in
milliseconds without re-doing the work — and without learning the
private witness.
    """)


# ── Witness editor (local, works during boot / before activation) ─────


st.header("Witness")
if spec.try_it:
    st.info(f"💡 **Try it:** {spec.try_it}", icon="💡")
st.caption(
    "Full assignment of values to every R1CS wire.  Wire 0 is always "
    "the constant 1.  The prover pads the remaining slots up to 1024 "
    "with zeros.  Hover the **?** on any wire for what it represents "
    "and what happens if you change it."
)


def _wire_key(i: int) -> str:
    return f"wire_{selected_key}_{i}"


# Witness values live in session_state as **decimal-string** so big
# integers (250-bit field elements in MiMC / Poseidon / Merkle wires)
# survive the JS round-trip — st.number_input uses JS float64 internally
# and loses precision past 2^53.  st.text_input keeps the full string.
for _i in range(spec.num_wires_orig):
    _k = _wire_key(_i)
    if _k not in st.session_state:
        st.session_state[_k] = "1" if _i == 0 else "0"


col_inputs, col_wires = st.columns([1, 2], gap="large")


with col_inputs:
    if selected_key == "sudoku_4x4":
        # ── Sudoku-specific 4×4 grid input ───────────────────────────
        st.subheader("Sudoku 4×4 grid")
        st.caption(
            "Each cell ∈ {1, 2, 3, 4}.  Verifier learns only that "
            "the predicate holds, never the values."
        )
        for row in range(4):
            cols = st.columns(4, gap="small")
            for col in range(4):
                i = row * 4 + col
                fld = spec.input_schema[i]
                with cols[col]:
                    st.number_input(
                        label = " ",                # collapse label to save space
                        value = int(fld["default"]),
                        min_value = 1,
                        max_value = 4,
                        step = 1,
                        key = f"input_{selected_key}_{fld['name']}",
                        label_visibility = "collapsed",
                        help = f"row {row}, column {col}",
                    )
    else:
        # ── Generic high-level inputs (other circuits) ──────────────
        st.subheader("High-level inputs")
        for fld in spec.input_schema:
            kwargs = dict(
                label = fld["label"],
                value = fld["default"],
                help  = fld.get("help") or None,
                key   = f"input_{selected_key}_{fld['name']}",
            )
            if fld["kind"] == "int":
                kwargs["step"] = 1
                if fld.get("min") is not None:
                    kwargs["min_value"] = fld["min"]
                if fld.get("max") is not None:
                    kwargs["max_value"] = fld["max"]
                st.number_input(**kwargs)
            elif fld["kind"] == "text":
                st.text_input(**kwargs)

    def _do_autofill():
        inputs = {
            f["name"]: st.session_state[f"input_{selected_key}_{f['name']}"]
            for f in spec.input_schema
        }
        try:
            local_spec  = _LOCAL_CIRCUITS[selected_key]
            new_witness = list(local_spec.witness_builder(**inputs))[
                : spec.num_wires_orig
            ]
            for i, val in enumerate(new_witness):
                # Decimal string — preserves precision for 250-bit values.
                st.session_state[_wire_key(i)] = str(int(val))
            st.session_state.pop("_last_check",      None)
            st.session_state.pop("_autofill_error",  None)
        except Exception as ex:
            st.session_state["_autofill_error"] = repr(ex)

    st.button(
        "⟳ Autofill witness from inputs above",
        use_container_width = True,
        on_click            = _do_autofill,
    )
    if "_autofill_error" in st.session_state:
        st.error(f"Autofill failed: {st.session_state['_autofill_error']}")

    st.caption(
        "Click **Autofill** to compute the witness from the high-level "
        "inputs (pure CPython, works during boot)."
    )

with col_wires:
    st.subheader(f"Witness vector  ({spec.num_wires_orig} wires)")
    # Pull wire helps from the CircuitInfo (one per wire, empty string if
    # no help was specified).  Hovering on the **?** beside each input
    # reveals what the wire represents.
    wire_helps_local = list(spec.wire_helps) if spec.wire_helps else []
    while len(wire_helps_local) < spec.num_wires_orig:
        wire_helps_local.append("")

    # Collapse long witness lists behind a scrolling container so the
    # main panel stays usable for big circuits (Sudoku has 82 wires,
    # Merkle has 126).
    use_scroll = spec.num_wires_orig > 12
    wire_container = (
        st.container(height=500, border=True) if use_scroll else st.container()
    )

    witness: list[int] = []
    parse_errors: list[tuple[int, str]] = []
    with wire_container:
        for i, lbl in enumerate(spec.wire_labels):
            wire_help = wire_helps_local[i] or None
            if i == 0:
                st.text_input(
                    lbl, value="1", disabled=True,
                    key=f"wire_locked_{selected_key}",
                    help=wire_help,
                )
                witness.append(1)
            else:
                val_str = st.text_input(
                    lbl, key=_wire_key(i), help=wire_help,
                )
                try:
                    witness.append(int(val_str or "0"))
                except ValueError:
                    parse_errors.append((i, val_str))
                    witness.append(0)
    if parse_errors:
        st.error(
            "These wires aren't valid integers; treated as 0 for now:\n"
            + "\n".join(f"  • wire {i}: `{v!r}`" for i, v in parse_errors[:5])
        )


# ── Manual R1CS check (local — works during boot) ─────────────────────


@st.cache_resource
def _local_padded_r1cs(circuit_key: str):
    raw = _LOCAL_CIRCUITS[circuit_key].r1cs_builder()
    return pad_r1cs(raw, target_num_constraints=1024, target_num_wires=1024)


col_check, col_check_status = st.columns([1, 3], vertical_alignment="center")
with col_check:
    check_clicked = st.button(
        "✓ Check witness",
        use_container_width = True,
        help = "Validate the current witness against the (padded) R1CS.  "
               "Runs locally — works while the daemon is still booting.",
    )

if check_clicked:
    try:
        padded   = _local_padded_r1cs(selected_key)
        padded_w = pad_witness(witness, target_num_wires=1024)
        ok       = verify_witness(padded, padded_w)
        st.session_state["_last_check"] = {
            "witness":  list(witness),
            "circuit":  selected_key,
            "is_valid": bool(ok),
        }
    except Exception as ex:
        st.error(f"Check failed: {ex!r}")

_last_check = st.session_state.get("_last_check")
# Three states:
#   • no current check  →  must click Check first
#   • current check passed  →  Generate proof enabled (witness_valid=True)
#   • current check failed  →  Generate proof gated behind an explicit
#                              "prove the wrong witness anyway" checkbox
#                              (demo: shows the verifier returning false)
witness_check_current = (
    _last_check is not None
    and _last_check.get("circuit") == selected_key
    and _last_check["witness"] == list(witness)
)
witness_valid = bool(witness_check_current and _last_check["is_valid"])

if not witness_check_current:
    with col_check_status:
        if _last_check is None or _last_check.get("circuit") != selected_key:
            st.info("ℹ️  No check yet — click **Check witness** to validate.")
        else:
            st.warning(
                "⚠️  Witness has changed since the last check.  Click "
                "**Check witness** again before generating a proof."
            )
    prove_invalid_opt_in = False
elif witness_valid:
    with col_check_status:
        st.success("✓ Witness satisfies the R1CS.")
    prove_invalid_opt_in = False
else:
    # Checked, current, but R1CS-invalid → offer the demo opt-in.
    with col_check_status:
        st.error("✗ Witness does **not** satisfy the R1CS.")
    prove_invalid_opt_in = st.checkbox(
        "🎯 Demo: prove the **invalid** witness anyway",
        value = False,
        help  = "Lets the prover encode + prove a witness that fails the "
                "R1CS, so the verifier's pairing check rejects it.  "
                "Use this to show that a wrong witness produces a "
                "syntactically valid proof which the verifier still "
                "catches — `verified = False`.",
    )


# ── Generate proof ─────────────────────────────────────────────────────


st.divider()

# Gate the button.  We auto-activate inside the submit handler, so the
# gate only cares about: (a) TPU env warm, (b) circuit ready/built,
# (c) witness checked + (valid OR user opted into the demo path).
prove_skip_r1cs_check = (not witness_valid) and prove_invalid_opt_in
ok_to_submit = witness_valid or prove_skip_r1cs_check

if not env.done:
    submit_help = "Waiting for TPU env to warm up…"
elif spec.status == "available":
    submit_help = "Click **Build** in the sidebar first."
elif spec.status == "building":
    submit_help = "Build still running — wait for it to finish."
elif spec.status == "error":
    submit_help = "Last build failed — rebuild from the sidebar."
elif not witness_check_current:
    submit_help = "Click **Check witness** above first."
elif not ok_to_submit:
    submit_help = ("Witness fails the R1CS.  Tick the demo box above "
                   "if you want to prove it anyway.")
else:
    submit_help = None
submit_disabled = submit_help is not None

submit_label = (
    "▶ Generate proof (expect verification to FAIL)"
    if prove_skip_r1cs_check else
    "▶ Generate proof"
)
submitted = st.button(
    submit_label,
    disabled = submit_disabled,
    help     = submit_help,
    type     = "primary",
)


# ── Submit handler — full prove + verify pipeline via daemon ──────────


if submitted:
    # If the selected circuit isn't active, activate it (drops the prior).
    if selected_key != active_key:
        with st.spinner(f"Activating {selected_key} (loading pkl into TPU)…"):
            try:
                client.activate(selected_key)
            except DaemonError as ex:
                st.error(f"Couldn't activate {selected_key}: {ex}")
                st.stop()

    st.subheader("Generating proof")
    bar_encode = st.progress(0.0, text="Phase 3 — encode witness")
    bar_prove  = st.progress(0.0, text="Phase 4 — prove (TPU)")
    bar_verify = st.progress(0.0, text="Phase 5 — verify (CPU pairings)")

    t_total = _time.time()
    bar_encode.progress(0.5, text="Phase 3 — encode (~10 ms)")
    bar_prove.progress(0.1,  text="Phase 4 — prove (sending to TPU)")
    bar_verify.progress(0.0, text="Phase 5 — verify (queued)")

    try:
        reply = client.prove(
            circuit         = selected_key,
            witness         = witness,
            seed            = seed,
            verify          = True,
            timeout         = 900.0,
            skip_r1cs_check = prove_skip_r1cs_check,
        )
    except DaemonError as ex:
        bar_encode.progress(1.0, text="Phase 3 — failed")
        bar_prove.progress(0.0,  text="Phase 4 — not started")
        bar_verify.progress(0.0, text="Phase 5 — not started")
        st.error(f"Daemon error: {ex}")
        st.stop()

    bar_encode.progress(1.0,
                        text=f"Phase 3 — encode ✓ ({reply.encode_ms:.1f} ms)")
    bar_prove.progress(1.0,
                       text=f"Phase 4 — prove ✓ ({reply.prove_ms:.1f} ms)")
    bar_verify.progress(1.0,
                        text=f"Phase 5 — verify ✓ "
                             f"({reply.verify_ms:.1f} ms, "
                             f"{'OK' if reply.verified else 'FAIL'})")

    total_ms = (_time.time() - t_total) * 1000

    if reply.verified:
        st.success(
            f"✓ Proof verified.  Total: {total_ms:.1f} ms "
            f"(encode {reply.encode_ms:.1f} + prove {reply.prove_ms:.1f} "
            f"+ verify {reply.verify_ms:.1f} + network/round-trip).",
            icon = "🟢",
        )
    else:
        st.error(
            f"✗ Verification FAILED.  Total: {total_ms:.1f} ms.",
            icon = "🔴",
        )

    # Side-by-side: without ZKP vs with ZKP
    st.divider()
    st.header("Without ZKP   vs   With ZKP")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.subheader("🔓 Direct (no ZKP)")
        st.write(
            f"To convince a verifier that **{spec.public_label} = "
            f"{reply.public_output}** comes from this circuit, the prover "
            "must send the **whole witness** so the verifier can recompute."
        )
        st.code(", ".join(reply.witness[:spec.num_wires_orig]),
                language="text")
        st.metric(
            "Bytes the verifier needs",
            f"{reply.witness_bytes:,} B",
            help = "32 bytes per Fr scalar × witness length.",
        )
        st.caption(
            "Verifier sees **every wire**, including the private inputs.  "
            "And must redo the math to check correctness."
        )

    with col_b:
        st.subheader("🔒 Groth16 (with ZKP)")
        st.write(
            f"The verifier sees only the **public output** "
            f"(`{spec.public_label} = {reply.public_output}`) and a "
            "constant-size proof.  Private wires stay private."
        )
        st.code(
            f"A = {reply.proof['A']}\n"
            f"B = {reply.proof['B']}\n"
            f"C = {reply.proof['C']}",
            language="json",
        )
        st.metric(
            "Bytes the verifier needs",
            f"{reply.proof_bytes:,} B",
            help = "Compressed: A 48 B + B 96 B + C 48 B = 192 B.",
        )
        st.caption(
            "Verifier runs **three pairings** on these 192 bytes — "
            f"~{reply.verify_ms:.0f} ms — and is convinced.  Witness "
            "stays secret."
        )

    # Per-phase TPU breakdown
    st.divider()
    st.subheader("Per-phase TPU breakdown")
    st.caption(
        "Wall time of each stage inside the prove kernel, with "
        "`block_until_ready()` between ops."
    )
    rows = [
        ("Phase 3 — encode witness (CPU)",       reply.encode_ms),
        ("Phase 4 — prove total (TPU)",          reply.prove_ms),
        ("  · MSM G1 — A_g1",                    reply.phase_ms["msm_g1_A_ms"]),
        ("  · MSM G1 — B_g1",                    reply.phase_ms["msm_g1_B_ms"]),
        ("  · MSM G1 — private_g1",              reply.phase_ms["msm_g1_private_ms"]),
        ("  · H pipeline (NTT + INTT + Fr)",     reply.phase_ms["h_pipeline_ms"]),
        ("  · H decode (host: DRNS → int)",      reply.phase_ms["h_decode_ms"]),
        ("  · MSM G1 — h_g1",                    reply.phase_ms["msm_g1_h_ms"]),
        ("  · MSM G2 — B_g2 (RCB projective)",   reply.phase_ms["msm_g2_B_ms"]),
        ("  · π_A composite (G1)",               reply.phase_ms["composite_piA_ms"]),
        ("  · π_B G1 composite",                 reply.phase_ms["composite_piB_g1_ms"]),
        ("  · π_B G2 composite (Fp2)",           reply.phase_ms["composite_piB_g2_ms"]),
        ("  · π_C composite (G1)",               reply.phase_ms["composite_piC_ms"]),
        ("  · CPU decode of EC points",          reply.phase_ms["decode_ms"]),
        ("Phase 5 — verify (CPU pairings)",      reply.verify_ms),
    ]
    st.dataframe(
        {
            "Phase":     [r[0] for r in rows],
            "Time (ms)": [f"{r[1]:.2f}" for r in rows],
        },
        hide_index = True,
        use_container_width = True,
    )

    if reply.seed_used is not None:
        st.caption(
            f"Deterministic mode: seed = {reply.seed_used}.  "
            "Rerun with the same witness + seed to get the same proof bytes."
        )
    else:
        st.caption(
            "Production mode: blinders sampled from `secrets.randbelow`.  "
            "Same witness → different valid proof every time (zero-knowledge)."
        )
