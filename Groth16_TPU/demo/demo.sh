#!/usr/bin/env bash
# Lifecycle wrapper for the TPU Groth16 demo.
#
# Subcommands:
#   start          — start the prover daemon (port 8000) and Streamlit UI (port 8501)
#   stop           — kill both
#   status         — show what's running
#   logs           — tail both logs
#   restart        — stop + start (both)
#   restart-ui     — restart ONLY the Streamlit UI (keeps daemon + TPU warm)
#   restart-daemon — restart ONLY the prover daemon (keeps UI page open)
#
# State (PID files, logs) lives in ``.demo_run/`` next to this script.

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"        # MAPLE root
RUN_DIR="$HERE/.demo_run"
mkdir -p "$RUN_DIR"

DAEMON_PID="$RUN_DIR/daemon.pid"
UI_PID="$RUN_DIR/ui.pid"
DAEMON_LOG="$RUN_DIR/daemon.log"
UI_LOG="$RUN_DIR/ui.log"

DAEMON_HOST="${MAPLE_DAEMON_HOST:-127.0.0.1}"
DAEMON_PORT="${MAPLE_DAEMON_PORT:-8000}"
UI_HOST="${MAPLE_UI_HOST:-0.0.0.0}"
UI_PORT="${MAPLE_UI_PORT:-8501}"

PY="${PYTHON:-python}"

# ─────────────────────────────────────────────────────────────────────

is_alive() {
    local pidfile="$1"
    [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null
}

show_status() {
    local label="$1" pidfile="$2" port="$3"
    if is_alive "$pidfile"; then
        local pid=$(cat "$pidfile")
        printf "  %-9s alive    PID %-7s  port %s\n" "$label" "$pid" "$port"
    else
        printf "  %-9s stopped                  port %s\n" "$label" "$port"
    fi
}

# Stop one component cleanly: SIGTERM, wait 5s, SIGKILL fallback.  Returns
# 0 if it stopped something, 1 if there was nothing to stop.
_stop_one() {
    local label="$1" pidfile="$2"
    if ! is_alive "$pidfile"; then
        return 1
    fi
    local pid
    pid=$(cat "$pidfile")
    echo "  stopping $label (PID $pid)"
    kill "$pid" 2>/dev/null || true
    for _ in {1..10}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "    forcing SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
    return 0
}

# ─────────────────────────────────────────────────────────────────────
# Per-component start / stop
# ─────────────────────────────────────────────────────────────────────

start_daemon() {
    cd "$ROOT"
    if is_alive "$DAEMON_PID"; then
        echo "  daemon already running (PID $(cat "$DAEMON_PID"))"
        return 0
    fi
    echo "  starting prover daemon on $DAEMON_HOST:$DAEMON_PORT"
    nohup "$PY" -m Groth16_TPU.demo.daemon \
        --host "$DAEMON_HOST" --port "$DAEMON_PORT" \
        > "$DAEMON_LOG" 2>&1 &
    echo "$!" > "$DAEMON_PID"
    sleep 1
    if is_alive "$DAEMON_PID"; then
        echo "  daemon  → PID $(cat "$DAEMON_PID"), log $DAEMON_LOG"
    else
        echo "  daemon failed to start — see $DAEMON_LOG"
        return 1
    fi
}

start_ui() {
    cd "$ROOT"
    if is_alive "$UI_PID"; then
        echo "  UI already running (PID $(cat "$UI_PID"))"
        return 0
    fi
    echo "  starting Streamlit UI on $UI_HOST:$UI_PORT"
    export MAPLE_DAEMON_URL="http://$DAEMON_HOST:$DAEMON_PORT"
    nohup "$PY" -m streamlit run "$HERE/demo_app.py" \
        --server.address "$UI_HOST" --server.port "$UI_PORT" \
        --server.headless true --browser.gatherUsageStats false \
        > "$UI_LOG" 2>&1 &
    echo "$!" > "$UI_PID"
    sleep 1
    if is_alive "$UI_PID"; then
        echo "  UI      → PID $(cat "$UI_PID"), log $UI_LOG"
    else
        echo "  UI failed to start — see $UI_LOG"
        return 1
    fi
}

stop_daemon() {
    _stop_one "daemon" "$DAEMON_PID" || true
    # Forkserver children + any in-flight prep_circuit subprocesses
    # only exist when the daemon is running, so this is safe to do
    # alongside a daemon stop.
    pkill -9 -f 'Groth16_TPU.demo.prep_circuit' 2>/dev/null || true
    pkill -9 -f 'multiprocessing.forkserver'    2>/dev/null || true
}

stop_ui() {
    _stop_one "UI" "$UI_PID" || true
}

# ─────────────────────────────────────────────────────────────────────
# Subcommands
# ─────────────────────────────────────────────────────────────────────

cmd_start() {
    start_daemon
    start_ui
    echo
    echo "  open  http://localhost:$UI_PORT  (SSH-tunnel if remote)"
    echo "  stop  ./demo.sh stop"
    echo "  logs  ./demo.sh logs"
}

cmd_stop() {
    local stopped_ui=1 stopped_daemon=1
    is_alive "$UI_PID"     && stopped_ui=0
    is_alive "$DAEMON_PID" && stopped_daemon=0
    stop_ui
    stop_daemon
    if [[ $stopped_ui -ne 0 && $stopped_daemon -ne 0 ]]; then
        echo "  nothing to stop"
    fi
}

cmd_status() {
    show_status "daemon" "$DAEMON_PID" "$DAEMON_PORT"
    show_status "UI"     "$UI_PID"     "$UI_PORT"

    if is_alive "$DAEMON_PID"; then
        echo
        echo "  /healthz:"
        curl -sS "http://$DAEMON_HOST:$DAEMON_PORT/healthz" 2>/dev/null \
            | sed 's/^/    /' || echo "    (no response)"
        echo "  /status:"
        curl -sS "http://$DAEMON_HOST:$DAEMON_PORT/status" 2>/dev/null \
            | "$PY" -c "import sys,json
o=json.load(sys.stdin)
print(f\"    phase={o['phase']}  progress={o['progress']:.0%}  done={o['done']}  elapsed={o['elapsed']:.1f}s\")" \
            2>/dev/null || echo "    (no response)"
    fi
}

cmd_logs() {
    echo "=== daemon ==="
    [[ -f "$DAEMON_LOG" ]] && tail -n 20 "$DAEMON_LOG" || echo "  (no log)"
    echo
    echo "=== UI ==="
    [[ -f "$UI_LOG" ]] && tail -n 20 "$UI_LOG" || echo "  (no log)"
}

cmd_restart() {
    cmd_stop
    sleep 1
    cmd_start
}

cmd_restart_ui() {
    # Front-end-only restart.  Daemon (and its ~90s TPU warmup) stays up.
    # Great for fast-iterating Streamlit UI changes.
    stop_ui
    sleep 1
    start_ui
    echo
    echo "  open  http://localhost:$UI_PORT  (refresh the browser tab)"
    if ! is_alive "$DAEMON_PID"; then
        echo
        echo "  ⚠ daemon is NOT running.  Start with ./demo.sh start"
    fi
}

cmd_restart_daemon() {
    # Back-end-only restart.  Streamlit UI keeps serving the browser
    # connection; it reconnects automatically once the daemon is back.
    stop_daemon
    sleep 1
    start_daemon
    echo
    if ! is_alive "$UI_PID"; then
        echo "  ⚠ UI is NOT running.  Start with ./demo.sh start"
    else
        echo "  UI is still running; refresh the browser tab to reconnect."
    fi
}

# ─────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $0 {start|stop|status|logs|restart|restart-ui|restart-daemon}

  start           start both prover daemon and Streamlit UI
  stop            stop both
  status          show what's running + daemon's /status
  logs            tail recent daemon + UI logs
  restart         stop then start (both)
  restart-ui      restart ONLY the Streamlit UI (keeps daemon + TPU warm)
                  — use this when iterating on demo_app.py / sidebar /
                    witness editor changes
  restart-daemon  restart ONLY the prover daemon (keeps the UI process up;
                  pays the ~90s TPU warmup again)

Env overrides:
  MAPLE_DAEMON_HOST  (default 127.0.0.1)
  MAPLE_DAEMON_PORT  (default 8000)
  MAPLE_UI_HOST      (default 0.0.0.0)
  MAPLE_UI_PORT      (default 8501)
  PYTHON             (default 'python')

State files:
  $RUN_DIR/
    daemon.pid  ui.pid  daemon.log  ui.log
EOF
    exit 2
}

[[ $# -eq 1 ]] || usage
case "$1" in
    start)          cmd_start          ;;
    stop)           cmd_stop           ;;
    status)         cmd_status         ;;
    logs)           cmd_logs           ;;
    restart)        cmd_restart        ;;
    restart-ui)     cmd_restart_ui     ;;
    restart-daemon) cmd_restart_daemon ;;
    *)              usage              ;;
esac
