#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p demo_traces

echo "Running degree-18 NTT"
python demo/demo_ntt.py

xprof --port 6006 --logdir ./demo_traces/
