#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p demo_traces

echo "Running CROSS modular multiplication"
python demo/demo_modmul_CROSS.py

echo "Running MORPH modular multiplication"
python demo/demo_modmul_MORPH.py

xprof --port 6006 --logdir ./demo_traces/