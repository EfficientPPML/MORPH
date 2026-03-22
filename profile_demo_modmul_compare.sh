echo "Running CROSS modular multiplication"
python demo_modmul_CROSS.py

echo "Running MORPH modular multiplication"
python demo_modmul_MORPH.py

xprof --port 6006 --logdir ./demo_traces/