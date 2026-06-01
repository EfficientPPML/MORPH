"""Streamlit-based GUI demo for the TPU Groth16 prover.

Public entry point: ``Groth16_TPU/demo/demo_app.py`` (run via
``streamlit run``).  Backend support modules:

  * :mod:`init_worker`   — background-thread TPU boot, polled by the UI
  * :mod:`demo_circuits` — registry of selectable predefined circuits
  * :mod:`demo_app`      — the Streamlit front-end
"""
