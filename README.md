# MORPH: Enable AI Accelerator for Zero Knowledge Proof
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)  

- [Tutorial](https://efficientppml.github.io/MORPH_Tutorial/)

# 1. What is MORPH?
MORPH is the first project to enable AI Accelerator, such as Google TPUs, to accelerate Zero Knowledge Proof Primitives (Multi-scalar Multiplication and Number Theory Transformation) and achieves the State-of-the-art (SotA) throughput and energy efficiency (performance per watt). The detailed flow is shown in the figure below.

It features 
- MXU Lazy Modular Reduction: bringing quadratic high-precision modular reduction down to linear operation.
- dataflow optimization for MSM and NTT.

This branch (`asplos`) contains demo scripts for profiling and comparing the two core workloads.

## Project Structure

```
├── finite_field_context.py           # Finite field arithmetic (MORPH & CROSS backends)
├── elliptic_curve_context.py         # Elliptic curve point arithmetic
├── multiscalar_multiplication_context.py  # Multi-scalar multiplication (MSM)
├── number_theory_transform_context.py     # Number Theoretic Transform (NTT)
├── utils.py                          # JAX kernel utilities, number theory helpers
├── profiler.py                       # Trace parsing and kernel profiling
├── configurations.toml               # Curve parameters (BLS12-377)
├── c_kernels/                        # Custom C kernels for TPU acceleration
├── deployments/                      # Serialized compiled JAX kernels
│
├── demo_modmul_MORPH.py              # Modular multiplication demo (MORPH backend)
├── demo_modmul_MORPH.py              # Modular multiplication demo (MORPH backend)
├── demo_msm_ba.py                    # MSM bucket accumulation demo
│
├── profile_demo_modmul_compare.sh    # Profile & compare MORPH vs MORPH modmul
└── profile_demo_msm_ba.sh            # Profile MSM bucket accumulation
```

## Prerequisites

- **Hardware**: Google Cloud TPU (v4 or later recommended)
- **Python**: 3.10+
- **Dependencies**: `jax`, `jaxlib` (TPU version), `numpy`, `pandas`, `toml`
- **Profiling**: `xprof` (for viewing JAX profiler traces)

## Demos

### 1. Modular Multiplication: MORPH vs MORPH

Runs a batch of 8192 modular multiplications over the BLS12-377 base field (~377-bit prime) using both the MORPH (DRNS-based) and MORPH backends, then launches `xprof` to visualize the profiler traces side by side.

```bash
bash profile_demo_modmul_compare.sh
```

This script:
1. Runs `demo_modmul_MORPH.py` — modular multiplication using `MORPHLazyContext`
2. Runs `demo_modmul_MORPH.py` — modular multiplication using `DRNSlazyContext`
3. Starts `xprof` on port 6006 serving traces from `./demo_traces/`

### 2. Multi-Scalar Multiplication (MSM) — Bucket Accumulation

Runs MSM bucket accumulation aMORPH all windows for 2^16 points on the BLS12-377 curve (Extended Twisted Edwards form), then launches `xprof` for profiling.

```bash
bash profile_demo_msm_ba.sh
```

This script:
1. Runs `demo_msm_ba.py` — initializes the MSM context, generates dummy inputs, and runs bucket accumulation
2. Starts `xprof` on port 6006 serving traces from `./demo_traces/`

### Viewing Traces

After running either demo, open `http://localhost:6006` in a browser to inspect the JAX/TPU profiler traces in the xprof UI.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **DRNS (Double RNS)** | Residue Number System representation enabling efficient large-integer modular arithmetic on TPU |
| **MORPH** | Alternative modular multiplication backend using chunk-based representation |
| **MSM** | Multi-scalar multiplication — computing $\sum_i s_i \cdot P_i$ over elliptic curve points |
| **Bucket Accumulation** | MSM decomposition strategy: scalars are sliced into windows, points accumulated into buckets per window |
| **Compiled Kernels** | Pre-compiled JAX/C kernels stored in `deployments/` for fast TPU execution |
| **Sharding** | Distribution of computation aMORPH TPU cores |

## Configuration

Curve parameters for BLS12-377 (both affine Weierstrass and extended twisted Edwards forms) are defined in `configurations.toml`. The demos use hardcoded parameters matching these configurations.
