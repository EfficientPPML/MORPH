import os
import random

import jax

import number_theory_transform_context as ntt_context


jax.config.update("jax_enable_x64", True)

DEGREE = 18
N = 2**DEGREE
R = 2**9
C = 2**9
NUM_MODULI = 21
PRECISION_BITS = 28
RADIX_BITS = 32

Q_PERF = 0x8000000000000000000000000000000000000000000000000000000070000001
PSI_PERF = 0x568FDDCD95737AC264EAADA546D74B051CA1B7FC5B8427DCE706674011E009E0
TRACE_DIR = "./demo_traces/ntt_degree_18"


def build_ntt_context():
  return ntt_context.NTT3Step(
      {
          "prime": Q_PERF,
          "psi": PSI_PERF,
          "r": R,
          "c": C,
          "finite_field_context": "drns",
          "num_moduli": NUM_MODULI,
          "precision_bits": PRECISION_BITS,
          "radix_bits": RADIX_BITS,
      }
  )


def main():
  os.makedirs(TRACE_DIR, exist_ok=True)

  print(f"Building degree-{DEGREE} NTT context")
  ntt_ctx = build_ntt_context()

  print(f"Generating {N} random coefficients")
  coeffs = [random.randrange(Q_PERF) for _ in range(N)]

  print("Converting input to computational format")
  coeffs_m = ntt_ctx.to_computational_format(coeffs)
  coeffs_m = coeffs_m.reshape(1, R, C, coeffs_m.shape[-1])

  ntt = jax.jit(ntt_ctx.ntt)

  print("Warming up JIT")
  warmup = ntt(coeffs_m)
  warmup.block_until_ready()

  with jax.profiler.trace(TRACE_DIR):
    print("Running profiled NTT")
    result = ntt(coeffs_m)
    result.block_until_ready()

  print("NTT done")


if __name__ == "__main__":
  main()
