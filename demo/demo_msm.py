import os

import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

import finite_field_context as ff_context
import elliptic_curve_context as ec_context
import multiscalar_multiplication_context as msm_context
import utils
from utils import hash_args

jax.config.update("jax_enable_x64", True)
MODULUS_377_INT = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
NUM_MODULI = 32
RNS_MODULI = utils.find_moduli_specified_number(NUM_MODULI, 28)

MSM_LENGTH_LIST = [2**18]
SLICE_BITS = 15
SCALAR_BITS = 253

def _build_msm_parameters(msm_length: int):
  ff_parameters = {
      "prime": MODULUS_377_INT,
      "rns_moduli": RNS_MODULI,
      "precision_bits": 28,
      "radix_bits": 32,
  }
  ec_parameters = {
      "finite_field_context_class": ff_context.DRNSlazyContext,
      "finite_field_parameters": ff_parameters,
      "prime": 258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177,
      "order": 8444461749428370424248824938781546531375899335154063827935233455917409239041,
      "a": -1,
      "twist_d": 122268283598675559488486339158635529096981886914877139579534153582033676785385790730042363341236035746924960903179,
      "alpha": -1,
      "b": 1,
      "s": 10189023633222963290707194929886294091415157242906428298294512798502806398782149227503530278436336312243746741931,
      "MA": 228097355113300204138531148905234651262148041026195375645000724271212049151994375092458297304264351187709081232384,
      "MB": 10189023633222963290707194929886294091415157242906428298294512798502806398782149227503530278436336312243746741931,
      "t": 23560188534917577818843641916571445935985386319233886518929971599490231428764380923487987729215299304184915158756,
      "generator": [
          71222569531709137229370268896323705690285216175189308202338047559628438110820800641278662592954630774340654489393,
          6177051365529633638563236407038680211609544222665285371549726196884440490905471891908272386851767077598415378235,
      ],
  }
  return {
      "elliptic_curve_context_class": ec_context.ExtendedTwistedEdwardsNDContext,
      "elliptic_curve_parameters": ec_parameters,
      "coordinate_dim": 4,
      "msm_length": msm_length,
      "tile_length": msm_length,
      "slice_bits": SLICE_BITS,
      "scalar_bits": SCALAR_BITS,
      "order": ec_parameters["order"],
      "c_kernel_ret_space_ratio": 2,
  }

  
def _build_fusion_ctx(msm_length: int):
  """Mirrors scratch_profile_msm_fused.py: sharded fused MSM with
  pre-compiled kernels. The context's multiscalar_multiply internally
  dispatches to these already-compiled, already-sharded kernels — so it
  must NOT be re-traced under jax.jit.
  """
  print(f"Building fused MSM context for length {msm_length}")
  params = _build_msm_parameters(msm_length)
  ctx = msm_context.FusionMSMContext(params)
  ctx.set_use_compiled_kernels(True)
  ctx.set_use_sharding(True)
  print("Compiling fused MSM kernels")
  ctx.compile(parameters={"use_fused": True})
  print("Finished compiling fused MSM kernels")
  return ctx


def main():
  msm_length = 2**18
  print("Starting MSM fusion demo")
  ctx = _build_fusion_ctx(msm_length)
  print("Preparing MSM input slices")
  tiled_slices = ctx.to_computational_format(None)
  with jax.profiler.trace("./demo_traces/msm_fusion"):
    print("Running profiled fused MSM")
    result = ctx.multiscalar_multiply(tiled_slices)
    result.block_until_ready()
  print("MSM fusion done")


if __name__ == "__main__":
  main()
