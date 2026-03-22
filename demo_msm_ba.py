import jax
import jax.numpy as jnp
import finite_field_context as ff_context
import elliptic_curve_context as ec_context
import multiscalar_multiplication_context as msm_context
import utils

MODULUS_377_INT = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
RNS_MODULI = utils.find_moduli_specified_number(32, 28)

ff_parameters = {"prime": MODULUS_377_INT, "rns_moduli": RNS_MODULI, "precision_bits": 28, "radix_bits": 32}

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

msm_length = 2**16

msm_parameters = {
    "elliptic_curve_context_class": ec_context.ExtendedTwistedEdwardsNDContext,
    "elliptic_curve_parameters": ec_parameters,
    "coordinate_dim": 4,
    "msm_length": msm_length,
    "slice_bits": 12,
    "scalar_bits": 253,
    "order": 8444461749428370424248824938781546531375899335154063827935233455917409239041,
    "c_kernel_ret_space_ratio": 1.1,
}

# Create DemoTPUMSM context (TPU-only, no CPU preprocessing)
print("MSM context initializing")
ctx = msm_context.DemoTPUMSMContext(msm_parameters)
ctx.set_use_sharding(True)
ctx.set_use_compiled_kernels(True)
ctx.compile()

# Generate dummy inputs and run TPU MSM (BA -> BR -> WM)
print("Generating dummy inputs")
regular_points, last_window_points = ctx.generate_dummy_inputs()

with jax.profiler.trace("./demo_traces/MSM"):
    print("Running MSM BA")
    result = ctx.bucket_accumulation_all_windows(ctx.all_buckets, regular_points, last_window_points).block_until_ready()
    print("Bucket accumulation all windows done")

