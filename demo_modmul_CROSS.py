import random
import jax
import jax.numpy as jnp
import finite_field_context as ff_context
import utils


MODULUS_377_INT = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
RNS_MODULI = utils.find_moduli_specified_number(32, 28)
BATCH_SIZE = 8192
my_context = ff_context.CROSSLazyContext(
    {"prime": MODULUS_377_INT, "chunk_num_u8": 48}
)
my_context.set_use_compiled_kernels(True)
my_context.set_use_sharding(True)
# my_context.serialize({"batch_shape": (BATCH_SIZE,)})

print("Loading compiled kernels")
my_context.compile({"batch_shape": (BATCH_SIZE,)})

print("Generating random inputs with batch size", BATCH_SIZE)
batch_a = [random.randint(0, MODULUS_377_INT) for _ in range(BATCH_SIZE)]
batch_b = [random.randint(0, MODULUS_377_INT) for _ in range(BATCH_SIZE)]

print("Converting inputs to computational format")
batch_a_m = my_context.to_computational_format(batch_a)
batch_b_m = my_context.to_computational_format(batch_b)



with jax.profiler.trace("./demo_traces/CROSS"):
    print("Modular Multiplying inputs")
    batch_c_m = my_context.modular_multiply(batch_a_m, batch_b_m)
    batch_c_m.block_until_ready()
    print("Modular multiplication done")









