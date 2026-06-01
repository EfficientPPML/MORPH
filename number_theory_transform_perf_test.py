"""Performance tests for batched NTT.

Profiles the six NTT designs produced by crossing
  * step count    (:class:`NTT3Step`, :class:`NTT5Step`, :class:`NTT7Step`)
  * backend       (DRNS via :class:`DRNSLazyExtensionContext`,
                   CROSS via :class:`CROSSLazyExtensionContext`)

The batch (leading) dimension is sharded across all available JAX
devices.  Performance is measured via ``jax.profiler`` through the
``KernelWrapper`` / ``Profiler`` helpers in ``profiler.py``, NOT
wall-clock time.  Sharded-correctness checks live in
``number_theory_transform_test.py``.
"""

import os

import jax
import jax.sharding as shd
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import number_theory_transform_context as ntt_context
import utils
from profiler import KernelWrapper, Profiler, collect_logs

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Trailing-dimension sizing — change these to sweep field-element width.
# NUM_MODULI drives the prime preset below (21 → 256-bit prime, 56 → 753-bit).
# ---------------------------------------------------------------------------
NUM_MODULI = 21           # DRNS: number of RNS moduli (trailing dim size)
PRECISION_BITS = 28       # DRNS: bit-width per modulus
RADIX_BITS = 32           # DRNS: Montgomery radix

# ---------------------------------------------------------------------------
# Prime & 2N-th-root presets, keyed by NUM_MODULI.  Each preset supplies a
# matching Q_PERF and PSI_PERF_BY_DEGREE (primitive 2*2^degree-th roots of
# unity mod Q_PERF).  The per-degree psi sidesteps utils.root_of_unity(),
# which trial-divides q-1 and is infeasible for the 753-bit prime.
# ---------------------------------------------------------------------------
_PERF_PRIME_PRESETS: dict[int, dict] = {
    # 256-bit prime — fits NUM_MODULI * PRECISION_BITS = 21 * 28 = 588 ≥ 256.
    21: {
        "q": 0x8000000000000000000000000000000000000000000000000000000070000001,
        "psi_by_degree": {
            14: 0x210d1d264152132ae3e5610b7e230bcd0058fe66fb35c5713527ea1fa40d1845,
            16: 0x40d7c3f33672325e7b65c4a20b0be07dd32f3ebb05c33dd8675d68eb3a8bdb6b,
            18: 0x568fddcd95737ac264eaada546d74b051ca1b7fc5b8427dce706674011e009e0,
            20: 0x3aae36a59e8e4f95e3118aa64270d0e122e0fc9585380815f737a67d613b5516,
            22: 0x23e461bcc11091f4a355ad034b454991f9cfa113272b8dbdf38e895c68be3702,
        },
    },
    # 753-bit prime — fits NUM_MODULI * PRECISION_BITS = 56 * 28 = 1568 ≥ 753.
    56: {
        "q": 0x100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000023c00001,
        "psi_by_degree": {
            14: 0x987b6025ab8620c5cb8376257cbc1fed3f6a9cd8003b6cb78c442378c5cb76df824ddfad28f53e6efed050f1193bd6b114ddbaf944860ce6ceec6a4eccc92d690996cd24ed61167e18b61f33c7f45ca1231f30751c16aa586da157bc12da,
            16: 0xc35116503813c2fea4aa3458b0f4f8b28bbcc8bee7004d34fb47b57fe855c6d764af4f7ce54c9334c6c957cde2d613405ad78f5bf210772600a9e8fbc797e35ddfcb55643e3f9ade2c9b4ad8d08f00d7224ee0e4e176c98e61232e23114c,
            18: 0x80de16a04909c865c8c3954f81d93780d98b4826b7c8aec95a3149697831b5ce3be84cea28866b38a0458484acdfbfc0fb26215fff7083f52721e3bbe094fa42ccaf9d9df1730211f6a211bed8c8128041609a585c1234113ac33fd11fd2,
            20: 0x4ae6b7005951673890ac5aea5dfca80f449c73138fbba6ef4e26d43dda812a3be60c97e60450c3aa294b751f6fcab9e3736c02191a19e87c08ab00f3a3d2b599cfc3d52a0886f3eda8941813bcdbcd51930f838b04cfe32a1aef52510324,
            22: 0x26dfa808fe842c5a3c50eed23d433805d7ca3bc5fe9edfa38c0a325159d75b8dbf7ed80054d92678dccc4cc6b9a1d47976dfda07d39c463f312ec45147860c46f8e4ace696ab8b7f789ebe170fe615c18902c15f97abc7300dbd9f1870a,
        },
    },
}

if NUM_MODULI not in _PERF_PRIME_PRESETS:
  raise ValueError(
      f"No Q_PERF/PSI preset for NUM_MODULI={NUM_MODULI}; "
      f"available: {sorted(_PERF_PRIME_PRESETS)}"
  )
Q_PERF: int = _PERF_PRIME_PRESETS[NUM_MODULI]["q"]
PSI_PERF_BY_DEGREE: dict[int, int] = _PERF_PRIME_PRESETS[NUM_MODULI]["psi_by_degree"]

# DRNS configs (BAT einsum + Montgomery/CRNS — runs fast, can push to deg=22).
PERF_CONFIGS_DRNS_3STEP = [
    {"degree": 14, "r1": 7,  "c": 7},
    {"degree": 16, "r1": 8,  "c": 8},
    {"degree": 18, "r1": 9,  "c": 9},
    {"degree": 20, "r1": 10, "c": 10},
    {"degree": 22, "r1": 11, "c": 11},
]

PERF_CONFIGS_DRNS_5STEP = [
    {"degree": 14, "r1": 5, "r2": 5, "c": 4},
    {"degree": 16, "r1": 5, "r2": 5, "c": 6},
    {"degree": 18, "r1": 6, "r2": 6, "c": 6},
    {"degree": 20, "r1": 6, "r2": 6, "c": 8},
    {"degree": 22, "r1": 7, "r2": 7, "c": 8},
]

PERF_CONFIGS_DRNS_7STEP = [
    {"degree": 16, "r1": 4, "r2": 4, "c1": 4, "c2": 4},
    {"degree": 20, "r1": 5, "r2": 5, "c1": 5, "c2": 5},
]

# CROSS configs — CROSS's fori_loop-based matmul is ~1000× slower than
# DRNS BAT per NTT, so keep sizes modest to finish in reasonable time.
PERF_CONFIGS_CROSS_3STEP = [
    {"degree": 14, "r1": 7, "c": 7},
    {"degree": 16, "r1": 8, "c": 8},
    {"degree": 18, "r1": 9,  "c": 9},
    {"degree": 20, "r1": 10, "c": 10},
    {"degree": 22, "r1": 11, "c": 11},
]

PERF_CONFIGS_CROSS_5STEP = [
    {"degree": 14, "r1": 5, "r2": 5, "c": 4},
    {"degree": 16, "r1": 5, "r2": 5, "c": 6},
    {"degree": 18, "r1": 6, "r2": 6, "c": 6},
    {"degree": 20, "r1": 6, "r2": 6, "c": 8},
    {"degree": 22, "r1": 7, "r2": 7, "c": 8},
]

PERF_CONFIGS_CROSS_7STEP = [
    {"degree": 16, "r1": 4, "r2": 4, "c1": 4, "c2": 4},
    {"degree": 20, "r1": 5, "r2": 5, "c1": 5, "c2": 5},
]


# ---------------------------------------------------------------------------
# Extension-context builders
# ---------------------------------------------------------------------------
def _create_drns_ff_ctx(prime):
  rns_moduli = utils.find_moduli_specified_number(NUM_MODULI, PRECISION_BITS)
  return ntt_context.DRNSLazyExtensionContext(
      {
          "prime": prime,
          "rns_moduli": rns_moduli,
          "precision_bits": PRECISION_BITS,
          "radix_bits": RADIX_BITS,
      }
  )


def _create_cross_ff_ctx(prime):
  params = {"prime": prime}
  return ntt_context.CROSSLazyExtensionContext(params)


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------
def _create_sharding():
  """Create default batch sharding for the current device mesh."""
  available_devices = jax.devices()
  if not available_devices:
    raise RuntimeError("No devices available for sharding test.")
  if len(available_devices) == 8:
    mesh_shape = (2, 4)
  elif len(available_devices) == 4:
    mesh_shape = (2, 2)
  elif len(available_devices) == 2:
    mesh_shape = (2, 1)
  else:
    mesh_shape = (1, 1)

  mesh = jax.make_mesh(mesh_shape, ('x', 'y'))
  return mesh, jax.sharding.PartitionSpec


def _batch_sharding(mesh, partition_spec, ndim):
  """NamedSharding that partitions only the leading (batch) axis."""
  axis_names = mesh.axis_names
  batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]
  spec = (batch_partition,) + (None,) * (ndim - 1)
  return jax.sharding.NamedSharding(mesh, partition_spec(*spec))


# ---------------------------------------------------------------------------
# Kernel-wrapper helpers
# ---------------------------------------------------------------------------
def _ntt_kernel(input_array, parameters):
  return parameters["ctx"].ntt(input_array)


def _intt_kernel(input_array, parameters):
  return parameters["ctx"].intt(input_array)


def _shard_mapped_kernel(method_name):
  """Build a kernel entry that shard-maps ``ctx.<method_name>`` over batch.

  CROSS's NTT path uses ``jax.vmap`` internally over every leading axis
  of ``_modular_multiply``.  Under a plain jit with a batch-sharded
  input, the broadcast of replicated twiddles against the sharded input
  creates vmap axis-spec mismatches.  Running the kernel under
  ``shard_map`` gives every device a shard-local (unsharded) view, so
  all broadcast / reshape / vmap ops inside the kernel see regular-
  strided arrays while the heavy lifting still parallelizes across all
  devices at the outer (shard_map) level.
  """

  def kernel(input_array, parameters):
    ctx = parameters["ctx"]
    mesh = parameters["mesh"]
    batch_spec = parameters["batch_spec"]
    fn = getattr(ctx, method_name)
    mapped = shard_map(fn, mesh=mesh, in_specs=batch_spec, out_specs=batch_spec,
                       check_rep=False)
    return mapped(input_array)

  return kernel


# ---------------------------------------------------------------------------
# Performance profiling (jax.profiler traces via KernelWrapper + Profiler).
# One test method per design: 3 DRNS designs + 3 CROSS designs = 6 tests.
# ---------------------------------------------------------------------------
class NTTShardedPerformanceTest(parameterized.TestCase):
  """Profile all six NTT designs (step-count × backend) at sharded batch."""

  def setUp(self):
    super().setUp()
    self.output_trace_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "log"
    )
    self.profiler_config = {
        "iterations": 3,
        "save_to_file": True,
        "enable_sharding": True,
    }

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    collect_logs(os.path.dirname(os.path.abspath(__file__)))

  # -------------------------------------------------------------------------
  # Profile driver: uniform entry point for DRNS and CROSS.  Selects the
  # right extension context, trailing-dim size, and per-device kernel
  # wrapping (plain jit for DRNS, shard_map-wrapped jit for CROSS).
  # -------------------------------------------------------------------------
  def _profile_design(
      self, variant_name, configs, ntt_cls, backend, make_params
  ):
    assert backend in ("drns", "cross")
    if backend == "drns":
      ff_ctx = _create_drns_ff_ctx(Q_PERF)
      trailing = len(ff_ctx.rns_moduli)
      trailing_key = "num_moduli"
      kernel_factory = lambda direction: (_ntt_kernel if direction == "ntt" else _intt_kernel)
    else:
      ff_ctx = _create_cross_ff_ctx(Q_PERF)
      trailing = ff_ctx.chunk_num_u32
      trailing_key = "chunk_num_u32"
      kernel_factory = lambda direction: _shard_mapped_kernel(direction)

    mesh, partition_spec = _create_sharding()
    num_devices = len(jax.devices())

    profiler = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"sharded_{variant_name}_{ntt_cls.__name__}",
        configuration=self.profiler_config,
    )

    for cfg in configs:
      degree = cfg["degree"]
      spatial_params, spatial_shape = make_params(cfg)
      params = {"prime": Q_PERF, "finite_field_context": ff_ctx, **spatial_params}
      psi = PSI_PERF_BY_DEGREE.get(degree)
      if psi is not None:
        params["psi"] = psi
      ntt_ctx = ntt_cls(params)

      ndim = 1 + len(spatial_shape) + 1
      batch_sharding = _batch_sharding(mesh, partition_spec, ndim)

      axis_names = mesh.axis_names
      batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]
      batch_spec = partition_spec(batch_partition, *([None] * (ndim - 1)))

      setting_base = {
          "context": ntt_cls.__name__,
          "backend": backend,
          "degree": degree,
          "spatial_shape": str(spatial_shape),
          trailing_key: trailing,
          "num_devices": num_devices,
      }

      batch = num_devices
      input_shape = (batch,) + spatial_shape + (trailing,)
      for direction in ("ntt", "intt"):
        name = f"{variant_name}_{direction}_deg{degree}_batch{batch}"
        wrapper = KernelWrapper(
            kernel_name=name,
            function_to_wrap=kernel_factory(direction),
            input_structs=[(input_shape, jnp.uint32)],
            parameters={"ctx": ntt_ctx, "mesh": mesh, "batch_spec": batch_spec},
            mesh=mesh,
            input_shardings=(batch_sharding,),
            output_sharding=batch_sharding,
            enable_sharding=True,
        )
        profiler.add_profile(
            name=name,
            kernel_wrapper=wrapper,
            kernel_setting_cols={**setting_base, "direction": direction, "batch": batch},
        )

    profiler.profile_all_profilers()
    profiler.post_process_all_profilers()

  # -------------------------------------------------------------------------
  # 3 DRNS-based designs
  # -------------------------------------------------------------------------
  def test_sharded_drns_3step(self):
    """DRNS 3-step NTT (``NTT3Step`` + ``DRNSLazyExtensionContext``)."""
    def make_params(cfg):
      r, c = 2 ** cfg["r1"], 2 ** cfg["c"]
      return {"r": r, "c": c}, (r, c)
    self._profile_design(
        "drns_3step", PERF_CONFIGS_DRNS_3STEP, ntt_context.NTT3Step, "drns", make_params
    )

  def test_sharded_drns_5step(self):
    """DRNS 5-step NTT (``NTT5Step`` + ``DRNSLazyExtensionContext``)."""
    def make_params(cfg):
      rr, rc, c = 2 ** cfg["r1"], 2 ** cfg["r2"], 2 ** cfg["c"]
      return {"rr": rr, "rc": rc, "c": c}, (rr, rc, c)
    self._profile_design(
        "drns_5step", PERF_CONFIGS_DRNS_5STEP, ntt_context.NTT5Step, "drns", make_params
    )

  def test_sharded_drns_7step(self):
    """DRNS 7-step NTT (``NTT7Step`` + ``DRNSLazyExtensionContext``)."""
    def make_params(cfg):
      rr, rc = 2 ** cfg["r1"], 2 ** cfg["r2"]
      cr, cc = 2 ** cfg["c1"], 2 ** cfg["c2"]
      return {"rr": rr, "rc": rc, "cr": cr, "cc": cc}, (rr, rc, cr, cc)
    self._profile_design(
        "drns_7step", PERF_CONFIGS_DRNS_7STEP, ntt_context.NTT7Step, "drns", make_params
    )

  # -------------------------------------------------------------------------
  # 3 CROSS-backed designs
  # -------------------------------------------------------------------------
  def test_sharded_cross_3step(self):
    """CROSS 3-step NTT (``NTT3Step`` + ``CROSSLazyExtensionContext``)."""
    def make_params(cfg):
      r, c = 2 ** cfg["r1"], 2 ** cfg["c"]
      return {"r": r, "c": c}, (r, c)
    self._profile_design(
        "cross_3step", PERF_CONFIGS_CROSS_3STEP, ntt_context.NTT3Step, "cross", make_params
    )

  def test_sharded_cross_5step(self):
    """CROSS 5-step NTT (``NTT5Step`` + ``CROSSLazyExtensionContext``)."""
    def make_params(cfg):
      rr, rc, c = 2 ** cfg["r1"], 2 ** cfg["r2"], 2 ** cfg["c"]
      return {"rr": rr, "rc": rc, "c": c}, (rr, rc, c)
    self._profile_design(
        "cross_5step", PERF_CONFIGS_CROSS_5STEP, ntt_context.NTT5Step, "cross", make_params
    )

  def test_sharded_cross_7step(self):
    """CROSS 7-step NTT (``NTT7Step`` + ``CROSSLazyExtensionContext``)."""
    def make_params(cfg):
      rr, rc = 2 ** cfg["r1"], 2 ** cfg["r2"]
      cr, cc = 2 ** cfg["c1"], 2 ** cfg["c2"]
      return {"rr": rr, "rc": rc, "cr": cr, "cc": cc}, (rr, rc, cr, cc)
    self._profile_design(
        "cross_7step", PERF_CONFIGS_CROSS_7STEP, ntt_context.NTT7Step, "cross", make_params
    )


if __name__ == "__main__":
  absltest.main()
