import os

import jax
import jax.numpy as jnp
import toml
from absl.testing import absltest
from absl.testing import parameterized

import finite_field_context as ff_context
import elliptic_curve_context as ec_context
import utils
from profiler import KernelWrapper, Profiler, collect_logs

jax.config.update("jax_enable_x64", True)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configurations.toml")

BATCH_SIZE_LIST = [128, 256, 512, 1024, 2048, 4096]

NUM_MODULI = 32

TEST_PARAMS_POINT_ADD = [
    ("point_add", BATCH_SIZE_LIST),
]

TEST_PARAMS_POINT_DOUBLE = [
    ("point_double", BATCH_SIZE_LIST),
]


def _build_ec_context():
  ec_config = toml.load(CONFIG_PATH)
  rns_moduli = utils.find_moduli_specified_number(NUM_MODULI, 28)
  finite_field_parameters = {
      "prime": ec_config["ec_parameters_bls12_377_affine"]["prime"],
      "rns_moduli": rns_moduli,
      "precision_bits": 28,
      "radix_bits": 32,
  }
  ete_cfg = ec_config["ec_parameters_bls12_377_extended_twisted_edwards"]
  ec_parameters = {
      "finite_field_context_class": ff_context.DRNSlazyContext,
      "finite_field_parameters": finite_field_parameters,
      "prime": ete_cfg["prime"],
      "order": ete_cfg["order"],
      "a": ete_cfg["a"],
      "twist_d": ete_cfg["d"],
      "alpha": ete_cfg["alpha"],
      "b": ete_cfg["b"],
      "s": ete_cfg["s"],
      "MA": ete_cfg["MA"],
      "MB": ete_cfg["MB"],
      "t": ete_cfg["t"],
      "generator": ete_cfg["generator"],
  }
  return ec_context.ExtendedTwistedEdwardsContext(ec_parameters)


def _point_add_kernel(point_a, point_b, parameters):
  return parameters["ctx"]._point_add(point_a, point_b)


def _point_double_kernel(point, parameters):
  return parameters["ctx"]._point_double(point)


class ECPointAddPerformanceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.output_trace_root = os.path.join(os.path.dirname(__file__), "log")
    self.profiler_config = {
        "iterations": 1,
        "save_to_file": True,
    }

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Collecting logs from: {root_dir}")
    collect_logs(root_dir)

  def _create_point_add_wrapper(self, kernel_name, ec_ctx, batch, num_moduli):
    point_shape = (4, batch, num_moduli)
    return KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=_point_add_kernel,
        input_structs=[
            (point_shape, jnp.uint32),
            (point_shape, jnp.uint32),
        ],
        parameters={"ctx": ec_ctx},
    )

  def _create_point_double_wrapper(self, kernel_name, ec_ctx, batch, num_moduli):
    point_shape = (4, batch, num_moduli)
    return KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=_point_double_kernel,
        input_structs=[
            (point_shape, jnp.uint32),
        ],
        parameters={"ctx": ec_ctx},
    )

  @parameterized.named_parameters(*TEST_PARAMS_POINT_ADD)
  def test_point_add_performance(self, batch_size_list):
    ec_ctx = _build_ec_context()
    profiler_instance = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming="ec_point_add",
        configuration=self.profiler_config,
    )

    for batch in batch_size_list:
      kernel_name = f"ec_point_add_b{batch}"
      kernel_wrapper = self._create_point_add_wrapper(
          kernel_name=kernel_name,
          ec_ctx=ec_ctx,
          batch=batch,
          num_moduli=NUM_MODULI,
      )
      profiler_instance.add_profile(
          name=kernel_name,
          kernel_wrapper=kernel_wrapper,
          kernel_setting_cols={
              "num_moduli": NUM_MODULI,
              "batch": batch,
          },
      )

    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()


if __name__ == "__main__":
  absltest.main()
