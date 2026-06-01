import jax
import finite_field_context as ff_context
import elliptic_curve_context as ec_context
from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import utils
import toml
import os

# NOTE: Ensure all tests point are on the curve
BLS12_377_TEST_CASES = [
    (
        "0",
        [
            [
                0x01AC3A384FC584EFD3E7F2C5A2927E7D454875C874A051027B9E7363D08942533EDE85DAE295D8CAB2751085206BCA76,
                0x011DB83AEC88460820F4868A73B12309EE2E910526E62DB4ACCB303ABF50F86C3985A072ED07A4B81FFB82D8DD247283,
            ],
            [
                0x0164DDDBF27670CE389E2992C0E7DAB7741F1B925EDBDC254D2BC0830BAF8E0B186F80F0DD4DE0F0EA6176E55934D45B,
                0x01908E9D77A0F8AD89AC41441F74248704E756BC59C38920617F51BFCDB738EE5B123876D489D09C9EB904A321A336EC,
            ],
        ],
        [
            [
                0x01546AF2ABB4E189E9BBC412FDBF2A8E5EC6E4A3B0AF132E21EE9CEC3EF5E226490FB98D662670FA3CFB3948B7E2A48C,
                0x002961A558A885DF227FDB09F8BDF57AF179CB9437FF8828F13E9DF01AE55502F409AAF5058B88F2F7CCC7BC0676A5D4,
            ],
            [
                0x00B0630E7F192D20443A93860275447074CE77DF559907FA1900F378D4674649BF25F85C893E2A1916B1DA57594F2E17,
                0x01ACC84F362CF60A265C011F0FE4360A15F51BECF7E2C3923FE07C66D5D113104B56E8486C64204A2A9ECD75BA0C41A7,
            ],
        ],
    ),
]


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configurations.toml")


def _build_ec_parameters():
  ec_config = toml.load(CONFIG_PATH)
  rns_moduli = utils.find_moduli_specified_number(32, 28)
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
  affine_cfg = ec_config["ec_parameters_bls12_377_affine"]
  ref_ec_parameters = {
      "finite_field_parameters": finite_field_parameters,
      "finite_field_context_class": ff_context.DRNSlazyContext,
      "prime": affine_cfg["prime"],
      "order": affine_cfg["order"],
      "a": affine_cfg["a"],
      "b": affine_cfg["b"],
      "generator": affine_cfg["generator"],
  }
  return ec_parameters, ref_ec_parameters


class BLS12_377_Test(parameterized.TestCase):

  def __init__(self, *args, **kwargs):
    super(BLS12_377_Test, self).__init__(*args, **kwargs)

  @parameterized.named_parameters(*BLS12_377_TEST_CASES)
  def test_ExtendedTwistedEdwards_point_add(self, point_batch_1, point_batch_2):
    ec_parameters, ref_ec_parameters = _build_ec_parameters()

    ec_ctx = ec_context.ExtendedTwistedEdwardsContext(ec_parameters)
    ref_ec_ctx = ec_context.CPUWeierstrassAffineContext(ref_ec_parameters)

    point_batch_1_m = ec_ctx.to_computational_format(point_batch_1)
    point_batch_2_m = ec_ctx.to_computational_format(point_batch_2)
    result_m = ec_ctx.point_add(point_batch_1_m, point_batch_2_m)
    result = ec_ctx.to_original_format(result_m)

    ref_result = ref_ec_ctx._point_add(point_batch_1, point_batch_2)

    np.testing.assert_array_equal(result, ref_result)

  @parameterized.named_parameters(*BLS12_377_TEST_CASES)
  def test_ExtendedTwistedEdwards_point_double(self, point_batch_1, point_batch_2):
    ec_parameters, ref_ec_parameters = _build_ec_parameters()

    ec_ctx = ec_context.ExtendedTwistedEdwardsContext(ec_parameters)
    ref_ec_ctx = ec_context.CPUWeierstrassAffineContext(ref_ec_parameters)

    point_batch_1_m = ec_ctx.to_computational_format(point_batch_1)
    result_m = ec_ctx.point_double(point_batch_1_m)
    result = ec_ctx.to_original_format(result_m)

    ref_result = ref_ec_ctx._point_double(point_batch_1)

    np.testing.assert_array_equal(result, ref_result)


# Test points for ND context tests
P1 = [
    0x01AC3A384FC584EFD3E7F2C5A2927E7D454875C874A051027B9E7363D08942533EDE85DAE295D8CAB2751085206BCA76,
    0x011DB83AEC88460820F4868A73B12309EE2E910526E62DB4ACCB303ABF50F86C3985A072ED07A4B81FFB82D8DD247283,
]
P2 = [
    0x0164DDDBF27670CE389E2992C0E7DAB7741F1B925EDBDC254D2BC0830BAF8E0B186F80F0DD4DE0F0EA6176E55934D45B,
    0x01908E9D77A0F8AD89AC41441F74248704E756BC59C38920617F51BFCDB738EE5B123876D489D09C9EB904A321A336EC,
]
P3 = [
    0x01546AF2ABB4E189E9BBC412FDBF2A8E5EC6E4A3B0AF132E21EE9CEC3EF5E226490FB98D662670FA3CFB3948B7E2A48C,
    0x002961A558A885DF227FDB09F8BDF57AF179CB9437FF8828F13E9DF01AE55502F409AAF5058B88F2F7CCC7BC0676A5D4,
]
P4 = [
    0x00B0630E7F192D20443A93860275447074CE77DF559907FA1900F378D4674649BF25F85C893E2A1916B1DA57594F2E17,
    0x01ACC84F362CF60A265C011F0FE4360A15F51BECF7E2C3923FE07C66D5D113104B56E8486C64204A2A9ECD75BA0C41A7,
]


class BLS12_377_ND_Test(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ec_parameters, self.ref_ec_parameters = _build_ec_parameters()
    self.ref_ec_ctx = ec_context.CPUWeierstrassAffineContext(self.ref_ec_parameters)

  def test_nd_1d_batch_matches_original(self):
    """1D batch: results must match the original ExtendedTwistedEdwardsContext."""
    batch_a = [P1, P2]
    batch_b = [P3, P4]

    orig_ctx = ec_context.ExtendedTwistedEdwardsContext(self.ec_parameters)
    nd_ctx = ec_context.ExtendedTwistedEdwardsNDContext(self.ec_parameters)

    orig_a = orig_ctx.to_computational_format(batch_a)
    orig_b = orig_ctx.to_computational_format(batch_b)
    orig_result = orig_ctx.to_original_format(orig_ctx.point_add(orig_a, orig_b))

    nd_a = nd_ctx.to_computational_format(batch_a)
    nd_b = nd_ctx.to_computational_format(batch_b)
    nd_result = nd_ctx.to_original_format(nd_ctx.point_add(nd_a, nd_b))

    np.testing.assert_array_equal(nd_result, orig_result)

  def test_nd_1d_batch_vs_cpu_ref(self):
    """1D batch: results must match the CPU Weierstrass reference."""
    batch_a = [P1, P2]
    batch_b = [P3, P4]

    nd_ctx = ec_context.ExtendedTwistedEdwardsNDContext(self.ec_parameters)
    nd_a = nd_ctx.to_computational_format(batch_a)
    nd_b = nd_ctx.to_computational_format(batch_b)
    nd_result = nd_ctx.to_original_format(nd_ctx.point_add(nd_a, nd_b))

    ref_result = self.ref_ec_ctx._point_add(batch_a, batch_b)
    np.testing.assert_array_equal(nd_result, ref_result)

  def test_nd_2d_batch(self):
    """2D batch: shape (4, 2, 2, precision). Each element checked against CPU ref."""
    batch_a = [[P1, P2], [P2, P1]]
    batch_b = [[P3, P4], [P4, P3]]

    nd_ctx = ec_context.ExtendedTwistedEdwardsNDContext(self.ec_parameters)
    nd_a = nd_ctx.to_computational_format(batch_a)
    nd_b = nd_ctx.to_computational_format(batch_b)

    self.assertEqual(nd_a.shape[0], 4)
    self.assertEqual(nd_a.shape[1], 2)
    self.assertEqual(nd_a.shape[2], 2)

    nd_result_m = nd_ctx.point_add(nd_a, nd_b)
    nd_result = nd_ctx.to_original_format(nd_result_m)

    self.assertLen(nd_result, 2)
    self.assertLen(nd_result[0], 2)

    for i in range(2):
      ref_result = self.ref_ec_ctx._point_add(batch_a[i], batch_b[i])
      np.testing.assert_array_equal(nd_result[i], ref_result)

  def test_nd_3d_batch(self):
    """3D batch: shape (4, 1, 2, 2, precision). Verified element-wise against CPU ref."""
    batch_a = [[[P1, P2], [P2, P1]]]
    batch_b = [[[P3, P4], [P4, P3]]]

    nd_ctx = ec_context.ExtendedTwistedEdwardsNDContext(self.ec_parameters)
    nd_a = nd_ctx.to_computational_format(batch_a)
    nd_b = nd_ctx.to_computational_format(batch_b)

    self.assertEqual(nd_a.shape[0], 4)
    self.assertEqual(nd_a.shape[1], 1)
    self.assertEqual(nd_a.shape[2], 2)
    self.assertEqual(nd_a.shape[3], 2)

    nd_result_m = nd_ctx.point_add(nd_a, nd_b)
    nd_result = nd_ctx.to_original_format(nd_result_m)

    self.assertLen(nd_result, 1)
    self.assertLen(nd_result[0], 2)
    self.assertLen(nd_result[0][0], 2)

    for i in range(2):
      ref_result = self.ref_ec_ctx._point_add(batch_a[0][i], batch_b[0][i])
      np.testing.assert_array_equal(nd_result[0][i], ref_result)

  def test_nd_single_point(self):
    """Single point: depth-1 input produces (4, 1, precision), verified against CPU ref."""
    nd_ctx = ec_context.ExtendedTwistedEdwardsNDContext(self.ec_parameters)
    nd_a = nd_ctx.to_computational_format(P1)
    nd_b = nd_ctx.to_computational_format(P3)

    self.assertEqual(nd_a.shape[0], 4)
    self.assertEqual(nd_a.shape[1], 1)

    nd_result_m = nd_ctx.point_add(nd_a, nd_b)
    nd_result = nd_ctx.to_original_format(nd_result_m)

    ref_result = self.ref_ec_ctx._point_add([P1], [P3])
    np.testing.assert_array_equal(nd_result, ref_result)

  def test_nd_2d_batch_consistency_with_1d(self):
    """2D batch results must be identical to 1D batch results for matching slices."""
    nd_ctx = ec_context.ExtendedTwistedEdwardsNDContext(self.ec_parameters)

    batch_1d_a = [P1, P2]
    batch_1d_b = [P3, P4]
    a_1d = nd_ctx.to_computational_format(batch_1d_a)
    b_1d = nd_ctx.to_computational_format(batch_1d_b)
    result_1d = nd_ctx.to_original_format(nd_ctx.point_add(a_1d, b_1d))

    batch_2d_a = [[P1, P2], [P2, P1]]
    batch_2d_b = [[P3, P4], [P4, P3]]
    a_2d = nd_ctx.to_computational_format(batch_2d_a)
    b_2d = nd_ctx.to_computational_format(batch_2d_b)
    result_2d = nd_ctx.to_original_format(nd_ctx.point_add(a_2d, b_2d))

    np.testing.assert_array_equal(result_2d[0], result_1d)

  def test_nd_point_double_vs_cpu_ref(self):
    """1D batch: point_double must match CPUWeierstrassAffineContext._point_double."""
    nd_ctx = ec_context.ExtendedTwistedEdwardsNDContext(self.ec_parameters)
    batch = [P1, P2]
    batch_m = nd_ctx.to_computational_format(batch)
    result = nd_ctx.to_original_format(nd_ctx.point_double(batch_m))
    ref_result = self.ref_ec_ctx._point_double(batch)
    np.testing.assert_array_equal(result, ref_result)

  def test_nd_point_double_equals_double_add(self):
    """point_double(P) must equal point_add(P, P)."""
    nd_ctx = ec_context.ExtendedTwistedEdwardsNDContext(self.ec_parameters)
    batch = [P1, P2]
    batch_m = nd_ctx.to_computational_format(batch)
    result_double = nd_ctx.to_original_format(nd_ctx.point_double(batch_m))
    result_add = nd_ctx.to_original_format(nd_ctx.point_add(batch_m, batch_m))
    np.testing.assert_array_equal(result_double, result_add)

  def test_nd_point_double_2d_batch(self):
    """2D batch: point_double verified element-wise against CPU ref."""
    nd_ctx = ec_context.ExtendedTwistedEdwardsNDContext(self.ec_parameters)
    batch = [[P1, P2], [P3, P4]]
    batch_m = nd_ctx.to_computational_format(batch)
    result = nd_ctx.to_original_format(nd_ctx.point_double(batch_m))
    for i, row in enumerate(batch):
      ref = self.ref_ec_ctx._point_double(row)
      np.testing.assert_array_equal(result[i], ref)


if __name__ == "__main__":
  absltest.main()
