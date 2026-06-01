"""Tests for ``Fq2PippengerMSMContext`` driving the XYZZ-Weierstrass backend
over Fp2 (BLS12-377 G2).

We cross-check against an *independent* oracle: the pure-Python Groth16
``msm_g2`` (projective-Jacobian Fp2 implementation in
``Groth16/kernels/msm.py``).  This is the "known-good oracle" called for in
``Groth16_TPU/G2_ETE_DERIVATION.md``.

Coverage:
  1. Identity MSM:  all-zero scalars → identity result.
  2. Single point:  k·G2_GEN reproduces ``msm_g2([G2_GEN], [k])``.
  3. Small batch:   16 random scalars over 16 random ``i·G2_GEN`` bases
                    matches the CPU oracle, exercising bucket
                    distribution, reduction, and window merge.
"""

import os
import random
import sys

import jax
import jax.numpy as jnp
from absl.testing import absltest

_HERE = os.path.dirname(os.path.abspath(__file__))
_GROTH16 = os.path.join(_HERE, "Groth16")
if _GROTH16 not in sys.path:
  sys.path.insert(0, _GROTH16)

import toml
import utils
import finite_field_context as ff_context
import field_extension_context as fe_context
import elliptic_curve_context as ec_context
import multiscalar_multiplication_context as msm_context

from bls12_377.params import G2_B, G2_GEN_X, G2_GEN_Y
from bls12_377.g2 import G2Affine
from kernels.msm import msm_g2 as cpu_msm_g2

jax.config.update("jax_enable_x64", True)

CONFIG_PATH = os.path.join(_HERE, "configurations.toml")
QNR = 5


def _build_xyzz_msm_params(slice_bits=4, scalar_bits=16):
  ec_config = toml.load(CONFIG_PATH)
  rns_moduli = utils.find_moduli_specified_number(32, 28)
  ete_cfg = ec_config["ec_parameters_bls12_377_extended_twisted_edwards"]
  ff_parameters = {
      "prime": ete_cfg["prime"],
      "rns_moduli": rns_moduli,
      "precision_bits": 28,
      "radix_bits": 32,
  }
  ec_params = {
      "prime": ete_cfg["prime"],
      "order": ete_cfg["order"],
      "quadratic_non_residue": QNR,
      "finite_field_context_class": ff_context.DRNSlazyContext,
      "finite_field_parameters": ff_parameters,
      "field_extension_context_class": fe_context.Fq2Context,
      "field_extension_parameters": {
          "prime": ete_cfg["prime"],
          "quadratic_non_residue": QNR,
          "finite_field_context_class": ff_context.DRNSlazyContext,
          "finite_field_parameters": ff_parameters,
      },
      "a": 0,
      "b": G2_B,
  }
  return {
      "elliptic_curve_context_class": ec_context.XYZZWeierstrassFq2NDContext,
      "elliptic_curve_parameters": ec_params,
      "slice_bits": slice_bits,
      "scalar_bits": scalar_bits,
      "order": ete_cfg["order"],
  }


def _g2_multiple(k: int):
  """Compute ``k·G2_GEN`` via the CPU oracle, return as G2Affine."""
  g = G2Affine(G2_GEN_X, G2_GEN_Y).to_projective()
  return g.scalar_mul(k).to_affine()


def _g2_affine_to_fp2_pair(pt: G2Affine):
  """G2Affine → ``[(x_c0, x_c1), (y_c0, y_c1)]`` for the EC context."""
  return [pt.x, pt.y]


class XYZZFq2PippengerMSMTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.params = _build_xyzz_msm_params(slice_bits=4, scalar_bits=16)
    self.msm_ctx = msm_context.Fq2PippengerMSMContext(self.params)
    self.p = self.params["elliptic_curve_parameters"]["prime"]

  # ------------------------------------------------------------------ #
  #  Helpers                                                            #
  # ------------------------------------------------------------------ #

  def _run_jax_msm(self, affine_pts_fp2, scalars):
    cf = self.msm_ctx.to_computational_format(affine_pts_fp2)
    out_cf = self.msm_ctx.multiscalar_multiply(cf, scalars)
    # ``to_original_format`` returns a length-1 list (the batch axis); unwrap.
    return self.msm_ctx.to_original_format(out_cf)[0]

  def _assert_affine_equal(self, got_fp2, expected_g2: G2Affine, label=""):
    """``got_fp2`` is ``[(c0, c1), (c0, c1)]``; ``expected_g2`` is G2Affine."""
    p = self.p
    if expected_g2.is_infinity():
      # XYZZ identity decodes to all-zeros affine by our convention.
      self.assertEqual(got_fp2[0][0] % p, 0, f"{label}: x.c0 (∞)")
      self.assertEqual(got_fp2[0][1] % p, 0, f"{label}: x.c1 (∞)")
      self.assertEqual(got_fp2[1][0] % p, 0, f"{label}: y.c0 (∞)")
      self.assertEqual(got_fp2[1][1] % p, 0, f"{label}: y.c1 (∞)")
      return
    self.assertEqual(got_fp2[0][0] % p, expected_g2.x[0] % p, f"{label}: x.c0")
    self.assertEqual(got_fp2[0][1] % p, expected_g2.x[1] % p, f"{label}: x.c1")
    self.assertEqual(got_fp2[1][0] % p, expected_g2.y[0] % p, f"{label}: y.c0")
    self.assertEqual(got_fp2[1][1] % p, expected_g2.y[1] % p, f"{label}: y.c1")

  # ------------------------------------------------------------------ #
  #  All-zero scalars → identity                                        #
  # ------------------------------------------------------------------ #

  def test_zero_scalars_produces_identity(self):
    """Σ 0 · P_i ≡ identity for any bases."""
    bases = [_g2_multiple(i + 1) for i in range(4)]
    pts_fp2 = [_g2_affine_to_fp2_pair(p) for p in bases]
    got = self._run_jax_msm(pts_fp2, [0, 0, 0, 0])
    # Identity at output: XYZZ identity (ZZ=ZZZ=0) decodes to (0, 0) per
    # the context's convention.
    p = self.p
    self.assertEqual(got[0][0] % p, 0)
    self.assertEqual(got[0][1] % p, 0)
    self.assertEqual(got[1][0] % p, 0)
    self.assertEqual(got[1][1] % p, 0)

  # ------------------------------------------------------------------ #
  #  Single base, single scalar — matches CPU msm_g2                    #
  # ------------------------------------------------------------------ #

  def test_single_point_matches_cpu_msm_g2(self):
    """For small k, k·G2_GEN agrees with the Groth16 CPU msm_g2."""
    g2_gen = G2Affine(G2_GEN_X, G2_GEN_Y)
    for k in (1, 2, 3, 5, 17, 1234):
      with self.subTest(k=k):
        got = self._run_jax_msm([_g2_affine_to_fp2_pair(g2_gen)], [k])
        expected = cpu_msm_g2([g2_gen], [k]).to_affine()
        self._assert_affine_equal(got, expected, label=f"k={k}")

  # ------------------------------------------------------------------ #
  #  Small random batch — matches CPU msm_g2                            #
  # ------------------------------------------------------------------ #

  def test_random_batch_matches_cpu_msm_g2(self):
    """16-point random-scalar MSM matches the Groth16 CPU oracle."""
    rng = random.Random(0xC0FFEE)
    bases = [_g2_multiple(i + 1) for i in range(16)]
    scalars = [rng.randrange(0, 1 << self.msm_ctx.scalar_bits) for _ in bases]

    pts_fp2 = [_g2_affine_to_fp2_pair(p) for p in bases]
    got = self._run_jax_msm(pts_fp2, scalars)
    expected = cpu_msm_g2(bases, scalars).to_affine()
    self._assert_affine_equal(got, expected, label="batch16")


if __name__ == "__main__":
  absltest.main()
