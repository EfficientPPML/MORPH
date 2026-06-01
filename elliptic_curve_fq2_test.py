"""Tests for ExtendedTwistedEdwardsFq2NDContext (elliptic_curve_context.py).

The new context performs extended-twisted-Edwards arithmetic over Fq2 =
Fq[u] / (u² + qnr).  We exercise it three ways:

  1. round-trip in/out conversion preserves input,
  2. CPU oracle ``_point_add`` (pure-Python Fp2 math) matches the JAX
     kernel ``point_add`` on random Fp2 points,
  3. when Fp1 points are *lifted* into Fp2 as ``(x_fp, 0)``, the Fq2 EC
     context produces results whose c0 components agree with the
     existing Fq-only ``ExtendedTwistedEdwardsNDContext``.  This is a
     sanity check that the Fp2 generalisation reduces to the Fp case
     on the Fp ⊆ Fp2 subfield.
"""

import os
import random

import numpy as np
import jax
import jax.numpy as jnp
from absl.testing import absltest

import toml
import utils
import finite_field_context as ff_context
import field_extension_context as fe_context
import elliptic_curve_context as ec_context

jax.config.update("jax_enable_x64", True)

# ── BLS12-377 parameters ──────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configurations.toml")
QNR = 5  # u² = -5 in Fp2 = Fp[u]/(u²+5) for BLS12-377


def _build_fq2_ec_parameters():
  """Build params for ExtendedTwistedEdwardsFq2NDContext using the BLS12-377
  Fp twist constants lifted into Fp2 (each constant becomes (c, 0))."""
  ec_config = toml.load(CONFIG_PATH)
  rns_moduli = utils.find_moduli_specified_number(32, 28)
  ete_cfg = ec_config["ec_parameters_bls12_377_extended_twisted_edwards"]
  finite_field_parameters = {
      "prime": ete_cfg["prime"],
      "rns_moduli": rns_moduli,
      "precision_bits": 28,
      "radix_bits": 32,
  }
  return {
      "prime": ete_cfg["prime"],
      "order": ete_cfg["order"],
      "quadratic_non_residue": QNR,
      "finite_field_context_class": ff_context.DRNSlazyContext,
      "finite_field_parameters": finite_field_parameters,
      "field_extension_context_class": fe_context.Fq2Context,
      "field_extension_parameters": {
          "prime": ete_cfg["prime"],
          "quadratic_non_residue": QNR,
          "finite_field_context_class": ff_context.DRNSlazyContext,
          "finite_field_parameters": finite_field_parameters,
      },
      "a":        ete_cfg["a"],
      "twist_d":  ete_cfg["d"],
      "alpha":    ete_cfg["alpha"],
      "b":        ete_cfg["b"],
      "s":        ete_cfg["s"],
      "MA":       ete_cfg["MA"],
      "MB":       ete_cfg["MB"],
      "t":        ete_cfg["t"],
      "generator": ete_cfg["generator"],
  }


def _build_fq_ec_parameters():
  ec_config = toml.load(CONFIG_PATH)
  rns_moduli = utils.find_moduli_specified_number(32, 28)
  ete_cfg = ec_config["ec_parameters_bls12_377_extended_twisted_edwards"]
  return {
      "finite_field_context_class": ff_context.DRNSlazyContext,
      "finite_field_parameters": {
          "prime": ete_cfg["prime"],
          "rns_moduli": rns_moduli,
          "precision_bits": 28,
          "radix_bits": 32,
      },
      "prime":   ete_cfg["prime"],
      "order":   ete_cfg["order"],
      "a":       ete_cfg["a"],
      "twist_d": ete_cfg["d"],
      "alpha":   ete_cfg["alpha"],
      "b":       ete_cfg["b"],
      "s":       ete_cfg["s"],
      "MA":      ete_cfg["MA"],
      "MB":      ete_cfg["MB"],
      "t":       ete_cfg["t"],
      "generator": ete_cfg["generator"],
  }


# Two distinct test points on BLS12-377 G1 (Fp).
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


def _lift_to_fp2(pt):
  """Lift an Fp affine point ``[x, y]`` to Fp2 ``[(x, 0), (y, 0)]``."""
  return [(pt[0], 0), (pt[1], 0)]


class Fq2ExtendedTwistedEdwardsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.params = _build_fq2_ec_parameters()
    self.ctx = ec_context.ExtendedTwistedEdwardsFq2NDContext(self.params)

  # ------------------------------------------------------------------ #
  #  Round-trip                                                         #
  # ------------------------------------------------------------------ #

  def test_roundtrip_single(self):
    pt = _lift_to_fp2(P1)
    cf = self.ctx.to_computational_format(pt)
    self.assertEqual(cf.shape[0], 4)
    self.assertEqual(cf.shape[-2], 2)
    out = self.ctx.to_original_format(cf)  # length-1 list wrapping the point
    p = self.params["prime"]
    (xc0, xc1), (yc0, yc1) = out[0]
    self.assertEqual(xc0 % p, P1[0] % p)
    self.assertEqual(yc0 % p, P1[1] % p)
    self.assertEqual(xc1 % p, 0)
    self.assertEqual(yc1 % p, 0)

  def test_roundtrip_batch(self):
    batch = [_lift_to_fp2(P1), _lift_to_fp2(P2), _lift_to_fp2(P3)]
    cf = self.ctx.to_computational_format(batch)
    self.assertEqual(cf.shape[0], 4)
    self.assertEqual(cf.shape[1], 3)
    self.assertEqual(cf.shape[-2], 2)
    out = self.ctx.to_original_format(cf)
    p = self.params["prime"]
    for got, src in zip(out, [P1, P2, P3]):
      (xc0, xc1), (yc0, yc1) = got
      self.assertEqual(xc0 % p, src[0] % p)
      self.assertEqual(yc0 % p, src[1] % p)
      self.assertEqual(xc1 % p, 0)
      self.assertEqual(yc1 % p, 0)

  # ------------------------------------------------------------------ #
  #  CPU oracle (pure-Python Fp2) ↔ JAX kernel parity                    #
  # ------------------------------------------------------------------ #

  def test_cpu_oracle_matches_jax_kernel_lifted(self):
    """Lift two Fp points to Fp2 and verify both backends agree."""
    a_aff = _lift_to_fp2(P1)
    b_aff = _lift_to_fp2(P3)

    # CPU oracle: convert to extended TE, do _point_add, convert back.
    a_ext = self.ctx._convert_to_extended_twisted_edwards(a_aff)
    b_ext = self.ctx._convert_to_extended_twisted_edwards(b_aff)
    cpu_ext = self.ctx._point_add(a_ext, b_ext)
    cpu_aff = self.ctx._convert_to_weierstrass_affine(cpu_ext)

    # JAX kernel.  Single-point inputs come back wrapped in a length-1
    # list (mirrors the Fq-only context); unwrap before comparing.
    a_cf = self.ctx.to_computational_format(a_aff)
    b_cf = self.ctx.to_computational_format(b_aff)
    jax_aff = self.ctx.to_original_format(self.ctx.point_add(a_cf, b_cf))[0]

    p = self.params["prime"]
    for cpu_coord, jax_coord in zip(cpu_aff, jax_aff):
      self.assertEqual(cpu_coord[0] % p, jax_coord[0] % p)
      self.assertEqual(cpu_coord[1] % p, jax_coord[1] % p)

  def test_cpu_oracle_matches_jax_kernel_random_fp2(self):
    """Random Fp2 points (non-trivial c1) — CPU vs JAX point_add."""
    p = self.params["prime"]
    rng = random.Random(0xC0DE)

    def rand_pt():
      # Build a random TE affine pair directly (not from a curve), so we
      # exercise the addition formula without needing a known-on-curve
      # Fp2 point.  The TE addition formula is a complete algebraic
      # identity for any inputs in (Fp2)⁴ encoded as [x, y, z, t]; we
      # just need both backends to agree.
      def rand_fp2():
        return (rng.randrange(p), rng.randrange(p))
      x, y = rand_fp2(), rand_fp2()
      z = (1, 0)
      t = self.ctx._fp2_mul(x, y)
      return [x, y, z, t]

    a = rand_pt()
    b = rand_pt()
    cpu = self.ctx._point_add(a, b)

    a_arr = self.ctx.fe_ctx.to_computational_format(a)  # (4, 2, M)
    b_arr = self.ctx.fe_ctx.to_computational_format(b)
    a_arr = jnp.expand_dims(a_arr, 1)  # (4, 1, 2, M) — single-element batch
    b_arr = jnp.expand_dims(b_arr, 1)
    out_arr = self.ctx.point_add(a_arr, b_arr)  # (4, 1, 2, M)
    jax_pt = self.ctx.fe_ctx.to_original_format(out_arr.squeeze(1))

    for cpu_coord, jax_coord in zip(cpu, jax_pt):
      self.assertEqual(cpu_coord[0] % p, jax_coord[0] % p)
      self.assertEqual(cpu_coord[1] % p, jax_coord[1] % p)

  # ------------------------------------------------------------------ #
  #  Fp1 ⊆ Fp2 reduction sanity                                          #
  # ------------------------------------------------------------------ #

  def test_lifted_fp1_matches_fq_only_context(self):
    """Lifting Fp points to Fp2 and adding via the Fq2 context yields
    results whose c0 component matches the Fq-only context, with c1=0."""
    fq_ctx = ec_context.ExtendedTwistedEdwardsNDContext(_build_fq_ec_parameters())

    batch_a = [P1, P2]
    batch_b = [P3, P4]

    # Fq baseline.
    fq_a = fq_ctx.to_computational_format(batch_a)
    fq_b = fq_ctx.to_computational_format(batch_b)
    fq_result = fq_ctx.to_original_format(fq_ctx.point_add(fq_a, fq_b))

    # Fq2 lifted.
    lifted_a = [_lift_to_fp2(pt) for pt in batch_a]
    lifted_b = [_lift_to_fp2(pt) for pt in batch_b]
    fq2_a = self.ctx.to_computational_format(lifted_a)
    fq2_b = self.ctx.to_computational_format(lifted_b)
    fq2_result = self.ctx.to_original_format(self.ctx.point_add(fq2_a, fq2_b))

    p = self.params["prime"]
    self.assertLen(fq2_result, 2)
    for fq_pt, fq2_pt in zip(fq_result, fq2_result):
      # fq_pt : [x_int, y_int]
      # fq2_pt: [(x_c0, x_c1), (y_c0, y_c1)]
      self.assertEqual(fq2_pt[0][0] % p, fq_pt[0] % p)
      self.assertEqual(fq2_pt[1][0] % p, fq_pt[1] % p)
      self.assertEqual(fq2_pt[0][1] % p, 0)
      self.assertEqual(fq2_pt[1][1] % p, 0)

  # ------------------------------------------------------------------ #
  #  Multi-D batch                                                       #
  # ------------------------------------------------------------------ #

  def test_2d_batch_matches_per_pair(self):
    """2D batch results match running each pair individually."""
    batch_a = [[_lift_to_fp2(P1), _lift_to_fp2(P2)],
               [_lift_to_fp2(P2), _lift_to_fp2(P1)]]
    batch_b = [[_lift_to_fp2(P3), _lift_to_fp2(P4)],
               [_lift_to_fp2(P4), _lift_to_fp2(P3)]]

    a_cf = self.ctx.to_computational_format(batch_a)
    b_cf = self.ctx.to_computational_format(batch_b)
    self.assertEqual(a_cf.shape[0], 4)
    self.assertEqual(a_cf.shape[1], 2)
    self.assertEqual(a_cf.shape[2], 2)
    self.assertEqual(a_cf.shape[3], 2)

    nd_result = self.ctx.to_original_format(self.ctx.point_add(a_cf, b_cf))
    self.assertLen(nd_result, 2)
    self.assertLen(nd_result[0], 2)

    p = self.params["prime"]
    for i in range(2):
      ref_a = self.ctx.to_computational_format(batch_a[i])
      ref_b = self.ctx.to_computational_format(batch_b[i])
      ref = self.ctx.to_original_format(self.ctx.point_add(ref_a, ref_b))
      for got, exp in zip(nd_result[i], ref):
        # Each got/exp is a [(x_c0, x_c1), (y_c0, y_c1)] pair.
        for got_coord, exp_coord in zip(got, exp):
          self.assertEqual(got_coord[0] % p, exp_coord[0] % p)
          self.assertEqual(got_coord[1] % p, exp_coord[1] % p)


if __name__ == "__main__":
  absltest.main()
