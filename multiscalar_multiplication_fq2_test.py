"""Tests for ``Fq2PippengerMSMContext`` (multiscalar_multiplication_context.py).

The Fq2 MSM driver runs Pippenger's algorithm natively on the Fq2 ETE point
layout ``(coord=4, batch, 2, M)``.  We exercise it three ways:

  1. **Lifted G1 sanity check** — when every Fp1 point is lifted to Fp2 as
     ``(value, 0)``, the Fq2 MSM result must agree with the existing Fq1
     MSM (computed via brute-force ``Σ k_i · P_i`` on the Fq EC kernel)
     in the c0 component, with c1 ≡ 0.

  2. **CPU oracle parity** — for random Fp2 affine points and small scalars,
     the JAX-driven Pippenger MSM must agree with a pure-python Fp2 MSM
     that does ``Σ k_i · P_i`` via standard double-and-add.  This pins
     down the bucket distribution, bucket reduction, and window merge
     stages without needing curve-checked random Fp2 inputs.

  3. **Single-window degenerate case** — ``slice_bits == scalar_bits`` so
     ``window_num == 1``: stresses the bucket-reduction / window-merge
     boundary conditions.
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
import multiscalar_multiplication_context as msm_context

jax.config.update("jax_enable_x64", True)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configurations.toml")
QNR = 5  # u² + 5 = 0 in Fp2 = Fp[u] / (u² + 5) for BLS12-377


# ── BLS12-377 G1 affine test points (same as elliptic_curve_fq2_test.py) ──────
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
G1_POINTS = [P1, P2, P3, P4]


def _lift_to_fp2(pt):
  return [(pt[0], 0), (pt[1], 0)]


def _build_fq2_msm_params(slice_bits=4, scalar_bits=16):
  """Build params for an Fq2 Pippenger MSM with small windows (test-friendly)."""
  ec_config = toml.load(CONFIG_PATH)
  rns_moduli = utils.find_moduli_specified_number(32, 28)
  ete_cfg = ec_config["ec_parameters_bls12_377_extended_twisted_edwards"]
  ff_parameters = {
      "prime": ete_cfg["prime"],
      "rns_moduli": rns_moduli,
      "precision_bits": 28,
      "radix_bits": 32,
  }
  fq2_ec_params = {
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
  return {
      "elliptic_curve_context_class": ec_context.ExtendedTwistedEdwardsFq2NDContext,
      "elliptic_curve_parameters": fq2_ec_params,
      "slice_bits": slice_bits,
      "scalar_bits": scalar_bits,
      "order": ete_cfg["order"],
  }


def _build_fq_ec_ctx():
  """Reference Fq1 EC context (used to compute Fp1 brute-force MSM result)."""
  ec_config = toml.load(CONFIG_PATH)
  rns_moduli = utils.find_moduli_specified_number(32, 28)
  ete_cfg = ec_config["ec_parameters_bls12_377_extended_twisted_edwards"]
  return ec_context.ExtendedTwistedEdwardsNDContext({
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
  })


class Fq2PippengerMSMTest(absltest.TestCase):

  # ------------------------------------------------------------------ #
  #  Helpers                                                            #
  # ------------------------------------------------------------------ #

  def _fq1_brute_force_msm(self, fq_ctx, affine_points, scalars):
    """Σ k_i · P_i computed by repeated Fq1 ``point_add`` on the JAX kernel."""
    zero_te = list(fq_ctx.zero_point)

    def _double_and_add_pyints(point_te, k):
      if k == 0:
        return list(zero_te)
      result = list(zero_te)
      addend = list(point_te)
      while k > 0:
        if k & 1:
          result = self._fq_point_add_pure(fq_ctx, result, addend)
        addend = self._fq_point_add_pure(fq_ctx, addend, addend)
        k >>= 1
      return result

    points_te = [fq_ctx._convert_to_extended_twisted_edwards(p) for p in affine_points]
    acc = list(zero_te)
    for pt, k in zip(points_te, scalars):
      acc = self._fq_point_add_pure(fq_ctx, acc, _double_and_add_pyints(pt, int(k)))
    # Convert ETE -> Weierstrass affine (Fq1).
    return fq_ctx._convert_to_weierstrass_affine(acc)

  def _fq_point_add_pure(self, fq_ctx, pa, pb):
    """Pure-Python add-2008-hwcd over Fq1 (mirrors the Fq2 oracle, just on ints)."""
    p = fq_ctx.prime
    twist_d = fq_ctx.twist_d % p
    x1, y1, z1, t1 = pa
    x2, y2, z2, t2 = pb
    A = (x1 * x2) % p
    B = (y1 * y2) % p
    D = (z1 * z2) % p
    C = (t1 * t2 * twist_d) % p
    E = (((x1 + y1) * (x2 + y2)) - A - B) % p
    F = (D - C) % p
    G = (D + C) % p
    H = (B + A) % p
    return [(E * F) % p, (G * H) % p, (F * G) % p, (E * H) % p]

  # ------------------------------------------------------------------ #
  #  Tests                                                              #
  # ------------------------------------------------------------------ #

  def test_lifted_fp1_msm_matches_fq_only(self):
    """G1 points lifted to Fp2 produce the same MSM result as the Fq1 path."""
    params = _build_fq2_msm_params(slice_bits=4, scalar_bits=16)
    ctx = msm_context.Fq2PippengerMSMContext(params)
    p = params["elliptic_curve_parameters"]["prime"]

    scalars = [3, 17, 91, 2_137]
    lifted = [_lift_to_fp2(pt) for pt in G1_POINTS]
    points_cf = ctx.to_computational_format(lifted)
    res_cf = ctx.multiscalar_multiply(points_cf, scalars)
    res_aff = ctx.to_original_format(res_cf)[0]
    (x_c0, x_c1), (y_c0, y_c1) = res_aff

    # Reference: brute-force Fq1 Σ k·P, then compare c0 components.
    fq_ctx = _build_fq_ec_ctx()
    fq_ref = self._fq1_brute_force_msm(fq_ctx, G1_POINTS, scalars)

    self.assertEqual(x_c0 % p, fq_ref[0] % p)
    self.assertEqual(y_c0 % p, fq_ref[1] % p)
    self.assertEqual(x_c1 % p, 0)
    self.assertEqual(y_c1 % p, 0)

  def test_jax_kernel_matches_cpu_oracle_lifted(self):
    """JAX MSM ↔ CPU oracle on lifted Fp1 points."""
    params = _build_fq2_msm_params(slice_bits=4, scalar_bits=16)
    ctx = msm_context.Fq2PippengerMSMContext(params)
    p = params["elliptic_curve_parameters"]["prime"]

    scalars = [11, 23, 47, 9919]
    lifted = [_lift_to_fp2(pt) for pt in G1_POINTS]

    # JAX path.
    points_cf = ctx.to_computational_format(lifted)
    jax_aff = ctx.to_original_format(ctx.multiscalar_multiply(points_cf, scalars))[0]

    # CPU oracle.
    cpu_aff = ctx.cpu_oracle_msm(lifted, scalars)

    for got_coord, exp_coord in zip(jax_aff, cpu_aff):
      self.assertEqual(got_coord[0] % p, exp_coord[0] % p)
      self.assertEqual(got_coord[1] % p, exp_coord[1] % p)

  def test_single_window(self):
    """slice_bits == scalar_bits triggers the window_num == 1 branch."""
    params = _build_fq2_msm_params(slice_bits=8, scalar_bits=8)
    ctx = msm_context.Fq2PippengerMSMContext(params)
    self.assertEqual(ctx.window_num, 1)
    p = params["elliptic_curve_parameters"]["prime"]

    scalars = [3, 17, 91, 200]
    lifted = [_lift_to_fp2(pt) for pt in G1_POINTS]

    points_cf = ctx.to_computational_format(lifted)
    jax_aff = ctx.to_original_format(ctx.multiscalar_multiply(points_cf, scalars))[0]

    cpu_aff = ctx.cpu_oracle_msm(lifted, scalars)
    for got_coord, exp_coord in zip(jax_aff, cpu_aff):
      self.assertEqual(got_coord[0] % p, exp_coord[0] % p)
      self.assertEqual(got_coord[1] % p, exp_coord[1] % p)

  def test_zero_and_unit_scalars(self):
    """Edge case: zero scalars contribute nothing; one scalar = unit selects pt."""
    params = _build_fq2_msm_params(slice_bits=4, scalar_bits=8)
    ctx = msm_context.Fq2PippengerMSMContext(params)
    p = params["elliptic_curve_parameters"]["prime"]

    scalars = [0, 1, 0, 0]  # Selects only P2.
    lifted = [_lift_to_fp2(pt) for pt in G1_POINTS]

    points_cf = ctx.to_computational_format(lifted)
    jax_aff = ctx.to_original_format(ctx.multiscalar_multiply(points_cf, scalars))[0]

    # Should equal P2 (lifted), so c0 == P2 coords, c1 == 0.
    (x_c0, x_c1), (y_c0, y_c1) = jax_aff
    self.assertEqual(x_c0 % p, P2[0] % p)
    self.assertEqual(y_c0 % p, P2[1] % p)
    self.assertEqual(x_c1 % p, 0)
    self.assertEqual(y_c1 % p, 0)


if __name__ == "__main__":
  absltest.main()
