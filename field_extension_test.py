"""Tests for Fq2Context (field_extension_context.py).

Reference arithmetic is pure-Python, matching the formulas in
Groth16/bls12_377/field.py.  Only the DRNSlazyContext backend is tested
since that is the backend used in production.
"""

import numpy as np
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

import utils
import finite_field_context as ff_context
import field_extension_context as fe_context

jax.config.update("jax_enable_x64", True)

# ── BLS12-377 parameters ──────────────────────────────────────────────────────

P = 258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177
QNR = 5  # u² = -5 in Fp2 = Fp[u]/(u²+5)

# ── Pure-Python Fp2 reference ─────────────────────────────────────────────────

def _fp_add(a, b): return (a + b) % P
def _fp_sub(a, b): return (a - b) % P
def _fp_mul(a, b): return (a * b) % P
def _fp_neg(a):    return (-a) % P

def ref_fp2_add(a, b):
    return (_fp_add(a[0], b[0]), _fp_add(a[1], b[1]))

def ref_fp2_sub(a, b):
    return (_fp_sub(a[0], b[0]), _fp_sub(a[1], b[1]))

def ref_fp2_neg(a):
    return (_fp_neg(a[0]), _fp_neg(a[1]))

def ref_fp2_mul(a, b):
    # (a0 + a1·u)(b0 + b1·u) = (a0·b0 − 5·a1·b1) + (a0·b1 + a1·b0)·u
    a0, a1 = a;  b0, b1 = b
    return (
        _fp_sub(_fp_mul(a0, b0), _fp_mul(QNR, _fp_mul(a1, b1))),
        _fp_add(_fp_mul(a0, b1), _fp_mul(a1, b0)),
    )

def ref_fp2_sq(a):
    # (a0 + a1·u)² = (a0²−5·a1²) + 2·a0·a1·u
    a0, a1 = a
    return (
        _fp_sub(_fp_mul(a0, a0), _fp_mul(QNR, _fp_mul(a1, a1))),
        _fp_mul(2, _fp_mul(a0, a1)),
    )

def ref_fp2_conj(a):
    return (a[0], _fp_neg(a[1]))

# ── Shared test vectors ───────────────────────────────────────────────────────
# Elements are (c0, c1) pairs of BLS12-377 Fp values.

A = (
    0x00BE4FBE5D03CE926E40E058BBDC3269C78CFAFED39796CD13EC8E9B0072DB2538DFFBCA05804574D9E2FF7EEB1DE219,
    0x008848DEFE740A67C8FC6225BF87FF5485951E2CAA9D41BB188282C8BD37CB5CD5481512FFCD394EEAB9B16EB21BE9EF,
)
B = (
    0x0082A0ED372BFAB8198D0667A1DC5E299C1F6C8FEB0ACD4D05A228325117BE63EAE5BABE6807F41C6C8016BDAC251CFE,
    0x01914A69C5102EFF1F674F5D30AFEEC4BD7FB348CA3E52D96D182AD44FB82305C2FE3D3634A9591AFD82DE55559C8EA6,
)
# A second distinct pair to exercise batched paths.
C = (
    0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000000,
    0x00BE4FBE5D03CE926E40E058BBDC3269C78CFAFED39796CD13EC8E9B0072DB2538DFFBCA05804574D9E2FF7EEB1DE219,
)
D = (
    0x008848DEFE740A67C8FC6225BF87FF5485951E2CAA9D41BB188282C8BD37CB5CD5481512FFCD394EEAB9B16EB21BE9EF,
    0x0082A0ED372BFAB8198D0667A1DC5E299C1F6C8FEB0ACD4D05A228325117BE63EAE5BABE6807F41C6C8016BDAC251CFE,
)

# Named-parameter triples: (test_name, fq2_elem_a, fq2_elem_b)
PAIRS = [
    ("pair_AB", A, B),
    ("pair_CD", C, D),
    ("pair_same", A, A),  # a * a must equal sq(a)
]

# ── Context factory ───────────────────────────────────────────────────────────

def make_ctx():
    rns_moduli = utils.find_moduli_specified_number(32, 28)
    return fe_context.Fq2Context({
        "prime": P,
        "quadratic_non_residue": QNR,
        "finite_field_context_class": ff_context.DRNSlazyContext,
        "finite_field_parameters": {
            "prime": P,
            "rns_moduli": rns_moduli,
            "precision_bits": 28,
            "radix_bits": 32,
        },
    })

# ── Helpers ───────────────────────────────────────────────────────────────────

def assert_fp2_equal(result, expected, msg=""):
    """Check two (c0, c1) Fp2 tuples are equal mod P."""
    r0, r1 = result
    e0, e1 = expected
    np.testing.assert_equal(r0 % P, e0 % P, err_msg=f"{msg} c0 mismatch")
    np.testing.assert_equal(r1 % P, e1 % P, err_msg=f"{msg} c1 mismatch")

def roundtrip(ctx, elem):
    """Convert a single (c0, c1) pair through the context and back.

    to_computational_format on a single pair produces shape [1, 2, moduli_dim].
    to_original_format always returns a list, so index [0] to get the pair.
    """
    return ctx.to_original_format(ctx.to_computational_format(elem))[0]

# ── Tests ─────────────────────────────────────────────────────────────────────

class Fq2ContextTest(parameterized.TestCase):

    # ------------------------------------------------------------------ #
    #  Round-trip: to_computational_format → to_original_format           #
    # ------------------------------------------------------------------ #

    def test_roundtrip_single(self):
        ctx = make_ctx()
        got = roundtrip(ctx, A)
        assert_fp2_equal(got, (A[0] % P, A[1] % P), "single roundtrip")

    def test_roundtrip_batch(self):
        ctx = make_ctx()
        batch = [A, B, C]
        cf = ctx.to_computational_format(batch)
        self.assertEqual(cf.shape[1], 2, "batch format should have 2 components")
        results = ctx.to_original_format(cf)
        for i, (elem, res) in enumerate(zip(batch, results)):
            assert_fp2_equal(res, (elem[0] % P, elem[1] % P), f"batch elem {i}")

    # ------------------------------------------------------------------ #
    #  Addition                                                           #
    # ------------------------------------------------------------------ #

    @parameterized.named_parameters(*PAIRS)
    def test_add(self, a, b):
        ctx = make_ctx()
        a_cf = ctx.to_computational_format([a, b])
        b_cf = ctx.to_computational_format([b, a])
        res = ctx.to_original_format(ctx.modular_add(a_cf, b_cf))
        for got, ea, eb in zip(res, [a, b], [b, a]):
            assert_fp2_equal(got, ref_fp2_add(ea, eb), "add")

    # ------------------------------------------------------------------ #
    #  Subtraction                                                        #
    # ------------------------------------------------------------------ #

    @parameterized.named_parameters(*PAIRS)
    def test_sub(self, a, b):
        ctx = make_ctx()
        a_cf = ctx.to_computational_format([a, b])
        b_cf = ctx.to_computational_format([b, a])
        res = ctx.to_original_format(ctx.modular_subtract(a_cf, b_cf))
        for got, ea, eb in zip(res, [a, b], [b, a]):
            assert_fp2_equal(got, ref_fp2_sub(ea, eb), "sub")

    # ------------------------------------------------------------------ #
    #  Negation                                                           #
    # ------------------------------------------------------------------ #

    @parameterized.named_parameters(*PAIRS)
    def test_neg(self, a, b):
        ctx = make_ctx()
        a_cf = ctx.to_computational_format([a, b])
        res = ctx.to_original_format(ctx.modular_negate(a_cf))
        for got, elem in zip(res, [a, b]):
            assert_fp2_equal(got, ref_fp2_neg(elem), "neg")

    # ------------------------------------------------------------------ #
    #  Multiplication                                                     #
    # ------------------------------------------------------------------ #

    @parameterized.named_parameters(*PAIRS)
    def test_mul(self, a, b):
        ctx = make_ctx()
        a_cf = ctx.to_computational_format([a, b])
        b_cf = ctx.to_computational_format([b, a])
        res = ctx.to_original_format(ctx.modular_multiply(a_cf, b_cf))
        for got, ea, eb in zip(res, [a, b], [b, a]):
            assert_fp2_equal(got, ref_fp2_mul(ea, eb), "mul")

    # ------------------------------------------------------------------ #
    #  Squaring                                                           #
    # ------------------------------------------------------------------ #

    @parameterized.named_parameters(*PAIRS)
    def test_square(self, a, b):
        ctx = make_ctx()
        a_cf = ctx.to_computational_format([a, b])
        res = ctx.to_original_format(ctx.modular_square(a_cf))
        for got, elem in zip(res, [a, b]):
            assert_fp2_equal(got, ref_fp2_sq(elem), "sq")

    @parameterized.named_parameters(*PAIRS)
    def test_square_matches_mul(self, a, b):
        """modular_square(a) must equal modular_multiply(a, a)."""
        ctx = make_ctx()
        a_cf = ctx.to_computational_format([a, b])
        sq_res  = ctx.to_original_format(ctx.modular_square(a_cf))
        mul_res = ctx.to_original_format(ctx.modular_multiply(a_cf, a_cf))
        for sq, ml in zip(sq_res, mul_res):
            assert_fp2_equal(sq, ml, "sq vs mul")

    # ------------------------------------------------------------------ #
    #  Conjugate                                                          #
    # ------------------------------------------------------------------ #

    @parameterized.named_parameters(*PAIRS)
    def test_conj(self, a, b):
        ctx = make_ctx()
        a_cf = ctx.to_computational_format([a, b])
        res = ctx.to_original_format(ctx.modular_conjugate(a_cf))
        for got, elem in zip(res, [a, b]):
            assert_fp2_equal(got, ref_fp2_conj(elem), "conj")

    # ------------------------------------------------------------------ #
    #  Algebraic identities                                               #
    # ------------------------------------------------------------------ #

    def test_add_neg_is_zero(self):
        """a + (-a) == (0, 0)."""
        ctx = make_ctx()
        a_cf = ctx.to_computational_format([A])
        res = ctx.to_original_format(ctx.modular_add(a_cf, ctx.modular_negate(a_cf)))
        assert_fp2_equal(res[0], (0, 0), "a + (-a)")

    def test_mul_commutativity(self):
        """a*b == b*a."""
        ctx = make_ctx()
        a_cf = ctx.to_computational_format([A])
        b_cf = ctx.to_computational_format([B])
        ab = ctx.to_original_format(ctx.modular_multiply(a_cf, b_cf))
        ba = ctx.to_original_format(ctx.modular_multiply(b_cf, a_cf))
        assert_fp2_equal(ab[0], ba[0], "commutativity")

    def test_sub_is_add_neg(self):
        """a - b == a + (-b)."""
        ctx = make_ctx()
        a_cf = ctx.to_computational_format([A])
        b_cf = ctx.to_computational_format([B])
        sub = ctx.to_original_format(ctx.modular_subtract(a_cf, b_cf))
        anb = ctx.to_original_format(ctx.modular_add(a_cf, ctx.modular_negate(b_cf)))
        assert_fp2_equal(sub[0], anb[0], "sub == add(neg)")

    def test_conj_twice_is_identity(self):
        """conj(conj(a)) == a."""
        ctx = make_ctx()
        a_cf = ctx.to_computational_format([A])
        res = ctx.to_original_format(ctx.modular_conjugate(ctx.modular_conjugate(a_cf)))
        assert_fp2_equal(res[0], (A[0] % P, A[1] % P), "double conj")

    # ------------------------------------------------------------------ #
    #  Scale by Fq scalar                                                 #
    # ------------------------------------------------------------------ #

    def test_scale_fq(self):
        """modular_scale_fq(a, s) == (s·c0, s·c1)."""
        ctx = make_ctx()
        s_int = 7
        fq = ctx.finite_field_context
        a_cf = ctx.to_computational_format([A])
        s_cf = fq.to_computational_format([s_int])
        res  = ctx.to_original_format(ctx.modular_scale_fq(a_cf, s_cf))
        exp  = (_fp_mul(A[0], s_int), _fp_mul(A[1], s_int))
        assert_fp2_equal(res[0], exp, "scale_fq")


if __name__ == "__main__":
    absltest.main()
