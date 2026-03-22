# Elliptic Curve System

This project's elliptic-curve layer lives in `elliptic_curve_context.py`.
It is designed as a thin curve-specific layer on top of the finite-field system in `finite_field_context.py`.

At the moment, the actively used implementation is:

- `ExtendedTwistedEdwardsContext`

It accepts affine Weierstrass points as external inputs, converts them into extended twisted Edwards coordinates for computation, and converts results back to affine Weierstrass coordinates on output.

## Main Classes

### `EllipticCurveContextBase`

Defines the common interface:

- `to_computational_format()`
- `to_original_format()`
- `point_add()`
- `point_double()`
- `get_finite_field_context()`

It also constructs the underlying finite-field context from:

- `finite_field_context_class`
- `finite_field_parameters`

That means EC arithmetic delegates all field operations to `self.ff_ctx`.

### `CPUWeierstrassAffineContext`

This is a CPU reference implementation for affine Weierstrass arithmetic.

What it is used for:

- reference checking in tests
- validating point addition against simple modular formulas

What it is not:

- not production-oriented
- not fully wired as a public API

Notes:

- `_point_add()` is implemented and used as the correctness reference
- `_point_double()` is implemented as a helper
- public `point_add()`, `point_double()`, `to_computational_format()`, and `to_original_format()` intentionally raise `NotImplementedError`

### `ExtendedTwistedEdwardsContextBase`

This base class holds the curve-geometry conversions:

- `_twist()`
- `_untwist()`
- `_convert_to_edwards_affine()`
- `_convert_to_extended_twisted_edwards()`
- `_convert_to_weierstrass_affine()`

The class stores the curve constants:

- `a`
- `twist_d`
- `alpha`
- `s`
- `MA`
- `MB`
- `t`
- `zero_point = [0, 1, 1, 0]`

These parameters are read from `configurations.toml` in the tests.

### `ExtendedTwistedEdwardsContext`

This is the JAX-backed implementation currently used by the repo.

It provides:

- batched coordinate conversion
- point addition in extended twisted Edwards form
- JAX parameter packaging for `twist_d`
- executable serialization/compilation hooks

## Coordinate Model

### External API format

Inputs and outputs are affine Weierstrass points:

`[x, y]`

For batches:

`[[x0, y0], [x1, y1], ...]`

### Internal computational format

Points are converted to extended twisted Edwards coordinates:

`[X, Y, Z, T]`

Then each coordinate is converted through the finite-field context into RNS form.

For a single point, `to_computational_format()` returns data arranged as:

- shape `(4, 1, num_moduli)`

For a batch of points:

- shape `(4, batch, num_moduli)`

The axis meaning is:

- axis 0: point coordinate (`X`, `Y`, `Z`, `T`)
- axis 1: batch
- axis 2: finite-field residue/modulus slot

## Conversion Flow

### Into computational format

When you call `to_computational_format(point)`:

1. affine Weierstrass point is twisted into Edwards affine coordinates
2. Edwards affine point becomes extended coordinates `[X, Y, 1, X*Y]`
3. each coordinate is encoded by the finite-field context

### Back to original format

When you call `to_original_format(point_m)`:

1. field coordinates are decoded back to integers
2. extended Edwards coordinates are converted to Edwards affine
3. Edwards affine is untwisted back to affine Weierstrass

This keeps the public boundary compatible with standard affine points while using a faster internal model for addition.

## Point Addition

`_point_add()` implements the extended twisted Edwards addition formula in batched JAX form.

High-level structure:

1. multiply coordinate pairs in the field
2. compute intermediate variables `a, b, c, d, e, f, g, h`
3. assemble the output coordinates using more field multiplies

All low-level arithmetic goes through the finite-field context:

- `_modular_multiply()`
- `_modular_add()`
- `_modular_subtract()`
- `_modular_negate()`

Public `point_add()` simply dispatches to:

- a cached compiled kernel, if enabled
- otherwise `_point_add()`

## Point Doubling Status

There is an important current limitation:

- public `point_double()` in `ExtendedTwistedEdwardsContext` raises `ValueError("point_double has logic bug")`

So, in the current state of the repo:

- point addition is the supported public EC operation
- point doubling exists as `_point_double()` but is explicitly marked unsafe for public use
- serialization still has a `point_double` path, but the public API intentionally blocks it

Any documentation or downstream code should treat `point_double()` as unavailable until the logic bug is fixed.

## Required Parameters

An EC context needs:

- `finite_field_context_class`
- `finite_field_parameters`
- `prime`
- `order`
- `a`
- `twist_d`
- `alpha`
- `b`
- `s`
- `MA`
- `MB`
- `t`
- `generator`

Typical construction:

```python
import toml
import utils
import finite_field_context as ff_context
import elliptic_curve_context as ec_context

cfg = toml.load("configurations.toml")
rns_moduli = utils.find_moduli_specified_number(32, 28)

finite_field_parameters = {
    "prime": cfg["ec_parameters_bls12_377_affine"]["prime"],
    "rns_moduli": rns_moduli,
    "precision_bits": 28,
    "radix_bits": 32,
}

ete_cfg = cfg["ec_parameters_bls12_377_extended_twisted_edwards"]
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

ec_ctx = ec_context.ExtendedTwistedEdwardsContext(ec_parameters)
```

## Example Usage

```python
point_batch_1 = [[x1, y1], [x2, y2]]
point_batch_2 = [[u1, v1], [u2, v2]]

point_batch_1_m = ec_ctx.to_computational_format(point_batch_1)
point_batch_2_m = ec_ctx.to_computational_format(point_batch_2)
result_m = ec_ctx.point_add(point_batch_1_m, point_batch_2_m)
result = ec_ctx.to_original_format(result_m)
```

This is the same flow used in `elliptic_curve_test.py`.

## Serialized/Compiled Kernels

`ExtendedTwistedEdwardsContext` also inherits `JaxKernelContextBase`.

Supported cacheable kernel names:

- `point_add`
- `point_double`

The EC shape contract is determined by:

- `parameters["batch_size"]`

which becomes:

- point shape `(4, batch_size, num_moduli)`

Example:

```python
ec_ctx.serialize({"batch_size": 1024})
ec_ctx.compile({"batch_size": 1024})
ec_ctx.set_use_compiled_kernels(True)
```

Even though `point_double` can be serialized internally, the public `point_double()` method is still blocked by the explicit runtime error.

## Testing and Performance Entry Points

- `elliptic_curve_test.py` checks `point_add()` against `CPUWeierstrassAffineContext._point_add()`
- `elliptic_curve_perf_test.py` profiles `_point_add()` and contains scaffolding for `_point_double()`

The correctness story today is therefore:

- point addition: tested
- point doubling: known issue

## Relationship to MSM

`multiscalar_multiplication_context.py` builds on this EC layer.
It relies on:

- `ec_ctx.zero_point`
- `ec_ctx.point_add()`
- `ec_ctx.to_computational_format()`
- `ec_ctx.to_original_format()`

Because MSM repeatedly accumulates buckets with point addition, the correctness and performance of `ExtendedTwistedEdwardsContext.point_add()` are central to the higher-level MSM pipeline.
