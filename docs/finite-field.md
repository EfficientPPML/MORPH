# Finite Field System

This project's finite-field layer is centered on `DRNSlazyContext` in `finite_field_context.py`.
It implements prime-field arithmetic over a large prime using:

- RNS (Residue Number System) representation
- Montgomery-style reduction inside each RNS modulus
- a CRNS reconstruction/reduction step to bring products back into the field
- JAX kernels for batched execution and optional executable caching

## Main Classes

### `FiniteFieldContextBase`

Defines the public interface:

- `to_computational_format()`
- `to_original_format()`
- `modular_multiply()`
- `modular_add()`
- `modular_subtract()`
- `modular_negate()`
- `modular_reduce()`

### `RNSContextBase`

Adds shared RNS machinery:

- stores `prime`, `rns_moduli`, `total_modulus`, and `precision_bits`
- computes CRT factors for reconstruction
- provides helpers for elementwise integer operations
- implements CRNS support used after multiplication

The key invariant is:

- `prod(rns_moduli) > prime`

Without that, reconstruction modulo the field prime is not valid.

### `DRNSlazyContextBase`

Adds the DRNS-lazy specific pieces:

- `radix_bits`
- modular inverses of each modulus with respect to `2^radix_bits`
- CRNS precomputation data: `matrix_E`, `vector_f_T`, `vector_g`
- `_elementwise_montgomery_reduce()`

This layer is still pure Python/list-based setup logic.

### `DRNSlazyContext`

This is the concrete JAX-backed implementation used by the tests and EC layer.

It adds:

- recursive conversion between Python integers/lists and RNS tensors
- JAX parameter packing in `_init_jax_parameters()`
- vectorized Montgomery reduction in `_jax_montgomery_reduce()`
- vectorized CRNS reduction in `_jax_crns()`
- public arithmetic entrypoints that can run either compiled kernels or the local implementations

## Data Representation

### Original format

Original values are normal Python integers modulo the field prime.

Examples:

- a single field element: `12345`
- a batch of elements: `[a0, a1, a2]`

### Computational format

`to_computational_format()` converts each element into an RNS vector with one residue per modulus.

For one integer:

- output shape: `(num_moduli,)`

For a batch of integers:

- output shape: `(batch, num_moduli)`

Internally, each residue is also shifted into a Montgomery-friendly form:

`((a % m) << radix_bits) % m`

### Conversion back

`to_original_format()` does:

1. Montgomery reduction per modulus
2. CRT reconstruction modulo `prod(rns_moduli)`
3. final reduction modulo `prime`

## Arithmetic Flow

### Multiplication

`modular_multiply(a, b)` follows this pipeline:

1. multiply corresponding residues in `uint64`
2. run Montgomery reduction
3. run CRNS reduction
4. run Montgomery reduction again

This is the core high-performance path.

### Addition and subtraction

`modular_add()` and `modular_subtract()` are intentionally lazy:

- `modular_add(a, b)` returns `a + b`
- `modular_subtract(a, b)` returns `a + modular_negate(b)`
- `modular_negate(a)` uses precomputed modulus offsets

These functions do not immediately canonicalize every residue. That is consistent with the "lazy" design, where normalization can be deferred until a later reduction or conversion back to original integers.

### Explicit reduction

`modular_reduce()` applies the CRNS + Montgomery cleanup path to bring a computational-format tensor back into the expected reduced range.

## Required Parameters

`DRNSlazyContext` expects a parameter dictionary with:

- `prime`: target field prime
- `rns_moduli`: tuple/list of pairwise coprime moduli
- `precision_bits`: fixed-point precision used by CRNS
- `radix_bits`: radix size for Montgomery reduction

Typical setup in this repo:

```python
import utils
from finite_field_context import DRNSlazyContext

prime = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
rns_moduli = utils.find_moduli_specified_number(32, 28)

ctx = DRNSlazyContext(
    {
        "prime": prime,
        "rns_moduli": rns_moduli,
        "precision_bits": 28,
        "radix_bits": 32,
    }
)
```

## Example Usage

```python
value_a = [11, 29]
value_b = [7, 5]

a_m = ctx.to_computational_format(value_a)
b_m = ctx.to_computational_format(value_b)
c_m = ctx.modular_multiply(a_m, b_m)
c = ctx.to_original_format(c_m)
```

The test in `finite_field_test.py` uses exactly this style and compares the decoded result against Python modular arithmetic.

## JAX and Cached Kernels

`DRNSlazyContext` inherits `JaxKernelContextBase`, so it supports:

- `serialize(parameters)`
- `compile(parameters)`
- `set_use_compiled_kernels(True/False)`

For finite-field kernels, the cacheable operations are:

- `modular_multiply`
- `modular_add`
- `modular_subtract`
- `modular_reduce`
- `modular_negate`

The required runtime shape is derived from:

- `parameters["batch_shape"]`

That shape is converted into:

- operand shape = `batch_shape + (num_moduli,)`

Example:

```python
ctx.serialize({"batch_shape": (1024,)})
ctx.compile({"batch_shape": (1024,)})
ctx.set_use_compiled_kernels(True)
```

## Helper Utilities

Useful helpers from `utils.py`:

- `modular_inverse()`
- `compute_crt_factors()`
- `to_rns()`
- `rns_reconstruct()`
- `find_moduli_specified_number()`

`find_moduli_specified_number()` is the helper most frequently used in tests to build a suitable RNS basis.

## Testing and Performance Entry Points

- `finite_field_test.py` validates modular multiplication correctness
- `finite_field_perf_test.py` profiles `_modular_multiply()` across batch sizes

The performance test uses `Profiler` and `KernelWrapper` from `profiler.py`, not the context-level `serialize()/compile()` cache.

## Practical Notes

- The implementation assumes 32-bit RNS residues in the JAX path.
- `precision_bits` and the RNS basis must be chosen carefully; the warning checks exist but are currently disabled.
- Because the implementation is lazy, not every intermediate tensor is canonically reduced after every operation.
- `to_original_format()` is the safest way to validate correctness at API boundaries.
