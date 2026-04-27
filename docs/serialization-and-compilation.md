# Serialization and Compilation System

This project has two separate but related JAX execution workflows:

1. immediate in-memory compilation with `jax.jit(...).lower(...).compile()`
2. persistent executable caching to disk through `store_jax_executable()` and `load_jax_executable()`

The second workflow is what the finite-field and elliptic-curve contexts expose through their `serialize()` and `compile()` methods.

## Where It Lives

The core implementation is in `utils.py`:

- `JaxKernelContextBase`
- `jax_jit_lower_compile()`
- `store_jax_executable()`
- `load_jax_executable()`
- `hash_args()`

There is also an older serialized-kernel path:

- `store_jax_kernel()`
- `load_jax_kernel()`

but the current context classes use the executable-based API, not the older exported-kernel API.

## Config File

`configurations.toml` controls two important pieces:

- `serialized_jax_kernel_dir`
- `hash_length`

Current defaults in this repo:

- serialized cache directory: `./deployments/`
- hash length: `4`

That means cached executables are written under `deployments/`, and their names use short hash suffixes.

## Base Interface

`JaxKernelContextBase` is a minimal mixin that gives contexts:

- `compiled_kernels`
- `use_compiled_kernels`
- `set_use_compiled_kernels()`

It also establishes the expected interface:

- `serialize()`
- `compile()`
- `context_hash()`

Concrete contexts implement those methods themselves.

## Hashing Model

The cache keys are intentionally split into two layers.

### 1. Context hash

Each context defines `context_hash()` from the properties that change kernel behavior.

Examples:

- `DRNSlazyContext.context_hash()` includes the class name, field prime, RNS moduli, `precision_bits`, and `radix_bits`
- `ExtendedTwistedEdwardsContext.context_hash()` includes the EC class name, the finite-field context hash, and all curve constants

This captures the semantic identity of the context.

### 2. Kernel hash

For a concrete compiled kernel, the code then combines:

- `context_hash()`
- runtime shape parameters passed to `serialize()` or `compile()`

using:

`hash_args(self.context_hash(), parameters)`

That lets one context cache multiple shape-specialized executables.

## What `serialize()` Actually Does

The name can be slightly misleading.

In the current codebase, `serialize(parameters)` does not just save an abstract kernel definition. It:

1. builds shape descriptors from the provided parameters
2. JIT-compiles the target function
3. serializes the compiled executable with `jax.experimental.serialize_executable.serialize()`
4. writes the executable bytes and its input/output pytrees to disk

So `serialize()` is best understood as:

- compile now, then persist the compiled executable

### Stored file layout

For each kernel name, the project creates a directory:

`<serialized_jax_kernel_dir>/<kernel_name>/`

Inside that directory it stores:

- `executable.jax`
- `in_out_trees.pkl`

### Store naming rule

The stored directory name is built as:

`<class_name>_<operation_name>_<kernel_hash>`

where:

- `class_name` is the concrete context class, such as `DRNSlazyContext` or `ExtendedTwistedEdwardsContext`
- `operation_name` is the private kernel being cached, such as `modular_multiply`, `modular_add`, `point_add`, or `point_double`
- `kernel_hash` is `hash_args(self.context_hash(), parameters)`

So the full on-disk path is:

`<serialized_jax_kernel_dir>/<class_name>_<operation_name>_<kernel_hash>/`

Examples from the current code:

- `deployments/DRNSlazyContext_modular_multiply_ab12/`
- `deployments/DRNSlazyContext_modular_reduce_ab12/`
- `deployments/ExtendedTwistedEdwardsContext_point_add_z9K3/`

The same naming rule is used by both `serialize()` and `compile()`, which is why `compile()` can deterministically reload a previously stored executable.

## What `compile()` Does

`compile(parameters)` is the runtime loader and fallback compiler.

For each expected kernel, it:

1. rebuilds the same hashed name from the context and shape parameters
2. tries `load_jax_executable(...)`
3. if present, stores the loaded callable in `self.compiled_kernels`
4. if absent, falls back to fresh in-memory compilation with `jax_jit_lower_compile(...)`
5. sets `self.use_compiled_kernels = True`

This makes `compile()` the normal entrypoint for using cached executables in practice.

## Finite-Field Integration

`DRNSlazyContext` supports serialization/compilation for:

- `modular_multiply`
- `modular_add`
- `modular_subtract`
- `modular_reduce`
- `modular_negate`

Its shape contract is based on:

- `parameters["batch_shape"]`

The operand shape becomes:

- `batch_shape + (num_moduli,)`

Example:

```python
ctx.serialize({"batch_shape": (4096,)})
ctx.compile({"batch_shape": (4096,)})

a = ctx.to_computational_format([1, 2, 3, 4])
b = ctx.to_computational_format([5, 6, 7, 8])
ctx.set_use_compiled_kernels(True)
c = ctx.modular_multiply(a, b)
```

## Elliptic-Curve Integration

`ExtendedTwistedEdwardsContext` supports serialization/compilation for:

- `point_add`
- `point_double`

Its shape contract is based on:

- `parameters["batch_size"]`

The point shape becomes:

- `(4, batch_size, num_moduli)`

Example:

```python
ec_ctx.serialize({"batch_size": 1024})
ec_ctx.compile({"batch_size": 1024})
ec_ctx.set_use_compiled_kernels(True)
```

Important caveat:

- the public `point_double()` API currently raises an error because the implementation is marked as buggy, even though the serialization path exists

## In-Memory Compiled Kernel Table

Both contexts store loaded or newly compiled callables in:

`self.compiled_kernels[shape_dtype_struct.__hash__()]`

At runtime, the public arithmetic methods look up the compiled callable by the input tensor's `jax.ShapeDtypeStruct`.

That means:

- compiled kernels are shape-specific
- switching batch shape requires a matching compiled entry
- if no compiled entry is registered for that shape, the public method will only work when `use_compiled_kernels` is `False`

## Relationship to `scratch_compiling.py`

`scratch_compiling.py` is a useful reference script for the lower-level JAX serialization APIs.
It demonstrates four separate execution modes:

- normal JIT-compiled execution
- execution from a deserialized compiled executable
- JIT compilation of a deserialized non-compiled kernel
- direct execution from a deserialized exported kernel

That file is effectively an experiment showing the primitives that the project later wraps in `utils.py`.

## Relationship to `profiler.py`

`profiler.py` uses a different path.

`KernelWrapper`:

- constructs `jax.ShapeDtypeStruct` values
- lowers and compiles immediately in memory
- optionally adds sharding metadata

This is for benchmarking and trace collection, not for persistent executable reuse.

So the repo currently has:

- context-level persistent caching for production-style repeated use
- profiler-level ephemeral compilation for benchmarking

## Typical Usage Pattern

For a context-based workflow, the intended sequence is:

1. construct the context with fixed mathematical parameters
2. call `serialize(shape_parameters)` once to populate the disk cache
3. later call `compile(shape_parameters)` to load cached executables
4. enable compiled execution with `set_use_compiled_kernels(True)`
5. call the public arithmetic methods

Example:

```python
ctx = DRNSlazyContext(ff_parameters)

shape_parameters = {"batch_shape": (1024,)}

ctx.serialize(shape_parameters)   # one-time cache population
ctx.compile(shape_parameters)     # load from cache or compile if missing
ctx.set_use_compiled_kernels(True)
```

## Practical Caveats

- Cache names depend on `hash_length`; the current value `4` is compact but increases collision risk compared with a longer hash.
- Cached executables are specific to both context parameters and input shape.
- `serialize()` performs compilation work, so it is not a cheap metadata-only step.
- The codebase still contains both exported-kernel and executable-serialization helpers; only the executable path is used by the current contexts.
- Any bug in the underlying private kernel implementation is preserved by serialization, so caching does not validate correctness on its own.
