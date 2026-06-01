import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from finite_field_context import FiniteFieldContextBase
import utils
from utils import JaxKernelContextBase, JaxParameters, hash_args, jax_jit_lower_compile, store_jax_executable, load_jax_executable


class FieldExtensionContextBase(ABC):
  def __init__(self, parameters: dict):
    self.parameters = parameters
    self.prime = parameters.get("prime", None)
    assert self.prime is not None, "prime must be provided"
    self.finite_field_context_class = parameters.get("finite_field_context_class", None)
    assert self.finite_field_context_class is not None, "finite_field_context_class must be provided"
    self.finite_field_context: FiniteFieldContextBase = self.finite_field_context_class(parameters.get("finite_field_parameters", {}))

  @abstractmethod
  def to_computational_format(self, a) -> jnp.ndarray:
    pass

  @abstractmethod
  def to_original_format(self, a: jnp.ndarray):
    pass


class Fq2Context(FieldExtensionContextBase, JaxKernelContextBase):
  """Degree-2 extension field Fq2 = Fq[u] / (u² + alpha) for TPU.

  Elements are (c0, c1) pairs representing c0 + c1·u, where u² = -alpha.

  Computational format: jnp.ndarray of shape [*leading, 2, moduli_dim] (uint32).
    axis -2 index 0 → c0 component
    axis -2 index 1 → c1 component
  ``leading`` may be any number of batch dimensions (including zero).  A
  single element is stored with one leading axis as [1, 2, moduli_dim].
  An EC point batch ``(4, *batch, 2, moduli_dim)`` flows through unchanged.

  All internal _modular_* methods accept and return [*leading, 2, moduli_dim]
  arrays and call the underlying Fq context's _modular_* methods on
  [*leading, moduli_dim] slices, which is compatible with both
  DRNSlazyContext and CROSSLazyContext (DRNS/CROSS reduce on the last axis).

  Karatsuba multiplication (3 Fq multiplications):
    t0  = a0·b0
    t1  = a1·b1
    c0  = t0 − alpha·t1
    c1  = (a0+a1)·(b0+b1) − t0 − t1

  Optimised squaring (2 Fq multiplications):
    t0  = a0²
    t1  = a1²
    c0  = t0 − alpha·t1
    c1  = (a0+a1)² − t0 − t1   [equals 2·a0·a1]
  """

  def __init__(self, parameters: dict):
    FieldExtensionContextBase.__init__(self, parameters)
    JaxKernelContextBase.__init__(self)
    self.quadratic_non_residue = parameters.get("quadratic_non_residue", None)
    assert self.quadratic_non_residue is not None, "quadratic_non_residue must be provided"
    # Precompute alpha in Fq computational format, shape [1, moduli_dim],
    # so it broadcasts against [N, moduli_dim] operands in _modular_multiply.
    fq = self.finite_field_context
    self._alpha_cf: jnp.ndarray = fq.to_computational_format([self.quadratic_non_residue])
    # CROSSLazyContext._modular_multiply is lazy: output may be slightly > p.
    # Detect this by the presence of the lazy-matrix attribute so that
    # _modular_multiply and _modular_square can normalize intermediate results
    # before passing them to _modular_subtract (which uses _sub_raw and
    # requires inputs ≤ p).  DRNSlazyContext is immune because its negate
    # adds a large precomputed multiple of p before subtracting.
    self._fq_needs_mul_reduce: bool = hasattr(fq, "_lazy_mat_jnp")

  # ---------------------------------------------------------------------------
  # Format conversion
  # ---------------------------------------------------------------------------

  def to_computational_format(self, a) -> jnp.ndarray:
    """Convert Fq2 elements to [*leading, 2, moduli_dim] uint32 arrays.

    Accepts:
      * a single pair ``(c0, c1)``  → output ``[1, 2, moduli_dim]``,
      * a flat list of pairs        → output ``[N, 2, moduli_dim]``,
      * an arbitrarily nested list  → output ``[*batch, 2, moduli_dim]``.
    """
    fq = self.finite_field_context
    if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], int):
      c0_cf = fq.to_computational_format([a[0]])  # [1, moduli_dim]
      c1_cf = fq.to_computational_format([a[1]])  # [1, moduli_dim]
      return jnp.stack([c0_cf, c1_cf], axis=-2)  # [1, 2, moduli_dim]

    # General nested case: recurse on the structure of ``a`` to extract two
    # parallel int trees, one for c0 and one for c1, then push each through
    # the underlying Fq context.  This produces matched [*batch, moduli_dim]
    # tensors which we stack at axis=-2.
    def split_tree(node):
      if isinstance(node, tuple) and len(node) == 2 and isinstance(node[0], int):
        return node[0], node[1]
      assert isinstance(node, (list, tuple)), \
          f"Fq2 input must be (int, int) tuple or nested list of such; got {type(node)}"
      c0_children = []
      c1_children = []
      for child in node:
        c0_child, c1_child = split_tree(child)
        c0_children.append(c0_child)
        c1_children.append(c1_child)
      return c0_children, c1_children

    c0_tree, c1_tree = split_tree(a)
    c0_cf = fq.to_computational_format(c0_tree)  # [*batch, moduli_dim]
    c1_cf = fq.to_computational_format(c1_tree)
    return jnp.stack([c0_cf, c1_cf], axis=-2)  # [*batch, 2, moduli_dim]

  def to_original_format(self, a: jnp.ndarray):
    """Convert [*leading, 2, moduli_dim] array back to nested (c0, c1) tuples.

    For input of shape ``[N, 2, moduli_dim]`` returns a flat list of N pairs.
    For higher-rank inputs returns a nested list mirroring the leading
    batch shape (each leaf a ``(c0, c1)`` int tuple).
    """
    fq = self.finite_field_context
    c0_data = fq.to_original_format(a[..., 0, :])
    c1_data = fq.to_original_format(a[..., 1, :])

    def zip_tree(c0_node, c1_node):
      if isinstance(c0_node, int):
        return (c0_node, c1_node)
      return [zip_tree(c0_child, c1_child)
              for c0_child, c1_child in zip(c0_node, c1_node)]
    return zip_tree(c0_data, c1_data)

  # ---------------------------------------------------------------------------
  # Internal JAX kernels  (operate on [N, 2, moduli_dim] arrays)
  # ---------------------------------------------------------------------------

  def _nr(self, x: jnp.ndarray) -> jnp.ndarray:
    """Normalize a [N, moduli_dim] Fq value after multiplication.

    For CROSSLazyContext the lazy multiply may output values slightly > p,
    which causes _modular_negate / _sub_raw to underflow.  Apply an explicit
    modular reduce to bring the value back into [0, p) before any subtraction.
    For DRNSlazyContext the output is already bounded and reduce would corrupt
    the Montgomery representation, so this is a no-op.
    """
    if self._fq_needs_mul_reduce:
      return self.finite_field_context._modular_reduce(x)
    return x

  def _canonicalize(self, x: jnp.ndarray) -> jnp.ndarray:
    """Normalize an Fq slot (shape ``[*leading, moduli_dim]``) so each slot
    is < m_i (DRNS) or < p (CROSS).

    Required at the *output* of Fq2 multiplication / squaring: the Karatsuba
    formula computes ``c1 = (t01 − t0) − t1`` with two chained Fq subtracts,
    leaving slot values up to ``~3·m + 2·sub``.  A subsequent
    ``fq._modular_negate`` would compute ``m − slot`` which underflows in
    ``uint32`` once ``slot > m``, silently corrupting the residue.  Forcing
    the output back into canonical form keeps all downstream lazy operations
    safe regardless of the host kernel's call pattern.
    """
    return self.finite_field_context._modular_reduce(x)

  def _modular_add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    fq = self.finite_field_context
    c0 = fq._modular_add(a[..., 0, :], b[..., 0, :])
    c1 = fq._modular_add(a[..., 1, :], b[..., 1, :])
    return jnp.stack([c0, c1], axis=-2)

  def _modular_subtract(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    fq = self.finite_field_context
    c0 = fq._modular_subtract(a[..., 0, :], b[..., 0, :])
    c1 = fq._modular_subtract(a[..., 1, :], b[..., 1, :])
    return jnp.stack([c0, c1], axis=-2)

  def _modular_negate(self, a: jnp.ndarray) -> jnp.ndarray:
    fq = self.finite_field_context
    c0 = fq._modular_negate(a[..., 0, :])
    c1 = fq._modular_negate(a[..., 1, :])
    return jnp.stack([c0, c1], axis=-2)

  def _modular_multiply(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Karatsuba Fq2 multiplication — 3 Fq multiplications."""
    fq = self.finite_field_context
    a0, a1 = a[..., 0, :], a[..., 1, :]
    b0, b1 = b[..., 0, :], b[..., 1, :]

    t0 = self._nr(fq._modular_multiply(a0, b0))
    t1 = self._nr(fq._modular_multiply(a1, b1))

    # c0 = t0 − alpha·t1
    alpha = jnp.broadcast_to(self._alpha_cf, t1.shape)
    alpha_t1 = self._nr(fq._modular_multiply(alpha, t1))
    c0 = fq._modular_subtract(t0, alpha_t1)

    # c1 = (a0+a1)·(b0+b1) − t0 − t1
    t01 = self._nr(fq._modular_multiply(fq._modular_add(a0, a1), fq._modular_add(b0, b1)))
    c1 = fq._modular_subtract(fq._modular_subtract(t01, t0), t1)

    # Bring both components back to canonical form.  c1 in particular went
    # through a depth-2 subtract chain and exceeds the lazy-negate input
    # bound; without this reduction a downstream Fq2 subtract corrupts c1.
    return jnp.stack([self._canonicalize(c0), self._canonicalize(c1)], axis=-2)

  def _modular_square(self, a: jnp.ndarray) -> jnp.ndarray:
    """Optimised Fq2 squaring — 2 Fq multiplications."""
    fq = self.finite_field_context
    a0, a1 = a[..., 0, :], a[..., 1, :]

    t0 = self._nr(fq._modular_multiply(a0, a0))
    t1 = self._nr(fq._modular_multiply(a1, a1))

    # c0 = t0 − alpha·t1
    alpha = jnp.broadcast_to(self._alpha_cf, t1.shape)
    alpha_t1 = self._nr(fq._modular_multiply(alpha, t1))
    c0 = fq._modular_subtract(t0, alpha_t1)

    # c1 = (a0+a1)² − t0 − t1  [equals 2·a0·a1]
    sum_a = fq._modular_add(a0, a1)
    t01 = self._nr(fq._modular_multiply(sum_a, sum_a))
    c1 = fq._modular_subtract(fq._modular_subtract(t01, t0), t1)

    return jnp.stack([self._canonicalize(c0), self._canonicalize(c1)], axis=-2)

  def _modular_conjugate(self, a: jnp.ndarray) -> jnp.ndarray:
    """Frobenius: (c0 + c1·u) → (c0 − c1·u) = (c0, −c1)."""
    fq = self.finite_field_context
    c0 = a[..., 0, :]
    c1 = fq._modular_negate(a[..., 1, :])
    return jnp.stack([c0, c1], axis=-2)

  def _modular_scale_fq(self, a: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    """Scale Fq2 element by an Fq scalar s.

    ``a`` has shape ``[*leading, 2, moduli_dim]``; ``s`` has shape
    ``[*leading, moduli_dim]`` (broadcastable against each c0/c1 slice).
    """
    fq = self.finite_field_context
    c0 = fq._modular_multiply(a[..., 0, :], s)
    c1 = fq._modular_multiply(a[..., 1, :], s)
    return jnp.stack([c0, c1], axis=-2)

  def _modular_reduce(self, a: jnp.ndarray) -> jnp.ndarray:
    fq = self.finite_field_context
    c0 = fq._modular_reduce(a[..., 0, :])
    c1 = fq._modular_reduce(a[..., 1, :])
    return jnp.stack([c0, c1], axis=-2)

  # ---------------------------------------------------------------------------
  # Public interface (routes through compiled kernels when available)
  # ---------------------------------------------------------------------------

  def modular_add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels and kernel_hash in self.compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_add"](a, b)
    return self._modular_add(a, b)

  def modular_subtract(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels and kernel_hash in self.compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_subtract"](a, b)
    return self._modular_subtract(a, b)

  def modular_negate(self, a: jnp.ndarray) -> jnp.ndarray:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels and kernel_hash in self.compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_negate"](a)
    return self._modular_negate(a)

  def modular_multiply(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels and kernel_hash in self.compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_multiply"](a, b)
    return self._modular_multiply(a, b)

  def modular_square(self, a: jnp.ndarray) -> jnp.ndarray:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels and kernel_hash in self.compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_square"](a)
    return self._modular_square(a)

  def modular_conjugate(self, a: jnp.ndarray) -> jnp.ndarray:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels and kernel_hash in self.compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_conjugate"](a)
    return self._modular_conjugate(a)

  def modular_scale_fq(self, a: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels and kernel_hash in self.compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_scale_fq"](a, s)
    return self._modular_scale_fq(a, s)

  def modular_reduce(self, a: jnp.ndarray) -> jnp.ndarray:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels and kernel_hash in self.compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_reduce"](a)
    return self._modular_reduce(a)

  # ---------------------------------------------------------------------------
  # Compilation and serialisation
  # ---------------------------------------------------------------------------

  def context_hash(self) -> str:
    return hash_args(
      self.__class__.__name__,
      self.prime,
      self.quadratic_non_residue,
      self.finite_field_context.__class__.__name__,
    )

  def _get_shape_dtype_structs(self, parameters: dict) -> list:
    """Return a ShapeDtypeStruct for a single [N, 2, moduli_dim] array.

    Required keys in parameters:
        "batch_shape"      — tuple, e.g. (N,)
        "fq_element_shape" — tuple, e.g. (num_moduli,) or (chunk_num_u32,)
        "dtype"            — optional jnp dtype, defaults to jnp.uint32
    """
    batch_shape = parameters["batch_shape"]
    fq_shape    = parameters["fq_element_shape"]
    dtype       = parameters.get("dtype", jnp.uint32)
    element_shape = batch_shape + (2,) + fq_shape
    return [jax.ShapeDtypeStruct(element_shape, dtype)]

  def compile(self, parameters: dict):
    """JIT-compile all Fq2 kernels for the given element shape.

    parameters must contain "batch_shape" and "fq_element_shape".
    After calling compile(), public modular_* methods use the compiled
    kernels for that shape.
    """
    sds = self._get_shape_dtype_structs(parameters)[0]
    kernel_hash = hash_args(sds.shape, sds.dtype.__str__())

    fq_sds = jax.ShapeDtypeStruct(
      sds.shape[:1] + sds.shape[2:],  # [N, moduli_dim]
      sds.dtype,
    )

    self.compiled_kernels[kernel_hash] = {
      "modular_add":       jax_jit_lower_compile(self._modular_add,       sds, sds),
      "modular_subtract":  jax_jit_lower_compile(self._modular_subtract,  sds, sds),
      "modular_negate":    jax_jit_lower_compile(self._modular_negate,    sds),
      "modular_multiply":  jax_jit_lower_compile(self._modular_multiply,  sds, sds),
      "modular_square":    jax_jit_lower_compile(self._modular_square,    sds),
      "modular_conjugate": jax_jit_lower_compile(self._modular_conjugate, sds),
      "modular_scale_fq":  jax_jit_lower_compile(self._modular_scale_fq,  sds, fq_sds),
      "modular_reduce":    jax_jit_lower_compile(self._modular_reduce,    sds),
    }
    self.use_compiled_kernels = True

  def serialize(self, parameters: dict):
    """Serialize all Fq2 compiled kernels to disk."""
    sds = self._get_shape_dtype_structs(parameters)[0]
    fq_sds = jax.ShapeDtypeStruct(sds.shape[:1] + sds.shape[2:], sds.dtype)
    kh = hash_args(self.context_hash(), parameters)
    cn = self.__class__.__name__

    store_jax_executable(self._modular_add,       sds, sds,    name=f"{cn}_modular_add_{kh}")
    store_jax_executable(self._modular_subtract,  sds, sds,    name=f"{cn}_modular_subtract_{kh}")
    store_jax_executable(self._modular_negate,    sds,         name=f"{cn}_modular_negate_{kh}")
    store_jax_executable(self._modular_multiply,  sds, sds,    name=f"{cn}_modular_multiply_{kh}")
    store_jax_executable(self._modular_square,    sds,         name=f"{cn}_modular_square_{kh}")
    store_jax_executable(self._modular_conjugate, sds,         name=f"{cn}_modular_conjugate_{kh}")
    store_jax_executable(self._modular_scale_fq,  sds, fq_sds, name=f"{cn}_modular_scale_fq_{kh}")
    store_jax_executable(self._modular_reduce,    sds,         name=f"{cn}_modular_reduce_{kh}")


# Alias kept for backward compatibility with existing test stubs.
TestFieldExtension2Context = Fq2Context