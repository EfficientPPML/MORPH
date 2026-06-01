from abc import ABC, abstractmethod
import math
import logging
import multiprocessing
from typing import Tuple, Union
import copy
import numpy as np
import warnings
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import jax
import jax.numpy as jnp

# Use 'forkserver' to avoid JAX multithreading + fork deadlock
_MP_CONTEXT = multiprocessing.get_context("forkserver")

import utils
from utils import JaxParameters, JaxKernelContextBase, hash_args, pad_jax_array, store_jax_executable, load_jax_executable, jax_jit_lower_compile
from finite_field_context import FiniteFieldContextBase
from field_extension_context import Fq2Context

jax.config.update("jax_enable_x64", True)


class EllipticCurveContextBase(ABC):
  """Abstract base class defining the interface for finite field operations.

  Subclasses must implement all abstract methods to provide concrete
  finite field arithmetic operations.
  """

  @abstractmethod
  def __init__(self, parameters: dict):
    """Initialize the finite field context.

    Args:
        parameters: Configuration dictionary containing field parameters.
    """
    self.parameters = parameters
    self.prime = parameters.get("prime", None)
    assert self.prime is not None, "prime must be provided"
    ff_ctx_class = parameters.get("finite_field_context_class", None)
    assert ff_ctx_class is not None, "finite_field_context_class must be provided"
    self.ff_ctx: FiniteFieldContextBase = ff_ctx_class(parameters.get("finite_field_parameters", {}))

  @abstractmethod
  def to_computational_format(self, a) -> jnp.ndarray:
    """Convert input to the internal computational representation.

    Args:
        a: Input value in standard format.

    Returns:
        Value converted to computational format (e.g., Montgomery form).
    """
    pass

  @abstractmethod
  def to_original_format(self, a) -> jnp.ndarray:
    """Convert from computational format back to standard representation.

    Args:
        a: Value in computational format.

    Returns:
        Value in standard integer representation.
    """
    pass

  @abstractmethod
  def point_add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Perform point addition: (a + b)

    Args:
        a: First operand in computational format.
        b: Second operand in computational format.

    Returns:
        Product in computational format.
    """
    pass

  @abstractmethod
  def point_double(self, a: jnp.ndarray) -> jnp.ndarray:
    """Perform point doubling: (2 * a)

    Args:
        a: First operand in computational format.

    Returns:
        Product in computational format.
    """
    pass

  def get_finite_field_context(self) -> FiniteFieldContextBase:
    return self.ff_ctx

  def _modular_multiply(self, a: int, b: int) -> int:
    return (a * b) % self.prime

  def _modular_reduce(self, a: int) -> int:
    return a % self.prime

  def _modular_divide(self, a: int, b: int) -> int:
    assert b != 0, "ec divide: b is zero"
    b_inv = pow(b, self.prime - 2, self.prime)
    return (a * b_inv) % self.prime


class CPUWeierstrassAffineContext(EllipticCurveContextBase):
  """CPU implementation of Weierstrass affine curve operations.

  This class provides CPU-based implementations for point addition and doubling
  on a Weierstrass curve in affine coordinates.
  This class is only for private functional testing, not for production use.
  """

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    # warnings.warn("CPUWeierstrassAffineContext is only for private functional testing, not for production use",UserWarning, stacklevel=2)

    # Curve configuration
    self.a = parameters["a"]
    self.b = parameters["b"]

  def point_add(self, point_a: jnp.ndarray, point_b: jnp.ndarray) -> jnp.ndarray:
    raise NotImplementedError("CPUWeierstrassAffineContext: point_add is not implemented")

  def point_double(self, point: jnp.ndarray) -> jnp.ndarray:
    raise NotImplementedError("CPUWeierstrassAffineContext: point_double is not implemented")

  def to_computational_format(self, a: list) -> jnp.ndarray:
    raise NotImplementedError("CPUWeierstrassAffineContext: to_computational_format is not implemented")

  def to_original_format(self, a: jnp.ndarray) -> list:
    raise NotImplementedError("CPUWeierstrassAffineContext: to_original_format is not implemented")

  def _point_add(self, point_a: list, point_b: list) -> list[int]:
    def single_point_add(point_a: list, point_b: list) -> list[int]:
      x1, y1 = point_a
      x2, y2 = point_b
      slope = self._modular_divide(self._modular_reduce(y2 - y1), self._modular_reduce(x2 - x1))
      x3 = self._modular_reduce(self._modular_multiply(slope, slope) - x1 - x2)
      y3 = self._modular_reduce(self._modular_multiply(slope, self._modular_reduce(x1 - x3)) - y1)
      return [x3, y3]

    list_depth = utils.nested_list_depth(point_a)
    if list_depth == 1:
      return single_point_add(point_a, point_b)
    elif list_depth == 2:
      return [single_point_add(point_a_i, point_b_i) for point_a_i, point_b_i in zip(point_a, point_b)]
    else:
      raise ValueError(f"Invalid list depth {list_depth} of input for point addition")

  def _point_double(self, point: list[int]) -> list[int]:
    def single_point_double(point: list) -> list[int]:
      x, y = point
      slope = self._modular_divide(self._modular_reduce(3 * x * x + self.a), self._modular_reduce(2 * y))
      x3 = self._modular_reduce(self._modular_multiply(slope, slope) - 2 * x)
      y3 = self._modular_reduce(self._modular_multiply(slope, x - x3) - y)
      return [x3, y3]

    list_depth = utils.nested_list_depth(point)
    if list_depth == 1:
      return single_point_double(point)
    elif list_depth == 2:
      return [single_point_double(point_i) for point_i in point]
    else:
      raise ValueError("Invalid list depth of input for point doubling")


class ExtendedTwistedEdwardsContextBase(EllipticCurveContextBase):

  def __init__(self, parameters: dict):
    super().__init__(parameters)

    # Curve configuration
    self.a = parameters["a"]
    self.twist_d = parameters["twist_d"]
    self.alpha = parameters["alpha"]
    self.s = parameters["s"]
    self.A = parameters["MA"]
    self.B = parameters["MB"]
    self.t = parameters["t"]
    self.k = self.twist_d + self.twist_d
    self.zero_point = [0, 1, 1, 0]

  def _twist(self, coordinates: list[int]) -> list[int]:
    assert len(coordinates) == 2, "Twisted Edwards coordinates must be of length 2"
    x, y = coordinates
    # Convert to montgomery (Notel it is ec montgomery not field montgomery)
    xm = self._modular_reduce(self.s * (x - self.alpha))
    ym = self._modular_reduce(self.s * y)
    # Convert to edwards
    if ym == 0:
      raise ValueError("ec twist: ym is zero")
    xt = self._modular_divide(xm, ym)

    yt_denom = xm + 1
    if yt_denom == 0:
      raise ValueError("ec twist: yt_denom is zero")
    yt = self._modular_divide(xm - 1, yt_denom)

    xt = self._modular_multiply(xt, self.t)
    return [xt, yt]

  def _untwist(self, coordinates: list[int]) -> list[int]:
    assert len(coordinates) == 2, "Twisted Edwards coordinates must be of length 2"
    xt, yt = coordinates
    xt = self._modular_divide(xt, self.t)
    # Convert to montgomery
    xm = self._modular_divide((1 + yt), (1 - yt))
    ym = self._modular_divide((1 + yt), self._modular_multiply((1 - yt), xt))
    # Convert to weierstrass
    x = self._modular_reduce(
        self._modular_divide(xm, self.B) + self._modular_divide(self.A, self._modular_multiply(3, self.B))
    )
    y = self._modular_divide(ym, self.B)
    return [x, y]

  def _convert_to_edwards_affine(self, coordinates: list[int]) -> list[int]:
    assert len(coordinates) == 4, "Twisted Edwards coordinates must be of length 2"
    x, y, z, t = coordinates
    z_inv = self._modular_divide(1, z)
    x = self._modular_multiply(x, z_inv)
    y = self._modular_multiply(y, z_inv)
    return [x, y]

  def _convert_to_extended_twisted_edwards(self, coordinates: list[int]) -> list[int]:
    assert len(coordinates) == 2, "Twisted Edwards coordinates must be of length 2"
    xt, yt = self._twist(coordinates)
    return [xt, yt, 1, self._modular_multiply(xt, yt)]

  def _convert_to_weierstrass_affine(self, coordinates: list[int]) -> list[int]:
    assert len(coordinates) == 4, "Extended Twisted Edwards coordinates must be of length 4"
    affine_coords = self._convert_to_edwards_affine(coordinates)
    untwisted_coords = self._untwist(affine_coords)
    return untwisted_coords


class ExtendedTwistedEdwardsContext(ExtendedTwistedEdwardsContextBase, JaxKernelContextBase):

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    JaxKernelContextBase.__init__(self)
    self.jax_parameters = JaxParameters()
    self._init_jax_parameters()

  def to_computational_format(self, a: list) -> jnp.ndarray:
    list_depth = utils.nested_list_depth(a)
    # NOTE: the dimension is (batch, coordinates)
    if list_depth == 1:
      twisted_coords = self._convert_to_extended_twisted_edwards(a)
    elif list_depth == 2:
      twisted_coords = [self._convert_to_extended_twisted_edwards(a_i) for a_i in a]
    else:
      raise ValueError("Invalid list depth of input for converting to extended twisted edwards coordinates")
    result = self.ff_ctx.to_computational_format(twisted_coords)
    if list_depth == 1:
      result = jnp.broadcast_to(result, (result.shape[0], 1, result.shape[1]))
    elif list_depth == 2:
      result = result.transpose(1, 0, 2)
    # NOTE: the computational format dim is (coordinates, batch, precision)
    if self.use_sharding:
      named_sharding, padded_shape = self.create_named_sharding(shape=result.shape, axes=[1])
      result = pad_jax_array(result, padded_shape)
      return result.to_device(named_sharding)
    else:
      return result.to_device(jax.devices("tpu")[0])


  def to_original_format(self, a: jnp.ndarray) -> list:
    dim = a.ndim
    # NOTE: the computational format dim is (coordinates, batch, precision)
    if dim == 3:
      a = a.transpose(1, 0, 2)  # (coordinates, batch, precision) -> (batch, coordinates, precision)
    a = self.ff_ctx.to_original_format(a)
    if dim == 2:
      affine_coords = self._convert_to_weierstrass_affine(a)
    elif dim == 3:
      affine_coords = [self._convert_to_weierstrass_affine(a_i) for a_i in a]
    else:
      raise ValueError("Invalid dimension of input for converting to weierstrass affine coordinates")
    return affine_coords

  def _init_jax_parameters(self):
    self.jax_parameters.set_parameter(
        twist_d=self.ff_ctx.to_computational_format(self.twist_d),
    )

  def _point_add(self, point_a: jnp.ndarray, point_b: jnp.ndarray) -> jnp.ndarray:
    twist_d = self.jax_parameters.twist_d

    inputsl = point_a
    inputsr = point_b
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    a, b, d, c = jnp.vsplit(outputs, 4)
    # print(a.shape, b.shape, d.shape, c.shape)

    pax, pay, _, _ = jnp.vsplit(point_a, 4)
    pbx, pby, _, _ = jnp.vsplit(point_b, 4)

    e1 = self.ff_ctx._modular_add(pax, pay)
    e2 = self.ff_ctx._modular_add(pbx, pby)
    twist_d_here = jnp.broadcast_to(twist_d.reshape(-1, twist_d.shape[0]), c.shape)
    if self.use_sharding:
      twist_d_here = jax.sharding.reshard(twist_d_here, jax.typeof(e2).sharding)
    inputsl = jnp.concatenate((e1, c), axis=0)
    inputsr = jnp.concatenate((e2, twist_d_here), axis=0)
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    e3, c = jnp.vsplit(outputs, 2)

    e = self.ff_ctx._modular_subtract(self.ff_ctx._modular_subtract(e3, a), b)
    f = self.ff_ctx._modular_subtract(d, c)
    g = self.ff_ctx._modular_add(d, c)
    h = self.ff_ctx._modular_add(a, b)

    inputsl = jnp.concatenate((e, g, f, e), axis=0)
    inputsr = jnp.concatenate((f, h, g, h), axis=0)
    # print(inputsl.shape, inputsr.shape)
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    return outputs.reshape(4, -1, outputs.shape[-1])

  def _point_double(self, point: jnp.ndarray) -> jnp.ndarray:
    x, y, z, t = point

    # dbl-2008-hwcd with a=-1:  A=X², B=Y², E=2XY, G=B-A, F=G-2Z², H=-(A+B)
    # Negate only fresh multiply outputs to avoid uint32 wrap in lazy DRNS negate.
    et = self.ff_ctx._modular_multiply(x, y)
    inputsl = jnp.vstack((x, y, z))
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsl)
    a, b, ct = jnp.vsplit(outputs, 3)

    h = self.ff_ctx._modular_add(self.ff_ctx._modular_negate(a), self.ff_ctx._modular_negate(b))
    e = self.ff_ctx._modular_add(et, et)
    g = self.ff_ctx._modular_subtract(b, a)
    f = self.ff_ctx._modular_subtract(self.ff_ctx._modular_subtract(g, ct), ct)

    inputsl = jnp.vstack((e, g, f, e))
    inputsr = jnp.vstack((f, h, g, h))
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    return outputs.reshape(4, -1, outputs.shape[-1])

  def point_add(self, point_a: jnp.ndarray, point_b: jnp.ndarray) -> jnp.ndarray:

    if self.use_compiled_kernels:
      kernel_hash = hash_args(point_a.shape, point_a.dtype.__str__())
      return self.compiled_kernels[kernel_hash]["point_add"](point_a, point_b)
    else:
      return self._point_add(point_a, point_b)

  def point_double(self, point: jnp.ndarray) -> jnp.ndarray:
    shape_dtype_struct = jax.ShapeDtypeStruct(point.shape, point.dtype)
    if self.use_compiled_kernels:
      return self.compiled_kernels[shape_dtype_struct.__hash__()]["point_double"](point)
    else:
      return self._point_double(point)

  def _get_shape_dtype_structs(self, parameters: dict) -> list[jax.ShapeDtypeStruct]:
    batch_size = parameters["batch_size"]
    num_moduli = self.jax_parameters.twist_d.shape[0]
    point_shape = (4, batch_size, num_moduli)
    if self.use_sharding:
      named_sharding, padded_shape = self.create_named_sharding(shape=point_shape, axes=[1])
      return [jax.ShapeDtypeStruct(padded_shape, jnp.uint32, sharding=named_sharding)]
    return [jax.ShapeDtypeStruct(point_shape, jnp.uint32)]

  def context_hash(self) -> str:
    return hash_args(
        self.__class__.__name__,
        self.ff_ctx.context_hash(),
        self.a,
        self.twist_d,
        self.alpha,
        self.s,
        self.A,
        self.B,
        self.t,
        self.use_sharding,
    )

  def serialize(self, parameters: dict):
    shape_dtype_structs = self._get_shape_dtype_structs(parameters)
    kernel_hash = hash_args(self.context_hash(), parameters)
    class_name = self.__class__.__name__

    store_jax_executable(
        self._point_add, shape_dtype_structs[0], shape_dtype_structs[0], name=f"{class_name}_point_add_{kernel_hash}"
    )
    store_jax_executable(self._point_double, shape_dtype_structs[0], name=f"{class_name}_point_double_{kernel_hash}")

  def compile(self, parameters: dict):
    shape_dtype_structs = self._get_shape_dtype_structs(parameters)
    kernel_hash = hash_args(self.context_hash(), parameters)
    class_name = self.__class__.__name__

    point_add_kernel = load_jax_executable(f"{class_name}_point_add_{kernel_hash}")
    point_double_kernel = load_jax_executable(f"{class_name}_point_double_{kernel_hash}")

    if None in [point_add_kernel, point_double_kernel]:
      warnings.warn(f"Not found stored serialized compiled kernels, compiling...", UserWarning, stacklevel=2)
    
    kernel_hash = hash_args(shape_dtype_structs[0].shape, shape_dtype_structs[0].dtype.__str__())
    self.compiled_kernels[kernel_hash] = {
        "point_add": point_add_kernel
        if point_add_kernel is not None
        else jax_jit_lower_compile(self._point_add, shape_dtype_structs[0], shape_dtype_structs[0]),
        "point_double": point_double_kernel
        if point_double_kernel is not None
        else jax_jit_lower_compile(self._point_double, shape_dtype_structs[0]),
    }
    self.use_compiled_kernels = True


def _twist_extend_and_rns_worker(point, s, alpha, prime, prime_m2, t, rns_moduli, radix_bits):
  """Module-level worker: twist + extend + RNS conversion using gmpy2 (must be picklable)."""
  import gmpy2
  x, y = gmpy2.mpz(point[0]), gmpy2.mpz(point[1])
  xm = (s * (x - alpha)) % prime
  ym = (s * y) % prime
  if ym == 0:
    raise ValueError("ec twist: ym is zero")
  xt = (xm * gmpy2.powmod(ym, prime_m2, prime)) % prime
  yt_denom = xm + 1
  if yt_denom == 0:
    raise ValueError("ec twist: yt_denom is zero")
  yt = ((xm - 1) * gmpy2.powmod(yt_denom, prime_m2, prime)) % prime
  xt = (xt * t) % prime
  coords = [int(xt), int(yt), 1, int((xt * yt) % prime)]
  # RNS conversion inline: (a % m) << radix_bits) % m for each coordinate
  return [[(((c % m) << radix_bits) % m) for m in rns_moduli] for c in coords]


class ExtendedTwistedEdwardsNDContext(ExtendedTwistedEdwardsContextBase, JaxKernelContextBase):
  """Extended Twisted Edwards context supporting arbitrary batch dimensions.

  Computational format layout: (coordinates=4, *batch_dims, precision)
  where batch_dims can be any number of dimensions, e.g.:
    - (4, batch, precision)                   -- 1D batch
    - (4, batch1, batch2, precision)          -- 2D batch
    - (4, batch1, batch2, batch3, precision)  -- 3D batch

  Input points are nested lists of [x, y] affine Weierstrass coordinates.
  Nesting depth determines batch dimensions:
    - [x, y]                       → (4, 1, precision)
    - [[x,y], ...]                 → (4, batch1, precision)
    - [[[x,y], ...], ...]          → (4, batch1, batch2, precision)
  """

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    JaxKernelContextBase.__init__(self)
    self.jax_parameters = JaxParameters()
    self._init_jax_parameters()

  def to_computational_format(self, a: list) -> jnp.ndarray:
    list_depth = utils.nested_list_depth(a)
    if list_depth < 1:
      raise ValueError(f"Invalid list depth {list_depth} for point conversion")

    # Use parallel processing with gmpy2 for large flat batches of points (depth==2)
    # Fuses twist + extend + RNS conversion into one parallel step
    _PARALLEL_THRESHOLD = 2048
    if list_depth == 2 and len(a) >= _PARALLEL_THRESHOLD:
      import gmpy2
      ff_ctx = self.ff_ctx
      worker = partial(
          _twist_extend_and_rns_worker,
          s=gmpy2.mpz(self.s), alpha=gmpy2.mpz(self.alpha),
          prime=gmpy2.mpz(self.prime), prime_m2=gmpy2.mpz(self.prime - 2),
          t=gmpy2.mpz(self.t),
          rns_moduli=ff_ctx.rns_moduli, radix_bits=ff_ctx.radix_bits,
      )
      num_workers = min(64, os.cpu_count() or 1, max(1, len(a) // 256))
      with ProcessPoolExecutor(max_workers=num_workers, mp_context=_MP_CONTEXT) as pool:
        rns_coords = list(pool.map(worker, a, chunksize=max(1, len(a) // num_workers)))
      # rns_coords: list of (4, moduli_num) per point → (N, 4, moduli_num) array
      result = jnp.array(np.array(rns_coords, dtype=np.uint32), dtype=jnp.uint32)
      # (N, 4, moduli_num) → (4, N, moduli_num)
      result = result.transpose(1, 0, 2)
    else:
      def recursive_twist(lst, depth):
        if depth == 1:
          return self._convert_to_extended_twisted_edwards(lst)
        return [recursive_twist(item, depth - 1) for item in lst]
      twisted_coords = recursive_twist(a, list_depth)

      result = self.ff_ctx.to_computational_format(twisted_coords)

      if list_depth == 1:
        # (4, precision) → (4, 1, precision)
        result = jnp.expand_dims(result, axis=1)
      else:
        # (*batch_dims, 4, precision) → (4, *batch_dims, precision)
        ndim = result.ndim
        perm = (ndim - 2,) + tuple(range(ndim - 2)) + (ndim - 1,)
        result = result.transpose(perm)

    if self.use_sharding:
      shard_axes = list(range(1, min(3, result.ndim - 1)))
      if not shard_axes:
        shard_axes = [1]
      named_sharding, padded_shape = self.create_named_sharding(shape=result.shape, axes=shard_axes)
      result = pad_jax_array(result, padded_shape)
      return result.to_device(named_sharding)
    else:
      return result.to_device(jax.devices("tpu")[0])

  def to_original_format(self, a: jnp.ndarray) -> list:
    ndim = a.ndim
    if ndim < 2:
      raise ValueError(f"Expected at least 2D array, got {ndim}D")

    if ndim == 2:
      a_orig = self.ff_ctx.to_original_format(a)
      return self._convert_to_weierstrass_affine(a_orig)

    # (4, *batch_dims, precision) → (*batch_dims, 4, precision)
    perm = tuple(range(1, ndim - 1)) + (0, ndim - 1)
    a = a.transpose(perm)
    a_orig = self.ff_ctx.to_original_format(a)

    batch_depth = ndim - 2
    def recursive_untwist(lst, depth):
      if depth == 0:
        return self._convert_to_weierstrass_affine(lst)
      return [recursive_untwist(item, depth - 1) for item in lst]

    return recursive_untwist(a_orig, batch_depth)

  def _init_jax_parameters(self):
    self.jax_parameters.set_parameter(
        twist_d=self.ff_ctx.to_computational_format(self.twist_d),
    )

  def _point_add(self, point_a: jnp.ndarray, point_b: jnp.ndarray) -> jnp.ndarray:
    twist_d = self.jax_parameters.twist_d

    inputsl = point_a
    inputsr = point_b
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    a, b, d, c = jnp.vsplit(outputs, 4)

    pax, pay, _, _ = jnp.vsplit(point_a, 4)
    pbx, pby, _, _ = jnp.vsplit(point_b, 4)

    e1 = self.ff_ctx._modular_add(pax, pay)
    e2 = self.ff_ctx._modular_add(pbx, pby)
    twist_d_here = jnp.broadcast_to(twist_d, c.shape)
    try:
      _e2_sh = jax.typeof(e2).sharding
      if _e2_sh is not None:
        twist_d_here = jax.sharding.reshard(twist_d_here, _e2_sh)
    except Exception:
      pass
    inputsl = jnp.concatenate((e1, c), axis=0)
    inputsr = jnp.concatenate((e2, twist_d_here), axis=0)
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    e3, c = jnp.vsplit(outputs, 2)

    e = self.ff_ctx._modular_subtract(self.ff_ctx._modular_subtract(e3, a), b)
    f = self.ff_ctx._modular_subtract(d, c)
    g = self.ff_ctx._modular_add(d, c)
    h = self.ff_ctx._modular_add(a, b)

    inputsl = jnp.concatenate((e, g, f, e), axis=0)
    inputsr = jnp.concatenate((f, h, g, h), axis=0)
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    return outputs

  def _point_double(self, point: jnp.ndarray) -> jnp.ndarray:
    original_shape = point.shape
    x, y, z, t = point

    # dbl-2008-hwcd with a=-1:  A=X², B=Y², E=2XY, G=B-A, F=G-2Z², H=-(A+B)
    # Negate only fresh multiply outputs to avoid uint32 wrap in lazy DRNS negate.
    et = self.ff_ctx._modular_multiply(x, y)
    inputsl = jnp.vstack((x, y, z))
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsl)
    a, b, ct = jnp.vsplit(outputs, 3)

    h = self.ff_ctx._modular_add(self.ff_ctx._modular_negate(a), self.ff_ctx._modular_negate(b))
    e = self.ff_ctx._modular_add(et, et)
    g = self.ff_ctx._modular_subtract(b, a)
    f = self.ff_ctx._modular_subtract(self.ff_ctx._modular_subtract(g, ct), ct)

    inputsl = jnp.vstack((e, g, f, e))
    inputsr = jnp.vstack((f, h, g, h))
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    return outputs.reshape(original_shape)

  def point_add(self, point_a: jnp.ndarray, point_b: jnp.ndarray) -> jnp.ndarray:
    if self.use_compiled_kernels:
      kernel_hash = hash_args(point_a.shape, point_a.dtype.__str__())
      return self.compiled_kernels[kernel_hash]["point_add"](point_a, point_b)
    else:
      return self._point_add(point_a, point_b)

  def point_double(self, point: jnp.ndarray) -> jnp.ndarray:
    shape_dtype_struct = jax.ShapeDtypeStruct(point.shape, point.dtype)
    if self.use_compiled_kernels:
      return self.compiled_kernels[shape_dtype_struct.__hash__()]["point_double"](point)
    else:
      return self._point_double(point)

  def _get_shape_dtype_structs(self, parameters: dict) -> list[jax.ShapeDtypeStruct]:
    batch_shape = parameters.get("batch_shape", None)
    if batch_shape is None:
      batch_shape = (parameters["batch_size"],)
    num_moduli = self.jax_parameters.twist_d.shape[0]
    point_shape = (4,) + tuple(batch_shape) + (num_moduli,)
    if self.use_sharding:
      shard_axes = list(range(1, min(3, len(point_shape) - 1)))
      if not shard_axes:
        shard_axes = [1]
      named_sharding, padded_shape = self.create_named_sharding(shape=point_shape, axes=shard_axes)
      return [jax.ShapeDtypeStruct(padded_shape, jnp.uint32, sharding=named_sharding)]
    return [jax.ShapeDtypeStruct(point_shape, jnp.uint32)]

  def context_hash(self) -> str:
    return hash_args(
        self.__class__.__name__,
        self.ff_ctx.context_hash(),
        self.a,
        self.twist_d,
        self.alpha,
        self.s,
        self.A,
        self.B,
        self.t,
        self.use_sharding,
    )

  def serialize(self, parameters: dict):
    shape_dtype_structs = self._get_shape_dtype_structs(parameters)
    kernel_hash = hash_args(self.context_hash(), parameters)
    class_name = self.__class__.__name__

    store_jax_executable(
        self._point_add, shape_dtype_structs[0], shape_dtype_structs[0], name=f"{class_name}_point_add_{kernel_hash}"
    )
    store_jax_executable(self._point_double, shape_dtype_structs[0], name=f"{class_name}_point_double_{kernel_hash}")

  def compile(self, parameters: dict):
    shape_dtype_structs = self._get_shape_dtype_structs(parameters)
    kernel_hash = hash_args(self.context_hash(), parameters)
    class_name = self.__class__.__name__

    point_add_kernel = load_jax_executable(f"{class_name}_point_add_{kernel_hash}")
    point_double_kernel = load_jax_executable(f"{class_name}_point_double_{kernel_hash}")

    if None in [point_add_kernel, point_double_kernel]:
      warnings.warn(f"Not found stored serialized compiled kernels, compiling...", UserWarning, stacklevel=2)

    kernel_hash = hash_args(shape_dtype_structs[0].shape, shape_dtype_structs[0].dtype.__str__())
    self.compiled_kernels[kernel_hash] = {
        "point_add": point_add_kernel
        if point_add_kernel is not None
        else jax_jit_lower_compile(self._point_add, shape_dtype_structs[0], shape_dtype_structs[0]),
        "point_double": point_double_kernel
        if point_double_kernel is not None
        else jax_jit_lower_compile(self._point_double, shape_dtype_structs[0]),
    }
    self.use_compiled_kernels = True


# =============================================================================
# Extended Twisted Edwards over Fq2  (G2-style curves over a quadratic extension)
# =============================================================================


def _as_fq2(value, prime: int) -> Tuple[int, int]:
  """Lift ``value`` to a canonical Fq2 element ``(c0, c1)``.

  Accepts either a raw int (interpreted as ``(int, 0)`` so plain-Fq curve
  parameters can be reused unchanged) or a ``(c0, c1)`` tuple/list.  All
  components are reduced modulo ``prime``.
  """
  if isinstance(value, int):
    return (value % prime, 0)
  if isinstance(value, (tuple, list)) and len(value) == 2 and \
      isinstance(value[0], int) and isinstance(value[1], int):
    return (value[0] % prime, value[1] % prime)
  raise TypeError(f"Cannot interpret {value!r} as an Fq2 element")


class ExtendedTwistedEdwardsFq2ContextBase(EllipticCurveContextBase):
  """Pure-Python Fp2 reference for the extended-twisted-Edwards model.

  Mirrors ``ExtendedTwistedEdwardsContextBase`` but with every curve
  constant and coordinate represented as an Fp2 element ``(c0, c1)``.
  Used both as a CPU oracle (``_point_add`` on lists) and as the host
  side of the JAX kernel context (``_twist`` / ``_untwist`` /
  ``_convert_*``).
  """

  def __init__(self, parameters: dict):
    self.parameters = parameters
    self.prime = parameters.get("prime", None)
    assert self.prime is not None, "prime must be provided"
    fe_ctx_class = parameters.get("field_extension_context_class", Fq2Context)
    fe_params = dict(parameters.get("field_extension_parameters", {}))
    fe_params.setdefault("prime", self.prime)
    fe_params.setdefault("finite_field_context_class",
                         parameters.get("finite_field_context_class"))
    fe_params.setdefault("finite_field_parameters",
                         parameters.get("finite_field_parameters", {}))
    fe_params.setdefault("quadratic_non_residue",
                         parameters.get("quadratic_non_residue"))
    self.fe_ctx: Fq2Context = fe_ctx_class(fe_params)
    self.ff_ctx = self.fe_ctx.finite_field_context
    self.qnr = parameters.get("quadratic_non_residue", fe_params.get("quadratic_non_residue"))
    assert self.qnr is not None, "quadratic_non_residue must be provided"

    # Curve constants.  Each is interpreted as an Fp2 element; plain ints
    # are auto-lifted to (int, 0) so the existing BLS12-377 G1 parameter
    # set can be reused for sanity tests.
    self.a        = _as_fq2(parameters["a"],        self.prime)
    self.twist_d  = _as_fq2(parameters["twist_d"],  self.prime)
    self.alpha    = _as_fq2(parameters["alpha"],    self.prime)
    self.s        = _as_fq2(parameters["s"],        self.prime)
    self.A        = _as_fq2(parameters["MA"],       self.prime)
    self.B        = _as_fq2(parameters["MB"],       self.prime)
    self.t        = _as_fq2(parameters["t"],        self.prime)
    self.k        = self._fp2_add(self.twist_d, self.twist_d)
    # Identity in extended twisted Edwards form, lifted to Fp2.
    self.zero_point = [(0, 0), (1, 0), (1, 0), (0, 0)]

  def get_finite_field_context(self) -> FiniteFieldContextBase:
    return self.ff_ctx

  def get_field_extension_context(self) -> Fq2Context:
    return self.fe_ctx

  # ------------------------------------------------------------------ #
  #  Pure-Python Fp2 arithmetic                                        #
  # ------------------------------------------------------------------ #

  def _fp2_add(self, a, b):
    return ((a[0] + b[0]) % self.prime, (a[1] + b[1]) % self.prime)

  def _fp2_sub(self, a, b):
    return ((a[0] - b[0]) % self.prime, (a[1] - b[1]) % self.prime)

  def _fp2_neg(self, a):
    return ((-a[0]) % self.prime, (-a[1]) % self.prime)

  def _fp2_mul(self, a, b):
    # (a0 + a1·u)·(b0 + b1·u) = (a0·b0 − qnr·a1·b1) + (a0·b1 + a1·b0)·u
    a0, a1 = a; b0, b1 = b
    p = self.prime
    c0 = (a0 * b0 - self.qnr * a1 * b1) % p
    c1 = (a0 * b1 + a1 * b0) % p
    return (c0, c1)

  def _fp2_sq(self, a):
    return self._fp2_mul(a, a)

  def _fp2_inv(self, a):
    a0, a1 = a
    p = self.prime
    denom = (a0 * a0 + self.qnr * a1 * a1) % p
    if denom == 0:
      raise ZeroDivisionError("fp2_inv on zero element")
    denom_inv = pow(denom, p - 2, p)
    return ((a0 * denom_inv) % p, ((-a1) * denom_inv) % p)

  def _fp2_div(self, a, b):
    return self._fp2_mul(a, self._fp2_inv(b))

  def _fp2_is_zero(self, a):
    return (a[0] % self.prime == 0) and (a[1] % self.prime == 0)

  def _fp2_eq(self, a, b):
    p = self.prime
    return (a[0] - b[0]) % p == 0 and (a[1] - b[1]) % p == 0

  def _fp2_scale_int(self, a, k: int):
    p = self.prime
    return ((a[0] * k) % p, (a[1] * k) % p)

  def _fp2_reduce(self, a):
    return (a[0] % self.prime, a[1] % self.prime)

  # ------------------------------------------------------------------ #
  #  Twist / untwist / coordinate conversions                          #
  # ------------------------------------------------------------------ #

  def _twist(self, coordinates):
    """Weierstrass-affine (over Fp2) → twisted-Edwards-affine (over Fp2)."""
    assert len(coordinates) == 2, "Weierstrass coordinates must be length 2"
    x = _as_fq2(coordinates[0], self.prime)
    y = _as_fq2(coordinates[1], self.prime)
    xm = self._fp2_mul(self.s, self._fp2_sub(x, self.alpha))
    ym = self._fp2_mul(self.s, y)
    if self._fp2_is_zero(ym):
      raise ValueError("ec twist (Fp2): ym is zero")
    xt = self._fp2_div(xm, ym)
    yt_denom = self._fp2_add(xm, (1, 0))
    if self._fp2_is_zero(yt_denom):
      raise ValueError("ec twist (Fp2): yt_denom is zero")
    yt = self._fp2_div(self._fp2_sub(xm, (1, 0)), yt_denom)
    xt = self._fp2_mul(xt, self.t)
    return [xt, yt]

  def _untwist(self, coordinates):
    """Twisted-Edwards-affine (over Fp2) → Weierstrass-affine (over Fp2)."""
    assert len(coordinates) == 2, "TE coordinates must be length 2"
    xt, yt = coordinates
    xt = self._fp2_div(xt, self.t)
    one = (1, 0)
    xm = self._fp2_div(self._fp2_add(one, yt), self._fp2_sub(one, yt))
    ym = self._fp2_div(
        self._fp2_add(one, yt),
        self._fp2_mul(self._fp2_sub(one, yt), xt),
    )
    three_B = self._fp2_scale_int(self.B, 3)
    x = self._fp2_add(
        self._fp2_div(xm, self.B),
        self._fp2_div(self.A, three_B),
    )
    y = self._fp2_div(ym, self.B)
    return [self._fp2_reduce(x), self._fp2_reduce(y)]

  def _convert_to_edwards_affine(self, coordinates):
    assert len(coordinates) == 4, "extended-TE coordinates must be length 4"
    x, y, z, _ = coordinates
    z_inv = self._fp2_inv(z)
    return [self._fp2_mul(x, z_inv), self._fp2_mul(y, z_inv)]

  def _convert_to_extended_twisted_edwards(self, coordinates):
    """[x_W, y_W] (Fp2 affine) → [xt, yt, 1, xt·yt] (Fp2 extended TE)."""
    xt, yt = self._twist(coordinates)
    return [xt, yt, (1, 0), self._fp2_mul(xt, yt)]

  def _convert_to_weierstrass_affine(self, coordinates):
    """[x, y, z, t] (Fp2 extended TE) → [x_W, y_W] (Fp2 affine)."""
    affine = self._convert_to_edwards_affine(coordinates)
    return self._untwist(affine)

  # ------------------------------------------------------------------ #
  #  CPU oracle (used by tests / Pippenger MSM helpers)                #
  # ------------------------------------------------------------------ #

  def _point_add(self, point_a, point_b):
    """Reference extended-twisted-Edwards add for points stored as
    length-4 lists of Fp2 tuples in [X, Y, Z, T] layout.

    Uses the ``add-2008-hwcd`` formula
    (https://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#addition-add-2008-hwcd)
    with the ``twist_d`` constant baked in.  For BLS12-377 the curve has
    ``a = -1``, which simplifies ``H = B − a·A`` to ``H = B + A`` —
    matching the JAX kernel's ``h = a + b`` exactly.

    Kept in pure Python so the JAX path can be cross-checked on random
    inputs without depending on a particular point-encoding pipeline.
    """
    def single(pa, pb):
      x1, y1, z1, t1 = pa
      x2, y2, z2, t2 = pb
      A = self._fp2_mul(x1, x2)
      B = self._fp2_mul(y1, y2)
      D = self._fp2_mul(z1, z2)
      C = self._fp2_mul(self._fp2_mul(t1, t2), self.twist_d)
      E = self._fp2_sub(
          self._fp2_sub(
              self._fp2_mul(self._fp2_add(x1, y1), self._fp2_add(x2, y2)),
              A,
          ),
          B,
      )
      F = self._fp2_sub(D, C)
      G = self._fp2_add(D, C)
      H = self._fp2_add(B, A)  # a = -1 → H = B − a·A = B + A
      X3 = self._fp2_mul(E, F)
      Y3 = self._fp2_mul(G, H)
      Z3 = self._fp2_mul(F, G)
      T3 = self._fp2_mul(E, H)
      return [X3, Y3, Z3, T3]

    depth = utils.nested_list_depth(point_a)
    # ``nested_list_depth`` doesn't recurse through tuples, so an Fp2
    # extended-TE point ``[(c0,c1), (c0,c1), (c0,c1), (c0,c1)]`` reports
    # depth==1 (single point) and a 1D batch of such points reports
    # depth==2.
    if depth == 1:
      return single(point_a, point_b)
    if depth == 2:
      return [single(pa, pb) for pa, pb in zip(point_a, point_b)]
    raise ValueError(f"Invalid list depth {depth} for Fp2 ETE point add")


class ExtendedTwistedEdwardsFq2NDContext(
    ExtendedTwistedEdwardsFq2ContextBase, JaxKernelContextBase):
  """Extended-Twisted-Edwards EC context whose base field is Fq2.

  Computational format layout: ``(coords=4, *batch_dims, 2, moduli_dim)``.

    * axis 0          — extended TE coordinate index ``[X, Y, Z, T]``
    * axis -2         — Fp2 component index ``[c0, c1]``
    * axis -1         — DRNS / lazy precision moduli
    * the middle axes carry an arbitrary nested batch shape

  Input points are nested lists of ``[x_W, y_W]`` Weierstrass-affine
  coordinates **over Fp2**, i.e. each coordinate is a ``(c0, c1)``
  tuple.  Nesting depth of the *list* part (excluding the inner Fp2
  tuples) determines the batch dimensionality:

    * ``[(x0,x1), (y0,y1)]``                                  → ``(4, 1, 2, M)``
    * ``[[(x0,x1),(y0,y1)], …]``                              → ``(4, N, 2, M)``
    * ``[[[(x,y), …], …], …]``                                → ``(4, N1, N2, 2, M)``
  """

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    JaxKernelContextBase.__init__(self)
    self.jax_parameters = JaxParameters()
    self._init_jax_parameters()

  # ------------------------------------------------------------------ #
  #  Format conversion                                                 #
  # ------------------------------------------------------------------ #

  @staticmethod
  def _affine_pair_depth(a):
    """Count nesting depth above the affine ``[x, y]`` Fp2-tuple leaf.

    ``[(c0,c1), (c0,c1)]``                       → 1   (single point)
    ``[[(c0,c1),(c0,c1)], …]``                   → 2   (1D batch)
    ``[[[(c0,c1),(c0,c1)], …], …]``              → 3   (2D batch)
    """
    def is_affine_pair(node):
      return (isinstance(node, list)
              and len(node) == 2
              and isinstance(node[0], tuple)
              and len(node[0]) == 2
              and isinstance(node[0][0], int))
    if is_affine_pair(a):
      return 1
    if not isinstance(a, list):
      raise ValueError(f"Bad input shape: {type(a)}")
    return 1 + max(ExtendedTwistedEdwardsFq2NDContext._affine_pair_depth(item)
                   for item in a)

  def to_computational_format(self, a) -> jnp.ndarray:
    depth = self._affine_pair_depth(a)
    if depth < 1:
      raise ValueError(f"Invalid list depth {depth} for Fp2 ETE point conversion")

    def recursive_twist(lst, d):
      if d == 1:
        return self._convert_to_extended_twisted_edwards(lst)
      return [recursive_twist(item, d - 1) for item in lst]

    twisted = recursive_twist(a, depth)

    # ``twisted`` is a nested list with the same batch structure as the
    # input but with 4 extended-TE coordinates at the leaf and each
    # coordinate an Fp2 ``(c0, c1)`` tuple.  fe_ctx.to_computational_format
    # turns the leaf-tuple level into the trailing ``(2, M)`` axes;
    # everything above it becomes batch dims.  The current axis order is
    # ``(*batch, 4, 2, M)`` (or ``(4, 2, M)`` when depth==1).  We move the
    # coord-axis to the front to land at ``(4, *batch, 2, M)``.
    result = self.fe_ctx.to_computational_format(twisted)

    if depth == 1:
      # (4, 2, M) → (4, 1, 2, M)
      result = jnp.expand_dims(result, axis=1)
    else:
      # (*batch, 4, 2, M) → (4, *batch, 2, M)
      ndim = result.ndim
      perm = (ndim - 3,) + tuple(range(ndim - 3)) + (ndim - 2, ndim - 1)
      result = result.transpose(perm)

    if self.use_sharding:
      shard_axes = list(range(1, min(3, result.ndim - 2)))
      if not shard_axes:
        shard_axes = [1]
      named_sharding, padded_shape = self.create_named_sharding(
          shape=result.shape, axes=shard_axes)
      result = pad_jax_array(result, padded_shape)
      return result.to_device(named_sharding)
    return result.to_device(jax.devices("tpu")[0])

  def to_original_format(self, a: jnp.ndarray):
    ndim = a.ndim
    if ndim < 3:
      raise ValueError(f"Expected at least 3D array (coord, 2, M), got {ndim}D")
    if ndim == 3:
      pts = self.fe_ctx.to_original_format(a)
      return self._convert_to_weierstrass_affine(pts)

    # (4, *batch, 2, M) → (*batch, 4, 2, M)
    perm = tuple(range(1, ndim - 2)) + (0, ndim - 2, ndim - 1)
    a = a.transpose(perm)
    pts = self.fe_ctx.to_original_format(a)

    batch_depth = ndim - 3

    def recursive_untwist(lst, d):
      if d == 0:
        return self._convert_to_weierstrass_affine(lst)
      return [recursive_untwist(item, d - 1) for item in lst]

    return recursive_untwist(pts, batch_depth)

  # ------------------------------------------------------------------ #
  #  JAX parameters                                                    #
  # ------------------------------------------------------------------ #

  def _init_jax_parameters(self):
    # ``twist_d`` is a single Fp2 element: shape ``[1, 2, M]``.
    self.jax_parameters.set_parameter(
        twist_d=self.fe_ctx.to_computational_format(self.twist_d),
    )

  # ------------------------------------------------------------------ #
  #  Point operations  (JAX kernels)                                   #
  # ------------------------------------------------------------------ #

  def _point_add_jax(self, point_a: jnp.ndarray, point_b: jnp.ndarray) -> jnp.ndarray:
    """Hisil-Wong-Carter-Dawson dedicated extended-TE addition over Fq2.

    All field arithmetic flows through ``self.fe_ctx`` (a ``Fq2Context``
    backed by a DRNS / lazy Fq context), so the kernel works for any
    leading batch shape ``(4, *batch, 2, M)``.
    """
    fe = self.fe_ctx
    twist_d = self.jax_parameters.twist_d  # shape (1, 2, M)

    inputsl = point_a
    inputsr = point_b
    outputs = fe._modular_multiply(inputsl, inputsr)
    a, b, d, c = jnp.vsplit(outputs, 4)

    pax, pay, _, _ = jnp.vsplit(point_a, 4)
    pbx, pby, _, _ = jnp.vsplit(point_b, 4)

    e1 = fe._modular_add(pax, pay)
    e2 = fe._modular_add(pbx, pby)
    twist_d_here = jnp.broadcast_to(twist_d, c.shape)
    try:
      _e2_sh = jax.typeof(e2).sharding
      if _e2_sh is not None:
        twist_d_here = jax.sharding.reshard(twist_d_here, _e2_sh)
    except Exception:
      pass
    inputsl = jnp.concatenate((e1, c), axis=0)
    inputsr = jnp.concatenate((e2, twist_d_here), axis=0)
    outputs = fe._modular_multiply(inputsl, inputsr)
    e3, c = jnp.vsplit(outputs, 2)

    e = fe._modular_subtract(fe._modular_subtract(e3, a), b)
    f = fe._modular_subtract(d, c)
    g = fe._modular_add(d, c)
    h = fe._modular_add(a, b)

    inputsl = jnp.concatenate((e, g, f, e), axis=0)
    inputsr = jnp.concatenate((f, h, g, h), axis=0)
    outputs = fe._modular_multiply(inputsl, inputsr)
    return outputs

  # Public API: keep the same name as the Fq context so call sites
  # interchangeably accept either backend.
  def point_add(self, point_a: jnp.ndarray, point_b: jnp.ndarray) -> jnp.ndarray:
    if self.use_compiled_kernels:
      kernel_hash = hash_args(point_a.shape, point_a.dtype.__str__())
      return self.compiled_kernels[kernel_hash]["point_add"](point_a, point_b)
    return self._point_add_jax(point_a, point_b)

  def point_double(self, point: jnp.ndarray) -> jnp.ndarray:
    raise NotImplementedError("Fp2 ExtendedTwistedEdwardsND: point_double not implemented")

  # ------------------------------------------------------------------ #
  #  Compile / serialise                                               #
  # ------------------------------------------------------------------ #

  def _get_shape_dtype_structs(self, parameters: dict) -> list[jax.ShapeDtypeStruct]:
    batch_shape = parameters.get("batch_shape", None)
    if batch_shape is None:
      batch_shape = (parameters["batch_size"],)
    moduli_dim = self.jax_parameters.twist_d.shape[-1]
    point_shape = (4,) + tuple(batch_shape) + (2, moduli_dim)
    if self.use_sharding:
      shard_axes = list(range(1, min(3, len(point_shape) - 2)))
      if not shard_axes:
        shard_axes = [1]
      named_sharding, padded_shape = self.create_named_sharding(
          shape=point_shape, axes=shard_axes)
      return [jax.ShapeDtypeStruct(padded_shape, jnp.uint32, sharding=named_sharding)]
    return [jax.ShapeDtypeStruct(point_shape, jnp.uint32)]

  def context_hash(self) -> str:
    return hash_args(
        self.__class__.__name__,
        self.fe_ctx.context_hash(),
        self.a, self.twist_d, self.alpha, self.s,
        self.A, self.B, self.t,
        self.use_sharding,
    )

  def serialize(self, parameters: dict):
    sds = self._get_shape_dtype_structs(parameters)[0]
    kernel_hash = hash_args(self.context_hash(), parameters)
    cn = self.__class__.__name__
    store_jax_executable(self._point_add_jax, sds, sds,
                         name=f"{cn}_point_add_{kernel_hash}")

  def compile(self, parameters: dict):
    sds = self._get_shape_dtype_structs(parameters)[0]
    kernel_hash = hash_args(self.context_hash(), parameters)
    cn = self.__class__.__name__
    point_add_kernel = load_jax_executable(f"{cn}_point_add_{kernel_hash}")
    if point_add_kernel is None:
      warnings.warn("Not found stored serialized compiled kernels, compiling...",
                    UserWarning, stacklevel=2)
    kernel_hash = hash_args(sds.shape, sds.dtype.__str__())
    self.compiled_kernels[kernel_hash] = {
        "point_add": point_add_kernel
        if point_add_kernel is not None
        else jax_jit_lower_compile(self._point_add_jax, sds, sds),
    }
    self.use_compiled_kernels = True


# ─────────────────────────────────────────────────────────────────────────────
# XYZZ-Weierstrass over Fq2  (G2-style curves where Edwards form doesn't exist)
# ─────────────────────────────────────────────────────────────────────────────


class XYZZWeierstrassFq2ContextBase(EllipticCurveContextBase):
  """Pure-Python Fp2 reference for the short-Weierstrass XYZZ model.

  Curve: y² = x³ + b over Fq2 (assumes a = 0, which matches all BLS-family
  G2 curves).  Coordinates: (X, Y, ZZ, ZZZ) with x = X/ZZ, y = Y/ZZZ and
  the on-curve invariant ZZZ² = ZZ³.  Affine encoding is (x, y, 1, 1).
  Identity is (1, 1, 0, 0); the JAX kernels do NOT special-case it — the
  MSM driver is responsible for not invoking ``point_add`` on it.

  Used both as a CPU oracle and as the host side of the JAX kernel
  context.  Reuses the ETE Fp2 base's pure-python field plumbing (so the
  oracle stays consistent with the existing Fq2 EC tests).
  """

  def __init__(self, parameters: dict):
    self.parameters = parameters
    self.prime = parameters.get("prime", None)
    assert self.prime is not None, "prime must be provided"
    fe_ctx_class = parameters.get("field_extension_context_class", Fq2Context)
    fe_params = dict(parameters.get("field_extension_parameters", {}))
    fe_params.setdefault("prime", self.prime)
    fe_params.setdefault("finite_field_context_class",
                         parameters.get("finite_field_context_class"))
    fe_params.setdefault("finite_field_parameters",
                         parameters.get("finite_field_parameters", {}))
    fe_params.setdefault("quadratic_non_residue",
                         parameters.get("quadratic_non_residue"))
    self.fe_ctx: Fq2Context = fe_ctx_class(fe_params)
    self.ff_ctx = self.fe_ctx.finite_field_context
    self.qnr = parameters.get("quadratic_non_residue", fe_params.get("quadratic_non_residue"))
    assert self.qnr is not None, "quadratic_non_residue must be provided"

    # Curve constants.  ``b`` is an Fp2 element; ``a`` is forced to zero
    # since the formulas in this context assume a = 0 (matches BLS G2).
    self.b = _as_fq2(parameters["b"], self.prime)
    a = parameters.get("a", 0)
    a_fp2 = _as_fq2(a, self.prime)
    if (a_fp2[0] % self.prime) != 0 or (a_fp2[1] % self.prime) != 0:
      raise ValueError("XYZZWeierstrassFq2Context: only a = 0 is supported")

    # Identity point (matches XYZZ convention).  X = Y = 1, ZZ = ZZZ = 0.
    self.zero_point = [(1, 0), (1, 0), (0, 0), (0, 0)]

  def get_finite_field_context(self) -> FiniteFieldContextBase:
    return self.ff_ctx

  def get_field_extension_context(self) -> Fq2Context:
    return self.fe_ctx

  # ------------------------------------------------------------------ #
  #  Pure-Python Fp2 arithmetic                                        #
  # ------------------------------------------------------------------ #

  def _fp2_add(self, a, b):
    return ((a[0] + b[0]) % self.prime, (a[1] + b[1]) % self.prime)

  def _fp2_sub(self, a, b):
    return ((a[0] - b[0]) % self.prime, (a[1] - b[1]) % self.prime)

  def _fp2_neg(self, a):
    return ((-a[0]) % self.prime, (-a[1]) % self.prime)

  def _fp2_mul(self, a, b):
    a0, a1 = a; b0, b1 = b
    p = self.prime
    c0 = (a0 * b0 - self.qnr * a1 * b1) % p
    c1 = (a0 * b1 + a1 * b0) % p
    return (c0, c1)

  def _fp2_sq(self, a):
    return self._fp2_mul(a, a)

  def _fp2_inv(self, a):
    a0, a1 = a
    p = self.prime
    denom = (a0 * a0 + self.qnr * a1 * a1) % p
    if denom == 0:
      raise ZeroDivisionError("fp2_inv on zero element")
    denom_inv = pow(denom, p - 2, p)
    return ((a0 * denom_inv) % p, ((-a1) * denom_inv) % p)

  def _fp2_div(self, a, b):
    return self._fp2_mul(a, self._fp2_inv(b))

  def _fp2_is_zero(self, a):
    return (a[0] % self.prime == 0) and (a[1] % self.prime == 0)

  def _fp2_eq(self, a, b):
    p = self.prime
    return (a[0] - b[0]) % p == 0 and (a[1] - b[1]) % p == 0

  def _fp2_scale_int(self, a, k: int):
    p = self.prime
    return ((a[0] * k) % p, (a[1] * k) % p)

  def _fp2_reduce(self, a):
    return (a[0] % self.prime, a[1] % self.prime)

  # ------------------------------------------------------------------ #
  #  Coordinate conversion (affine ↔ XYZZ)                             #
  # ------------------------------------------------------------------ #

  def _convert_to_xyzz(self, coordinates):
    """[x_W, y_W] (Fp2 affine) → [X, Y, ZZ, ZZZ] with ZZ = ZZZ = (1, 0)."""
    assert len(coordinates) == 2, "Weierstrass coordinates must be length 2"
    x = _as_fq2(coordinates[0], self.prime)
    y = _as_fq2(coordinates[1], self.prime)
    return [x, y, (1, 0), (1, 0)]

  def _convert_to_weierstrass_affine(self, coordinates):
    """[X, Y, ZZ, ZZZ] → [x_W, y_W].  Returns [(0,0),(0,0)] for identity.

    The identity (ZZ = 0) cannot be expressed in affine coordinates;
    callers should treat the all-zeros Fp2 output as the point at infinity.
    """
    assert len(coordinates) == 4, "XYZZ coordinates must be length 4"
    X, Y, ZZ, ZZZ = (_as_fq2(c, self.prime) for c in coordinates)
    if self._fp2_is_zero(ZZ) or self._fp2_is_zero(ZZZ):
      return [(0, 0), (0, 0)]
    x = self._fp2_div(X, ZZ)
    y = self._fp2_div(Y, ZZZ)
    return [self._fp2_reduce(x), self._fp2_reduce(y)]

  # ------------------------------------------------------------------ #
  #  CPU oracle (used by tests / Pippenger MSM helpers)                #
  # ------------------------------------------------------------------ #

  def _point_add(self, point_a, point_b):
    """Reference XYZZ add (Sutherland add-2008-s) over Fp2.

    Handles identity branches in Python so that the oracle matches the
    intended semantics of the JAX kernel + MSM-driver pristine tracking.
    """
    def single(pa, pb):
      X1, Y1, ZZ1, ZZZ1 = pa
      X2, Y2, ZZ2, ZZZ2 = pb
      # Identity branches.
      if self._fp2_is_zero(ZZ1):
        return [X2, Y2, ZZ2, ZZZ2]
      if self._fp2_is_zero(ZZ2):
        return [X1, Y1, ZZ1, ZZZ1]

      U1 = self._fp2_mul(X1, ZZ2)
      U2 = self._fp2_mul(X2, ZZ1)
      S1 = self._fp2_mul(Y1, ZZZ2)
      S2 = self._fp2_mul(Y2, ZZZ1)
      P  = self._fp2_sub(U2, U1)
      R  = self._fp2_sub(S2, S1)

      if self._fp2_is_zero(P):
        if self._fp2_is_zero(R):
          # P = Q: caller must use _point_double; fall through anyway.
          return self._point_double_single(pa)
        # P = -Q: result is identity.
        return [(1, 0), (1, 0), (0, 0), (0, 0)]

      PP  = self._fp2_sq(P)
      PPP = self._fp2_mul(P, PP)
      Q   = self._fp2_mul(U1, PP)
      twoQ = self._fp2_add(Q, Q)
      X3  = self._fp2_sub(self._fp2_sub(self._fp2_sq(R), PPP), twoQ)
      Y3  = self._fp2_sub(
          self._fp2_mul(R, self._fp2_sub(Q, X3)),
          self._fp2_mul(S1, PPP),
      )
      ZZ3  = self._fp2_mul(self._fp2_mul(ZZ1, ZZ2), PP)
      ZZZ3 = self._fp2_mul(self._fp2_mul(ZZZ1, ZZZ2), PPP)
      return [self._fp2_reduce(X3), self._fp2_reduce(Y3),
              self._fp2_reduce(ZZ3), self._fp2_reduce(ZZZ3)]

    depth = utils.nested_list_depth(point_a)
    if depth == 1:
      return single(point_a, point_b)
    if depth == 2:
      return [single(pa, pb) for pa, pb in zip(point_a, point_b)]
    raise ValueError(f"Invalid list depth {depth} for Fp2 XYZZ point add")

  def _point_double_single(self, pa):
    X1, Y1, ZZ1, ZZZ1 = pa
    if self._fp2_is_zero(ZZ1):
      return [(1, 0), (1, 0), (0, 0), (0, 0)]
    U  = self._fp2_add(Y1, Y1)              # 2·Y1
    V  = self._fp2_sq(U)                     # U²
    W  = self._fp2_mul(U, V)                 # U·V
    S  = self._fp2_mul(X1, V)                # X1·V
    X1_sq = self._fp2_sq(X1)
    M  = self._fp2_scale_int(X1_sq, 3)       # 3·X1²   (a = 0)
    X3 = self._fp2_sub(self._fp2_sq(M),
                       self._fp2_add(S, S))  # M² - 2S
    Y3 = self._fp2_sub(self._fp2_mul(M, self._fp2_sub(S, X3)),
                       self._fp2_mul(W, Y1))
    ZZ3  = self._fp2_mul(V, ZZ1)
    ZZZ3 = self._fp2_mul(W, ZZZ1)
    return [self._fp2_reduce(X3), self._fp2_reduce(Y3),
            self._fp2_reduce(ZZ3), self._fp2_reduce(ZZZ3)]

  def _point_double(self, point):
    """Reference XYZZ double (EFD dbl-2008-s-1, a = 0) over Fp2."""
    depth = utils.nested_list_depth(point)
    if depth == 1:
      return self._point_double_single(point)
    if depth == 2:
      return [self._point_double_single(p) for p in point]
    raise ValueError(f"Invalid list depth {depth} for Fp2 XYZZ point double")

  # ------------------------------------------------------------------ #
  #  Curve membership helper (for tests)                               #
  # ------------------------------------------------------------------ #

  def _is_on_curve_affine(self, affine):
    """y² == x³ + b over Fp2?  Identity (x=y=0) returns True by convention."""
    x = _as_fq2(affine[0], self.prime)
    y = _as_fq2(affine[1], self.prime)
    if self._fp2_is_zero(x) and self._fp2_is_zero(y):
      return True
    lhs = self._fp2_sq(y)
    rhs = self._fp2_add(self._fp2_mul(x, self._fp2_sq(x)), self.b)
    return self._fp2_eq(lhs, rhs)


class XYZZWeierstrassFq2NDContext(
    XYZZWeierstrassFq2ContextBase, JaxKernelContextBase):
  """JAX/TPU XYZZ-Weierstrass EC context over Fq2.

  Computational format layout: ``(coords=4, *batch_dims, 2, moduli_dim)``,
  same as ``ExtendedTwistedEdwardsFq2NDContext``:

    * axis 0          — XYZZ coordinate index ``[X, Y, ZZ, ZZZ]``
    * axis -2         — Fp2 component index ``[c0, c1]``
    * axis -1         — DRNS / lazy precision moduli
    * the middle axes carry an arbitrary nested batch shape

  Input points are nested lists of ``[x_W, y_W]`` Weierstrass-affine
  coordinates over Fp2; the format conversion injects ZZ = ZZZ = (1, 0).

  Identity handling is the *caller's* responsibility — `_point_add_jax`
  unconditionally evaluates the Sutherland add-2008-s formula, which
  degenerates when either operand is the identity (ZZ = 0) or when the
  two operands are equal.  The Pippenger driver tracks "pristine" buckets
  in Python and routes doublings to `point_double`.
  """

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    JaxKernelContextBase.__init__(self)
    self.jax_parameters = JaxParameters()
    self._init_jax_parameters()

  # ------------------------------------------------------------------ #
  #  Format conversion                                                 #
  # ------------------------------------------------------------------ #

  @staticmethod
  def _affine_pair_depth(a):
    """Count nesting depth above the affine ``[x, y]`` Fp2-tuple leaf."""
    def is_affine_pair(node):
      return (isinstance(node, list)
              and len(node) == 2
              and isinstance(node[0], tuple)
              and len(node[0]) == 2
              and isinstance(node[0][0], int))
    if is_affine_pair(a):
      return 1
    if not isinstance(a, list):
      raise ValueError(f"Bad input shape: {type(a)}")
    return 1 + max(XYZZWeierstrassFq2NDContext._affine_pair_depth(item)
                   for item in a)

  def to_computational_format(self, a) -> jnp.ndarray:
    depth = self._affine_pair_depth(a)
    if depth < 1:
      raise ValueError(f"Invalid list depth {depth} for Fp2 XYZZ point conversion")

    def recursive_extend(lst, d):
      if d == 1:
        return self._convert_to_xyzz(lst)
      return [recursive_extend(item, d - 1) for item in lst]

    extended = recursive_extend(a, depth)
    result = self.fe_ctx.to_computational_format(extended)

    if depth == 1:
      # (4, 2, M) → (4, 1, 2, M)
      result = jnp.expand_dims(result, axis=1)
    else:
      # (*batch, 4, 2, M) → (4, *batch, 2, M)
      ndim = result.ndim
      perm = (ndim - 3,) + tuple(range(ndim - 3)) + (ndim - 2, ndim - 1)
      result = result.transpose(perm)

    if self.use_sharding:
      shard_axes = list(range(1, min(3, result.ndim - 2)))
      if not shard_axes:
        shard_axes = [1]
      named_sharding, padded_shape = self.create_named_sharding(
          shape=result.shape, axes=shard_axes)
      result = pad_jax_array(result, padded_shape)
      return result.to_device(named_sharding)
    return result.to_device(jax.devices("tpu")[0])

  def to_original_format(self, a: jnp.ndarray):
    ndim = a.ndim
    if ndim < 3:
      raise ValueError(f"Expected at least 3D array (coord, 2, M), got {ndim}D")
    if ndim == 3:
      pts = self.fe_ctx.to_original_format(a)
      return self._convert_to_weierstrass_affine(pts)

    # (4, *batch, 2, M) → (*batch, 4, 2, M)
    perm = tuple(range(1, ndim - 2)) + (0, ndim - 2, ndim - 1)
    a = a.transpose(perm)
    pts = self.fe_ctx.to_original_format(a)

    batch_depth = ndim - 3

    def recursive_decode(lst, d):
      if d == 0:
        return self._convert_to_weierstrass_affine(lst)
      return [recursive_decode(item, d - 1) for item in lst]

    return recursive_decode(pts, batch_depth)

  # ------------------------------------------------------------------ #
  #  JAX parameters                                                    #
  # ------------------------------------------------------------------ #

  def _init_jax_parameters(self):
    # Encoded "1" in Fp2 form; useful for compile-stage shape derivation.
    self.jax_parameters.set_parameter(
        one=self.fe_ctx.to_computational_format((1, 0)),
    )

  # ------------------------------------------------------------------ #
  #  Point operations  (JAX kernels)                                   #
  # ------------------------------------------------------------------ #

  def _point_add_jax(self, point_a: jnp.ndarray, point_b: jnp.ndarray) -> jnp.ndarray:
    """Sutherland add-2008-s XYZZ addition (a = 0) over Fq2.

    Layout convention: ``point_*`` has shape ``(4, *batch, 2, M)`` with
    coord axis 0 = [X, Y, ZZ, ZZZ].  No identity / doubling handling —
    callers must avoid those degenerate cases.

    All field arithmetic flows through ``fe_ctx._modular_*``, so the
    formula works against any leading batch shape.
    """
    fe = self.fe_ctx
    X1, Y1, ZZ1, ZZZ1 = jnp.vsplit(point_a, 4)
    X2, Y2, ZZ2, ZZZ2 = jnp.vsplit(point_b, 4)

    # ── U1 = X1·ZZ2, U2 = X2·ZZ1, S1 = Y1·ZZZ2, S2 = Y2·ZZZ1 (4 muls)
    inputsl = jnp.concatenate((X1, X2, Y1, Y2), axis=0)
    inputsr = jnp.concatenate((ZZ2, ZZ1, ZZZ2, ZZZ1), axis=0)
    U1U2S1S2 = fe._modular_multiply(inputsl, inputsr)
    U1, U2, S1, S2 = jnp.vsplit(U1U2S1S2, 4)

    P = fe._modular_subtract(U2, U1)
    R = fe._modular_subtract(S2, S1)

    # ── PP = P², R² (2 squares)  +  ZZ1·ZZ2, ZZZ1·ZZZ2 (2 muls)
    sq_in = jnp.concatenate((P, R), axis=0)
    sq_out = fe._modular_square(sq_in)
    PP, RR = jnp.vsplit(sq_out, 2)

    z_in_l = jnp.concatenate((ZZ1, ZZZ1), axis=0)
    z_in_r = jnp.concatenate((ZZ2, ZZZ2), axis=0)
    z_prods = fe._modular_multiply(z_in_l, z_in_r)
    ZZ_prod, ZZZ_prod = jnp.vsplit(z_prods, 2)

    # ── PPP = P·PP, Q = U1·PP (2 muls)
    m1_l = jnp.concatenate((P, U1), axis=0)
    m1_r = jnp.concatenate((PP, PP), axis=0)
    PPP_Q = fe._modular_multiply(m1_l, m1_r)
    PPP, Q = jnp.vsplit(PPP_Q, 2)

    # ── X3 = R² − PPP − 2Q  (chained subtracts on canonical Q to avoid
    #    feeding a non-canonical `2Q` into _modular_subtract).
    X3 = fe._modular_subtract(
        fe._modular_subtract(fe._modular_subtract(RR, PPP), Q),
        Q,
    )
    # Canonicalize X3 — it's about to be the *second* operand of
    # `Q - X3`, which requires ≤ m on the subtrahend.
    X3 = fe._modular_reduce(X3)

    # ── ZZ3 = ZZ_prod·PP,  ZZZ3 = ZZZ_prod·PPP,
    #    Y3 part A = R·(Q − X3),  Y3 part B = S1·PPP   (4 muls)
    Q_minus_X3 = fe._modular_subtract(Q, X3)
    m2_l = jnp.concatenate((ZZ_prod, ZZZ_prod, R,         S1),  axis=0)
    m2_r = jnp.concatenate((PP,      PPP,      Q_minus_X3, PPP), axis=0)
    m2_out = fe._modular_multiply(m2_l, m2_r)
    ZZ3, ZZZ3, Y3A, Y3B = jnp.vsplit(m2_out, 4)
    Y3 = fe._modular_subtract(Y3A, Y3B)

    return jnp.concatenate((X3, Y3, ZZ3, ZZZ3), axis=0)

  def _point_double_jax(self, point: jnp.ndarray) -> jnp.ndarray:
    """EFD dbl-2008-s-1 (a = 0) XYZZ doubling over Fq2.

    No identity handling — caller must avoid doubling the identity.
    """
    fe = self.fe_ctx
    X1, Y1, ZZ1, ZZZ1 = jnp.vsplit(point, 4)

    # U = 2·Y1
    U = fe._modular_add(Y1, Y1)

    # V = U², X1² (2 squares)
    sq_in = jnp.concatenate((U, X1), axis=0)
    sq_out = fe._modular_square(sq_in)
    V, X1_sq = jnp.vsplit(sq_out, 2)

    # M = 3·X1²  (one add: 2·X1_sq, then +X1_sq)
    twoX1_sq = fe._modular_add(X1_sq, X1_sq)
    M = fe._modular_add(twoX1_sq, X1_sq)

    # W = U·V,  S = X1·V,  ZZ3 = V·ZZ1   (3 muls)
    m1_l = jnp.concatenate((U,  X1, V),   axis=0)
    m1_r = jnp.concatenate((V,  V,  ZZ1), axis=0)
    m1_out = fe._modular_multiply(m1_l, m1_r)
    W, S, ZZ3 = jnp.vsplit(m1_out, 3)

    # M² (1 square)
    M_sq = fe._modular_square(M)

    # X3 = M² − 2S  (chained subtract on canonical S; avoids feeding a
    # non-canonical `2S` as the subtrahend of _modular_subtract).
    X3 = fe._modular_subtract(fe._modular_subtract(M_sq, S), S)
    # Canonicalize X3 before using it as the second operand of S - X3.
    X3 = fe._modular_reduce(X3)

    # Y3 = M·(S − X3) − W·Y1,  ZZZ3 = W·ZZZ1   (3 muls)
    S_minus_X3 = fe._modular_subtract(S, X3)
    m2_l = jnp.concatenate((M,          W,  W),    axis=0)
    m2_r = jnp.concatenate((S_minus_X3, Y1, ZZZ1), axis=0)
    m2_out = fe._modular_multiply(m2_l, m2_r)
    Y3A, WY1, ZZZ3 = jnp.vsplit(m2_out, 3)
    Y3 = fe._modular_subtract(Y3A, WY1)

    return jnp.concatenate((X3, Y3, ZZ3, ZZZ3), axis=0)

  # Public API matching ExtendedTwistedEdwardsFq2NDContext.
  def point_add(self, point_a: jnp.ndarray, point_b: jnp.ndarray) -> jnp.ndarray:
    if self.use_compiled_kernels:
      kernel_hash = hash_args(point_a.shape, point_a.dtype.__str__())
      if kernel_hash in self.compiled_kernels:
        return self.compiled_kernels[kernel_hash]["point_add"](point_a, point_b)
    return self._point_add_jax(point_a, point_b)

  def point_double(self, point: jnp.ndarray) -> jnp.ndarray:
    if self.use_compiled_kernels:
      kernel_hash = hash_args(point.shape, point.dtype.__str__())
      if kernel_hash in self.compiled_kernels:
        return self.compiled_kernels[kernel_hash]["point_double"](point)
    return self._point_double_jax(point)

  # ------------------------------------------------------------------ #
  #  Compile / serialise                                               #
  # ------------------------------------------------------------------ #

  def _get_shape_dtype_structs(self, parameters: dict) -> list[jax.ShapeDtypeStruct]:
    batch_shape = parameters.get("batch_shape", None)
    if batch_shape is None:
      batch_shape = (parameters["batch_size"],)
    moduli_dim = self.jax_parameters.one.shape[-1]
    point_shape = (4,) + tuple(batch_shape) + (2, moduli_dim)
    if self.use_sharding:
      shard_axes = list(range(1, min(3, len(point_shape) - 2)))
      if not shard_axes:
        shard_axes = [1]
      named_sharding, padded_shape = self.create_named_sharding(
          shape=point_shape, axes=shard_axes)
      return [jax.ShapeDtypeStruct(padded_shape, jnp.uint32, sharding=named_sharding)]
    return [jax.ShapeDtypeStruct(point_shape, jnp.uint32)]

  def context_hash(self) -> str:
    return hash_args(
        self.__class__.__name__,
        self.fe_ctx.context_hash(),
        self.b,
        self.use_sharding,
    )

  def serialize(self, parameters: dict):
    sds = self._get_shape_dtype_structs(parameters)[0]
    kernel_hash = hash_args(self.context_hash(), parameters)
    cn = self.__class__.__name__
    store_jax_executable(self._point_add_jax, sds, sds,
                         name=f"{cn}_point_add_{kernel_hash}")
    store_jax_executable(self._point_double_jax, sds,
                         name=f"{cn}_point_double_{kernel_hash}")

  def compile(self, parameters: dict):
    sds = self._get_shape_dtype_structs(parameters)[0]
    kernel_hash = hash_args(self.context_hash(), parameters)
    cn = self.__class__.__name__
    point_add_kernel = load_jax_executable(f"{cn}_point_add_{kernel_hash}")
    point_double_kernel = load_jax_executable(f"{cn}_point_double_{kernel_hash}")
    if point_add_kernel is None or point_double_kernel is None:
      warnings.warn("Not found stored serialized compiled kernels, compiling...",
                    UserWarning, stacklevel=2)
    kernel_hash = hash_args(sds.shape, sds.dtype.__str__())
    self.compiled_kernels[kernel_hash] = {
        "point_add": point_add_kernel
        if point_add_kernel is not None
        else jax_jit_lower_compile(self._point_add_jax, sds, sds),
        "point_double": point_double_kernel
        if point_double_kernel is not None
        else jax_jit_lower_compile(self._point_double_jax, sds),
    }
    self.use_compiled_kernels = True


# ─────────────────────────────────────────────────────────────────────────────
# Projective with RCB-2015 complete formulas over Fq2  (identity-safe G2)
# ─────────────────────────────────────────────────────────────────────────────


class ProjectiveCompleteFq2ContextBase(EllipticCurveContextBase):
  """Pure-Python Fp2 reference for projective short-Weierstrass with
  Renes-Costello-Batina 2015 complete formulas.

  Curve:  y² = x³ + b over Fq2 with ``a = 0``.
  Coords: ``(X, Y, Z)`` with x = X/Z, y = Y/Z.  The identity is *any*
  point with ``Z = 0`` (convention: ``(0, 1, 0)``).

  Unlike :class:`XYZZWeierstrassFq2ContextBase`, the JAX kernels here use
  RCB Algorithm 7 (add) and Algorithm 9 (double), which are *unified*:
  the same straight-line formula produces the correct result for every
  input pair, including ``P + 0``, ``0 + Q``, ``P + (-P)``, ``P + P``,
  and ``0 + 0``.  No identity tracking, no pristine masks — MSM drivers
  can call ``point_add`` unconditionally.
  """

  def __init__(self, parameters: dict):
    self.parameters = parameters
    self.prime = parameters.get("prime", None)
    assert self.prime is not None, "prime must be provided"
    fe_ctx_class = parameters.get("field_extension_context_class", Fq2Context)
    fe_params = dict(parameters.get("field_extension_parameters", {}))
    fe_params.setdefault("prime", self.prime)
    fe_params.setdefault("finite_field_context_class",
                         parameters.get("finite_field_context_class"))
    fe_params.setdefault("finite_field_parameters",
                         parameters.get("finite_field_parameters", {}))
    fe_params.setdefault("quadratic_non_residue",
                         parameters.get("quadratic_non_residue"))
    self.fe_ctx: Fq2Context = fe_ctx_class(fe_params)
    self.ff_ctx = self.fe_ctx.finite_field_context
    self.qnr = parameters.get(
        "quadratic_non_residue", fe_params.get("quadratic_non_residue"))
    assert self.qnr is not None, "quadratic_non_residue must be provided"

    self.b = _as_fq2(parameters["b"], self.prime)
    a = parameters.get("a", 0)
    a_fp2 = _as_fq2(a, self.prime)
    if (a_fp2[0] % self.prime) != 0 or (a_fp2[1] % self.prime) != 0:
      raise ValueError(
          "ProjectiveCompleteFq2Context: only a = 0 is supported")
    # Curve constant 3·b precomputed once.
    b0, b1 = self.b
    self.b3 = ((3 * b0) % self.prime, (3 * b1) % self.prime)

    # Identity = (0 : 1 : 0).
    self.zero_point = [(0, 0), (1, 0), (0, 0)]

  def get_finite_field_context(self) -> FiniteFieldContextBase:
    return self.ff_ctx

  def get_field_extension_context(self) -> Fq2Context:
    return self.fe_ctx

  # ------------------------------------------------------------------ #
  #  Pure-Python Fp2 arithmetic (reused for the affine ↔ projective    #
  #  conversion and the on-curve oracle)                               #
  # ------------------------------------------------------------------ #

  def _fp2_add(self, a, b):
    return ((a[0] + b[0]) % self.prime, (a[1] + b[1]) % self.prime)

  def _fp2_sub(self, a, b):
    return ((a[0] - b[0]) % self.prime, (a[1] - b[1]) % self.prime)

  def _fp2_mul(self, a, b):
    a0, a1 = a; b0, b1 = b
    p = self.prime
    c0 = (a0 * b0 - self.qnr * a1 * b1) % p
    c1 = (a0 * b1 + a1 * b0) % p
    return (c0, c1)

  def _fp2_sq(self, a):
    return self._fp2_mul(a, a)

  def _fp2_inv(self, a):
    a0, a1 = a
    p = self.prime
    denom = (a0 * a0 + self.qnr * a1 * a1) % p
    if denom == 0:
      raise ZeroDivisionError("fp2_inv on zero element")
    denom_inv = pow(denom, p - 2, p)
    return ((a0 * denom_inv) % p, ((-a1) * denom_inv) % p)

  def _fp2_is_zero(self, a):
    return (a[0] % self.prime == 0) and (a[1] % self.prime == 0)

  def _fp2_reduce(self, a):
    return (a[0] % self.prime, a[1] % self.prime)

  # ------------------------------------------------------------------ #
  #  Coordinate conversion (affine ↔ projective)                       #
  # ------------------------------------------------------------------ #

  def _convert_to_proj(self, coordinates):
    """[x, y] (Fp2 affine) → [X, Y, Z] with Z = (1, 0).

    Identity-encoded affine ``[(0,0), (0,0)]`` round-trips via the
    Pippenger encoder by setting Z = 0 instead; this helper itself
    always emits Z = (1, 0) and leaves identity injection to the caller
    via ``identity_mask``.
    """
    assert len(coordinates) == 2, "Affine coords must be length 2"
    x = _as_fq2(coordinates[0], self.prime)
    y = _as_fq2(coordinates[1], self.prime)
    return [x, y, (1, 0)]

  def _convert_to_affine(self, coordinates):
    """[X, Y, Z] → [x, y].  Returns [(0,0),(0,0)] for identity (Z = 0)."""
    assert len(coordinates) == 3, "Projective coords must be length 3"
    X, Y, Z = (_as_fq2(c, self.prime) for c in coordinates)
    if self._fp2_is_zero(Z):
      return [(0, 0), (0, 0)]
    Zi = self._fp2_inv(Z)
    return [self._fp2_reduce(self._fp2_mul(X, Zi)),
            self._fp2_reduce(self._fp2_mul(Y, Zi))]

  # ------------------------------------------------------------------ #
  #  CPU oracle (RCB Algorithm 7 / 9 in Python)                        #
  # ------------------------------------------------------------------ #

  def _point_add_single(self, pa, pb):
    X1, Y1, Z1 = (_as_fq2(c, self.prime) for c in pa)
    X2, Y2, Z2 = (_as_fq2(c, self.prime) for c in pb)
    b3 = self.b3
    f = self
    t0 = f._fp2_mul(X1, X2)
    t1 = f._fp2_mul(Y1, Y2)
    t2 = f._fp2_mul(Z1, Z2)
    t3 = f._fp2_add(X1, Y1)
    t4 = f._fp2_add(X2, Y2)
    t3 = f._fp2_mul(t3, t4)
    t4 = f._fp2_add(t0, t1)
    t3 = f._fp2_sub(t3, t4)
    t4 = f._fp2_add(Y1, Z1)
    X3 = f._fp2_add(Y2, Z2)
    t4 = f._fp2_mul(t4, X3)
    X3 = f._fp2_add(t1, t2)
    t4 = f._fp2_sub(t4, X3)
    X3 = f._fp2_add(X1, Z1)
    Y3 = f._fp2_add(X2, Z2)
    X3 = f._fp2_mul(X3, Y3)
    Y3 = f._fp2_add(t0, t2)
    Y3 = f._fp2_sub(X3, Y3)
    X3 = f._fp2_add(t0, t0)
    t0 = f._fp2_add(X3, t0)
    t2 = f._fp2_mul(b3, t2)
    Z3 = f._fp2_add(t1, t2)
    t1 = f._fp2_sub(t1, t2)
    Y3 = f._fp2_mul(b3, Y3)
    X3 = f._fp2_mul(t4, Y3)
    t2 = f._fp2_mul(t3, t1)
    X3 = f._fp2_sub(t2, X3)
    Y3 = f._fp2_mul(Y3, t0)
    t1 = f._fp2_mul(t1, Z3)
    Y3 = f._fp2_add(t1, Y3)
    t0 = f._fp2_mul(t0, t3)
    Z3 = f._fp2_mul(Z3, t4)
    Z3 = f._fp2_add(Z3, t0)
    return [f._fp2_reduce(X3), f._fp2_reduce(Y3), f._fp2_reduce(Z3)]

  def _point_double_single(self, pa):
    X, Y, Z = (_as_fq2(c, self.prime) for c in pa)
    b3 = self.b3
    f = self
    t0 = f._fp2_mul(Y, Y)
    Z3 = f._fp2_add(t0, t0)
    Z3 = f._fp2_add(Z3, Z3)
    Z3 = f._fp2_add(Z3, Z3)
    t1 = f._fp2_mul(Y, Z)
    t2 = f._fp2_mul(Z, Z)
    t2 = f._fp2_mul(b3, t2)
    X3 = f._fp2_mul(t2, Z3)
    Y3 = f._fp2_add(t0, t2)
    Z3 = f._fp2_mul(t1, Z3)
    t1 = f._fp2_add(t2, t2)
    t2 = f._fp2_add(t1, t2)
    t0 = f._fp2_sub(t0, t2)
    Y3 = f._fp2_mul(t0, Y3)
    Y3 = f._fp2_add(X3, Y3)
    t1 = f._fp2_mul(X, Y)
    X3 = f._fp2_mul(t0, t1)
    X3 = f._fp2_add(X3, X3)
    return [f._fp2_reduce(X3), f._fp2_reduce(Y3), f._fp2_reduce(Z3)]

  def _point_add(self, point_a, point_b):
    depth = utils.nested_list_depth(point_a)
    if depth == 1:
      return self._point_add_single(point_a, point_b)
    if depth == 2:
      return [self._point_add_single(pa, pb)
              for pa, pb in zip(point_a, point_b)]
    raise ValueError(f"Invalid list depth {depth} for Fp2 projective add")

  def _point_double(self, point):
    depth = utils.nested_list_depth(point)
    if depth == 1:
      return self._point_double_single(point)
    if depth == 2:
      return [self._point_double_single(p) for p in point]
    raise ValueError(f"Invalid list depth {depth} for Fp2 projective double")

  # ------------------------------------------------------------------ #
  #  Curve membership helper                                           #
  # ------------------------------------------------------------------ #

  def _is_on_curve_affine(self, affine):
    """y² == x³ + b over Fp2?  Identity (x=y=0) returns True by convention."""
    x = _as_fq2(affine[0], self.prime)
    y = _as_fq2(affine[1], self.prime)
    if self._fp2_is_zero(x) and self._fp2_is_zero(y):
      return True
    lhs = self._fp2_sq(y)
    rhs = self._fp2_add(self._fp2_mul(x, self._fp2_sq(x)), self.b)
    return ((lhs[0] - rhs[0]) % self.prime == 0
            and (lhs[1] - rhs[1]) % self.prime == 0)


class ProjectiveCompleteFq2NDContext(
    ProjectiveCompleteFq2ContextBase, JaxKernelContextBase):
  """JAX/TPU EC context: projective short-Weierstrass with RCB-2015
  complete add/double, identity-safe under unified formulas.

  Computational format layout: ``(coords=3, *batch_dims, 2, moduli_dim)``:

    * axis 0   — projective coordinate index ``[X, Y, Z]``
    * axis -2  — Fp2 component index ``[c0, c1]``
    * axis -1  — DRNS / lazy precision moduli
    * middle axes — arbitrary nested batch shape

  Input points are nested lists of ``[x_W, y_W]`` Weierstrass-affine
  coordinates over Fp2.  ``to_computational_format`` injects Z = (1, 0).

  Identity-safe: ``point_add(P, Q)`` returns the correct result for any
  pair including identity operands.  This lets MSM drivers initialize
  buckets to ``(0, 1, 0)`` and always do ``bucket = bucket + point``
  without pristine bookkeeping.
  """

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    JaxKernelContextBase.__init__(self)
    self.jax_parameters = JaxParameters()
    self._init_jax_parameters()
    # Precomputed Fp2 constant 3·b in DRNS computational form; shape ``(2, M)``
    self._b3_cf: jnp.ndarray = self.fe_ctx.to_computational_format(self.b3)[0]

  # ------------------------------------------------------------------ #
  #  Format conversion                                                 #
  # ------------------------------------------------------------------ #

  @staticmethod
  def _affine_pair_depth(a):
    def is_affine_pair(node):
      return (isinstance(node, list)
              and len(node) == 2
              and isinstance(node[0], tuple)
              and len(node[0]) == 2
              and isinstance(node[0][0], int))
    if is_affine_pair(a):
      return 1
    if not isinstance(a, list):
      raise ValueError(f"Bad input shape: {type(a)}")
    return 1 + max(
        ProjectiveCompleteFq2NDContext._affine_pair_depth(item) for item in a)

  def to_computational_format(self, a) -> jnp.ndarray:
    depth = self._affine_pair_depth(a)
    if depth < 1:
      raise ValueError(
          f"Invalid list depth {depth} for Fp2 projective conversion")

    def recursive_extend(lst, d):
      if d == 1:
        return self._convert_to_proj(lst)
      return [recursive_extend(item, d - 1) for item in lst]

    extended = recursive_extend(a, depth)
    result = self.fe_ctx.to_computational_format(extended)

    if depth == 1:
      # (3, 2, M) → (3, 1, 2, M)
      result = jnp.expand_dims(result, axis=1)
    else:
      # (*batch, 3, 2, M) → (3, *batch, 2, M)
      ndim = result.ndim
      perm = (ndim - 3,) + tuple(range(ndim - 3)) + (ndim - 2, ndim - 1)
      result = result.transpose(perm)

    if self.use_sharding:
      shard_axes = list(range(1, min(3, result.ndim - 2)))
      if not shard_axes:
        shard_axes = [1]
      named_sharding, padded_shape = self.create_named_sharding(
          shape=result.shape, axes=shard_axes)
      result = pad_jax_array(result, padded_shape)
      return result.to_device(named_sharding)
    return result.to_device(jax.devices("tpu")[0])

  def to_original_format(self, a: jnp.ndarray):
    ndim = a.ndim
    if ndim < 3:
      raise ValueError(
          f"Expected at least 3D array (coord, 2, M), got {ndim}D")
    if ndim == 3:
      pts = self.fe_ctx.to_original_format(a)
      return self._convert_to_affine(pts)

    # (3, *batch, 2, M) → (*batch, 3, 2, M)
    perm = tuple(range(1, ndim - 2)) + (0, ndim - 2, ndim - 1)
    a = a.transpose(perm)
    pts = self.fe_ctx.to_original_format(a)

    batch_depth = ndim - 3

    def recursive_decode(lst, d):
      if d == 0:
        return self._convert_to_affine(lst)
      return [recursive_decode(item, d - 1) for item in lst]

    return recursive_decode(pts, batch_depth)

  # ------------------------------------------------------------------ #
  #  JAX parameters                                                    #
  # ------------------------------------------------------------------ #

  def _init_jax_parameters(self):
    self.jax_parameters.set_parameter(
        one=self.fe_ctx.to_computational_format((1, 0)),
    )

  # ------------------------------------------------------------------ #
  #  Point operations  (JAX kernels — RCB 2015 Algorithm 7 / 9)        #
  # ------------------------------------------------------------------ #

  def _fsub(self, a, b):
    """Fq2 subtract with canonicalized subtrahend.

    The DRNS-lazy ``_modular_negate`` underflows when its input exceeds the
    canonical bound (e.g. after an ``_modular_add``), so we reduce ``b``
    before subtracting.  Matches the XYZZ context's manual reduce-before-
    subtract calls (see ``XYZZWeierstrassFq2NDContext._point_add_jax``).
    """
    fe = self.fe_ctx
    return fe._modular_subtract(a, fe._modular_reduce(b))

  def _point_add_jax(self, point_a: jnp.ndarray, point_b: jnp.ndarray) -> jnp.ndarray:
    """RCB-2015 Algorithm 7 (complete add, a = 0) over Fq2.

    Layout: ``point_a, point_b`` have shape ``(3, *batch, 2, M)`` with
    coord axis 0 = ``[X, Y, Z]``.  Identity-safe under all input pairs.
    """
    fe = self.fe_ctx
    X1, Y1, Z1 = jnp.vsplit(point_a, 3)
    X2, Y2, Z2 = jnp.vsplit(point_b, 3)
    b3 = jnp.broadcast_to(self._b3_cf, X1.shape)

    t0 = fe._modular_multiply(X1, X2)
    t1 = fe._modular_multiply(Y1, Y2)
    t2 = fe._modular_multiply(Z1, Z2)
    t3 = fe._modular_add(X1, Y1)
    t4 = fe._modular_add(X2, Y2)
    t3 = fe._modular_multiply(t3, t4)
    t4 = fe._modular_add(t0, t1)
    t3 = self._fsub(t3, t4)
    t4 = fe._modular_add(Y1, Z1)
    X3 = fe._modular_add(Y2, Z2)
    t4 = fe._modular_multiply(t4, X3)
    X3 = fe._modular_add(t1, t2)
    t4 = self._fsub(t4, X3)
    X3 = fe._modular_add(X1, Z1)
    Y3 = fe._modular_add(X2, Z2)
    X3 = fe._modular_multiply(X3, Y3)
    Y3 = fe._modular_add(t0, t2)
    Y3 = self._fsub(X3, Y3)
    X3 = fe._modular_add(t0, t0)
    t0 = fe._modular_add(X3, t0)
    t2 = fe._modular_multiply(b3, t2)
    Z3 = fe._modular_add(t1, t2)
    t1 = self._fsub(t1, t2)
    Y3 = fe._modular_multiply(b3, Y3)
    X3 = fe._modular_multiply(t4, Y3)
    t2 = fe._modular_multiply(t3, t1)
    X3 = self._fsub(t2, X3)
    Y3 = fe._modular_multiply(Y3, t0)
    t1 = fe._modular_multiply(t1, Z3)
    Y3 = fe._modular_add(t1, Y3)
    t0 = fe._modular_multiply(t0, t3)
    Z3 = fe._modular_multiply(Z3, t4)
    Z3 = fe._modular_add(Z3, t0)

    return jnp.concatenate((X3, Y3, Z3), axis=0)

  def _point_double_jax(self, point: jnp.ndarray) -> jnp.ndarray:
    """RCB-2015 Algorithm 9 (complete double, a = 0) over Fq2."""
    fe = self.fe_ctx
    X, Y, Z = jnp.vsplit(point, 3)
    b3 = jnp.broadcast_to(self._b3_cf, X.shape)

    t0 = fe._modular_multiply(Y, Y)
    Z3 = fe._modular_add(t0, t0)
    Z3 = fe._modular_add(Z3, Z3)
    Z3 = fe._modular_add(Z3, Z3)
    t1 = fe._modular_multiply(Y, Z)
    t2 = fe._modular_multiply(Z, Z)
    t2 = fe._modular_multiply(b3, t2)
    X3 = fe._modular_multiply(t2, Z3)
    Y3 = fe._modular_add(t0, t2)
    Z3 = fe._modular_multiply(t1, Z3)
    t1 = fe._modular_add(t2, t2)
    t2 = fe._modular_add(t1, t2)
    t0 = self._fsub(t0, t2)
    Y3 = fe._modular_multiply(t0, Y3)
    Y3 = fe._modular_add(X3, Y3)
    t1 = fe._modular_multiply(X, Y)
    X3 = fe._modular_multiply(t0, t1)
    X3 = fe._modular_add(X3, X3)

    return jnp.concatenate((X3, Y3, Z3), axis=0)

  def point_add(self, point_a: jnp.ndarray, point_b: jnp.ndarray) -> jnp.ndarray:
    if self.use_compiled_kernels:
      kernel_hash = hash_args(point_a.shape, point_a.dtype.__str__())
      if kernel_hash in self.compiled_kernels:
        return self.compiled_kernels[kernel_hash]["point_add"](point_a, point_b)
    return self._point_add_jax(point_a, point_b)

  def point_double(self, point: jnp.ndarray) -> jnp.ndarray:
    if self.use_compiled_kernels:
      kernel_hash = hash_args(point.shape, point.dtype.__str__())
      if kernel_hash in self.compiled_kernels:
        return self.compiled_kernels[kernel_hash]["point_double"](point)
    return self._point_double_jax(point)

  # ------------------------------------------------------------------ #
  #  Compile / serialise                                               #
  # ------------------------------------------------------------------ #

  def _get_shape_dtype_structs(self, parameters: dict) -> list[jax.ShapeDtypeStruct]:
    batch_shape = parameters.get("batch_shape", None)
    if batch_shape is None:
      batch_shape = (parameters["batch_size"],)
    moduli_dim = self.jax_parameters.one.shape[-1]
    point_shape = (3,) + tuple(batch_shape) + (2, moduli_dim)
    if self.use_sharding:
      shard_axes = list(range(1, min(3, len(point_shape) - 2)))
      if not shard_axes:
        shard_axes = [1]
      named_sharding, padded_shape = self.create_named_sharding(
          shape=point_shape, axes=shard_axes)
      return [jax.ShapeDtypeStruct(padded_shape, jnp.uint32, sharding=named_sharding)]
    return [jax.ShapeDtypeStruct(point_shape, jnp.uint32)]

  def context_hash(self) -> str:
    return hash_args(
        self.__class__.__name__,
        self.fe_ctx.context_hash(),
        self.b,
        self.use_sharding,
    )

  def serialize(self, parameters: dict):
    sds = self._get_shape_dtype_structs(parameters)[0]
    kernel_hash = hash_args(self.context_hash(), parameters)
    cn = self.__class__.__name__
    store_jax_executable(self._point_add_jax, sds, sds,
                         name=f"{cn}_point_add_{kernel_hash}")
    store_jax_executable(self._point_double_jax, sds,
                         name=f"{cn}_point_double_{kernel_hash}")

  def compile(self, parameters: dict):
    sds = self._get_shape_dtype_structs(parameters)[0]
    kernel_hash = hash_args(self.context_hash(), parameters)
    cn = self.__class__.__name__
    point_add_kernel = load_jax_executable(f"{cn}_point_add_{kernel_hash}")
    point_double_kernel = load_jax_executable(f"{cn}_point_double_{kernel_hash}")
    if point_add_kernel is None or point_double_kernel is None:
      warnings.warn(
          "Not found stored serialized compiled kernels, compiling...",
          UserWarning, stacklevel=2)
    kernel_hash = hash_args(sds.shape, sds.dtype.__str__())
    self.compiled_kernels[kernel_hash] = {
        "point_add": point_add_kernel
        if point_add_kernel is not None
        else jax_jit_lower_compile(self._point_add_jax, sds, sds),
        "point_double": point_double_kernel
        if point_double_kernel is not None
        else jax_jit_lower_compile(self._point_double_jax, sds),
    }
    self.use_compiled_kernels = True
