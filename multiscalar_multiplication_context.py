from abc import ABC, abstractmethod
import ctypes
import math
import logging
import random
from typing import Optional, Tuple, Union
import copy
import numpy as np
import warnings
import jax
import jax.numpy as jnp
import jax.ffi as ffi
import warnings

import utils
from utils import JaxKernelContextBase, JaxParameters, hash_args, jax_jit_lower_compile, store_jax_executable, load_jax_executable
from finite_field_context import FiniteFieldContextBase
from elliptic_curve_context import (
    EllipticCurveContextBase,
    ExtendedTwistedEdwardsFq2NDContext,
    XYZZWeierstrassFq2NDContext,
    ProjectiveCompleteFq2NDContext,
)
from field_extension_context import Fq2Context

jax.config.update("jax_enable_x64", True)


class MultiscalarMultiplicationContextBase(ABC):

  def __init__(self, parameters: dict):
    self.parameters = parameters
    self.ec_ctx_class = parameters.get("elliptic_curve_context_class", None)
    assert self.ec_ctx_class is not None, "elliptic_curve_context_class must be provided"
    self.ec_ctx: EllipticCurveContextBase = self.ec_ctx_class(parameters.get("elliptic_curve_parameters", {}))

  def _padd(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return self.ec_ctx.point_add(a, b)

  @abstractmethod
  def to_computational_format(self, a: list) -> list:
    pass

  @abstractmethod
  def to_original_format(self, a: list) -> list:
    pass

  @abstractmethod
  def multiscalar_multiply(self, tiled_slices: jnp.ndarray):
    pass



class CPUDistributionMSMContextBase(MultiscalarMultiplicationContextBase):
  def __init__(self, parameters: dict):
    super().__init__(parameters)
    self._init_config_parameters()
    self._init_jax_data()
    self._init_cpu_kernels()
    self._init_point_parameters()


  def _init_point_parameters(self):
    raw_points = utils.read_external_msm_file(self.parameters.get("points_path"), "points")
    with jax.default_device(jax.devices("cpu")[0]):
      points = self.ec_ctx.to_computational_format(raw_points).to_device(jax.devices("cpu")[0])
    self.points = points.transpose(1, 0, 2)

  def _preprocess_scalars(self, scalars: list):
    tiled_scalar_list = utils.split_list(scalars, self.tile_length)
    tiled_slices_list = []
    for tiled_scalars in tiled_scalar_list:
      sliced_scalars = utils.slice_scalars(tiled_scalars, self.scalar_bits, self.slice_bits)
      tiled_slices_list.append(sliced_scalars)

    with jax.default_device(jax.devices("cpu")[0]):
      tiled_slices_list = jnp.array(tiled_slices_list, dtype=jnp.int32)
    return tiled_slices_list

  

class OldCPUDistributionMSMContext(CPUDistributionMSMContextBase):
  def __init__(self, parameters: dict):
    CPUDistributionMSMContextBase.__init__(self, parameters)

  def _init_config_parameters(self):
    self.coordinate_dim = self.parameters.get("coordinate_dim", 4)
    self.msm_length = self.parameters.get("msm_length")
    self.tile_length = self.parameters.get("tile_length")
    assert self.msm_length % self.tile_length == 0, "msm_length must be divisible by tile_length"
    self.tile_num = self.msm_length // self.tile_length
    self.slice_bits = self.parameters.get("slice_bits")
    self.scalar_bits = self.parameters.get("scalar_bits")
    self.order = self.parameters.get("order")
    self.window_num = int(math.ceil(self.scalar_bits / self.slice_bits))  #
    self.batch_window_num = self.window_num
    self.bucket_num_per_window = 2**self.slice_bits - 1  # Note: here remove the bucket_0
    self.bucket_num_last_window = self.order >> ((self.window_num - 1) * self.slice_bits)
    self.moduli_num = self.ec_ctx.get_finite_field_context().get_moduli_num()

    # Special bucket optimization
    self.log_special_duplication_ratio = math.ceil(math.log2(self.bucket_num_per_window / self.bucket_num_last_window))
    self.special_duplication_ratio = 2**self.log_special_duplication_ratio
    self.bucket_num_duplication = self.bucket_num_last_window * self.special_duplication_ratio

  def _init_jax_data(self):
    self.zero_point = self.ec_ctx.get_finite_field_context().to_computational_format(self.ec_ctx.zero_point)
    self.all_buckets = jnp.broadcast_to(
        self.zero_point.reshape(1, self.coordinate_dim, 1, self.moduli_num),
        (self.batch_window_num, self.coordinate_dim, self.bucket_num_per_window, self.moduli_num),
    )
    self.temp_sum = jnp.array([self.zero_point for _ in range(self.batch_window_num)]).transpose(1, 0, 2)
    self.window_sum = jnp.array([self.zero_point for _ in range(self.batch_window_num)]).transpose(1, 0, 2)

  def _init_cpu_kernels(self):
    expected_regular_bucket_size = self.tile_length / (self.bucket_num_per_window + 1)
    expected_special_bucket_size = math.ceil(self.tile_length / self.bucket_num_duplication)
    # print("expected_regular_bucket_size", expected_regular_bucket_size)
    # print("expected_special_bucket_size", expected_special_bucket_size)
    self.expend_ratio = self.parameters["c_kernel_ret_space_ratio"]
    self.c_kernel_regular_bucket_size = int(expected_regular_bucket_size * self.expend_ratio)
    self.c_kernel_special_bucket_size = int(expected_special_bucket_size * self.expend_ratio)

    regular_shape = (
        self.window_num - 1,
        self.bucket_num_per_window,
        self.c_kernel_regular_bucket_size,
        self.coordinate_dim,
        self.moduli_num,
    )

    special_shape = (
        self.bucket_num_last_window,
        self.special_duplication_ratio,
        self.c_kernel_special_bucket_size,
        self.coordinate_dim,
        self.moduli_num,
    )
    self.ba_input_regular_shape = regular_shape
    self.ba_input_special_shape = special_shape

    lib = ctypes.cdll.LoadLibrary("./c_kernels/distribution.so")
    jax.ffi.register_ffi_target("distribute_buf", jax.ffi.pycapsule(lib.DistributeBuf), platform="cpu")
    self.distribution_buf_c_kernel_call = ffi.ffi_call(
        "distribute_buf",
        (
            jax.ShapeDtypeStruct(regular_shape, jnp.uint32),
            jax.ShapeDtypeStruct(special_shape, jnp.uint32),
            jax.ShapeDtypeStruct((2,), jnp.uint32),
        ),
    )


  def _bucket_accumulation_per_window(self, buckets: jnp.ndarray, window_points: jnp.ndarray) -> jnp.ndarray:
    """Accumulate points within buckets for a single window.

    Args:
        buckets: Initial bucket values (coordinate_dim, bucket_dim, precision_dim).
        window_points: Points to accumulate (bucket_size_dim, coordinate_dim, bucket_dim, precision_dim).
        parameters: Computation parameters.

    Returns:
        Accumulated bucket values.
    """
    bucket_size_dim = window_points.shape[0]

    def scan_body(buckets, points):
      buckets = self._padd(buckets, points)
      return buckets, None

    buckets, _ = jax.lax.scan(scan_body, buckets, window_points, length=bucket_size_dim)
    return buckets

  def _bucket_accumulation_regular_windows(
      self,
      regular_buckets: jnp.ndarray,
      all_points: jnp.ndarray,
  ) -> jnp.ndarray:
    """Accumulate points for all regular windows.

    Args:
        regular_buckets: Initial bucket values (window_dim, coordinate_dim, bucket_dim, precision_dim).
        all_points: Points for all windows (window_dim, bucket_size_dim, coordinate_dim, bucket_dim, precision_dim).

    Returns:
        Accumulated bucket values for all regular windows.
    """
    window_dim = regular_buckets.shape[0]

    def scan_body(empty, window_bucket_point_pack):
      window_buckets, points = window_bucket_point_pack
      buckets = self._bucket_accumulation_per_window(window_buckets, points)
      return None, buckets

    _, buckets = jax.lax.scan(scan_body, None, (regular_buckets, all_points), length=window_dim)
    return buckets

  def _bucket_accumulation_last_window(
      self,
      buckets_in: jnp.ndarray,
      window_points: jnp.ndarray,
  ) -> jnp.ndarray:
    """Accumulate points for the last (special) window with duplication handling.

    Args:
        buckets_in: Initial bucket values (coordinate_dim, bucket_dim, precision_dim).
        window_points: Points with duplication (bucket_size_dim, coordinate_dim, bucket_dup_dim, bucket_dim, precision_dim).
        parameters: Computation parameters.

    Returns:
        Accumulated bucket values.
    """
    coordinate_dim, buckets_dim, precision_dim = buckets_in.shape
    bucket_size_dim, _, bucket_dup_dim, _, _ = window_points.shape

    # Reshape for processing
    window_points = window_points.reshape(bucket_size_dim, coordinate_dim, -1, precision_dim)
    base_dup_buckets = window_points[0]

    def scan_body(buckets, points):
      buckets = self._padd(buckets, points)
      return buckets, None

    dup_buckets, _ = jax.lax.scan(scan_body, base_dup_buckets, window_points[1:], length=bucket_size_dim - 1)

    # Reduce duplicated buckets using tree reduction
    log_bucket_dup_dim = int(math.log2(bucket_dup_dim))
    for _ in range(log_bucket_dup_dim):
      buckets_split = jnp.split(dup_buckets, 2, axis=1)
      dup_buckets = self._padd(buckets_split[0], buckets_split[1])

    # Add to input buckets
    buckets_in = self._padd(buckets_in, dup_buckets)
    return buckets_in

  def _bucket_accumulation_all_windows(
      self,
      all_buckets: jnp.ndarray,
      regular_points: jnp.ndarray,
      last_window_points: jnp.ndarray,
  ) -> jnp.ndarray:
    """Accumulate points for all windows with distributed optimization.

    This is the main bucket accumulation kernel that handles both regular
    windows and the special last window with duplication optimization.

    Args:
        all_buckets: Initial bucket values (window_dim, coordinate_dim, bucket_dim, precision_dim).
        regular_points: Points for regular windows (window_dim-1, bucket_num, bucket_size, coord_dim, prec_dim).
        last_window_points: Points for last window (bucket_num, dup_ratio, bucket_size, coord_dim, prec_dim).
        parameters: Computation parameters.

    Returns:
        Accumulated bucket values for all windows.
    """
    # Transpose for computation
    regular_points = regular_points.transpose(0, 2, 3, 1, 4)
    last_window_points = last_window_points.transpose(2, 3, 1, 0, 4)

    window_dim, coordinate_dim, buckets_dim, precision_dim = all_buckets.shape
    _, _, last_window_bucket_dup, last_window_bucket_dim, _ = last_window_points.shape

    # Process regular windows
    regular_buckets = all_buckets[: window_dim - 1]
    regular_buckets = self._bucket_accumulation_regular_windows(regular_buckets, regular_points)

    # Process last window
    last_point_buckets = all_buckets[window_dim - 1, :, :last_window_bucket_dim, :]
    last_blank_buckets = all_buckets[window_dim - 1, :, last_window_bucket_dim:, :]
    last_point_buckets = self._bucket_accumulation_last_window(last_point_buckets, last_window_points)

    # Combine last window buckets
    last_buckets = jax.lax.broadcast(jnp.concatenate((last_point_buckets, last_blank_buckets), axis=1), (1,))

    # Combine all buckets
    all_buckets = jnp.concatenate((regular_buckets, last_buckets), axis=0)
    return all_buckets

  def _bucket_reduction(
      self, all_buckets: jnp.ndarray, temp_sum: jnp.ndarray, window_sum: jnp.ndarray
  ) -> jnp.ndarray:
    """Reduce buckets to window sums using scan algorithm.

    Implements the bucket reduction phase of Pippenger's algorithm using
    a scan-based approach for efficiency.

    Args:
        all_buckets: Bucket values (window_dim, coordinate_dim, bucket_dim, precision_dim).
        temp_sum: Temporary sum array (coordinate_dim, window_dim, precision_dim).
        window_sum: Initial window sum (coordinate_dim, window_dim, precision_dim).
        bucket_num_in_window: Number of buckets per window.
        parameters: Computation parameters.

    Returns:
        Window sums (coordinate_dim, window_dim, precision_dim).
    """
    # Transpose for scan
    bucket_num_in_window = all_buckets.shape[2]
    all_buckets = all_buckets.transpose(2, 1, 0, 3)
    
    def scan_body(temp_and_window_sum_pack, buckets):
      temp_sum, window_sum = temp_and_window_sum_pack
      temp_sum = self._padd(temp_sum, buckets)
      window_sum = self._padd(window_sum, temp_sum)
      return (temp_sum, window_sum), None

    (_, window_sum), _ = jax.lax.scan(
        scan_body, (temp_sum, window_sum), all_buckets[:bucket_num_in_window], length=bucket_num_in_window , reverse=True
    )
    return window_sum

  def _window_merge(self, window_sum: jnp.ndarray) -> jnp.ndarray:
    """Merge window results into final MSM result using scan algorithm.

    Implements the window merging phase of Pippenger's algorithm.

    Args:
        window_sum: Window sums (coordinate_dim, window_dim, precision_dim).
        slice_length: Bit width of each window.
        parameters: Computation parameters.

    Returns:
        Final MSM result (coordinate_dim, precision_dim).
    """
    coordinate_dim, window_dim, precision_dim = window_sum.shape
    window_sum = window_sum.transpose(1, 0, 2).reshape((window_dim, coordinate_dim, 1, precision_dim))
    result = window_sum[window_dim - 1]

    def fori_loop_body(i, result):
      result = self._padd(result, result)
      return result

    def scan_body(result, window_sum):
      result = jax.lax.fori_loop(0, self.slice_bits, fori_loop_body, result)
      result = self._padd(result, window_sum)
      return result, None

    result, _ = jax.lax.scan(scan_body, result, window_sum[: window_dim - 1], reverse=True, length=window_dim - 1)
    result = result.reshape((coordinate_dim, precision_dim))
    return result

  def distribute_buckets(self, tiled_slices: jnp.ndarray, tiled_points: jnp.ndarray) -> jnp.ndarray:
    with jax.default_device(jax.devices("cpu")[0]):
      tiled_slices = jax.device_put(tiled_slices, jax.devices("cpu")[0])
      tiled_points = jax.device_put(tiled_points, jax.devices("cpu")[0])
      zero_point = jax.device_put(self.zero_point, jax.devices("cpu")[0])
      regular_buckets, last_window_buckets, metadata = self.distribution_buf_c_kernel_call(
          tiled_slices,
          tiled_points,
          zero_point,
          window_num=np.uint32(self.window_num),
          regular_bucket_num=np.uint32(self.bucket_num_per_window),
          special_bucket_num=np.uint32(self.bucket_num_last_window),
          msm_length=np.uint32(self.tile_length),
          fixed_regular_padding_size=np.uint32(self.c_kernel_regular_bucket_size),
          fixed_special_padding_size=np.uint32(self.c_kernel_special_bucket_size * self.special_duplication_ratio),
      )
      return regular_buckets, last_window_buckets

  def multiscalar_multiply(self, tiled_slices: jnp.ndarray):

    for tile_index in range(self.tile_num):
      idx_start = tile_index * self.tile_length
      idx_end = idx_start + self.tile_length
      tiled_slices_tile = tiled_slices[tile_index]
      tiled_points_tile = self.points[idx_start:idx_end]
      regular_buckets, last_window_buckets = self.distribute_buckets(tiled_slices_tile, tiled_points_tile)
      regular_buckets = jax.device_put(regular_buckets, jax.devices("tpu")[0])
      last_window_buckets = jax.device_put(last_window_buckets, jax.devices("tpu")[0])
      self.all_buckets = self._bucket_accumulation_all_windows(self.all_buckets, regular_buckets, last_window_buckets)


    window_sum = self._bucket_reduction(self.all_buckets, self.temp_sum, self.window_sum)
    result = self._window_merge(window_sum)
    return result

  def to_original_format(self, a: jnp.ndarray) -> list:
    return self.ec_ctx.to_original_format(a)

  def to_computational_format(self, scalars: list) -> jnp.ndarray:
    tiled_slices = self._preprocess_scalars(scalars)
    return tiled_slices

class CPUDistributionMSMContext(OldCPUDistributionMSMContext, JaxKernelContextBase):
  def __init__(self, parameters: dict):
    super().__init__(parameters)
    JaxKernelContextBase.__init__(self)

  def _init_config_parameters(self):
    self.coordinate_dim = self.parameters.get("coordinate_dim", 4)
    self.msm_length = self.parameters.get("msm_length")
    self.tile_length = self.parameters.get("tile_length")
    assert self.msm_length % self.tile_length == 0, "msm_length must be divisible by tile_length"
    self.tile_num = self.msm_length // self.tile_length
    self.slice_bits = self.parameters.get("slice_bits")
    self.scalar_bits = self.parameters.get("scalar_bits")
    self.order = self.parameters.get("order")
    self.window_num = int(math.ceil(self.scalar_bits / self.slice_bits))  #
    self.batch_window_num = self.window_num
    self.bucket_num_per_window = 2**self.slice_bits  # Note: Include Bucket 0
    orig_bucket_num_last_window = (self.order >> ((self.window_num - 1) * self.slice_bits)) + 1  # Note: Include Bucket 0
    # Pad to nearest value so bucket_num_last_window % 8 == 0.  The
    # kernel's bucket-distribution layout requires this invariant; the
    # earlier code dropped the padding when ratio > 10% (giving wrong
    # MSM results, e.g. orig=5, padding=3 → set to 0 corrupts the
    # duplication path).  Always pay the padding — a few extra empty
    # buckets cost nothing and keep the kernel correct.
    added_padding = (8 - orig_bucket_num_last_window % 8) % 8
    if added_padding > 0:
      print(
          f"[bucket_num_last_window] Was {orig_bucket_num_last_window}, "
          f"added {added_padding} to make it divisible by 8"
      )
    self.bucket_num_last_window = orig_bucket_num_last_window + added_padding
    self.moduli_num = self.ec_ctx.get_finite_field_context().get_moduli_num()

    # Special bucket optimization
    self.log_special_duplication_ratio = math.ceil(math.log2(self.bucket_num_per_window / self.bucket_num_last_window))
    self.special_duplication_ratio = 2**self.log_special_duplication_ratio
    self.bucket_num_duplication = self.bucket_num_last_window * self.special_duplication_ratio

  def _init_jax_data(self):
    self.zero_point = self.ec_ctx.get_finite_field_context().to_computational_format(self.ec_ctx.zero_point)
    self.all_buckets = jnp.broadcast_to(
        self.zero_point.reshape(1, self.coordinate_dim, 1, self.moduli_num),
        (self.batch_window_num, self.coordinate_dim, self.bucket_num_per_window, self.moduli_num),
    )
    self.temp_sum = jnp.array([self.zero_point for _ in range(self.batch_window_num)]).transpose(1, 0, 2)
    self.window_sum = jnp.array([self.zero_point for _ in range(self.batch_window_num)]).transpose(1, 0, 2)
    self.window_sum = jnp.broadcast_to(self.window_sum.reshape(self.coordinate_dim, 1, self.batch_window_num, self.moduli_num), ( self.coordinate_dim, self.bucket_num_per_window, self.batch_window_num, self.moduli_num))

    # Pre-store shardings for bucket accumulation inputs
    self.ba_all_buckets_sharding = None
    self.ba_regular_points_sharding = None
    self.ba_special_points_sharding = None
    # Pre-store shardings for bucket reduction inputs
    self.br_all_buckets_sharding = None
    self.br_window_sum_sharding = None
    # Pre-store shardings for window merge input
    self.wm_window_sum_sharding = None

  def _init_shardings(self):
    """Pre-compute and store all shardings for compiled/sharded execution."""
    all_buckets_shape = (self.batch_window_num, self.coordinate_dim, self.bucket_num_per_window, self.moduli_num)
    regular_points_shape = self.ba_input_regular_shape
    special_points_shape = self.ba_input_special_shape
    window_sum_shape = (self.coordinate_dim, self.bucket_num_per_window, self.batch_window_num, self.moduli_num)
    wm_window_sum_shape = (self.coordinate_dim, self.batch_window_num, self.moduli_num)

    # Bucket accumulation shardings
    self.ba_all_buckets_sharding, _ = self.create_named_sharding(shape=all_buckets_shape, axes=[2])
    self.ba_regular_points_sharding, _ = self.create_named_sharding(shape=regular_points_shape, axes=[1])
    self.ba_special_points_sharding, _ = self.create_named_sharding(shape=special_points_shape, axes=[0, 1])
    # Bucket reduction shardings
    self.br_all_buckets_sharding, _ = self.create_named_sharding(shape=all_buckets_shape, axes=[2])
    self.br_window_sum_sharding, _ = self.create_named_sharding(shape=window_sum_shape, axes=[1])
    # Window merge shardings
    self.wm_window_sum_sharding, _ = self.create_named_sharding(shape=wm_window_sum_shape, axes=[])

  def set_use_sharding(self, use_sharding: bool):
    super().set_use_sharding(use_sharding)
    if use_sharding:
      self._init_shardings()

  def _init_cpu_kernels(self):
    expected_regular_bucket_size = self.tile_length / (self.bucket_num_per_window)
    expected_special_bucket_size = math.ceil(self.tile_length / self.bucket_num_duplication)
    self.expend_ratio = self.parameters["c_kernel_ret_space_ratio"]
    self.c_kernel_regular_bucket_size = int(expected_regular_bucket_size * self.expend_ratio)
    self.c_kernel_special_bucket_size = int(expected_special_bucket_size * self.expend_ratio)

    regular_shape = (
        self.window_num - 1,
        self.bucket_num_per_window,
        self.c_kernel_regular_bucket_size,
        self.coordinate_dim,
        self.moduli_num,
    )

    special_shape = (
        self.bucket_num_last_window,
        self.special_duplication_ratio,
        self.c_kernel_special_bucket_size,
        self.coordinate_dim,
        self.moduli_num,
    )
    self.ba_input_regular_shape = regular_shape
    self.ba_input_special_shape = special_shape

    lib = ctypes.cdll.LoadLibrary("./c_kernels/distribution.so")
    jax.ffi.register_ffi_target("distribute_buf", jax.ffi.pycapsule(lib.DistributeBufZero), platform="cpu")
    self.distribution_buf_c_kernel_call = ffi.ffi_call(
        "distribute_buf",
        (
            jax.ShapeDtypeStruct(regular_shape, jnp.uint32),
            jax.ShapeDtypeStruct(special_shape, jnp.uint32),
            jax.ShapeDtypeStruct((2,), jnp.uint32),
        ),
    )

  def _bucket_accumulation_regular_windows_opt(
        self,
        regular_buckets: jnp.ndarray,
        all_points: jnp.ndarray,
    ) -> jnp.ndarray:
      """Accumulate points for all regular windows.

      Args:
          regular_buckets: Initial bucket values (window_dim, coordinate_dim, bucket_dim, precision_dim).
          all_points: Points for all windows (window_dim, bucket_size_dim, coordinate_dim, bucket_dim, precision_dim).

      Returns:
          Accumulated bucket values for all regular windows.
      """
      
      bucket_size_dim = all_points.shape[1]
      regular_buckets = regular_buckets.transpose(1,0,2,3)
      all_points = all_points.transpose(1, 2, 0, 3, 4)

      def scan_body(buckets, points):
        buckets = self._padd(buckets, points)
        return buckets, None

      buckets, _ = jax.lax.scan(scan_body, regular_buckets, all_points, length=bucket_size_dim)
      buckets = buckets.transpose(1, 0, 2, 3)
      return buckets

  def _bucket_accumulation_all_windows_opt(
      self,
      all_buckets: jnp.ndarray,
      regular_points: jnp.ndarray,
      last_window_points: jnp.ndarray,
  ) -> jnp.ndarray:
    """Accumulate points for all windows with distributed optimization.

    This is the main bucket accumulation kernel that handles both regular
    windows and the special last window with duplication optimization.

    Args:
        all_buckets: Initial bucket values (window_dim, coordinate_dim, bucket_dim, precision_dim).
        regular_points: Points for regular windows (window_dim-1, bucket_num, bucket_size, coord_dim, prec_dim).
        last_window_points: Points for last window (bucket_num, dup_ratio, bucket_size, coord_dim, prec_dim).
        parameters: Computation parameters.

    Returns:
        Accumulated bucket values for all windows.
    """
    # Transpose for computation
    regular_points = regular_points.transpose(0, 2, 3, 1, 4)
    last_window_points = last_window_points.transpose(2, 3, 1, 0, 4)

    window_dim, coordinate_dim, buckets_dim, precision_dim = all_buckets.shape
    _, _, last_window_bucket_dup, last_window_bucket_dim, _ = last_window_points.shape

    # Process regular windows
    regular_buckets = all_buckets[: window_dim - 1]
    regular_buckets = self._bucket_accumulation_regular_windows_opt(regular_buckets, regular_points)
    

    # Process last window
    last_point_buckets = all_buckets[window_dim - 1, :, :last_window_bucket_dim, :]
    last_blank_buckets = all_buckets[window_dim - 1, :, last_window_bucket_dim:, :]
    last_point_buckets = self._bucket_accumulation_last_window(last_point_buckets, last_window_points)

    # Combine last window buckets
    last_buckets = jax.lax.broadcast(jnp.concatenate((last_point_buckets, last_blank_buckets), axis=1), (1,))

    # Combine all buckets
    all_buckets = jnp.concatenate((regular_buckets, last_buckets), axis=0)
    return all_buckets

  def _bucket_reduction(
      self, all_buckets: jnp.ndarray, temp_sum: jnp.ndarray, window_sum: jnp.ndarray
  ) -> jnp.ndarray:
    """Reduce buckets to window sums using tree-based parallel algorithm.

    Computes S = sum_{i=0}^{n-1} i * B[i] per window via adjacent-pair
    tree reduction in O(log n) parallel steps.

    Tracks H = 2^k * bucket_sums (doubling each level) so that H[1::2]
    provides the correction weight directly. No separate bucket array needed.

    Args:
        all_buckets: Bucket values (window_dim, coordinate_dim, bucket_dim, precision_dim).
        temp_sum: Unused (kept for interface compatibility).
        window_sum: Initial window sum (coordinate_dim, bucket_dim, window_dim, precision_dim).

    Returns:
        Window sums (coordinate_dim, window_dim, precision_dim).
    """
    bucket_num_in_window = all_buckets.shape[2]
    # (window, coord, bucket, prec) → (coord, bucket, window, prec)
    all_buckets = all_buckets.transpose(1, 2, 0, 3)
    iter_num = int(math.log2(bucket_num_in_window))

    cd, m, wd, pd = all_buckets.shape
    for _ in range(iter_num):
      m = all_buckets.shape[1]
      half = m // 2

      all_buckets = all_buckets.reshape(cd, half, 2, wd, pd)
      window_sum = window_sum.reshape(cd, half, 2, wd, pd)

      all_buckets_left, all_buckets_right = all_buckets[:, :, 0], all_buckets[:, :, 1]
      window_sum_left, window_sum_right = window_sum[:, :, 0], window_sum[:, :, 1]

      window_sum = self._padd(self._padd(window_sum_left, window_sum_right), all_buckets_right)
      bucket_sum = self._padd(all_buckets_left, all_buckets_right)
      all_buckets = self._padd(bucket_sum, bucket_sum)

    return window_sum[:, 0]

  def _get_ba_shape_dtype_structs(self):
    """Get ShapeDtypeStructs for bucket_accumulation_all_windows inputs."""
    all_buckets_shape = (self.batch_window_num, self.coordinate_dim, self.bucket_num_per_window, self.moduli_num)
    regular_points_shape = self.ba_input_regular_shape
    special_points_shape = self.ba_input_special_shape

    if self.use_sharding:
      return [
          jax.ShapeDtypeStruct(all_buckets_shape, jnp.uint32, sharding=self.ba_all_buckets_sharding),
          jax.ShapeDtypeStruct(regular_points_shape, jnp.uint32, sharding=self.ba_regular_points_sharding),
          jax.ShapeDtypeStruct(special_points_shape, jnp.uint32, sharding=self.ba_special_points_sharding),
      ]
    return [
        jax.ShapeDtypeStruct(all_buckets_shape, jnp.uint32),
        jax.ShapeDtypeStruct(regular_points_shape, jnp.uint32),
        jax.ShapeDtypeStruct(special_points_shape, jnp.uint32),
    ]

  def _get_br_shape_dtype_structs(self):
    """Get ShapeDtypeStructs for bucket_reduction inputs."""
    all_buckets_shape = (self.batch_window_num, self.coordinate_dim, self.bucket_num_per_window, self.moduli_num)
    temp_sum_shape = (self.coordinate_dim, self.batch_window_num, self.moduli_num)
    window_sum_shape = (self.coordinate_dim, self.bucket_num_per_window, self.batch_window_num, self.moduli_num)

    if self.use_sharding:
      return [
          jax.ShapeDtypeStruct(all_buckets_shape, jnp.uint32, sharding=self.br_all_buckets_sharding),
          jax.ShapeDtypeStruct(temp_sum_shape, jnp.uint32),
          jax.ShapeDtypeStruct(window_sum_shape, jnp.uint32, sharding=self.br_window_sum_sharding),
      ]
    return [
        jax.ShapeDtypeStruct(all_buckets_shape, jnp.uint32),
        jax.ShapeDtypeStruct(temp_sum_shape, jnp.uint32),
        jax.ShapeDtypeStruct(window_sum_shape, jnp.uint32),
    ]

  def _get_wm_shape_dtype_structs(self):
    """Get ShapeDtypeStructs for window_merge input."""
    window_sum_shape = (self.coordinate_dim, self.batch_window_num, self.moduli_num)

    if self.use_sharding:
      return [jax.ShapeDtypeStruct(window_sum_shape, jnp.uint32, sharding=self.wm_window_sum_sharding)]
    return [jax.ShapeDtypeStruct(window_sum_shape, jnp.uint32)]

  def context_hash(self) -> str:
    return hash_args(
        self.__class__.__name__,
        self.ec_ctx.context_hash() if hasattr(self.ec_ctx, 'context_hash') else str(self.ec_ctx.__class__.__name__),
        self.slice_bits,
        self.scalar_bits,
        self.msm_length,
        self.tile_length,
        self.bucket_num_per_window,
        self.bucket_num_last_window,
        self.use_sharding,
    )

  def serialize(self, parameters: dict = None):
    ba_structs = self._get_ba_shape_dtype_structs()
    br_structs = self._get_br_shape_dtype_structs()
    wm_structs = self._get_wm_shape_dtype_structs()
    kernel_hash = hash_args(self.context_hash(), parameters if parameters else {})
    class_name = self.__class__.__name__

    store_jax_executable(
        self._bucket_accumulation_all_windows,
        ba_structs[0], ba_structs[1], ba_structs[2],
        name=f"{class_name}_bucket_accumulation_all_windows_{kernel_hash}",
    )
    store_jax_executable(
        self._bucket_reduction,
        br_structs[0], br_structs[1], br_structs[2],
        name=f"{class_name}_bucket_reduction_{kernel_hash}",
    )
    store_jax_executable(
        self._window_merge,
        wm_structs[0],
        name=f"{class_name}_window_merge_{kernel_hash}",
    )

  def compile(self, parameters: dict = None):
    ba_structs = self._get_ba_shape_dtype_structs()
    br_structs = self._get_br_shape_dtype_structs()
    wm_structs = self._get_wm_shape_dtype_structs()
    kernel_hash = hash_args(self.context_hash(), parameters if parameters else {})
    class_name = self.__class__.__name__

    ba_kernel = load_jax_executable(f"{class_name}_bucket_accumulation_all_windows_{kernel_hash}")
    br_kernel = load_jax_executable(f"{class_name}_bucket_reduction_{kernel_hash}")
    wm_kernel = load_jax_executable(f"{class_name}_window_merge_{kernel_hash}")

    if None in [ba_kernel, br_kernel, wm_kernel]:
      warnings.warn("Not found stored serialized compiled kernels for MSM, compiling...", UserWarning, stacklevel=2)

    ba_hash = hash_args(
        ba_structs[0].shape, ba_structs[0].dtype.__str__(),
        ba_structs[0].shape, ba_structs[0].dtype.__str__(),
    )
    br_hash = hash_args(
        br_structs[0].shape, br_structs[0].dtype.__str__(),
        br_structs[0].shape, br_structs[0].dtype.__str__(),
    )
    wm_hash = hash_args(
        wm_structs[0].shape, wm_structs[0].dtype.__str__(),
        wm_structs[0].shape, wm_structs[0].dtype.__str__(),
    )
    self.compiled_kernels.setdefault(ba_hash, {})["bucket_accumulation_all_windows"] = (
        ba_kernel
        if ba_kernel is not None
        else jax_jit_lower_compile(self._bucket_accumulation_all_windows, ba_structs[0], ba_structs[1], ba_structs[2])
    )
    self.compiled_kernels.setdefault(br_hash, {})["bucket_reduction"] = (
        br_kernel
        if br_kernel is not None
        else jax_jit_lower_compile(self._bucket_reduction, br_structs[0], br_structs[1], br_structs[2])
    )
    self.compiled_kernels.setdefault(wm_hash, {})["window_merge"] = (
        wm_kernel
        if wm_kernel is not None
        else jax_jit_lower_compile(self._window_merge, wm_structs[0])
    )
    self.use_compiled_kernels = True

  def bucket_accumulation_all_windows(self, all_buckets, regular_points, last_window_points):
    if self.use_compiled_kernels:
      kernel_hash = hash_args(
          all_buckets.shape, all_buckets.dtype.__str__(),
          all_buckets.shape, all_buckets.dtype.__str__(),
      )
      return self.compiled_kernels[kernel_hash]["bucket_accumulation_all_windows"](all_buckets, regular_points, last_window_points)
    else:
      return self._bucket_accumulation_all_windows_opt(all_buckets, regular_points, last_window_points)

  def bucket_reduction(self, all_buckets, temp_sum, window_sum):
    if self.use_compiled_kernels:
      kernel_hash = hash_args(
          all_buckets.shape, all_buckets.dtype.__str__(),
          all_buckets.shape, all_buckets.dtype.__str__(),
      )
      return self.compiled_kernels[kernel_hash]["bucket_reduction"](all_buckets, temp_sum, window_sum)
    else:
      return self._bucket_reduction(all_buckets, temp_sum, window_sum)

  def window_merge(self, window_sum):
    if self.use_compiled_kernels:
      kernel_hash = hash_args(
          window_sum.shape, window_sum.dtype.__str__(),
          window_sum.shape, window_sum.dtype.__str__(),
      )
      return self.compiled_kernels[kernel_hash]["window_merge"](window_sum)
    else:
      return self._window_merge(window_sum)

  def multiscalar_multiply(self, tiled_slices: jnp.ndarray):

    for tile_index in range(self.tile_num):
      idx_start = tile_index * self.tile_length
      idx_end = idx_start + self.tile_length
      tiled_slices_tile = tiled_slices[tile_index]
        # tiled_points_tile = self.points[:, idx_start:idx_end].transpose(1, 0, 2)
      tiled_points_tile = self.points[idx_start:idx_end]
      regular_buckets, last_window_buckets = self.distribute_buckets(tiled_slices_tile, tiled_points_tile)
      if self.use_sharding:
        regular_buckets = regular_buckets.to_device(self.ba_regular_points_sharding)
        last_window_buckets = last_window_buckets.to_device(self.ba_special_points_sharding)
      else:
        regular_buckets = jax.device_put(regular_buckets, jax.devices("tpu")[0])
        last_window_buckets = jax.device_put(last_window_buckets, jax.devices("tpu")[0])
      self.all_buckets = self.bucket_accumulation_all_windows(self.all_buckets, regular_buckets, last_window_buckets)
      
    window_sum = self.bucket_reduction(self.all_buckets, self.temp_sum, self.window_sum)
    result = self.window_merge(window_sum)
    return result



class TPUDistributionMSMContext(CPUDistributionMSMContext):
  """MSM context where bucket distribution runs on TPU using a sort-based
  bucketize algorithm instead of a CPU C kernel.

  Distribution becomes a major TPU kernel alongside bucket_accumulation,
  bucket_reduction, and window_merge — all four are compiled by `compile()`
  and dispatched the same way.
  """

  def __init__(self, parameters: dict):
    MultiscalarMultiplicationContextBase.__init__(self, parameters)
    self._init_config_parameters()
    self._init_jax_data()
    self._init_kernels()
    self._init_point_parameters()
    JaxKernelContextBase.__init__(self)

  def _init_kernels(self):
    expected_regular_bucket_size = self.tile_length / self.bucket_num_per_window
    expected_special_bucket_size = math.ceil(self.tile_length / self.bucket_num_duplication)
    self.expend_ratio = self.parameters["c_kernel_ret_space_ratio"]
    self.regular_bucket_size = int(expected_regular_bucket_size * self.expend_ratio)
    self.special_bucket_size = int(expected_special_bucket_size * self.expend_ratio)

    self.ba_input_regular_shape = (
        self.window_num - 1,
        self.bucket_num_per_window,
        self.regular_bucket_size,
        self.coordinate_dim,
        self.moduli_num,
    )
    self.ba_input_special_shape = (
        self.bucket_num_last_window,
        self.special_duplication_ratio,
        self.special_bucket_size,
        self.coordinate_dim,
        self.moduli_num,
    )
    self.dist_input_slices_shape = (self.window_num, self.tile_length)
    self.dist_input_points_shape = (self.tile_length, self.coordinate_dim, self.moduli_num)

  def _init_point_parameters(self):
    raw_points = utils.read_external_msm_file(self.parameters.get("points_path"), "points")
    points = self.ec_ctx.to_computational_format(raw_points)
    points = points.transpose(1, 0, 2)  # (N, coordinate_dim, moduli_num)
    self.points = jax.device_put(points, jax.devices("tpu")[0])

  def _bucketize_regular_windows(self, items: jnp.ndarray, bucket_ids: jnp.ndarray) -> jnp.ndarray:
    """Sort-based bucketization for regular windows.

    Args:
        items: Points to distribute (tile_length, coordinate_dim, moduli_num).
        bucket_ids: Slice values for the regular windows (window_num - 1, tile_length).

    Returns:
        Bucketed points (window_num - 1, bucket_num_per_window, regular_bucket_size,
                         coordinate_dim, moduli_num).
    """
    n = self.tile_length
    b = self.window_num - 1

    bucket_ids_i32 = bucket_ids.astype(jnp.int32)
    sorted_indices = jnp.argsort(bucket_ids_i32, axis=1, stable=True)
    sorted_buckets = jnp.take_along_axis(bucket_ids_i32, sorted_indices, axis=1)

    is_boundary = jnp.concatenate([
        jnp.ones_like(sorted_buckets[:, :1], dtype=jnp.bool_),
        sorted_buckets[:, 1:] != sorted_buckets[:, :-1],
    ], axis=1)
    positions = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32), (b, n))
    boundary_positions = jnp.where(is_boundary, positions, jnp.int32(0))
    bucket_starts = jax.lax.associative_scan(jnp.maximum, boundary_positions, axis=1)
    within_rank = positions - bucket_starts

    
    batch_idx = jnp.broadcast_to(jnp.arange(b, dtype=jnp.int32)[:, None], (b, n))
    zeros_bn = jnp.zeros((b, n), dtype=jnp.int32)
    inv_rank = zeros_bn.at[batch_idx, sorted_indices].set(within_rank)
    items_b = jnp.broadcast_to(items[None, :, :, :], (b, n, 4, 32))

    # items_sorted = items[sorted_indices]


    output = jnp.broadcast_to(
        self.zero_point,
        (b, self.bucket_num_per_window, self.regular_bucket_size, self.coordinate_dim, self.moduli_num),
    ).copy()
    # output = output.at[batch_idx, sorted_buckets, within_rank].set(items_sorted)
    output = output.at[batch_idx, bucket_ids_i32, inv_rank].set(items_b)
    return output

  def _bucketize_last_window(self, items: jnp.ndarray, last_slices: jnp.ndarray) -> jnp.ndarray:
    """Sort-based bucketization for the last (special) window with duplication.

    Each logical bucket s in [0, bucket_num_last_window) is replicated
    special_duplication_ratio times. Item i is mapped to duplicate (i mod sdup),
    spreading items across the duplicates so the per-slot capacity stays at
    special_bucket_size. Reduction back to logical buckets is handled later by
    _bucket_accumulation_last_window.

    Args:
        items: Points to distribute (tile_length, coordinate_dim, moduli_num).
        last_slices: Slice values for the last window (tile_length,).

    Returns:
        Bucketed points (bucket_num_last_window, special_duplication_ratio,
                         special_bucket_size, coordinate_dim, moduli_num).
    """
    n = self.tile_length
    sdup = self.special_duplication_ratio

    dup_ids = jnp.arange(n, dtype=jnp.int32) % sdup
    bucket_ids_i32 = last_slices.astype(jnp.int32) * sdup + dup_ids

    sorted_indices = jnp.argsort(bucket_ids_i32, stable=True)
    sorted_buckets = bucket_ids_i32[sorted_indices]

    is_boundary = jnp.concatenate([
        jnp.ones((1,), dtype=jnp.bool_),
        sorted_buckets[1:] != sorted_buckets[:-1],
    ])
    positions = jnp.arange(n, dtype=jnp.int32)
    boundary_positions = jnp.where(is_boundary, positions, jnp.int32(0))
    bucket_starts = jax.lax.associative_scan(jnp.maximum, boundary_positions)
    within_rank = positions - bucket_starts

    zeros_n = jnp.zeros((n,), dtype=jnp.int32)
    inv_rank = zeros_n.at[sorted_indices].set(within_rank)

    output = jnp.broadcast_to(
        self.zero_point,
        (self.bucket_num_duplication, self.special_bucket_size, self.coordinate_dim, self.moduli_num),
    ).copy()
    output = output.at[bucket_ids_i32, inv_rank].set(items)
    return output.reshape(
        self.bucket_num_last_window,
        self.special_duplication_ratio,
        self.special_bucket_size,
        self.coordinate_dim,
        self.moduli_num,
    )

  def _distribute_buckets(self, regular_slices: jnp.ndarray, last_window_slice: jnp.ndarray, tiled_points: jnp.ndarray):
    """Major TPU kernel: distribute one tile of points into buckets.

    Args:
        regular_slices: Slice values for regular windows (window_num - 1, tile_length).
        last_window_slice: Slice values for the last window (tile_length,).
        tiled_points: Points for one tile (tile_length, coordinate_dim, moduli_num).

    Returns:
        regular_buckets: (window_num - 1, bucket_num_per_window, regular_bucket_size,
                          coordinate_dim, moduli_num).
        last_window_buckets: (bucket_num_last_window, special_duplication_ratio,
                              special_bucket_size, coordinate_dim, moduli_num).
    """
    regular_buckets = self._bucketize_regular_windows(tiled_points, regular_slices)
    last_window_buckets = self._bucketize_last_window(tiled_points, last_window_slice)
    return regular_buckets, last_window_buckets

  def _get_dist_shape_dtype_structs(self):
    """Get ShapeDtypeStructs for _distribute_buckets inputs."""
    regular_slices_shape = (self.window_num - 1, self.tile_length)
    last_window_slice_shape = (self.tile_length,)
    return [
        jax.ShapeDtypeStruct(regular_slices_shape, jnp.int32),
        jax.ShapeDtypeStruct(last_window_slice_shape, jnp.int32),
        jax.ShapeDtypeStruct(self.dist_input_points_shape, jnp.uint32),
    ]

  def compile(self, parameters: dict = None):
    dist_structs = self._get_dist_shape_dtype_structs()
    ba_structs = self._get_ba_shape_dtype_structs()
    br_structs = self._get_br_shape_dtype_structs()
    wm_structs = self._get_wm_shape_dtype_structs()

    dist_hash = hash_args(
        dist_structs[0].shape, dist_structs[0].dtype.__str__(),
        dist_structs[1].shape, dist_structs[1].dtype.__str__(),
        dist_structs[2].shape, dist_structs[2].dtype.__str__(),
    )
    ba_hash = hash_args(
        ba_structs[0].shape, ba_structs[0].dtype.__str__(),
        ba_structs[0].shape, ba_structs[0].dtype.__str__(),
    )
    br_hash = hash_args(
        br_structs[0].shape, br_structs[0].dtype.__str__(),
        br_structs[0].shape, br_structs[0].dtype.__str__(),
    )
    wm_hash = hash_args(
        wm_structs[0].shape, wm_structs[0].dtype.__str__(),
        wm_structs[0].shape, wm_structs[0].dtype.__str__(),
    )

    self.compiled_kernels.setdefault(dist_hash, {})["distribute_buckets"] = (
        jax_jit_lower_compile(self._distribute_buckets, dist_structs[0], dist_structs[1], dist_structs[2])
    )
    self.compiled_kernels.setdefault(ba_hash, {})["bucket_accumulation_all_windows"] = (
        jax_jit_lower_compile(self._bucket_accumulation_all_windows_opt, ba_structs[0], ba_structs[1], ba_structs[2])
    )
    self.compiled_kernels.setdefault(br_hash, {})["bucket_reduction"] = (
        jax_jit_lower_compile(self._bucket_reduction, br_structs[0], br_structs[1], br_structs[2])
    )
    self.compiled_kernels.setdefault(wm_hash, {})["window_merge"] = (
        jax_jit_lower_compile(self._window_merge, wm_structs[0])
    )
    self.use_compiled_kernels = True

  def distribute_buckets(self, regular_slices, last_window_slice, tiled_points):
    if self.use_compiled_kernels:
      kernel_hash = hash_args(
          regular_slices.shape, regular_slices.dtype.__str__(),
          last_window_slice.shape, last_window_slice.dtype.__str__(),
          tiled_points.shape, tiled_points.dtype.__str__(),
      )
      return self.compiled_kernels[kernel_hash]["distribute_buckets"](regular_slices, last_window_slice, tiled_points)
    else:
      return self._distribute_buckets(regular_slices, last_window_slice, tiled_points)

  def multiscalar_multiply(self, tiled_slices: jnp.ndarray):
    tpu = jax.devices("tpu")[0]
    tiled_slices = jax.device_put(tiled_slices, tpu)

    for tile_index in range(self.tile_num):
      idx_start = tile_index * self.tile_length
      idx_end = idx_start + self.tile_length
      tiled_slices_tile = tiled_slices[tile_index]
      regular_slices_tile = tiled_slices_tile[: self.window_num - 1]
      last_window_slice_tile = tiled_slices_tile[self.window_num - 1]
      tiled_points_tile = self.points[idx_start:idx_end]
      regular_buckets, last_window_buckets = self.distribute_buckets(regular_slices_tile, last_window_slice_tile, tiled_points_tile)
      self.all_buckets = self.bucket_accumulation_all_windows(self.all_buckets, regular_buckets, last_window_buckets)

    window_sum = self.bucket_reduction(self.all_buckets, self.temp_sum, self.window_sum)
    result = self.window_merge(window_sum)
    return result

  
class FusionMSMContext(MultiscalarMultiplicationContextBase, JaxKernelContextBase):
  def __init__(self, parameters: dict):
    super().__init__(parameters)
    MultiscalarMultiplicationContextBase.__init__(self, parameters)
    self._init_config_parameters()
    self._init_jax_data()
    self._init_point_parameters()
    JaxKernelContextBase.__init__(self)
    self.use_fused = False

  def _init_config_parameters(self):
    self.coordinate_dim = self.parameters.get("coordinate_dim", 4)
    self.msm_length = self.parameters.get("msm_length")
    self.tile_length = self.parameters.get("tile_length")
    assert self.msm_length % self.tile_length == 0 and self.msm_length >= self.tile_length, "msm_length must be divisible by tile_length and greater than or equal to tile_length"

    self.tile_num = self.msm_length // self.tile_length
    self.slice_bits = self.parameters.get("slice_bits")
    if self.slice_bits != 15:
      warnings.warn(f"Slice bits {self.slice_bits} may cause performance issue, using 15 instead.")
    if 2**self.slice_bits > self.tile_length:
      warnings.warn(f"2**{self.slice_bits} is greater than tile_length, which may cause performance issue.")
    self.scalar_bits = self.parameters.get("scalar_bits")
    self.order = self.parameters.get("order")
    self.window_num = int(math.ceil(self.scalar_bits / self.slice_bits))  #
    self.batch_window_num = self.window_num
    self.bucket_num_per_window = 2**self.slice_bits  # Note: Include Bucket 0
    orig_bucket_num_last_window = (self.order >> ((self.window_num - 1) * self.slice_bits)) + 1  # Note: Include Bucket 0
    # Pad to nearest value so bucket_num_last_window % 8 == 0.  The
    # kernel's bucket-distribution layout requires this invariant; the
    # earlier code dropped the padding when ratio > 10% (giving wrong
    # MSM results, e.g. orig=5, padding=3 → set to 0 corrupts the
    # duplication path).  Always pay the padding — a few extra empty
    # buckets cost nothing and keep the kernel correct.
    added_padding = (8 - orig_bucket_num_last_window % 8) % 8
    if added_padding > 0:
      print(
          f"[bucket_num_last_window] Was {orig_bucket_num_last_window}, "
          f"added {added_padding} to make it divisible by 8"
      )
    self.bucket_num_last_window = orig_bucket_num_last_window + added_padding
    self.moduli_num = self.ec_ctx.get_finite_field_context().get_moduli_num()

    # Special bucket optimization
    self.log_special_duplication_ratio = math.ceil(math.log2(self.bucket_num_per_window / self.bucket_num_last_window))
    self.special_duplication_ratio = 2**self.log_special_duplication_ratio
    self.bucket_num_duplication = self.bucket_num_last_window * self.special_duplication_ratio
    expected_regular_bucket_size = self.tile_length / self.bucket_num_per_window
    expected_special_bucket_size = math.ceil(self.tile_length / self.bucket_num_duplication)
    self.expend_ratio = self.parameters["c_kernel_ret_space_ratio"]
    self.regular_bucket_size = int(expected_regular_bucket_size * self.expend_ratio)
    self.special_bucket_size = int(expected_special_bucket_size * self.expend_ratio)

  def _init_point_parameters(self):
    if self.parameters.get("points_path") is not None:
      raw_points = utils.read_external_msm_file(self.parameters.get("points_path"), "points")
      points = self.ec_ctx.to_computational_format(raw_points)
      points = points.transpose(1, 0, 2)  # (N, coordinate_dim, moduli_num)
    else:
      points = jax.random.randint(jax.random.PRNGKey(0), (self.msm_length, self.coordinate_dim, self.moduli_num), 0, 2**16, dtype=jnp.uint32)
    # Reshape once here into per-tile layout so both execution paths index
    # tiles with a single leading-axis lookup (self.points[tile_index]) and
    # the fused path can pass self.points directly as tiled_points.
    self.points = points.reshape(
        self.tile_num, self.tile_length, self.coordinate_dim, self.moduli_num,
    )

  def _init_jax_data(self):
    self.zero_point = self.ec_ctx.get_finite_field_context().to_computational_format(self.ec_ctx.zero_point)
    all_buckets = jnp.broadcast_to(
        self.zero_point.reshape(self.coordinate_dim, 1, 1, self.moduli_num),
        (self.coordinate_dim, self.batch_window_num,  self.bucket_num_per_window, self.moduli_num),
    )
    self.regular_window_buckets = all_buckets[:, : self.window_num - 1]
    self.last_window_buckets = all_buckets[:, self.window_num - 1]
    self.window_sum = jnp.broadcast_to(
        self.zero_point.reshape(self.coordinate_dim, 1, 1, self.moduli_num),
        (self.coordinate_dim, self.bucket_num_per_window, self.batch_window_num, self.moduli_num),
    )
    # Cache the initial zero-broadcasts so ``reset()`` is a single pointer
    # swap per MSM call instead of rebuilding the broadcasts.  The shapes
    # never change across calls, so this is paid exactly once at init.
    self._zero_templates = {
        "regular_window_buckets": self.regular_window_buckets,
        "last_window_buckets":    self.last_window_buckets,
        "window_sum":             self.window_sum,
    }

  def reset(self):
    """Snap the mutable per-call bucket / window-sum state back to its
    initial all-identity form.  Callers should invoke this between
    successive :meth:`multiscalar_multiply` calls (the kernel mutates
    these attributes in place during the call)."""
    for name, zero in self._zero_templates.items():
      setattr(self, name, zero)

  def _preprocess_scalars(self, scalars: list):
    tiled_scalar_list = utils.split_list(scalars, self.tile_length)
    tiled_slices_list = []
    for tiled_scalars in tiled_scalar_list:
      sliced_scalars = utils.slice_scalars(tiled_scalars, self.scalar_bits, self.slice_bits)
      tiled_slices_list.append(sliced_scalars)
    tiled_slices_list = jnp.array(tiled_slices_list, dtype=jnp.int32)
    return tiled_slices_list

  def to_original_format(self, a: jnp.ndarray) -> list:
    return self.ec_ctx.to_original_format(a)

  def to_computational_format(self, scalars: Optional[list] = None) -> jnp.ndarray:
    if scalars is None:
      warnings.warn("No scalars provided, generating random scalars.")
      scalars = [random.randint(0, self.order - 1) for _ in range(self.msm_length)]
    tiled_slices = self._preprocess_scalars(scalars)

    # When sharding is enabled, pad tiled_slices along the tile_length axis
    # once here so every per-tile slice (regular + last) already matches the
    # padded shape expected by the compiled BBA kernel. The hot loop can then
    # just `to_device(...)` without any shape-time work.
    if self.use_sharding:
      from utils import pad_jax_array
      padded_tile_length = self.bba_reg_slices_padded[1]
      if tiled_slices.shape[-1] != padded_tile_length:
        warnings.warn(f"Tiled slices shape {tiled_slices.shape} does not match padded tile length {padded_tile_length}, padding to {target_shape}.")
        target_shape = (tiled_slices.shape[0], tiled_slices.shape[1], padded_tile_length)
        tiled_slices = pad_jax_array(tiled_slices, target_shape)
    return tiled_slices

  def _bucketize_regular_windows(self, items: jnp.ndarray, bucket_ids: jnp.ndarray) -> jnp.ndarray:
    """Sort-based bucketization for regular windows.

    Args:
        items: Points to distribute (tile_length, coordinate_dim, moduli_num).
        bucket_ids: Slice values for the regular windows (window_num - 1, tile_length).

    Returns:
        Bucketed points (regular_bucket_size, coordinate_dim,
                         window_num - 1, bucket_num_per_window, moduli_num).
    """
    n = self.tile_length
    b = self.window_num - 1

    bucket_ids_i32 = bucket_ids.astype(jnp.int32)
    sorted_indices = jnp.argsort(bucket_ids_i32, axis=1, stable=True)
    sorted_buckets = jnp.take_along_axis(bucket_ids_i32, sorted_indices, axis=1)

    is_boundary = jnp.concatenate([
        jnp.ones_like(sorted_buckets[:, :1], dtype=jnp.bool_),
        sorted_buckets[:, 1:] != sorted_buckets[:, :-1],
    ], axis=1)
    positions = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32), (b, n))
    boundary_positions = jnp.where(is_boundary, positions, jnp.int32(0))
    bucket_starts = jax.lax.associative_scan(jnp.maximum, boundary_positions, axis=1)
    within_rank = positions - bucket_starts

    batch_idx = jnp.broadcast_to(jnp.arange(b, dtype=jnp.int32)[:, None], (b, n))
    zeros_bn = jnp.zeros((b, n), dtype=jnp.int32)
    inv_rank = zeros_bn.at[batch_idx, sorted_indices].set(within_rank)
    items_b = jnp.broadcast_to(items[None, :, :, :], (b, n, self.coordinate_dim, self.moduli_num))

    output = jnp.broadcast_to(
        self.zero_point.reshape(1, self.coordinate_dim, 1, 1, self.moduli_num),
        (self.regular_bucket_size, self.coordinate_dim, b, self.bucket_num_per_window, self.moduli_num),
    ).copy()
    output = output.at[inv_rank, :, batch_idx, bucket_ids_i32].set(items_b)
    return output

  def _bucketize_last_window(self, items: jnp.ndarray, last_slices: jnp.ndarray) -> jnp.ndarray:
    """Sort-based bucketization for the last (special) window with duplication.

    Each logical bucket s in [0, bucket_num_last_window) is replicated
    special_duplication_ratio times. Item i is mapped to duplicate (i mod sdup),
    spreading items across the duplicates so the per-slot capacity stays at
    special_bucket_size. Reduction back to logical buckets is handled later by
    _bucket_accumulation_last_window.

    Args:
        items: Points to distribute (tile_length, coordinate_dim, moduli_num).
        last_slices: Slice values for the last window (tile_length,).

    Returns:
        Bucketed points (special_bucket_size, coordinate_dim,
                         special_duplication_ratio, bucket_num_last_window, moduli_num).
    """
    n = self.tile_length
    sdup = self.special_duplication_ratio

    dup_ids = jnp.arange(n, dtype=jnp.int32) % sdup
    bucket_ids_i32 = dup_ids * self.bucket_num_last_window + last_slices.astype(jnp.int32)

    sorted_indices = jnp.argsort(bucket_ids_i32, stable=True)
    sorted_buckets = bucket_ids_i32[sorted_indices]

    is_boundary = jnp.concatenate([
        jnp.ones((1,), dtype=jnp.bool_),
        sorted_buckets[1:] != sorted_buckets[:-1],
    ])
    positions = jnp.arange(n, dtype=jnp.int32)
    boundary_positions = jnp.where(is_boundary, positions, jnp.int32(0))
    bucket_starts = jax.lax.associative_scan(jnp.maximum, boundary_positions)
    within_rank = positions - bucket_starts

    zeros_n = jnp.zeros((n,), dtype=jnp.int32)
    inv_rank = zeros_n.at[sorted_indices].set(within_rank)

    output = jnp.broadcast_to(
        self.zero_point.reshape(1, self.coordinate_dim, 1, self.moduli_num),
        (self.special_bucket_size, self.coordinate_dim, self.bucket_num_duplication, self.moduli_num),
    ).copy()
    output = output.at[inv_rank, :, bucket_ids_i32].set(items)
    return output.reshape(
        self.special_bucket_size,
        self.coordinate_dim,
        self.special_duplication_ratio,
        self.bucket_num_last_window,
        self.moduli_num,
    )

  def _distribute_buckets(self, regular_slices: jnp.ndarray, last_window_slice: jnp.ndarray, tiled_points: jnp.ndarray):
    """Major TPU kernel: distribute one tile of points into buckets.

    Args:
        regular_slices: Slice values for regular windows (window_num - 1, tile_length).
        last_window_slice: Slice values for the last window (tile_length,).
        tiled_points: Points for one tile (tile_length, coordinate_dim, moduli_num).

    Returns:
        regular_buckets: (regular_bucket_size, coordinate_dim,
                          window_num - 1, bucket_num_per_window, moduli_num).
        last_window_buckets: (special_bucket_size, coordinate_dim,
                              special_duplication_ratio, bucket_num_last_window, moduli_num).
    """
    regular_buckets = self._bucketize_regular_windows(tiled_points, regular_slices)
    last_window_buckets = self._bucketize_last_window(tiled_points, last_window_slice)
    return regular_buckets, last_window_buckets

  def _bucket_accumulation_regular_windows_2d_parallel(
        self,
        regular_buckets: jnp.ndarray,
        all_points: jnp.ndarray,
    ) -> jnp.ndarray:
      """Accumulate points for all regular windows.

      Args:
          regular_buckets: Initial bucket values (coordinate_dim, window_dim, bucket_dim, precision_dim).
          all_points: Points for all windows (bucket_size_dim, coordinate_dim, window_dim, bucket_dim, precision_dim).

      Returns:
          Accumulated bucket values for all regular windows.
      """
      bucket_size_dim = all_points.shape[0]

      def scan_body(buckets, points):
        buckets = self._padd(buckets, points)
        return buckets, None

      buckets, _ = jax.lax.scan(scan_body, regular_buckets, all_points, length=bucket_size_dim)
      return buckets

  def _bucket_accumulation_last_window(
      self,
      buckets_in: jnp.ndarray,
      window_points: jnp.ndarray,
  ) -> jnp.ndarray:
    """Accumulate points for the last (special) window with duplication handling.

    Args:
        buckets_in: Initial bucket values (coordinate_dim, bucket_dim, precision_dim).
        window_points: Points with duplication (bucket_size_dim, coordinate_dim, bucket_dup_dim, bucket_dim, precision_dim).
        parameters: Computation parameters.

    Returns:
        Accumulated bucket values.
    """
    coordinate_dim, buckets_dim, precision_dim = buckets_in.shape
    bucket_size_dim, _, bucket_dup_dim, _, _ = window_points.shape

    # Reshape for processing
    window_points = window_points.reshape(bucket_size_dim, coordinate_dim, -1, precision_dim)
    base_dup_buckets = window_points[0]

    def scan_body(buckets, points):
      buckets = self._padd(buckets, points)
      return buckets, None

    dup_buckets, _ = jax.lax.scan(scan_body, base_dup_buckets, window_points[1:], length=bucket_size_dim - 1)

    # Reduce duplicated buckets using tree reduction
    log_bucket_dup_dim = int(math.log2(bucket_dup_dim))
    for _ in range(log_bucket_dup_dim):
      buckets_split = jnp.split(dup_buckets, 2, axis=1)
      dup_buckets = self._padd(buckets_split[0], buckets_split[1])

    # Add to input buckets
    buckets_in = self._padd(buckets_in, dup_buckets)
    return buckets_in

  def _bucket_accumulation_all_windows(
      self,
      regular_window_buckets: jnp.ndarray,
      last_window_buckets: jnp.ndarray,
      regular_window_points: jnp.ndarray,
      last_window_points: jnp.ndarray,
  ) -> jnp.ndarray:
    """Accumulate points for all windows with optimized regular window processing.

    Uses the opt variant for regular windows, which processes all windows
    simultaneously in a single scan instead of scanning window-by-window.

    Args:
        regular_buckets: Initial bucket values for regular windows
            (coord_dim, window_dim-1, bucket_dim, prec_dim).
        last_window_buckets: Initial bucket values for the last window
            (coord_dim, bucket_dim, prec_dim).
        regular_window_points: Points for regular windows
            (bucket_size, coord_dim, window_dim-1, bucket_num, prec_dim).
        last_window_points: Points for last window
            (bucket_size, coord_dim, dup_ratio, bucket_num, prec_dim).

    Returns:
        (regular_buckets, last_window_buckets): Accumulated bucket values,
            same shapes as the inputs.
    """
    _, _, last_window_bucket_dup, last_window_bucket_dim, _ = last_window_points.shape

    regular_window_buckets = self._bucket_accumulation_regular_windows_2d_parallel(regular_window_buckets, regular_window_points)

    last_point_buckets = last_window_buckets[:, :last_window_bucket_dim, :]
    last_blank_buckets = last_window_buckets[:, last_window_bucket_dim:, :]
    last_point_buckets = self._bucket_accumulation_last_window(last_point_buckets, last_window_points)

    last_window_buckets = jnp.concatenate((last_point_buckets, last_blank_buckets), axis=1)

    # Combine all buckets
    return regular_window_buckets, last_window_buckets
  
  def _bucketize_and_bucket_accumulation(self, 
                                        regular_window_slices: jnp.ndarray,
                                        last_window_slices: jnp.ndarray,
                                        tiled_points: jnp.ndarray,
                                        regular_window_buckets: jnp.ndarray,
                                        last_window_buckets: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    regular_window_points = self._bucketize_regular_windows(tiled_points, regular_window_slices)
    last_window_points = self._bucketize_last_window(tiled_points, last_window_slices)
    regular_window_buckets, last_window_buckets = self._bucket_accumulation_all_windows(regular_window_buckets, last_window_buckets, regular_window_points, last_window_points)

    return regular_window_buckets, last_window_buckets

  def _bucket_reduction(self,
      regular_window_buckets: jnp.ndarray, last_window_buckets: jnp.ndarray, window_sum: jnp.ndarray
  ) -> jnp.ndarray:
    """Reduce buckets to window sums using tree-based parallel algorithm.

    Computes S = sum_{i=0}^{n-1} i * B[i] per window via adjacent-pair
    tree reduction in O(log n) parallel steps.

    Tracks H = 2^k * bucket_sums (doubling each level) so that H[1::2]
    provides the correction weight directly. No separate bucket array needed.

    Args:
        all_buckets: Bucket values (window_dim, coordinate_dim, bucket_dim, precision_dim).
        window_sum: Initial window sum (coordinate_dim, bucket_dim, window_dim, precision_dim).

    Returns:
        Window sums (coordinate_dim, window_dim, precision_dim).
    """
    all_buckets = jnp.concatenate((regular_window_buckets, last_window_buckets[:, jnp.newaxis]), axis=1)
    bucket_num_in_window = all_buckets.shape[2]
    # (coord, window, bucket, prec) → (coord, bucket, window, prec)
    all_buckets = all_buckets.transpose(0, 2, 1, 3)
    iter_num = int(math.log2(bucket_num_in_window))

    cd, m, wd, pd = all_buckets.shape
    for _ in range(iter_num):
      m = all_buckets.shape[1]
      half = m // 2

      all_buckets = all_buckets.reshape(cd, half, 2, wd, pd)
      window_sum = window_sum.reshape(cd, half, 2, wd, pd)

      all_buckets_left, all_buckets_right = all_buckets[:, :, 0], all_buckets[:, :, 1]
      window_sum_left, window_sum_right = window_sum[:, :, 0], window_sum[:, :, 1]

      window_sum = self._padd(self._padd(window_sum_left, window_sum_right), all_buckets_right)
      bucket_sum = self._padd(all_buckets_left, all_buckets_right)
      all_buckets = self._padd(bucket_sum, bucket_sum)

    return window_sum[:, 0]

  def _window_merge(self, window_sum: jnp.ndarray) -> jnp.ndarray:
    """Merge window results into final MSM result using scan algorithm.

    Implements the window merging phase of Pippenger's algorithm.

    Args:
        window_sum: Window sums (coordinate_dim, window_dim, precision_dim).
        slice_length: Bit width of each window.
        parameters: Computation parameters.

    Returns:
        Final MSM result (coordinate_dim, precision_dim).
    """
    coordinate_dim, window_dim, precision_dim = window_sum.shape
    window_sum = window_sum.transpose(1, 0, 2).reshape((window_dim, coordinate_dim, 1, precision_dim))
    result = window_sum[window_dim - 1]

    def fori_loop_body(i, result):
      result = self._padd(result, result)
      return result

    def scan_body(result, window_sum):
      result = jax.lax.fori_loop(0, self.slice_bits, fori_loop_body, result)
      result = self._padd(result, window_sum)
      return result, None

    result, _ = jax.lax.scan(scan_body, result, window_sum[: window_dim - 1], reverse=True, length=window_dim - 1)
    result = result.reshape((coordinate_dim, precision_dim))
    return result

  def _bba_and_bws(
      self,
      regular_window_slices: jnp.ndarray,
      last_window_slices: jnp.ndarray,
      tiled_points: jnp.ndarray,
      regular_window_buckets: jnp.ndarray,
      last_window_buckets: jnp.ndarray,
      window_sum: jnp.ndarray,
  ) -> jnp.ndarray:
    """Fused kernel: run the last-tile BBA and then BWS in a single jit.

    Saves a kernel launch + the BBA->BWS round-trip via the host/HBM: XLA
    sees the accumulated buckets immediately feed into bucket reduction and
    can overlap the window-axis -> bucket-axis reshard with the reduction
    tree.
    """
    regular_window_buckets, last_window_buckets = self._bucketize_and_bucket_accumulation(
        regular_window_slices, last_window_slices, tiled_points,
        regular_window_buckets, last_window_buckets,
    )
    # Sharding conversion at the BBA -> BWS boundary.
    #   BBA output: regular_window_buckets on axis[1] (window),
    #               last_window_buckets on axis[1] (bucket).
    #   BWS input : regular_window_buckets on axis[2] (bucket),
    #               last_window_buckets on axis[1] (bucket, unchanged).
    # The last_window_buckets layout already matches BWS; only the
    # regular buckets need a window-axis -> bucket-axis reshard.
    if self.use_sharding:
      P = jax.sharding.PartitionSpec
      regular_window_buckets = self.shard_constraint(
          regular_window_buckets, P(None, None, self.mesh_axes, None))
    return self._bucket_and_window_sum(regular_window_buckets, last_window_buckets, window_sum)

  def _bucket_and_window_sum(self, regular_window_buckets: jnp.ndarray, last_window_buckets: jnp.ndarray, window_sum: jnp.ndarray
  ) -> jnp.ndarray:
    bucket_summations = self._bucket_reduction(regular_window_buckets, last_window_buckets, window_sum)
    # bucket_summations: (C, W, M). Window merge has no natural batch axis
    # to shard along, so force replication here. Without this constraint,
    # XLA may propagate BR's bucket-axis sharding forward and emit awkward
    # collectives inside the window-merge scan.
    if self.use_sharding:
      P = jax.sharding.PartitionSpec
      bucket_summations = self.shard_constraint(bucket_summations, P(None, None, None))
    window_summations = self._window_merge(bucket_summations)
    return window_summations

  # ----- Sharding setup -----

  def _init_shardings(self):
    """Pre-compute named shardings and padded shapes for BBA/BWS kernels.

    Sharding plan (mirrors scratch_profile_msm_sharding.py references):

    BBA (bucketize + bucket accumulation) — analogous to DA fused:
      regular_slices        (W-1, tile_length)   -> axis[0]  (window)
      last_window_slice     (tile_length,)        -> replicated
      tiled_points          (tile_length, C, M)   -> replicated
      regular_window_buckets (C, W-1, B, M)       -> axis[1]  (window)
      last_window_buckets    (C, B, M)            -> axis[1]  (bucket)

    BWS (bucket reduction + window merge) — analogous to BR_new_opt + WM:
      regular_window_buckets (C, W-1, B, M)       -> axis[2]  (bucket)
      last_window_buckets    (C, B, M)            -> axis[1]  (bucket)
      window_sum             (C, B, W, M)         -> axis[1]  (bucket)

    Note: regular_window_buckets is bucket-axis sharded at BWS entry but
    window-axis sharded during BBA. XLA inserts the reshard at the BWS
    boundary.

    Any axis whose extent is not divisible by the mesh size triggers a
    warning (from create_named_sharding) and is padded up.
    """
    C = self.coordinate_dim
    B = self.bucket_num_per_window
    M = self.moduli_num
    W = self.window_num
    T = self.tile_length

    # Shapes
    reg_slices_shape = (W - 1, T)
    last_slice_shape = (T,)
    points_shape = (T, C, M)
    reg_buckets_shape = (C, W - 1, B, M)
    last_buckets_shape = (C, B, M)
    window_sum_shape = (C, B, W, M)

    def _check(name, shape, padded):
      if tuple(shape) != tuple(padded):
        warnings.warn(
            f"[FusionMSMContext sharding] '{name}' shape {shape} not "
            f"divisible by mesh; padded to {padded}.",
            stacklevel=2,
        )

    # ----- BBA kernel shardings -----
    self.bba_reg_slices_sharding, self.bba_reg_slices_padded = (
        self.create_named_sharding(shape=reg_slices_shape, axes=[0]))
    _check("bba_regular_slices", reg_slices_shape, self.bba_reg_slices_padded)

    self.bba_last_slice_sharding, self.bba_last_slice_padded = (
        self.create_named_sharding(shape=last_slice_shape, axes=[]))
    _check("bba_last_window_slice", last_slice_shape, self.bba_last_slice_padded)

    self.bba_points_sharding, self.bba_points_padded = (
        self.create_named_sharding(shape=points_shape, axes=[]))
    _check("bba_tiled_points", points_shape, self.bba_points_padded)

    self.bba_reg_buckets_sharding, self.bba_reg_buckets_padded = (
        self.create_named_sharding(shape=reg_buckets_shape, axes=[1]))
    _check("bba_regular_window_buckets", reg_buckets_shape, self.bba_reg_buckets_padded)

    self.bba_last_buckets_sharding, self.bba_last_buckets_padded = (
        self.create_named_sharding(shape=last_buckets_shape, axes=[1]))
    _check("bba_last_window_buckets", last_buckets_shape, self.bba_last_buckets_padded)

    # ----- BWS kernel shardings -----
    self.bws_reg_buckets_sharding, self.bws_reg_buckets_padded = (
        self.create_named_sharding(shape=reg_buckets_shape, axes=[2]))
    _check("bws_regular_window_buckets", reg_buckets_shape, self.bws_reg_buckets_padded)

    self.bws_last_buckets_sharding, self.bws_last_buckets_padded = (
        self.create_named_sharding(shape=last_buckets_shape, axes=[1]))
    _check("bws_last_window_buckets", last_buckets_shape, self.bws_last_buckets_padded)

    self.bws_window_sum_sharding, self.bws_window_sum_padded = (
        self.create_named_sharding(shape=window_sum_shape, axes=[1]))
    _check("bws_window_sum", window_sum_shape, self.bws_window_sum_padded)

    # ----- Whole-MSM fused kernel shardings -----
    # Slices get a tile axis prepended, so the regular window axis shifts
    # from axis[0] to axis[1]. Last slices + points are replicated.
    fused_reg_slices_shape = (self.tile_num, W - 1, T)
    fused_last_slices_shape = (self.tile_num, T)
    fused_tiled_points_shape = (self.tile_num, T, C, M)
    self.fused_reg_slices_sharding, self.fused_reg_slices_padded = (
        self.create_named_sharding(shape=fused_reg_slices_shape, axes=[1]))
    _check("fused_regular_tiled_slices", fused_reg_slices_shape, self.fused_reg_slices_padded)
    self.fused_last_slices_sharding, self.fused_last_slices_padded = (
        self.create_named_sharding(shape=fused_last_slices_shape, axes=[]))
    _check("fused_last_tiled_slices", fused_last_slices_shape, self.fused_last_slices_padded)
    self.fused_tiled_points_sharding, self.fused_tiled_points_padded = (
        self.create_named_sharding(shape=fused_tiled_points_shape, axes=[]))
    _check("fused_tiled_points", fused_tiled_points_shape, self.fused_tiled_points_padded)

  def set_use_sharding(self, use_sharding: bool):
    super().set_use_sharding(use_sharding)
    if use_sharding:
      self._init_shardings()
      self._place_sharded_state()

  def _place_sharded_state(self):
    """Pad + place persistent state (accumulators, points) onto the correct
    shardings so compiled kernels receive inputs whose layout matches what
    they were compiled for. Must be called after _init_shardings().
    """
    from utils import pad_jax_array
    # Accumulators — BBA input layouts.
    self.regular_window_buckets = pad_jax_array(
        jnp.asarray(self.regular_window_buckets), self.bba_reg_buckets_padded
    ).to_device(self.bba_reg_buckets_sharding)
    self.last_window_buckets = pad_jax_array(
        jnp.asarray(self.last_window_buckets), self.bba_last_buckets_padded
    ).to_device(self.bba_last_buckets_sharding)
    # window_sum — BWS input layout.
    self.window_sum = pad_jax_array(
        jnp.asarray(self.window_sum), self.bws_window_sum_padded
    ).to_device(self.bws_window_sum_sharding)
    # Points — replicated on the mesh. Use the 4D fused sharding so that
    # the runtime placement matches what the compiled fused kernel was
    # lowered with (both paths see the same PartitionSpec).
    self.points = jax.device_put(jnp.asarray(self.points), self.fused_tiled_points_sharding)

  # ----- Compile system (mirrors TPUDistributionMSMContext) -----

  def _get_bba_shape_dtype_structs(self):
    """ShapeDtypeStructs for _bucketize_and_bucket_accumulation inputs.

    When sharding is enabled, inputs carry the shardings/padded shapes
    established in _init_shardings().
    """
    if self.use_sharding:
      return [
          jax.ShapeDtypeStruct(self.bba_reg_slices_padded, jnp.int32, sharding=self.bba_reg_slices_sharding),
          jax.ShapeDtypeStruct(self.bba_last_slice_padded, jnp.int32, sharding=self.bba_last_slice_sharding),
          jax.ShapeDtypeStruct(self.bba_points_padded, jnp.uint32, sharding=self.bba_points_sharding),
          jax.ShapeDtypeStruct(self.bba_reg_buckets_padded, jnp.uint32, sharding=self.bba_reg_buckets_sharding),
          jax.ShapeDtypeStruct(self.bba_last_buckets_padded, jnp.uint32, sharding=self.bba_last_buckets_sharding),
      ]
    regular_slices_shape = (self.window_num - 1, self.tile_length)
    last_window_slice_shape = (self.tile_length,)
    tiled_points_shape = (self.tile_length, self.coordinate_dim, self.moduli_num)
    regular_window_buckets_shape = (
        self.coordinate_dim, self.window_num - 1, self.bucket_num_per_window, self.moduli_num,
    )
    last_window_buckets_shape = (
        self.coordinate_dim, self.bucket_num_per_window, self.moduli_num,
    )
    return [
        jax.ShapeDtypeStruct(regular_slices_shape, jnp.int32),
        jax.ShapeDtypeStruct(last_window_slice_shape, jnp.int32),
        jax.ShapeDtypeStruct(tiled_points_shape, jnp.uint32),
        jax.ShapeDtypeStruct(regular_window_buckets_shape, jnp.uint32),
        jax.ShapeDtypeStruct(last_window_buckets_shape, jnp.uint32),
    ]

  def _get_bws_shape_dtype_structs(self):
    """ShapeDtypeStructs for _bucket_and_window_sum inputs."""
    if self.use_sharding:
      return [
          jax.ShapeDtypeStruct(self.bws_reg_buckets_padded, jnp.uint32, sharding=self.bws_reg_buckets_sharding),
          jax.ShapeDtypeStruct(self.bws_last_buckets_padded, jnp.uint32, sharding=self.bws_last_buckets_sharding),
          jax.ShapeDtypeStruct(self.bws_window_sum_padded, jnp.uint32, sharding=self.bws_window_sum_sharding),
      ]
    regular_window_buckets_shape = (
        self.coordinate_dim, self.window_num - 1, self.bucket_num_per_window, self.moduli_num,
    )
    last_window_buckets_shape = (
        self.coordinate_dim, self.bucket_num_per_window, self.moduli_num,
    )
    window_sum_shape = (
        self.coordinate_dim, self.bucket_num_per_window, self.batch_window_num, self.moduli_num,
    )
    return [
        jax.ShapeDtypeStruct(regular_window_buckets_shape, jnp.uint32),
        jax.ShapeDtypeStruct(last_window_buckets_shape, jnp.uint32),
        jax.ShapeDtypeStruct(window_sum_shape, jnp.uint32),
    ]

  def _get_bba_bws_shape_dtype_structs(self):
    """ShapeDtypeStructs for the fused _bba_and_bws inputs.

    Slices/points + BBA-layout accumulators + BWS-layout window_sum. The
    reshard from BBA's window-axis bucket layout to BWS's bucket-axis
    layout happens inside the kernel via with_sharding_constraint.
    """
    bba = self._get_bba_shape_dtype_structs()
    bws = self._get_bws_shape_dtype_structs()
    # 3 slice/point inputs + 2 BBA accumulators + BWS window_sum.
    return bba + [bws[2]]

  def compile(self, parameters: dict = None):
    """Compile the kernels needed for the chosen execution path.

    Args:
      parameters: Compile-time options. Recognised keys:
        - ``use_fused`` (bool, default False): when True, compile the
          whole-MSM fused kernel (``_multiscalar_multiply_fused``) and
          only the subkernels it uses internally (``_bba_and_bws`` for
          the last tile, plus ``_bucketize_and_bucket_accumulation``
          for the scan body when ``tile_num > 1``). When False, compile
          the per-stage path: BBA + BWS, and additionally
          ``_bba_and_bws`` for the last-tile fusion that
          ``_multiscalar_multiply`` uses. Also skips compiling the
          standalone BBA when ``tile_num == 1``.
    """
    parameters = parameters or {}
    use_fused = parameters.get("use_fused", False)
    self.use_fused = use_fused

    bba_structs = self._get_bba_shape_dtype_structs()
    bws_structs = self._get_bws_shape_dtype_structs()
    bba_bws_structs = self._get_bba_bws_shape_dtype_structs()

    bba_hash = hash_args(
        *(v for s in bba_structs for v in (s.shape, s.dtype.__str__()))
    )
    bws_hash = hash_args(
        *(v for s in bws_structs for v in (s.shape, s.dtype.__str__()))
    )
    bba_bws_hash = hash_args(
        *(v for s in bba_bws_structs for v in (s.shape, s.dtype.__str__()))
    )

    # Always compile the fused BBA+BWS kernel: it handles the last tile
    # in both execution paths, and is the single kernel used by
    # tile_num == 1 in the non-fused path.
    self.compiled_kernels.setdefault(bba_bws_hash, {})["bba_and_bws"] = (
        jax_jit_lower_compile(
            self._bba_and_bws,
            bba_bws_structs[0], bba_bws_structs[1], bba_bws_structs[2],
            bba_bws_structs[3], bba_bws_structs[4], bba_bws_structs[5],
        )
    )

    if use_fused:
      fused_structs = self._get_fused_shape_dtype_structs()
      fused_hash = hash_args(
          *(v for s in fused_structs for v in (s.shape, s.dtype.__str__()))
      )
      self.compiled_kernels.setdefault(fused_hash, {})["multiscalar_multiply_fused"] = (
          jax_jit_lower_compile(
              self._multiscalar_multiply_fused,
              *fused_structs,
          )
      )
      # The whole-MSM fused kernel absorbs BBA (via scan) and BWS
      # internally, so the standalone per-stage compiles aren't needed.
    else:
      # Non-fused path uses standalone BBA for tiles 0..N-2, and
      # _bba_and_bws for the last tile (already compiled above). When
      # tile_num == 1 there are no non-last tiles, so skip the
      # standalone BBA compile entirely.
      if self.tile_num > 1:
        self.compiled_kernels.setdefault(bba_hash, {})["bucketize_and_bucket_accumulation"] = (
            jax_jit_lower_compile(
                self._bucketize_and_bucket_accumulation,
                bba_structs[0], bba_structs[1], bba_structs[2],
                bba_structs[3], bba_structs[4],
            )
        )
      # BWS standalone is still useful if any external caller invokes
      # bucket_and_window_sum directly. Keep compiling it.
      self.compiled_kernels.setdefault(bws_hash, {})["bucket_and_window_sum"] = (
          jax_jit_lower_compile(
              self._bucket_and_window_sum,
              bws_structs[0], bws_structs[1], bws_structs[2],
          )
      )

    self.use_compiled_kernels = True

  def bba_and_bws(
      self, regular_slices, last_window_slice, tiled_points,
      regular_window_buckets, last_window_buckets, window_sum,
  ):
    if self.use_compiled_kernels:
      kernel_hash = hash_args(
          regular_slices.shape, regular_slices.dtype.__str__(),
          last_window_slice.shape, last_window_slice.dtype.__str__(),
          tiled_points.shape, tiled_points.dtype.__str__(),
          regular_window_buckets.shape, regular_window_buckets.dtype.__str__(),
          last_window_buckets.shape, last_window_buckets.dtype.__str__(),
          window_sum.shape, window_sum.dtype.__str__(),
      )
      return self.compiled_kernels[kernel_hash]["bba_and_bws"](
          regular_slices, last_window_slice, tiled_points,
          regular_window_buckets, last_window_buckets, window_sum,
      )
    return self._bba_and_bws(
        regular_slices, last_window_slice, tiled_points,
        regular_window_buckets, last_window_buckets, window_sum,
    )

  def bucketize_and_bucket_accumulation(
      self, regular_slices, last_window_slice, tiled_points,
      regular_window_buckets, last_window_buckets,
  ):
    if self.use_compiled_kernels:
      kernel_hash = hash_args(
          regular_slices.shape, regular_slices.dtype.__str__(),
          last_window_slice.shape, last_window_slice.dtype.__str__(),
          tiled_points.shape, tiled_points.dtype.__str__(),
          regular_window_buckets.shape, regular_window_buckets.dtype.__str__(),
          last_window_buckets.shape, last_window_buckets.dtype.__str__(),
      )
      return self.compiled_kernels[kernel_hash]["bucketize_and_bucket_accumulation"](
          regular_slices, last_window_slice, tiled_points,
          regular_window_buckets, last_window_buckets,
      )
    return self._bucketize_and_bucket_accumulation(
        regular_slices, last_window_slice, tiled_points,
        regular_window_buckets, last_window_buckets,
    )

  def bucket_and_window_sum(self, regular_window_buckets, last_window_buckets, window_sum):
    if self.use_compiled_kernels:
      kernel_hash = hash_args(
          regular_window_buckets.shape, regular_window_buckets.dtype.__str__(),
          last_window_buckets.shape, last_window_buckets.dtype.__str__(),
          window_sum.shape, window_sum.dtype.__str__(),
      )
      return self.compiled_kernels[kernel_hash]["bucket_and_window_sum"](
          regular_window_buckets, last_window_buckets, window_sum,
      )
    return self._bucket_and_window_sum(regular_window_buckets, last_window_buckets, window_sum)

  def context_hash(self) -> str:
    return hash_args(
        self.__class__.__name__,
        self.ec_ctx.context_hash() if hasattr(self.ec_ctx, 'context_hash') else str(self.ec_ctx.__class__.__name__),
        self.slice_bits,
        self.scalar_bits,
        self.msm_length,
        self.tile_length,
        self.bucket_num_per_window,
        self.bucket_num_last_window,
        self.use_sharding,
    )

  def _multiscalar_multiply(self, tiled_slices: jnp.ndarray):
    # Padding (if any) was applied in to_computational_format. Here we only
    # place the per-tile slices onto the right sharding — points are already
    # placed on the replicated BBA points sharding via _place_sharded_state.
    last_tile = self.tile_num - 1
    for tile_index in range(self.tile_num):
      tiled_slices_tile = tiled_slices[tile_index]
      regular_slices_tile = tiled_slices_tile[: self.window_num - 1]
      last_window_slice_tile = tiled_slices_tile[self.window_num - 1]
      tiled_points_tile = self.points[tile_index]

      if self.use_sharding:
        regular_slices_tile = regular_slices_tile.to_device(self.bba_reg_slices_sharding)
        last_window_slice_tile = last_window_slice_tile.to_device(self.bba_last_slice_sharding)

      if tile_index == last_tile:
        # Fused BBA + BWS: reshard (BBA -> BWS) stays inside one compiled
        # kernel, so XLA can overlap it with the reduction tree.
        return self.bba_and_bws(
            regular_slices_tile, last_window_slice_tile, tiled_points_tile,
            self.regular_window_buckets, self.last_window_buckets, self.window_sum,
        )

      self.regular_window_buckets, self.last_window_buckets = self.bucketize_and_bucket_accumulation(
          regular_slices_tile, last_window_slice_tile, tiled_points_tile,
          self.regular_window_buckets, self.last_window_buckets,
      )

  def _multiscalar_multiply_fused(
      self,
      regular_tiled_slices: jnp.ndarray,
      last_tiled_slices: jnp.ndarray,
      tiled_points: jnp.ndarray,
      regular_window_buckets: jnp.ndarray,
      last_window_buckets: jnp.ndarray,
      window_sum: jnp.ndarray,
  ) -> jnp.ndarray:
    """Whole-MSM fused kernel — one jit across all tiles + BWS.

    Regular and last-window slices are passed as separate tensors because
    they require different input shardings: regular is sharded along its
    window axis, while the last-window slice has no batch-like axis to
    shard over and is replicated. Packing them together would force a
    single sharding on the combined tensor, losing this distinction.

    Args:
      regular_tiled_slices:   (tile_num, W-1, tile_length)   — shard axis[1]
      last_tiled_slices:      (tile_num, tile_length)         — replicated
      tiled_points:           (tile_num, tile_length, C, M)   — replicated
      regular_window_buckets: (C, W-1, B, M)                  — BBA axis[1]
      last_window_buckets:    (C, B, M)                       — BBA axis[1]
      window_sum:             (C, B, W, M)                    — BWS axis[1]

    Hybrid structure:
      - Tiles ``0..N-2`` run through ``jax.lax.scan``.
      - The last tile goes through ``_bba_and_bws`` (reshard + BR->WM
        constraint live inside that kernel).
      - When ``tile_num == 1``, the scan is skipped.
    """
    def scan_body(carry, inputs):
      reg_buckets, last_buckets = carry
      regular_slices, last_window_slice, points_tile = inputs
      reg_buckets, last_buckets = self._bucketize_and_bucket_accumulation(
          regular_slices, last_window_slice, points_tile,
          reg_buckets, last_buckets,
      )
      return (reg_buckets, last_buckets), None

    if self.tile_num > 1:
      (regular_window_buckets, last_window_buckets), _ = jax.lax.scan(
          scan_body,
          (regular_window_buckets, last_window_buckets),
          (regular_tiled_slices[:-1], last_tiled_slices[:-1], tiled_points[:-1]),
          length=self.tile_num - 1,
      )

    return self._bba_and_bws(
        regular_tiled_slices[-1],
        last_tiled_slices[-1],
        tiled_points[-1],
        regular_window_buckets,
        last_window_buckets,
        window_sum,
    )

  def _get_fused_shape_dtype_structs(self):
    """ShapeDtypeStructs for the whole-MSM fused kernel.

    Regular / last slices are produced as separate structs so each can
    carry its own sharding (regular: window-axis sharded; last:
    replicated).
    """
    bba = self._get_bba_shape_dtype_structs()
    bws = self._get_bws_shape_dtype_structs()
    per_tile_length = bba[0].shape[1]

    regular_tiled_slices_shape = (self.tile_num, self.window_num - 1, per_tile_length)
    last_tiled_slices_shape = (self.tile_num, per_tile_length)
    tiled_points_shape = (self.tile_num,) + tuple(bba[2].shape)

    if self.use_sharding:
      # Use the shardings/padded shapes pre-computed in _init_shardings so
      # the compile-time and run-time placements agree.
      return [
          jax.ShapeDtypeStruct(self.fused_reg_slices_padded, jnp.int32, sharding=self.fused_reg_slices_sharding),
          jax.ShapeDtypeStruct(self.fused_last_slices_padded, jnp.int32, sharding=self.fused_last_slices_sharding),
          jax.ShapeDtypeStruct(self.fused_tiled_points_padded, jnp.uint32, sharding=self.fused_tiled_points_sharding),
          bba[3], bba[4], bws[2],
      ]
    return [
        jax.ShapeDtypeStruct(regular_tiled_slices_shape, jnp.int32),
        jax.ShapeDtypeStruct(last_tiled_slices_shape, jnp.int32),
        jax.ShapeDtypeStruct(tiled_points_shape, jnp.uint32),
        bba[3], bba[4], bws[2],
    ]

  def multiscalar_multiply(self, tiled_slices: jnp.ndarray):
    if not self.use_fused:
      return self._multiscalar_multiply(tiled_slices)

    # Fused path: split regular / last slices so they can carry independent
    # shardings. self.points is already placed in per-tile layout.
    regular_tiled_slices = tiled_slices[:, : self.window_num - 1]
    last_tiled_slices = tiled_slices[:, self.window_num - 1]

    if self.use_sharding:
      regular_tiled_slices = regular_tiled_slices.to_device(self.fused_reg_slices_sharding)
      last_tiled_slices = last_tiled_slices.to_device(self.fused_last_slices_sharding)

    args = (
        regular_tiled_slices, last_tiled_slices, self.points,
        self.regular_window_buckets, self.last_window_buckets, self.window_sum,
    )
    if self.use_compiled_kernels:
      kernel_hash = hash_args(
          *(v for a in args for v in (a.shape, a.dtype.__str__()))
      )
      return self.compiled_kernels[kernel_hash]["multiscalar_multiply_fused"](*args)
    return self._multiscalar_multiply_fused(*args)


# =============================================================================
# Fq2 MSM (Pippenger over an Fq2 elliptic curve)
# =============================================================================


class Fq2PippengerMSMContext(MultiscalarMultiplicationContextBase, JaxKernelContextBase):
  """Pippenger multi-scalar multiplication over an Fq2 elliptic curve.

  Operates natively on the ``ExtendedTwistedEdwardsFq2NDContext`` layout
  ``(coord=4, batch, 2, moduli)``: every accumulator (``zero``, bucket,
  window sum, running merge) keeps the ``2`` Fp2-component axis intact,
  and all elliptic-curve arithmetic flows through ``ec_ctx.point_add``.
  Because that ``point_add`` is itself expressed entirely in Fq2Context
  primitives (which in turn route through the underlying DRNS / lazy Fq
  context), this MSM doesn't depend on any specific Fq encoding pipeline
  or on the C-kernel distribution path used by the Fp1 MSM contexts.

  The implementation favours clarity over raw throughput:

    * Window slicing is the standard Pippenger decomposition
      (``window_num = ceil(scalar_bits / slice_bits)`` windows of
      ``2**slice_bits`` buckets each).
    * Bucket distribution is a Python-level scatter into
      ``bucket_sums[window][slot]``: each non-zero slot gets a JAX
      ``point_add`` against the running bucket value, so the heavy
      EC arithmetic still runs on TPU even though the dispatch is
      driven from the host.
    * Bucket reduction uses the standard "running-prefix" trick
      ``out_w = Σ_{s=1}^{B-1} s · bucket[s]`` evaluated as
      ``acc, out``-style accumulators.
    * Window merge is the canonical
      ``result = ((((W_{n-1} · 2^k) + W_{n-2}) · 2^k) + … ) + W_0``.

  Use small ``slice_bits`` (8 or below) for correctness tests — the
  bucket axis grows as ``2**slice_bits`` and the per-window reduction
  walks every slot.

  Required parameters:

    * ``elliptic_curve_context_class``  — should be
      ``ExtendedTwistedEdwardsFq2NDContext``.
    * ``elliptic_curve_parameters``     — its construction kwargs.
    * ``slice_bits``, ``scalar_bits``   — Pippenger window config.
    * ``order``                         — curve scalar-field order
      (only used for sanity checks / parameter hashing).
  """

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    JaxKernelContextBase.__init__(self)

    self.slice_bits = parameters.get("slice_bits", 8)
    self.scalar_bits = parameters.get("scalar_bits")
    assert self.scalar_bits is not None, "scalar_bits must be provided"
    self.order = parameters.get("order")
    self.window_num = int(math.ceil(self.scalar_bits / self.slice_bits))
    self.bucket_num_per_window = 1 << self.slice_bits
    self.coordinate_dim = 4

    # Field plumbing.  We need both the field-extension context (Fq2) and
    # the underlying Fq context for moduli sizing.  Accepts:
    #   * ExtendedTwistedEdwardsFq2NDContext  — unified add formula, used
    #     by curves with rational 2-torsion in Fp2 (e.g. G1-style).
    #   * XYZZWeierstrassFq2NDContext         — short-Weierstrass XYZZ,
    #     non-unified.  Driver routes self-add through ``point_double``
    #     and tracks pristine buckets to avoid identity-add.
    #   * ProjectiveCompleteFq2NDContext      — RCB-2015 complete add,
    #     identity-safe under all input pairs.  Pristine bookkeeping is
    #     redundant but harmless; the driver doesn't need to special-case
    #     identity or self-add.
    if not isinstance(self.ec_ctx,
                      (ExtendedTwistedEdwardsFq2NDContext,
                       XYZZWeierstrassFq2NDContext,
                       ProjectiveCompleteFq2NDContext)):
      raise TypeError(
          "Fq2PippengerMSMContext expects an Fq2 EC backend; "
          f"got {type(self.ec_ctx).__name__}"
      )
    # Only XYZZ-Weierstrass needs the explicit ``point_double`` call for
    # self-add.  ETE and RCB-Projective both have unified ``point_add``
    # that handles ``P + P`` correctly.
    self._needs_xyzz_doubling = isinstance(
        self.ec_ctx, XYZZWeierstrassFq2NDContext)
    # Coordinate axis cardinality (4 for XYZZ / ETE, 3 for projective RCB).
    self.coordinate_dim = (
        3 if isinstance(self.ec_ctx, ProjectiveCompleteFq2NDContext) else 4
    )

    self.fe_ctx: Fq2Context = self.ec_ctx.get_field_extension_context()
    self.ff_ctx: FiniteFieldContextBase = self.ec_ctx.get_finite_field_context()
    self.moduli_num = self.ff_ctx.get_moduli_num()

    # Identity / "zero" point on the curve, in the EC context's native
    # 4-coord form (Fp2 inner).  Stored both as a length-4 list of Fp2
    # tuples (for CPU oracles / tests) and in JAX computational format
    # with a leading single-batch axis so it can feed straight into
    # ``ec_ctx.point_add`` (when the context's add is identity-safe).
    zero_te = self.ec_ctx.zero_point
    zero_te_cf = self.fe_ctx.to_computational_format(zero_te)  # (4, 2, M)
    self._zero_pt = jnp.expand_dims(zero_te_cf, axis=1)  # (4, 1, 2, M)

  # --------------------------------------------------------------------- #
  #  Format conversion (delegates to the EC context)                       #
  # --------------------------------------------------------------------- #

  def to_computational_format(self, points: list) -> jnp.ndarray:
    """Convert a list of Fp2 affine points to ``(4, N, 2, M)`` Fq2 EC form."""
    return self.ec_ctx.to_computational_format(points)

  def to_original_format(self, point_cf: jnp.ndarray):
    """Inverse of ``to_computational_format``."""
    return self.ec_ctx.to_original_format(point_cf)

  def reset(self):
    """No-op for API parity with :class:`FusionMSMContext`.  The Fq2
    Pippenger driver doesn't hold mutable between-call state — its
    bucket arrays are local to ``multiscalar_multiply``."""
    return

  # --------------------------------------------------------------------- #
  #  Internals                                                             #
  # --------------------------------------------------------------------- #

  def _slice_scalars(self, scalars: list) -> list:
    """Standard Pippenger slicing — returns a length-W list of length-N
    Python int slice tables, one per window."""
    return utils.slice_scalars(scalars, self.scalar_bits, self.slice_bits)

  def _zero_buckets(self) -> list:
    """A length-B list of identity points sharing the JAX zero point."""
    return [self._zero_pt] * self.bucket_num_per_window

  # --------------------------------------------------------------------- #
  #  Public MSM                                                            #
  # --------------------------------------------------------------------- #

  def multiscalar_multiply(
      self,
      points_cf: jnp.ndarray,
      scalars: list,
  ) -> jnp.ndarray:
    """Compute ``Σ_i scalars[i] · points_cf[:, i]`` via Pippenger.

    Args:
      points_cf: Fq2 EC computational-format point batch with shape
        ``(4, N, 2, M)``.  Use :meth:`to_computational_format` to obtain
        from a list of Fp2 affine points.
      scalars: length-``N`` Python list of ``int`` scalars (each treated
        modulo ``2**scalar_bits``; values up to the curve order are fine).

    Returns:
      Single Fq2 EC point in computational format with shape
      ``(4, 1, 2, M)``.
    """
    if points_cf.ndim != 4 or points_cf.shape[0] != self.coordinate_dim:
      raise ValueError(
          f"points_cf must have shape ({self.coordinate_dim}, N, 2, M); "
          f"got {points_cf.shape}"
      )
    n = points_cf.shape[1]
    if len(scalars) != n:
      raise ValueError(
          f"scalars/points length mismatch: {len(scalars)} vs {n}"
      )

    sliced = self._slice_scalars(scalars)  # list[W][N] of int
    W = self.window_num
    B = self.bucket_num_per_window

    # Self-add primitive.  XYZZ add formulas degenerate when P == Q, so
    # we route window doublings through ``point_double``.  ETE's add is
    # unified, so for it ``point_add(P, P)`` is exactly 2P.
    def _double(p):
      if self._needs_xyzz_doubling:
        return self.ec_ctx.point_double(p)
      return self.ec_ctx.point_add(p, p)

    # ----- Bucket distribution -----
    # bucket_sums[w][s] = Σ {points_cf[:, i] : sliced[w][i] == s}.
    # We track ``bucket_pristine[w][s]`` so the very first point that
    # lands in a bucket is *copied* rather than added to the identity —
    # XYZZ add doesn't handle the identity, and even for ETE this
    # saves one (point_add, identity, P) per bucket.
    bucket_sums = [self._zero_buckets() for _ in range(W)]
    bucket_pristine = [[True] * B for _ in range(W)]
    for i in range(n):
      pt_i = jax.lax.dynamic_slice_in_dim(points_cf, i, 1, axis=1)  # (4, 1, 2, M)
      window_slices_i = [int(sliced[w][i]) for w in range(W)]
      for w, s in enumerate(window_slices_i):
        if s == 0:
          continue
        if bucket_pristine[w][s]:
          bucket_sums[w][s] = pt_i
          bucket_pristine[w][s] = False
        else:
          bucket_sums[w][s] = self.ec_ctx.point_add(bucket_sums[w][s], pt_i)

    # ----- Bucket reduction (per window) -----
    # Compute  W_w = Σ_{s=1}^{B-1} s · bucket[s]  via the running-prefix
    # identity:  acc_s = bucket[s] + bucket[s+1] + ... ; out_s = acc_s + acc_{s+1} + ...
    # so out_1 = Σ s · bucket[s].
    #
    # ``acc_pristine`` / ``out_pristine`` track whether the running
    # accumulator still equals the identity, so we can fold the first
    # contributing bucket into it without calling identity-add.
    window_sums = []
    window_sums_pristine = []
    for w in range(W):
      acc = self._zero_pt
      out = self._zero_pt
      acc_pristine = True
      out_pristine = True
      buckets_w = bucket_sums[w]
      pristine_w = bucket_pristine[w]
      for s in range(B - 1, 0, -1):
        # acc += bucket[s]
        if not pristine_w[s]:
          if acc_pristine:
            acc = buckets_w[s]
            acc_pristine = False
          else:
            acc = self.ec_ctx.point_add(acc, buckets_w[s])
        # out += acc
        if not acc_pristine:
          if out_pristine:
            out = acc
            out_pristine = False
          else:
            # When the previous iteration promoted out from pristine via
            # ``out = acc`` and the current iteration did not touch acc,
            # out and acc are the same JAX object → XYZZ add degenerates,
            # so dispatch to point_double instead.  Harmless for ETE
            # (point_add(P, P) == 2P with unified formulas) but still
            # cheaper than a full add.
            if out is acc:
              out = _double(out)
            else:
              out = self.ec_ctx.point_add(out, acc)
      window_sums.append(out)
      window_sums_pristine.append(out_pristine)

    # ----- Window merge -----
    # result = (((W_{n-1} << k) + W_{n-2}) << k) + … + W_0
    # where ``<< k`` is k repeated EC doublings.
    #
    # We seed ``result`` lazily from the first non-pristine window sum
    # to keep identity out of the EC kernel; subsequent windows fold
    # in via point_add.  Doubling identity is a no-op, so we just skip
    # the doubling chain while result is still pristine.
    result = self._zero_pt
    result_pristine = True
    for w in range(W - 1, -1, -1):
      if not result_pristine:
        for _ in range(self.slice_bits):
          result = _double(result)
      if window_sums_pristine[w]:
        continue
      if result_pristine:
        result = window_sums[w]
        result_pristine = False
      else:
        result = self.ec_ctx.point_add(result, window_sums[w])

    return result

  # --------------------------------------------------------------------- #
  #  CPU oracle — pure-python Fp2 reference for test cross-checking        #
  # --------------------------------------------------------------------- #

  def cpu_oracle_msm(self, affine_points: list, scalars: list) -> list:
    """Reference MSM run entirely in pure-python Fp2 arithmetic.

    Args:
      affine_points: length-N list of Fp2 affine ``[(x_c0, x_c1), (y_c0, y_c1)]``
        Weierstrass points.
      scalars: matching length-N list of ``int`` scalars.

    Returns:
      Single ``[(x_c0, x_c1), (y_c0, y_c1)]`` Weierstrass-affine point on
      the curve, equal (mod ``prime``) to ``Σ k_i · P_i``.
    """
    ec = self.ec_ctx
    # Push every input through the same affine→internal pipeline used
    # by the JAX path so we compare apples to apples.  ETE and XYZZ
    # expose different conversion helpers.
    if isinstance(ec, XYZZWeierstrassFq2NDContext):
      points_te = [ec._convert_to_xyzz(p) for p in affine_points]
    else:
      points_te = [ec._convert_to_extended_twisted_edwards(p) for p in affine_points]
    zero_te = list(ec.zero_point)

    def double_and_add(point_te, k):
      """k * P via double-and-add over Fp2 ETE.  Returns identity for k == 0."""
      if k == 0:
        return list(zero_te)
      result = list(zero_te)
      addend = list(point_te)
      while k > 0:
        if k & 1:
          result = ec._point_add(result, addend)
        addend = ec._point_add(addend, addend)
        k >>= 1
      return result

    acc = list(zero_te)
    for pt, k in zip(points_te, scalars):
      acc = ec._point_add(acc, double_and_add(pt, int(k)))

    return ec._convert_to_weierstrass_affine(acc)

  # --------------------------------------------------------------------- #
  #  Hashing / serialisation                                               #
  # --------------------------------------------------------------------- #

  def context_hash(self) -> str:
    return hash_args(
        self.__class__.__name__,
        self.ec_ctx.context_hash() if hasattr(self.ec_ctx, "context_hash")
        else self.ec_ctx.__class__.__name__,
        self.slice_bits,
        self.scalar_bits,
        self.order,
        self.use_sharding,
    )


# =============================================================================
# Fq2 G2 MSM — same fused architecture as FusionMSMContext, Fp2 axis inserted
# =============================================================================


class Fq2FusionMSMContext(MultiscalarMultiplicationContextBase, JaxKernelContextBase):
  """G2 version of :class:`FusionMSMContext`.

  Mirrors the fused per-stage AOT-compiled driver structure of
  :class:`FusionMSMContext` exactly — same Pippenger algorithm, same
  ``_bba_and_bws`` / ``_bucketize_and_bucket_accumulation`` /
  ``_bucket_and_window_sum`` kernels, same ``compile({use_fused})`` API.

  The only difference vs the G1 driver is that every *point* tensor
  carries an additional Fp2-component axis (size 2) inserted directly
  before the moduli axis:

      G1 point shape:  ``(*, coord, *, M)``
      G2 point shape:  ``(*, coord, *, 2, M)``

  All EC ops route through ``self.ec_ctx.point_add`` / ``point_double``
  which accept arbitrary leading batch dims and operate on the Fp2 axis
  via the underlying ``Fq2Context._modular_*`` ops, so no algorithm
  change is needed.

  Requires an identity-safe EC context (e.g.
  :class:`ProjectiveCompleteFq2NDContext`): the driver initialises
  buckets to the EC ctx's ``zero_point`` and freely calls ``point_add``
  without pristine bookkeeping.
  """

  # --------------------------------------------------------------------- #
  #  Init                                                                  #
  # --------------------------------------------------------------------- #

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    MultiscalarMultiplicationContextBase.__init__(self, parameters)
    self._init_config_parameters()
    self._init_jax_data()
    self._init_point_parameters()
    JaxKernelContextBase.__init__(self)
    self.use_fused = False
    # Lazy jit wrapper for ``_bba_and_bws`` — compiled on first call,
    # cached by JAX shape signature thereafter.  Avoids per-call eager
    # dispatch of the ~300 inner point_adds.  We don't AOT compile here
    # because the trace is heavy (RCB + DRNS lazy carry while_loops) and
    # ``jax.jit`` defers compile to first use; users who want init-time
    # compile should call ``self.compile(...)`` explicitly.
    self._jit_bba_and_bws = jax.jit(self._bba_and_bws)

  def _init_config_parameters(self):
    # coordinate_dim = 3 for projective RCB; pulled from the EC ctx.
    self.coordinate_dim = self.parameters.get(
        "coordinate_dim",
        getattr(self.ec_ctx, "coordinate_dim", None),
    )
    if self.coordinate_dim is None:
      # ProjectiveCompleteFq2NDContext stores coord count in zero_point's
      # length (3 for projective). Fall back to that.
      self.coordinate_dim = len(self.ec_ctx.zero_point)
    self.msm_length  = self.parameters.get("msm_length")
    self.tile_length = self.parameters.get("tile_length")
    assert (self.msm_length % self.tile_length == 0
            and self.msm_length >= self.tile_length), \
        "msm_length must be divisible by tile_length and ≥ tile_length"

    self.tile_num   = self.msm_length // self.tile_length
    self.slice_bits = self.parameters.get("slice_bits")
    self.scalar_bits = self.parameters.get("scalar_bits")
    self.order      = self.parameters.get("order")
    self.window_num = int(math.ceil(self.scalar_bits / self.slice_bits))
    self.batch_window_num = self.window_num
    self.bucket_num_per_window = 2 ** self.slice_bits  # incl. bucket 0

    orig_bucket_num_last_window = (
        self.order >> ((self.window_num - 1) * self.slice_bits)) + 1
    added_padding = (8 - orig_bucket_num_last_window % 8) % 8
    if added_padding > 0:
      print(
          f"[Fq2 bucket_num_last_window] Was {orig_bucket_num_last_window}, "
          f"added {added_padding} to make it divisible by 8")
    self.bucket_num_last_window = orig_bucket_num_last_window + added_padding

    self.moduli_num = self.ec_ctx.get_finite_field_context().get_moduli_num()
    # Fp2 component axis cardinality (always 2 for Fp2).
    self.fe_dim = 2

    # Special-window duplication ratio (same logic as G1).
    self.log_special_duplication_ratio = math.ceil(
        math.log2(self.bucket_num_per_window / self.bucket_num_last_window))
    self.special_duplication_ratio = 2 ** self.log_special_duplication_ratio
    self.bucket_num_duplication = (
        self.bucket_num_last_window * self.special_duplication_ratio)

    expected_regular_bucket_size = self.tile_length / self.bucket_num_per_window
    expected_special_bucket_size = math.ceil(
        self.tile_length / self.bucket_num_duplication)
    self.expend_ratio = self.parameters["c_kernel_ret_space_ratio"]
    self.regular_bucket_size = int(expected_regular_bucket_size * self.expend_ratio)
    self.special_bucket_size = int(expected_special_bucket_size * self.expend_ratio)

  def _init_point_parameters(self):
    """Allocate a placeholder ``self.points`` tensor of shape
    ``(tile_num, tile_length, coord, 2, M)``.

    Real points are assigned later via ``self.points = ...`` (the
    prover sets this from the precomputed proving key)."""
    self.points = jnp.zeros(
        (self.tile_num, self.tile_length, self.coordinate_dim,
         self.fe_dim, self.moduli_num),
        dtype=jnp.uint32,
    )

  def _init_jax_data(self):
    """Initial all-identity bucket and window-sum tensors.

    Shapes (Fp2 axis inserted just before moduli axis):

      regular_window_buckets : (coord, W-1,         B, 2, M)
      last_window_buckets    : (coord,              B, 2, M)
      window_sum             : (coord,              B, W, 2, M)
    """
    # zero_point: EC ctx returns a list of self.coordinate_dim Fp2 tuples
    # (e.g. [(0,0), (1,0), (0,0)] for projective identity).  fe_ctx
    # to_computational_format gives shape (coord, 2, M).
    fe_ctx = self.ec_ctx.get_field_extension_context()
    self.zero_point = fe_ctx.to_computational_format(
        self.ec_ctx.zero_point)  # (coord, 2, M)
    assert self.zero_point.shape == (
        self.coordinate_dim, self.fe_dim, self.moduli_num), (
        f"unexpected zero_point shape {self.zero_point.shape}")

    self.regular_window_buckets = jnp.broadcast_to(
        self.zero_point.reshape(
            self.coordinate_dim, 1, 1, self.fe_dim, self.moduli_num),
        (self.coordinate_dim, self.window_num - 1,
         self.bucket_num_per_window, self.fe_dim, self.moduli_num),
    )
    self.last_window_buckets = jnp.broadcast_to(
        self.zero_point.reshape(
            self.coordinate_dim, 1, self.fe_dim, self.moduli_num),
        (self.coordinate_dim, self.bucket_num_per_window,
         self.fe_dim, self.moduli_num),
    )
    self.window_sum = jnp.broadcast_to(
        self.zero_point.reshape(
            self.coordinate_dim, 1, 1, self.fe_dim, self.moduli_num),
        (self.coordinate_dim, self.bucket_num_per_window,
         self.batch_window_num, self.fe_dim, self.moduli_num),
    )
    self._zero_templates = {
        "regular_window_buckets": self.regular_window_buckets,
        "last_window_buckets":    self.last_window_buckets,
        "window_sum":             self.window_sum,
    }

  def reset(self):
    for name, zero in self._zero_templates.items():
      setattr(self, name, zero)

  # --------------------------------------------------------------------- #
  #  Public format conversion + scalar preprocessing                       #
  # --------------------------------------------------------------------- #

  def _preprocess_scalars(self, scalars: list):
    tiled_scalar_list = utils.split_list(scalars, self.tile_length)
    tiled_slices_list = [
        utils.slice_scalars(s, self.scalar_bits, self.slice_bits)
        for s in tiled_scalar_list
    ]
    return jnp.array(tiled_slices_list, dtype=jnp.int32)

  def to_original_format(self, a: jnp.ndarray) -> list:
    return self.ec_ctx.to_original_format(a)

  def to_computational_format(self, scalars: Optional[list] = None) -> jnp.ndarray:
    if scalars is None:
      warnings.warn("No scalars provided, generating random scalars.")
      scalars = [random.randint(0, self.order - 1)
                 for _ in range(self.msm_length)]
    return self._preprocess_scalars(scalars)

  # --------------------------------------------------------------------- #
  #  Bucketization kernels (sort-based, Fp2-aware)                         #
  # --------------------------------------------------------------------- #

  def _bucketize_regular_windows(
      self,
      items: jnp.ndarray,        # (tile_length, coord, 2, M)
      bucket_ids: jnp.ndarray,   # (W-1, tile_length)
  ) -> jnp.ndarray:
    """Sort-based scatter for regular windows.

    Returns: ``(reg_bucket_size, coord, W-1, B, 2, M)``
    """
    n = self.tile_length
    b = self.window_num - 1

    bucket_ids_i32 = bucket_ids.astype(jnp.int32)
    sorted_indices = jnp.argsort(bucket_ids_i32, axis=1, stable=True)
    sorted_buckets = jnp.take_along_axis(bucket_ids_i32, sorted_indices, axis=1)

    is_boundary = jnp.concatenate([
        jnp.ones_like(sorted_buckets[:, :1], dtype=jnp.bool_),
        sorted_buckets[:, 1:] != sorted_buckets[:, :-1],
    ], axis=1)
    positions = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32), (b, n))
    boundary_positions = jnp.where(is_boundary, positions, jnp.int32(0))
    bucket_starts = jax.lax.associative_scan(
        jnp.maximum, boundary_positions, axis=1)
    within_rank = positions - bucket_starts

    batch_idx = jnp.broadcast_to(jnp.arange(b, dtype=jnp.int32)[:, None], (b, n))
    zeros_bn = jnp.zeros((b, n), dtype=jnp.int32)
    inv_rank = zeros_bn.at[batch_idx, sorted_indices].set(within_rank)
    items_b = jnp.broadcast_to(
        items[None, :, :, :, :],
        (b, n, self.coordinate_dim, self.fe_dim, self.moduli_num),
    )

    output = jnp.broadcast_to(
        self.zero_point.reshape(
            1, self.coordinate_dim, 1, 1, self.fe_dim, self.moduli_num),
        (self.regular_bucket_size, self.coordinate_dim, b,
         self.bucket_num_per_window, self.fe_dim, self.moduli_num),
    ).copy()
    output = output.at[inv_rank, :, batch_idx, bucket_ids_i32].set(items_b)
    return output

  def _bucketize_last_window(
      self,
      items: jnp.ndarray,        # (tile_length, coord, 2, M)
      last_slices: jnp.ndarray,  # (tile_length,)
  ) -> jnp.ndarray:
    """Sort-based scatter for the last (special) window with duplication.

    Returns: ``(special_bucket_size, coord, dup_ratio, bucket_num_last, 2, M)``
    """
    n = self.tile_length
    sdup = self.special_duplication_ratio

    dup_ids = jnp.arange(n, dtype=jnp.int32) % sdup
    bucket_ids_i32 = (
        dup_ids * self.bucket_num_last_window
        + last_slices.astype(jnp.int32))

    sorted_indices = jnp.argsort(bucket_ids_i32, stable=True)
    sorted_buckets = bucket_ids_i32[sorted_indices]

    is_boundary = jnp.concatenate([
        jnp.ones((1,), dtype=jnp.bool_),
        sorted_buckets[1:] != sorted_buckets[:-1],
    ])
    positions = jnp.arange(n, dtype=jnp.int32)
    boundary_positions = jnp.where(is_boundary, positions, jnp.int32(0))
    bucket_starts = jax.lax.associative_scan(jnp.maximum, boundary_positions)
    within_rank = positions - bucket_starts

    zeros_n = jnp.zeros((n,), dtype=jnp.int32)
    inv_rank = zeros_n.at[sorted_indices].set(within_rank)

    output = jnp.broadcast_to(
        self.zero_point.reshape(
            1, self.coordinate_dim, 1, self.fe_dim, self.moduli_num),
        (self.special_bucket_size, self.coordinate_dim,
         self.bucket_num_duplication, self.fe_dim, self.moduli_num),
    ).copy()
    output = output.at[inv_rank, :, bucket_ids_i32].set(items)
    return output.reshape(
        self.special_bucket_size,
        self.coordinate_dim,
        self.special_duplication_ratio,
        self.bucket_num_last_window,
        self.fe_dim,
        self.moduli_num,
    )

  def _distribute_buckets(
      self,
      regular_slices: jnp.ndarray,
      last_window_slice: jnp.ndarray,
      tiled_points: jnp.ndarray,
  ):
    return (
        self._bucketize_regular_windows(tiled_points, regular_slices),
        self._bucketize_last_window(tiled_points, last_window_slice),
    )

  # --------------------------------------------------------------------- #
  #  Bucket accumulation (BA)                                              #
  # --------------------------------------------------------------------- #

  def _bucket_accumulation_regular_windows_2d_parallel(
      self,
      regular_buckets: jnp.ndarray,    # (coord, W-1, B, 2, M)
      all_points: jnp.ndarray,          # (reg_bucket_size, coord, W-1, B, 2, M)
  ) -> jnp.ndarray:
    bucket_size_dim = all_points.shape[0]

    def scan_body(buckets, points):
      return self._padd(buckets, points), None

    buckets, _ = jax.lax.scan(
        scan_body, regular_buckets, all_points, length=bucket_size_dim)
    return buckets

  def _bucket_accumulation_last_window(
      self,
      buckets_in: jnp.ndarray,     # (coord, bucket_num_last, 2, M)
      window_points: jnp.ndarray,  # (special_bucket_size, coord, dup, bucket_num_last, 2, M)
  ) -> jnp.ndarray:
    coordinate_dim, buckets_dim, fe_dim, precision_dim = buckets_in.shape
    bucket_size_dim, _, bucket_dup_dim, _, _, _ = window_points.shape

    # Collapse dup × bucket axes for the scan body: (size, coord, dup·B, 2, M)
    window_points = window_points.reshape(
        bucket_size_dim, coordinate_dim, -1, fe_dim, precision_dim)
    base_dup_buckets = window_points[0]

    def scan_body(buckets, points):
      return self._padd(buckets, points), None

    dup_buckets, _ = jax.lax.scan(
        scan_body, base_dup_buckets, window_points[1:],
        length=bucket_size_dim - 1)

    # Reduce duplicated buckets via tree split along the dup·B axis (axis 1).
    log_bucket_dup_dim = int(math.log2(bucket_dup_dim))
    for _ in range(log_bucket_dup_dim):
      buckets_split = jnp.split(dup_buckets, 2, axis=1)
      dup_buckets = self._padd(buckets_split[0], buckets_split[1])

    return self._padd(buckets_in, dup_buckets)

  def _bucket_accumulation_all_windows(
      self,
      regular_window_buckets: jnp.ndarray,
      last_window_buckets: jnp.ndarray,
      regular_window_points: jnp.ndarray,
      last_window_points: jnp.ndarray,
  ):
    _, _, _, last_window_bucket_dim, _, _ = last_window_points.shape

    regular_window_buckets = (
        self._bucket_accumulation_regular_windows_2d_parallel(
            regular_window_buckets, regular_window_points))

    last_point_buckets = last_window_buckets[:, :last_window_bucket_dim, :, :]
    last_blank_buckets = last_window_buckets[:, last_window_bucket_dim:, :, :]
    last_point_buckets = self._bucket_accumulation_last_window(
        last_point_buckets, last_window_points)

    last_window_buckets = jnp.concatenate(
        (last_point_buckets, last_blank_buckets), axis=1)

    return regular_window_buckets, last_window_buckets

  def _bucketize_and_bucket_accumulation(
      self,
      regular_window_slices: jnp.ndarray,
      last_window_slices: jnp.ndarray,
      tiled_points: jnp.ndarray,
      regular_window_buckets: jnp.ndarray,
      last_window_buckets: jnp.ndarray,
  ):
    regular_window_points = self._bucketize_regular_windows(
        tiled_points, regular_window_slices)
    last_window_points = self._bucketize_last_window(
        tiled_points, last_window_slices)
    return self._bucket_accumulation_all_windows(
        regular_window_buckets, last_window_buckets,
        regular_window_points, last_window_points,
    )

  # --------------------------------------------------------------------- #
  #  Bucket reduction (BR) — tree algorithm, Fp2 axis preserved            #
  # --------------------------------------------------------------------- #

  def _bucket_reduction(
      self,
      regular_window_buckets: jnp.ndarray,
      last_window_buckets: jnp.ndarray,
      window_sum: jnp.ndarray,
  ) -> jnp.ndarray:
    """Tree reduction matching ``FusionMSMContext._bucket_reduction`` with
    the Fp2 axis appearing as ``axis=-2`` throughout."""
    # regular: (C, W-1, B, 2, M)
    # last:    (C, B, 2, M) → expand to (C, 1, B, 2, M) then concat
    all_buckets = jnp.concatenate(
        (regular_window_buckets, last_window_buckets[:, jnp.newaxis]),
        axis=1)
    bucket_num_in_window = all_buckets.shape[2]
    # (C, W, B, 2, M) → (C, B, W, 2, M)
    all_buckets = all_buckets.transpose(0, 2, 1, 3, 4)
    iter_num = int(math.log2(bucket_num_in_window))

    cd, m, wd, fed, pd = all_buckets.shape  # m = current bucket axis size
    for _ in range(iter_num):
      m = all_buckets.shape[1]
      half = m // 2

      # Split bucket axis into adjacent pairs.
      all_buckets = all_buckets.reshape(cd, half, 2, wd, fed, pd)
      window_sum  = window_sum.reshape (cd, half, 2, wd, fed, pd)

      all_buckets_left,  all_buckets_right  = all_buckets[:, :, 0], all_buckets[:, :, 1]
      window_sum_left,   window_sum_right   = window_sum [:, :, 0], window_sum [:, :, 1]

      window_sum = self._padd(
          self._padd(window_sum_left, window_sum_right), all_buckets_right)
      bucket_sum = self._padd(all_buckets_left, all_buckets_right)
      all_buckets = self._padd(bucket_sum, bucket_sum)

    return window_sum[:, 0]   # (C, W, 2, M)

  # --------------------------------------------------------------------- #
  #  Window merge (WM)                                                     #
  # --------------------------------------------------------------------- #

  def _window_merge(self, window_sum: jnp.ndarray) -> jnp.ndarray:
    """``window_sum: (C, W, 2, M) → (C, 2, M)``.

    Same Horner-style scan as G1 but with the Fp2 axis preserved.
    """
    coordinate_dim, window_dim, fe_dim, precision_dim = window_sum.shape
    # (C, W, 2, M) → (W, C, 1, 2, M)
    window_sum = window_sum.transpose(1, 0, 2, 3).reshape(
        (window_dim, coordinate_dim, 1, fe_dim, precision_dim))
    result = window_sum[window_dim - 1]

    def fori_loop_body(i, result):
      return self._padd(result, result)

    def scan_body(result, ws):
      result = jax.lax.fori_loop(0, self.slice_bits, fori_loop_body, result)
      result = self._padd(result, ws)
      return result, None

    result, _ = jax.lax.scan(
        scan_body, result, window_sum[: window_dim - 1],
        reverse=True, length=window_dim - 1)
    return result.reshape((coordinate_dim, fe_dim, precision_dim))

  # --------------------------------------------------------------------- #
  #  Fused BBA + BWS  (the per-tile kernel)                                #
  # --------------------------------------------------------------------- #

  def _bucket_and_window_sum(
      self,
      regular_window_buckets: jnp.ndarray,
      last_window_buckets: jnp.ndarray,
      window_sum: jnp.ndarray,
  ) -> jnp.ndarray:
    bucket_summations = self._bucket_reduction(
        regular_window_buckets, last_window_buckets, window_sum)
    return self._window_merge(bucket_summations)

  def _bba_and_bws(
      self,
      regular_window_slices: jnp.ndarray,
      last_window_slices: jnp.ndarray,
      tiled_points: jnp.ndarray,
      regular_window_buckets: jnp.ndarray,
      last_window_buckets: jnp.ndarray,
      window_sum: jnp.ndarray,
  ) -> jnp.ndarray:
    regular_window_buckets, last_window_buckets = (
        self._bucketize_and_bucket_accumulation(
            regular_window_slices, last_window_slices, tiled_points,
            regular_window_buckets, last_window_buckets,
        ))
    return self._bucket_and_window_sum(
        regular_window_buckets, last_window_buckets, window_sum)

  # --------------------------------------------------------------------- #
  #  Shape-dtype structs for AOT compile                                   #
  # --------------------------------------------------------------------- #

  def _get_bba_shape_dtype_structs(self):
    """ShapeDtypeStructs for _bucketize_and_bucket_accumulation inputs.

    Fp2 variant of :meth:`FusionMSMContext._get_bba_shape_dtype_structs`:
    every points / bucket tensor carries an extra ``fe_dim`` axis for the
    Fp2 extension coordinate.  ``tiled_points`` is the per-tile points
    tensor; including it as a ShapeDtypeStruct here makes ``points`` an
    argument to the AOT-compiled kernel rather than a closure capture,
    so swapping circuits never triggers a re-trace.
    """
    regular_slices_shape = (self.window_num - 1, self.tile_length)
    last_window_slice_shape = (self.tile_length,)
    tiled_points_shape = (
        self.tile_length, self.coordinate_dim, self.fe_dim, self.moduli_num)
    regular_window_buckets_shape = (
        self.coordinate_dim, self.window_num - 1, self.bucket_num_per_window,
        self.fe_dim, self.moduli_num)
    last_window_buckets_shape = (
        self.coordinate_dim, self.bucket_num_per_window,
        self.fe_dim, self.moduli_num)
    return [
        jax.ShapeDtypeStruct(regular_slices_shape, jnp.int32),
        jax.ShapeDtypeStruct(last_window_slice_shape, jnp.int32),
        jax.ShapeDtypeStruct(tiled_points_shape, jnp.uint32),
        jax.ShapeDtypeStruct(regular_window_buckets_shape, jnp.uint32),
        jax.ShapeDtypeStruct(last_window_buckets_shape, jnp.uint32),
    ]

  def _get_bws_shape_dtype_structs(self):
    regular_window_buckets_shape = (
        self.coordinate_dim, self.window_num - 1, self.bucket_num_per_window,
        self.fe_dim, self.moduli_num)
    last_window_buckets_shape = (
        self.coordinate_dim, self.bucket_num_per_window,
        self.fe_dim, self.moduli_num)
    window_sum_shape = (
        self.coordinate_dim, self.bucket_num_per_window,
        self.batch_window_num, self.fe_dim, self.moduli_num)
    return [
        jax.ShapeDtypeStruct(regular_window_buckets_shape, jnp.uint32),
        jax.ShapeDtypeStruct(last_window_buckets_shape, jnp.uint32),
        jax.ShapeDtypeStruct(window_sum_shape, jnp.uint32),
    ]

  def _get_bba_bws_shape_dtype_structs(self):
    bba = self._get_bba_shape_dtype_structs()
    bws = self._get_bws_shape_dtype_structs()
    return bba + [bws[2]]

  def _get_fused_shape_dtype_structs(self):
    bba = self._get_bba_shape_dtype_structs()
    bws = self._get_bws_shape_dtype_structs()
    per_tile_length = bba[0].shape[1]
    regular_tiled_slices_shape = (self.tile_num, self.window_num - 1, per_tile_length)
    last_tiled_slices_shape    = (self.tile_num, per_tile_length)
    tiled_points_shape         = (self.tile_num,) + tuple(bba[2].shape)
    return [
        jax.ShapeDtypeStruct(regular_tiled_slices_shape, jnp.int32),
        jax.ShapeDtypeStruct(last_tiled_slices_shape, jnp.int32),
        jax.ShapeDtypeStruct(tiled_points_shape, jnp.uint32),
        bba[3], bba[4], bws[2],
    ]

  # --------------------------------------------------------------------- #
  #  Compile + dispatch                                                    #
  # --------------------------------------------------------------------- #

  def compile(self, parameters: dict = None):
    parameters = parameters or {}
    use_fused = parameters.get("use_fused", False)
    self.use_fused = use_fused

    bba_structs = self._get_bba_shape_dtype_structs()
    bws_structs = self._get_bws_shape_dtype_structs()
    bba_bws_structs = self._get_bba_bws_shape_dtype_structs()

    bba_hash = hash_args(
        *(v for s in bba_structs for v in (s.shape, s.dtype.__str__())))
    bws_hash = hash_args(
        *(v for s in bws_structs for v in (s.shape, s.dtype.__str__())))
    bba_bws_hash = hash_args(
        *(v for s in bba_bws_structs for v in (s.shape, s.dtype.__str__())))

    # The fused-last-tile kernel is used by both paths.
    self.compiled_kernels.setdefault(bba_bws_hash, {})["bba_and_bws"] = (
        jax_jit_lower_compile(self._bba_and_bws, *bba_bws_structs)
    )

    if use_fused:
      fused_structs = self._get_fused_shape_dtype_structs()
      fused_hash = hash_args(
          *(v for s in fused_structs for v in (s.shape, s.dtype.__str__())))
      self.compiled_kernels.setdefault(fused_hash, {})[
          "multiscalar_multiply_fused"] = (
              jax_jit_lower_compile(self._multiscalar_multiply_fused,
                                    *fused_structs)
          )
    else:
      if self.tile_num > 1:
        self.compiled_kernels.setdefault(bba_hash, {})[
            "bucketize_and_bucket_accumulation"] = (
                jax_jit_lower_compile(self._bucketize_and_bucket_accumulation,
                                      *bba_structs)
            )
      self.compiled_kernels.setdefault(bws_hash, {})[
          "bucket_and_window_sum"] = (
              jax_jit_lower_compile(self._bucket_and_window_sum, *bws_structs)
          )

    self.use_compiled_kernels = True

  def bba_and_bws(self, *args):
    if self.use_compiled_kernels:
      kernel_hash = hash_args(
          *(v for a in args for v in (a.shape, a.dtype.__str__())))
      return self.compiled_kernels[kernel_hash]["bba_and_bws"](*args)
    # Use the jit-wrapped function so the ~300 inner padds get fused
    # into a single compiled HLO graph (cached by JAX after first call).
    return self._jit_bba_and_bws(*args)

  def bucketize_and_bucket_accumulation(self, *args):
    if self.use_compiled_kernels:
      kernel_hash = hash_args(
          *(v for a in args for v in (a.shape, a.dtype.__str__())))
      return self.compiled_kernels[kernel_hash][
          "bucketize_and_bucket_accumulation"](*args)
    return self._bucketize_and_bucket_accumulation(*args)

  def bucket_and_window_sum(self, *args):
    if self.use_compiled_kernels:
      kernel_hash = hash_args(
          *(v for a in args for v in (a.shape, a.dtype.__str__())))
      return self.compiled_kernels[kernel_hash][
          "bucket_and_window_sum"](*args)
    return self._bucket_and_window_sum(*args)

  def context_hash(self) -> str:
    return hash_args(
        self.__class__.__name__,
        self.ec_ctx.context_hash() if hasattr(self.ec_ctx, "context_hash")
        else self.ec_ctx.__class__.__name__,
        self.slice_bits, self.scalar_bits,
        self.msm_length, self.tile_length,
        self.bucket_num_per_window, self.bucket_num_last_window,
        self.fe_dim, self.coordinate_dim,
    )

  # --------------------------------------------------------------------- #
  #  Drivers                                                               #
  # --------------------------------------------------------------------- #

  def _multiscalar_multiply(self, tiled_slices: jnp.ndarray):
    last_tile = self.tile_num - 1
    for tile_index in range(self.tile_num):
      tiled_slices_tile = tiled_slices[tile_index]
      regular_slices_tile = tiled_slices_tile[: self.window_num - 1]
      last_window_slice_tile = tiled_slices_tile[self.window_num - 1]
      tiled_points_tile = self.points[tile_index]

      if tile_index == last_tile:
        return self.bba_and_bws(
            regular_slices_tile, last_window_slice_tile, tiled_points_tile,
            self.regular_window_buckets, self.last_window_buckets, self.window_sum,
        )

      self.regular_window_buckets, self.last_window_buckets = (
          self.bucketize_and_bucket_accumulation(
              regular_slices_tile, last_window_slice_tile, tiled_points_tile,
              self.regular_window_buckets, self.last_window_buckets,
          ))

  def _multiscalar_multiply_fused(
      self,
      regular_tiled_slices: jnp.ndarray,
      last_tiled_slices: jnp.ndarray,
      tiled_points: jnp.ndarray,
      regular_window_buckets: jnp.ndarray,
      last_window_buckets: jnp.ndarray,
      window_sum: jnp.ndarray,
  ) -> jnp.ndarray:
    def scan_body(carry, inputs):
      reg_buckets, last_buckets = carry
      regular_slices, last_window_slice, points_tile = inputs
      reg_buckets, last_buckets = self._bucketize_and_bucket_accumulation(
          regular_slices, last_window_slice, points_tile,
          reg_buckets, last_buckets,
      )
      return (reg_buckets, last_buckets), None

    if self.tile_num > 1:
      (regular_window_buckets, last_window_buckets), _ = jax.lax.scan(
          scan_body,
          (regular_window_buckets, last_window_buckets),
          (regular_tiled_slices[:-1], last_tiled_slices[:-1], tiled_points[:-1]),
          length=self.tile_num - 1,
      )

    return self._bba_and_bws(
        regular_tiled_slices[-1],
        last_tiled_slices[-1],
        tiled_points[-1],
        regular_window_buckets,
        last_window_buckets,
        window_sum,
    )

  def multiscalar_multiply(self, tiled_slices: jnp.ndarray):
    if not self.use_fused:
      return self._multiscalar_multiply(tiled_slices)

    regular_tiled_slices = tiled_slices[:, : self.window_num - 1]
    last_tiled_slices    = tiled_slices[:, self.window_num - 1]
    args = (
        regular_tiled_slices, last_tiled_slices, self.points,
        self.regular_window_buckets, self.last_window_buckets, self.window_sum,
    )
    if self.use_compiled_kernels:
      kernel_hash = hash_args(
          *(v for a in args for v in (a.shape, a.dtype.__str__())))
      return self.compiled_kernels[kernel_hash]["multiscalar_multiply_fused"](*args)
    return self._multiscalar_multiply_fused(*args)

