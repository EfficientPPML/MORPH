from abc import ABC, abstractmethod
import ctypes
import math
import logging
from typing import Tuple, Union
import copy
import numpy as np
import warnings
import jax
import jax.numpy as jnp
import jax.ffi as ffi
from jax import sharding as shd

import utils
from utils import JaxKernelContextBase, JaxParameters, hash_args, jax_jit_lower_compile, store_jax_executable, load_jax_executable
from finite_field_context import FiniteFieldContextBase
from elliptic_curve_context import EllipticCurveContextBase

jax.config.update("jax_enable_x64", True)


class MultiscalarMultiplicationContextBase(ABC):

  def __init__(self, parameters: dict):
    self.parameters = parameters
    self.ec_ctx_class = parameters.get("elliptic_curve_context_class", None)
    assert self.ec_ctx_class is not None, "elliptic_curve_context_class must be provided"
    self.ec_ctx: EllipticCurveContextBase = self.ec_ctx_class(parameters.get("elliptic_curve_parameters", {}))

  @abstractmethod
  def to_computational_format(self, a: list) -> list:
    pass

  @abstractmethod
  def to_original_format(self, a: list) -> list:
    pass

  @abstractmethod
  def multiscalar_multiply(self, tiled_slices: jnp.ndarray):
    pass


class OldCPUDistributionMSMContext(MultiscalarMultiplicationContextBase):

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    self._init_config_parameters()
    self._init_jax_data()
    self._init_cpu_kernels()
    self._init_point_parameters()

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

  def _padd(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return self.ec_ctx.point_add(a, b)

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
      tiled_points_tile = self.points[:, idx_start:idx_end].transpose(1, 0, 2)
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



class PerfTestDistributionMSMContext(MultiscalarMultiplicationContextBase, JaxKernelContextBase):
  def __init__(self, parameters: dict):
    super().__init__(parameters)
    JaxKernelContextBase.__init__(self)
    self.slice_bits = parameters.get("slice_bits")


  def multiscalar_multiply(self, tiled_slices: jnp.ndarray):
    pass

  def to_original_format(self, a: jnp.ndarray) -> list:
    pass

  def to_computational_format(self, scalars: list) -> jnp.ndarray:
    pass

  def _padd(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return self.ec_ctx.point_add(a, b)

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
        parameters: Computation parameters.

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
      self, all_buckets: jnp.ndarray, temp_sum: jnp.ndarray, window_sum: jnp.ndarray) -> jnp.ndarray:
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
        scan_body, (temp_sum, window_sum), all_buckets[:bucket_num_in_window], length=bucket_num_in_window, reverse=True
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

  def _bucket_reduction_new(
      self, all_buckets: jnp.ndarray,  window_sum: jnp.ndarray
  ) -> jnp.ndarray:
    """Reduce buckets to window sums using tree-based parallel algorithm.

    Computes S = sum_{i=0}^{n-1} i * B[i] per window via adjacent-pair
    tree reduction in O(log n) parallel steps.

    At each level k, adjacent pairs are merged:
      window_sum = ws_left + ws_right + scaled_right
      all_buckets = left + right
      scaled = 2 * (scaled_left + scaled_right)

    where scaled tracks 2^k * bucket_sums to provide the correction weight.

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

    for _ in range(iter_num):
      all_buckets_left, all_buckets_right = all_buckets[:, 0::2], all_buckets[:, 1::2]
      window_sum_left, window_sum_right = window_sum[:, 0::2], window_sum[:, 1::2]

      window_sum = self._padd(self._padd(window_sum_left, window_sum_right), all_buckets_right)
      bucket_sum = self._padd(all_buckets_left, all_buckets_right)
      all_buckets = self._padd(bucket_sum, bucket_sum)

    return window_sum[:, 0]

  def _bucket_reduction_new2(self, all_buckets: jnp.ndarray, window_sum: jnp.ndarray) -> jnp.ndarray:
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
    self.bucket_num_per_window = 2**self.slice_bits  # Note: Inlcude Bucket 0
    self.bucket_num_last_window = (self.order >> ((self.window_num - 1) * self.slice_bits))+1 # Note: Inlcude Bucket 0
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

    # Move jax data to sharded devices
    self.all_buckets = self.all_buckets.to_device(self.ba_all_buckets_sharding)
    self.window_sum = self.window_sum.to_device(self.br_window_sum_sharding)

  def set_use_sharding(self, use_sharding: bool):
    super().set_use_sharding(use_sharding)
    if use_sharding:
      self._init_shardings()

  def _init_cpu_kernels(self):
    expected_regular_bucket_size = self.tile_length / (self.bucket_num_per_window)
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
    jax.ffi.register_ffi_target("distribute_buf", jax.ffi.pycapsule(lib.DistributeBufZero), platform="cpu")
    self.distribution_buf_c_kernel_call = ffi.ffi_call(
        "distribute_buf",
        (
            jax.ShapeDtypeStruct(regular_shape, jnp.uint32),
            jax.ShapeDtypeStruct(special_shape, jnp.uint32),
            jax.ShapeDtypeStruct((2,), jnp.uint32),
        ),
    )

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

    for _ in range(iter_num):
      all_buckets_left, all_buckets_right = all_buckets[:, 0::2], all_buckets[:, 1::2]
      window_sum_left, window_sum_right = window_sum[:, 0::2], window_sum[:, 1::2]

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

    self._compiled_kernel_hash = hash_args(
        ba_structs[0].shape, ba_structs[0].dtype.__str__(),
        br_structs[0].shape, br_structs[0].dtype.__str__(),
    )
    self.compiled_kernels[self._compiled_kernel_hash] = {
        "bucket_accumulation_all_windows": ba_kernel
        if ba_kernel is not None
        else jax_jit_lower_compile(self._bucket_accumulation_all_windows, ba_structs[0], ba_structs[1], ba_structs[2]),
        "bucket_reduction": br_kernel
        if br_kernel is not None
        else jax_jit_lower_compile(self._bucket_reduction, br_structs[0], br_structs[1], br_structs[2]),
        "window_merge": wm_kernel
        if wm_kernel is not None
        else jax_jit_lower_compile(self._window_merge, wm_structs[0]),
    }
    self.use_compiled_kernels = True

  def bucket_accumulation_all_windows(self, all_buckets, regular_points, last_window_points):
    if self.use_compiled_kernels:
      return self.compiled_kernels[self._compiled_kernel_hash]["bucket_accumulation_all_windows"](all_buckets, regular_points, last_window_points)
    else:
      return self._bucket_accumulation_all_windows(all_buckets, regular_points, last_window_points)

  def bucket_reduction(self, all_buckets, temp_sum, window_sum):
    if self.use_compiled_kernels:
      return self.compiled_kernels[self._compiled_kernel_hash]["bucket_reduction"](all_buckets, temp_sum, window_sum)
    else:
      return self._bucket_reduction(all_buckets, temp_sum, window_sum)

  def window_merge(self, window_sum):
    if self.use_compiled_kernels:
      return self.compiled_kernels[self._compiled_kernel_hash]["window_merge"](window_sum)
    else:
      return self._window_merge(window_sum)

  def distribute_buckets(self, tiled_slices: jnp.ndarray, tiled_points: jnp.ndarray) -> jnp.ndarray:
    if self.use_sharding:
      prev_mesh = shd.set_mesh(None)
      try:
        result = super().distribute_buckets(tiled_slices, tiled_points)
      finally:
        shd.set_mesh(prev_mesh)
      return result
    else:
      return super().distribute_buckets(tiled_slices, tiled_points)

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


class DemoTPUMSMContext(MultiscalarMultiplicationContextBase, JaxKernelContextBase):
  """TPU-only MSM demo: runs only BA (bucket accumulation), BR (bucket reduction),
  and WM (window merge) on TPU. No CPU preprocessing or distribution.
  Single tile where tile_length == msm_length.
  """

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    JaxKernelContextBase.__init__(self)
    self._init_config_parameters()
    self._init_jax_data()

  def _init_config_parameters(self):
    self.coordinate_dim = self.parameters.get("coordinate_dim", 4)
    self.msm_length = self.parameters.get("msm_length")
    self.tile_length = self.msm_length  # Single tile = full MSM length
    self.tile_num = 1
    self.slice_bits = self.parameters.get("slice_bits")
    self.scalar_bits = self.parameters.get("scalar_bits")
    self.order = self.parameters.get("order")
    self.window_num = int(math.ceil(self.scalar_bits / self.slice_bits))
    self.batch_window_num = self.window_num
    self.bucket_num_per_window = 2**self.slice_bits  # Include Bucket 0
    self.bucket_num_last_window = (self.order >> ((self.window_num - 1) * self.slice_bits)) + 1  # Include Bucket 0
    self.moduli_num = self.ec_ctx.get_finite_field_context().get_moduli_num()

    # Special bucket optimization
    self.log_special_duplication_ratio = math.ceil(math.log2(self.bucket_num_per_window / self.bucket_num_last_window))
    self.special_duplication_ratio = 2**self.log_special_duplication_ratio
    self.bucket_num_duplication = self.bucket_num_last_window * self.special_duplication_ratio

    # Compute BA input shapes (same logic as CPUDistributionMSMContext but without loading C kernels)
    self.expend_ratio = self.parameters.get("c_kernel_ret_space_ratio", 4)
    expected_regular_bucket_size = self.tile_length / self.bucket_num_per_window
    expected_special_bucket_size = math.ceil(self.tile_length / self.bucket_num_duplication)
    self.c_kernel_regular_bucket_size = int(expected_regular_bucket_size * self.expend_ratio)
    self.c_kernel_special_bucket_size = int(expected_special_bucket_size * self.expend_ratio)

    self.ba_input_regular_shape = (
        self.window_num - 1,
        self.bucket_num_per_window,
        self.c_kernel_regular_bucket_size,
        self.coordinate_dim,
        self.moduli_num,
    )
    self.ba_input_special_shape = (
        self.bucket_num_last_window,
        self.special_duplication_ratio,
        self.c_kernel_special_bucket_size,
        self.coordinate_dim,
        self.moduli_num,
    )

  def _init_jax_data(self):
    self.zero_point = self.ec_ctx.get_finite_field_context().to_computational_format(self.ec_ctx.zero_point)
    self.all_buckets = jnp.broadcast_to(
        self.zero_point.reshape(1, self.coordinate_dim, 1, self.moduli_num),
        (self.batch_window_num, self.coordinate_dim, self.bucket_num_per_window, self.moduli_num),
    )
    self.temp_sum = jnp.array([self.zero_point for _ in range(self.batch_window_num)]).transpose(1, 0, 2)
    self.window_sum = jnp.array([self.zero_point for _ in range(self.batch_window_num)]).transpose(1, 0, 2)
    self.window_sum = jnp.broadcast_to(
        self.window_sum.reshape(self.coordinate_dim, 1, self.batch_window_num, self.moduli_num),
        (self.coordinate_dim, self.bucket_num_per_window, self.batch_window_num, self.moduli_num),
    )

    # Pre-store shardings
    self.ba_all_buckets_sharding = None
    self.ba_regular_points_sharding = None
    self.ba_special_points_sharding = None
    self.br_all_buckets_sharding = None
    self.br_window_sum_sharding = None
    self.wm_window_sum_sharding = None

  def _init_shardings(self):
    all_buckets_shape = (self.batch_window_num, self.coordinate_dim, self.bucket_num_per_window, self.moduli_num)
    regular_points_shape = self.ba_input_regular_shape
    special_points_shape = self.ba_input_special_shape
    window_sum_shape = (self.coordinate_dim, self.bucket_num_per_window, self.batch_window_num, self.moduli_num)
    wm_window_sum_shape = (self.coordinate_dim, self.batch_window_num, self.moduli_num)

    self.ba_all_buckets_sharding, _ = self.create_named_sharding(shape=all_buckets_shape, axes=[2])
    self.ba_regular_points_sharding, _ = self.create_named_sharding(shape=regular_points_shape, axes=[1])
    self.ba_special_points_sharding, _ = self.create_named_sharding(shape=special_points_shape, axes=[0, 1])
    self.br_all_buckets_sharding, _ = self.create_named_sharding(shape=all_buckets_shape, axes=[2])
    self.br_window_sum_sharding, _ = self.create_named_sharding(shape=window_sum_shape, axes=[1])
    self.wm_window_sum_sharding, _ = self.create_named_sharding(shape=wm_window_sum_shape, axes=[])

    self.all_buckets = self.all_buckets.to_device(self.ba_all_buckets_sharding)
    self.window_sum = self.window_sum.to_device(self.br_window_sum_sharding)

  def set_use_sharding(self, use_sharding: bool):
    super().set_use_sharding(use_sharding)
    if use_sharding:
      self._init_shardings()

  def generate_dummy_inputs(self):
    """Generate dummy inputs for the TPU MSM pipeline (BA -> BR -> WM).

    Returns:
        (regular_points, last_window_points): Dummy inputs for bucket accumulation.
    """
    regular_points = jnp.zeros(self.ba_input_regular_shape, dtype=jnp.uint32)
    last_window_points = jnp.zeros(self.ba_input_special_shape, dtype=jnp.uint32)

    if self.use_sharding:
      regular_points = regular_points.to_device(self.ba_regular_points_sharding)
      last_window_points = last_window_points.to_device(self.ba_special_points_sharding)
    else:
      regular_points = jax.device_put(regular_points, jax.devices("tpu")[0])
      last_window_points = jax.device_put(last_window_points, jax.devices("tpu")[0])

    return regular_points, last_window_points

  def _padd(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return self.ec_ctx.point_add(a, b)

  def _bucket_accumulation_per_window(self, buckets: jnp.ndarray, window_points: jnp.ndarray) -> jnp.ndarray:
    bucket_size_dim = window_points.shape[0]

    def scan_body(buckets, points):
      buckets = self._padd(buckets, points)
      return buckets, None

    buckets, _ = jax.lax.scan(scan_body, buckets, window_points, length=bucket_size_dim)
    return buckets

  def _bucket_accumulation_regular_windows(self, regular_buckets: jnp.ndarray, all_points: jnp.ndarray) -> jnp.ndarray:
    window_dim = regular_buckets.shape[0]

    def scan_body(empty, window_bucket_point_pack):
      window_buckets, points = window_bucket_point_pack
      buckets = self._bucket_accumulation_per_window(window_buckets, points)
      return None, buckets

    _, buckets = jax.lax.scan(scan_body, None, (regular_buckets, all_points), length=window_dim)
    return buckets

  def _bucket_accumulation_last_window(self, buckets_in: jnp.ndarray, window_points: jnp.ndarray) -> jnp.ndarray:
    coordinate_dim, buckets_dim, precision_dim = buckets_in.shape
    bucket_size_dim, _, bucket_dup_dim, _, _ = window_points.shape

    window_points = window_points.reshape(bucket_size_dim, coordinate_dim, -1, precision_dim)
    base_dup_buckets = window_points[0]

    def scan_body(buckets, points):
      buckets = self._padd(buckets, points)
      return buckets, None

    dup_buckets, _ = jax.lax.scan(scan_body, base_dup_buckets, window_points[1:], length=bucket_size_dim - 1)

    log_bucket_dup_dim = int(math.log2(bucket_dup_dim))
    for _ in range(log_bucket_dup_dim):
      buckets_split = jnp.split(dup_buckets, 2, axis=1)
      dup_buckets = self._padd(buckets_split[0], buckets_split[1])

    buckets_in = self._padd(buckets_in, dup_buckets)
    return buckets_in

  def _bucket_accumulation_all_windows(
      self, all_buckets: jnp.ndarray, regular_points: jnp.ndarray, last_window_points: jnp.ndarray
  ) -> jnp.ndarray:
    regular_points = regular_points.transpose(0, 2, 3, 1, 4)
    last_window_points = last_window_points.transpose(2, 3, 1, 0, 4)

    window_dim, coordinate_dim, buckets_dim, precision_dim = all_buckets.shape
    _, _, last_window_bucket_dup, last_window_bucket_dim, _ = last_window_points.shape

    regular_buckets = all_buckets[: window_dim - 1]
    regular_buckets = self._bucket_accumulation_regular_windows(regular_buckets, regular_points)

    last_point_buckets = all_buckets[window_dim - 1, :, :last_window_bucket_dim, :]
    last_blank_buckets = all_buckets[window_dim - 1, :, last_window_bucket_dim:, :]
    last_point_buckets = self._bucket_accumulation_last_window(last_point_buckets, last_window_points)

    last_buckets = jax.lax.broadcast(jnp.concatenate((last_point_buckets, last_blank_buckets), axis=1), (1,))
    all_buckets = jnp.concatenate((regular_buckets, last_buckets), axis=0)
    return all_buckets

  def _bucket_reduction(self, all_buckets: jnp.ndarray, temp_sum: jnp.ndarray, window_sum: jnp.ndarray) -> jnp.ndarray:
    bucket_num_in_window = all_buckets.shape[2]
    all_buckets = all_buckets.transpose(1, 2, 0, 3)
    iter_num = int(math.log2(bucket_num_in_window))

    for _ in range(iter_num):
      all_buckets_left, all_buckets_right = all_buckets[:, 0::2], all_buckets[:, 1::2]
      window_sum_left, window_sum_right = window_sum[:, 0::2], window_sum[:, 1::2]

      window_sum = self._padd(self._padd(window_sum_left, window_sum_right), all_buckets_right)
      bucket_sum = self._padd(all_buckets_left, all_buckets_right)
      all_buckets = self._padd(bucket_sum, bucket_sum)

    return window_sum[:, 0]

  def _window_merge(self, window_sum: jnp.ndarray) -> jnp.ndarray:
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

  def _get_ba_shape_dtype_structs(self):
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
      warnings.warn("Not found stored serialized compiled kernels for DemoTPUMSM, compiling...", UserWarning, stacklevel=2)

    self._compiled_kernel_hash = hash_args(
        ba_structs[0].shape, ba_structs[0].dtype.__str__(),
        br_structs[0].shape, br_structs[0].dtype.__str__(),
    )
    self.compiled_kernels[self._compiled_kernel_hash] = {
        "bucket_accumulation_all_windows": ba_kernel
        if ba_kernel is not None
        else jax_jit_lower_compile(self._bucket_accumulation_all_windows, ba_structs[0], ba_structs[1], ba_structs[2]),
        "bucket_reduction": br_kernel
        if br_kernel is not None
        else jax_jit_lower_compile(self._bucket_reduction, br_structs[0], br_structs[1], br_structs[2]),
        "window_merge": wm_kernel
        if wm_kernel is not None
        else jax_jit_lower_compile(self._window_merge, wm_structs[0]),
    }
    self.use_compiled_kernels = True

  def bucket_accumulation_all_windows(self, all_buckets, regular_points, last_window_points):
    if self.use_compiled_kernels:
      return self.compiled_kernels[self._compiled_kernel_hash]["bucket_accumulation_all_windows"](all_buckets, regular_points, last_window_points)
    else:
      return self._bucket_accumulation_all_windows(all_buckets, regular_points, last_window_points)

  def bucket_reduction(self, all_buckets, temp_sum, window_sum):
    if self.use_compiled_kernels:
      return self.compiled_kernels[self._compiled_kernel_hash]["bucket_reduction"](all_buckets, temp_sum, window_sum)
    else:
      return self._bucket_reduction(all_buckets, temp_sum, window_sum)

  def window_merge(self, window_sum):
    if self.use_compiled_kernels:
      return self.compiled_kernels[self._compiled_kernel_hash]["window_merge"](window_sum)
    else:
      return self._window_merge(window_sum)

  def multiscalar_multiply(self, regular_points: jnp.ndarray = None, last_window_points: jnp.ndarray = None):
    """Run TPU-only MSM: BA -> BR -> WM.

    Args:
        regular_points: Regular window points. If None, uses generate_dummy_inputs().
        last_window_points: Last window points. If None, uses generate_dummy_inputs().

    Returns:
        Final MSM result (coordinate_dim, precision_dim).
    """
    # if regular_points is None or last_window_points is None:
    #   regular_points, last_window_points = self.generate_dummy_inputs()

    all_buckets = self.bucket_accumulation_all_windows(self.all_buckets, regular_points, last_window_points)
    window_sum = self.bucket_reduction(all_buckets, self.temp_sum, self.window_sum)
    result = self.window_merge(window_sum)
    return all_buckets


  def to_original_format(self, a: jnp.ndarray) -> list:
    return self.ec_ctx.to_original_format(a)

  def to_computational_format(self, scalars: list) -> jnp.ndarray:
    raise NotImplementedError("DemoTPUMSM does not support scalar preprocessing. Use generate_dummy_inputs() instead.")