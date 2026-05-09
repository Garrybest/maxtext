# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Muon.

Implementation of the
[Muon optimizer](https://github.com/KellerJordan/modded-nanogpt)
by Keller Jordan
"""

# pylint: disable=unnecessary-lambda-assignment


import functools
import itertools
import math
from typing import Any, Callable, NamedTuple, Optional, Union, Sequence, Literal

import jax
import jax.numpy as jnp

from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
from optax.transforms import _masking
import optax.tree

ReshapeFn = Callable[[jax.Array], jax.Array]

_PRECONDITIONINGS = ["frobenius", "spectral", "aol", "schatten"]
_DEFAULT_NS_COEFFS = (3.4445, -4.7750, 2.0315)
_DION_NS_COEFFS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]
_NS_COEFFS_PRESET_DICT = {
    "standard": _DEFAULT_NS_COEFFS,
    "dion": _DION_NS_COEFFS,
}


class MuonDimensionNumbers(NamedTuple):
  """Specification for which weight axes participate in matrix projection.

  Muon defines an orthogonalization for 2D matrix weights for matrix-vector
  products:

  .. math::
    x W = y

  where the first matrix dimension is the reduction axis and the second matrix
  dimension is the output axis. Thus, the default spec consists of 0 and 1
  reduction and output axes respectively.

  .. warning::
    The batch axes are implicit, all axes not specified as reduction or output
    axes are considered batch axes and will be considered independently in the
    orthogonalization (via jax.vmap).

  When ``component_splits`` is set, the output axis is split into components
  (e.g., MLA nope/rope) and the batch axes are merged into each component's
  output dimension before NS, matching Megatron's per-component treatment.
  Each component gets its own NS orthogonalization and scale factor.
  """

  reduction_axis: Sequence[int] | int = 0
  output_axis: Sequence[int] | int = 1
  component_splits: tuple[int, ...] | None = None


WeightDimNumOrFn = MuonDimensionNumbers | base.Params | Callable[[base.Params], base.Params | None]


_is_weight_dim_nums = lambda x: isinstance(x, MuonDimensionNumbers)


def _normalize_axes(x: jax.Array, dim_nums: MuonDimensionNumbers) -> tuple[tuple[int, ...], tuple[int, ...]]:
  """Normalize axes in dimension numbers to two tuples of non-negative ints."""
  if isinstance(dim_nums.reduction_axis, int):
    dim_nums = dim_nums._replace(reduction_axis=(dim_nums.reduction_axis,))
  reduction_axes = tuple(ax % x.ndim for ax in dim_nums.reduction_axis)

  if isinstance(dim_nums.output_axis, int):
    dim_nums = dim_nums._replace(output_axis=(dim_nums.output_axis,))
  output_axes = tuple(ax % x.ndim for ax in dim_nums.output_axis)
  return reduction_axes, output_axes


def _compute_muon_reshape(x: jax.Array, dim_nums: MuonDimensionNumbers) -> tuple[ReshapeFn, ReshapeFn]:
  """Compute the reshape and inverse functions for an array from a spec."""
  if x.ndim < 2:
    raise ValueError("Muon optimized parameters must have rank >= 2, got" f" {x.ndim=}")
  reduction_axes, output_axes = _normalize_axes(x, dim_nums)
  if set(reduction_axes) & set(output_axes):
    raise ValueError(
        "Normalized reduction axes and output axes must be"
        f" disjoint, got {reduction_axes} and {output_axes}."
        f" Originally {dim_nums=} and {x.shape=}"
    )
  batch_axes = tuple(sorted(set(range(x.ndim)) - set(reduction_axes) - set(output_axes)))
  transpose = batch_axes + reduction_axes + output_axes
  inv_transpose = tuple(sorted(range(x.ndim), key=lambda i: transpose[i]))
  axes2shape = lambda axes: tuple(x.shape[ax] for ax in axes)
  # Reshape to (batch, reduction, output) to match the (reduction, output)
  # structure of the original muon for 2D weights.
  flat_shape = (
      math.prod(axes2shape(batch_axes)),
      math.prod(axes2shape(reduction_axes)),
      math.prod(axes2shape(output_axes)),
  )
  unflat_shape = axes2shape(batch_axes) + axes2shape(reduction_axes) + axes2shape(output_axes)
  reshape_fn = lambda x: x.transpose(transpose).reshape(flat_shape)
  inverse_fn = lambda x: x.reshape(unflat_shape).transpose(inv_transpose)
  return reshape_fn, inverse_fn


def _get_shape_products(x: jax.Array, dim_nums: MuonDimensionNumbers) -> tuple[float, float]:
  reduction_axes, output_axes = _normalize_axes(x, dim_nums)
  fan_in = math.prod(x.shape[ax] for ax in reduction_axes)
  fan_out = math.prod(x.shape[ax] for ax in output_axes)
  return fan_in, fan_out


def _build_component_scale_tensor(
    update: jax.Array,
    dim_nums: MuonDimensionNumbers,
    scale_fn: Callable[[float, float], float],
) -> jax.Array:
  """Build a broadcast-compatible scale tensor for component_splits.

  Each component region along the output axis gets its own scale factor,
  computed from the per-component fan values (where batch dims are merged
  into the output, matching Megatron's per-component treatment).
  """
  reduction_axes, output_axes = _normalize_axes(update, dim_nums)
  batch_axes = tuple(sorted(set(range(update.ndim)) - set(reduction_axes) - set(output_axes)))
  output_axis = output_axes[0]
  splits = dim_nums.component_splits

  fan_in = math.prod(update.shape[ax] for ax in reduction_axes)
  batch_prod = math.prod(update.shape[ax] for ax in batch_axes)

  # Build a 1D array of scales along the output axis.
  scales = []
  for s in splits:
    fan_out = batch_prod * s
    scale = scale_fn(fan_in, fan_out)
    scales.extend([scale] * s)

  # Reshape to broadcast: all dims are 1 except the output axis.
  shape = [1] * update.ndim
  shape[output_axis] = sum(splits)
  return jnp.asarray(scales, dtype=update.dtype).reshape(shape)


def _scale_update_for_width_transfer(update: jax.Array, dim_nums: MuonDimensionNumbers):
  """Apply width scaling from <https://github.com/KellerJordan/Muon>."""
  if getattr(dim_nums, "component_splits", None) is not None:
    return update * _build_component_scale_tensor(
        update, dim_nums, lambda fan_in, fan_out: math.sqrt(max(1, fan_out / fan_in))
    )
  fan_in, fan_out = _get_shape_products(update, dim_nums)
  scale = jnp.sqrt(jnp.maximum(1, fan_out / fan_in))
  # Cast scale to update's dtype to prevent JAX type promotion from bf16 to
  # fp32.  Megatron-LM fuses shape-scale and LR into a single scalar multiply
  # on the bf16 NS output (_update.mul_(-adjusted_lr)), keeping the update in
  # bf16 until it is added to fp32 params.  We match that by preserving the
  # update dtype here so the downstream chain (add_decayed_weights,
  # scale_by_learning_rate) also stays in bf16.
  return update * jnp.asarray(scale, dtype=update.dtype)


def _scale_update_for_consistent_rms(
    update: jax.Array, dim_nums: MuonDimensionNumbers, consistent_rms: jax.typing.ArrayLike
):
  """Apply consistent RMS scaling from <https://arxiv.org/abs/2502.16982>."""
  if getattr(dim_nums, "component_splits", None) is not None:
    return update * _build_component_scale_tensor(
        update, dim_nums, lambda fan_in, fan_out: (math.sqrt(max(fan_in, fan_out)) * float(consistent_rms))
    )
  fan_in, fan_out = _get_shape_products(update, dim_nums)
  scale = jnp.sqrt(jnp.maximum(fan_in, fan_out)) * consistent_rms
  # Keep update in its original dtype (bf16) to match Megatron-LM's behavior
  # where LR * shape_scale is applied as a single bf16 scalar multiply.
  return update * jnp.asarray(scale, dtype=update.dtype)


def scale_by_shape(
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
) -> base.GradientTransformation:
  """Scale updates by factors derived from parameter shape.

  Args:
    weight_dimension_numbers: An optional tree with the same structure as the
      params of `MuonDimensionNumbers`s, specifying how to reshape the
      parameters before and after the orthogonalization OR a callable returning
      such a tree. None implies that all parameters are 2D matrices.
    consistent_rms: An optional float to activate consistent RMS scaling.
      If float, scales updates by `sqrt(max(fan_in, fan_out)) * consistent_rms`.
      If None, uses width scaling `sqrt(max(1, fan_out / fan_in))`.

  Returns:
    A `GradientTransformation` object.
  """

  def update_fn(updates, state, params=None):
    del params
    if callable(weight_dimension_numbers):
      # Populate weight_dim_nums if it's a callable. Use updates instead of
      # actual params since only shapes matter and params may not be provided.
      resolved_weight_dim_nums = weight_dimension_numbers(updates)
    else:
      resolved_weight_dim_nums = weight_dimension_numbers

    if consistent_rms is not None:
      scaling_fn = functools.partial(_scale_update_for_consistent_rms, consistent_rms=consistent_rms)
    else:
      scaling_fn = _scale_update_for_width_transfer

    scaled_updates = jax.tree.map(
        scaling_fn,
        updates,
        resolved_weight_dim_nums,
        is_leaf=_is_weight_dim_nums,
    )
    return scaled_updates, state

  # Use the standard empty_state initializer, as this transform is stateless
  return base.GradientTransformation(base.init_empty_state, update_fn)


def _aol_first_newton_schulz_iteration(
    x: jax.Array,
    coeffs: jax.Array,
    eps: jax.typing.ArrayLike = 1e-8,
) -> jax.Array:
  """'Almost Orthogonal Layer' Preconditioning with Newton-Schulz iteration."""
  # Implements the first Newton-Schulz step with AOL preconditioning
  # which allows for better orthogonalization performance.
  a = x @ x.T.conj()
  rescaling = jnp.clip(jnp.abs(a).sum(axis=-1), min=eps)
  s = jnp.expand_dims(jax.lax.rsqrt(rescaling), -1)
  x, a = x * s, a * s * s.transpose(-1, -2)
  b = coeffs[1] * a + coeffs[2] * a @ a
  return coeffs[0] * x + b @ x


def _schatten_first_newton_schulz_iteration(
    x: jax.Array,
    coeffs: jax.Array,
    eps: jax.typing.ArrayLike = 1e-8,
) -> jax.Array:
  """Schatten-4 Preconditioning with Newton-Schulz iteration."""
  # Implements the first Newton-Schulz step with Schatten-4 norm
  # preconditioning which allows for better orthogonalization performance.
  a = x @ x.T
  rescaling = jnp.clip(jnp.linalg.norm(a, ord="fro", axis=(-2, -1)), min=eps)
  s = jnp.expand_dims(jax.lax.rsqrt(rescaling), (0, -1))
  x, a = x * s, a * s**2
  b = coeffs[1] * a + coeffs[2] * a @ a
  return coeffs[0] * x + b @ x


def _base_newton_schulz_iteration(x: jax.Array, coeffs: jax.Array) -> jax.Array:
  # Implements Newton-Schulz step f(X) = c_0 X + c_1 (XX^T)X + c_2 (XX^T)^2X,
  # with quintic form f(X) = c_0 X + (c_1 A + c_2 AA)X, where A = XX^T.
  # The NS step has the property f(X) = f(X^T)^T. That is, we can get equivalent
  # result by transposing input and output. In particular, we may transpose X
  # when rows > cols for efficiency.
  a = x @ x.T.conj()
  b = coeffs[1] * a + coeffs[2] * a @ a
  return coeffs[0] * x + b @ x


_newton_schulz_iterator = _base_newton_schulz_iteration  # backwards compat


def _aol_ns_iterator(i, x, coeffs):
  # Modified first step using AOL rescaling
  return jax.lax.cond(
      i == 0,
      lambda x: _aol_first_newton_schulz_iteration(x, coeffs),
      lambda x: _base_newton_schulz_iteration(x, coeffs),
      x,
  )


def _schatten_ns_iterator(i, x, coeffs):
  # Modified first step using Schatten-4 norm rescaling
  return jax.lax.cond(
      i == 0,
      lambda x: _schatten_first_newton_schulz_iteration(x, coeffs),
      lambda x: _base_newton_schulz_iteration(x, coeffs),
      x,
  )


def _base_ns_iterator(i, x, coeffs):
  del i
  return _base_newton_schulz_iteration(x, coeffs)


def _orthogonalize_components(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike,
    preconditioning: str,
    eps: jax.typing.ArrayLike,
    dimension_numbers: MuonDimensionNumbers,
) -> jax.Array:
  """Orthogonalize with per-component splitting on the output axis.

  Matches Megatron's per-component treatment of MLA projections: splits the
  output axis into semantic components (e.g., nope/rope), merges batch axes
  (heads) into each component, and applies NS independently to each resulting
  2D matrix.  Each component sees ``(reduction, batch * comp_size)`` where
  ``comp_size`` is the component's slice of the output axis.

  Args:
    x: The weight/gradient tensor with shape including reduction, batch (head),
      and output axes.
    ns_coeffs: Newton-Schulz coefficients.
    ns_steps: Number of NS iterations.
    preconditioning: Preconditioning method.
    eps: Numerical stability epsilon.
    dimension_numbers: MuonDimensionNumbers with component_splits set.

  Returns:
    Orthogonalized tensor with the same shape as the input.
  """
  reduction_axes, output_axes = _normalize_axes(x, dimension_numbers)
  batch_axes = tuple(sorted(set(range(x.ndim)) - set(reduction_axes) - set(output_axes)))
  splits = dimension_numbers.component_splits

  if len(output_axes) != 1:
    raise ValueError(f"component_splits requires exactly 1 output axis, got {output_axes}")
  output_axis = output_axes[0]

  if sum(splits) != x.shape[output_axis]:
    raise ValueError(f"component_splits {splits} must sum to output axis size " f"{x.shape[output_axis]}")

  # Transpose to (reduction..., batch..., output) canonical order.
  perm = tuple(reduction_axes) + tuple(batch_axes) + (output_axis,)
  x_t = jnp.transpose(x, perm)

  n_reduction = len(reduction_axes)
  n_batch = len(batch_axes)
  reduction_size = math.prod(x_t.shape[:n_reduction])
  batch_size = math.prod(x_t.shape[n_reduction : n_reduction + n_batch])

  # Split along the last axis (output) into components.
  split_indices = list(itertools.accumulate(splits[:-1]))
  components = jnp.split(x_t, split_indices, axis=-1)

  # Process each component: merge batch into output → 2D → NS → reshape back.
  results = []
  for comp in components:
    comp_size = comp.shape[-1]
    # Reshape to 2D: (reduction_prod, batch_prod * comp_size).
    comp_2d = comp.reshape(reduction_size, batch_size * comp_size)
    # Apply NS as a standard 2D matrix (no component_splits in the recursive
    # call since MuonDimensionNumbers(0, 1) defaults to None).
    comp_ortho = orthogonalize_via_newton_schulz(comp_2d, ns_coeffs, ns_steps, preconditioning, eps)
    # Reshape back to (reduction..., batch..., comp_size).
    comp_back = comp_ortho.reshape(x_t.shape[: n_reduction + n_batch] + (comp_size,))
    results.append(comp_back)

  # Concatenate along output axis and inverse transpose.
  result_t = jnp.concatenate(results, axis=-1)
  inv_perm = tuple(sorted(range(len(perm)), key=lambda i: perm[i]))
  return jnp.transpose(result_t, inv_perm)


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike = 5,
    preconditioning: Literal["frobenius", "spectral", "aol", "schatten"] = "frobenius",
    eps: jax.typing.ArrayLike = 1e-8,
    dimension_numbers: MuonDimensionNumbers | None = None,
) -> jax.Array:
  r"""Orthogonalize via Newton-Schulz iteration.

  We opt to use a quintic iteration whose coefficients are selected to maximize
  the slope at zero. For the purpose of minimizing steps, it turns out to be
  empirically effective to keep increasing the slope at zero even beyond the
  point where the iteration no longer converges all the way to one everywhere
  on the interval. This iteration therefore does not produce UV^T but rather
  something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
  which turns out not to hurt model performance at all relative to UV^T, where
  USV^T = G is the SVD.

  Args:
    x: A matrix to orthogonalize.
    ns_coeffs: Coefficients for the Newton-schulz iterators.
      Must have shape (n, 3) where n is the number of iterations.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a 2D array.
    preconditioning: Which preconditioning method to use.
    eps: Term added to denominators to improve numerical stability.
    dimension_numbers: Optional spec for reshaping a tensor before and after the
      orthogonalization, to support non-2D parameters.

  Returns:
    The orthogonalized matrix.
  """
  # Dispatch to component-split path before any other logic.
  if isinstance(dimension_numbers, MuonDimensionNumbers) and dimension_numbers.component_splits is not None:
    return _orthogonalize_components(x, ns_coeffs, ns_steps, preconditioning, eps, dimension_numbers)

  if x.ndim != 2 and not isinstance(dimension_numbers, MuonDimensionNumbers):
    raise ValueError(
        f"Input must have shape (m, n) or weight dimension numbers must be"
        f" provided. Got shape={x.shape} and {dimension_numbers=}."
    )
  if x.ndim == 2:
    dimension_numbers = MuonDimensionNumbers(reduction_axis=0, output_axis=1)
  if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
    raise ValueError("Newton-Schulz coefficients must have shape (3,) or" f" (n, 3), got {ns_coeffs.shape}")

  def _orthogonalize(x):
    transposed = False
    if x.shape[0] > x.shape[1]:
      x = x.T
      transposed = True

    ns_iterators = {
        "frobenius": _base_ns_iterator,
        "spectral": _base_ns_iterator,
        "aol": _aol_ns_iterator,
        "schatten": _schatten_ns_iterator,
    }
    if preconditioning not in _PRECONDITIONINGS:
      raise ValueError(f"Unknown preconditioning {preconditioning}")
    _ns_iterator = ns_iterators[preconditioning]

    if preconditioning == "frobenius":
      x /= jnp.linalg.norm(x, ord="fro") + eps
    elif preconditioning == "spectral":
      x /= jnp.linalg.norm(x, ord=2) + eps
    else:
      pass

    ns_coeffs_ = ns_coeffs.astype(x.dtype)

    if ns_coeffs_.ndim == 1:
      x = jax.lax.fori_loop(
          0, ns_steps, lambda i, x: _ns_iterator(i, x, ns_coeffs_), x, unroll=True
      )  # Unroll to ensure efficient composition with jax.vmap.
    else:

      def _scan_body(carry, coeffs_step):
        i, x = carry
        x_new = _ns_iterator(i, x, coeffs_step)
        return (i + 1, x_new), None

      init_carry = (jnp.asarray(0, dtype=jnp.int32), x)
      (_, x), _ = jax.lax.scan(_scan_body, init_carry, ns_coeffs_)

    if transposed:
      x = x.T
    return x

  reshape_fn, inverse_fn = _compute_muon_reshape(x, dimension_numbers)
  return inverse_fn(jax.vmap(_orthogonalize)(reshape_fn(x)))


class MuonState(NamedTuple):
  """State for the Muon algorithm."""

  count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  ns_coeffs: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.


def _batch_orthogonalize_tree(
    updates: base.Updates,
    weight_dim_nums: base.Params,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike,
    preconditioning: str,
    eps: jax.typing.ArrayLike,
    batch_update_size: int | None,
) -> base.Updates:
  """Orthogonalize updates by grouping same-shape matrices and batching NS.

  Instead of running Newton-Schulz on each parameter independently, groups
  parameters by their reshaped 2D matrix shape (m, n), stacks them along the
  batch dimension, and runs NS on the combined batch. This amortizes overhead
  for many small matrices. Results are numerically identical to unbatched mode.

  Args:
    updates: PyTree of gradient updates (already momentum-processed).
    weight_dim_nums: PyTree of MuonDimensionNumbers (same structure as updates).
    ns_coeffs: Newton-Schulz coefficients array.
    ns_steps: Number of NS iterations.
    preconditioning: Preconditioning method.
    eps: Numerical stability epsilon.
    batch_update_size: If set, chunk each shape group into batches of this size.

  Returns:
    PyTree of orthogonalized updates (same structure as input).
  """
  flat_updates, treedef = jax.tree.flatten(updates, is_leaf=_is_weight_dim_nums)

  # Handle weight_dim_nums=None: create matching default dim nums for 2D params.
  if weight_dim_nums is None:
    flat_dim_nums = [None] * len(flat_updates)
  else:
    flat_dim_nums, _ = jax.tree.flatten(weight_dim_nums, is_leaf=_is_weight_dim_nums)

  # For each leaf, reshape to (batch, m, n) and record metadata for regrouping.
  # Group key is (m, n) after ensuring m <= n.
  groups: dict[tuple[int, int], list] = {}
  results = [None] * len(flat_updates)

  for i, (leaf, dim_num) in enumerate(zip(flat_updates, flat_dim_nums)):
    if leaf.ndim < 2:
      # Scalar/1D params shouldn't reach here (they go to Adam), but be safe.
      results[i] = leaf
      continue

    # Component-split params can't be batched with others; process individually.
    if dim_num is not None and getattr(dim_num, "component_splits", None) is not None:
      results[i] = orthogonalize_via_newton_schulz(leaf, ns_coeffs, ns_steps, preconditioning, eps, dim_num)
      continue

    if leaf.ndim == 2:
      dn = MuonDimensionNumbers(reduction_axis=0, output_axis=1)
    else:
      dn = dim_num

    reshape_fn, inverse_fn = _compute_muon_reshape(leaf, dn)
    reshaped = reshape_fn(leaf)  # (batch, m, n)
    _, m, n = reshaped.shape
    transposed = m > n
    if transposed:
      reshaped = reshaped.transpose(0, 2, 1)
      m, n = n, m

    key = (m, n)
    if key not in groups:
      groups[key] = []
    groups[key].append(
        {
            "idx": i,
            "reshaped": reshaped,
            "inverse_fn": inverse_fn,
            "transposed": transposed,
            "batch_size": reshaped.shape[0],
        }
    )

  # Process each shape group.
  for (m, n), items in groups.items():
    # Concatenate all items along batch dim.
    all_reshaped = jnp.concatenate([item["reshaped"] for item in items], axis=0)

    # Chunk if batch_update_size is set.
    total_batch = all_reshaped.shape[0]
    if batch_update_size is not None and total_batch > batch_update_size:
      chunks = []
      for start in range(0, total_batch, batch_update_size):
        end = min(start + batch_update_size, total_batch)
        chunk = all_reshaped[start:end]
        chunks.append(
            orthogonalize_via_newton_schulz(
                chunk,
                ns_coeffs,
                ns_steps,
                preconditioning,
                eps,
                MuonDimensionNumbers(reduction_axis=1, output_axis=2),
            )
        )
      all_ortho = jnp.concatenate(chunks, axis=0)
    else:
      all_ortho = orthogonalize_via_newton_schulz(
          all_reshaped,
          ns_coeffs,
          ns_steps,
          preconditioning,
          eps,
          MuonDimensionNumbers(reduction_axis=1, output_axis=2),
      )

    # Split back to individual items.
    batch_sizes = [item["batch_size"] for item in items]
    split_indices = list(itertools.accumulate(batch_sizes[:-1]))
    split_results = jnp.split(all_ortho, split_indices, axis=0)

    for item, ortho in zip(items, split_results):
      if item["transposed"]:
        ortho = ortho.transpose(0, 2, 1)
      results[item["idx"]] = item["inverse_fn"](ortho)

  return treedef.unflatten(results)


def scale_by_muon(
    ns_coeffs: Union[
        tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
        tuple[
            tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
            ...,
        ],
    ] = _DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-7,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    ns_dtype: Optional[jax.typing.DTypeLike] = jnp.bfloat16,
    *,
    nesterov: bool = True,
    nesterov_style: Literal["ema", "sgd"] = "ema",
    adaptive: bool = False,
    preconditioning: Literal["frobenius", "spectral", "aol", "schatten"] = "frobenius",
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    batch_update: bool = False,
    batch_update_size: int | None = None,
) -> base.GradientTransformation:
  r"""Rescale updates according to the Muon algorithm.

  Muon is a variant of Shampoo that uses the Newton-schulz method to
  orthogonalize the momentum accumulated by the optimizer. Mathematically, it
  does steepest descent under the Schatten-p norm, for some large p. With
  p=infty, it is equivalent to Shampoo without accumulation, or steepest
  descent under the Spectral norm.

  Args:
    ns_coeffs: Coefficients for the Newton-schulz method.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    eps: Term added to denominators to improve numerical stability.
      Default 1e-7 matches Megatron-LM's hardcoded NS normalization epsilon.
    mu_dtype: Data type of the momentum accumulator.
    ns_dtype: Data type for Newton-Schulz orthogonalization. Input is cast to
      this dtype before NS iterations. The output stays in ns_dtype and flows
      through subsequent transforms; JAX promotes to fp32 at param update.
      This matches Megatron-LM where the bf16 NS output is scaled by LR in
      bf16, then added to fp32 params. Default ``jnp.bfloat16``. Set to
      ``None`` to skip casting (use the momentum dtype as-is).
    nesterov: Whether to use Nesterov momentum.
    nesterov_style: Style of momentum accumulation. 'ema' uses the optax
      default (EMA accumulation with bias correction). 'sgd' uses classic
      SGD-style momentum (buf = beta*buf + grad) with Nesterov look-ahead
      (g = grad + beta*buf) and no bias correction, matching Megatron-LM.
    adaptive: Whether to scale the updates by the dual norm of the
      original updates. See <https://arxiv.org/abs/2409.20325>
    preconditioning: What type of preconditioning to use before NS iterations.
      Available options are:
      - 'frobenius' (default): Use Frobenius rescaling before NS.
      - 'spectral' : Use Spectral norm rescaling before NS.
      - 'aol': Use AOL rescaling to improve orthogonality.
      - 'schatten': Use the Schatten-4 norm for rescaling.
    weight_dimension_numbers: An optional tree with the same structure as the
      params of `MuonDimensionNumbers`s, specifying how to reshape the
      parameters before and after the orthogonalization OR a callable returning
      such a tree. None implies that all parameters are 2D matrices.
    batch_update: If True, group parameters by their reshaped 2D matrix shape
      and stack them along the batch dimension before running Newton-Schulz.
      This amortizes overhead for many small matrices. Results are numerically
      identical to unbatched mode.
    batch_update_size: If set (and batch_update is True), chunk each shape
      group into batches of at most this size before running Newton-Schulz.

  Returns:
    A `GradientTransformation` object.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024

    Liu et al., `Muon is Scalable for LLM Training`,
    <https://arxiv.org/abs/2502.16982>`_, 2025

    Boissin et al., `Turbo-Muon: Accelerating Orthogonality-Based
    Optimization with Pre-Conditioning`,
    <https://arxiv.org/abs/2512.04632>`_, 2025

    Ahn et al., `Dion: Distributed Orthonormalized Updates`,
    <https://arxiv.org/abs/2504.05295>`_, 2025

    Grishina et al., `Accelerating Newton-Schulz Iteration for Orthogonalization
    via Chebyshev-type Polynomials`,
    <https://arxiv.org/abs/2506.10935>`_, 2025

    Amsel et al., `The Polar Express: Optimal Matrix Sign Methods and Their
    Application to the Muon Algorithm`,
    <https://arxiv.org/pdf/2505.16932>`, 2025
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
    ns_coeffs_ = jnp.asarray(ns_coeffs)

    if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
      raise ValueError(f"ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}")
    if ns_coeffs_.ndim == 2:
      if not ns_coeffs_.shape[0] <= ns_steps:
        raise ValueError(f"Not enough coeffs to perform {ns_steps} steps")
      ns_coeffs_ = ns_coeffs_[-ns_steps:]

    return MuonState(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        ns_coeffs=ns_coeffs_,
    )

  def update_fn(updates, state, params=None):
    del params
    # TODO(rdyro): extend to _masking._mask_callable
    if callable(weight_dimension_numbers):
      # Populate weight_dim_nums if it's a callable. Use updates instead of
      # actual params since only shapes matter and params may not be provided.
      resolved_weight_dim_nums = weight_dimension_numbers(updates)
    else:
      resolved_weight_dim_nums = weight_dimension_numbers

    if nesterov_style == "sgd":
      # Classic SGD-style momentum (Megatron-LM):
      #   buf = beta * buf + grad
      #   g   = grad + beta * buf    (Nesterov look-ahead, no bias correction)
      mu = jax.tree.map(lambda g, m: beta * m + g, updates, state.mu)
      if nesterov:
        mu_hat = jax.tree.map(lambda g, m: g + beta * m, updates, mu)
      else:
        mu_hat = mu
    else:
      # Default optax EMA-style momentum with bias correction
      mu = optax.tree.update_moment(updates, state.mu, beta, 1)
      count_inc = numerics.safe_increment(state.count)
      if nesterov:
        mu_hat = jax.tree.map(
            lambda m, g: beta * m + (1 - beta) * g,
            optax.tree.bias_correction(mu, beta, numerics.safe_increment(count_inc)),
            optax.tree.bias_correction(updates, beta, count_inc),
        )
      else:
        mu_hat = optax.tree.bias_correction(mu, beta, count_inc)

    count_inc = numerics.safe_increment(state.count)
    # Cast to ns_dtype before Newton-Schulz orthogonalization.
    # Megatron-LM casts to bfloat16 here (g.bfloat16()) and keeps the update
    # in bf16 through LR scaling; the cast back to fp32 happens implicitly
    # when the update is added to fp32 parameters (param.data.add_(update)).
    # We match this by NOT casting back here — the bf16 update flows through
    # the rest of the optax chain (scale_by_shape, add_decayed_weights,
    # scale_by_learning_rate) and JAX promotes to fp32 at param addition.
    if ns_dtype is not None:
      ns_input = optax.tree.cast(mu_hat, ns_dtype)
    else:
      ns_input = mu_hat
    # Apply Newton-schulz orthogonalization.
    if batch_update:
      updates = _batch_orthogonalize_tree(
          ns_input, resolved_weight_dim_nums, state.ns_coeffs, ns_steps, preconditioning, eps, batch_update_size
      )
    else:
      updates = jax.tree.map(
          lambda x, dim_num: orthogonalize_via_newton_schulz(x, state.ns_coeffs, ns_steps, preconditioning, eps, dim_num),
          ns_input,
          resolved_weight_dim_nums,
          is_leaf=_is_weight_dim_nums,
      )
    if adaptive:
      # Scale the orthogonalized updates by the dual norm of the original
      # updates. See https://arxiv.org/abs/2409.20325 for the derivation.
      updates = jax.tree.map(lambda x, y: jnp.sum(x.conj() * y) * y, mu_hat, updates)

    mu = optax.tree.cast(mu, mu_dtype)
    return updates, MuonState(
        count=count_inc,
        mu=mu,
        ns_coeffs=state.ns_coeffs,
    )

  return base.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: Union[
        tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
        tuple[
            tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
            ...,
        ],
        str,
    ] = _DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-7,
    weight_decay: jax.typing.ArrayLike = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    ns_dtype: Optional[jax.typing.DTypeLike] = jnp.bfloat16,
    *,
    nesterov: bool = True,
    nesterov_style: Literal["ema", "sgd"] = "ema",
    adaptive: bool = False,
    preconditioning: Literal["frobenius", "spectral", "aol", "schatten"] = "frobenius",
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps: jax.typing.ArrayLike | None = None,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_nesterov: bool = False,
    adam_weight_decay: jax.typing.ArrayLike = 0.0,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    adam_weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    muon_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
    batch_update: bool = False,
    batch_update_size: int | None = None,
) -> base.GradientTransformation:
  r"""Muon: Momentum Orthogonalized by Newton-schulz.

  Muon is a variant of Shampoo that uses the Newton-schulz method to
  orthogonalize the momentum accumulated by the optimizer. Mathematically, it
  does steepest descent under the Schatten-p norm, for some large p. With
  p=infty, it is equivalent to Shampoo without accumulation, or steepest
  descent under the Spectral norm.

  Note that Muon is currently only defined for 2D parameters, i.e. matrices.
  This is because the Newton-Schulz iterator expects a matrix as input.
  The non-2D parameters are instead passed through an AdamW optimizer
  (using a weight decay of 0 as default).

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    ns_coeffs: Coefficients for the Newton-schulz method (can be a string
      indicator for a preset). Existing presets: `muon`, `dion`.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    eps: Term added to the denominator to improve numerical stability.
      Default 1e-7 matches Megatron-LM's hardcoded NS normalization epsilon.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip.
    mu_dtype: Data type of the momentum accumulator.
    ns_dtype: Data type for Newton-Schulz orthogonalization. Input is cast to
      this dtype before NS iterations. The output stays in ns_dtype through
      LR scaling; JAX promotes to fp32 at param update, matching Megatron-LM.
      Default ``jnp.bfloat16``. Set to ``None`` to skip casting.
    nesterov: Whether to use Nesterov momentum.
    nesterov_style: Style of momentum accumulation. 'ema' uses the optax
      default (EMA accumulation with bias correction). 'sgd' uses classic
      SGD-style momentum (buf = beta*buf + grad) with Nesterov look-ahead
      (g = grad + beta*buf) and no bias correction, matching Megatron-LM.
    adaptive: Whether to scale the updates by the dual norm of the
      original updates. See <https://arxiv.org/abs/2409.20325>
    preconditioning: What type of preconditioning to use before NS iterations.
      Available options are:
      - 'frobenius' (default): Use Frobenius rescaling before NS:
        safe, standard, but degrades orthogonalization quality when using
        less than 5 NS steps.
      - 'spectral' : Use Spectral norm rescaling before NS:
        much more computationally intensive, but better orthogonalization
        quality.
      - 'aol': Use AOL rescalings to improve orthogonality with little to
        no overhead, usually allows the user to remove one iterative NS step.
        See <https://arxiv.org/abs/2512.04632>.
      - 'schatten': Use the Schatten-4 norm for rescaling,
        allows for better performance with little to no extra cost.
        See <https://arxiv.org/abs/2506.10935>.
    adam_b1: Exponential decay rate for Adam's first moment estimates.
    adam_b2: Exponential decay rate for Adam's second moment estimates.
    adam_nesterov: Whether to use Nesterov momentum in the Adam partition.
      Default ``False`` to match Megatron-LM's standard Adam (no Nesterov).
    adam_eps_root: Epsilon to stabilize division in Adam, square root version.
    adam_weight_decay: Weight decay factor for Adam.
    adam_learning_rate: Auxiliary learning rate for the Adam optimizer.
      If `None`, the learning rate for Adam defaults to the same as Muon.
    adam_weight_decay_mask: A tree with same structure as (or a prefix of) the
      params PyTree, or a Callable that returns such a pytree given the params.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply Adam weight decay to, and `False` for those you want to skip.
    muon_weight_dimension_numbers: An optional tree of `MuonDimensionNumbers`s,
      specifying how to reshape the parameters for orthogonalization otherwise
      muon parameters are assumed to be 2D matrices. A `None` value indicates
      that the parameter is not a muon parameter and will be optimized with
      Adam. A callable takes as input the params and returns a possibly masked
      pytree of specs, similar to `weight_decay_mask`. If not provided, muon is
      applied to all 2D parameters.
    consistent_rms: An optional float to activate consistent RMS scaling.
      Scales updates by `sqrt(max(fan_in, fan_out)) * consistent_rms` to make
      root mean square (RMS) shape-independent, like AdamW. `0.2` is recommended
      to match AdamW's empirical RMS. See <https://arxiv.org/abs/2502.16982>.
      If `None`, uses width scaling `sqrt(max(1, fan_out / fan_in))`.
    batch_update: If True, group same-shape Muon parameters and run
      Newton-Schulz on stacked batches for efficiency. Numerically identical
      to unbatched mode.
    batch_update_size: Maximum batch size per NS call when batch_update is True.
      If None, all same-shape parameters are stacked into one batch.

  Returns:
    The corresponding `GradientTransformation`.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024

    Liu et al., `Muon is Scalable for LLM Training`,
    <https://arxiv.org/abs/2502.16982>`_, 2025

    Boissin et al., `Turbo-Muon: Accelerating Orthogonality-Based
    Optimization with Pre-Conditioning`,
    <https://arxiv.org/abs/2512.04632>`_, 2025

    Ahn et al., `Dion: Distributed Orthonormalized Updates`,
    <https://arxiv.org/abs/2504.05295>`_, 2025

    Grishina et al., `Accelerating Newton-Schulz Iteration for Orthogonalization
    via Chebyshev-type Polynomials`,
    <https://arxiv.org/abs/2506.10935>`_, 2025

    Amsel et al., `The Polar Express: Optimal Matrix Sign Methods and Their
    Application to the Muon Algorithm`,
    <https://arxiv.org/pdf/2505.16932>`, 2025
  """

  if adam_learning_rate is None:
    adam_learning_rate = learning_rate

  if isinstance(ns_coeffs, str):
    if ns_coeffs not in _NS_COEFFS_PRESET_DICT:
      raise ValueError(f"Unknown ns_coeff preset string: {ns_coeffs}")
    ns_coeffs_ = _NS_COEFFS_PRESET_DICT[ns_coeffs]
  else:
    ns_coeffs_ = ns_coeffs

  # None at root indicates the default 2D rule.
  if muon_weight_dimension_numbers is None:
    param_labels = lambda params: jax.tree.map(lambda x: "muon" if x.ndim == 2 else "adam", params)
    muon_weight_dimension_numbers = MuonDimensionNumbers()
  else:

    def param_labels(params):
      dim_nums = (
          muon_weight_dimension_numbers(params)
          if callable(muon_weight_dimension_numbers)
          else muon_weight_dimension_numbers
      )
      populate_subtree_ = lambda dim_num, x: jax.tree.map(lambda y: "muon" if dim_num is not None else "adam", x)
      # Dimension numbers come first since they can be a prefix mask.
      return jax.tree.map(populate_subtree_, dim_nums, params, is_leaf=lambda x: x is None or _is_weight_dim_nums(x))

  # We need to normalize the dimension numbers because they have to match the
  # tree structure of the masked muon state tree (see `combine.partition`).
  def muon_weight_dim_nums_fn(params):
    # if muon_weight_dimension_numbers is None:
    #   return None
    # Normalize the dimension numbers for `combine.partition`.
    # Insert MaskedNode() where muon state will be masked out.
    dim_nums = (
        muon_weight_dimension_numbers(params)
        if callable(muon_weight_dimension_numbers)
        else muon_weight_dimension_numbers
    )
    mask = jax.tree.map(lambda label: label == "muon", param_labels(params))
    is_leaf = lambda x: (x is None or _is_weight_dim_nums(x) or isinstance(x, _masking.MaskedNode))
    populate_subtree_ = lambda dim_nums, submask: jax.tree.map(
        lambda m: dim_nums if m else _masking.MaskedNode(), submask
    )
    return jax.tree.map(populate_subtree_, dim_nums, mask, is_leaf=is_leaf)

  return combine.partition(
      transforms={
          "muon": combine.chain(
              scale_by_muon(
                  ns_coeffs=ns_coeffs_,
                  ns_steps=ns_steps,
                  beta=beta,
                  eps=eps,
                  mu_dtype=mu_dtype,
                  ns_dtype=ns_dtype,
                  nesterov=nesterov,
                  nesterov_style=nesterov_style,
                  adaptive=adaptive,
                  preconditioning=preconditioning,
                  weight_dimension_numbers=muon_weight_dim_nums_fn,
                  batch_update=batch_update,
                  batch_update_size=batch_update_size,
              ),
              scale_by_shape(
                  weight_dimension_numbers=muon_weight_dim_nums_fn,
                  consistent_rms=consistent_rms,
              ),
              transform.add_decayed_weights(weight_decay, weight_decay_mask),
              transform.scale_by_learning_rate(learning_rate),
          ),
          "adam": alias.adamw(
              learning_rate=adam_learning_rate,
              b1=adam_b1,
              b2=adam_b2,
              eps=adam_eps if adam_eps is not None else eps,
              eps_root=adam_eps_root,
              weight_decay=adam_weight_decay,
              mask=adam_weight_decay_mask,
              mu_dtype=mu_dtype,
              nesterov=adam_nesterov,
          ),
      },
      param_labels=param_labels,
  )
