# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MoE auxiliary helpers for the pre-training loop.

This module hosts the trainer-side helpers that:

* count MoE layers (incl. MTP);
* read and aggregate sown intermediates (load-balance loss, z-loss, expert
  counts, router stats) across heterogeneous decoder layouts (DeepSeek /
  LING2 / LING3, scan / unscan, with optional MTP);
* apply the loss-free balancing routed-bias updates back onto the params.

Design notes:

* Helpers fail loud (``RuntimeError``) when an expected sown path is missing.
  Silent zero-fallbacks have historically masked layout drifts where, e.g.,
  the LING3 scan layout sows under ``moe_layers/layers_{j}/...`` while the
  trainer was reading ``moe_layers/...`` and getting ``0``.
* Optional MTP paths use ``required=False`` to keep the historical
  log-and-skip behavior.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from maxtext.common.common_types import DecoderBlockType
from maxtext.layers.decoders import compute_ling3_scan_layout
from maxtext.layers.moe import expert_counts_to_bias_update
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils


# --- Stringly-typed param/intermediate path constants -----------------------

_MOE_BLOCK_NAMES = {
    DecoderBlockType.DEEPSEEK: "DeepSeekMoeBlock_0",
    DecoderBlockType.LING2: "mlp",
    DecoderBlockType.LING3: "mlp",
}
_GATE_BIAS_TAIL = ("MoeBlock_0", "gate", "bias")


def _moe_block_name(decoder_block) -> str | None:
  """Return the sub-module name that wraps ``MoeBlock_0`` for this decoder, or None."""
  return _MOE_BLOCK_NAMES.get(decoder_block)


# --- Layer counting ---------------------------------------------------------


def count_moe_layers(config) -> int:
  """Return the number of MoE layers (backbone + MTP) for this config."""
  moe_layer_freq = getattr(config, "moe_layer_freq", None)
  if moe_layer_freq:
    count = sum(moe_layer_freq[: config.num_decoder_layers])
  else:
    num_moe_candidates = config.num_decoder_layers - config.first_num_dense_layers
    if config.interleave_moe_layer_step > 1:
      num_moe_candidates = num_moe_candidates // config.interleave_moe_layer_step
    count = max(num_moe_candidates, 1)
  if getattr(config, "mtp_num_layers", 0) > 0 and config.num_experts > 1:
    count += config.mtp_num_layers
  return count


# --- Sown-intermediate helpers ---------------------------------------------


def _unwrap_sow(raw: Any) -> Any:
  """Flax/NNX ``sow()`` accumulates into a tuple; unwrap to the latest entry."""
  return raw[-1] if isinstance(raw, tuple) else raw


def _get_sown(
    intermediate_outputs,
    nested_key,
    *,
    context: str,
    required: bool = True,
    unwrap: bool = True,
):
  """Read a sown intermediate at ``nested_key``.

  Args:
    intermediate_outputs: Mutable intermediates pytree.
    nested_key: Sequence of keys describing the path inside the pytree.
    context: Free-form context string included in error messages so failures
      pinpoint the (decoder_block, scan_layers, key) combo that broke.
    required: When True (default) raise if the path is missing. When False
      return ``None`` instead — used for genuinely optional paths.
    unwrap: When True (default) strip the outer sow-tuple. Set False to keep
      the raw tuple (e.g. when a downstream caller wants ``counts[0]``).
  """
  if not maxtext_utils.has_nested_key(intermediate_outputs, nested_key):
    if required:
      raise RuntimeError(
          f"Expected sown intermediate at {'/'.join(nested_key)} but it is "
          f"missing ({context}). Possible causes: model layout drift "
          f"(sow path moved), the layer was never executed, or trainer-side "
          f"path assumptions are wrong."
      )
    return None
  value = maxtext_utils.get_nested_value(intermediate_outputs, nested_key)
  return _unwrap_sow(value) if unwrap else value


# --- Sum/mean collection across layers --------------------------------------


def _collect_ling3_scan_intermediate_sum(config, intermediate_outputs, key):
  """Read sown values across the LING3 scan layout (Phase 1b prefix + Phase 2 scan).

  Phase 1b prefix: intermediates/decoder/moe_layers_{i}/{key}            (i in range(num_moe_prefix))
  Phase 2 scan:    intermediates/decoder/moe_layers/layers_{j}/{key}     (j in range(interval))
  """
  ctx = f"decoder_block=LING3, scan_layers=True, key={key!r}"
  _, num_moe_prefix, _ = compute_ling3_scan_layout(config)
  values = [
      _get_sown(intermediate_outputs, ("intermediates", "decoder", f"moe_layers_{i}", key), context=ctx)
      for i in range(num_moe_prefix)
  ]
  interval = config.inhomogeneous_layer_cycle_interval
  values.extend(
      _get_sown(intermediate_outputs, ("intermediates", "decoder", "moe_layers", f"layers_{j}", key), context=ctx)
      for j in range(interval)
  )
  return values


def collect_moe_intermediate_sum(config, intermediate_outputs, key):
  """Collect and sum a sowed intermediate (e.g. ``moe_lb_loss``) across MoE layers.

  Handles all decoder block types and scan/unscan modes, mirroring the
  ``collect_moe_expert_counts`` collection logic. Raises ``RuntimeError`` if a
  path that *must* exist (per the decoder block / scan_layers / num_decoder_layers
  config) is missing.
  """
  ctx = f"decoder_block={config.decoder_block}, scan_layers={config.scan_layers}, key={key!r}"
  if config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.LING2, DecoderBlockType.LING3):
    if not config.scan_layers:
      num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
      values = [
          _get_sown(intermediate_outputs, ("intermediates", "decoder", f"moe_layers_{i}", key), context=ctx)
          for i in range(num_moe_layers)
      ]
    elif config.decoder_block == DecoderBlockType.LING3:
      values = _collect_ling3_scan_intermediate_sum(config, intermediate_outputs, key)
    else:
      # DeepSeek scan: a single scanned moe_layer; intermediates land at
      # decoder/moe_layers/{key} as a stacked tensor of shape (scan_length, ...).
      # LING2 + scan is rejected upstream in decoders.get_decoder_layers().
      values = [_get_sown(intermediate_outputs, ("intermediates", "decoder", "moe_layers", key), context=ctx)]
  else:
    # Mixtral, Llama4, GPT_OSS, Qwen3, etc.
    if config.scan_layers:
      values = [_get_sown(intermediate_outputs, ("intermediates", "decoder", "layers", key), context=ctx)]
    else:
      values = [
          _get_sown(intermediate_outputs, ("intermediates", "decoder", f"layers_{lyr}", key), context=ctx)
          for lyr in range(config.num_decoder_layers)
      ]
  # Each entry is either a scalar or a 1-D scan-stacked tensor; sum them all together.
  backbone_sum = sum(jnp.sum(jnp.asarray(v)) for v in values)

  # Collect from MTP block MoE layers (optional — MTP can be absent entirely).
  mtp_sum = 0.0
  if getattr(config, "mtp_num_layers", 0) > 0 and config.num_experts > 1:
    for k in range(1, config.mtp_num_layers + 1):
      mtp_sum += _get_sown(
          intermediate_outputs,
          ("intermediates", "mtp_block", f"mtp_layer_{k}", f"mtp_{k}_transformer_layer", key),
          context=ctx,
      )

  return backbone_sum + mtp_sum


def collect_moe_intermediate_mean(config, intermediate_outputs, key):
  """Collect and average a sowed intermediate value across all MoE layers."""
  total = collect_moe_intermediate_sum(config, intermediate_outputs, key)
  return total / max(count_moe_layers(config), 1)


# --- Routed-bias updates ----------------------------------------------------


def _moe_bias_target_path(*decoder_segments: str, moe_block_name: str) -> tuple[str, ...]:
  """Build the gate-bias param path under ``params/decoder/<segments>/{moe_block_name}/MoeBlock_0/gate/bias``.

  ``decoder_segments`` may be one segment (unscan / Phase 1b: ``moe_layers_{i}``)
  or two (Phase 2 scan: ``moe_layers``, ``layers_{j}``).
  """
  return ("params", "decoder", *decoder_segments, moe_block_name) + _GATE_BIAS_TAIL


def _try_update_bias(
    config,
    new_state,
    target_path,
    expert_counts,
    path_label,
    *,
    required=True,
    zero_mean_axis=-1,
    transpose_update=False,
):
  """Try to apply a single MoE gate bias update at ``target_path``.

  When ``required`` is True (default), raise if the target path is missing —
  this catches layout drift bugs. Optional MTP paths should pass
  ``required=False`` to keep the log-and-skip behavior.

  ``zero_mean_axis`` controls which axis ``update_state_param`` re-centers
  along when ``routed_bias_zero_mean_update=True``. For 1D bias of shape
  ``(num_experts,)`` the default of -1 is correct; for scan-stacked 2D bias of
  shape ``(num_experts, scan_length)`` callers must pass ``zero_mean_axis=0``.

  ``transpose_update`` transposes the bias-update tensor before it is added to
  the param. Needed when ``counts`` has shape ``(scan_length, num_experts)``
  (so ``expert_counts_to_bias_update`` averages along the correct axis) but
  the underlying bias param is shape ``(num_experts, scan_length)``.
  """
  if not maxtext_utils.has_nested_key(new_state.params, target_path):
    msg = (
        f"Routed-bias update target {'/'.join(target_path)} not found in params "
        f"(decoder_block={config.decoder_block}, scan_layers={config.scan_layers}, "
        f"label={path_label!r})."
    )
    if required:
      raise RuntimeError(msg + " This typically means trainer-side path assumptions disagree with the model layout.")
    max_logging.log("Skipping " + msg)
    return new_state
  counts = jnp.array(expert_counts[0])
  update_value = expert_counts_to_bias_update(counts, config.num_experts, config.routed_bias_update_rate)
  if transpose_update:
    update_value = update_value.T
  return maxtext_utils.update_state_param(
      new_state,
      target_path,
      update_value,
      zero_mean_update=config.routed_bias_zero_mean_update,
      zero_mean_axis=zero_mean_axis,
  )


def _update_unscan_moe_bias(config, new_state, moe_block_name, moe_expert_counts):
  """Shared unscan path: per-layer 1D bias updates for DeepSeek/LING2/LING3."""
  num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
  for i in range(num_moe_layers):
    if moe_expert_counts[i] is None:
      continue
    target_path = _moe_bias_target_path(f"moe_layers_{i}", moe_block_name=moe_block_name)
    new_state = _try_update_bias(config, new_state, target_path, moe_expert_counts[i], f"moe_layers_{i}")
  return new_state


def _update_deepseek_scan_bias(config, new_state, moe_block_name, moe_expert_counts):
  """DeepSeek scan: single stacked layer, bias shape (num_experts, scan_length).

  LING2 + scan is rejected upstream in decoders.get_decoder_layers().
  """
  target_path = _moe_bias_target_path("moe_layers", moe_block_name=moe_block_name)
  return _try_update_bias(
      config,
      new_state,
      target_path,
      moe_expert_counts,
      "moe_expert_counts",
      zero_mean_axis=0,
      transpose_update=True,
  )


def _update_ling3_scan_bias(config, new_state, moe_block_name, moe_expert_counts):
  """LING3 scan bias update — both phases of the cycle.

  ``moe_expert_counts`` is a dict produced by ``collect_moe_expert_counts``:
    counts["prefix"]: list of length num_moe_prefix, each (num_experts,)        — Phase 1b
    counts["scan"]:   list of length interval, each (scan_length, num_experts)  — Phase 2

  Bias param layout:
    params/decoder/moe_layers_{i}/.../bias                     shape (num_experts,)
    params/decoder/moe_layers/layers_{j}/.../bias              shape (num_experts, scan_length)

  Phase 2 needs ``transpose_update=True`` (counts have experts on the last
  axis, bias has experts on axis 0) and ``zero_mean_axis=0`` (re-center over
  experts).
  """
  for i, counts_i in enumerate(moe_expert_counts["prefix"]):
    target_path = _moe_bias_target_path(f"moe_layers_{i}", moe_block_name=moe_block_name)
    new_state = _try_update_bias(config, new_state, target_path, counts_i, f"ling3_prefix_moe_layers_{i}")
  for j, counts_j in enumerate(moe_expert_counts["scan"]):
    target_path = _moe_bias_target_path("moe_layers", f"layers_{j}", moe_block_name=moe_block_name)
    new_state = _try_update_bias(
        config,
        new_state,
        target_path,
        counts_j,
        f"ling3_scan_layers_{j}",
        zero_mean_axis=0,
        transpose_update=True,
    )
  return new_state


def collect_moe_expert_counts(config, intermediate_outputs):
  """Collect per-layer routed-expert counts for DeepSeek/LING2/LING3 backbones.

  Returns either:
    * dict ``{"prefix": [...], "scan": [...]}`` for LING3 scan (Phase 1b 1D
      counts + Phase 2 2D ``(scan_length, num_experts)`` counts);
    * the raw stacked sown tuple for DeepSeek scan;
    * a list of per-layer sown tuples for any unscanned config.

  Raises ``RuntimeError`` when an expected-to-exist path is missing.
  """
  ctx_base = f"decoder_block={config.decoder_block}"
  if not config.scan_layers:
    num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
    return [
        _get_sown(
            intermediate_outputs,
            ("intermediates", "decoder", f"moe_layers_{i}", "moe_expert_counts"),
            context=f"unscan moe_expert_counts ({ctx_base})",
            unwrap=False,
        )
        for i in range(num_moe_layers)
    ]

  if config.decoder_block == DecoderBlockType.LING3:
    _, num_moe_prefix, _ = compute_ling3_scan_layout(config)
    interval = config.inhomogeneous_layer_cycle_interval
    prefix = [
        _get_sown(
            intermediate_outputs,
            ("intermediates", "decoder", f"moe_layers_{i}", "moe_expert_counts"),
            context="LING3 Phase 1b moe_expert_counts",
            unwrap=False,
        )
        for i in range(num_moe_prefix)
    ]
    scan = [
        _get_sown(
            intermediate_outputs,
            ("intermediates", "decoder", "moe_layers", f"layers_{j}", "moe_expert_counts"),
            context="LING3 Phase 2 moe_expert_counts",
            unwrap=False,
        )
        for j in range(interval)
    ]
    return {"prefix": prefix, "scan": scan}

  # DeepSeek scan: single stacked tensor.  LING2 + scan is rejected upstream
  # in decoders.get_decoder_layers().
  return _get_sown(
      intermediate_outputs,
      ("intermediates", "decoder", "moe_layers", "moe_expert_counts"),
      context=f"scan moe_expert_counts ({ctx_base})",
      unwrap=False,
  )


def collect_mtp_expert_counts(config, intermediate_outputs):
  """Collect per-MTP-layer routed-expert counts, or ``None`` if no MTP layer sowed any.

  MTP layers may legitimately be absent (e.g. MTP disabled, or a non-MoE MTP
  block); the historical behavior is to silently return ``None`` in that case.
  """
  if not (
      config.routed_bias
      and config.routed_bias_update_rate > 0.0
      and getattr(config, "mtp_num_layers", 0) > 0
      and config.num_experts > 1
  ):
    return None
  per_layer = [
      maxtext_utils.get_nested_value(
          intermediate_outputs,
          (
              "intermediates",
              "mtp_block",
              f"mtp_layer_{k}",
              f"mtp_{k}_transformer_layer",
              "moe_expert_counts",
          ),
          None,
      )
      for k in range(1, config.mtp_num_layers + 1)
  ]
  return per_layer if any(u is not None for u in per_layer) else None


def apply_moe_bias_updates(config, new_state, moe_expert_counts, mtp_expert_counts):
  """Apply Auxiliary-Loss-Free load balancing bias updates from expert counts.

  ``moe_expert_counts`` contains raw per-expert token counts (possibly summed
  across gradient-accumulation micro-batches by the GA scan). We convert them
  to bias updates here so the direction is computed from the *total* counts,
  matching Megatron's accumulate-then-update semantics.
  """
  moe_block_name = _moe_block_name(config.decoder_block)

  if moe_expert_counts is not None:
    if moe_block_name is None:
      max_logging.log("Skipping moe_expert_counts: unsupported decoder block type.")
    elif not config.scan_layers:
      new_state = _update_unscan_moe_bias(config, new_state, moe_block_name, moe_expert_counts)
    elif config.decoder_block == DecoderBlockType.LING3:
      new_state = _update_ling3_scan_bias(config, new_state, moe_block_name, moe_expert_counts)
    elif config.decoder_block == DecoderBlockType.DEEPSEEK:
      new_state = _update_deepseek_scan_bias(config, new_state, moe_block_name, moe_expert_counts)

  # MTP MoE expert bias updates — matches Megatron's recursive module
  # traversal which updates ALL routers including MTP layers.
  if mtp_expert_counts is not None:
    if moe_block_name is not None:
      for k, expert_counts_for_layer in enumerate(mtp_expert_counts, start=1):
        if expert_counts_for_layer is None:
          continue
        target_path = (
            "params",
            "mtp_block",
            f"mtp_layer_{k}",
            f"mtp_{k}_transformer_layer",
            moe_block_name,
        ) + _GATE_BIAS_TAIL
        new_state = _try_update_bias(config, new_state, target_path, expert_counts_for_layer, f"mtp_layer_{k}")
    else:
      max_logging.log("Skipping mtp_expert_counts: unsupported decoder block type.")

  return new_state
