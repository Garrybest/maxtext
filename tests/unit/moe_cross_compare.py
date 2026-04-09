#!/usr/bin/env python3
"""Cross-framework MoE / Dense-MLP module-level precision comparison.

Loads Megatron Argus dump (via ``argus.load_tensor``), runs MaxText MoE or
Dense MLP forward/backward with the same input and weights, then compares
outputs at each module boundary to pinpoint precision divergence sources.

Requires: ``pip install -e /path/to/Argus``

Usage (MoE):
  python tools/moe_cross_compare.py \
      --megatron-dump-dir /path/to/megatron/argus_dump/step_1 \
      --orbax-ckpt-path /path/to/orbax/checkpoint \
      --layers 1,2,3 \
      --ling2-profile

Usage (Dense MLP):
  python tools/moe_cross_compare.py \
      --megatron-dump-dir /path/to/megatron/argus_dump/step_1 \
      --orbax-ckpt-path /path/to/orbax/checkpoint \
      --layers 0 \
      --mode dense \
      --ling2-profile
"""

from __future__ import annotations

# pylint: disable=import-outside-toplevel
# Heavy dependencies (jax, flax, orbax) are imported lazily inside functions
# to keep module import fast and avoid hard dependency when only using subsets.

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Precision metrics
# ---------------------------------------------------------------------------


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
  diff = (a.astype(np.float64) - b.astype(np.float64)).ravel()
  ref = b.astype(np.float64).ravel()
  norm_diff = np.linalg.norm(diff)
  if norm_diff == 0:
    return 0.0
  norm_ref = np.linalg.norm(ref)
  if norm_ref == 0:
    return float("inf")
  return float(norm_diff / norm_ref)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
  a64 = a.astype(np.float64).ravel()
  b64 = b.astype(np.float64).ravel()
  dot = np.dot(a64, b64)
  na = np.linalg.norm(a64)
  nb = np.linalg.norm(b64)
  if na == 0 or nb == 0:
    return 1.0 if (na == 0 and nb == 0) else 0.0
  return float(dot / (na * nb))


def _bf16_precision_stats(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
  """Compute precision metrics using absolute bf16 ULP distance.

  Converts float values to ordered bf16 integers and computes the integer
  distance, which equals the number of representable bf16 values between them.
  """
  a32 = a.astype(np.float32).ravel()
  b32 = b.astype(np.float32).ravel()

  # Truncate float32 → bf16 uint16
  a_u16 = (a32.view(np.uint32) >> 16).astype(np.uint16)
  b_u16 = (b32.view(np.uint32) >> 16).astype(np.uint16)
  exact_pct = float((a_u16 == b_u16).mean() * 100)

  # Map bf16 uint16 to ordered integers for correct ULP distance.
  # Positive bf16 (0x0000..0x7FFF): ordered = value (monotonically increasing)
  # Negative bf16 (0x8000..0xFFFF): ordered = 0x7FFF - value (reversed)
  # This gives: -max → most negative, -0 → -1, +0 → 0, +max → most positive
  def _to_ordered(u16):
    i = u16.astype(np.int64)
    return np.where(i >= 0x8000, np.int64(0x7FFF) - i, i)

  a_ord = _to_ordered(a_u16)
  b_ord = _to_ordered(b_u16)
  ulp_dist = np.abs(a_ord - b_ord)

  diff_mask = ulp_dist > 0
  if diff_mask.any():
    d = ulp_dist[diff_mask]
    ulp_p99 = float(np.percentile(d, 99.9))
  else:
    ulp_p99 = 0.0

  return {
      "bf16_exact_pct": exact_pct,
      "bf16_le1ulp_pct": float((ulp_dist <= 1).mean() * 100),
      "ulp_max": int(ulp_dist.max()),
      "ulp_p999": ulp_p99,
      "ulp_mean": float(ulp_dist.mean()),
  }


def _compare_pair(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
  """Return a metrics dict for two float tensors."""
  a32 = a.astype(np.float32)
  b32 = b.astype(np.float32)
  result = {
      "max_abs": _max_abs(a32, b32),
      "mean_abs": _mean_abs(a32, b32),
      "rel_l2": _rel_l2(a32, b32),
      "cosine": _cosine(a32, b32),
  }
  result.update(_bf16_precision_stats(a32, b32))
  return result


# ---------------------------------------------------------------------------
# Argus data loading
# ---------------------------------------------------------------------------


def _import_argus(dp_mode: str = "default"):
  """Import argus.load_tensor / list_tensors or fail fast.

  Args:
      dp_mode: Argus dp_mode. "replica" enables single-rank fallback for
          dumps where the gatherer cannot handle multi-dim shard_by (ep+dp).
          "default" uses upstream argus behavior without fallback.
  """
  try:
    from argus import load_tensor as _raw_load, list_tensors  # type: ignore  # pylint: disable=import-outside-toplevel

    if dp_mode == "replica":
      from argus.compare.gatherer import ShardGatherer  # pylint: disable=import-outside-toplevel

      def load_tensor(dump_dir, name, **kwargs):
        """Wrapper that falls back to single-rank loading on gather errors."""
        try:
          return _raw_load(dump_dir, name, dp_mode="replica", **kwargs)
        except (KeyError, ValueError):
          # Multi-rank gather can fail when shard_by has multiple dims (ep+dp).
          # Fall back by pointing a gatherer at rank_0 only.
          category, tensor_name = name.split(":", 1)
          from glob import glob as _glob  # pylint: disable=import-outside-toplevel

          rank0 = sorted(_glob(f"{dump_dir}/rank_*"))[0]
          g = ShardGatherer(rank0, dp_mode="replica")
          return g.gather(category, tensor_name)

      return load_tensor, list_tensors

    return _raw_load, list_tensors
  except ImportError as e:
    print(f"ERROR: Failed to import argus: {type(e).__name__}: {e}")
    print("Install with: pip install -e /path/to/Argus")
    sys.exit(1)


# Argus tensor names that Megatron MoE dumps via forward/backward hooks.
# (short_name, argus_category, module_suffix, io_suffix)
#   → full argus name = "{category}:layer_{idx}_{module_suffix}_{io_suffix}"
_ARGUS_FORWARD_POINTS = [
    ("moe_input", "forward", "moe", "input"),
    ("moe_output", "forward", "moe", "output"),
    ("moe_input_0", "forward", "moe", "input_0"),
    ("moe_output_0", "forward", "moe", "output_0"),
    ("moe_router_input", "forward", "moe_router", "input"),
    ("moe_router_output", "forward", "moe_router", "output"),
    ("moe_router_input_0", "forward", "moe_router", "input_0"),
    ("moe_router_output_0", "forward", "moe_router", "output_0"),
    ("moe_experts_input", "forward", "moe_experts", "input"),
    ("moe_experts_output", "forward", "moe_experts", "output"),
    ("moe_experts_input_0", "forward", "moe_experts", "input_0"),
    ("moe_experts_output_0", "forward", "moe_experts", "output_0"),
    ("shared_expert_input", "forward", "shared_expert", "input"),
    ("shared_expert_output", "forward", "shared_expert", "output"),
    ("shared_expert_input_0", "forward", "shared_expert", "input_0"),
    ("shared_expert_output_0", "forward", "shared_expert", "output_0"),
]

_ARGUS_BACKWARD_POINTS = [
    ("moe_input_grad", "backward", "moe", "input_grad"),
    ("moe_output_grad", "backward", "moe", "output_grad"),
    ("moe_router_input_grad", "backward", "moe_router", "input_grad"),
    ("moe_router_output_grad", "backward", "moe_router", "output_grad"),
    ("moe_experts_input_grad", "backward", "moe_experts", "input_grad"),
    ("moe_experts_output_grad", "backward", "moe_experts", "output_grad"),
    ("shared_expert_input_grad", "backward", "shared_expert", "input_grad"),
    ("shared_expert_output_grad", "backward", "shared_expert", "output_grad"),
]

# Dense MLP Argus hook points
# New dump format uses "_0" suffix (e.g. dense_mlp_input_0, dense_mlp_output_0).
# We try both old and new names; the extra-tensor discovery loop also picks up
# anything with the layer prefix that we didn't list here.
_ARGUS_DENSE_FORWARD_POINTS = [
    ("dense_mlp_input", "forward", "dense_mlp", "input"),
    ("dense_mlp_output", "forward", "dense_mlp", "output"),
    ("dense_mlp_input_0", "forward", "dense_mlp", "input_0"),
    ("dense_mlp_output_0", "forward", "dense_mlp", "output_0"),
    ("dense_mlp_fc1_output", "forward", "dense_mlp_fc1_output", "input"),
    ("dense_mlp_after_activation", "forward", "dense_mlp_after_activation", "input"),
    ("dense_mlp_fc2_output", "forward", "dense_mlp_fc2_output", "input"),
]

_ARGUS_DENSE_BACKWARD_POINTS = [
    ("dense_mlp_input_grad", "backward", "dense_mlp", "input_grad"),
    ("dense_mlp_output_grad", "backward", "dense_mlp", "output_grad"),
]


def _argus_name(layer_idx: int, category: str, module: str, suffix: str) -> str:
  return f"{category}:layer_{layer_idx}_{module}_{suffix}"


def load_megatron_data(
    dump_dir: str,
    layer_idx: int,
    *,
    micro_batch_size: int = 2,
    load_backward: bool = False,
    mode: str = "moe",
    dp_mode: str = "default",
) -> dict[str, np.ndarray]:
  """Load all available Megatron tensors for one layer via Argus API.

  Args:
    mode: "moe" or "dense" — selects which Argus hook points to look for.
    dp_mode: Argus dp_mode. "replica" for single-rank fallback, "default" for upstream behavior.

  Returns dict with short_name keys (e.g. "moe_input", "dense_mlp_input", etc.)
  plus any extra forward/backward tensors discovered for this layer,
  plus param grads from the ``grads:`` category.
  """
  load_tensor, list_tensors = _import_argus(dp_mode=dp_mode)

  # list_tensors may or may not include the category prefix depending on
  # the Argus version.  Normalise to "category:name" so lookups are uniform.
  def _prefixed(names, category):
    prefix = f"{category}:"
    return {n if n.startswith(prefix) else f"{prefix}{n}" for n in names}

  fwd_names = list_tensors(dump_dir, "forward")
  all_names: set[str] = _prefixed(fwd_names, "forward")
  if load_backward:
    try:
      bwd_names = list_tensors(dump_dir, "backward")
      all_names |= _prefixed(bwd_names, "backward")
    except (OSError, ValueError):
      bwd_names = []
    try:
      grad_names = list_tensors(dump_dir, "grads")
      all_names |= _prefixed(grad_names, "grads")
    except (OSError, ValueError):
      grad_names = []

  layer_prefix = f"layer_{layer_idx}_"
  result: dict[str, np.ndarray] = {}

  loaded_count = 0

  def _try_load(argus_name: str, short_name: str):
    nonlocal loaded_count
    if argus_name not in all_names:
      return
    try:
      loaded_count += 1
      print(f"    [{loaded_count}] Loading {argus_name} ...", end="", flush=True)
      result[short_name] = load_tensor(dump_dir, argus_name)
      print(f" shape={result[short_name].shape}")
    except (ValueError, OSError, pickle.UnpicklingError) as e:
      # Object arrays (pickle) are typically None/bias placeholders — skip silently
      if "allow_pickle" in str(e):
        print(" skipped (non-tensor data)")
      else:
        print(f" FAILED: {e}")

  # Select hook points based on mode
  fwd_points = _ARGUS_DENSE_FORWARD_POINTS if mode == "dense" else _ARGUS_FORWARD_POINTS
  bwd_points = _ARGUS_DENSE_BACKWARD_POINTS if mode == "dense" else _ARGUS_BACKWARD_POINTS

  # Known forward points
  for short, cat, module, suffix in fwd_points:
    _try_load(_argus_name(layer_idx, cat, module, suffix), short)

  # Extra forward tensors for this layer not in our known list
  # Filter by mode to avoid loading unrelated modules (attn, layernorm, etc.)
  module_filter = "dense_mlp" if mode == "dense" else "moe"
  known_fwd = {_argus_name(layer_idx, cat, mod, suf) for _, cat, mod, suf in fwd_points}
  for name in all_names:
    if not name.startswith("forward:"):
      continue
    if name not in known_fwd:
      tensor_part = name.split(":", 1)[1]
      if tensor_part.startswith(layer_prefix):
        short = tensor_part[len(layer_prefix) :]
        if module_filter in short:
          _try_load(name, short)

  if load_backward:
    # Known backward points
    for short, cat, module, suffix in bwd_points:
      _try_load(_argus_name(layer_idx, cat, module, suffix), short)

    # Extra backward tensors — also filter by mode
    known_bwd = {_argus_name(layer_idx, cat, mod, suf) for _, cat, mod, suf in bwd_points}
    for name in all_names:
      if not name.startswith("backward:"):
        continue
      if name not in known_bwd:
        tensor_part = name.split(":", 1)[1]
        if tensor_part.startswith(layer_prefix):
          short = tensor_part[len(layer_prefix) :]
          if module_filter in short:
            _try_load(name, short)

    # Param grads — filter by layer AND module type (dense mlp vs moe).
    # In Megatron, both dense MLP and MoE blocks live under the `.mlp.` sub-module,
    # so the filter excludes attention/layernorm grads in both modes.
    # For MoE we use a broader filter to also catch gate/router params that may
    # sit outside the `.mlp.` subtree in some checkpoint formats.
    layer_grad_pattern = f"decoder.layers.{layer_idx}."
    grad_module_filter = ".mlp." if mode == "dense" else ".mlp."
    # MoE grads may also live under router/gate paths; load all layer grads
    # and rely on GRAD_MAP matching to pick the right ones.
    for name in all_names:
      if not name.startswith("grads:"):
        continue
      param_name = name.split(":", 1)[1]
      if not layer_grad_pattern in param_name:
        continue
      if mode == "dense":
        # Dense mode: only load MLP-related param grads
        if grad_module_filter not in param_name:
          continue
      # MoE mode: load all grads for this layer (gate, experts, shared, mlp)
      _try_load(name, f"param_grad:{param_name}")

  return result


# ---------------------------------------------------------------------------
# Orbax checkpoint utilities
# ---------------------------------------------------------------------------


def _is_leaf(x: Any) -> bool:
  """Check if x is a tensor-like leaf (has shape and dtype)."""
  return hasattr(x, "shape") and hasattr(x, "dtype")


def _flatten_tree(tree: Any, prefix: str = "") -> dict[str, Any]:
  """Recursively flatten a nested dict/list checkpoint tree into dot-separated keys."""
  out: dict[str, Any] = {}
  if _is_leaf(tree):
    out[prefix] = tree
    return out
  if isinstance(tree, dict):
    for k, v in tree.items():
      p = f"{prefix}.{k}" if prefix else str(k)
      out.update(_flatten_tree(v, p))
    return out
  if isinstance(tree, (list, tuple)):
    for i, v in enumerate(tree):
      p = f"{prefix}[{i}]" if prefix else f"[{i}]"
      out.update(_flatten_tree(v, p))
    return out
  return out


def _choose_tensor_key(
    flat: dict[str, Any],
    key_scope_marker: str,
    suffix_candidates: tuple[str, ...],
    expected_shape: tuple[int, ...],
    required: bool = True,
) -> tuple[np.ndarray | None, str | None]:
  """Find and return a tensor from flat checkpoint by scope marker and suffix candidates."""
  matches = [
      k
      for k in flat
      if (not key_scope_marker or key_scope_marker in k) and any(k.endswith(sfx) for sfx in suffix_candidates)
  ]
  if not matches:
    if required:
      raise KeyError(
          f"No tensor matched marker='{key_scope_marker}' "
          f"suffixes={suffix_candidates} expected_shape={expected_shape}"
      )
    return None, None
  if expected_shape is not None:
    shape_matched = [k for k in matches if tuple(np.asarray(flat[k]).shape) == expected_shape]
    candidates = sorted(shape_matched or matches, key=lambda x: (len(x), x))
  else:
    candidates = sorted(matches, key=lambda x: (len(x), x))
  chosen = candidates[0]
  arr = np.asarray(flat[chosen], dtype=np.float32)
  if expected_shape is not None and tuple(arr.shape) != expected_shape:
    raise ValueError(f"Shape mismatch for {chosen}: {arr.shape} vs {expected_shape}")
  return arr, chosen


# ---------------------------------------------------------------------------
# MaxText module construction
# ---------------------------------------------------------------------------


def build_maxtext_moe(args: argparse.Namespace):
  """Build MaxText MoE module (single-device mesh). Returns (model, cfg, mesh)."""
  import jax
  import jax.numpy as jnp
  from flax import nnx
  from jax.sharding import Mesh

  from maxtext.configs import pyconfig
  from maxtext.utils.globals import MAXTEXT_PKG_DIR
  from maxtext.layers import moe
  from maxtext.layers.initializers import nd_dense_init

  cfg = pyconfig.initialize(
      [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
      run_name="moe_cross_compare",
      enable_checkpointing=False,
      model_name="ling2" if args.ling2_profile else "mixtral-8x7b",
      dtype="bfloat16" if args.compute_dtype == "bf16" else "float32",
      matmul_precision=args.matmul_precision,
      max_target_length=args.seq_len,
      max_prefill_predict_length=args.seq_len,
      per_device_batch_size=args.batch_size,
      base_emb_dim=args.hidden_size,
      num_experts=args.num_experts,
      num_experts_per_tok=args.top_k,
      base_mlp_dim=args.ffn_hidden_size,
      base_moe_mlp_dim=args.ffn_hidden_size,
      shared_experts=1 if args.enable_shared_expert else 0,
      moe_shared_expert_dim=args.shared_ffn_hidden_size,
      routed_score_func=args.routed_score_func,
      routed_scaling_factor=args.routed_scaling_factor,
      norm_topk_prob=args.norm_topk_prob,
      n_routing_groups=args.n_routing_groups,
      topk_routing_group=args.topk_routing_group,
      routed_bias=args.routed_bias,
      routed_bias_update_rate=0.0,
      load_balance_loss_weight=0.0,
      decoder_block="ling2" if args.ling2_profile else "default",
      activations_in_float32=args.activations_in_float32,
      ici_fsdp_parallelism=1,
      ici_tensor_parallelism=1,
      ici_expert_parallelism=1,
      ici_sequence_parallelism=1,
      ici_autoregressive_parallelism=1,
      dcn_fsdp_parallelism=1,
      dcn_tensor_parallelism=1,
      dcn_expert_parallelism=1,
      dcn_sequence_parallelism=1,
      dcn_autoregressive_parallelism=1,
  )

  n_axes = len(cfg.mesh_axes)
  devices_array = np.array(jax.devices()[:1]).reshape([1] * n_axes)
  mesh = Mesh(devices_array, cfg.mesh_axes)
  compute_dtype = jnp.bfloat16 if args.compute_dtype == "bf16" else jnp.float32

  if args.enable_shared_expert:
    model = moe.RoutedAndSharedMoE(
        config=cfg,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        rngs=nnx.Rngs(params=42),
        dtype=compute_dtype,
        weight_dtype=compute_dtype,
    )
  else:
    model = moe.RoutedMoE(
        config=cfg,
        num_experts=args.num_experts,
        num_experts_per_tok=args.top_k,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        rngs=nnx.Rngs(params=42),
        intermediate_dim=args.ffn_hidden_size,
        dtype=compute_dtype,
        weight_dtype=compute_dtype,
    )

  return model, cfg, mesh


def build_maxtext_dense_mlp(args: argparse.Namespace):
  """Build MaxText dense MLP module (single-device mesh). Returns (model, cfg, mesh)."""
  import jax
  import jax.numpy as jnp
  from flax import nnx
  from jax.sharding import Mesh

  from maxtext.configs import pyconfig
  from maxtext.utils.globals import MAXTEXT_PKG_DIR
  from maxtext.layers.linears import MlpBlock
  from maxtext.layers.initializers import nd_dense_init

  cfg = pyconfig.initialize(
      [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
      run_name="dense_cross_compare",
      enable_checkpointing=False,
      model_name="ling2" if args.ling2_profile else "mixtral-8x7b",
      dtype="bfloat16" if args.compute_dtype == "bf16" else "float32",
      matmul_precision=args.matmul_precision,
      max_target_length=args.seq_len,
      max_prefill_predict_length=args.seq_len,
      per_device_batch_size=args.batch_size,
      base_emb_dim=args.hidden_size,
      base_mlp_dim=args.dense_ffn_hidden_size,
      fused_mlp=args.fused_mlp,
      activations_in_float32=args.activations_in_float32,
      ici_fsdp_parallelism=1,
      ici_tensor_parallelism=1,
      ici_expert_parallelism=1,
      ici_sequence_parallelism=1,
      ici_autoregressive_parallelism=1,
      dcn_fsdp_parallelism=1,
      dcn_tensor_parallelism=1,
      dcn_expert_parallelism=1,
      dcn_sequence_parallelism=1,
      dcn_autoregressive_parallelism=1,
  )

  n_axes = len(cfg.mesh_axes)
  devices_array = np.array(jax.devices()[:1]).reshape([1] * n_axes)
  mesh = Mesh(devices_array, cfg.mesh_axes)
  compute_dtype = jnp.bfloat16 if args.compute_dtype == "bf16" else jnp.float32

  model = MlpBlock(
      config=cfg,
      mesh=mesh,
      in_features=args.hidden_size,
      intermediate_dim=args.dense_ffn_hidden_size,
      activations=cfg.mlp_activations,
      kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
      intermediate_dropout_rate=0.0,
      dtype=compute_dtype,
      weight_dtype=compute_dtype,
      rngs=nnx.Rngs(params=42),
  )

  return model, cfg, mesh


def _load_orbax_checkpoint(orbax_ckpt_path: str) -> dict[str, Any]:
  """Load and flatten Orbax checkpoint with progress messages."""
  import time
  import orbax.checkpoint as ocp

  print(f"  [1/2] Restoring Orbax checkpoint: {orbax_ckpt_path} ...")
  t0 = time.time()
  # Handle composite checkpoints where data lives under items/
  items_path = os.path.join(orbax_ckpt_path, "items")
  restore_path = items_path if os.path.exists(os.path.join(items_path, "_METADATA")) else orbax_ckpt_path
  ckpt = ocp.StandardCheckpointer().restore(restore_path)
  t1 = time.time()
  print(f"  [1/2] Restore done ({t1 - t0:.1f}s)")

  print("  [2/2] Flattening checkpoint tree ...")
  flat = _flatten_tree(ckpt)
  t2 = time.time()
  print(f"  [2/2] Flatten done ({t2 - t1:.1f}s, {len(flat)} tensors)")
  return flat


def load_orbax_dense_weights(model, orbax_ckpt_path, layer_idx, orbax_layer_prefix, fused_mlp=False):
  """Load weights from Orbax checkpoint into dense MLP model."""
  import jax.numpy as jnp

  flat = _load_orbax_checkpoint(orbax_ckpt_path)
  marker = f"dense_layers_{layer_idx}" if orbax_layer_prefix is None else ""
  prefix = orbax_layer_prefix or f"params.params.decoder.dense_layers_{layer_idx}"

  if fused_mlp:
    # fused_mlp: model.wi.kernel has shape [in_features, num_activations, ffn_hidden]
    # Load wi_0 and wi_1 separately, then stack into [in, 2, ffn]
    wi_0_key = f"{prefix}.mlp.wi_0.kernel"
    wi_1_key = f"{prefix}.mlp.wi_1.kernel"
    wi_0_arr, _ = _choose_tensor_key(
        flat,
        marker,
        (wi_0_key, f"{prefix}.MlpBlock_0.wi_0.kernel", f"{prefix}.wi_0.kernel"),
        None,
        required=False,
    )
    wi_1_arr, _ = _choose_tensor_key(
        flat,
        marker,
        (wi_1_key, f"{prefix}.MlpBlock_0.wi_1.kernel", f"{prefix}.wi_1.kernel"),
        None,
        required=False,
    )
    if wi_0_arr is not None and wi_1_arr is not None:
      # Stack: [in, ffn] + [in, ffn] → [in, 2, ffn]
      fused = np.stack([wi_0_arr, wi_1_arr], axis=1)
      model.wi.kernel.value = jnp.array(fused, dtype=jnp.float32)
      print(f"    wi.kernel (fused from wi_0+wi_1)  shape={fused.shape}")
    else:
      raise KeyError("Cannot find wi_0/wi_1 in checkpoint for fused_mlp")
  else:
    for wn in ("wi_0", "wi_1"):
      attr = getattr(model, wn, None)
      if attr is None:
        continue
      param = attr.kernel
      shape = tuple(np.asarray(param[...]).shape)
      arr, key = _choose_tensor_key(
          flat,
          marker,
          (f"{prefix}.mlp.{wn}.kernel", f"{prefix}.MlpBlock_0.{wn}.kernel", f"{prefix}.{wn}.kernel", f".{wn}.kernel"),
          shape,
      )
      param.value = jnp.array(arr, dtype=jnp.float32)
      print(f"    {wn}.kernel  ← {key}  shape={shape}")

  # wo is the same regardless of fused_mlp
  wo_param = model.wo.kernel
  wo_shape = tuple(np.asarray(wo_param[...]).shape)
  wo_arr, wo_key = _choose_tensor_key(
      flat,
      marker,
      (f"{prefix}.mlp.wo.kernel", f"{prefix}.MlpBlock_0.wo.kernel", f"{prefix}.wo.kernel", ".wo.kernel"),
      wo_shape,
  )
  wo_param.value = jnp.array(wo_arr, dtype=jnp.float32)
  print(f"    wo.kernel  ← {wo_key}  shape={wo_shape}")

  print(f"  Loaded Orbax dense MLP weights (layer {layer_idx})")


def load_orbax_weights(model, orbax_ckpt_path, layer_idx, orbax_layer_prefix, enable_shared_expert, orbax_moe_idx=None):
  """Load weights from Orbax checkpoint into the model."""
  import jax.numpy as jnp

  flat = _load_orbax_checkpoint(orbax_ckpt_path)
  moe_idx = orbax_moe_idx if orbax_moe_idx is not None else layer_idx
  layer_prefix = orbax_layer_prefix or f"params.params.decoder.moe_layers_{moe_idx}.ALMoeBlock_0"
  marker = f"moe_layers_{moe_idx}"
  routed = model.routed_moe if enable_shared_expert else model

  # Gate kernel
  gate_shape = tuple(np.asarray(routed.gate.kernel[...]).shape)
  gate, gate_key = _choose_tensor_key(
      flat,
      marker,
      (f"{layer_prefix}.MoeBlock_0.gate.kernel", f"{layer_prefix}.gate.kernel", ".MoeBlock_0.gate.kernel"),
      gate_shape,
  )
  print(f"    gate.kernel: {gate_key} shape={gate.shape}")
  routed.gate.kernel.value = jnp.array(gate, dtype=jnp.float32)

  # Gate bias
  if getattr(routed.gate, "bias", None) is not None:
    bias_shape = tuple(np.asarray(routed.gate.bias[...]).shape)
    gate_bias, bias_key = _choose_tensor_key(
        flat,
        marker,
        (f"{layer_prefix}.MoeBlock_0.gate.bias", f"{layer_prefix}.gate.bias", ".MoeBlock_0.gate.bias"),
        bias_shape,
        required=False,
    )
    print(f"    gate.bias:   {bias_key} shape={gate_bias.shape if gate_bias is not None else 'None'}")
    routed.gate.bias.value = jnp.array(gate_bias if gate_bias is not None else np.zeros(bias_shape), dtype=jnp.float32)

  # Expert weights
  for wn in ("wi_0", "wi_1", "wo"):
    param = getattr(routed, wn)
    shape = tuple(np.asarray(param[...]).shape)
    arr, matched_key = _choose_tensor_key(
        flat,
        marker,
        (f"{layer_prefix}.MoeBlock_0.{wn}", f"{layer_prefix}.{wn}", f".MoeBlock_0.{wn}"),
        shape,
    )
    print(f"    {wn}: {matched_key} shape={arr.shape}")
    param.value = jnp.array(arr, dtype=jnp.float32)

  # Shared expert weights
  if enable_shared_expert:
    for wn in ("wi_0", "wi_1", "wo"):
      param = getattr(model.shared_experts, wn).kernel
      shape = tuple(np.asarray(param[...]).shape)
      arr, matched_key = _choose_tensor_key(
          flat,
          marker,
          (
              f"{layer_prefix}.shared_experts.{wn}.kernel",
              f"{layer_prefix}.MoeBlock_0.shared_experts.{wn}.kernel",
              f".shared_experts.{wn}.kernel",
          ),
          shape,
      )
      print(f"    shared.{wn}: {matched_key} shape={arr.shape}")
      param.value = jnp.array(arr, dtype=jnp.float32)

  print(f"  Loaded Orbax weights (layer {layer_idx})")


# ---------------------------------------------------------------------------
# MaxText forward / backward
# ---------------------------------------------------------------------------


def run_maxtext_forward(model, input_tensor: np.ndarray, args) -> dict[str, np.ndarray]:
  """Run MaxText MoE forward, collecting outputs at the same module boundaries as Argus.

  Returns dict with the same short_name keys as Megatron Argus dump:
    moe_input, moe_output, moe_router_output, moe_experts_output,
    shared_expert_output, etc.
  """
  import jax
  import jax.numpy as jnp

  compute_dtype = jnp.bfloat16 if args.compute_dtype == "bf16" else jnp.float32
  # MaxText expects 3D (batch, seq, hidden).  Argus load_tensor returns
  # 2D (tokens, hidden) for consistent token ordering.  Reshape to 3D
  # using seq_len to recover the original (seq, batch, hidden) layout.
  if input_tensor.ndim == 2:
    total_tokens, hidden = input_tensor.shape
    seq_len = args.seq_len
    if total_tokens % seq_len == 0:
      batch = total_tokens // seq_len
    else:
      batch = 1
      seq_len = total_tokens
    input_tensor = input_tensor.reshape(seq_len, batch, hidden)
  x = jnp.array(input_tensor, dtype=compute_dtype)
  enable_shared = args.enable_shared_expert
  routed = model.routed_moe if enable_shared else model

  # Gate logits — compute raw logits manually to also compare pre-sigmoid
  gate_kernel = jnp.asarray(routed.gate.kernel[...], jnp.float32)
  x_flat_f32 = x.reshape(-1, x.shape[-1]).astype(jnp.float32)
  raw_logits = x_flat_f32 @ gate_kernel  # (tokens, num_experts), before sigmoid

  gate_result = routed.gate(x)
  gate_logits = gate_result[0]  # post-bias (sigmoid + bias)
  pre_bias_logits = gate_result[1] if len(gate_result) > 1 else gate_logits  # pre-bias (sigmoid only)
  # Flatten to (tokens, num_experts) to match Megatron's shape
  orig_shape = gate_logits.shape
  gate_flat = gate_logits.reshape(-1, orig_shape[-1])
  pre_bias_flat = pre_bias_logits.reshape(-1, orig_shape[-1])
  raw_logits_flat = raw_logits.reshape(-1, orig_shape[-1])

  # Top-k using model's actual routing logic (grouped top-k + deepseek scaling)
  topk_vals, topk_idx = routed.get_topk(gate_logits, pre_bias_logits)

  # Run routed MoE and shared expert separately to get each output
  routed_output_tuple = routed(x)
  routed_out = routed_output_tuple[0]

  result: dict[str, np.ndarray] = {
      "moe_input": np.asarray(jax.device_get(x), dtype=np.float32),
      "moe_router_output": np.asarray(jax.device_get(gate_flat), dtype=np.float32),
      "moe_router_pre_bias": np.asarray(jax.device_get(pre_bias_flat), dtype=np.float32),
      "moe_router_raw_logits": np.asarray(jax.device_get(raw_logits_flat), dtype=np.float32),
      "moe_topk_indices": np.asarray(jax.device_get(topk_idx), dtype=np.int32),
      "moe_topk_weights": np.asarray(jax.device_get(topk_vals), dtype=np.float32),
  }

  if enable_shared and hasattr(model, "shared_experts"):
    shared_out = model.shared_experts(x)
    total_output = routed_out + shared_out
    result["shared_expert_output"] = np.asarray(jax.device_get(shared_out), dtype=np.float32)
    result["shared_expert_input"] = result["moe_input"].copy()
    result["moe_experts_output"] = np.asarray(jax.device_get(routed_out), dtype=np.float32)
  else:
    total_output = routed_out

  result["moe_output"] = np.asarray(jax.device_get(total_output), dtype=np.float32)

  return result


def run_maxtext_backward(
    model,
    input_tensor: np.ndarray,
    output_grad: np.ndarray,
    args,
) -> dict[str, np.ndarray]:
  """Run MaxText backward using Megatron's output_grad as the upstream gradient.

  Instead of inventing a synthetic loss, we take the actual dL/d_output from
  Megatron's Argus dump (backward:layer_{i}_moe_output_grad) and push it
  through MaxText via jax.vjp so both sides see the same upstream signal.

  Returns param grads + moe_input_grad.
  """
  import jax
  import jax.numpy as jnp
  from flax import nnx

  compute_dtype = jnp.bfloat16 if args.compute_dtype == "bf16" else jnp.float32

  # Reshape 2D (tokens, hidden) → 3D (seq, batch, hidden) to match model sharding
  if input_tensor.ndim == 2:
    total_tokens, hidden = input_tensor.shape
    seq_len = args.seq_len
    if total_tokens % seq_len == 0:
      batch = total_tokens // seq_len
    else:
      batch = 1
      seq_len = total_tokens
    input_tensor = input_tensor.reshape(seq_len, batch, hidden)
    output_grad = output_grad.reshape(seq_len, batch, hidden)

  x = jnp.array(input_tensor, dtype=compute_dtype)
  dy = jnp.array(output_grad, dtype=compute_dtype)

  # Split model state into differentiable params and the rest.
  graphdef, params, rest = nnx.split(model, nnx.Param, ...)

  def _forward(params, x):
    merged = nnx.merge(graphdef, params, rest)
    outputs = merged(x)
    y = outputs[0] if isinstance(outputs, tuple) else outputs
    return y

  # Forward + backward with Megatron's output_grad as cotangent
  _, vjp_fn = jax.vjp(_forward, params, x)
  param_grads, input_grad = vjp_fn(dy)

  grads: dict[str, np.ndarray] = {}

  # Param grads: flatten nnx state
  for path, var_state in param_grads.flat_state():
    key = "__".join(str(p) for p in path)
    grads[key] = np.asarray(jax.device_get(var_state[...]), dtype=np.float32)

  # Activation grads
  grads["moe_input_grad"] = np.asarray(jax.device_get(input_grad), dtype=np.float32)
  grads["moe_output_grad"] = np.asarray(jax.device_get(dy), dtype=np.float32)

  print(f"  MaxText backward done (vjp with Megatron output_grad). {len(grads)} grad tensors")
  return grads


def run_maxtext_dense_forward(model, input_tensor: np.ndarray, args) -> dict[str, np.ndarray]:
  """Run MaxText dense MLP forward using model(x) directly."""
  import jax
  import jax.numpy as jnp

  # Diagnostic: verify MlpBlock config
  print(f"  [diag] model.dtype={model.dtype}, model.weight_dtype={model.weight_dtype}")
  print(f"  [diag] model.activations={model.activations}")
  print(f"  [diag] model.in_features={model.in_features}, model.intermediate_dim={model.intermediate_dim}")
  print(f"  [diag] model.use_bias={model.use_bias}, model.use_pre_norm={model.use_pre_norm}")
  print(f"  [diag] config.matmul_precision={model.config.matmul_precision}")
  print(f"  [diag] config.fused_mlp={model.config.fused_mlp}")

  # Print weight checksums to verify Orbax loading
  weight_names = ("wi", "wo") if hasattr(model, "wi") else ("wi_0", "wi_1", "wo")
  for wn in weight_names:
    attr = getattr(model, wn, None)
    if attr is not None:
      w = np.asarray(attr.kernel[...])
      print(f"    weight {wn}: shape={w.shape} dtype={w.dtype} mean={w.mean():.6e} std={w.std():.6e} sum={w.sum():.6e}")

  compute_dtype = jnp.bfloat16 if args.compute_dtype == "bf16" else jnp.float32
  # MaxText expects 3D (batch, seq, hidden).  Argus load_tensor returns
  # 2D (tokens, hidden) for consistent token ordering.  Reshape to 3D
  # using seq_len to recover the original (seq, batch, hidden) layout.
  if input_tensor.ndim == 2:
    total_tokens, hidden = input_tensor.shape
    seq_len = args.seq_len
    if total_tokens % seq_len == 0:
      batch = total_tokens // seq_len
    else:
      batch = 1
      seq_len = total_tokens
    input_tensor = input_tensor.reshape(seq_len, batch, hidden)
  x = jnp.array(input_tensor, dtype=compute_dtype)
  print(f"  [diag] input x.dtype={x.dtype}, x.shape={x.shape}")

  result: dict[str, np.ndarray] = {
      "dense_mlp_input": np.asarray(jax.device_get(x), dtype=np.float32),
  }

  output = model(x, deterministic=True)
  result["dense_mlp_output"] = np.asarray(jax.device_get(output), dtype=np.float32)

  return result


def run_maxtext_dense_backward(
    model,
    input_tensor: np.ndarray,
    output_grad: np.ndarray,
    args,
) -> dict[str, np.ndarray]:
  """Run MaxText dense MLP backward using Megatron's output_grad via jax.vjp."""
  import jax
  import jax.numpy as jnp
  from flax import nnx

  compute_dtype = jnp.bfloat16 if args.compute_dtype == "bf16" else jnp.float32
  # Reshape 2D (tokens, hidden) → 3D (seq, batch, hidden) to match model sharding
  if input_tensor.ndim == 2:
    total_tokens, hidden = input_tensor.shape
    seq_len = args.seq_len
    if total_tokens % seq_len == 0:
      batch = total_tokens // seq_len
    else:
      batch = 1
      seq_len = total_tokens
    input_tensor = input_tensor.reshape(seq_len, batch, hidden)
    output_grad = output_grad.reshape(seq_len, batch, hidden)
  x = jnp.array(input_tensor, dtype=compute_dtype)
  dy = jnp.array(output_grad, dtype=compute_dtype)

  graphdef, params, rest = nnx.split(model, nnx.Param, ...)

  def _forward(params, x):
    merged = nnx.merge(graphdef, params, rest)
    return merged(x, deterministic=True)

  _, vjp_fn = jax.vjp(_forward, params, x)
  param_grads, input_grad = vjp_fn(dy)

  grads: dict[str, np.ndarray] = {}
  for path, var_state in param_grads.flat_state():
    key = "__".join(str(p) for p in path)
    grads[key] = np.asarray(jax.device_get(var_state[...]), dtype=np.float32)

  grads["dense_mlp_input_grad"] = np.asarray(jax.device_get(input_grad), dtype=np.float32)
  grads["dense_mlp_output_grad"] = np.asarray(jax.device_get(dy), dtype=np.float32)

  print(f"  MaxText dense backward done (vjp). {len(grads)} grad tensors")
  return grads


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

# Forward stage map: (display_name, maxtext_key, megatron_key)
FORWARD_STAGE_MAP = [
    ("moe_input", "moe_input", "moe_input"),
    # NOTE: Megatron's moe_router_output is the final normalized+scaled weights
    # (masked, sum=scaling_factor), NOT raw/pre-bias/post-bias gate logits.
    # MaxText gate stages have no Megatron equivalent in the dump, so we skip them.
    ("moe_topk_indices", "moe_topk_indices", "moe_topk_indices"),
    ("moe_output", "moe_output", "moe_output"),
    ("shared_expert_input", "shared_expert_input", "shared_expert_input"),
    ("shared_expert_output", "shared_expert_output", "shared_expert_output"),
]

DENSE_FORWARD_STAGE_MAP = [
    ("dense_mlp_input", "dense_mlp_input", "dense_mlp_input"),
    ("dense_mlp_output", "dense_mlp_output", "dense_mlp_output"),
]

DENSE_GRAD_MAP = [
    ("wi_0.kernel", "wi_0__kernel", ["param_grad:wi_0.weight", "param_grad:wi_0"]),
    ("wi_1.kernel", "wi_1__kernel", ["param_grad:wi_1.weight", "param_grad:wi_1"]),
    ("wo.kernel", "wo__kernel", ["param_grad:wo.weight", "param_grad:wo"]),
    ("input_grad", "dense_mlp_input_grad", ["dense_mlp_input_grad"]),
]

DENSE_GRAD_MAP_FUSED = [
    ("wi.kernel", "wi__kernel", ["param_grad:wi.weight", "param_grad:wi"]),
    ("wo.kernel", "wo__kernel", ["param_grad:wo.weight", "param_grad:wo"]),
    ("input_grad", "dense_mlp_input_grad", ["dense_mlp_input_grad"]),
]

# Param grad map: (display_name, maxtext_key_substring, megatron_key_candidates)
# MaxText keys: nnx flat_state paths joined by "__"
# Megatron keys: from Argus grads category → "param_grad:{name}"
GRAD_MAP = [
    ("gate.kernel", "gate__kernel", ["param_grad:gate.weight", "param_grad:gate"]),
    ("gate.bias", "gate__bias", ["param_grad:gate.bias", "param_grad:gate_bias"]),
    ("wi_0", "wi_0__value", ["param_grad:wi_0"]),
    ("wi_1", "wi_1__value", ["param_grad:wi_1"]),
    ("wo", "wo__value", ["param_grad:wo"]),
    ("shared_wi_0", "shared_experts__wi_0", ["param_grad:shared_wi_0"]),
    ("shared_wi_1", "shared_experts__wi_1", ["param_grad:shared_wi_1"]),
    ("shared_wo", "shared_experts__wo", ["param_grad:shared_wo"]),
    ("input_grad", "moe_input_grad", ["moe_input_grad"]),
]


def _find_key(d: dict[str, np.ndarray], substring: str) -> tuple[str | None, np.ndarray | None]:
  """Find a key containing substring. Exact match first, then substring, prefer shortest."""
  if substring in d:
    return substring, d[substring]
  matches = sorted([k for k in d if substring in k], key=len)
  return (matches[0], d[matches[0]]) if matches else (None, None)


def _compare_row(
    stage: str,
    mx_val: np.ndarray | None,
    mg_val: np.ndarray | None,
) -> dict[str, Any]:
  """Build one comparison row."""
  row: dict[str, Any] = {"stage": stage}

  if mx_val is None and mg_val is None:
    return {}
  if mx_val is None:
    row["status"] = "MX_MISSING"
    row["mg_shape"] = str(mg_val.shape)
    return row
  if mg_val is None:
    row["status"] = "MG_MISSING"
    row["mx_shape"] = str(mx_val.shape)
    return row

  row["mx_shape"] = str(mx_val.shape)
  row["mg_shape"] = str(mg_val.shape)

  if mx_val.shape != mg_val.shape:
    # Auto-reshape if element count matches (e.g. (4096,16,256) vs (65536,256))
    if mx_val.size == mg_val.size:
      mx_val = mx_val.reshape(mg_val.shape)
      row["mx_shape"] = f"{row['mx_shape']}→{str(mx_val.shape)}"
    else:
      row["status"] = "SHAPE_MISMATCH"
      return row

  if np.issubdtype(mx_val.dtype, np.integer) and np.issubdtype(mg_val.dtype, np.integer):
    exact = bool(np.array_equal(mx_val, mg_val))
    if exact:
      row["status"] = "OK"
      row["set_match"] = True
    elif mx_val.ndim >= 2:
      flat_a = mx_val.reshape(-1, mx_val.shape[-1])
      flat_b = mg_val.reshape(-1, mg_val.shape[-1])
      sm = all(set(flat_a[i]) == set(flat_b[i]) for i in range(flat_a.shape[0]))
      row["status"] = "OK" if sm else "MISMATCH"
      row["set_match"] = sm
    else:
      row["status"] = "MISMATCH"
    return row

  row["status"] = "OK"
  row.update(_compare_pair(mx_val, mg_val))
  return row


def compare_forward(
    mx_data: dict[str, np.ndarray],
    mg_data: dict[str, np.ndarray],
    stages: list[str] | None,
    stage_map: list[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
  """Compare forward outputs at module boundaries."""
  if stage_map is None:
    stage_map = FORWARD_STAGE_MAP
  results = []

  # Known stages
  for display, mx_key, mg_key in stage_map:
    if stages and display not in stages:
      continue
    mx_val = mx_data.get(mx_key)
    mg_val = mg_data.get(mg_key)
    row = _compare_row(display, mx_val, mg_val)
    if row:
      results.append(row)

  # Auto-discover extra keys present on both sides
  # Exclude keys that share a name but have different semantics across frameworks
  # (e.g. Megatron moe_router_output = normalized topk weights,
  #  MaxText moe_router_output = post-bias sigmoid logits)
  _EXTRA_EXCLUDE = {"moe_router_output", "moe_topk_weights", "moe_experts_output"}
  known_mx = {mx_key for _, mx_key, _ in stage_map}
  known_mg = {mg_key for _, _, mg_key in stage_map}
  extra = (set(mx_data.keys()) - known_mx) & (set(mg_data.keys()) - known_mg) - _EXTRA_EXCLUDE
  for key in sorted(extra):
    if stages and key not in stages:
      continue
    row = _compare_row(key, mx_data[key], mg_data[key])
    if row:
      results.append(row)

  return results


def compare_backward(
    mx_grads: dict[str, np.ndarray],
    mg_grads: dict[str, np.ndarray],
    grad_map: list[tuple[str, str, list[str]]] | None = None,
) -> list[dict[str, Any]]:
  """Compare param grads + activation grads."""
  if grad_map is None:
    grad_map = GRAD_MAP
  results = []
  matched_mx: set[str] = set()
  matched_mg: set[str] = set()

  for display, mx_sub, mg_candidates in grad_map:
    mx_key, mx_val = _find_key(mx_grads, mx_sub)
    if mx_val is None:
      continue
    matched_mx.add(mx_key)

    mg_key, mg_val = None, None
    for cand in mg_candidates:
      mg_key, mg_val = _find_key(mg_grads, cand)
      if mg_val is not None:
        matched_mg.add(mg_key)
        break

    row = _compare_row(f"grad:{display}", mx_val, mg_val)
    if row:
      results.append(row)

  # Auto-match remaining keys by exact name
  for mx_key in sorted(set(mx_grads.keys()) - matched_mx):
    if mx_key in mg_grads:
      row = _compare_row(f"grad:(auto) {mx_key}", mx_grads[mx_key], mg_grads[mx_key])
      if row:
        results.append(row)
      matched_mg.add(mx_key)

  # Info about unmatched
  mx_only = sorted(set(mx_grads.keys()) - matched_mx - set(mg_grads.keys()))
  mg_only = sorted(set(mg_grads.keys()) - matched_mg)
  if mx_only:
    results.append({"stage": "grad:(info) MX-only", "status": ", ".join(mx_only)})
  if mg_only:
    results.append({"stage": "grad:(info) MG-only", "status": ", ".join(mg_only)})

  return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def print_report(layer_idx: int, rows: list[dict[str, Any]]) -> None:
  """Print a markdown-formatted comparison report for one layer."""
  if not rows:
    print(f"  Layer {layer_idx}: No comparable data found.")
    return

  # Markdown table header
  print(
      f"| {'Layer':>5} | {'Stage':<35} | {'Shape':<20} | {'Status':<14} | "
      f"{'MaxAbs':<10} | {'RelL2':<10} | {'Cosine':<10} | "
      f"{'ULPmax':>6} | {'ULPp99':>6} | {'ULPavg':>7} | {'Exact%':>7} | {'≤1ULP':>6} |"
  )
  print(
      f"|{'-' * 7}|{'-' * 37}|{'-' * 22}|{'-' * 16}|"
      f"{'-' * 12}|{'-' * 12}|{'-' * 12}|"
      f"{'-' * 8}|{'-' * 8}|{'-' * 9}|{'-' * 9}|{'-' * 8}|"
  )

  for row in rows:
    stage = row["stage"]
    shape = row.get("mx_shape", row.get("mg_shape", "?"))
    status = row.get("status", "?")

    if status == "OK" and "set_match" in row:
      # Integer comparison (e.g. topk indices) — show set_match instead of numeric stats
      print(
          f"| {layer_idx:>5} | {stage:<35} | {shape:<20} | {'OK':<14} | "
          f"{'-':<10} | {'-':<10} | {'-':<10} | "
          f"{'':>6} | {'':>6} | {'':>7} | {'':>7} | {'':>6} | set_match={row['set_match']}"
      )
    elif status == "OK":
      print(
          f"| {layer_idx:>5} | {stage:<35} | {shape:<20} | {status:<14} | "
          f"{row['max_abs']:<10.2e} | "
          f"{row['rel_l2']:<10.2e} | {row['cosine']:<10.6f} | "
          f"{row.get('ulp_max', 0):>6} | {row.get('ulp_p999', 0):>6.1f} | "
          f"{row.get('ulp_mean', 0):>7.2f} | "
          f"{row.get('bf16_exact_pct', 0):>6.1f}% | "
          f"{row.get('bf16_le1ulp_pct', 0):>5.1f}% |"
      )
    elif status in ("MISMATCH", "MX_MISSING", "MG_MISSING", "SHAPE_MISMATCH"):
      extra = ""
      if "set_match" in row:
        extra += f" set_match={row['set_match']}"
      print(
          f"| {layer_idx:>5} | {stage:<35} | {shape:<20} | {status:<14} | "
          f"{'-':<10} | {'-':<10} | {'-':<10} | "
          f"{'':>6} | {'':>6} | {'':>7} | {'':>7} | {'':>6} |{extra}"
      )
    else:
      print(f"| {layer_idx:>5} | {stage:<35} | {'':20} | {status} |")


def write_json_report(
    all_results: dict[int, list[dict]],
    output_path: str | None,
) -> None:
  """Write comparison results to a JSON file."""
  if not output_path:
    return

  def _convert(obj):
    if isinstance(obj, (np.integer,)):
      return int(obj)
    if isinstance(obj, (np.floating,)):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return obj

  summary = {f"layer_{idx}": rows for idx, rows in all_results.items()}
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=_convert)
  print(f"\nJSON report written to {output_path}")


# ---------------------------------------------------------------------------
# CI pass/fail check
# ---------------------------------------------------------------------------

# Stages that are checked for CI pass/fail (output-level comparisons).
# Stages like "input" are trivially exact; intermediate stages may be missing.
_CI_CHECK_STAGES = {"moe_output", "dense_mlp_output", "shared_expert_output"}


def _ci_check(all_results: dict[int, list[dict]], args: argparse.Namespace) -> int:
  """Check comparison results against CI thresholds. Returns 0 on pass, 1 on fail."""
  max_rel_l2 = args.max_rel_l2
  min_cosine = args.min_cosine
  max_max_abs = args.max_max_abs

  print("\n" + "=" * 60)
  print("  CI Assertion Check")
  print(f"  Thresholds: rel_l2 < {max_rel_l2}, cosine > {min_cosine}, max_abs < {max_max_abs}")
  print(f"  Checked stages: {sorted(_CI_CHECK_STAGES)}")
  print("=" * 60)

  all_passed = True
  checked = 0

  for layer_idx, rows in sorted(all_results.items()):
    for row in rows:
      stage = row.get("stage", "")
      status = row.get("status", "")

      # Only check output stages with numeric metrics
      if stage not in _CI_CHECK_STAGES:
        continue
      if status != "OK":
        print(f"  [FAIL] layer {layer_idx} / {stage}: status={status}")
        all_passed = False
        checked += 1
        continue

      rel_l2 = row.get("rel_l2", float("inf"))
      cosine = row.get("cosine", 0.0)
      max_abs = row.get("max_abs", float("inf"))
      checked += 1

      passed = rel_l2 < max_rel_l2 and cosine > min_cosine and max_abs < max_max_abs
      tag = "PASS" if passed else "FAIL"
      print(f"  [{tag}] layer {layer_idx} / {stage}  " f"rel_l2={rel_l2:.3e}  cosine={cosine:.6f}  max_abs={max_abs:.3e}")
      if not passed:
        all_passed = False

  if checked == 0:
    print("  [FAIL] No output stages found to check!")
    all_passed = False

  print(f"\n  Overall: {'PASS' if all_passed else 'FAIL'} ({checked} stages checked)")
  print(f"{'=' * 60}")
  return 0 if all_passed else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _slice_batch(data: dict[str, np.ndarray], seq_len: int, max_batch: int) -> dict[str, np.ndarray]:
  """Slice all 2D/3D tensors in *data* along the batch dimension.

  Argus dump tensors are typically 2D ``(tokens, hidden)`` where
  ``tokens = seq_len * batch``.  This reshapes to ``(seq, batch, hidden)``,
  slices to ``[:, :max_batch, :]``, and flattens back to 2D so downstream
  code (forward/backward) works unchanged.

  Tensors that don't divide evenly by *seq_len* or have ndim < 2 are left
  untouched.  Returns a **new** dict (original is not mutated).
  """
  out: dict[str, np.ndarray] = {}
  for key, arr in data.items():
    if arr.ndim == 2:
      total_tokens, hidden = arr.shape
      if total_tokens % seq_len == 0:
        batch = total_tokens // seq_len
        if batch > max_batch:
          arr = arr.reshape(seq_len, batch, hidden)[:, :max_batch, :].reshape(seq_len * max_batch, hidden)
    elif arr.ndim == 3:
      _, b, _ = arr.shape
      if b > max_batch:
        arr = arr[:, :max_batch, :]
    out[key] = arr
  return out


def _detect_mode(dump_dir: str, layer_idx: int) -> str:
  """Auto-detect whether a layer is MoE or dense based on available tensors."""
  _, list_tensors = _import_argus()
  all_names = list_tensors(dump_dir)
  has_moe = any(f"layer_{layer_idx}_moe_" in n for n in all_names)
  has_dense = any(f"layer_{layer_idx}_dense_mlp_" in n for n in all_names)
  if has_moe and not has_dense:
    return "moe"
  if has_dense and not has_moe:
    return "dense"
  if has_moe and has_dense:
    return "moe"  # prefer MoE if both present
  return "moe"  # default


def run(args: argparse.Namespace) -> int:
  """Run cross-framework comparison for all specified layers. Returns exit code."""
  if args.ling2_profile:
    args.hidden_size = 2048
    args.ffn_hidden_size = 512
    args.shared_ffn_hidden_size = 2048
    args.dense_ffn_hidden_size = getattr(args, "dense_ffn_hidden_size", 5120) or 5120
    args.num_experts = 256
    args.top_k = 8
    args.enable_shared_expert = True
    args.routed_score_func = "sigmoid"
    args.routed_scaling_factor = getattr(args, "routed_scaling_factor", 2.5) or 2.5
    args.norm_topk_prob = False
    args.n_routing_groups = 8
    args.topk_routing_group = 4
    args.routed_bias = True

  layers = [int(x) for x in args.layers.split(",")] if args.layers else [0]
  stages = None
  if args.stages and args.stages != "all":
    stages = [s.strip() for s in args.stages.split(",")]

  all_results: dict[int, list[dict]] = {}

  for layer_idx in layers:
    # Determine mode per layer
    mode = args.mode
    if mode == "auto":
      mode = _detect_mode(args.megatron_dump_dir, layer_idx)
      print(f"  Auto-detected mode: {mode}")

    print(f"\n{'=' * 135}")
    print(f"Layer {layer_idx} (mode={mode})")
    print(f"{'=' * 135}")

    # Load Megatron data via Argus
    print("  Loading Megatron Argus dump...")
    mg_data = load_megatron_data(
        args.megatron_dump_dir,
        layer_idx,
        micro_batch_size=args.micro_batch_size,
        load_backward=args.with_backward,
        mode=mode,
        dp_mode=args.dp_mode,
    )
    if not mg_data:
      print(f"  WARNING: No data found for layer {layer_idx}. Skipping.")
      continue

    print(f"  Megatron tensors ({len(mg_data)}):")
    for k, v in sorted(mg_data.items()):
      print(f"    {k:35s} shape={str(v.shape):25s} dtype={v.dtype}")

    if mode == "dense":
      _run_dense_layer(args, mg_data, layer_idx, stages, all_results)
    else:
      _run_moe_layer(args, mg_data, layer_idx, stages, all_results)

  write_json_report(all_results, args.output_json)
  return _ci_check(all_results, args) if args.ci else 0


def _run_moe_layer(args, mg_data, layer_idx, stages, all_results):
  """Run MoE comparison for one layer."""
  # Normalize new-format keys (e.g. moe_input_0 → moe_input)
  _normalize_dump_keys(mg_data)

  # Slice batch dimension if --max-batch is set (reduces HBM for backward)
  max_batch = getattr(args, "max_batch", 0)
  if max_batch > 0:
    mg_data = _slice_batch(mg_data, args.seq_len, max_batch)
    print(f"  Sliced batch to max_batch={max_batch} (seq_len={args.seq_len})")

  # Derive top-k indices/weights from Megatron routing_map + gate scores
  routing_map = mg_data.get("moe_router_output_1")  # (tokens, experts) bool
  gate_scores = mg_data.get("moe_router_output")  # (tokens, experts) float32
  if routing_map is not None and gate_scores is not None:
    # Extract top-k indices from bool routing_map using masked gate scores
    topk = args.top_k
    # Set non-selected experts to -inf, then argsort to get top-k
    masked_scores = np.where(routing_map, gate_scores, -np.inf)
    # Get top-k indices per token by argpartition + sort
    mg_topk_idx = np.argsort(-masked_scores, axis=-1)[:, :topk].astype(np.int32)
    # Gather the corresponding weights
    mg_topk_wts = np.take_along_axis(gate_scores, mg_topk_idx.astype(np.intp), axis=-1).astype(np.float32)
    mg_data["moe_topk_indices"] = mg_topk_idx
    mg_data["moe_topk_weights"] = mg_topk_wts
    print(f"  Derived Megatron top-k indices/weights from routing_map (shape={mg_topk_idx.shape})")

  # Get MoE input
  mg_input = mg_data.get("moe_input")
  if mg_input is None:
    print("  ERROR: 'moe_input' (forward:layer_{i}_moe_input) not found. Cannot run MaxText forward.")
    return

  # Build model (lazy, once)
  if not hasattr(args, "cached_moe_model"):
    print("  Building MaxText MoE module...")
    args.cached_moe_model, args.cached_moe_cfg, args.cached_moe_mesh = build_maxtext_moe(args)
  model = args.cached_moe_model

  # Load Orbax weights
  if args.orbax_ckpt_path:
    orbax_moe_idx = layer_idx - args.num_dense_layers
    load_orbax_weights(
        model,
        args.orbax_ckpt_path,
        layer_idx,
        args.orbax_layer_prefix,
        args.enable_shared_expert,
        orbax_moe_idx=orbax_moe_idx,
    )

  # Forward
  print(f"  Running MaxText MoE forward (input shape={mg_input.shape})...")
  mx_data = run_maxtext_forward(model, mg_input, args)
  print(f"  MaxText tensors ({len(mx_data)}):")
  for k, v in sorted(mx_data.items()):
    print(f"    {k:35s} shape={str(v.shape):25s} dtype={v.dtype}")

  print("\n  --- Forward Comparison ---\n")
  rows = compare_forward(mx_data, mg_data, stages)

  # Backward
  if args.with_backward:
    print("\n  --- Backward Comparison ---\n")
    mg_grads = {k: v for k, v in mg_data.items() if k.endswith("_grad") or k.startswith("param_grad:")}
    mg_output_grad = mg_grads.get("moe_output_grad")
    if mg_output_grad is None:
      print("  ERROR: 'moe_output_grad' not found. Cannot run aligned backward.")
      print(f"  Available grad keys: {sorted(mg_grads.keys())}")
    else:
      print(f"  Using Megatron moe_output_grad (shape={mg_output_grad.shape}) as upstream gradient")
      print("  Running MaxText backward (vjp)...")
      mx_grads = run_maxtext_backward(model, mg_input, mg_output_grad, args)
      if mg_grads:
        print(f"  Megatron grad tensors ({len(mg_grads)}):")
        for k, v in sorted(mg_grads.items()):
          print(f"    {k:35s} shape={str(v.shape):25s} dtype={v.dtype}")
        rows += compare_backward(mx_grads, mg_grads)
      else:
        print("  WARNING: No Megatron backward/grad data in this dump.")

  print()
  print_report(layer_idx, rows)
  all_results[layer_idx] = rows


def _resolve_mg_key(mg_data, *candidates):
  """Return the first key that exists in mg_data, or None."""
  for k in candidates:
    if k in mg_data:
      return k
  return None


def _normalize_dump_keys(mg_data):
  """Normalize new-format dump keys (with _0 suffix) to canonical names.

  New argus dump appends '_0' to module input/output names (e.g.
  'moe_input_0', 'dense_mlp_output_0').  Map them to canonical names
  used by stage maps and downstream code.  Only adds the alias if the
  canonical name is not already present.
  """
  # Build aliases dynamically: any key ending in '_0' where removing
  # '_0' gives a recognized canonical name pattern.
  canonical_suffixes = ("_input", "_output", "_input_grad", "_output_grad")
  aliases = {}
  for key in list(mg_data.keys()):
    if key.endswith("_0"):
      base = key[:-2]  # strip '_0'
      if any(base.endswith(s) for s in canonical_suffixes):
        aliases[key] = base

  for new_key, canonical in aliases.items():
    if canonical not in mg_data:
      mg_data[canonical] = mg_data[new_key]


def _run_dense_layer(args, mg_data, layer_idx, stages, all_results):
  """Run dense MLP comparison for one layer."""
  # Normalize new-format keys (e.g. dense_mlp_input_0 → dense_mlp_input)
  _normalize_dump_keys(mg_data)

  # Slice batch dimension if --max-batch is set
  max_batch = getattr(args, "max_batch", 0)
  if max_batch > 0:
    mg_data = _slice_batch(mg_data, args.seq_len, max_batch)
    print(f"  Sliced batch to max_batch={max_batch} (seq_len={args.seq_len})")

  # Get dense MLP input
  mg_input = mg_data.get("dense_mlp_input")
  if mg_input is None:
    avail = sorted(k for k in mg_data if "dense_mlp" in k or "mlp" in k)
    print("  ERROR: 'dense_mlp_input' not found.")
    print(f"  Available MLP-related keys: {avail}")
    return

  # Build model (lazy, once)
  if not hasattr(args, "cached_dense_model"):
    print("  Building MaxText dense MLP module...")
    args.cached_dense_model, args.cached_dense_cfg, args.cached_dense_mesh = build_maxtext_dense_mlp(args)
  model = args.cached_dense_model

  # Load Orbax weights
  if args.orbax_ckpt_path:
    load_orbax_dense_weights(
        model, args.orbax_ckpt_path, layer_idx, args.orbax_layer_prefix, fused_mlp=getattr(args, "fused_mlp", False)
    )

  # Forward
  print(f"  Running MaxText dense MLP forward (input shape={mg_input.shape})...")
  mx_data = run_maxtext_dense_forward(model, mg_input, args)
  print(f"  MaxText tensors ({len(mx_data)}):")
  for k, v in sorted(mx_data.items()):
    print(f"    {k:35s} shape={str(v.shape):25s} dtype={v.dtype}")

  print("\n  --- Forward Comparison (Dense MLP) ---\n")
  rows = compare_forward(mx_data, mg_data, stages, stage_map=DENSE_FORWARD_STAGE_MAP)

  # Backward
  if args.with_backward:
    print("\n  --- Backward Comparison (Dense MLP) ---\n")
    mg_grads = {k: v for k, v in mg_data.items() if k.endswith("_grad") or k.startswith("param_grad:")}
    mg_output_grad = mg_grads.get("dense_mlp_output_grad")
    if mg_output_grad is None:
      print("  ERROR: 'dense_mlp_output_grad' not found. Cannot run aligned backward.")
      print(f"  Available grad keys: {sorted(mg_grads.keys())}")
    else:
      print(f"  Using Megatron dense_mlp_output_grad (shape={mg_output_grad.shape}) as upstream gradient")
      print("  Running MaxText dense backward (vjp)...")
      mx_grads = run_maxtext_dense_backward(model, mg_input, mg_output_grad, args)
      if mg_grads:
        print(f"  Megatron grad tensors ({len(mg_grads)}):")
        for k, v in sorted(mg_grads.items()):
          print(f"    {k:35s} shape={str(v.shape):25s} dtype={v.dtype}")
        dense_grad_map = DENSE_GRAD_MAP_FUSED if getattr(args, "fused_mlp", False) else DENSE_GRAD_MAP
        rows += compare_backward(mx_grads, mg_grads, grad_map=dense_grad_map)
      else:
        print("  WARNING: No Megatron backward/grad data in this dump.")

  print()
  print_report(layer_idx, rows)
  all_results[layer_idx] = rows


def parse_args() -> argparse.Namespace:
  """Parse command-line arguments for cross-framework MoE comparison."""
  parser = argparse.ArgumentParser(
      description="Cross-framework MoE precision comparison via Argus dump.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "--megatron-dump-dir", required=True, help="Megatron Argus dump dir (e.g. .../step_1, containing rank_* subdirs)."
  )
  parser.add_argument("--orbax-ckpt-path", type=str, default=None, help="Orbax checkpoint path for MaxText weights.")
  parser.add_argument("--orbax-layer-prefix", type=str, default=None, help="Explicit Orbax key prefix for the MoE layer.")
  parser.add_argument(
      "--dp-mode",
      type=str,
      default="default",
      choices=["default", "replica"],
      help="Argus dp_mode. 'replica' loads single-rank data with fallback for ep+dp shard issues. "
      "'default' uses upstream argus multi-rank gather.",
  )
  parser.add_argument("--layers", type=str, default="0", help="Comma-separated layer indices (default: 0).")
  parser.add_argument(
      "--mode",
      type=str,
      default="auto",
      choices=["auto", "moe", "dense"],
      help="Comparison mode: 'moe', 'dense', or 'auto' (detect from dump).",
  )
  parser.add_argument("--stages", type=str, default="all", help="Comma-separated forward stages to compare, or 'all'.")
  parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output path.")
  parser.add_argument("--ling2-profile", action="store_true", help="Apply LING2 MoE config preset.")
  parser.add_argument("--micro-batch-size", type=int, default=2, help="Per-rank micro batch size for Argus load_tensor.")
  parser.add_argument(
      "--with-backward", action=argparse.BooleanOptionalAction, default=False, help="Also compare backward gradients."
  )

  # Model dimensions (overridden by --ling2-profile)
  parser.add_argument("--hidden-size", type=int, default=2048)
  parser.add_argument("--ffn-hidden-size", type=int, default=512)
  parser.add_argument("--shared-ffn-hidden-size", type=int, default=2048)
  parser.add_argument(
      "--dense-ffn-hidden-size", type=int, default=5120, help="Dense MLP intermediate dim (default: 5120 for LING2)."
  )
  parser.add_argument("--num-experts", type=int, default=256)
  parser.add_argument("--top-k", type=int, default=8)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--seq-len", type=int, default=4)
  parser.add_argument(
      "--max-batch",
      type=int,
      default=0,
      help="Slice input batch to at most N samples to reduce HBM usage for backward. "
      "0 = no slicing (default). E.g. --max-batch 4 reduces 16-batch dump to 4.",
  )

  # MoE config
  parser.add_argument("--enable-shared-expert", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--routed-score-func", type=str, default="sigmoid")
  parser.add_argument("--routed-scaling-factor", type=float, default=2.5)
  parser.add_argument("--norm-topk-prob", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--n-routing-groups", type=int, default=8)
  parser.add_argument("--topk-routing-group", type=int, default=4)
  parser.add_argument("--routed-bias", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
      "--num-dense-layers",
      type=int,
      default=1,
      help="Number of dense decoder layers before MoE layers (default: 1). "
      "Used to compute Orbax moe_layers index: orbax_idx = layer_idx - num_dense_layers.",
  )

  # Compute
  parser.add_argument("--compute-dtype", type=str, default="bf16", choices=["bf16", "fp32"])
  parser.add_argument(
      "--matmul-precision", type=str, default="default", choices=["default", "high", "highest", "bfloat16", "float32"]
  )
  parser.add_argument("--activations-in-float32", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument(
      "--fused-mlp",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Use fused MLP (single wi kernel) instead of separate wi_0/wi_1.",
  )

  # CI pass/fail
  parser.add_argument(
      "--ci",
      action="store_true",
      help="Enable CI mode: assert output precision meets thresholds, exit non-zero on failure.",
  )
  parser.add_argument(
      "--max-rel-l2", type=float, default=0.01, help="CI threshold: maximum relative L2 norm (default: 0.01)."
  )
  parser.add_argument(
      "--min-cosine", type=float, default=0.999, help="CI threshold: minimum cosine similarity (default: 0.999)."
  )
  parser.add_argument(
      "--max-max-abs", type=float, default=0.1, help="CI threshold: maximum absolute difference (default: 0.1)."
  )

  return parser.parse_args()


if __name__ == "__main__":
  sys.exit(run(parse_args()))
