"""CI test: compare MaxText MLA against Megatron argus dumps.

Usage:
    python tools/mla_alignment/test_mla_ci.py \
        --dump-dir /path/to/step_1 \
        --ckpt-path /path/to/orbax/items \
        --atol 1e-2 --rtol 1e-2
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import argus
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from maxtext.utils import maxtext_utils
from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.layers.attention_mla import MLA


# ── CI pass/fail thresholds ──────────────────────────────────────────────────

MAX_REL_L2 = 1e-2
MAX_OUT_OF_TOL_PCT = 1.0  # percent
MAX_MEAN_ABS_DIFF = 1e-2
IGNORE_ULP = 2
ORIG_DTYPE = "bfloat16"

# MLA layers to test (MTP layer 20 has no MLA weights in checkpoint)
MLA_LAYER_INDICES = (4, 9, 14, 19)


# ── Orbax checkpoint helpers ──────────────────────────────────────────────────


def _is_leaf(x: Any) -> bool:
  """Check if x is an array-like leaf node."""
  return hasattr(x, "shape") and hasattr(x, "dtype")


def _flatten_tree(tree: Any, prefix: str = "") -> dict[str, Any]:
  """Recursively flatten a nested tree into a flat dict of leaves."""
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
    expected_shape: tuple[int, ...] | None = None,
    required: bool = True,
) -> tuple[np.ndarray | None, str | None]:
  """Find and return a tensor matching the scope marker and suffix candidates."""
  matches: list[str] = []
  for k in flat.keys():
    if key_scope_marker and key_scope_marker not in k:
      continue
    if any(k.endswith(sfx) for sfx in suffix_candidates):
      matches.append(k)

  if not matches:
    if required:
      raise KeyError(
          f"No tensor matched marker='{key_scope_marker}' suffixes={suffix_candidates} "
          f"expected_shape={expected_shape}"
      )
    return None, None

  if expected_shape is not None:
    shape_matched = [k for k in matches if tuple(np.asarray(flat[k]).shape) == expected_shape]
    candidates = shape_matched if shape_matched else matches
  else:
    candidates = matches

  candidates = sorted(candidates, key=lambda x: (len(x), x))
  chosen = candidates[0]
  arr = np.asarray(flat[chosen], dtype=np.float32)
  if expected_shape is not None and tuple(arr.shape) != expected_shape:
    raise ValueError(
        f"Tensor shape mismatch for key={chosen}: got {arr.shape}, expected {expected_shape}. "
        f"Matches considered: {matches}"
    )
  return arr, chosen


def _load_orbax_checkpoint(orbax_ckpt_path: str) -> dict[str, Any]:
  """Load and flatten an Orbax checkpoint."""
  import orbax.checkpoint as ocp  # pylint: disable=import-outside-toplevel

  # Auto-detect items/ subdirectory (composite checkpoint format)
  items_subdir = os.path.join(orbax_ckpt_path, "items")
  if os.path.isdir(items_subdir) and os.path.exists(os.path.join(items_subdir, "_METADATA")):
    orbax_ckpt_path = items_subdir

  ckpt = ocp.StandardCheckpointer().restore(orbax_ckpt_path)
  return _flatten_tree(ckpt)


def _load_mla_weights_from_orbax(
    flat: dict[str, Any],
    layer_index: int,
    layer_prefix: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
  """Load MLA weights from a flattened Orbax checkpoint tree."""
  marker = f"layers_{layer_index}"
  pfx = layer_prefix or f"params.params.decoder.layers_{layer_index}.ALDenseBlock_0"

  weight_specs = [
      (
          "pre_norm_scale",
          f"{pfx}.pre_self_attention_layer_norm.scale",
          ".pre_self_attention_layer_norm.scale",
      ),
      ("wq_a", f"{pfx}.self_attention.attention.wq_a.kernel", ".wq_a.kernel"),
      (
          "q_norm_scale",
          f"{pfx}.self_attention.attention.q_norm.scale",
          ".q_norm.scale",
      ),
      ("wq_b", f"{pfx}.self_attention.attention.wq_b.kernel", ".wq_b.kernel"),
      ("wkv_a", f"{pfx}.self_attention.attention.wkv_a.kernel", ".wkv_a.kernel"),
      (
          "kv_norm_scale",
          f"{pfx}.self_attention.attention.kv_norm.scale",
          ".kv_norm.scale",
      ),
      ("wkv_b", f"{pfx}.self_attention.attention.wkv_b.kernel", ".wkv_b.kernel"),
      ("out", f"{pfx}.self_attention.attention.out.kernel", ".out.kernel"),
  ]

  weights: dict[str, np.ndarray] = {}
  key_map: dict[str, str] = {}
  for name, primary_suffix, fallback_suffix in weight_specs:
    arr, chosen_key = _choose_tensor_key(
        flat,
        marker,
        (primary_suffix, fallback_suffix),
        expected_shape=None,
        required=True,
    )
    weights[name] = arr
    key_map[name] = chosen_key or ""

  return weights, key_map


# ── MaxText config / MLA builder ─────────────────────────────────────────────


def build_config(batch_size: int, seq_len: int, dtype_str: str | None, attention_kernel: str):
  """Initialize MaxText config for MLA alignment testing."""
  extra_kwargs = {}
  if dtype_str is not None:
    extra_kwargs["dtype"] = dtype_str
    extra_kwargs["weight_dtype"] = dtype_str

  cfg = pyconfig.initialize(
      [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
      model_name="ling2",
      run_name="mla_alignment_test",
      per_device_batch_size=batch_size,
      max_target_length=seq_len,
      max_prefill_predict_length=seq_len,
      enable_checkpointing=False,
      attention=attention_kernel,
      matmul_precision="highest",
      float32_qk_product=True,
      float32_logits=True,
      activations_in_float32=True,
      override_model_config=True,
      **extra_kwargs,
  )
  return cfg


def build_mla(cfg, mesh, dtype):
  """Construct an MLA module following the attention_test_util pattern."""
  B = cfg.global_batch_size_to_train_on
  L = cfg.max_target_length

  mla = MLA(
      config=cfg,
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      inputs_q_shape=(B, L, cfg.base_emb_dim),
      inputs_kv_shape=(B, L, cfg.base_emb_dim),
      max_target_length=cfg.max_target_length,
      max_prefill_predict_length=cfg.max_prefill_predict_length,
      mesh=mesh,
      attention_kernel=cfg.attention,
      dtype=dtype,
      weight_dtype=dtype,
      dropout_rate=cfg.dropout_rate,
      attention_type=cfg.attention_type,
      q_lora_rank=cfg.q_lora_rank,
      kv_lora_rank=cfg.kv_lora_rank,
      qk_nope_head_dim=cfg.qk_nope_head_dim,
      qk_rope_head_dim=cfg.qk_rope_head_dim,
      v_head_dim=cfg.v_head_dim,
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      model_mode=MODEL_MODE_TRAIN,
      rngs=nnx.Rngs(params=42),
  )
  return mla


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
  """Parse command-line arguments for MLA CI alignment test."""
  parser = argparse.ArgumentParser(description="CI test: compare MaxText MLA against Megatron argus dumps")
  parser.add_argument("--dump-dir", required=True, help="Megatron argus dump dir (contains rank_* subdirs)")
  parser.add_argument("--ckpt-path", required=True, help="Orbax checkpoint path (e.g. .../items/)")
  parser.add_argument(
      "--layers",
      type=str,
      default=None,
      help="Comma-separated layer indices (e.g. '3,7,11'). " "If omitted, auto-discovers MLA layers from dump.",
  )
  parser.add_argument("--atol", type=float, default=1e-2)
  parser.add_argument("--rtol", type=float, default=1e-2)
  parser.add_argument(
      "--batch-size", type=int, default=2, help="Per-device batch size (must match dump's micro-batch-size)"
  )
  parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length (must match dump)")
  parser.add_argument(
      "--dtype",
      type=str,
      default=None,
      choices=["float32", "bfloat16"],
      help="Override dtype (default: from ling2.yml)",
  )
  parser.add_argument(
      "--ckpt-layer-offset",
      type=int,
      default=-1,
      help="Offset from dump layer index to checkpoint layer index. "
      "ckpt_layer = dump_layer + offset. Default -1 "
      "(dense_layers_0 occupies layer 0 in dump).",
  )
  return parser.parse_args()


# ── Layer discovery ──────────────────────────────────────────────────────────


def discover_mla_layers(dump_dir: str) -> list[int]:
  """Auto-discover layer indices that have mla_attn tensors in the dump."""
  tensors = argus.list_tensors(dump_dir, category="forward")
  pattern = re.compile(r"(?:forward:)?layer_(\d+)_mla_attn_input_0")
  layers = sorted(int(m.group(1)) for t in tensors if (m := pattern.match(t)))
  return layers


# ── Megatron dump loading ────────────────────────────────────────────────────


def load_megatron_tensors(
    dump_dir: str, layer_idx: int, micro_batch_size: int = 2, seq_len: int = 4096, num_dp_ranks: int = 8
) -> dict[str, np.ndarray]:
  """Load forward/backward MLA tensors for a single layer from Megatron dump.

  Uses dp_mode="replica" to load a single rank's data (S, B_micro, H),
  then transposes to (B_micro, S, H).
  """

  def _load_and_transpose(name: str) -> np.ndarray:
    arr = argus.load_tensor(dump_dir, name, dp_mode="replica")
    if arr.ndim == 3:
      arr = np.transpose(arr, (1, 0, 2))  # (S, B, H) -> (B, S, H)
    return arr

  prefix = f"layer_{layer_idx}_mla_attn"
  tensors = {}
  tensors["input"] = _load_and_transpose(f"forward:{prefix}_input_0")
  tensors["output"] = _load_and_transpose(f"forward:{prefix}_output_0")
  tensors["output_grad"] = _load_and_transpose(f"backward:{prefix}_output_grad_0")
  tensors["input_grad"] = _load_and_transpose(f"backward:{prefix}_input_grad_0")
  return tensors


# ── MaxText MLA construction ─────────────────────────────────────────────────


def setup_config_and_mesh(batch_size: int, seq_len: int, dtype_str: str | None, num_devices: int = 1):
  """Build MaxText config and mesh.

  Args:
      batch_size: Per-device batch size (matches dump's micro-batch-size).
      num_devices: Number of devices (default 1 for single-rank replica data).
  """
  cfg = build_config(batch_size, seq_len, dtype_str, attention_kernel="flash")
  jax_dtype = jnp.float32 if cfg.dtype == "float32" else jnp.bfloat16
  cfg._flat_config["ici_data_parallelism"] = num_devices  # pylint: disable=protected-access
  cfg._flat_config["ici_fsdp_parallelism"] = 1  # pylint: disable=protected-access
  devices = jax.devices()[:num_devices]
  devices_array = maxtext_utils.create_device_mesh(cfg, devices=devices)
  mesh = Mesh(devices_array, cfg.mesh_axes)
  global_batch = batch_size * num_devices
  cfg._flat_config["global_batch_size_to_train_on"] = global_batch  # pylint: disable=protected-access
  return cfg, mesh, jax_dtype, num_devices


def build_mla_for_layer(cfg, mesh, jax_dtype, ckpt_flat: dict, ckpt_layer_idx: int):
  """Build MLA module with Orbax weights for a specific checkpoint layer.

  Args:
      ckpt_layer_idx: Layer index in the Orbax checkpoint (e.g. 3 for moe_layers_3).

  Returns mla ready for forward/backward. No pre_norm — dump provides
  post-layernorm hidden states as MLA input.
  """
  with mesh:
    mla = build_mla(cfg, mesh, jax_dtype)

  # Load and inject Orbax weights for this checkpoint layer
  orbax_weights, _ = _load_mla_weights_from_orbax(ckpt_flat, ckpt_layer_idx, None)
  mla.wq_a.kernel.value = jnp.array(orbax_weights["wq_a"])
  mla.q_norm.scale.value = jnp.array(orbax_weights["q_norm_scale"])
  mla.wq_b.kernel.value = jnp.array(orbax_weights["wq_b"])
  mla.wkv_a.kernel.value = jnp.array(orbax_weights["wkv_a"])
  mla.kv_norm.scale.value = jnp.array(orbax_weights["kv_norm_scale"])
  mla.wkv_b.kernel.value = jnp.array(orbax_weights["wkv_b"])
  mla.out.kernel.value = jnp.array(orbax_weights["out"])

  return mla


# ── Forward / backward ───────────────────────────────────────────────────────


def run_forward(mla, hidden_states_np, mesh):
  """Run MLA forward pass with global-batch input.

  Input is already the full global batch (B_global, S, H) from all DP ranks.
  JAX DP mesh shards it across devices on the batch dim automatically.
  """
  global_batch, seq_len = hidden_states_np.shape[:2]

  positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (global_batch, seq_len))
  seg_ids = jnp.ones((global_batch, seq_len), dtype=jnp.int32)
  # Dump stores bf16 values as fp32 dtype — cast to bf16 for correct forward
  x = jnp.array(hidden_states_np, dtype=jnp.bfloat16)

  with mesh:
    output, _ = mla(
        x,
        x,
        inputs_positions=positions,
        decoder_segment_ids=seg_ids,
        model_mode=MODEL_MODE_TRAIN,
        deterministic=True,
    )
  output_np = np.array(output, dtype=np.float32)
  return output_np, positions, seg_ids


def run_backward(mla, hidden_states_np, output_grad_np, positions, seg_ids, mesh):
  """Run MLA backward via jax.vjp, return input_grad as numpy (B_global, S, H).

  Uses nnx.split/merge for correct NNX parameter tracing under jax.vjp.
  Input is already full global batch.
  """
  # Dump stores bf16 values as fp32 dtype — cast to bf16
  x = jnp.array(hidden_states_np, dtype=jnp.bfloat16)
  output_grad = jnp.array(output_grad_np, dtype=jnp.bfloat16)

  mla_graphdef, mla_params, mla_rest = nnx.split(mla, nnx.Param, ...)

  def fwd_fn(mla_params, h):
    mla_model = nnx.merge(mla_graphdef, mla_params, mla_rest)
    out, _ = mla_model(
        h,
        h,
        inputs_positions=positions,
        decoder_segment_ids=seg_ids,
        model_mode=MODEL_MODE_TRAIN,
        deterministic=True,
    )
    return out

  with mesh:
    _, vjp_fn = jax.vjp(fwd_fn, mla_params, x)
    _, input_grad = vjp_fn(output_grad)

  return np.array(input_grad, dtype=np.float32)


# ── In-memory tensor comparison ──────────────────────────────────────────────


@dataclass
class TensorCompareResult:
  """Result of comparing two tensors with CI pass/fail criteria."""

  name: str
  shape: tuple
  cosine_sim: float
  rel_l2: float
  mean_abs_diff: float
  max_abs_diff: float
  num_elements: int
  num_out_of_tol: int
  num_ulp_ignored: int
  passed: bool


def compare_tensors(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    atol: float,
    rtol: float,
    ignore_ulp: int = IGNORE_ULP,
    orig_dtype: str = ORIG_DTYPE,
) -> TensorCompareResult:
  """Compare two tensors in memory with ULP filtering and CI pass/fail criteria."""
  from argus.compare.comparator import (  # pylint: disable=import-outside-toplevel
      _cosine_similarity,
  )

  actual = actual.astype(np.float32).ravel()
  expected = expected.astype(np.float32).ravel()
  num_elements = actual.size

  # ULP filtering: mask differences <= ignore_ulp ULPs in orig_dtype
  num_ulp_ignored = 0
  if ignore_ulp > 0:
    if orig_dtype == "bfloat16":
      cast_actual = np.asarray(jnp.array(actual, dtype=jnp.bfloat16), dtype=np.float32)
      cast_expected = np.asarray(jnp.array(expected, dtype=jnp.bfloat16), dtype=np.float32)
      # bfloat16 = upper 16 bits of float32; shift right to get bfloat16 integer repr
      actual_int = cast_actual.view(np.int32) >> 16
      expected_int = cast_expected.view(np.int32) >> 16
    elif orig_dtype == "float16":
      cast_actual = actual.astype(np.float16)
      cast_expected = expected.astype(np.float16)
      actual_int = cast_actual.view(np.int16).astype(np.int32)
      expected_int = cast_expected.view(np.int16).astype(np.int32)
    else:
      cast_actual = actual
      cast_expected = expected
      actual_int = cast_actual.view(np.int32)
      expected_int = cast_expected.view(np.int32)
    ulp_dist = np.abs(actual_int - expected_int).astype(np.int64)
    ulp_mask = ulp_dist <= ignore_ulp
    num_ulp_ignored = int(np.sum(ulp_mask & (actual != expected)))
    expected = np.where(ulp_mask, actual, expected)

  # Metrics
  diff = actual - expected
  abs_diff = np.abs(diff)
  cosine_sim = float(_cosine_similarity(actual, expected))
  actual_l2 = float(np.linalg.norm(actual))
  diff_l2 = float(np.linalg.norm(diff))
  rel_l2 = diff_l2 / actual_l2 if actual_l2 > 0 else 0.0
  mean_abs_diff = float(np.mean(abs_diff))
  max_abs_diff = float(np.max(abs_diff))
  close_mask = np.isclose(actual, expected, atol=atol, rtol=rtol)
  num_out_of_tol = int(np.sum(~close_mask))
  out_of_tol_pct = 100.0 * num_out_of_tol / num_elements if num_elements > 0 else 0.0

  passed = rel_l2 < MAX_REL_L2 and out_of_tol_pct < MAX_OUT_OF_TOL_PCT and mean_abs_diff < MAX_MEAN_ABS_DIFF

  return TensorCompareResult(
      name=name,
      shape=tuple(actual.shape),
      cosine_sim=cosine_sim,
      rel_l2=rel_l2,
      mean_abs_diff=mean_abs_diff,
      max_abs_diff=max_abs_diff,
      num_elements=num_elements,
      num_out_of_tol=num_out_of_tol,
      num_ulp_ignored=num_ulp_ignored,
      passed=passed,
  )


def print_result(r: TensorCompareResult) -> None:
  """Print a single tensor comparison result."""
  status = "PASS" if r.passed else "FAIL"
  out_of_tol_pct = 100.0 * r.num_out_of_tol / r.num_elements if r.num_elements > 0 else 0.0
  print(f"  [{status}] {r.name}")
  print(
      f"         cosine={r.cosine_sim:.6f}  rel_l2={r.rel_l2:.3e}  "
      f"mean_diff={r.mean_abs_diff:.3e}  max_diff={r.max_abs_diff:.3e}"
  )
  print(
      f"         out_of_tol={r.num_out_of_tol}/{r.num_elements} ({out_of_tol_pct:.4f}%)  "
      f"ulp_ignored={r.num_ulp_ignored}"
  )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
  args = parse_args()

  # 1. Discover or parse layers
  if args.layers and args.layers.strip():
    layers = [int(x.strip()) for x in args.layers.split(",")]
  else:
    layers = discover_mla_layers(args.dump_dir)
    if not layers:
      print("ERROR: No MLA layers found in dump")
      return 1

  print(f"Discovered MLA layers: {layers}")

  # 2. Build config and mesh ONCE (pyconfig.initialize is not reentrant)
  cfg, mesh, jax_dtype, num_devices = setup_config_and_mesh(args.batch_size, args.seq_len, args.dtype)

  # MTP layer (layer 20) has no MLA weights in checkpoint; only test main MLA layers
  layers = [l for l in layers if l in MLA_LAYER_INDICES]
  print(f"Testing MLA layers: {layers}")

  # 3. Load Orbax checkpoint (once, shared across layers)
  print(f"\n=== Loading Orbax checkpoint: {args.ckpt_path} ===")
  ckpt_flat = _load_orbax_checkpoint(args.ckpt_path)
  print(f"  Flattened tree: {len(ckpt_flat)} leaves")

  # 4. Per-layer: forward, backward, compare in memory
  all_results: list[TensorCompareResult] = []

  for layer_idx in layers:
    print(f"\n{'='*60}")
    print(f"  Layer {layer_idx}: MLA alignment test")
    print(f"{'='*60}")

    # Load Megatron tensors (global batch from all DP ranks)
    meg = load_megatron_tensors(
        args.dump_dir, layer_idx, micro_batch_size=args.batch_size, seq_len=args.seq_len, num_dp_ranks=num_devices
    )
    print(f"  Megatron input:  {meg['input'].shape}")
    print(f"  Megatron output: {meg['output'].shape}")

    # Build MaxText MLA with checkpoint weights
    ckpt_layer_idx = layer_idx + args.ckpt_layer_offset
    print(f"  Checkpoint layer: {ckpt_layer_idx} " f"(dump {layer_idx} + offset {args.ckpt_layer_offset})")
    mla = build_mla_for_layer(cfg, mesh, jax_dtype, ckpt_flat, ckpt_layer_idx)

    # Forward
    maxtext_output, positions, seg_ids = run_forward(mla, meg["input"], mesh)
    print(f"  Forward done: {maxtext_output.shape}")

    # Backward
    maxtext_input_grad = run_backward(mla, meg["input"], meg["output_grad"], positions, seg_ids, mesh)
    print(f"  Backward done: {maxtext_input_grad.shape}")

    # Compare forward output
    prefix = f"layer_{layer_idx}"
    r_fwd = compare_tensors(f"{prefix}_fwd_output", maxtext_output, meg["output"], args.atol, args.rtol)
    print_result(r_fwd)
    all_results.append(r_fwd)

    # Compare backward input_grad
    r_bwd = compare_tensors(f"{prefix}_bwd_input_grad", maxtext_input_grad, meg["input_grad"], args.atol, args.rtol)
    print_result(r_bwd)
    all_results.append(r_bwd)

  # 5. Summary
  num_passed = sum(1 for r in all_results if r.passed)
  num_total = len(all_results)
  all_passed = num_passed == num_total

  print(f"\n{'='*60}")
  print(f"  Summary: {num_passed}/{num_total} passed")
  print(f"  Thresholds: rel_l2 < {MAX_REL_L2}, out_of_tol < {MAX_OUT_OF_TOL_PCT}%, " f"mean_diff < {MAX_MEAN_ABS_DIFF}")
  print(f"  ULP filter: ignore_ulp={IGNORE_ULP}, orig_dtype={ORIG_DTYPE}")
  for r in all_results:
    status = "PASS" if r.passed else "FAIL"
    print(f"    [{status}] {r.name}  rel_l2={r.rel_l2:.3e}  mean_diff={r.mean_abs_diff:.3e}")
  print(f"  Overall: {'PASS' if all_passed else 'FAIL'}")
  print(f"{'='*60}")

  return 0 if all_passed else 1


if __name__ == "__main__":
  sys.exit(main())
