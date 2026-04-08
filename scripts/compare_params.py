#!/usr/bin/env python3
# Copyright 2023-2025 Google LLC
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

"""Compare MaxText params against HF safetensors (name mapping applied).

Supports all models registered in PARAM_MAPPING. Sources: Argus dump or Orbax
checkpoint.  The HF config.json in --hf-dir is loaded automatically.

Usage:
    # MaxText Argus dump vs HF safetensors
    python scripts/compare_params.py \
        --model-name ling2 \
        --dump-dir /models/argus_dump/maxtext/step_0/rank_0/ \
        --hf-dir /path/to/hf-safetensors/

    # Orbax checkpoint vs HF safetensors
    python scripts/compare_params.py \
        --model-name ling2 \
        --orbax-items-dir /models/gpu-ckpt/.../0/items/ \
        --hf-dir /path/to/hf-safetensors/

    # List MaxText keys only
    python scripts/compare_params.py --model-name ling2 --dump-dir ... --list-keys

    # Filter by layer
    python scripts/compare_params.py --model-name ling2 --dump-dir ... --hf-dir ... --layers 0,1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare_orbax_safetensors import build_mapping, load_hf_config, make_maxtext_config, numel  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────


def _to_f32(arr: np.ndarray) -> np.ndarray:
  """Convert to float32, handling bfloat16 stored as void/V2."""
  if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:
    return (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
  if arr.dtype != np.float32:
    return arr.astype(np.float32)
  return arr


def _parse_dump_key(raw_key: str) -> tuple[str, int]:
  """'decoder/norm/scale#local_2' -> ('decoder/norm/scale', 2)."""
  if "#local_" in raw_key:
    base, idx_str = raw_key.rsplit("#local_", 1)
    return base, int(idx_str)
  return raw_key, 0


def _parse_param_key(raw_key: str) -> tuple[str, int]:
  """Parse a dump key into (dotted_name, local_index)."""
  base, local_idx = _parse_dump_key(raw_key)
  clean = base.replace("/", ".")
  for prefix in ("params.params.", "params."):
    if clean.startswith(prefix):
      clean = clean[len(prefix) :]
      break
  return clean, local_idx


def _load_metadata(rank_dir: Path) -> tuple[dict[str, tuple[int, ...]], dict[str, list]]:
  """Read full_shape and local_slices from metadata.yaml."""
  import yaml  # pylint: disable=import-outside-toplevel

  meta_path = rank_dir / "metadata.yaml"
  if not meta_path.exists():
    return {}, {}
  with open(meta_path, encoding="utf-8") as f:
    meta = yaml.safe_load(f)
  layout = meta.get("tensor_layout", {})
  shapes = {}
  for k, v in layout.items():
    category_and_name = k.split(":", 1)
    if len(category_and_name) == 2:
      clean = category_and_name[1].replace("/", ".")
      for prefix in ("params.params.", "params."):
        if clean.startswith(prefix):
          clean = clean[len(prefix) :]
          break
      shapes[k] = tuple(v["full_shape"])
      shapes[clean] = tuple(v["full_shape"])
  local_slices = meta.get("local_slices", {})
  return shapes, local_slices


def _reconstruct_from_slices(
    shards: dict[int, np.ndarray],
    slice_info: dict[int, list],
    full_shape: tuple[int, ...],
) -> np.ndarray:
  """Reconstruct full tensor by placing shards at their recorded slice positions."""
  out = np.zeros(full_shape, dtype=shards[min(shards)].dtype)
  for idx, arr in shards.items():
    sl = slice_info.get(idx)
    if sl is not None:
      out[tuple(slice(*s) for s in sl)] = arr
  return out


def _reconstruct_shards(
    shards: dict[int, np.ndarray],
    full_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
  """Fallback: concatenate shards when no slice info is available."""
  if len(shards) == 1:
    return list(shards.values())[0]
  ordered = [shards[i] for i in sorted(shards.keys())]
  if full_shape is not None and ordered[0].shape == full_shape:
    return ordered[0]
  if full_shape is not None:
    for axis, _ in enumerate(full_shape):
      if ordered[0].shape[axis] * len(ordered) == full_shape[axis]:
        if all(ordered[0].shape[a] == full_shape[a] for a in range(len(full_shape)) if a != axis):
          return np.concatenate(ordered, axis=axis)
  return np.concatenate(ordered, axis=0)


def compare_pair(
    a: np.ndarray,
    b: np.ndarray,
    transpose_b: bool = False,
) -> dict:
  """Compare two tensors and return diff statistics."""
  if transpose_b and b.ndim == 2:
    b = b.T
  if a.shape != b.shape and numel(a.shape) == numel(b.shape):
    b = b.reshape(a.shape)
  if a.shape != b.shape:
    return {"status": "shape_mismatch", "a_shape": list(a.shape), "b_shape": list(b.shape)}
  diff = np.abs(a - b)
  a_flat, b_flat = a.flatten(), b.flatten()
  cos = float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12))
  return {
      "status": "ok",
      "max_diff": float(diff.max()),
      "mean_diff": float(diff.mean()),
      "cosine": cos,
      "relative_err": float(diff.max() / (np.abs(a).max() + 1e-12)),
  }


def _print_table_header():
  """Print comparison table header."""
  print(f"\n{'='*120}")
  print(f"  {'Key':60s}  {'shape':20s}  {'max_diff':>10s}  {'mean_diff':>10s}  {'cosine':>10s}  {'rel_err':>10s}")
  print(f"  {'-'*60}  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")


def _print_result(key: str, stats: dict, shape_str: str = ""):
  """Print a single comparison result row."""
  if stats["status"] == "ok":
    flag = " !!!" if stats["max_diff"] > 1e-3 else ""
    print(
        f"  {key:60s}  {shape_str:20s}"
        f"  {stats['max_diff']:10.2e}"
        f"  {stats['mean_diff']:10.2e}"
        f"  {stats['cosine']:10.8f}"
        f"  {stats['relative_err']:10.2e}{flag}"
    )
  elif stats["status"] == "shape_mismatch":
    print(f"  {key:60s}  SHAPE MISMATCH {stats.get('a_shape')} vs {stats.get('b_shape')}")
  else:
    print(f"  {key:60s}  {stats['status'].upper():>54s}")


# ── Loaders ─────────────────────────────────────────────────────────


def load_dump_params(
    rank_dir: Path,
    full_shapes: dict | None = None,
    local_slices: dict | None = None,
) -> dict[str, np.ndarray]:
  """Load params from Argus dump, reconstructing shards using slice info."""
  cat_dir = rank_dir / "params"
  if not cat_dir.exists():
    print(f"  Warning: {cat_dir} not found")
    return {}

  shard_map: dict[str, dict[int, tuple[np.ndarray, list | None]]] = {}
  for f in sorted(cat_dir.glob("chunk_*.npz")):
    data = np.load(f)
    for raw_key in data.files:
      clean, local_idx = _parse_param_key(raw_key)
      arr = _to_f32(data[raw_key])
      meta_key = f"params:{raw_key}"
      sl = None
      if local_slices and meta_key in local_slices:
        entries = local_slices[meta_key]
        if entries and "local_index" in entries[0]:
          sl = entries[0]["local_index"]
      shard_map.setdefault(clean, {})[local_idx] = (arr, sl)

  tensors = {}
  for key, shards_with_slices in shard_map.items():
    fs = full_shapes.get(key) if full_shapes else None
    plain_shards = {idx: arr for idx, (arr, _) in shards_with_slices.items()}
    slice_info = {idx: sl for idx, (_, sl) in shards_with_slices.items() if sl is not None}
    if fs is not None and slice_info:
      tensors[key] = _reconstruct_from_slices(plain_shards, slice_info, fs)
    else:
      tensors[key] = _reconstruct_shards(plain_shards, fs)
  return tensors


def load_from_orbax(orbax_items_dir: Path) -> dict[str, np.ndarray]:
  """Load params from Orbax checkpoint."""
  import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
  import orbax.checkpoint as ocp  # pylint: disable=import-outside-toplevel

  ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
  tree = ckptr.restore(orbax_items_dir)

  # Handle both double-wrapped and single params
  params = tree["params"]
  if isinstance(params, dict) and "params" in params:
    params = params["params"]

  out = {}
  stack = [("", params)]
  while stack:
    prefix, cur = stack.pop()
    if isinstance(cur, dict):
      for k, v in cur.items():
        stack.append((f"{prefix}.{k}" if prefix else k, v))
    else:
      arr = np.asarray(cur)
      if arr.dtype == jnp.bfloat16 or (arr.dtype.kind == "V" and arr.dtype.itemsize == 2):
        arr = arr.view(np.uint16).astype(np.uint32) << 16
        arr = arr.view(np.float32)
      elif arr.dtype != np.float32:
        arr = arr.astype(np.float32)
      out[prefix] = arr
  return out


def load_hf_tensors(hf_dir: Path) -> dict[str, np.ndarray]:
  """Load all tensors from HF safetensors files."""
  from safetensors import safe_open  # pylint: disable=import-outside-toplevel

  out = {}
  for sf in sorted(hf_dir.glob("*.safetensors")):
    with safe_open(str(sf), framework="numpy") as f:
      for k in f.keys():
        arr = f.get_tensor(k)
        out[k] = arr.astype(np.float32) if arr.dtype != np.float32 else arr
  return out


# ── Compare ─────────────────────────────────────────────────────────


def compare_params(args):
  """Run the MaxText vs HF parameter comparison."""
  # Load MaxText side
  if args.orbax_items_dir:
    print("Loading Orbax checkpoint...")
    mt_params = load_from_orbax(Path(args.orbax_items_dir))
  else:
    print("Loading Argus dump params...")
    full_shapes, local_slices = _load_metadata(Path(args.dump_dir))
    mt_params = load_dump_params(Path(args.dump_dir), full_shapes, local_slices)
  print(f"  MaxText: {len(mt_params)} leaves")

  if args.list_keys:
    for k in sorted(mt_params.keys()):
      print(f"  {k:80s}  {str(mt_params[k].shape)}")
    return

  if not args.hf_dir:
    print("Error: --hf-dir required for comparison")
    sys.exit(1)

  print("Loading HF safetensors...")
  hf_dir = Path(args.hf_dir)
  hf_tensors = load_hf_tensors(hf_dir)
  print(f"  HF: {len(hf_tensors)} tensors")

  hf_config = load_hf_config(hf_dir)
  maxtext_config = make_maxtext_config(hf_config)
  mapping = build_mapping(args.model_name, hf_config, maxtext_config)

  layer_filter = {int(x.strip()) for x in args.layers.split(",")} if args.layers else set()

  ok_count, bad_count = 0, 0
  results = []
  _print_table_header()

  for mt_key in sorted(mt_params.keys()):
    if layer_filter:
      skip = True
      for li in layer_filter:
        if f"layers_{li}." in mt_key:
          skip = False
          break
      if mt_key in ("token_embedder.embedding", "decoder.decoder_norm.scale", "decoder.logits_dense.kernel"):
        skip = False
      if skip:
        continue

    hf_keys = mapping.get(mt_key)
    if hf_keys is None:
      _print_result(mt_key, {"status": "no_mapping"})
      bad_count += 1
      results.append({"key": mt_key, "status": "no_mapping"})
      continue

    mt_arr = mt_params[mt_key]
    needs_transpose = mt_key.endswith((".kernel", ".wi_0", ".wi_1", ".wo"))

    if len(hf_keys) == 1:
      hf_arr = hf_tensors.get(hf_keys[0])
      if hf_arr is None:
        _print_result(mt_key, {"status": "hf_missing"})
        bad_count += 1
        continue
      stats = compare_pair(mt_arr, hf_arr, transpose_b=needs_transpose)
      _print_result(mt_key, stats, str(list(mt_arr.shape)))
      stats["key"] = mt_key
      results.append(stats)
      ok_count += 1 if stats["status"] == "ok" else 0
      bad_count += 0 if stats["status"] == "ok" else 1
    else:
      # Expert stacking: 1 MT tensor -> N HF tensors
      expert_diffs = []
      for ei, hk in enumerate(hf_keys):
        hf_arr = hf_tensors.get(hk)
        if hf_arr is None:
          continue
        s = compare_pair(mt_arr[ei], hf_arr, transpose_b=needs_transpose)
        if s["status"] == "ok":
          expert_diffs.append(s)
      if not expert_diffs:
        _print_result(mt_key, {"status": "hf_experts_missing"})
        bad_count += 1
        continue
      agg = {
          "status": "ok",
          "max_diff": max(d["max_diff"] for d in expert_diffs),
          "mean_diff": (sum(d["mean_diff"] for d in expert_diffs) / len(expert_diffs)),
          "cosine": min(d["cosine"] for d in expert_diffs),
          "relative_err": max(d["relative_err"] for d in expert_diffs),
      }
      label = f"{mt_key} [{len(expert_diffs)} experts]"
      _print_result(label, agg, str(list(mt_arr.shape)))
      agg["key"] = mt_key
      results.append(agg)
      ok_count += 1

  print(f"\nSummary: {ok_count} matched, {bad_count} issues")
  if args.report_json:
    with open(args.report_json, "w", encoding="utf-8") as f:
      json.dump(results, f, indent=2, default=str)
    print(f"Report saved to {args.report_json}")


# ── Main ────────────────────────────────────────────────────────────


def main():
  """Parse arguments and run comparison."""
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--model-name", required=True, help="Model key in PARAM_MAPPING (e.g. ling2, gemma2-2b)")
  src = ap.add_mutually_exclusive_group(required=True)
  src.add_argument("--dump-dir", help="Argus dump rank directory (e.g. step_0/rank_0/)")
  src.add_argument("--orbax-items-dir", help="Orbax checkpoint items/ directory")
  ap.add_argument("--hf-dir", help="HF safetensors directory (must contain config.json)")
  ap.add_argument("--layers", default="", help="Comma-separated layer indices, empty=all")
  ap.add_argument("--list-keys", action="store_true", help="List keys and exit")
  ap.add_argument("--report-json", default="", help="Save report to JSON")
  args = ap.parse_args()

  compare_params(args)


if __name__ == "__main__":
  main()
