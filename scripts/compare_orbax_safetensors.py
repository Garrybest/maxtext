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

"""Compare Orbax checkpoint leaves against HF safetensors keys/shapes.

Supports all models registered in PARAM_MAPPING (gemma, qwen, llama, deepseek,
ling2, etc.). The HF config.json in --hf-dir is loaded automatically to derive
model-specific parameters.

Usage:
    python scripts/compare_orbax_safetensors.py \
        --model-name ling2 \
        --orbax-items-dir /path/to/orbax/0/items/ \
        --hf-dir /path/to/hf-safetensors/
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


# ── Config helpers ──────────────────────────────────────────────────


def load_hf_config(hf_dir: Path) -> dict:
  """Load config.json from HF directory."""
  config_path = hf_dir / "config.json"
  with open(config_path, encoding="utf-8") as f:
    return json.load(f)


def make_maxtext_config(hf_config: dict) -> SimpleNamespace:
  """Create a permissive maxtext config derived from HF config.

  The PARAM_MAPPING functions read HF config first, falling back to
  maxtext_config attributes.  Deriving defaults from the same HF config
  ensures the fallbacks are consistent without requiring a real MaxText
  pyconfig (which needs JAX).
  """
  return SimpleNamespace(
      scan_layers=False,
      # Ling2 / DeepSeek style
      first_num_dense_layers=int(hf_config.get("first_k_dense_replace", 0)),
      num_experts=int(hf_config.get("num_experts", hf_config.get("n_routed_experts", 0))),
      inhomogeneous_layer_cycle_interval=int(hf_config.get("layer_group_size", 1)),
      q_lora_rank=int(hf_config.get("q_lora_rank", 0)),
      mtp_num_layers=int(hf_config.get("num_nextn_predict_layers", 0)),
      # Generic
      num_hidden_layers=int(hf_config.get("num_hidden_layers", 0)),
  )


# ── Key format helpers ──────────────────────────────────────────────


def _pm_key_to_orbax_key(pm_key: str) -> str:
  """Convert param_mapping key to Orbax flat key.

  'params-decoder-dense_layers_0-mlp-wi_0-kernel'
  -> 'decoder.dense_layers_0.mlp.wi_0.kernel'
  """
  key = pm_key
  if key.startswith("params-"):
    key = key[len("params-") :]
  return key.replace("-", ".")


def build_mapping(
    model_name: str,
    hf_config: dict,
    maxtext_config: SimpleNamespace,
) -> dict[str, list[str]]:
  """Build Orbax-key -> HF-key(s) mapping via PARAM_MAPPING registry."""
  from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING  # pylint: disable=import-outside-toplevel

  if model_name not in PARAM_MAPPING:
    raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(PARAM_MAPPING.keys())}")
  pm = PARAM_MAPPING[model_name](hf_config, maxtext_config)
  mapping = {}
  for pm_key, hf_val in pm.items():
    orbax_key = _pm_key_to_orbax_key(pm_key)
    mapping[orbax_key] = hf_val if isinstance(hf_val, list) else [hf_val]
  return mapping


# ── Loaders ─────────────────────────────────────────────────────────


def load_orbax_shapes(orbax_items_dir: Path) -> dict[str, tuple[int, ...]]:
  """Load tensor shapes from an Orbax checkpoint (metadata only)."""
  import orbax.checkpoint as ocp  # pylint: disable=import-outside-toplevel

  ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
  meta = ckptr.metadata(orbax_items_dir)
  tree = meta.item_metadata.tree

  # Handle both double-wrapped (NNX: params.params) and single (Linen: params)
  if "params" in tree:
    node = tree["params"]
    if isinstance(node, dict) and "params" in node:
      node = node["params"]
  else:
    node = tree

  out = {}
  stack = [("", node)]
  while stack:
    prefix, cur = stack.pop()
    if isinstance(cur, dict):
      for k, v in cur.items():
        nxt = f"{prefix}.{k}" if prefix else k
        stack.append((nxt, v))
    else:
      shape = tuple(int(x) for x in getattr(cur, "shape", ()))
      out[prefix] = shape
  return out


def load_hf_shapes(hf_dir: Path) -> dict[str, tuple[int, ...]]:
  """Load tensor shapes from HF safetensors files."""
  from safetensors import safe_open  # pylint: disable=import-outside-toplevel

  out = {}
  for sf in sorted(hf_dir.glob("*.safetensors")):
    with safe_open(str(sf), framework="pt") as f:
      for k in f.keys():
        out[k] = tuple(int(x) for x in f.get_slice(k).get_shape())
  return out


def numel(shape: tuple[int, ...]) -> int:
  """Return the total number of elements for the given shape."""
  n = 1
  for d in shape:
    n *= int(d)
  return n


# ── Main ────────────────────────────────────────────────────────────


def main():
  """Parse arguments and run Orbax vs HF shape comparison."""
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--model-name", required=True, help="Model key in PARAM_MAPPING (e.g. ling2, gemma2-2b, llama3.1-8b)")
  ap.add_argument("--orbax-items-dir", required=True, help="Orbax checkpoint items/ directory")
  ap.add_argument("--hf-dir", required=True, help="HF safetensors directory (must contain config.json)")
  ap.add_argument("--report-json", default="", help="Save report to JSON file")
  ap.add_argument("--dump-orbax", default="", help="Dump Orbax keys/shapes to file")
  ap.add_argument("--dump-hf", default="", help="Dump HF keys/shapes to file")
  args = ap.parse_args()

  hf_dir = Path(args.hf_dir)
  hf_config = load_hf_config(hf_dir)
  maxtext_config = make_maxtext_config(hf_config)
  mapping = build_mapping(args.model_name, hf_config, maxtext_config)

  orbax_shapes = load_orbax_shapes(Path(args.orbax_items_dir))
  hf_shapes = load_hf_shapes(hf_dir)

  if args.dump_orbax:
    with open(args.dump_orbax, "w", encoding="utf-8") as f:
      for k in sorted(orbax_shapes):
        f.write(f"{k}\t{orbax_shapes[k]}\n")
  if args.dump_hf:
    with open(args.dump_hf, "w", encoding="utf-8") as f:
      for k in sorted(hf_shapes):
        f.write(f"{k}\t{hf_shapes[k]}\n")

  missing_mapping = []
  missing_hf = []
  shape_mismatch = []
  mapped_ok = 0
  used_hf = set()

  for mt_key, mt_shape in sorted(orbax_shapes.items()):
    hf_keys = mapping.get(mt_key)
    if hf_keys is None:
      missing_mapping.append([mt_key, mt_shape])
      continue

    if len(hf_keys) == 1:
      hk = hf_keys[0]
      hs = hf_shapes.get(hk)
      if hs is None:
        missing_hf.append([mt_key, hk, mt_shape, None])
        continue
      used_hf.add(hk)
      if numel(mt_shape) != numel(hs):
        shape_mismatch.append([mt_key, mt_shape, [hk], [hs], "numel_mismatch"])
      else:
        mapped_ok += 1
    else:
      # Expert stacking: 1 MT tensor -> N HF tensors
      if len(mt_shape) < 1 or mt_shape[0] != len(hf_keys):
        shape_mismatch.append([mt_key, mt_shape, hf_keys[:2], [hf_shapes.get(hf_keys[0])], "expert_count_mismatch"])
        continue
      mt_inner = numel(mt_shape[1:])
      bad = False
      hss = []
      for hk in hf_keys:
        hs = hf_shapes.get(hk)
        if hs is None:
          missing_hf.append([mt_key, hk, mt_shape, None])
          bad = True
          break
        used_hf.add(hk)
        hss.append(hs)
        if numel(hs) != mt_inner:
          bad = True
      if bad:
        shape_mismatch.append([mt_key, mt_shape, hf_keys[:2], hss[:2], "expert_inner_numel_mismatch"])
      else:
        mapped_ok += 1

  report = {
      "model_name": args.model_name,
      "orbax_leaf_count": len(orbax_shapes),
      "hf_tensor_count": len(hf_shapes),
      "mapped_ok_count": mapped_ok,
      "missing_mapping_count": len(missing_mapping),
      "missing_hf_count": len(missing_hf),
      "shape_mismatch_count": len(shape_mismatch),
      "unused_hf_count": len(set(hf_shapes) - used_hf),
      "missing_mapping_sample": missing_mapping[:30],
      "missing_hf_sample": missing_hf[:30],
      "shape_mismatch_sample": shape_mismatch[:30],
      "unused_hf_sample": sorted(set(hf_shapes) - used_hf)[:50],
  }

  print(json.dumps(report, ensure_ascii=False, indent=2))
  if args.report_json:
    with open(args.report_json, "w", encoding="utf-8") as f:
      json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
  main()
