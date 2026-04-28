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

"""Unified checkpoint comparison: HuggingFace ↔ MaxText Orbax.

Sources: `hf:<dir>` (safetensors + config.json + index.json) or
`orbax:<dir>` (MaxText Orbax items directory).

Modes: `structure` (keys + shapes + dtypes; fast), `numerical` (load every
tensor and check bit-exact / within atol; slow), or `both` (default).

`--model-name` is required whenever the two sides are in different formats; the
PARAM_MAPPING registry bridges MaxText ↔ HF parameter names.

Examples:
    # Validate MaxText→HF output (bit-exact HF↔HF)
    compare_checkpoints.py --left hf:/ref --right hf:/converted

    # Fast structural check after to_maxtext.py
    compare_checkpoints.py --left orbax:/ckpt/0/items --right hf:/hf \\
        --model-name ling2 --mode structure

    # Full numerical check of Orbax vs HF
    compare_checkpoints.py --left orbax:/ckpt/0/items --right hf:/hf --model-name ling2

    # Subset / inspection
    compare_checkpoints.py --left orbax:/ckpt/0/items --right hf:/hf --model-name ling2 --layers 0,4,19
    compare_checkpoints.py --left orbax:/ckpt/0/items --list-keys
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


# ── Source abstractions ─────────────────────────────────────────────


class Source:
  """Abstract tensor source: provides key/shape/dtype/load APIs."""

  def keys(self) -> set[str]:
    raise NotImplementedError

  def shape(self, key: str) -> tuple[int, ...]:
    raise NotImplementedError

  def dtype(self, key: str) -> str:
    raise NotImplementedError

  def load(self, key: str):
    """Load tensor as numpy fp32 array."""
    raise NotImplementedError

  @property
  def kind(self) -> str:
    raise NotImplementedError


def _to_f32_numpy(arr):
  """Convert numpy array (possibly bf16 stored as void, bfloat16, or uint16) to fp32."""
  import numpy as np  # pylint: disable=import-outside-toplevel

  # bf16 stored as an opaque 2-byte void (numpy lacks native bf16 support in some builds)
  if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:
    return (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
  # bf16 passed as uint16 bit pattern
  if arr.dtype == np.uint16:
    return (arr.astype(np.uint32) << 16).view(np.float32)
  if arr.dtype != np.float32:
    return arr.astype(np.float32)
  return arr


class HFSource(Source):
  """HuggingFace safetensors directory source."""

  kind = "hf"

  def __init__(self, path: Path):
    self.path = path
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
      with open(index_path, encoding="utf-8") as f:
        self._index = json.load(f)["weight_map"]
    else:
      # Single-shard fallback
      single = path / "model.safetensors"
      if not single.exists():
        raise FileNotFoundError(f"Neither {index_path} nor {single} exists")
      self._index = {}
      from safetensors import safe_open  # pylint: disable=import-outside-toplevel

      with safe_open(str(single), framework="flax") as f:
        for k in f.keys():
          self._index[k] = "model.safetensors"
    self._shard_cache = {}
    self._shape_cache = {}
    self._dtype_cache = {}

  def _shard(self, name):
    if name not in self._shard_cache:
      from safetensors import safe_open  # pylint: disable=import-outside-toplevel

      self._shard_cache[name] = safe_open(str(self.path / name), framework="flax")
    return self._shard_cache[name]

  def keys(self):
    return set(self._index.keys())

  def shape(self, key):
    if key not in self._shape_cache:
      sl = self._shard(self._index[key]).get_slice(key)
      self._shape_cache[key] = tuple(int(x) for x in sl.get_shape())
    return self._shape_cache[key]

  def dtype(self, key):
    if key not in self._dtype_cache:
      self._dtype_cache[key] = str(self._shard(self._index[key]).get_slice(key).get_dtype())
    return self._dtype_cache[key]

  def load(self, key):
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel

    t = self._shard(self._index[key]).get_tensor(key)
    return np.asarray(t.astype(jnp.float32))

  def config(self) -> dict:
    """Load HF config.json."""
    with open(self.path / "config.json", encoding="utf-8") as f:
      return json.load(f)


class OrbaxSource(Source):
  """MaxText Orbax checkpoint items directory source."""

  kind = "orbax"

  def __init__(self, path: Path):
    self.path = path
    self._loaded = False
    self._tensors = None
    self._shapes_only = None

  def _load_shapes_only(self):
    """Metadata-only shape extraction (fast)."""
    if self._shapes_only is not None:
      return
    import orbax.checkpoint as ocp  # pylint: disable=import-outside-toplevel

    ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    meta = ckptr.metadata(self.path)
    tree = meta.item_metadata.tree
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
          stack.append((f"{prefix}.{k}" if prefix else k, v))
      else:
        out[prefix] = tuple(int(x) for x in getattr(cur, "shape", ()))
    self._shapes_only = out

  def _load_all(self):
    """Eager-load all tensors (slow, used for numerical phase)."""
    if self._loaded:
      return
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel
    import orbax.checkpoint as ocp  # pylint: disable=import-outside-toplevel

    ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    tree = ckptr.restore(self.path)
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
        if arr.dtype == jnp.bfloat16:
          arr = arr.view(np.uint16)
        out[prefix] = _to_f32_numpy(arr)
    self._tensors = out
    self._loaded = True

  def keys(self):
    if self._tensors is not None:
      return set(self._tensors.keys())
    self._load_shapes_only()
    return set(self._shapes_only.keys())

  def shape(self, key):
    if self._tensors is not None:
      return tuple(self._tensors[key].shape)
    self._load_shapes_only()
    return self._shapes_only[key]

  def dtype(self, key):
    # Orbax metadata has this but we normalize to fp32 on load; for structure
    # phase, just report fp32 since that's what numerical phase sees.
    return "F32"

  def load(self, key):
    self._load_all()
    return self._tensors[key]


# ── Cross-format PARAM_MAPPING bridge ───────────────────────────────


def _pm_key_to_orbax_key(pm_key: str) -> str:
  """'params-decoder-dense_layers_0-mlp-wi_0-kernel' -> 'decoder.dense_layers_0.mlp.wi_0.kernel'."""
  key = pm_key[len("params-") :] if pm_key.startswith("params-") else pm_key
  return key.replace("-", ".")


def _make_maxtext_config(hf_config: dict) -> SimpleNamespace:
  """Permissive MaxText config derived from HF config for PARAM_MAPPING fallbacks."""
  return SimpleNamespace(
      scan_layers=False,
      first_num_dense_layers=int(hf_config.get("first_k_dense_replace", 0)),
      num_experts=int(hf_config.get("num_experts", hf_config.get("n_routed_experts", 0))),
      inhomogeneous_layer_cycle_interval=int(hf_config.get("layer_group_size", 1)),
      q_lora_rank=int(hf_config.get("q_lora_rank", 0)),
      mtp_num_layers=int(hf_config.get("num_nextn_predict_layers", 0)),
      num_hidden_layers=int(hf_config.get("num_hidden_layers", 0)),
  )


def _build_mt_to_hf_mapping(model_name: str, hf_config: dict) -> dict:
  """Build {maxtext_orbax_key: [hf_keys...]} for cross-format compare."""
  from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING  # pylint: disable=import-outside-toplevel

  if model_name not in PARAM_MAPPING:
    raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(PARAM_MAPPING)}")
  pm = PARAM_MAPPING[model_name](hf_config, _make_maxtext_config(hf_config))
  out = {}
  for pm_key, hf_val in pm.items():
    out[_pm_key_to_orbax_key(pm_key)] = hf_val if isinstance(hf_val, list) else [hf_val]
  return out


def _build_mt_hook_fn_map(model_name: str, hf_config: dict) -> dict:
  """Build {maxtext_orbax_key: hook_fn_or_list} matching the conversion-time hook chain.

  The numerical compare phase needs to apply the same HF→MT hooks that
  to_maxtext.py used during conversion (e.g. reshape_kernel, reshape_depthwise_conv).
  Without this, layout transformations like the depthwise-conv squeeze+transpose
  cannot be inverted by the simple b.reshape(a.shape) fallback in _compare_pair.
  """
  from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS  # pylint: disable=import-outside-toplevel

  if model_name not in HOOK_FNS:
    return {}
  hooks = HOOK_FNS[model_name](hf_config, _make_maxtext_config(hf_config), saving_to_hf=False)
  return {_pm_key_to_orbax_key(pm_key): hook for pm_key, hook in hooks.items()}


def _numel(shape) -> int:
  n = 1
  for d in shape:
    n *= int(d)
  return n


# ── Compare phases ──────────────────────────────────────────────────


def _compare_pair(a, b, transpose_b=False):
  """Numerical compare. a is MT-side (or ref), b is HF-side (or target)."""
  import numpy as np  # pylint: disable=import-outside-toplevel

  if transpose_b and b.ndim == 2:
    b = b.T
  if a.shape != b.shape and _numel(a.shape) == _numel(b.shape):
    b = b.reshape(a.shape)
  if a.shape != b.shape:
    return {"status": "shape_mismatch", "a_shape": list(a.shape), "b_shape": list(b.shape)}
  diff = np.abs(a - b)
  max_diff = float(diff.max())
  mean_diff = float(diff.mean())
  af, bf = a.flatten(), b.flatten()
  denom = float(np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-12
  cos = float(np.dot(af, bf) / denom)
  rel_err = float(max_diff / (float(np.abs(a).max()) + 1e-12))
  return {"status": "ok", "max_diff": max_diff, "mean_diff": mean_diff, "cosine": cos, "relative_err": rel_err}


def _phase_structure_hf_hf(left: HFSource, right: HFSource):
  """HF↔HF structure phase: direct key/shape/dtype check."""
  lk, rk = left.keys(), right.keys()
  common = sorted(lk & rk)
  missing = sorted(lk - rk)
  extra = sorted(rk - lk)
  shape_mismatches, dtype_mismatches = [], []
  for k in common:
    if left.shape(k) != right.shape(k):
      shape_mismatches.append({"key": k, "left": list(left.shape(k)), "right": list(right.shape(k))})
    if left.dtype(k) != right.dtype(k):
      dtype_mismatches.append({"key": k, "left": left.dtype(k), "right": right.dtype(k)})
  return {
      "mode": "hf_hf",
      "left_keys": len(lk),
      "right_keys": len(rk),
      "common": len(common),
      "missing_in_right": missing,
      "extra_in_right": extra,
      "shape_mismatches": shape_mismatches,
      "dtype_mismatches": dtype_mismatches,
  }


def _phase_structure_mt_hf(mt: Source, hf: HFSource, mapping: dict):
  """MT↔HF structure phase: apply PARAM_MAPPING, check numel compatibility."""
  mt_keys = sorted(mt.keys())
  hf_shapes = {k: hf.shape(k) for k in hf.keys()}
  missing_mapping, missing_hf, shape_mismatch = [], [], []
  mapped_ok, used_hf = 0, set()
  for mt_key in mt_keys:
    hf_keys = mapping.get(mt_key)
    mt_shape = mt.shape(mt_key)
    if hf_keys is None:
      missing_mapping.append([mt_key, list(mt_shape)])
      continue
    if len(hf_keys) == 1:
      hk = hf_keys[0]
      hs = hf_shapes.get(hk)
      if hs is None:
        missing_hf.append([mt_key, hk])
        continue
      used_hf.add(hk)
      if _numel(mt_shape) != _numel(hs):
        shape_mismatch.append({"mt_key": mt_key, "mt_shape": list(mt_shape), "hf_key": hk, "hf_shape": list(hs)})
      else:
        mapped_ok += 1
    else:
      # Expert stacking
      if len(mt_shape) < 1 or mt_shape[0] != len(hf_keys):
        shape_mismatch.append(
            {"mt_key": mt_key, "mt_shape": list(mt_shape), "hf_keys": hf_keys[:2], "reason": "expert_count"}
        )
        continue
      mt_inner = _numel(mt_shape[1:])
      bad = False
      for hk in hf_keys:
        hs = hf_shapes.get(hk)
        if hs is None:
          missing_hf.append([mt_key, hk])
          bad = True
          break
        used_hf.add(hk)
        if _numel(hs) != mt_inner:
          shape_mismatch.append(
              {"mt_key": mt_key, "mt_shape": list(mt_shape), "hf_key": hk, "hf_shape": list(hs), "reason": "expert_inner"}
          )
          bad = True
          break
      if not bad:
        mapped_ok += 1
  return {
      "mode": "mt_hf",
      "mt_leaves": len(mt_keys),
      "hf_tensors": len(hf_shapes),
      "mapped_ok": mapped_ok,
      "missing_mapping_count": len(missing_mapping),
      "missing_hf_count": len(missing_hf),
      "shape_mismatch_count": len(shape_mismatch),
      "unused_hf_count": len(set(hf_shapes) - used_hf),
      "missing_mapping_sample": missing_mapping[:30],
      "missing_hf_sample": missing_hf[:30],
      "shape_mismatch_sample": shape_mismatch[:30],
      "unused_hf_sample": sorted(set(hf_shapes) - used_hf)[:50],
  }


def _phase_numerical_hf_hf(left: HFSource, right: HFSource, sample, atol, verbose):
  """HF↔HF numerical: bit-exact per-tensor compare.

  When running --mode numerical without the structure phase, this also surfaces
  key-set mismatches (missing_in_right / extra_in_right) so a missing or extra
  tensor cannot yield a false `verdict_ok=True`.
  """
  import numpy as np  # pylint: disable=import-outside-toplevel

  lk, rk = left.keys(), right.keys()
  common = sorted(lk & rk)
  missing_in_right = sorted(lk - rk)
  extra_in_right = sorted(rk - lk)
  keys = (
      common
      if (sample is None or sample <= 0 or sample >= len(common))
      else common[:: max(1, len(common) // sample)][:sample]
  )
  bit_exact = 0
  diffs = []
  t0 = time.time()
  for i, k in enumerate(keys):
    a, b = left.load(k), right.load(k)
    if np.array_equal(a, b):
      bit_exact += 1
      continue
    stats = _compare_pair(a, b)
    stats["key"] = k
    diffs.append(stats)
    if verbose:
      print(f"  [DIFF] {k}: max_diff={stats.get('max_diff', 'N/A')}")
    if (i + 1) % 1000 == 0:
      eta = (len(keys) - i - 1) / (i + 1) * (time.time() - t0)
      print(f"    [{i+1}/{len(keys)}] bit-exact={bit_exact} diff={len(diffs)} eta={eta:.0f}s")
  values_ok = not diffs or all(d.get("status") == "ok" and d.get("max_diff", float("inf")) <= atol for d in diffs)
  verdict_ok = values_ok and not missing_in_right and not extra_in_right
  return {
      "mode": "hf_hf",
      "compared": len(keys),
      "bit_exact": bit_exact,
      "missing_in_right": missing_in_right,
      "extra_in_right": extra_in_right,
      "diffs": diffs[:50],
      "diff_count": len(diffs),
      "verdict_ok": verdict_ok,
      "worst_max_diff": max((d.get("max_diff", 0.0) for d in diffs), default=0.0),
  }


def _hf_layer_idx(hf_path: str):
  """Extract global layer index from an HF param path like 'model.layers.N.*'."""
  import re  # pylint: disable=import-outside-toplevel

  m = re.match(r"model\.layers\.(\d+)\.", hf_path)
  return int(m.group(1)) if m else None


_MT_NON_LAYER_KEYS = frozenset(
    {
        "params-token_embedder-embedding",
        "params-decoder-decoder_norm-scale",
        "params-decoder-logits_dense-kernel",
        # Also accept the flattened-key form emitted by OrbaxSource
        "token_embedder.embedding",
        "decoder.decoder_norm.scale",
        "decoder.logits_dense.kernel",
    }
)


def _phase_numerical_mt_hf(
    mt: Source,
    hf: HFSource,
    mapping: dict,
    hooks_map: dict,
    layer_filter: set,
    atol: float,
):
  """MT↔HF numerical: apply PARAM_MAPPING + HOOK_FNS, compare element-wise.

  When a hook is registered for a given mt_key, it is applied to the HF tensor
  (HF→MT direction) before the diff. This is required for params like KDA
  depthwise conv kernels where MT layout ([K, C]) is not a flat-buffer view of
  HF layout ([C, 1, K]) — only the conversion-time hook can invert the
  squeeze+transpose. When no hook is registered, fall back to the legacy
  transpose_b/reshape heuristic.
  """
  from maxtext.checkpoint_conversion.utils.utils import apply_hook_fns  # pylint: disable=import-outside-toplevel

  results = []
  ok = 0
  bad = 0
  # Cache HF key membership once so per-expert checks stay O(1) for large MoE
  # configs (256 experts × N layers would otherwise rebuild the set many times).
  hf_key_set = hf.keys()
  consumed_hf = set()
  skipped_by_layer_filter = False
  for mt_key in sorted(mt.keys()):
    if layer_filter:
      # Resolve layer filter against the HF target's "model.layers.N" index,
      # which reflects the user's logical layer numbering even for heterogeneous
      # models (e.g. Ling2 where global layer 4 is MT 'moe_layers_3').
      targets = mapping.get(mt_key) or []
      if not isinstance(targets, list):
        targets = [targets]
      hf_layers = {idx for idx in (_hf_layer_idx(h) for h in targets) if idx is not None}
      if not (hf_layers & layer_filter) and mt_key not in _MT_NON_LAYER_KEYS:
        skipped_by_layer_filter = True
        continue
    hf_keys = mapping.get(mt_key)
    if hf_keys is None:
      results.append({"key": mt_key, "status": "no_mapping"})
      bad += 1
      continue
    mt_arr = mt.load(mt_key)
    needs_transpose = mt_key.endswith((".kernel", ".wi_0", ".wi_1", ".wo"))
    if len(hf_keys) == 1:
      if hf_keys[0] not in hf_key_set:
        results.append({"key": mt_key, "status": "hf_missing"})
        bad += 1
        continue
      consumed_hf.add(hf_keys[0])
      hf_arr = hf.load(hf_keys[0])
      hook = hooks_map.get(mt_key)
      if hook is not None:
        hf_arr = apply_hook_fns(hf_arr, mt_arr.shape, hook)
      stats = _compare_pair(mt_arr, hf_arr, transpose_b=(needs_transpose and hook is None))
      stats["key"] = mt_key
      results.append(stats)
      if stats["status"] == "ok" and stats.get("max_diff", 0.0) <= atol:
        ok += 1
      else:
        bad += 1
    else:
      # MoE expert stacking: mt_arr shape is (num_experts, ...); each hf_keys[ei]
      # is the HF path for expert ei.
      expert_diffs = []
      expert_errors = []
      mt_expert_count = mt_arr.shape[0] if mt_arr.ndim > 0 else 0
      # Flag any MaxText experts beyond what the HF mapping enumerates, so a
      # checkpoint with extra experts doesn't silently pass.
      if mt_expert_count > len(hf_keys):
        expert_errors.append(
            {
                "expert": len(hf_keys),
                "status": "mt_has_extra_experts",
                "mt_count": mt_expert_count,
                "hf_count": len(hf_keys),
            }
        )
      for ei, hk in enumerate(hf_keys):
        if ei >= mt_expert_count:
          expert_errors.append({"expert": ei, "hf_key": hk, "status": "mt_expert_out_of_range"})
          continue
        if hk not in hf_key_set:
          expert_errors.append({"expert": ei, "hf_key": hk, "status": "hf_missing"})
          continue
        consumed_hf.add(hk)
        hf_arr = hf.load(hk)
        hook = hooks_map.get(mt_key)
        if hook is not None:
          hf_arr = apply_hook_fns(hf_arr, mt_arr[ei].shape, hook)
        s = _compare_pair(mt_arr[ei], hf_arr, transpose_b=(needs_transpose and hook is None))
        if s["status"] == "ok":
          expert_diffs.append(s)
        else:
          expert_errors.append({"expert": ei, "hf_key": hk, **s})
      if expert_errors or not expert_diffs:
        results.append(
            {
                "key": mt_key,
                "status": "expert_errors",
                "error_count": len(expert_errors),
                "errors_sample": expert_errors[:5],
                "ok_count": len(expert_diffs),
            }
        )
        bad += 1
        continue
      agg = {
          "key": mt_key,
          "status": "ok",
          "max_diff": max(d["max_diff"] for d in expert_diffs),
          "mean_diff": sum(d["mean_diff"] for d in expert_diffs) / len(expert_diffs),
          "cosine": min(d["cosine"] for d in expert_diffs),
          "relative_err": max(d["relative_err"] for d in expert_diffs),
          "experts": len(expert_diffs),
      }
      results.append(agg)
      if agg["max_diff"] <= atol:
        ok += 1
      else:
        bad += 1
  # Catch HF tensors the mapping never reached. Skip this check under --layers
  # (user intentionally compared a subset).
  unused_hf = sorted(hf_key_set - consumed_hf) if not skipped_by_layer_filter else []
  return {
      "mode": "mt_hf",
      "matched": ok,
      "issues": bad,
      "results": results,
      "unused_hf_count": len(unused_hf),
      "unused_hf_sample": unused_hf[:30],
      "verdict_ok": bad == 0 and not unused_hf,
  }


# ── Printing helpers ────────────────────────────────────────────────


def _print_structure_report(r):
  """Print Phase 1 summary. Returns True if structure matches."""
  mode = r.get("mode")
  if mode == "hf_hf":
    print(f"  keys: left={r['left_keys']} right={r['right_keys']} common={r['common']}")
    print(f"  missing in right: {len(r['missing_in_right'])}  extra in right: {len(r['extra_in_right'])}")
    print(f"  shape mismatches: {len(r['shape_mismatches'])}  dtype mismatches: {len(r['dtype_mismatches'])}")
    for m in r["missing_in_right"][:5]:
      print(f"    - missing: {m}")
    for m in r["shape_mismatches"][:5]:
      print(f"    - shape:   {m['key']}: L={m['left']} vs R={m['right']}")
    return not (r["missing_in_right"] or r["extra_in_right"] or r["shape_mismatches"] or r["dtype_mismatches"])
  # mt_hf
  print(f"  leaves: mt={r['mt_leaves']} hf={r['hf_tensors']} mapped_ok={r['mapped_ok']}")
  print(f"  missing_mapping: {r['missing_mapping_count']}  missing_hf: {r['missing_hf_count']}")
  print(f"  shape_mismatch: {r['shape_mismatch_count']}  unused_hf: {r['unused_hf_count']}")
  for item in r["missing_mapping_sample"][:5]:
    print(f"    - no mapping: {item[0]}  {item[1]}")
  for item in r["shape_mismatch_sample"][:5]:
    print(f"    - shape: {item}")
  return (
      r["missing_mapping_count"] == 0
      and r["shape_mismatch_count"] == 0
      and r["missing_hf_count"] == 0
      and r["unused_hf_count"] == 0
  )


def _print_numerical_report(r):
  """Print Phase 2 summary. Returns True on verdict_ok."""
  mode = r.get("mode")
  if mode == "hf_hf":
    missing = r.get("missing_in_right", [])
    extra = r.get("extra_in_right", [])
    print(f"  compared: {r['compared']}  bit-exact: {r['bit_exact']}  diff: {r['diff_count']}")
    print(f"  worst max_diff: {r.get('worst_max_diff', 0.0):.3e}")
    if missing or extra:
      print(f"  missing in right: {len(missing)}  extra in right: {len(extra)}")
      for m in missing[:5]:
        print(f"    - missing: {m}")
      for e in extra[:5]:
        print(f"    - extra:   {e}")
    for d in r["diffs"][:5]:
      print(f"    - {d.get('key')}: max={d.get('max_diff')}")
    return r["verdict_ok"]
  # mt_hf
  print(f"  matched: {r['matched']}  issues: {r['issues']}")
  if r.get("unused_hf_count", 0):
    print(f"  unused_hf: {r['unused_hf_count']}")
    for k in r.get("unused_hf_sample", [])[:5]:
      print(f"    - unused: {k}")
  # Show top 10 worst or issue keys
  bad = [x for x in r["results"] if x.get("status") != "ok" or x.get("max_diff", 0.0) > 0]
  if bad:
    for b in bad[:10]:
      print(f"    - {b}")
  return r["verdict_ok"]


# ── Source parsing / CLI ────────────────────────────────────────────


def _parse_source(spec: str) -> Source:
  """Parse TYPE:PATH spec into a Source instance."""
  if ":" not in spec:
    raise argparse.ArgumentTypeError(f"expected TYPE:PATH, got '{spec}'")
  kind, path = spec.split(":", 1)
  path = Path(path)
  if kind == "hf":
    return HFSource(path)
  if kind == "orbax":
    return OrbaxSource(path)
  raise argparse.ArgumentTypeError(f"unknown source type '{kind}' (want hf|orbax)")


def main():
  """CLI entry point."""
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--left", required=True, type=_parse_source, help="Left source: hf:<dir> | orbax:<dir>")
  ap.add_argument("--right", type=_parse_source, help="Right source (omit when --list-keys is set)")
  ap.add_argument("--model-name", default=None, help="Required for cross-format compare (PARAM_MAPPING key)")
  ap.add_argument("--mode", choices=["structure", "numerical", "both"], default="both")
  ap.add_argument("--sample", type=int, default=None, help="Numerical: only compare first N common tensors (HF↔HF)")
  ap.add_argument("--atol", type=float, default=0.0, help="Max allowed |diff| (default: 0.0 = bit-exact)")
  ap.add_argument("--layers", default="", help="MT↔HF: comma-separated layer indices to compare")
  ap.add_argument("--list-keys", action="store_true", help="List left source keys and exit")
  ap.add_argument("--report-json", default=None, help="Write JSON report to this path")
  ap.add_argument("--verbose", action="store_true", help="Print per-tensor results")
  args = ap.parse_args()

  if args.list_keys:
    for k in sorted(args.left.keys()):
      print(f"{k}\t{args.left.shape(k)}")
    return

  if args.right is None:
    ap.error("--right is required unless --list-keys is set")

  left, right = args.left, args.right
  cross = left.kind != right.kind
  if cross and not args.model_name:
    ap.error(f"--model-name is required when comparing {left.kind} against {right.kind}")

  mapping = None
  hf_side = None
  mt_side = None
  if cross:
    # One side must be HF (for config.json + mapping target).
    if left.kind == "hf":
      hf_side, mt_side = left, right
    elif right.kind == "hf":
      hf_side, mt_side = right, left
    else:
      ap.error(f"cross-format compare between {left.kind} and {right.kind} not supported (need one HF side)")
    hf_config = hf_side.config()
    mapping = _build_mt_to_hf_mapping(args.model_name, hf_config)
    hooks_map = _build_mt_hook_fn_map(args.model_name, hf_config)

  report = {"args": {k: str(v) for k, v in vars(args).items() if k not in ("left", "right")}}
  structure_ok = True

  if args.mode in ("structure", "both"):
    print("\n[Phase 1: Structure]")
    if cross:
      phase_r = _phase_structure_mt_hf(mt_side, hf_side, mapping)
    else:
      phase_r = _phase_structure_hf_hf(left, right)
    structure_ok = _print_structure_report(phase_r)
    report["structure"] = phase_r

  if args.mode in ("numerical", "both"):
    if args.mode == "both" and not structure_ok:
      print("\n[Phase 2: Numerical] SKIPPED (structure phase found issues)")
      report["numerical"] = {"skipped": True, "reason": "structure_mismatch"}
    else:
      print("\n[Phase 2: Numerical]")
      t0 = time.time()
      if cross:
        layer_filter = {int(x.strip()) for x in args.layers.split(",") if x.strip()}
        phase_r = _phase_numerical_mt_hf(mt_side, hf_side, mapping, hooks_map, layer_filter, args.atol)
      else:
        phase_r = _phase_numerical_hf_hf(left, right, args.sample, args.atol, args.verbose)
      phase_r["elapsed_sec"] = time.time() - t0
      structure_ok = _print_numerical_report(phase_r) and structure_ok
      report["numerical"] = phase_r

  verdict_ok = structure_ok and report.get("numerical", {}).get("verdict_ok", True)
  print(f"\nVERDICT: {'OK ✅' if verdict_ok else 'MISMATCH ❌'}")

  if args.report_json:
    with open(args.report_json, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2, default=str)
    print(f"Report: {args.report_json}")

  sys.exit(0 if verdict_ok else 1)


if __name__ == "__main__":
  main()
