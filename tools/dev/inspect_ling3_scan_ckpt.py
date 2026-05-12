#!/usr/bin/env python3
"""Inspect a Ling3-tiny scan-mode Orbax checkpoint to verify the layer layout
and routed-bias state of Phase 1b unscan-prefix MoE layers.

Usage (on the Pod):
    python tools/dev/inspect_ling3_scan_ckpt.py /models/pretrain/ling3/maxtext_ckpt/ling3-tiny-scan-2/

The path can be:
  - the checkpoint root (we'll try `<root>/items` if no metadata is at root)
  - a specific step directory containing `items/`
  - the `items/` directory directly

What this verifies (per docs/ling3-scan-loss-analysis.md §4 K1/K2):
  1. The param tree has separate keys for the Phase 1b unscan prefix:
       decoder/moe_layers_0, _1, _2  (3 layers, each with their own gate.bias [num_experts])
     plus a stacked Phase 2 scan region:
       decoder/moe_layers              (gate.bias shape [scan_length, num_experts])
  2. The norms / nonzero-fraction of the gate.bias for these 4 paths.

Verdict logic:
  - If unscan-prefix biases are exactly 0 AND scan-region biases (per row) are
    nonzero, the §2 root-cause hypothesis is confirmed (this ckpt has been
    trained for >0 steps and the bug ate the prefix updates).
  - If everything is 0: ckpt is fresh-from-conversion, can't confirm bug from
    bias state alone — but structure check still useful.
  - If unscan-prefix biases are also nonzero: hypothesis is wrong, look elsewhere.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
  import orbax.checkpoint as ocp
except ImportError:
  print("orbax-checkpoint not installed; run inside the maxtext venv", file=sys.stderr)
  sys.exit(1)


# Bias paths we care about. These are tuples of nested dict keys;
# the first element may be "params" (Linen) or absent (NNX); script handles both.
PHASE1B_PATHS = [
    ("decoder", "moe_layers_0", "mlp", "MoeBlock_0", "gate", "bias"),
    ("decoder", "moe_layers_1", "mlp", "MoeBlock_0", "gate", "bias"),
    ("decoder", "moe_layers_2", "mlp", "MoeBlock_0", "gate", "bias"),
]
SCAN_PATH = ("decoder", "moe_layers", "mlp", "MoeBlock_0", "gate", "bias")


def resolve_ckpt_path(p: Path) -> Path:
  """Return the orbax-loadable directory.

  Tries (in order): <p>, <p>/items, <p>/0/items.
  """
  candidates = [p, p / "items", p / "0" / "items"]
  for c in candidates:
    if (c / "_CHECKPOINT_METADATA").exists() or (c / "manifest.ocdbt").exists() or (c / "metadata").exists():
      return c
    # Older format: directory has subdirs of array names
    if c.is_dir() and any(child.is_dir() for child in c.iterdir()):
      return c
  return p


def walk_metadata(node, path=()):
  """Yield (path_tuple, shape, dtype) for each array leaf in metadata tree."""
  if hasattr(node, "shape"):
    yield path, tuple(node.shape), str(node.dtype)
    return
  if isinstance(node, dict):
    for k, v in node.items():
      yield from walk_metadata(v, path + (str(k),))
    return
  if isinstance(node, (list, tuple)):
    for i, v in enumerate(node):
      yield from walk_metadata(v, path + (str(i),))


def get_at_path(tree, path):
  """Walk `tree` (nested dict / orbax PyTreeMetadata) by string keys."""
  cur = tree
  for k in path:
    if isinstance(cur, dict):
      if k not in cur:
        return None
      cur = cur[k]
    else:
      attr = getattr(cur, k, None)
      if attr is None:
        return None
      cur = attr
  return cur


def find_actual_path(metadata_tree, candidate_paths):
  """Try each candidate path with optional 'params' prefix; return the first hit."""
  for p in candidate_paths:
    for prefix in ((), ("params",)):
      full = prefix + p
      node = get_at_path(metadata_tree, full)
      if node is not None and hasattr(node, "shape"):
        return full, node
  return None, None


def array_summary(name: str, arr: np.ndarray, scan_axis: int | None = None) -> str:
  """Return a one-line summary of an array. If scan_axis is set, also report per-row stats."""
  arr = np.asarray(arr)
  zero_frac = float((arr == 0).sum()) / arr.size
  out = (
      f"{name}: shape={arr.shape} dtype={arr.dtype} "
      f"L2={np.linalg.norm(arr.astype(np.float32)):.6e} "
      f"min={arr.min():.4e} max={arr.max():.4e} "
      f"mean_abs={np.mean(np.abs(arr)):.4e} zero_frac={zero_frac:.3f}"
  )
  if scan_axis is not None and arr.ndim > 1:
    rows = []
    for i in range(arr.shape[scan_axis]):
      sl = [slice(None)] * arr.ndim
      sl[scan_axis] = i
      row = arr[tuple(sl)]
      rows.append(
          f"    row[{i}]: L2={np.linalg.norm(row.astype(np.float32)):.6e} "
          f"mean_abs={np.mean(np.abs(row)):.4e} zero_frac={float((row==0).sum())/row.size:.3f}"
      )
    out += "\n" + "\n".join(rows)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("ckpt_path", type=Path)
  ap.add_argument("--show-tree", action="store_true",
                  help="Print every array path under decoder/ (filtered to MoE/gate keys)")
  args = ap.parse_args()

  ckpt_path = resolve_ckpt_path(args.ckpt_path.expanduser().resolve())
  print(f"=== Inspecting: {ckpt_path}\n")

  ckptr = ocp.PyTreeCheckpointer()

  # --- Step 1: structural inspection via metadata (cheap) ---
  meta = ckptr.metadata(ckpt_path)
  # orbax versions vary: try the most common access pattern
  meta_tree = getattr(meta, "tree", meta)

  if args.show_tree:
    print("--- All array paths under decoder/* matching moe_layers / gate / bias ---")
    for path, shape, dtype in walk_metadata(meta_tree):
      key = "/".join(path)
      if "moe_layers" in key and ("gate" in key or "bias" in key):
        print(f"  {key}  shape={shape} dtype={dtype}")
    print()

  # --- Step 2: locate the 4 bias paths ---
  print("--- Locating bias paths ---")
  located = {}
  for p in PHASE1B_PATHS:
    full, node = find_actual_path(meta_tree, [p])
    label = "/".join(p)
    if full is None:
      print(f"  [MISSING] {label}")
    else:
      print(f"  [OK]      {'/'.join(full)}  shape={tuple(node.shape)} dtype={node.dtype}")
      located[label] = full
  full_scan, node_scan = find_actual_path(meta_tree, [SCAN_PATH])
  scan_label = "/".join(SCAN_PATH)
  if full_scan is None:
    print(f"  [MISSING] {scan_label}")
  else:
    print(f"  [OK]      {'/'.join(full_scan)}  shape={tuple(node_scan.shape)} dtype={node_scan.dtype}")
    located[scan_label] = full_scan
  print()

  if not located:
    print("Found none of the expected bias paths. Re-run with --show-tree to print the actual structure.",
          file=sys.stderr)
    sys.exit(2)

  # --- Step 3: load just those leaves, summarize ---
  print("--- Loading bias arrays ---")
  # Build a sparse restore_args matching `located` only.
  def build_restore(node, path, target_paths):
    if path in target_paths:
      return ocp.RestoreArgs()
    if hasattr(node, "shape"):
      return None  # don't load
    if isinstance(node, dict):
      return {k: build_restore(v, path + (k,), target_paths) for k, v in node.items()}
    return None

  target_set = set(tuple(v) for v in located.values())
  restore_args = build_restore(meta_tree, (), target_set)
  restored = ckptr.restore(ckpt_path, restore_args=restore_args)

  print()
  print("=== Phase 1b unscan-prefix biases (expected: per the bug, frozen at init) ===")
  for label, full in [(l, located[l]) for l in [
      "decoder/moe_layers_0/mlp/MoeBlock_0/gate/bias",
      "decoder/moe_layers_1/mlp/MoeBlock_0/gate/bias",
      "decoder/moe_layers_2/mlp/MoeBlock_0/gate/bias",
  ] if l in located]:
    arr = get_at_path(restored, full)
    print(array_summary(label, arr))

  print()
  print("=== Phase 2 scan region biases (expected: each row updated independently per scan iter) ===")
  if scan_label in located:
    arr = get_at_path(restored, located[scan_label])
    arr = np.asarray(arr)
    # Stacked along param_scan_axis. For Linen scan with axis=0, shape is [scan_length, num_experts].
    # Detect scan axis as the first axis whose length matches expected scan_length=5 for ling3-tiny.
    scan_axis = 0 if arr.shape[0] in (4, 5, 6) else (1 if arr.ndim > 1 and arr.shape[1] in (4, 5, 6) else 0)
    print(array_summary(scan_label, arr, scan_axis=scan_axis))

  # --- Step 4: verdict ---
  print()
  print("=== Verdict ===")
  prefix_zero = []
  prefix_norms = []
  for label in ["decoder/moe_layers_0/mlp/MoeBlock_0/gate/bias",
                "decoder/moe_layers_1/mlp/MoeBlock_0/gate/bias",
                "decoder/moe_layers_2/mlp/MoeBlock_0/gate/bias"]:
    if label in located:
      arr = np.asarray(get_at_path(restored, located[label]))
      n = float(np.linalg.norm(arr.astype(np.float32)))
      prefix_norms.append(n)
      prefix_zero.append(n == 0.0)
  scan_nonzero_per_row = []
  if scan_label in located:
    arr = np.asarray(get_at_path(restored, located[scan_label]))
    sa = 0 if arr.shape[0] in (4, 5, 6) else 1
    for i in range(arr.shape[sa]):
      sl = [slice(None)] * arr.ndim
      sl[sa] = i
      scan_nonzero_per_row.append(float(np.linalg.norm(arr[tuple(sl)].astype(np.float32))) > 0)

  print(f"  Phase 1b prefix bias L2 norms: {prefix_norms}")
  print(f"  Phase 2  scan rows nonzero?  : {scan_nonzero_per_row}")
  if prefix_zero and all(prefix_zero) and scan_nonzero_per_row and all(scan_nonzero_per_row):
    print("  >> CONFIRMED: prefix biases are exactly zero while scan biases are nonzero.")
    print("  >> §2 root cause is supported by ckpt evidence.")
  elif prefix_zero and all(prefix_zero) and scan_nonzero_per_row and not any(scan_nonzero_per_row):
    print("  >> INCONCLUSIVE: every bias is zero. This looks like a fresh-from-conversion")
    print("     ckpt (no training steps). Re-run on a ckpt saved AFTER N>0 training steps.")
  elif prefix_zero and not all(prefix_zero):
    print("  >> HYPOTHESIS REJECTED: at least one prefix bias is nonzero. The trainer")
    print("     IS updating these layers somehow; re-read train.py / metric_logger to find where.")
  else:
    print("  >> Indeterminate: collect more context (training step count, training script).")


if __name__ == "__main__":
  main()
