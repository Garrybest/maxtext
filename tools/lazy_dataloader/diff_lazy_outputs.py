#!/usr/bin/env python3
# pylint: disable=import-outside-toplevel
"""Compare antllm .pt batch outputs with MaxText .npy sample outputs.

antllm's check_outputs.py dumps batches as .pt files (torch tensors).
Our check_lazy_outputs.py dumps individual samples as .npy files.
This script loads both, flattens batches into samples, and compares
them element-by-element to verify alignment.

Usage:

  # Compare antllm .pt directory with maxtext .npy directory
  python diff_lazy_outputs.py \
      --antllm-dir /path/to/antllm_outputs/dataloader \
      --maxtext-dir /path/to/maxtext_outputs

  # Compare two .npy directories (both from check_lazy_outputs.py)
  python diff_lazy_outputs.py \
      --maxtext-dir /path/to/run1 \
      --maxtext-dir2 /path/to/run2

  # Show first N mismatched tokens per sample
  python diff_lazy_outputs.py \
      --antllm-dir /path/to/antllm_outputs/dataloader \
      --maxtext-dir /path/to/maxtext_outputs \
      --show-diff-tokens 20

  # Compare only first N samples
  python diff_lazy_outputs.py \
      --antllm-dir /path/to/antllm_outputs/dataloader \
      --maxtext-dir /path/to/maxtext_outputs \
      --max-samples 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def get_args():
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Compare antllm .pt outputs with MaxText .npy outputs",
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("--antllm-dir", type=str, default=None, help="Directory with antllm .pt batch files")
  parser.add_argument("--maxtext-dir", type=str, required=True, help="Directory with MaxText .npy sample files")
  parser.add_argument("--maxtext-dir2", type=str, default=None, help="Second .npy directory (for npy-vs-npy comparison)")
  parser.add_argument("--max-samples", type=int, default=None, help="Compare at most N samples")
  parser.add_argument(
      "--show-diff-tokens",
      type=int,
      default=10,
      help="Show first N mismatched token positions per sample",
  )
  parser.add_argument(
      "--batch-key", type=str, default=None, help="Key to extract tokens from .pt batch dict (auto-detected if None)"
  )
  parser.add_argument("--quiet", action="store_true", help="Only print summary, not per-sample details")
  parser.add_argument(
      "--aggregate-steps",
      action="store_true",
      help="Aggregate all ranks per step, then compare step-level token sets. "
      "antllm .pt files are grouped by step; MaxText uses --simulate-global output.",
  )
  return parser.parse_args()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _numeric_sort_key(name: str):
  """Sort filenames numerically by extracting trailing integer before extension.

  e.g. 'train_1_0_0_10.pt' -> ('train_1_0_0_', 10) so that step 2 sorts
  before step 10 (lexicographic sort would put '10' before '2').
  """
  import re

  m = re.match(r"^(.*?)(\d+)\.pt$", name)
  if m:
    return (m.group(1), int(m.group(2)))
  return (name, 0)


def load_antllm_samples(pt_dir: str, batch_key: str | None = None) -> list[np.ndarray]:
  """Load antllm .pt batch files and flatten into individual samples.

  antllm's check_outputs.py saves batches with filenames like:
    train_{world_size}_{rank}_{scatter_id}_{step}.pt

  Each .pt file is a torch.save'd object — typically a dict or tuple
  containing token tensors of shape (batch_size, seq_length).
  """
  try:
    import torch
  except ImportError:
    print("ERROR: torch is required to load .pt files. Install with: pip install torch")
    sys.exit(1)

  pt_dir = Path(pt_dir)
  pt_files = sorted(pt_dir.glob("*.pt"), key=lambda p: _numeric_sort_key(p.name))
  if not pt_files:
    raise FileNotFoundError(f"No .pt files found in {pt_dir}")

  samples = []
  for pt_file in pt_files:
    batch = torch.load(pt_file, map_location="cpu", weights_only=False)
    tokens = _extract_tokens_from_batch(batch, batch_key)
    if tokens is None:
      print(f"  Warning: could not extract tokens from {pt_file.name}, skipping")
      continue
    # tokens shape: (batch_size, seq_length) or (batch_size, seq_length+1)
    tokens_np = tokens.numpy() if hasattr(tokens, "numpy") else np.asarray(tokens)
    if tokens_np.ndim == 1:
      samples.append(tokens_np)
    elif tokens_np.ndim == 2:
      for row in tokens_np:
        samples.append(row)
    else:
      print(f"  Warning: unexpected tensor shape {tokens_np.shape} in {pt_file.name}")

  print(f"  Loaded {len(samples)} samples from {len(pt_files)} .pt files in {pt_dir}")
  return samples


def _extract_tokens_from_batch(batch, batch_key: str | None):
  """Extract token tensor from various batch formats."""
  import torch

  # Direct tensor
  if isinstance(batch, (torch.Tensor, np.ndarray)):
    return batch

  # Dict with known keys
  if isinstance(batch, dict):
    if batch_key and batch_key in batch:
      return batch[batch_key]
    # Try common keys in order of likelihood
    for key in ["text", "input_ids", "tokens", "inputs", "input_tokens"]:
      if key in batch:
        return batch[key]
    # Fallback: first tensor-like value
    for key, val in batch.items():
      if isinstance(val, (torch.Tensor, np.ndarray)):
        if val.ndim >= 1 and val.dtype in (torch.int32, torch.int64, torch.long, np.int32, np.int64):
          print(f"    Auto-detected batch key: '{key}' (shape={tuple(val.shape)})")
          return val

  # Tuple/list — first element is usually tokens
  if isinstance(batch, (tuple, list)):
    if len(batch) > 0:
      first = batch[0]
      if isinstance(first, (torch.Tensor, np.ndarray)):
        return first
      if isinstance(first, dict):
        return _extract_tokens_from_batch(first, batch_key)

  return None


def load_npy_samples(npy_dir: str) -> list[np.ndarray]:
  """Load .npy sample files from a directory."""
  npy_dir = Path(npy_dir)
  npy_files = sorted(npy_dir.glob("*.npy"), key=lambda p: p.name)
  if not npy_files:
    raise FileNotFoundError(f"No .npy files found in {npy_dir}")

  samples = []
  for npy_file in npy_files:
    arr = np.load(npy_file)
    if arr.ndim == 1:
      samples.append(arr)
    elif arr.ndim == 2:
      for row in arr:
        samples.append(row)
    else:
      print(f"  Warning: unexpected shape {arr.shape} in {npy_file.name}")

  print(f"  Loaded {len(samples)} samples from {len(npy_files)} .npy files in {npy_dir}")
  return samples


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def load_antllm_steps(pt_dir: str, batch_key: str | None = None) -> dict[int, list[np.ndarray]]:
  """Load antllm .pt files grouped by step, aggregating all ranks.

  Filename format: {stage}_{world_size}_{rank}_{scatter_id}_{step}.pt
  Returns: {step: [sample_array, ...]} with all ranks' samples merged per step.
  """
  import re

  try:
    import torch
  except ImportError:
    print("ERROR: torch is required to load .pt files. Install with: pip install torch")
    sys.exit(1)

  pt_dir = Path(pt_dir)
  pt_files = sorted(pt_dir.glob("*.pt"))
  if not pt_files:
    raise FileNotFoundError(f"No .pt files found in {pt_dir}")

  steps: dict[int, list[np.ndarray]] = {}
  for pt_file in pt_files:
    m = re.match(r"^(.+?)_(\d+)\.pt$", pt_file.name)
    if not m:
      continue
    step = int(m.group(2))
    batch = torch.load(pt_file, map_location="cpu", weights_only=False)
    tokens = _extract_tokens_from_batch(batch, batch_key)
    if tokens is None:
      continue
    tokens_np = tokens.numpy() if hasattr(tokens, "numpy") else np.asarray(tokens)
    if step not in steps:
      steps[step] = []
    if tokens_np.ndim == 1:
      steps[step].append(tokens_np)
    elif tokens_np.ndim == 2:
      for row in tokens_np:
        steps[step].append(row)

  print(f"  Loaded {len(steps)} steps from {len(pt_files)} .pt files in {pt_dir}")
  return steps


def compare_steps(
    steps_a: dict[int, list[np.ndarray]],
    steps_b: dict[int, list[np.ndarray]],
    label_a: str,
    label_b: str,
    show_diff_tokens: int = 10,
    quiet: bool = False,
) -> dict:
  """Compare per-step aggregated token sets between two systems."""
  all_steps = sorted(set(steps_a.keys()) | set(steps_b.keys()))
  n_match = 0
  n_mismatch = 0

  print(f"\nComparing {len(all_steps)} steps ({label_a} vs {label_b})")
  print()

  for step in all_steps:
    a_samples = steps_a.get(step, [])
    b_samples = steps_b.get(step, [])

    if len(a_samples) != len(b_samples):
      n_mismatch += 1
      if not quiet:
        print(f"  Step {step}: SAMPLE COUNT MISMATCH ({len(a_samples)} vs {len(b_samples)})")
      continue

    # Sort both by content for set comparison (order within step doesn't matter)
    a_sorted = sorted(a_samples, key=lambda x: x.tobytes())
    b_sorted = sorted(b_samples, key=lambda x: x.tobytes())

    step_match = True
    for i, (sa, sb) in enumerate(zip(a_sorted, b_sorted)):
      if not np.array_equal(sa, sb):
        step_match = False
        if not quiet:
          diff_positions = np.where(sa != sb)[0]
          print(f"  Step {step}: MISMATCH in sorted sample {i}, {len(diff_positions)} positions differ")
          show_n = min(show_diff_tokens, len(diff_positions))
          for j in range(show_n):
            pos = int(diff_positions[j])
            print(f"    pos {pos:>6d}: {label_a}={int(sa[pos]):>6d}  {label_b}={int(sb[pos]):>6d}")
        break

    if step_match:
      n_match += 1
      if not quiet:
        print(f"  Step {step}: MATCH ({len(a_samples)} samples)")
    else:
      n_mismatch += 1

  print()
  print("=" * 60)
  print("Step-Level Aggregate Comparison")
  print("=" * 60)
  print(f"  Steps compared:  {len(all_steps)}")
  print(f"  Matched:         {n_match}")
  print(f"  Mismatched:      {n_mismatch}")
  if n_mismatch == 0:
    print("  Result:          ALL STEPS MATCH")
  else:
    print("  Result:          DIFFERENCES FOUND")
  print("=" * 60)

  return {"n_steps": len(all_steps), "n_match": n_match, "n_mismatch": n_mismatch}


def compare_samples(
    samples_a: list[np.ndarray],
    samples_b: list[np.ndarray],
    label_a: str,
    label_b: str,
    max_samples: int | None = None,
    show_diff_tokens: int = 10,
    quiet: bool = False,
) -> dict:
  """Compare two lists of samples element-by-element."""
  n_a, n_b = len(samples_a), len(samples_b)
  n_compare = min(n_a, n_b)
  if max_samples is not None:
    n_compare = min(n_compare, max_samples)

  print(f"\nComparing {n_compare} samples ({label_a}: {n_a} total, {label_b}: {n_b} total)")
  if n_a != n_b:
    print(f"  WARNING: sample counts differ ({n_a} vs {n_b})")
  print()

  n_match = 0
  n_mismatch = 0
  n_length_mismatch = 0
  first_mismatch_idx = None
  mismatch_positions = []  # (sample_idx, first_diff_pos)

  for i in range(n_compare):
    a, b = samples_a[i], samples_b[i]

    if len(a) != len(b):
      n_length_mismatch += 1
      n_mismatch += 1
      if first_mismatch_idx is None:
        first_mismatch_idx = i
      if not quiet:
        print(f"  Sample {i}: LENGTH MISMATCH ({len(a)} vs {len(b)})")
      continue

    if np.array_equal(a, b):
      n_match += 1
    else:
      n_mismatch += 1
      diff_mask = a != b
      diff_positions = np.where(diff_mask)[0]
      n_diffs = len(diff_positions)
      mismatch_positions.append((i, int(diff_positions[0])))

      if first_mismatch_idx is None:
        first_mismatch_idx = i

      if not quiet:
        print(f"  Sample {i}: MISMATCH at {n_diffs}/{len(a)} positions")
        show_n = min(show_diff_tokens, n_diffs)
        for j in range(show_n):
          pos = int(diff_positions[j])
          print(f"    pos {pos:>6d}: {label_a}={int(a[pos]):>6d}  {label_b}={int(b[pos]):>6d}")
        if n_diffs > show_n:
          print(f"    ... and {n_diffs - show_n} more differences")

  # Summary
  print()
  print("=" * 60)
  print("Comparison Summary")
  print("=" * 60)
  print(f"  Samples compared:      {n_compare}")
  print(f"  Matched:               {n_match}")
  print(f"  Mismatched:            {n_mismatch}")
  if n_length_mismatch > 0:
    print(f"    (length mismatch):   {n_length_mismatch}")
  if n_mismatch == 0:
    print("  Result:                ALL MATCH")
  else:
    print("  Result:                DIFFERENCES FOUND")
    print(f"  First mismatch at:     sample {first_mismatch_idx}")
  if n_a != n_b:
    print(f"  Uncompared samples:    {abs(n_a - n_b)}")
  print("=" * 60)

  return {
      "n_compare": n_compare,
      "n_match": n_match,
      "n_mismatch": n_mismatch,
      "n_length_mismatch": n_length_mismatch,
      "first_mismatch_idx": first_mismatch_idx,
  }


def load_npy_groups(group_dir: str) -> dict[int, list[np.ndarray]]:
  """Load per-group step .npy files from check_lazy_outputs.py --all-groups output.

  Directory structure: <group_dir>/<group_id>/step_<N>.npy
  Each .npy has shape (batch_size, seq_len).
  Returns: {step: [sample_array, ...]} with all groups' samples merged per step.
  """
  import re

  group_dir = Path(group_dir)
  sub_dirs = sorted([d for d in group_dir.iterdir() if d.is_dir()], key=lambda p: int(p.name))
  if not sub_dirs:
    raise FileNotFoundError(f"No group subdirectories found in {group_dir}")

  steps: dict[int, list[np.ndarray]] = {}
  for gdir in sub_dirs:
    npy_files = sorted(gdir.glob("step_*.npy"))
    for npy_file in npy_files:
      m = re.match(r"step_(\d+)\.npy", npy_file.name)
      if not m:
        continue
      step = int(m.group(1))
      arr = np.load(npy_file)
      if step not in steps:
        steps[step] = []
      if arr.ndim == 1:
        steps[step].append(arr)
      elif arr.ndim == 2:
        for row in arr:
          steps[step].append(row)

  n_samples = sum(len(v) for v in steps.values())
  print(f"  Loaded {len(steps)} steps, {n_samples} total samples from {len(sub_dirs)} groups in {group_dir}")
  return steps


def main():
  args = get_args()

  if args.aggregate_steps:
    # Step-level aggregate comparison
    if args.antllm_dir is None and args.maxtext_dir2 is None:
      print("ERROR: --aggregate-steps requires --antllm-dir or --maxtext-dir2")
      sys.exit(1)

    # Load side A (antllm .pt or maxtext npy groups)
    if args.antllm_dir:
      print(f"Loading antllm .pt steps from {args.antllm_dir}...")
      steps_a = load_antllm_steps(args.antllm_dir, args.batch_key)
      label_a = "antllm"
    else:
      print(f"Loading maxtext .npy group steps from {args.maxtext_dir2}...")
      steps_a = load_npy_groups(args.maxtext_dir2)
      label_a = "maxtext2"

    # Load side B (maxtext npy groups)
    print(f"Loading maxtext .npy group steps from {args.maxtext_dir}...")
    steps_b = load_npy_groups(args.maxtext_dir)
    label_b = "maxtext"

    result = compare_steps(
        steps_a,
        steps_b,
        label_a=label_a,
        label_b=label_b,
        show_diff_tokens=args.show_diff_tokens,
        quiet=args.quiet,
    )
    sys.exit(0 if result["n_mismatch"] == 0 else 1)

  if args.antllm_dir is None and args.maxtext_dir2 is None:
    print("ERROR: provide --antllm-dir (for pt-vs-npy) or --maxtext-dir2 (for npy-vs-npy)")
    sys.exit(1)

  if args.antllm_dir and args.maxtext_dir2:
    print("ERROR: use either --antllm-dir or --maxtext-dir2, not both")
    sys.exit(1)

  # Load samples
  if args.antllm_dir:
    print(f"Loading antllm .pt samples from {args.antllm_dir}...")
    samples_a = load_antllm_samples(args.antllm_dir, args.batch_key)
    label_a = "antllm"
  else:
    print(f"Loading .npy samples from {args.maxtext_dir2}...")
    samples_a = load_npy_samples(args.maxtext_dir2)
    label_a = "maxtext2"

  print(f"Loading MaxText .npy samples from {args.maxtext_dir}...")
  samples_b = load_npy_samples(args.maxtext_dir)
  label_b = "maxtext"

  # Compare
  result = compare_samples(
      samples_a,
      samples_b,
      label_a=label_a,
      label_b=label_b,
      max_samples=args.max_samples,
      show_diff_tokens=args.show_diff_tokens,
      quiet=args.quiet,
  )

  sys.exit(0 if result["n_mismatch"] == 0 else 1)


if __name__ == "__main__":
  main()
