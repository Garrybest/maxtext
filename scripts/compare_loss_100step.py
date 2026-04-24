#!/usr/bin/env python3
"""Compare MaxText training loss against a reference for 100 steps.

Validation criteria (thresholds configurable via CLI):
  - Step 0 (first step): relative diff <= 0.01% for lm_loss and mtp_loss
  - Steps 1-99: average relative diff <= 0.1% for lm_loss and mtp_loss

Usage:
  python3 scripts/compare_loss_100step.py <metrics_file> [--reference PATH] \\
      [--first-step-tol FLOAT] [--avg-tol FLOAT]

The metrics file is the JSONL emitted by MaxText's metric_logger
(train.py sets `metrics_file=...` via config). The reference file is a
JSON document with schema:
    {"losses": [{"maxtext_step": int, "lm_loss": float, "mtp_loss": float}, ...]}
By default the reference is read from the GCS-Fuse mount that the CI
loss-validation JobSet mounts at /models.
"""

import argparse
import json
import os
import sys

DEFAULT_REFERENCE = "/models/ci-reference/loss_ling2_100step.json"
DEFAULT_FIRST_STEP_TOL = 0.0001  # 0.01%
DEFAULT_AVG_TOL = 0.001  # 0.1%


def parse_reference_losses(ref_file):
  """Load reference losses from JSON file.

  Returns:
    List of (maxtext_step, lm_loss, mtp_loss) sorted by step.
  """
  with open(ref_file, "r", encoding="utf-8") as f:
    data = json.load(f)
  return [(e["maxtext_step"], e["lm_loss"], e["mtp_loss"]) for e in data["losses"]]


def parse_maxtext_metrics(metrics_file):
  """Parse lm_loss and raw_mtp_loss from MaxText metrics JSONL file.

  Returns:
    List of (step, lm_loss, mtp_loss) sorted by step.
  """
  losses = []
  with open(metrics_file, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      data = json.loads(line)
      step = int(data["step"])
      lm_loss = data["learning/lm_loss"]
      raw_mtp_loss = data["learning/raw_mtp_loss"]
      losses.append((step, lm_loss, raw_mtp_loss))
  losses.sort(key=lambda x: x[0])
  return losses


def relative_diff(ref, actual):
  """Compute relative difference: |ref - actual| / |ref|."""
  if ref == 0:
    return abs(actual)
  return abs(ref - actual) / abs(ref)


def main():
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("metrics_file", help="MaxText metrics JSONL file")
  parser.add_argument(
      "--reference",
      default=DEFAULT_REFERENCE,
      help=f"Reference losses JSON (default: {DEFAULT_REFERENCE})",
  )
  parser.add_argument(
      "--first-step-tol",
      type=float,
      default=DEFAULT_FIRST_STEP_TOL,
      help=f"Step-0 relative diff tolerance (default: {DEFAULT_FIRST_STEP_TOL})",
  )
  parser.add_argument(
      "--avg-tol",
      type=float,
      default=DEFAULT_AVG_TOL,
      help=f"Average relative diff tolerance for steps 1-N (default: {DEFAULT_AVG_TOL})",
  )
  args = parser.parse_args()

  ref_file = args.reference
  metrics_file = args.metrics_file

  if not os.path.exists(ref_file):
    print(f"ERROR: Reference file not found: {ref_file}")
    sys.exit(1)
  if not os.path.exists(metrics_file):
    print(f"ERROR: Metrics file not found: {metrics_file}")
    sys.exit(1)

  print("=" * 70)
  print("  100-Step Loss Validation: MaxText vs Reference")
  print("=" * 70)

  print(f"\n[1/3] Loading reference losses from {ref_file}...")
  ref_losses = parse_reference_losses(ref_file)
  print(f"  Found {len(ref_losses)} reference steps")

  print(f"\n[2/3] Reading MaxText metrics from {metrics_file}...")
  mxt_losses = parse_maxtext_metrics(metrics_file)
  print(f"  Found {len(mxt_losses)} steps")

  if not ref_losses or not mxt_losses:
    print("  ERROR: No steps to compare!")
    sys.exit(1)

  ref_by_step = {s: (lm, mtp) for s, lm, mtp in ref_losses}
  mxt_by_step = {s: (lm, mtp) for s, lm, mtp in mxt_losses}
  required_steps = sorted(ref_by_step.keys())

  missing_mxt = [s for s in required_steps if s not in mxt_by_step]
  if missing_mxt:
    print(f"  ERROR: MaxText metrics missing required steps: {missing_mxt}")
    sys.exit(1)

  print("\n[3/3] Comparing losses...")
  failures = []
  lm_diffs = []
  mtp_diffs = []

  for step in required_steps:
    ref_lm, ref_mtp = ref_by_step[step]
    mxt_lm, mxt_mtp = mxt_by_step[step]

    lm_diff = relative_diff(ref_lm, mxt_lm)
    mtp_diff = relative_diff(ref_mtp, mxt_mtp)

    lm_diffs.append(lm_diff)
    mtp_diffs.append(mtp_diff)

    print(
        f"  step {step:3d} | "
        f"lm: MaxText={mxt_lm:.6e} Ref={ref_lm:.6e} diff={lm_diff:.4%} | "
        f"mtp: MaxText={mxt_mtp:.6e} Ref={ref_mtp:.6e} diff={mtp_diff:.4%}"
    )

  print(f"\n--- Step 0 Check (threshold: {args.first_step_tol:.2%}) ---")
  for name, diff in [("lm_loss", lm_diffs[0]), ("mtp_loss", mtp_diffs[0])]:
    status = "PASS" if diff <= args.first_step_tol else "FAIL"
    print(f"  [{status}] step 0 {name}: diff={diff:.4%}")
    if diff > args.first_step_tol:
      failures.append(f"step 0 {name}: diff={diff:.4%} > {args.first_step_tol:.2%}")

  if len(required_steps) > 1:
    avg_lm_diff = sum(lm_diffs[1:]) / len(lm_diffs[1:])
    avg_mtp_diff = sum(mtp_diffs[1:]) / len(mtp_diffs[1:])

    print(f"\n--- Steps 1-{required_steps[-1]} Average Check (threshold: {args.avg_tol:.2%}) ---")
    for name, avg_diff in [("lm_loss", avg_lm_diff), ("mtp_loss", avg_mtp_diff)]:
      status = "PASS" if avg_diff <= args.avg_tol else "FAIL"
      print(f"  [{status}] average {name}: diff={avg_diff:.4%}")
      if avg_diff > args.avg_tol:
        failures.append(f"steps 1-{required_steps[-1]} average {name}: diff={avg_diff:.4%} > {args.avg_tol:.2%}")

  print()
  if failures:
    print("=" * 70)
    print(f"  FAILED: {len(failures)} check(s) failed:")
    print("=" * 70)
    for f in failures:
      print(f"  - {f}")
    sys.exit(1)
  else:
    print("=" * 70)
    print("  PASSED: All loss comparisons within thresholds")
    print(f"    Step 0:       lm={lm_diffs[0]:.4%}, mtp={mtp_diffs[0]:.4%} (limit {args.first_step_tol:.2%})")
    if len(required_steps) > 1:
      print(
          f"    Steps 1-{required_steps[-1]:3d}: "
          f"lm_avg={avg_lm_diff:.4%}, mtp_avg={avg_mtp_diff:.4%} "
          f"(limit {args.avg_tol:.2%})"
      )
    print("=" * 70)


if __name__ == "__main__":
  main()
