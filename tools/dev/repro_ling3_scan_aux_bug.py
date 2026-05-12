#!/usr/bin/env python3
"""Reproduce the Ling3 scan-mode aux-collection bug (docs/ling3-scan-loss-analysis.md §2).

Standalone — no maxtext / jax imports needed. Copies the relevant function
logic verbatim from src/maxtext/trainers/pre_train/train.py (as of branch
chore/ling3-submit-helpers-and-pallas-bump, 2026-05-12).

Run on the Pod (or locally; only needs numpy):

    python tools/dev/repro_ling3_scan_aux_bug.py

The script:
  1. Builds a mock `config` matching ling3-tiny essentials.
  2. Builds two equivalent mock `intermediate_outputs`:
       (U) Unscan layout: 23 separate `moe_layers_{i}/moe_lb_loss` scalars.
       (S) Scan   layout: 3 unscan-prefix `moe_layers_{0,1,2}/moe_lb_loss`
                          + 1 stacked `moe_layers/.../moe_lb_loss` (the rest).
     Each MoE layer contributes lb_loss = 1.0, so the LOGICAL total is 23.0
     in both modes (true ground truth from the model).
  3. Runs the CURRENT (buggy) `_collect_moe_intermediate_sum` against (S).
  4. Runs a PROPOSED (fixed) version against (S).
  5. Runs the same function against (U) for the unscan baseline.
  6. Prints all three numbers and a verdict.

We probe both plausible scan-region structures because Ling3 wraps 4 MoE
sub-layers inside Ling3ScannableBlock, which may end up at:
    A) `decoder/moe_layers/moe_lb_loss`  (flat shape (scan_length,) — what
        the current code expects, matching DeepSeek's structure)
    B) `decoder/moe_layers/layers_{0..3}/moe_lb_loss`  (per sub-layer, each
        shape (scan_length,) — what NNX/Linen scan would naturally produce
        for a parent module with named children)

If real structure is (A): only Phase 1b prefix is missed → bug as documented.
If real structure is (B): the entire scan region ALSO returns 0.0 → much worse.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


# ---------------------------------------------------------------------------
# Verbatim copies from src/maxtext/trainers/pre_train/train.py
# ---------------------------------------------------------------------------

class _DecoderBlockType:
  """Stand-in for maxtext.common.common_types.DecoderBlockType (string-equal)."""
  DEEPSEEK = "deepseek"
  LING2 = "ling2"
  LING3 = "ling3"


def _get_nested_value(d, keys, default):
  """Stand-in for maxtext_utils.get_nested_value."""
  cur = d
  for k in keys:
    if isinstance(cur, dict) and k in cur:
      cur = cur[k]
    else:
      return default
  return cur


def collect_moe_intermediate_sum_BUGGY(config, intermediate_outputs, key):
  """Copy of train.py:100-148 (current code)."""
  def _get_sow_scalar(nested_key):
    raw = _get_nested_value(intermediate_outputs, nested_key, 0.0)
    return raw[-1] if isinstance(raw, tuple) else raw

  if config.decoder_block in (_DecoderBlockType.DEEPSEEK, _DecoderBlockType.LING2, _DecoderBlockType.LING3):
    if config.scan_layers:
      nested_key = ("intermediates", "decoder", "moe_layers", key)
      values = _get_nested_value(intermediate_outputs, nested_key, 0.0)
    else:
      num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
      values = []
      for i in range(num_moe_layers):
        nested_key = ("intermediates", "decoder", f"moe_layers_{i}", key)
        values.append(_get_sow_scalar(nested_key))
  else:
    raise NotImplementedError("only ling3 path covered in this repro")
  backbone_sum = float(np.sum(np.asarray(values, dtype=np.float64)))

  mtp_sum = 0.0
  if getattr(config, "mtp_num_layers", 0) > 0 and config.num_experts > 1:
    for k in range(1, config.mtp_num_layers + 1):
      nested_key = (
          "intermediates", "mtp_block", f"mtp_layer_{k}",
          f"mtp_{k}_transformer_layer", key,
      )
      mtp_sum += _get_sow_scalar(nested_key)
  return backbone_sum + mtp_sum


def _ling3_unscan_moe_prefix_count(config):
  """How many MoE layers live in the Phase 1b unscan prefix."""
  interval = config.inhomogeneous_layer_cycle_interval
  if config.first_num_dense_layers > 0:
    unscan_prefix = ((config.first_num_dense_layers + interval - 1) // interval) * interval
  else:
    unscan_prefix = 0
  return max(unscan_prefix - config.first_num_dense_layers, 0)


def collect_moe_intermediate_sum_FIXED(config, intermediate_outputs, key):
  """Proposed fix: also collect Phase 1b unscan-prefix MoE layers in scan mode.

  Also probes the scan-region structure both as a flat tensor under
  `moe_layers/{key}` AND as per-sub-layer tensors under `moe_layers/layers_*/{key}`
  to be robust against either layout.
  """
  def _get_sow_scalar(nested_key):
    raw = _get_nested_value(intermediate_outputs, nested_key, 0.0)
    return raw[-1] if isinstance(raw, tuple) else raw

  if config.decoder_block in (_DecoderBlockType.DEEPSEEK, _DecoderBlockType.LING2, _DecoderBlockType.LING3):
    if config.scan_layers:
      values = []
      # 1. Phase 1b unscan-prefix MoE layers (LING3-specific; 0 for DeepSeek/LING2 typical configs)
      prefix = _ling3_unscan_moe_prefix_count(config)
      for i in range(prefix):
        values.append(_get_sow_scalar(("intermediates", "decoder", f"moe_layers_{i}", key)))
      # 2. Phase 2 scan region — try flat path first
      flat = _get_nested_value(intermediate_outputs, ("intermediates", "decoder", "moe_layers", key), None)
      if flat is not None and not isinstance(flat, dict):
        values.append(flat)
      else:
        # Fallback: per sub-layer
        sub_dict = _get_nested_value(intermediate_outputs, ("intermediates", "decoder", "moe_layers"), {})
        if isinstance(sub_dict, dict):
          for sub_key in sorted(sub_dict.keys()):
            if sub_key.startswith("layers_"):
              raw = sub_dict[sub_key].get(key) if isinstance(sub_dict[sub_key], dict) else None
              if raw is not None:
                values.append(raw[-1] if isinstance(raw, tuple) else raw)
    else:
      num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
      values = []
      for i in range(num_moe_layers):
        values.append(_get_sow_scalar(("intermediates", "decoder", f"moe_layers_{i}", key)))
  else:
    raise NotImplementedError("only ling3 path covered in this repro")
  flat_array = np.concatenate([np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel() for v in values]) \
      if values else np.zeros((0,), dtype=np.float64)
  backbone_sum = float(flat_array.sum())

  mtp_sum = 0.0
  if getattr(config, "mtp_num_layers", 0) > 0 and config.num_experts > 1:
    for k in range(1, config.mtp_num_layers + 1):
      mtp_sum += _get_sow_scalar((
          "intermediates", "mtp_block", f"mtp_layer_{k}",
          f"mtp_{k}_transformer_layer", key,
      ))
  return backbone_sum + mtp_sum


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

def make_ling3_tiny_config(scan_layers: bool):
  return SimpleNamespace(
      decoder_block=_DecoderBlockType.LING3,
      scan_layers=scan_layers,
      num_decoder_layers=24,
      first_num_dense_layers=1,
      inhomogeneous_layer_cycle_interval=4,
      num_experts=128,
      mtp_num_layers=1,
  )


def make_unscan_intermediates(per_layer_value=1.0, mtp_value=1.0):
  """23 separate moe_layers_{i}/moe_lb_loss (Flax sow tuple wrapping)."""
  decoder = {}
  for i in range(23):  # num_decoder_layers - first_num_dense_layers = 24 - 1
    decoder[f"moe_layers_{i}"] = {"moe_lb_loss": (np.float32(per_layer_value),)}
  return {
      "intermediates": {
          "decoder": decoder,
          "mtp_block": {
              "mtp_layer_1": {
                  "mtp_1_transformer_layer": {"moe_lb_loss": (np.float32(mtp_value),)},
              },
          },
      },
  }


def make_scan_intermediates_flat(per_layer_value=1.0, mtp_value=1.0):
  """Layout (A): scan region as a flat array under moe_layers/moe_lb_loss."""
  decoder = {}
  # Phase 1b unscan prefix: 3 layers
  for i in range(3):
    decoder[f"moe_layers_{i}"] = {"moe_lb_loss": (np.float32(per_layer_value),)}
  # Phase 2 scan region: 5 ScannableBlock × 4 sub-layers = 20 sown values
  # If the structure flattens to moe_layers/moe_lb_loss directly, shape would be (20,)
  # (or (5, 4) if scan stacks separately and inside-block 4-loop concats).
  decoder["moe_layers"] = {"moe_lb_loss": np.full(20, per_layer_value, dtype=np.float32)}
  return {
      "intermediates": {
          "decoder": decoder,
          "mtp_block": {
              "mtp_layer_1": {
                  "mtp_1_transformer_layer": {"moe_lb_loss": (np.float32(mtp_value),)},
              },
          },
      },
  }


def make_scan_intermediates_nested(per_layer_value=1.0, mtp_value=1.0):
  """Layout (B): scan region as moe_layers/layers_{0..3}/moe_lb_loss, each shape (5,).

  This is what NNX/Linen scan would naturally produce when the scanned module
  (Ling3ScannableBlock) has 4 named sub-layers each sowing inside their own scope.
  """
  decoder = {}
  for i in range(3):
    decoder[f"moe_layers_{i}"] = {"moe_lb_loss": (np.float32(per_layer_value),)}
  scan_subdict = {}
  for j in range(4):
    scan_subdict[f"layers_{j}"] = {"moe_lb_loss": np.full(5, per_layer_value, dtype=np.float32)}
  decoder["moe_layers"] = scan_subdict
  return {
      "intermediates": {
          "decoder": decoder,
          "mtp_block": {
              "mtp_layer_1": {
                  "mtp_1_transformer_layer": {"moe_lb_loss": (np.float32(mtp_value),)},
              },
          },
      },
  }


# ---------------------------------------------------------------------------
# Repro
# ---------------------------------------------------------------------------

def report(label, value, expected):
  status = "OK" if abs(value - expected) < 1e-5 else "WRONG"
  print(f"  {label:<70s}  -> {value:>8.3f}  (expected {expected:>5.1f})  [{status}]")


def main():
  cfg_unscan = make_ling3_tiny_config(scan_layers=False)
  cfg_scan = make_ling3_tiny_config(scan_layers=True)

  inter_unscan = make_unscan_intermediates()
  inter_scan_flat = make_scan_intermediates_flat()
  inter_scan_nested = make_scan_intermediates_nested()

  GROUND_TRUTH = 23 + 1  # 23 backbone MoE + 1 MTP, all contributing 1.0

  print("=" * 90)
  print("Ground truth: every MoE layer (23 backbone + 1 MTP) sows lb_loss=1.0")
  print(f"=> _collect_moe_intermediate_sum SHOULD return {GROUND_TRUTH:.1f} in all modes/layouts.")
  print("=" * 90)

  print("\n[1] Unscan baseline (sanity check the function works at all)")
  v = collect_moe_intermediate_sum_BUGGY(cfg_unscan, inter_unscan, "moe_lb_loss")
  report("unscan + current code", v, GROUND_TRUTH)

  print("\n[2] Scan mode, layout (A) = flat moe_layers/moe_lb_loss tensor of length 20")
  v = collect_moe_intermediate_sum_BUGGY(cfg_scan, inter_scan_flat, "moe_lb_loss")
  report("scan + current code (BUGGY) on flat layout", v, GROUND_TRUTH)
  v = collect_moe_intermediate_sum_FIXED(cfg_scan, inter_scan_flat, "moe_lb_loss")
  report("scan + proposed FIX on flat layout", v, GROUND_TRUTH)

  print("\n[3] Scan mode, layout (B) = nested moe_layers/layers_{0..3}/moe_lb_loss, each (5,)")
  v = collect_moe_intermediate_sum_BUGGY(cfg_scan, inter_scan_nested, "moe_lb_loss")
  report("scan + current code (BUGGY) on nested layout", v, GROUND_TRUTH)
  v = collect_moe_intermediate_sum_FIXED(cfg_scan, inter_scan_nested, "moe_lb_loss")
  report("scan + proposed FIX on nested layout", v, GROUND_TRUTH)

  print("\n" + "=" * 90)
  print("Interpretation:")
  print("  - If [1] returns 24.0 -> baseline OK, the function works for unscan.")
  print("  - If [2] BUGGY returns 21.0 -> bug as documented (3 unscan-prefix layers lost).")
  print("  - If [2] BUGGY returns 1.0 -> the BUGGY code only sees MTP, not the scan tensor")
  print("    either (something else wrong with how it reads `values`).")
  print("  - If [3] BUGGY returns 1.0 -> if real structure is nested, the entire scan")
  print("    region's lb_loss is being SILENTLY DROPPED. That's a much bigger bug.")
  print("  - The FIX line in [2]/[3] should hit 24.0 in BOTH layouts.")
  print("=" * 90)
  print()
  print("Next step: dump the real intermediates structure on the Pod with R1 from")
  print("the analysis doc to determine which layout (A or B) actually applies.")


if __name__ == "__main__":
  main()
