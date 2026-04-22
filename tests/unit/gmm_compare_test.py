#!/usr/bin/env python3
"""Compare MaxText blockwise FP8 GMM outputs with Megatron FP8 MoE expert dumps.

Full MoE forward comparison (GmmMoEForwardTest):
  Loads moe_input from dump, runs full MaxText MoE forward (FP8 blockwise)
  with Orbax weights, compares moe_output with the dump reference.
  Works with both BF16 and FP8 dumps (automatically detects).

Both sides use blockwise FP8 quantization:
  - Megatron: TE Float8BlockScaling (block_scaling_dim=1)
  - MaxText:  Qwix fp8_blockwise (tile_size=128, absmax calibration)

Environment variables:
  DUMP_DIR  - Megatron dump base dir (auto-detect)
  CKPT_PATH - Orbax checkpoint path

Run via pytest:
  DUMP_DIR=/path/to/dump python3 -m pytest tests/gmm_compare_test.py -v -s
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure repo root and src/ are on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_CKPT_PATH = "/models/gpu-ckpt-ling2.5/AL_MODEL_HF20E256_ORBAX_MTP/0/items/"

# Full MoE forward thresholds (BF16 MaxText vs FP8 Megatron dump).
# Relaxed vs BF16-vs-BF16 comparison due to FP8 quantization noise.
MOE_FWD_MIN_COSINE = 0.995
MOE_FWD_MAX_REL_L2 = 0.05

# ALModel MoE config.
MOE_LAYER_IDX = 1  # First MoE layer (layer_0 is dense)


# ── Helpers ──────────────────────────────────────────────────────────────────


def find_dump_dir(base):
  """Locate the step directory containing rank_* subdirs."""
  entries = os.listdir(base)
  step_dirs = sorted(d for d in entries if d.startswith("step_"))
  if not step_dirs:
    raise ValueError(f"No step_* dirs in {base}")
  step_path = os.path.join(base, step_dirs[0])
  if not any(d.startswith("rank_") for d in os.listdir(step_path)):
    raise ValueError(f"No rank_* dirs in {step_path}")
  return step_path


# ── Full MoE forward comparison ──────────────────────────────────────────────

# ALModel MoE profile (matches --al-model-profile in moe_cross_compare)
AL_MODEL_MOE_CONFIG = {
    "hidden_size": 2048,
    "ffn_hidden_size": 512,
    "shared_ffn_hidden_size": 2048,
    "num_experts": 256,
    "top_k": 8,
    "enable_shared_expert": True,
    "routed_score_func": "sigmoid",
    "routed_scaling_factor": 2.5,
    "norm_topk_prob": False,
    "n_routing_groups": 8,
    "topk_routing_group": 4,
    "routed_bias": True,
    "batch_size": 1,
    "seq_len": 4,
    "compute_dtype": "bf16",
    "matmul_precision": "default",
    "activations_in_float32": False,
    "ling2_profile": True,
    "orbax_layer_prefix": None,
    "num_dense_layers": 1,
    "quantization": "fp8_blockwise",
    "use_qwix_quantization": True,
    "use_tokamax_gmm": True,
}


class GmmMoEForwardTest(unittest.TestCase):
  """Full MoE forward comparison: MaxText FP8 blockwise vs Megatron FP8 dump.

  Loads moe_input from dump, runs full MaxText MoE forward with FP8
  blockwise quantization (Qwix) and Orbax weights, compares moe_output
  with Megatron's FP8 (TE Float8BlockScaling) dump output.
  """

  @classmethod
  def setUpClass(cls):
    from tests.unit.moe_cross_compare import (  # pylint: disable=import-outside-toplevel
        build_maxtext_moe,
        load_megatron_data,
        load_orbax_weights,
        run_maxtext_forward,
        _normalize_dump_keys,
        _compare_pair,
    )

    cls._compare_pair = staticmethod(_compare_pair)

    dump_dir_env = os.environ.get("DUMP_DIR")
    if not dump_dir_env:
      raise unittest.SkipTest("DUMP_DIR not set; skipping MoE forward test")

    cls.ckpt_path = os.environ.get("CKPT_PATH", DEFAULT_CKPT_PATH)
    cls.dump_dir = find_dump_dir(dump_dir_env)

    import argparse  # pylint: disable=import-outside-toplevel

    cls.args = argparse.Namespace(**dict(AL_MODEL_MOE_CONFIG, orbax_ckpt_path=cls.ckpt_path))

    # Load Megatron MoE data via Argus
    cls.mg_data = load_megatron_data(cls.dump_dir, MOE_LAYER_IDX, mode="moe", dp_mode="replica")
    if not cls.mg_data:
      raise unittest.SkipTest(f"No MoE data for layer {MOE_LAYER_IDX} in {cls.dump_dir}")

    _normalize_dump_keys(cls.mg_data)

    # Derive top-k from routing_map + gate scores
    routing_map = cls.mg_data.get("moe_router_output_1")
    gate_scores = cls.mg_data.get("moe_router_output")
    if routing_map is not None and gate_scores is not None:
      topk = cls.args.top_k
      masked = np.where(routing_map, gate_scores, -np.inf)
      cls.mg_data["moe_topk_indices"] = np.argsort(-masked, axis=-1)[:, :topk].astype(np.int32)

    cls.mg_input = cls.mg_data.get("moe_input")
    if cls.mg_input is None:
      raise unittest.SkipTest("'moe_input' not found in dump")
    cls.mg_output = cls.mg_data.get("moe_output")
    if cls.mg_output is None:
      raise unittest.SkipTest("'moe_output' not found in dump")

    # Build MaxText MoE model, load weights, run forward
    print("  Building MaxText MoE module...")
    cls.model, cls.cfg, cls.mesh = build_maxtext_moe(cls.args)
    print("  Loading Orbax weights...")
    orbax_moe_idx = MOE_LAYER_IDX - cls.args.num_dense_layers
    load_orbax_weights(
        cls.model,
        cls.ckpt_path,
        MOE_LAYER_IDX,
        cls.args.orbax_layer_prefix,
        cls.args.enable_shared_expert,
        orbax_moe_idx=orbax_moe_idx,
    )

    print(f"  Running MaxText MoE forward (input={cls.mg_input.shape})...")
    cls.mx_data = run_maxtext_forward(cls.model, cls.mg_input, cls.args)

  def test_moe_output(self):
    """MoE output cosine similarity and relative L2 within thresholds."""
    metrics = self._compare_pair(self.mx_data["moe_output"], self.mg_output)
    print(
        f"\n  moe_output: cosine={metrics['cosine']:.6f} "
        f"rel_l2={metrics['rel_l2']:.3e} max_abs={metrics['max_abs']:.3e}"
    )
    self.assertGreater(
        metrics["cosine"], MOE_FWD_MIN_COSINE,
        f"Cosine similarity {metrics['cosine']:.6f} < {MOE_FWD_MIN_COSINE}"
    )
    self.assertLess(
        metrics["rel_l2"], MOE_FWD_MAX_REL_L2,
        f"Relative L2 {metrics['rel_l2']:.3e} >= {MOE_FWD_MAX_REL_L2}"
    )

  def test_shared_expert_output(self):
    """Shared expert output matches if present in both frameworks."""
    mx_shared = self.mx_data.get("shared_expert_output")
    mg_shared = self.mg_data.get("shared_expert_output")
    if mx_shared is None or mg_shared is None:
      self.skipTest("Shared expert output not available on both sides")

    metrics = self._compare_pair(mx_shared, mg_shared)
    print(f"\n  shared_expert_output: cosine={metrics['cosine']:.6f} " f"rel_l2={metrics['rel_l2']:.3e}")
    self.assertGreater(
        metrics["cosine"], MOE_FWD_MIN_COSINE, f"Shared expert cosine {metrics['cosine']:.6f} < {MOE_FWD_MIN_COSINE}"
    )


if __name__ == "__main__":
  unittest.main()
