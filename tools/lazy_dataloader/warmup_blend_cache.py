#!/usr/bin/env python3
"""Pre-populate the lazy blending index cache from a dataset YAML.

The lazy dataloader's ``build_blending_indices()`` is O(size) in pure
Python and can take minutes for 500+ datasets.  This tool runs the
computation offline so that production TPU jobs hit the cache directly.

Usage:
  python tools/lazy_dataloader/warmup_blend_cache.py \
      --datasets-yaml scripts/datasets/ant_datasets_dev.yml

  # Override cache dir or scatter from YAML defaults
  python tools/lazy_dataloader/warmup_blend_cache.py \
      --datasets-yaml scripts/datasets/ant_datasets_dev.yml \
      --cache-dir /tmp/blend_cache --scatter -1

  # Only warm specific scatter groups
  python tools/lazy_dataloader/warmup_blend_cache.py \
      --datasets-yaml scripts/datasets/ant_datasets_dev.yml \
      --scatter-groups 0,2

  # Dry run: show what would be computed without building
  python tools/lazy_dataloader/warmup_blend_cache.py \
      --datasets-yaml scripts/datasets/ant_datasets_dev.yml --dry-run
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

print("[warmup] importing maxtext modules ...", flush=True)
from maxtext.input_pipeline._lazy_blending import (
    _blend_cache_key,
    _save_blend_cache,
    _try_load_blend_cache,
)
from maxtext.input_pipeline._megatron_blending import build_blending_indices
from maxtext.input_pipeline.lazy_data_processing import (
    _build_source,
    _parse_datasets_and_weights,
)

print("[warmup] imports done", flush=True)

# jax/absl hijacks the root logger during import — force it back to INFO
# so that library-level logs (e.g. build_blending_indices progress) are visible.
logging.root.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.root.addHandler(_handler)

logger = logging.getLogger(__name__)

_YAML_CONFIG_KEYS = {
    "lazy_loader_mode": "sliding_window",
    "lazy_dataset_weight_mode": "epoch",
    "lazy_loader_scatter": -1,
    "lazy_loader_seed": 1234,
    "lazy_loader_online_shuffle": False,
    "lazy_blend_shuffle_seed": -1,
    "lazy_blend_shuffle_only_dataset": False,
    "lazy_blend_cache_dir": "",
    "lazy_drop_last": True,
    "lazy_bfd_pack": "",
    "lazy_bfd_pack_sort_by_lens": False,
    "lazy_pack_divisible_by": -1,
    "lazy_data_type": "text",
    "lazy_data_root": "",
    "lazy_eos_token_id": 2,
    "lazy_cls_token_id": -1,
    "lazy_add_cls": False,
    "lazy_index_mapping_path": "",
    "lazy_bin_index_path": "",
    "lazy_data_size_B_tokens": 0.0,
    "max_target_length": 8192,
    "reset_attention_mask": False,
}


def _parse_yaml_config(yaml_path: str) -> dict:
  """Parse dataset YAML and merge with default config keys."""
  with open(yaml_path, encoding="utf-8") as f:
    raw = yaml.safe_load(f)
  cfg = dict(_YAML_CONFIG_KEYS)
  for k in _YAML_CONFIG_KEYS:
    if k in raw:
      cfg[k] = raw[k]
  if "lazy_train_files" in raw:
    cfg["lazy_train_files"] = raw["lazy_train_files"]
  else:
    raise ValueError(f"YAML must contain 'lazy_train_files': {yaml_path}")
  return cfg


def _resolve_scatter_path(cfg: dict, path: str) -> str:
  data_root = cfg["lazy_data_root"]
  if data_root:
    return f"{data_root}/{path}.scatter"
  return path


def _get_source_lengths(cfg: dict, paths: list[str], scatter_id: int) -> list[int]:
  """Build sources one at a time to get lengths, freeing each after measuring."""
  config = SimpleNamespace(**cfg)
  lengths = []
  for i, path in enumerate(paths):
    scatter_path = _resolve_scatter_path(cfg, path)
    src = _build_source(config, scatter_path, stage="train", process_index=scatter_id)
    lengths.append(len(src))
    if hasattr(src, "close"):
      src.close()
    del src
    if (i + 1) % 50 == 0:
      gc.collect()
      logger.info("  measured %d/%d dataset lengths ...", i + 1, len(paths))
  gc.collect()
  return lengths


def _compute_blend_params(cfg: dict, source_lengths: list[int], weights: list[float]):
  """Compute normalized weights and total blend size from config."""
  weight_mode = cfg["lazy_dataset_weight_mode"]
  if weight_mode == "epoch":
    raw_weights = [w * sl for w, sl in zip(weights, source_lengths)]
    total = sum(raw_weights)
    size = int(total)
    norm_weights = np.array([w / size for w in raw_weights], dtype=np.float64)
  elif weight_mode == "ratio":
    norm_weights = np.array(weights, dtype=np.float64)
    norm_weights = norm_weights / np.sum(norm_weights)
    scatter = max(abs(cfg["lazy_loader_scatter"]), 1)
    size = int(math.ceil(cfg["lazy_data_size_B_tokens"] * 1e9 / cfg["max_target_length"] / scatter))
  else:
    raise ValueError(f"Unknown lazy_dataset_weight_mode: {weight_mode}")
  if size <= 0:
    raise ValueError(f"Computed blend size is {size}, must be positive")
  return norm_weights, size


def warmup_one_group(
    cfg: dict,
    paths: list[str],
    weights: list[float],
    scatter_id: int,
    cache_dir: str,
    force: bool = False,
    dry_run: bool = False,
):
  """Build and cache blending indices for a single scatter group."""
  shuffle_seed = cfg["lazy_blend_shuffle_seed"]
  shuffle_only_dataset = cfg["lazy_blend_shuffle_only_dataset"]

  logger.info("--- scatter_group=%d: measuring source lengths ...", scatter_id)
  t0 = time.perf_counter()
  source_lengths = _get_source_lengths(cfg, paths, scatter_id)
  t_src = time.perf_counter() - t0
  logger.info("  %d dataset lengths measured in %.1fs", len(source_lengths), t_src)

  norm_weights, size = _compute_blend_params(cfg, source_lengths, weights)

  cache_key = _blend_cache_key(
      norm_weights,
      size,
      len(weights),
      shuffle_seed,
      shuffle_only_dataset,
      scatter_id=scatter_id,
      source_lengths=source_lengths,
  )
  di_path = os.path.join(cache_dir, f"{cache_key}.dataset_index.npy")

  logger.info(
      "  scatter_group=%d: size=%s, num_datasets=%d, cache_key=%s", scatter_id, f"{size:,}", len(weights), cache_key
  )

  if dry_run:
    exists = os.path.isfile(di_path)
    logger.info("  [dry-run] cache %s: %s", "EXISTS" if exists else "MISSING", di_path)
    return

  if not force:
    cached = _try_load_blend_cache(cache_dir, cache_key, size)
    if cached is not None:
      logger.info("  cache HIT — skipping (use --force to recompute)")
      return

  logger.info("  computing blend indices (size=%s) ...", f"{size:,}")
  t0 = time.perf_counter()
  dataset_index = np.zeros(size, dtype=np.int16)
  dataset_sample_index = np.zeros(size, dtype=np.int64)
  build_blending_indices(
      dataset_index=dataset_index,
      dataset_sample_index=dataset_sample_index,
      weights=norm_weights,
      num_datasets=len(weights),
      size=size,
  )
  t_blend = time.perf_counter() - t0
  logger.info("  blend indices built in %.1fs", t_blend)

  if shuffle_seed > 0:
    logger.info("  shuffling (seed=%d, only_dataset=%s) ...", shuffle_seed, shuffle_only_dataset)
    rng = np.random.RandomState(shuffle_seed)
    inds = np.arange(size, dtype=np.int64)
    rng.shuffle(inds)
    dataset_index = dataset_index[inds]
    if shuffle_only_dataset:
      for i in range(len(weights)):
        mask = dataset_index == i
        dataset_sample_index[mask] = np.arange(mask.sum())
    else:
      dataset_sample_index = dataset_sample_index[inds]
    del inds

  _save_blend_cache(cache_dir, cache_key, dataset_index, dataset_sample_index)
  logger.info("  saved: %s", di_path)


def main():
  parser = argparse.ArgumentParser(
      description="Pre-populate lazy blending index cache from a dataset YAML",
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("--datasets-yaml", required=True, help="Dataset YAML file path")
  parser.add_argument("--cache-dir", default=None, help="Override lazy_blend_cache_dir from YAML")
  parser.add_argument("--scatter", type=int, default=None, help="Override lazy_loader_scatter from YAML")
  parser.add_argument(
      "--scatter-groups",
      default=None,
      help="Comma-separated scatter group IDs to warm (default: all)",
  )
  parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
  parser.add_argument("--dry-run", action="store_true", help="Show what would be computed, skip build")
  args = parser.parse_args()

  cfg = _parse_yaml_config(args.datasets_yaml)

  if args.scatter is not None:
    cfg["lazy_loader_scatter"] = args.scatter
  if args.cache_dir is not None:
    cfg["lazy_blend_cache_dir"] = args.cache_dir

  cache_dir = cfg["lazy_blend_cache_dir"]
  if not cache_dir:
    parser.error("No cache dir: set lazy_blend_cache_dir in YAML or use --cache-dir")

  paths, weights = _parse_datasets_and_weights(cfg["lazy_train_files"])
  if not paths:
    parser.error("No datasets found in lazy_train_files")
  if weights is None:
    weights = [1.0] * len(paths)

  abs_scatter = max(abs(cfg["lazy_loader_scatter"]), 1)
  if args.scatter_groups is not None:
    groups = [int(g) for g in args.scatter_groups.split(",")]
    for g in groups:
      if g < 0 or g >= abs_scatter:
        parser.error(f"scatter_group {g} out of range [0, {abs_scatter})")
  else:
    groups = list(range(abs_scatter))

  logger.info("=== Blend Cache Warmup ===")
  logger.info("  YAML:       %s", args.datasets_yaml)
  logger.info("  cache_dir:  %s", cache_dir)
  logger.info("  scatter:    %d (groups: %s)", cfg["lazy_loader_scatter"], groups)
  logger.info("  datasets:   %d", len(paths))
  logger.info("  force:      %s", args.force)
  logger.info("  dry_run:    %s", args.dry_run)
  logger.info("==========================")

  t_total = time.perf_counter()
  for scatter_id in groups:
    warmup_one_group(
        cfg,
        paths,
        weights,
        scatter_id=scatter_id,
        cache_dir=cache_dir,
        force=args.force,
        dry_run=args.dry_run,
    )
  elapsed = time.perf_counter() - t_total
  logger.info("Done. Total time: %.1fs", elapsed)


if __name__ == "__main__":
  main()
