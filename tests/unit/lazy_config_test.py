"""Tests for LazyDataset config parsing.

Validates that:
- LazyDataset fields are accessible on the main config object
- Default values are correct and backward-compatible
- CLI overrides work for lazy_* fields
- Invalid values for Literal fields are rejected
- Existing models are not affected by new lazy fields
"""

import os
import unittest

from pydantic import ValidationError

from maxtext.configs.pyconfig import initialize_pydantic
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

_BASE_CONFIG_PATH = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")


class LazyConfigDefaultsTest(unittest.TestCase):
  """Tests that LazyDataset fields have correct defaults on base config."""

  @classmethod
  def setUpClass(cls):
    cls.cfg = initialize_pydantic(["", _BASE_CONFIG_PATH])

  def test_lazy_train_files_default(self):
    self.assertEqual(self.cfg.lazy_train_files, "")

  def test_lazy_valid_files_default(self):
    self.assertEqual(self.cfg.lazy_valid_files, "")

  def test_lazy_test_files_default(self):
    self.assertEqual(self.cfg.lazy_test_files, "")

  def test_lazy_data_root_default(self):
    self.assertEqual(self.cfg.lazy_data_root, "")

  def test_lazy_split_default(self):
    self.assertEqual(self.cfg.lazy_split, "")

  def test_lazy_data_type_default(self):
    self.assertEqual(self.cfg.lazy_data_type, "text")

  def test_lazy_loader_mode_default(self):
    self.assertEqual(self.cfg.lazy_loader_mode, "sliding_window")

  def test_lazy_eos_token_id_default(self):
    self.assertEqual(self.cfg.lazy_eos_token_id, 2)

  def test_lazy_cls_token_id_default(self):
    self.assertEqual(self.cfg.lazy_cls_token_id, -1)

  def test_lazy_add_cls_default(self):
    self.assertFalse(self.cfg.lazy_add_cls)

  def test_lazy_drop_last_default(self):
    self.assertTrue(self.cfg.lazy_drop_last)

  def test_lazy_index_mapping_path_default(self):
    self.assertEqual(self.cfg.lazy_index_mapping_path, "")

  def test_lazy_bin_index_path_default(self):
    self.assertEqual(self.cfg.lazy_bin_index_path, "")

  def test_lazy_loader_online_shuffle_default(self):
    self.assertFalse(self.cfg.lazy_loader_online_shuffle)

  def test_lazy_loader_seed_default(self):
    self.assertEqual(self.cfg.lazy_loader_seed, 1234)

  def test_lazy_blend_shuffle_seed_default(self):
    self.assertEqual(self.cfg.lazy_blend_shuffle_seed, -1)

  def test_lazy_blend_shuffle_only_dataset_default(self):
    self.assertFalse(self.cfg.lazy_blend_shuffle_only_dataset)

  def test_lazy_dataset_weight_mode_default(self):
    self.assertEqual(self.cfg.lazy_dataset_weight_mode, "ratio")

  def test_lazy_data_size_B_tokens_default(self):
    self.assertEqual(self.cfg.lazy_data_size_B_tokens, 0)

  def test_lazy_loader_scatter_default(self):
    self.assertEqual(self.cfg.lazy_loader_scatter, -1)

  def test_lazy_blend_cache_dir_default(self):
    self.assertEqual(self.cfg.lazy_blend_cache_dir, "")

  def test_lazy_bfd_pack_default(self):
    self.assertEqual(self.cfg.lazy_bfd_pack, "")

  def test_lazy_no_attnmask_data_default(self):
    self.assertEqual(self.cfg.lazy_no_attnmask_data, "")

  def test_lazy_bfd_pack_sort_by_lens_default(self):
    self.assertFalse(self.cfg.lazy_bfd_pack_sort_by_lens)

  def test_lazy_pack_divisible_by_default(self):
    self.assertEqual(self.cfg.lazy_pack_divisible_by, -1)


class LazyConfigCLIOverrideTest(unittest.TestCase):
  """Tests that lazy_* fields can be overridden via CLI args."""

  def test_override_lazy_train_files(self):
    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "lazy_train_files=0.7 ds1 0.3 ds2"])
    self.assertEqual(cfg.lazy_train_files, "0.7 ds1 0.3 ds2")

  def test_override_lazy_loader_mode_pack(self):
    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "lazy_loader_mode=pack"])
    self.assertEqual(cfg.lazy_loader_mode, "pack")

  def test_override_lazy_loader_mode_map(self):
    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "lazy_loader_mode=map"])
    self.assertEqual(cfg.lazy_loader_mode, "map")

  def test_override_lazy_eos_token_id(self):
    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "lazy_eos_token_id=100"])
    self.assertEqual(cfg.lazy_eos_token_id, 100)

  def test_override_lazy_loader_scatter(self):
    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "lazy_loader_scatter=-8"])
    self.assertEqual(cfg.lazy_loader_scatter, -8)

  def test_override_lazy_drop_last_false(self):
    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "lazy_drop_last=false"])
    self.assertFalse(cfg.lazy_drop_last)

  def test_override_lazy_loader_online_shuffle(self):
    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "lazy_loader_online_shuffle=true"])
    self.assertTrue(cfg.lazy_loader_online_shuffle)

  def test_override_lazy_dataset_weight_mode_epoch(self):
    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "lazy_dataset_weight_mode=epoch"])
    self.assertEqual(cfg.lazy_dataset_weight_mode, "epoch")

  def test_override_lazy_bfd_pack(self):
    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "lazy_bfd_pack=ds1,ds2"])
    self.assertEqual(cfg.lazy_bfd_pack, "ds1,ds2")

  def test_override_lazy_no_attnmask_data(self):
    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "lazy_no_attnmask_data=ds_a,ds_b"])
    self.assertEqual(cfg.lazy_no_attnmask_data, "ds_a,ds_b")

  def test_override_lazy_bfd_pack_all(self):
    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "lazy_bfd_pack=ALL"])
    self.assertEqual(cfg.lazy_bfd_pack, "ALL")

  def test_override_lazy_split(self):
    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "lazy_split=0.98,0.01,0.01"])
    self.assertEqual(cfg.lazy_split, "0.98,0.01,0.01")


class LazyConfigValidationTest(unittest.TestCase):
  """Tests that invalid lazy config values are rejected."""

  def test_invalid_lazy_loader_mode_rejected(self):
    with self.assertRaises((ValidationError, SystemExit)):
      initialize_pydantic(["", _BASE_CONFIG_PATH, "lazy_loader_mode=invalid_mode"])

  def test_invalid_lazy_dataset_weight_mode_rejected(self):
    with self.assertRaises((ValidationError, SystemExit)):
      initialize_pydantic(["", _BASE_CONFIG_PATH, "lazy_dataset_weight_mode=invalid"])


class LazyConfigBackwardCompatTest(unittest.TestCase):
  """Tests that existing models are not affected by lazy fields."""

  @classmethod
  def setUpClass(cls):
    cls.cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "model_name=llama2-7b"])

  def test_existing_model_has_lazy_defaults(self):
    """Lazy fields should exist with safe defaults on any model config."""
    cfg = self.cfg
    self.assertEqual(cfg.lazy_train_files, "")
    self.assertEqual(cfg.lazy_loader_mode, "sliding_window")
    self.assertEqual(cfg.lazy_loader_scatter, -1)
    self.assertFalse(cfg.lazy_loader_online_shuffle)
    self.assertEqual(cfg.lazy_bfd_pack, "")
    self.assertEqual(cfg.lazy_no_attnmask_data, "")

  def test_mmap_fields_unaffected(self):
    """MMapDataset fields (reset_attention_mask etc.) remain accessible."""
    cfg = self.cfg
    self.assertTrue(cfg.reset_attention_mask)
    self.assertFalse(cfg.eod_mask_loss)


if __name__ == "__main__":
  unittest.main()
