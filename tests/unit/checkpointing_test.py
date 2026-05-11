# Copyright 2023–2025 Google LLC
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

"""Unit tests for checkpointing utilities."""

import tempfile

from etils import epath

from maxtext.common.checkpointing import (
    JsonCheckpointHandler,
    load_trainer_state,
    should_save_by_samples,
)


class TestShouldSaveBySamples:
  """Tests for should_save_by_samples()."""

  def test_disabled_when_save_samples_is_zero(self):
    assert should_save_by_samples(1000, 0, 64) is False

  def test_disabled_when_save_samples_is_negative(self):
    assert should_save_by_samples(1000, -100, 64) is False

  def test_exact_multiple(self):
    # consumed_samples is exactly a multiple of save_samples
    assert should_save_by_samples(10000, 5000, 64) is True

  def test_remainder_within_half_batch(self):
    # remainder = 10, half_batch = 32 -> 10 <= 32 -> True
    assert should_save_by_samples(5010, 5000, 64) is True

  def test_close_to_next_multiple(self):
    # save_samples - remainder = 5000 - 4980 = 20, half_batch = 32 -> 20 <= 32 -> True
    assert should_save_by_samples(4980, 5000, 64) is True

  def test_far_from_boundary(self):
    # remainder = 2500, half_batch = 32 -> 2500 > 32 and (5000 - 2500) = 2500 > 32 -> False
    assert should_save_by_samples(2500, 5000, 64) is False

  def test_zero_consumed_samples(self):
    assert should_save_by_samples(0, 5000, 64) is False

  def test_large_batch_relative_to_save_period(self):
    # global_batch_size = 4096, save_samples = 5000
    # remainder = 3000, half_batch = 2048 -> 3000 > 2048
    # save_samples - remainder = 2000, half_batch = 2048 -> 2000 <= 2048 -> True
    assert should_save_by_samples(3000, 5000, 4096) is True

  def test_boundary_exactly_half_batch(self):
    # remainder = 32 (== half_batch), tie breaks to pre-boundary side -> False
    assert should_save_by_samples(5032, 5000, 64) is False

  def test_half_batch_tie_saves_on_pre_boundary_side_only(self):
    assert should_save_by_samples(4990, 5000, 20) is True
    assert should_save_by_samples(5010, 5000, 20) is False


class TestTrainerStatePersistence:
  """Tests for load_trainer_state() with Orbax item layout."""

  def test_load_from_orbax_item_layout(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      trainer_state = {
          "consumed_train_tokens": 24998051840,
          "consumed_train_samples": 6103040,
          "global_step": 1192,
      }
      item_dir = epath.Path(tmpdir) / "1192" / "trainer_state"
      JsonCheckpointHandler().save(item_dir, item=trainer_state)

      loaded = load_trainer_state(tmpdir, 1192)
      assert loaded == trainer_state

  def test_load_nonexistent_returns_none(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      loaded = load_trainer_state(tmpdir, 999)
      assert loaded is None
