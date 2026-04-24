# Copyright 2023-2026 Google LLC
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

"""Tests for Ling3 decoder layer wiring.

Validates that:
- decoders.Decoder.get_decoder_layers() dispatches LING3 to Ling3 Linen wrappers
  in both unscan (2-tuple) and scan (3-tuple, with MoE last for MTP) modes
- get_norm_layer returns rms_norm for LING3
- KDA-position layer constructs KimiDeltaAttention, MLA-position uses MLA
- Ling3ScannableBlock bundles the correct KDA/MLA attention distribution
- The two-phase scan helper exists and the unscan_prefix boundary assertion holds
- multi_token_prediction.py wires layer_idx for LING3
- decoders.py unscan + scan branches include LING3
"""

import os
import unittest

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs.pyconfig import initialize_pydantic
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

_BASE_CONFIG_PATH = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")


class Ling3DecoderDispatchTest(unittest.TestCase):
  """Tests for decoders.Decoder dispatch on LING3."""

  @classmethod
  def setUpClass(cls):
    cls.cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "model_name=ling3-tiny", "scan_layers=False"])

  def _build_decoder(self, cfg):
    # Local import — avoid heavyweight imports at module load time.
    from maxtext.layers import decoders  # pylint: disable=import-outside-toplevel

    # We don't need a real mesh for get_decoder_layers / get_norm_layer; pass None.
    # Decoder is a Linen module; we instantiate it but never call it, so setup() is not run.
    return decoders.Decoder(config=cfg, mesh=None)

  def test_ling3_in_decoder_layers_dispatch_unscan(self):
    """LING3 + scan_layers=False returns [Ling3DenseDecoderLayerToLinen, Ling3MoEDecoderLayerToLinen]."""
    from maxtext.models import ling3  # pylint: disable=import-outside-toplevel

    decoder = self._build_decoder(self.cfg)
    layers = decoder.get_decoder_layers()
    self.assertEqual(len(layers), 2)
    self.assertIs(layers[0], ling3.Ling3DenseDecoderLayerToLinen)
    self.assertIs(layers[1], ling3.Ling3MoEDecoderLayerToLinen)

  def test_ling3_in_decoder_layers_dispatch_scan(self):
    """LING3 + scan_layers=True returns [Dense, ScannableBlock, MoE].

    The MoE wrapper MUST be last so MTP picks it up via `layer_types[-1]`
    instead of the heterogeneous ScannableBlock (RFC-0012 §4.5).
    """
    from maxtext.models import ling3  # pylint: disable=import-outside-toplevel

    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "model_name=ling3-tiny", "scan_layers=True"])
    decoder = self._build_decoder(cfg)
    layers = decoder.get_decoder_layers()
    self.assertEqual(len(layers), 3)
    self.assertIs(layers[0], ling3.Ling3DenseDecoderLayerToLinen)
    self.assertIs(layers[1], ling3.Ling3ScannableBlockToLinen)
    self.assertIs(layers[-1], ling3.Ling3MoEDecoderLayerToLinen)

  def test_ling3_norm_layer_is_rms(self):
    """LING3 uses rms_norm (matches Ling2 / DeepSeek family)."""
    from maxtext.layers.normalizations import rms_norm  # pylint: disable=import-outside-toplevel

    decoder = self._build_decoder(self.cfg)
    norm_partial = decoder.get_norm_layer(num_features=self.cfg.emb_dim)
    # functools.partial wraps rms_norm
    self.assertIs(norm_partial.func, rms_norm)


class Ling3LayerConstructionTest(unittest.TestCase):
  """Tests construction of individual Ling3 decoder layers."""

  @classmethod
  def setUpClass(cls):
    cls.cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "model_name=ling3-tiny", "scan_layers=False"])

  def test_ling3_kda_position_constructs(self):
    """KDA-position layer (layer_idx=0) constructs KimiDeltaAttention."""
    from flax import nnx  # pylint: disable=import-outside-toplevel
    import jax  # pylint: disable=import-outside-toplevel
    from maxtext.layers import attention_kda  # pylint: disable=import-outside-toplevel
    from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
    from maxtext.models import ling3  # pylint: disable=import-outside-toplevel

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

    # layer_idx=0 with cycle interval 4 → KDA position (only idx % 4 == 3 is MLA)
    layer = ling3.Ling3MoEDecoderLayer(
        config=self.cfg,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        layer_idx=0,
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
    )
    self.assertIsInstance(layer.attention, attention_kda.KimiDeltaAttention)

  def test_ling3_mla_position_constructs(self):
    """MLA-position layer (last in cycle) constructs MLA attention."""
    from flax import nnx  # pylint: disable=import-outside-toplevel
    import jax  # pylint: disable=import-outside-toplevel
    from maxtext.layers import attention_mla  # pylint: disable=import-outside-toplevel
    from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
    from maxtext.models import ling3  # pylint: disable=import-outside-toplevel

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

    # layer_idx=3 with cycle interval 4 → MLA position ((3+1) % 4 == 0)
    layer = ling3.Ling3MoEDecoderLayer(
        config=self.cfg,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        layer_idx=3,
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
    )
    self.assertIsInstance(layer.attention, attention_mla.MLA)


class Ling3ScannableBlockTest(unittest.TestCase):
  """Tests for Ling3ScannableBlock structure.

  The block bundles `inhomogeneous_layer_cycle_interval` MoE layers; in the
  Ling3 cycle the first interval-1 are KDA positions and the last is MLA.
  Uses the real ling3-tiny interval=4 config.
  """

  @classmethod
  def setUpClass(cls):
    cls.cfg = initialize_pydantic(
        [
            "",
            _BASE_CONFIG_PATH,
            "model_name=ling3-tiny",
            "scan_layers=True",
        ]
    )

  def _build_block(self):
    from flax import nnx  # pylint: disable=import-outside-toplevel
    import jax  # pylint: disable=import-outside-toplevel
    from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
    from maxtext.models import ling3  # pylint: disable=import-outside-toplevel

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)
    return ling3.Ling3ScannableBlock(
        config=self.cfg,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
    )

  def test_ling3_scannable_block_layer_count(self):
    """Block holds exactly `inhomogeneous_layer_cycle_interval` sub-layers (named layers_{i})."""
    block = self._build_block()
    interval = self.cfg.inhomogeneous_layer_cycle_interval
    for i in range(interval):
      self.assertTrue(hasattr(block, f"layers_{i}"), f"missing layers_{i}")
    self.assertFalse(hasattr(block, f"layers_{interval}"), "block holds extra sub-layer")

  def test_ling3_scannable_block_attention_types(self):
    """Layers 0..interval-2 use KDA, layer interval-1 uses MLA."""
    from maxtext.layers import attention_kda  # pylint: disable=import-outside-toplevel
    from maxtext.layers import attention_mla  # pylint: disable=import-outside-toplevel

    block = self._build_block()
    interval = self.cfg.inhomogeneous_layer_cycle_interval
    for i in range(interval - 1):
      layer = getattr(block, f"layers_{i}")
      self.assertIsInstance(
          layer.attention,
          attention_kda.KimiDeltaAttention,
          f"layers_{i} should be KDA but got {type(layer.attention).__name__}",
      )
    last = getattr(block, f"layers_{interval - 1}")
    self.assertIsInstance(last.attention, attention_mla.MLA)


class Ling3SourceLevelTest(unittest.TestCase):
  """Source-level checks that don't require module construction.

  Mirror the pattern from tests/unit/ling3_config_test.py:test_ling3_in_layer_map.
  """

  def test_ling3_in_decoders_unscan_branch(self):
    """decoders.py unscan dense+MoE branch tuple includes LING3."""
    decoders_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "layers", "decoders.py")
    with open(decoders_path, "r", encoding="utf-8") as f:
      source = f.read()
    # The tuple in the unscan branch
    self.assertIn(
        "(DecoderBlockType.DEEPSEEK, DecoderBlockType.LING2, DecoderBlockType.LING3)",
        source,
    )
    # The inner global_layer_idx wiring
    self.assertIn(
        "if cfg.decoder_block in (DecoderBlockType.LING2, DecoderBlockType.LING3):",
        source,
    )

  def test_ling3_in_decoders_dispatch(self):
    """decoders.py get_decoder_layers has a LING3 case wired to ling3 Linen wrappers."""
    decoders_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "layers", "decoders.py")
    with open(decoders_path, "r", encoding="utf-8") as f:
      source = f.read()
    self.assertIn("case DecoderBlockType.LING3:", source)
    self.assertIn("ling3.Ling3DenseDecoderLayerToLinen", source)
    self.assertIn("ling3.Ling3MoEDecoderLayerToLinen", source)
    self.assertIn("ling3.Ling3ScannableBlockToLinen", source)

  def test_ling3_in_decoders_scan_helper(self):
    """decoders.py defines _apply_ling3_scan_layers and routes LING3 scan to it."""
    decoders_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "layers", "decoders.py")
    with open(decoders_path, "r", encoding="utf-8") as f:
      source = f.read()
    self.assertIn("def _apply_ling3_scan_layers(", source)
    self.assertIn("self._apply_ling3_scan_layers(", source)

  def test_ling3_in_mtp_layer_idx_path(self):
    """multi_token_prediction.py wires layer_idx for LING3 (same path as LING2)."""
    mtp_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "layers", "multi_token_prediction.py")
    with open(mtp_path, "r", encoding="utf-8") as f:
      source = f.read()
    self.assertIn(
        "if cfg.decoder_block in (DecoderBlockType.LING2, DecoderBlockType.LING3):",
        source,
    )


class Ling3ScanLayerMathTest(unittest.TestCase):
  """Boundary-math tests for `_apply_ling3_scan_layers`.

  These exercise the unscan_prefix / num_moe_prefix / scan_length computation
  and the assert that the scan region divides evenly by `interval`. They use
  a stand-in `Decoder` with `scan_decoder_layers` and the layer constructors
  patched out so no real Flax modules need to be built.
  """

  def _run_helper(self, *, num_decoder_layers, first_num_dense_layers, interval):
    """Drive `_apply_ling3_scan_layers` and capture the layer names it constructs.

    Returns a list of (`prefix`, layer_idx) tuples for Phase 1, plus the
    Phase 2 `scan_length` (or None if Phase 2 was skipped).
    """
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel
    from unittest import mock  # pylint: disable=import-outside-toplevel
    from maxtext.layers import decoders  # pylint: disable=import-outside-toplevel

    cfg = SimpleNamespace(
        num_decoder_layers=num_decoder_layers,
        first_num_dense_layers=first_num_dense_layers,
        inhomogeneous_layer_cycle_interval=interval,
    )

    captured_phase1 = []
    captured_phase2 = {"scan_length": None}

    def fake_layer_factory(*, name, **_kwargs):
      captured_phase1.append(name)

      def call(*_args, **_kwargs2):
        return None, None

      return call

    def fake_scan(_cfg, _block, scan_length, _name, *_args, **_kwargs):
      captured_phase2["scan_length"] = scan_length

      def call(*_call_args, **_call_kwargs):
        return None, None

      return call

    fake_self = mock.Mock(spec=["config", "mesh", "model_mode", "quant", "scan_decoder_layers"])
    fake_self.config = cfg
    fake_self.mesh = None
    fake_self.model_mode = "train"
    fake_self.quant = None
    fake_self.scan_decoder_layers.side_effect = fake_scan

    remat_layers = (fake_layer_factory, mock.sentinel.scannable, fake_layer_factory)
    # pylint: disable=protected-access
    decoders.Decoder._apply_ling3_scan_layers(fake_self, remat_layers, y=None, broadcast_args=())
    return captured_phase1, captured_phase2["scan_length"]

  def test_no_dense_prefix(self):
    """first_num_dense_layers=0 → unscan_prefix=0, entire model in scan."""
    phase1, scan_length = self._run_helper(num_decoder_layers=8, first_num_dense_layers=0, interval=4)
    self.assertEqual(phase1, [])
    self.assertEqual(scan_length, 2)

  def test_dense_prefix_aligned(self):
    """first_num_dense_layers divisible by interval → no MoE transition."""
    phase1, scan_length = self._run_helper(num_decoder_layers=12, first_num_dense_layers=4, interval=4)
    self.assertEqual(
        phase1,
        ["dense_layers_0", "dense_layers_1", "dense_layers_2", "dense_layers_3"],
    )
    self.assertEqual(scan_length, 2)

  def test_dense_prefix_needs_moe_transition(self):
    """first_num_dense_layers=1, interval=4 → 1 dense + 3 MoE transition layers."""
    phase1, scan_length = self._run_helper(num_decoder_layers=12, first_num_dense_layers=1, interval=4)
    self.assertEqual(
        phase1,
        ["dense_layers_0", "moe_layers_0", "moe_layers_1", "moe_layers_2"],
    )
    self.assertEqual(scan_length, 2)  # (12 - 4) / 4 = 2

  def test_unscan_prefix_covers_all_layers_raises(self):
    """unscan_prefix == num_decoder_layers trips the assert (no MoE region left)."""
    with self.assertRaises(AssertionError) as ctx:
      self._run_helper(num_decoder_layers=4, first_num_dense_layers=4, interval=4)
    self.assertIn("unscan_prefix", str(ctx.exception))

  def test_scan_region_not_divisible_raises(self):
    """If num_decoder_layers - unscan_prefix is not a multiple of interval, assert trips."""
    # num=10, dense=1, interval=4 → unscan_prefix=4, scan_region=6, 6%4 != 0
    with self.assertRaises(AssertionError) as ctx:
      self._run_helper(num_decoder_layers=10, first_num_dense_layers=1, interval=4)
    self.assertIn("divisible", str(ctx.exception))


if __name__ == "__main__":
  unittest.main()
