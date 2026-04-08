"""Mock verification tests for MLA interleaved RoPE de-interleave logic.

Tests three scenarios:
1. Yarn + interleave=True + mla_interleaved_rope=True: external de-interleave skipped
2. mla_interleaved_rope=False: no de-interleave at all
3. Non-Yarn RoPE + mla_interleaved_rope=True: external de-interleave applied

Uses numpy only (no JAX dependency) to verify the branching logic.
"""

import unittest

import numpy as np


def _deinterleave(arr):
  """De-interleave: [x0,y0,x1,y1,...] -> [x0,x1,...,y0,y1,...]"""
  return np.concatenate([arr[..., 0::2], arr[..., 1::2]], axis=-1)


class YarnRoPEStub:
  """Stub for YarnRotaryEmbedding."""

  _is_yarn = True

  def __init__(self, interleave=True):
    self.interleave = interleave
    self.last_input = None

  def __call__(self, inputs, position=None):
    self.last_input = inputs.copy()
    return inputs


class DefaultRoPEStub:
  """Stub for RotaryEmbedding (default, no interleave handling)."""

  _is_yarn = False

  def __init__(self):
    self.last_input = None

  def __call__(self, inputs, position=None):
    self.last_input = inputs.copy()
    return inputs


def apply_rotary_embedding_logic(
    inputs,
    rope,
    attention_type,
    mla_interleaved_rope,
    partial_rotary_factor,
    qk_rope_head_dim,
):
  """Replicate the apply_rotary_embedding logic from attentions.py.

  This mirrors the exact branching structure in the actual code.
  """
  is_mla = attention_type == "mla"

  def _should_deinterleave():
    return is_mla and mla_interleaved_rope and not (getattr(rope, "_is_yarn", False) and rope.interleave)

  if is_mla and partial_rotary_factor < 1.0:
    rotary_dim = int(qk_rope_head_dim * partial_rotary_factor)
    inputs_rot = inputs[..., :rotary_dim]
    inputs_pass = inputs[..., rotary_dim:]

    if _should_deinterleave():
      inputs_rot = _deinterleave(inputs_rot)

    inputs_rot = rope(inputs_rot, position=None)
    return np.concatenate([inputs_rot, inputs_pass], axis=-1)
  else:
    if _should_deinterleave():
      inputs = _deinterleave(inputs)
    return rope(inputs, position=None)


class TestMlaInterleavedRope(unittest.TestCase):
  """Test the MLA interleaved RoPE de-interleave logic."""

  def _make_input(self, dim=8):
    """Interleaved input: [0,1,2,3,4,5,6,7] = [x0,y0,x1,y1,x2,y2,x3,y3]."""
    return np.arange(dim, dtype=np.float32).reshape(1, 1, 1, dim)

  # --- Scenario 1: Yarn interleave=True, mla_interleaved_rope=True ---

  def test_yarn_interleave_true_skips_external_deinterleave(self):
    """Yarn internally handles interleave -> no external de-interleave."""
    rope = YarnRoPEStub(interleave=True)
    inputs = self._make_input()

    apply_rotary_embedding_logic(
        inputs,
        rope,
        "mla",
        mla_interleaved_rope=True,
        partial_rotary_factor=1.0,
        qk_rope_head_dim=8,
    )

    # Input passed to Yarn as-is (no de-interleave)
    np.testing.assert_array_equal(rope.last_input, inputs)

  # --- Scenario 2: mla_interleaved_rope=False ---

  def test_mla_interleaved_rope_false_no_deinterleave(self):
    """No de-interleave when mla_interleaved_rope=False."""
    rope = DefaultRoPEStub()
    inputs = self._make_input()

    apply_rotary_embedding_logic(
        inputs,
        rope,
        "mla",
        mla_interleaved_rope=False,
        partial_rotary_factor=1.0,
        qk_rope_head_dim=8,
    )

    np.testing.assert_array_equal(rope.last_input, inputs)

  # --- Scenario 3: Non-Yarn + mla_interleaved_rope=True ---

  def test_non_yarn_mla_interleaved_deinterleaves(self):
    """Non-Yarn RoPE + mla_interleaved_rope=True: external de-interleave."""
    rope = DefaultRoPEStub()
    inputs = self._make_input()  # [0,1,2,3,4,5,6,7]

    apply_rotary_embedding_logic(
        inputs,
        rope,
        "mla",
        mla_interleaved_rope=True,
        partial_rotary_factor=1.0,
        qk_rope_head_dim=8,
    )

    # Expected: [0,2,4,6, 1,3,5,7]
    expected = _deinterleave(inputs)
    np.testing.assert_array_equal(rope.last_input, expected)

  # --- Scenario 3b: Partial rotary + non-Yarn + mla_interleaved_rope=True ---

  def test_partial_rotary_non_yarn_deinterleaves_rot_portion(self):
    """Partial rotary: de-interleave only the rotary portion."""
    rope = DefaultRoPEStub()
    inputs = self._make_input()  # [0,1,2,3,4,5,6,7]

    result = apply_rotary_embedding_logic(
        inputs,
        rope,
        "mla",
        mla_interleaved_rope=True,
        partial_rotary_factor=0.5,
        qk_rope_head_dim=8,
    )

    # rotary_dim = 4, so first 4 dims get rotated
    # inputs_rot = [0,1,2,3] -> de-interleave -> [0,2,1,3]
    expected_rot = np.array([[[[0.0, 2.0, 1.0, 3.0]]]])
    np.testing.assert_array_equal(rope.last_input, expected_rot)

    # Pass-through portion [4,5,6,7] preserved
    np.testing.assert_array_equal(result[..., 4:], np.array([[[[4.0, 5.0, 6.0, 7.0]]]]))

  # --- Scenario 4: Yarn interleave=False + mla_interleaved_rope=True ---

  def test_yarn_interleave_false_deinterleaves(self):
    """Yarn with interleave=False: should apply external de-interleave."""
    rope = YarnRoPEStub(interleave=False)
    inputs = self._make_input()

    apply_rotary_embedding_logic(
        inputs,
        rope,
        "mla",
        mla_interleaved_rope=True,
        partial_rotary_factor=1.0,
        qk_rope_head_dim=8,
    )

    expected = _deinterleave(inputs)
    np.testing.assert_array_equal(rope.last_input, expected)

  # --- Scenario 5: Non-MLA attention type ---

  def test_non_mla_no_deinterleave(self):
    """Non-MLA attention type: no de-interleave regardless of config."""
    rope = DefaultRoPEStub()
    inputs = self._make_input()

    apply_rotary_embedding_logic(
        inputs,
        rope,
        "dot_product",
        mla_interleaved_rope=True,
        partial_rotary_factor=1.0,
        qk_rope_head_dim=8,
    )

    np.testing.assert_array_equal(rope.last_input, inputs)


if __name__ == "__main__":
  unittest.main(verbosity=2)
