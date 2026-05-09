# Copyright 2026 Google LLC
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

"""Unit tests for _inject_loop_variant_dep (GA weight hoist prevention).

Tests verify that the injection function:
1. Is numerically identity across dtypes (bf16, fp32, int32)
2. Preserves pytree structure and shapes
3. Handles edge cases (scalars, nested pytrees)
"""

import unittest

import jax.numpy as jnp
import numpy as np

from maxtext.utils.gradient_accumulation import _inject_loop_variant_dep


class InjectLoopVariantDepIdentityTest(unittest.TestCase):
  """T1: Verify _inject_loop_variant_dep is numerically identity."""

  def _make_data(self, dtype=jnp.float32):
    """Create a minimal data pytree (simulating microbatch input)."""
    return {"input_ids": jnp.ones((4, 8), dtype=dtype)}

  def test_identity_bf16(self):
    params = {"w": jnp.arange(12, dtype=jnp.bfloat16).reshape(3, 4)}
    data = self._make_data(jnp.bfloat16)
    result = _inject_loop_variant_dep(params, data)
    np.testing.assert_array_equal(result["w"], params["w"])

  def test_identity_fp32(self):
    params = {"w": jnp.arange(12, dtype=jnp.float32).reshape(3, 4)}
    data = self._make_data(jnp.float32)
    result = _inject_loop_variant_dep(params, data)
    np.testing.assert_array_equal(result["w"], params["w"])

  def test_identity_int32(self):
    params = {"w": jnp.arange(12, dtype=jnp.int32).reshape(3, 4)}
    data = self._make_data(jnp.float32)
    result = _inject_loop_variant_dep(params, data)
    np.testing.assert_array_equal(result["w"], params["w"])

  def test_identity_nested_pytree(self):
    params = {
        "layer_0": {
            "kernel": jnp.ones((4, 4), dtype=jnp.float32),
            "bias": jnp.zeros((4,), dtype=jnp.float32),
        },
        "layer_1": {
            "kernel": jnp.ones((4, 2), dtype=jnp.bfloat16) * 3.0,
        },
    }
    data = self._make_data()
    result = _inject_loop_variant_dep(params, data)
    for key, value in params.items():
      for sub_key, sub_value in value.items():
        np.testing.assert_array_equal(
            result[key][sub_key],
            sub_value,
            err_msg=f"Mismatch at {key}/{sub_key}",
        )

  def test_identity_scalar_param(self):
    params = {"scale": jnp.float32(2.5)}
    data = self._make_data()
    result = _inject_loop_variant_dep(params, data)
    np.testing.assert_array_equal(result["scale"], params["scale"])

  def test_preserves_shapes(self):
    shapes = [(1,), (4, 4), (2, 3, 5), (1, 1, 1, 1)]
    params = {f"p{i}": jnp.ones(s, dtype=jnp.float32) for i, s in enumerate(shapes)}
    data = self._make_data()
    result = _inject_loop_variant_dep(params, data)
    for key in params:
      self.assertEqual(result[key].shape, params[key].shape, f"Shape mismatch for {key}")

  def test_preserves_dtypes(self):
    params = {
        "f32": jnp.ones((4,), dtype=jnp.float32),
        "bf16": jnp.ones((4,), dtype=jnp.bfloat16),
        "i32": jnp.ones((4,), dtype=jnp.int32),
    }
    data = self._make_data()
    result = _inject_loop_variant_dep(params, data)
    for key, value in params.items():
      self.assertEqual(result[key].dtype, value.dtype, f"Dtype mismatch for {key}")

  def test_different_data_values_same_output(self):
    """Verify identity holds regardless of the data token value."""
    params = {"w": jnp.arange(8, dtype=jnp.float32)}
    data_a = {"input_ids": jnp.ones((4, 8), dtype=jnp.float32) * 42.0}
    data_b = {"input_ids": jnp.ones((4, 8), dtype=jnp.float32) * -7.0}
    result_a = _inject_loop_variant_dep(params, data_a)
    result_b = _inject_loop_variant_dep(params, data_b)
    np.testing.assert_array_equal(result_a["w"], params["w"])
    np.testing.assert_array_equal(result_b["w"], params["w"])


if __name__ == "__main__":
  unittest.main()
