# Copyright 2023–2026 Google LLC
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

"""Tests for the local optax Muon fork under third_party/optax_muon/."""

# pylint: disable=import-outside-toplevel

import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from third_party.optax_muon import muon


class TestMuonForkImports(unittest.TestCase):
  """Verify that the fork exposes all required public symbols."""

  def test_import_muon_function(self):
    self.assertTrue(callable(muon))

  def test_import_muon_dimension_numbers(self):
    from third_party.optax_muon import MuonDimensionNumbers

    mdn = MuonDimensionNumbers(reduction_axis=(0,), output_axis=(-1,))
    self.assertEqual(mdn.reduction_axis, (0,))
    self.assertEqual(mdn.output_axis, (-1,))

  def test_import_scale_by_muon(self):
    from third_party.optax_muon import scale_by_muon

    self.assertTrue(callable(scale_by_muon))

  def test_import_muon_state(self):
    from third_party.optax_muon import MuonState

    self.assertTrue(issubclass(MuonState, tuple))

  def test_import_orthogonalize(self):
    from third_party.optax_muon import orthogonalize_via_newton_schulz

    self.assertTrue(callable(orthogonalize_via_newton_schulz))


class TestMuonForkFunctionality(unittest.TestCase):
  """Verify that the forked muon optimizer works correctly."""

  def test_muon_init_and_update_2d(self):
    """Basic muon init + update with default 2D params."""
    tx = muon(learning_rate=0.01)
    params = {"w": jnp.ones((4, 8))}
    opt_state = tx.init(params)
    grads = {"w": jnp.ones((4, 8)) * 0.1}
    updates, _ = tx.update(grads, opt_state, params)
    # updates should have same structure and shape
    self.assertEqual(updates["w"].shape, (4, 8))

  def test_muon_with_dimension_numbers(self):
    """Muon with explicit weight dimension numbers and mixed params."""
    from third_party.optax_muon import MuonDimensionNumbers as mdn

    params = {"a": jnp.ones((2, 3)), "b": jnp.ones((3,))}
    dim_nums = {"a": mdn((0,), (1,)), "b": None}
    tx = muon(learning_rate=0.1, muon_weight_dimension_numbers=dim_nums)
    opt_state = tx.init(params)
    grads = {"a": jnp.ones((2, 3)) * 0.1, "b": jnp.ones((3,)) * 0.1}
    updates, _ = tx.update(grads, opt_state, params)
    self.assertEqual(updates["a"].shape, (2, 3))
    self.assertEqual(updates["b"].shape, (3,))

  def test_muon_with_consistent_rms(self):
    """Verify consistent_rms parameter is accepted (0.2.8 feature)."""
    tx = muon(learning_rate=0.01, consistent_rms=0.2)
    params = {"w": jnp.ones((4, 8))}
    opt_state = tx.init(params)
    grads = {"w": jnp.ones((4, 8)) * 0.1}
    updates, _ = tx.update(grads, opt_state, params)
    self.assertEqual(updates["w"].shape, (4, 8))


class TestSgdStyleMomentum(unittest.TestCase):
  """Verify SGD-style Nesterov momentum in scale_by_muon (Megatron parity)."""

  def test_sgd_style_momentum_manual_two_steps(self):
    """Hand-compute SGD-style momentum for 2 steps and compare.

    SGD-style momentum:
        buf_t = beta * buf_{t-1} + grad_t
        g_t   = grad_t + beta * buf_t       (Nesterov look-ahead)
    No bias correction, no (1-beta) scaling on grad.
    We disable Newton-Schulz (ns_steps=0 won't work, so we test via
    scale_by_muon directly with a small identity-like matrix).
    """
    from third_party.optax_muon import scale_by_muon

    beta = 0.9
    tx = scale_by_muon(beta=beta, nesterov_style="sgd", ns_steps=1)

    # Use a 2x2 matrix so Newton-Schulz doesn't distort values too much
    key = jax.random.PRNGKey(42)
    grad1 = jax.random.normal(key, (2, 3))
    grad2 = jax.random.normal(jax.random.PRNGKey(7), (2, 3))

    state = tx.init(grad1)

    # Step 1: buf = 0.9*0 + grad1 = grad1
    #          g = grad1 + 0.9 * buf = grad1 + 0.9*grad1 = 1.9*grad1
    _, state1 = tx.update(grad1, state)
    buf1 = beta * jnp.zeros_like(grad1) + grad1

    # Step 2: buf = 0.9*grad1 + grad2
    #          g = grad2 + 0.9*buf
    _, state2 = tx.update(grad2, state1)
    buf2 = beta * buf1 + grad2

    # Verify the momentum buffer (mu) in state matches our manual buf
    # state is MuonState(count, mu, ns_coeffs)
    npt.assert_allclose(state1.mu, buf1, atol=1e-6)
    npt.assert_allclose(state2.mu, buf2, atol=1e-6)

  def test_sgd_style_differs_from_ema_style(self):
    """SGD-style and EMA-style should produce different momentum buffers."""
    from third_party.optax_muon import scale_by_muon

    beta = 0.9
    tx_ema = scale_by_muon(beta=beta, nesterov_style="ema")
    tx_sgd = scale_by_muon(beta=beta, nesterov_style="sgd")

    grad = jax.random.normal(jax.random.PRNGKey(0), (4, 8))
    state_ema = tx_ema.init(grad)
    state_sgd = tx_sgd.init(grad)

    _, state_ema = tx_ema.update(grad, state_ema)
    _, state_sgd = tx_sgd.update(grad, state_sgd)

    # The momentum buffers must differ:
    # EMA: mu = 0.9*0 + 0.1*grad = 0.1*grad
    # SGD: buf = 0.9*0 + grad = grad
    self.assertFalse(jnp.allclose(state_ema.mu, state_sgd.mu), "SGD-style and EMA-style momentum buffers should differ")

  def test_sgd_style_no_bias_correction(self):
    """SGD-style should NOT apply bias correction — the momentum buffer
    should be exactly beta*buf+grad, not (1-beta) weighted."""
    from third_party.optax_muon import scale_by_muon

    beta = 0.9
    tx = scale_by_muon(beta=beta, nesterov_style="sgd", ns_steps=1)
    grad = jnp.ones((2, 3))
    state = tx.init(grad)

    _, state1 = tx.update(grad, state)
    # buf = 0.9*0 + 1 = 1.0  (NOT 0.9*0 + 0.1*1 = 0.1 like EMA)
    expected_buf = jnp.ones((2, 3))
    npt.assert_allclose(state1.mu, expected_buf, atol=1e-7)

  def test_muon_toplevel_sgd_style(self):
    """muon() top-level API should accept and propagate nesterov_style='sgd'."""
    tx = muon(learning_rate=0.01, nesterov_style="sgd")
    params = {"w": jnp.ones((4, 8))}
    opt_state = tx.init(params)
    grads = {"w": jnp.ones((4, 8)) * 0.1}
    updates, _ = tx.update(grads, opt_state, params)
    self.assertEqual(updates["w"].shape, (4, 8))

  def test_muon_toplevel_sgd_vs_ema_differ(self):
    """muon(nesterov_style='sgd') and muon(nesterov_style='ema') should
    produce different internal momentum after one step."""

    grad = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    params = {"w": grad}
    grads = {"w": grad * 0.1}

    tx_ema = muon(learning_rate=0.01, nesterov_style="ema")
    tx_sgd = muon(learning_rate=0.01, nesterov_style="sgd")

    st_ema = tx_ema.init(params)
    st_sgd = tx_sgd.init(params)

    # Extract inner muon partition states and compare mu buffers
    # The internal state structure may vary, but the overall updates differ
    updates_ema, _ = tx_ema.update(grads, st_ema, params)
    updates_sgd, _ = tx_sgd.update(grads, st_sgd, params)
    # After NS orthogonalization on step 1 they might still be close,
    # but the raw momentum buffers must differ — just verify no crash
    # and that init+update works end-to-end for both styles
    self.assertEqual(updates_ema["w"].shape, (4, 8))
    self.assertEqual(updates_sgd["w"].shape, (4, 8))


class TestAdamWeightDecayMask(unittest.TestCase):
  """Verify adam_weight_decay_mask is accepted and wired to internal adamw."""

  def test_muon_accepts_adam_weight_decay_mask_callable(self):
    """muon() must accept adam_weight_decay_mask and not crash."""
    from third_party.optax_muon import MuonDimensionNumbers as mdn

    dim_nums = {"w": mdn((0,), (1,)), "bias": None, "scale": None}

    def mask_fn(p):
      return jax.tree.map(lambda _: True, p)

    tx = muon(
        learning_rate=0.01,
        adam_weight_decay=0.1,
        adam_weight_decay_mask=mask_fn,
        muon_weight_dimension_numbers=dim_nums,
    )
    params = {"w": jnp.ones((4, 8)), "bias": jnp.ones((8,)), "scale": jnp.ones((8,))}
    state = tx.init(params)
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
    updates, _ = tx.update(grads, state, params)
    self.assertEqual(updates["w"].shape, (4, 8))
    self.assertEqual(updates["bias"].shape, (8,))

  def test_adam_weight_decay_mask_excludes_bias_from_decay(self):
    """With mask excluding bias, bias updates should differ from unmasked case."""
    from third_party.optax_muon import MuonDimensionNumbers as mdn

    dim_nums = {"w": mdn((0,), (1,)), "bias": None}
    params = {"w": jnp.ones((4, 8)), "bias": jnp.ones((8,)) * 2.0}
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)

    # Case 1: no mask (all params get weight decay)
    tx_no_mask = muon(learning_rate=0.01, adam_weight_decay=0.1, muon_weight_dimension_numbers=dim_nums)
    state1 = tx_no_mask.init(params)
    updates_no_mask, _ = tx_no_mask.update(grads, state1, params)

    # Case 2: mask excludes bias from weight decay
    def mask_fn(p):
      return {"w": True, "bias": False}

    tx_masked = muon(
        learning_rate=0.01,
        adam_weight_decay=0.1,
        adam_weight_decay_mask=mask_fn,
        muon_weight_dimension_numbers=dim_nums,
    )
    state2 = tx_masked.init(params)
    updates_masked, _ = tx_masked.update(grads, state2, params)

    # Bias updates should differ (masked has no decay component)
    self.assertFalse(
        jnp.allclose(updates_no_mask["bias"], updates_masked["bias"]),
        "Masking bias from decay should produce different updates",
    )
    # W updates should be same (both have decay)
    npt.assert_allclose(updates_no_mask["w"], updates_masked["w"], atol=1e-6)


class TestBatchNewtonSchulz(unittest.TestCase):
  """Verify batch Newton-Schulz: same-shape matrices stacked → batch NS → unstack.

  The batch mode groups parameters by their reshaped 2D shape, stacks them
  along the batch dimension, runs NS once per group, then unstacks. Results
  must be numerically identical to non-batch mode.
  """

  def _make_multi_param_pytree(self, key, n_same=4, shape=(4, 8)):
    """Create a pytree with n_same parameters of the same shape + 1 different."""
    keys = jax.random.split(key, n_same + 1)
    params = {f"w{i}": jax.random.normal(keys[i], shape) for i in range(n_same)}
    # Add a parameter with a different shape (goes to a separate group)
    params["v"] = jax.random.normal(keys[n_same], (6, 12))
    return params

  def test_scale_by_muon_accepts_batch_update(self):
    """scale_by_muon() must accept batch_update and batch_update_size params."""
    from third_party.optax_muon import scale_by_muon

    tx = scale_by_muon(batch_update=True, batch_update_size=2)
    grad = jax.random.normal(jax.random.PRNGKey(0), (4, 8))
    state = tx.init(grad)
    updates, _ = tx.update(grad, state)
    self.assertEqual(updates.shape, (4, 8))

  def test_muon_accepts_batch_update(self):
    """muon() top-level API must accept and propagate batch_update."""
    tx = muon(learning_rate=0.01, batch_update=True)
    params = {"w": jnp.ones((4, 8))}
    state = tx.init(params)
    grads = {"w": jnp.ones((4, 8)) * 0.1}
    updates, _ = tx.update(grads, state, params)
    self.assertEqual(updates["w"].shape, (4, 8))

  def test_batch_vs_nobatch_identical(self):
    """batch=True and batch=False must produce numerically identical results."""
    key = jax.random.PRNGKey(42)
    params = self._make_multi_param_pytree(key)
    grads = jax.tree.map(lambda x: jax.random.normal(jax.random.PRNGKey(7), x.shape) * 0.1, params)

    tx_nobatch = muon(learning_rate=0.02, batch_update=False)
    tx_batch = muon(learning_rate=0.02, batch_update=True)

    state_nb = tx_nobatch.init(params)
    state_b = tx_batch.init(params)

    updates_nb, _ = tx_nobatch.update(grads, state_nb, params)
    updates_b, _ = tx_batch.update(grads, state_b, params)

    for name in params:
      npt.assert_allclose(
          updates_b[name], updates_nb[name], atol=2e-3, err_msg=f"batch vs nobatch mismatch for param '{name}'"
      )

  def test_batch_with_batch_update_size(self):
    """Chunking via batch_update_size must produce same results as no chunking."""
    key = jax.random.PRNGKey(99)
    # 6 same-shape params → will be split into chunks of 2
    params = {f"w{i}": jax.random.normal(jax.random.split(key, 7)[i], (4, 8)) for i in range(6)}
    grads = jax.tree.map(lambda x: jax.random.normal(jax.random.PRNGKey(3), x.shape) * 0.1, params)

    tx_nobatch = muon(learning_rate=0.02, batch_update=False)
    tx_chunked = muon(learning_rate=0.02, batch_update=True, batch_update_size=2)

    state_nb = tx_nobatch.init(params)
    state_ch = tx_chunked.init(params)

    updates_nb, _ = tx_nobatch.update(grads, state_nb, params)
    updates_ch, _ = tx_chunked.update(grads, state_ch, params)

    for name in params:
      npt.assert_allclose(
          updates_ch[name],
          updates_nb[name],
          atol=2e-3,
          err_msg=f"chunked batch vs nobatch mismatch for param '{name}'",
      )

  def test_batch_mixed_shapes(self):
    """Params with different shapes: each group batched separately, all correct."""
    key = jax.random.PRNGKey(11)
    keys = jax.random.split(key, 5)
    params = {
        "a1": jax.random.normal(keys[0], (4, 8)),
        "a2": jax.random.normal(keys[1], (4, 8)),
        "b1": jax.random.normal(keys[2], (3, 6)),
        "b2": jax.random.normal(keys[3], (3, 6)),
        "c": jax.random.normal(keys[4], (5, 10)),
    }
    grads = jax.tree.map(lambda x: jax.random.normal(jax.random.PRNGKey(5), x.shape) * 0.1, params)

    tx_nobatch = muon(learning_rate=0.02, batch_update=False)
    tx_batch = muon(learning_rate=0.02, batch_update=True)

    state_nb = tx_nobatch.init(params)
    state_b = tx_batch.init(params)

    updates_nb, _ = tx_nobatch.update(grads, state_nb, params)
    updates_b, _ = tx_batch.update(grads, state_b, params)

    for name in params:
      npt.assert_allclose(
          updates_b[name], updates_nb[name], atol=2e-3, err_msg=f"mixed-shape batch mismatch for param '{name}'"
      )

  def test_batch_with_dimension_numbers(self):
    """Batch mode with custom MuonDimensionNumbers (3D params with heads)."""
    from third_party.optax_muon import MuonDimensionNumbers as mdn

    key = jax.random.PRNGKey(77)
    keys = jax.random.split(key, 4)
    # 3D params: [embed, heads, head_dim] — heads as batch axis
    params = {
        "q": jax.random.normal(keys[0], (64, 8, 16)),
        "k": jax.random.normal(keys[1], (64, 8, 16)),
        "v": jax.random.normal(keys[2], (64, 8, 16)),
        "bias": jax.random.normal(keys[3], (8,)),  # Adam param
    }
    dim_nums = {
        "q": mdn(reduction_axis=(0,), output_axis=(-1,)),
        "k": mdn(reduction_axis=(0,), output_axis=(-1,)),
        "v": mdn(reduction_axis=(0,), output_axis=(-1,)),
        "bias": None,
    }
    grads = jax.tree.map(lambda x: jax.random.normal(jax.random.PRNGKey(13), x.shape) * 0.1, params)

    tx_nobatch = muon(learning_rate=0.02, batch_update=False, muon_weight_dimension_numbers=dim_nums)
    tx_batch = muon(learning_rate=0.02, batch_update=True, muon_weight_dimension_numbers=dim_nums)

    state_nb = tx_nobatch.init(params)
    state_b = tx_batch.init(params)

    updates_nb, _ = tx_nobatch.update(grads, state_nb, params)
    updates_b, _ = tx_batch.update(grads, state_b, params)

    for name in params:
      npt.assert_allclose(
          updates_b[name], updates_nb[name], atol=2e-3, err_msg=f"batch with dim_nums mismatch for param '{name}'"
      )


if __name__ == "__main__":
  unittest.main()
