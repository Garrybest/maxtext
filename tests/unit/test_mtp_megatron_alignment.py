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

"""Alignment tests between Megatron-LM (PyTorch) and MaxText (JAX) for MTP.

Validates numerical equivalence of Multi-Token Prediction forward pass,
backward pass (gradients), and optimizer updates between the two frameworks.

Requires:
  - PyTorch >= 2.4.0 (for torch.nn.RMSNorm)
  - megatron-core importable (add Megatron-LM to PYTHONPATH)
  - JAX with CPU backend

Run:
  python -m pytest tests/unit/test_mtp_megatron_alignment.py -v
"""

# pylint: disable=import-outside-toplevel,protected-access

import os

# Force JAX to use CPU backend for deterministic IEEE float32 alignment.
os.environ["JAX_PLATFORMS"] = "cpu"

import unittest
from functools import reduce
from operator import mul
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Framework imports are deferred to setUp so tests can be collected even if
# a dependency is missing.  The module-level flag lets us skip gracefully.
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
_MEGATRON_AVAILABLE = False
_JAX_AVAILABLE = False

try:
  import torch
  import torch.nn.functional as F

  _TORCH_AVAILABLE = True
except ImportError:
  pass

try:
  import megatron  # noqa: F401  # pylint: disable=unused-import

  _MEGATRON_AVAILABLE = True
except ImportError:
  pass

try:
  import jax
  import jax.numpy as jnp
  from flax import nnx

  _JAX_AVAILABLE = True
except ImportError:
  pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 128
NUM_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 32
FFN_HIDDEN_SIZE = 512
VOCAB_SIZE = 256
SEQ_LEN = 16
BATCH_SIZE = 2
MTP_NUM_LAYERS = 1
LAYERNORM_EPS = 1e-5
SEED = 42

# Tolerances for float32 CPU (both PyTorch and JAX on CPU)
ATOL_NORM = 1e-6
RTOL_NORM = 1e-6
ATOL_FWD = 1e-5
RTOL_FWD = 1e-5
ATOL_FWD_COMPOUND = 5e-3  # compound ops (multi-matmul + nonlinearity)
RTOL_FWD_COMPOUND = 2e-3
ATOL_GRAD = 1e-4
RTOL_GRAD = 1e-4
ATOL_LOSS = 1e-4
RTOL_LOSS = 1e-4
ATOL_OPT = 1e-4
RTOL_OPT = 1e-4

# ---------------------------------------------------------------------------
# Runtime feature detection — MaxText branches differ in MTP support
# ---------------------------------------------------------------------------
_MTP_FINAL_LAYERNORM: bool | None = None  # set by _detect_mtp_features()


def _detect_mtp_features() -> None:
  """Detect which MTP features the current MaxText installation supports."""
  global _MTP_FINAL_LAYERNORM  # pylint: disable=global-statement
  if _MTP_FINAL_LAYERNORM is not None:
    return
  try:
    import inspect

    from maxtext.layers.multi_token_prediction import MultiTokenPredictionLayer

    src = inspect.getsource(MultiTokenPredictionLayer.__init__)
    _MTP_FINAL_LAYERNORM = "final_layernorm" in src
  except Exception:  # pylint: disable=broad-except
    _MTP_FINAL_LAYERNORM = False


def _require_all():
  """Skip the test if any framework is missing."""
  if not _TORCH_AVAILABLE:
    raise unittest.SkipTest("PyTorch not available")
  if not _MEGATRON_AVAILABLE:
    raise unittest.SkipTest("megatron-core not available")
  if not _JAX_AVAILABLE:
    raise unittest.SkipTest("JAX not available")


# ========================================================================
# Megatron CPU helpers
# ========================================================================


class MegatronCPUSetup:
  """Singleton that initialises Megatron parallel-state for CPU testing."""

  _initialised = False

  @classmethod
  def initialise(cls) -> None:
    """Set up single-process gloo backend and TP=1 / PP=1."""
    if cls._initialised:
      return

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    if not torch.distributed.is_initialized():
      torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

    from megatron.core import parallel_state as ps

    if not ps.model_parallel_is_initialized():
      ps.initialize_model_parallel(
          tensor_model_parallel_size=1,
          pipeline_model_parallel_size=1,
      )

    cls._patch_cuda_deps()
    cls._initialised = True

  @classmethod
  def _patch_cuda_deps(cls) -> None:
    """Monkey-patch Megatron utilities that call cuda APIs."""
    from megatron.core import utils as mcore_utils

    # GlobalMemoryBuffer — allocate on CPU instead of cuda
    def _cpu_get_tensor(self: Any, shape: tuple, dtype: torch.dtype, name: str) -> torch.Tensor:
      required = reduce(mul, shape, 1)
      key = (name, dtype)
      if self.buffer.get(key) is None or self.buffer[key].numel() < required:
        self.buffer[key] = torch.empty(required, dtype=dtype, device="cpu", requires_grad=False)
      return self.buffer[key][:required].view(*shape)

    mcore_utils.GlobalMemoryBuffer.get_tensor = _cpu_get_tensor

    # dump_tensor — no-op on CPU
    try:
      import megatron.training.argus_dump as argus

      argus.dump_tensor = lambda *a, **kw: None
    except (ImportError, AttributeError):
      pass

    # model_parallel_cuda_manual_seed — fall back to CPU manual_seed
    # Patch CUDA RNG tracker with a CPU-compatible dummy.
    try:
      from contextlib import nullcontext
      from megatron.core.tensor_parallel import random as tp_random

      def _cpu_manual_seed(seed):
        torch.manual_seed(seed)

      tp_random.model_parallel_cuda_manual_seed = _cpu_manual_seed

      class _CPURNGTracker:
        """Minimal RNG tracker that avoids CUDA calls."""

        def __init__(self):
          self.states_ = {}

        def add(self, name, seed):
          self.states_[name] = seed

        def fork(self, name="model-parallel-rng"):
          return nullcontext()

      tp_random._CUDA_RNG_STATE_TRACKER = _CPURNGTracker()
      tp_random._CUDA_RNG_STATE_TRACKER_INITIALIZED = True
    except (ImportError, AttributeError):
      pass

    # MTPLossLoggingHelper — allocate tracker on CPU instead of CUDA
    try:
      from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

      def _cpu_save_loss(
          loss: torch.Tensor,
          layer_number: int,
          num_layers: int,
          reduce_group: Any = None,
          avg_group: Any = None,
      ) -> None:
        if layer_number is None:
          return
        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
          tracker["values"] = torch.zeros(num_layers, device="cpu")
        tracker["values"][layer_number] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

      MTPLossLoggingHelper.save_loss_to_tracker = staticmethod(_cpu_save_loss)
    except (ImportError, AttributeError):
      pass

  @classmethod
  def get_config(cls) -> "TransformerConfig":
    """Return a small TransformerConfig for CPU alignment tests."""
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_query_groups=NUM_HEADS,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        use_cpu_initialization=True,
        bf16=False,
        fp16=False,
        params_dtype=torch.float32,
        pipeline_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        normalization="RMSNorm",
        layernorm_epsilon=LAYERNORM_EPS,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        masked_softmax_fusion=False,
        persist_layer_norm=False,
        sequence_parallel=False,
        mtp_num_layers=MTP_NUM_LAYERS,
        mtp_loss_scaling_factor=0.1,
        apply_residual_connection_post_layernorm=False,
    )

  @classmethod
  def build_mtp_block(cls, config: "TransformerConfig") -> "MultiTokenPredictionBlock":
    """Create an MTP block using the local (non-TE) spec."""
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_mtp_block_spec,
    )
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock

    transformer_spec = get_gpt_layer_local_spec(normalization="RMSNorm")
    mtp_spec = get_gpt_mtp_block_spec(
        config=config,
        spec=transformer_spec,
        use_transformer_engine=False,
    )
    return MultiTokenPredictionBlock(config=config, spec=mtp_spec)


# ========================================================================
# Weight conversion helpers
# ========================================================================


def torch_to_np(t: "torch.Tensor") -> np.ndarray:
  """Detach, move to CPU, convert to float32 numpy."""
  return t.detach().cpu().float().numpy()


def np_to_jax(a: np.ndarray) -> "jnp.ndarray":
  """Numpy → JAX array."""
  return jnp.array(a)


def np_to_torch(a: np.ndarray, requires_grad: bool = False) -> "torch.Tensor":
  """Numpy → PyTorch tensor."""
  return torch.from_numpy(a.copy()).float().requires_grad_(requires_grad)


def convert_rmsnorm_weights(megatron_weight: "torch.Tensor") -> np.ndarray:
  """Megatron RMSNorm weight [H] → MaxText scale [H]. Direct copy."""
  return torch_to_np(megatron_weight)


def convert_linear_weights(megatron_weight: "torch.Tensor") -> np.ndarray:
  """Megatron Linear weight [out, in] → MaxText kernel [in, out]. Transpose."""
  return torch_to_np(megatron_weight).T


def convert_qkv_weights(
    fused_weight: "torch.Tensor",
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Megatron fused QKV weight → MaxText separate Q, K, V kernels.

  Megatron layout: [num_kv_heads * (heads_per_group + 2) * head_dim, hidden]
  At TP=1 with MHA: [(num_heads + 2*num_kv_heads) * head_dim, hidden]

  MaxText layout: query.kernel [hidden, num_heads, head_dim]
                  key.kernel   [hidden, num_kv_heads, head_dim]
                  value.kernel [hidden, num_kv_heads, head_dim]
  """
  w = torch_to_np(fused_weight)  # [out, in]
  hidden = w.shape[1]
  heads_per_group = num_heads // num_kv_heads

  # Reshape to [num_kv_heads, (heads_per_group + 2) * head_dim, hidden]
  w = w.reshape(num_kv_heads, (heads_per_group + 2) * head_dim, hidden)

  q_part = w[:, : heads_per_group * head_dim, :]  # [ng, hpg*hd, H]
  k_part = w[:, heads_per_group * head_dim : (heads_per_group + 1) * head_dim, :]  # [ng, hd, H]
  v_part = w[:, (heads_per_group + 1) * head_dim :, :]  # [ng, hd, H]

  # → MaxText: [hidden, num_heads, head_dim]
  q_kernel = q_part.reshape(num_heads, head_dim, hidden).transpose(2, 0, 1)
  k_kernel = k_part.reshape(num_kv_heads, head_dim, hidden).transpose(2, 0, 1)
  v_kernel = v_part.reshape(num_kv_heads, head_dim, hidden).transpose(2, 0, 1)
  return q_kernel, k_kernel, v_kernel


def convert_output_proj_weights(
    megatron_weight: "torch.Tensor",
    num_heads: int,
    head_dim: int,
) -> np.ndarray:
  """Megatron output projection [hidden, num_heads*head_dim]
  → MaxText [num_heads, head_dim, hidden]."""
  w = torch_to_np(megatron_weight)  # [hidden, num_heads*head_dim]
  return w.T.reshape(num_heads, head_dim, -1)  # [num_heads, head_dim, hidden]


def convert_swiglu_fc1_weights(megatron_weight: "torch.Tensor") -> tuple[np.ndarray, np.ndarray]:
  """Megatron SwiGLU fc1 [2*ffn, hidden] → MaxText wi_0 and wi_1 [hidden, ffn]."""
  w = torch_to_np(megatron_weight)
  gate, up = np.split(w, 2, axis=0)  # each [ffn, hidden]
  return gate.T, up.T  # each [hidden, ffn]


def convert_fc2_weights(megatron_weight: "torch.Tensor") -> np.ndarray:
  """Megatron fc2 [hidden, ffn] → MaxText wo [ffn, hidden]."""
  return torch_to_np(megatron_weight).T


# ========================================================================
# MaxText setup helpers
# ========================================================================


def _build_maxtext_config(**overrides: Any) -> "Config":
  """Build a MaxText Config for alignment tests.

  Conditionally includes ``mtp_final_layernorm`` only when the current
  MaxText installation supports it, allowing the test to run on branches
  with or without that feature.
  """
  _detect_mtp_features()
  from maxtext.configs import pyconfig
  from tests.utils.test_helpers import get_decoupled_parallelism_overrides, get_test_config_path

  extra = get_decoupled_parallelism_overrides()
  extra.update(overrides)

  kwargs: dict[str, Any] = {
      "run_name": "mtp_alignment_test",
      "skip_jax_distributed_system": True,
      "per_device_batch_size": BATCH_SIZE,
      "base_emb_dim": HIDDEN_SIZE,
      "base_num_query_heads": NUM_HEADS,
      "base_num_kv_heads": NUM_HEADS,
      "base_mlp_dim": FFN_HIDDEN_SIZE,
      "vocab_size": VOCAB_SIZE,
      "max_target_length": SEQ_LEN,
      "mtp_num_layers": MTP_NUM_LAYERS,
      "mtp_loss_scaling_factor": 0.1,
      "normalization_layer_epsilon": LAYERNORM_EPS,
      "dropout_rate": 0.0,
      "attention_type": "global",
      "logits_via_embedding": False,
      "enable_dropout": False,
      "dtype": "float32",
  }
  if _MTP_FINAL_LAYERNORM:
    kwargs["mtp_final_layernorm"] = True
  kwargs.update(extra)

  return pyconfig.initialize([None, get_test_config_path()], **kwargs)


def _build_maxtext_mtp_layer(
    config: "Config",
    mesh: "jax.sharding.Mesh",
    rngs: "nnx.Rngs",
) -> "MultiTokenPredictionLayer":
  """Build a standalone MaxText MTP layer for testing."""
  from maxtext.layers.multi_token_prediction import MultiTokenPredictionLayer
  from maxtext.layers.nnx_decoders import NNXDecoderLayer

  return MultiTokenPredictionLayer(
      config=config,
      mesh=mesh,
      layer_number=1,
      transformer_layer_module=NNXDecoderLayer,
      rngs=rngs,
  )


# ========================================================================
# Shared input generators
# ========================================================================


def _make_shared_inputs() -> dict[str, np.ndarray]:
  """Generate deterministic inputs as numpy arrays for both frameworks."""
  rng = np.random.RandomState(SEED)
  return {
      "hidden_state": rng.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).astype(np.float32),
      "embedding": rng.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).astype(np.float32),
      "input_ids": rng.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).astype(np.int32),
      "target_ids": rng.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).astype(np.int32),
      "target_mask": np.ones((BATCH_SIZE, SEQ_LEN), dtype=np.float32),
      "position_ids": np.tile(np.arange(SEQ_LEN, dtype=np.int32), (BATCH_SIZE, 1)),
  }


# ========================================================================
# Alignment test decoder (lightweight mock for MTP block testing)
# ========================================================================


class _AlignmentTestDecoder:
  """Minimal decoder providing the interface expected by MultiTokenPredictionBlock.

  Uses shared_embedding(input_ids) for token lookup and a direct float32
  dot product for logit projection (bypasses Embed.attend's bfloat16 cast).
  """

  def __init__(self, config: Any):
    self.config = config
    self.model_mode = "train"

  def _apply_embedding(
      self,
      shared_embedding: Any,
      input_ids: Any,
      position_ids: Any,
      deterministic: bool,
      model_mode: str,
  ) -> Any:
    """Look up token embeddings (no position embedding)."""
    return shared_embedding(input_ids)

  def apply_output_projection(
      self,
      shared_embedding: Any,
      hidden_state: Any,
      deterministic: bool,
      model_mode: str,
  ) -> Any:
    """Project to logits via embedding transpose (no decoder_norm)."""
    embedding = shared_embedding.embedding.value  # [V, H]
    return jnp.dot(hidden_state, embedding.T)

  def apply_output_head(
      self,
      shared_embedding: Any,
      hidden_state: Any,
      deterministic: bool,
      model_mode: str,
  ) -> Any:
    """Same as apply_output_projection (no decoder_norm in test decoder)."""
    embedding = shared_embedding.embedding.value
    return jnp.dot(hidden_state, embedding.T)


# ========================================================================
# Test classes
# ========================================================================


@unittest.skipUnless(_TORCH_AVAILABLE and _JAX_AVAILABLE, "Requires both PyTorch and JAX")
class TestLevel1SubModule(unittest.TestCase):
  """Level 1: Sub-module numerical alignment (RMSNorm, Linear)."""

  def test_rmsnorm_alignment(self) -> None:
    """Same weights + same input → same output for RMSNorm."""
    np_input = np.random.RandomState(SEED).randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).astype(np.float32)
    np_scale = np.random.RandomState(SEED + 1).randn(HIDDEN_SIZE).astype(np.float32)

    # --- PyTorch ---
    pt_norm = torch.nn.RMSNorm(HIDDEN_SIZE, eps=LAYERNORM_EPS)
    with torch.no_grad():
      pt_norm.weight.copy_(torch.from_numpy(np_scale))
    pt_out = pt_norm(torch.from_numpy(np_input)).detach().numpy()

    # --- JAX ---
    from maxtext.layers.normalizations import RMSNorm

    jax_norm = RMSNorm(
        num_features=HIDDEN_SIZE,
        epsilon=LAYERNORM_EPS,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        kernel_axes=("norm",),
        rngs=nnx.Rngs(params=0),
    )
    jax_norm.scale.value = jnp.array(np_scale)
    jax_out = np.array(jax_norm(jnp.array(np_input)))

    np.testing.assert_allclose(jax_out, pt_out, rtol=RTOL_NORM, atol=ATOL_NORM)

  def test_linear_projection_alignment(self) -> None:
    """Same (transposed) weights + same input → same output for linear."""
    rng = np.random.RandomState(SEED)
    np_input = rng.randn(BATCH_SIZE, SEQ_LEN, 2 * HIDDEN_SIZE).astype(np.float32)
    np_weight = rng.randn(HIDDEN_SIZE, 2 * HIDDEN_SIZE).astype(np.float32)  # [out, in]

    # --- PyTorch ---
    pt_out = F.linear(torch.from_numpy(np_input), torch.from_numpy(np_weight)).detach().numpy()

    # --- JAX ---
    from maxtext.layers.linears import DenseGeneral

    jax_linear = DenseGeneral(
        in_features_shape=2 * HIDDEN_SIZE,
        out_features_shape=HIDDEN_SIZE,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        use_bias=False,
        kernel_axes=("concat_embed", "embed"),
        rngs=nnx.Rngs(params=0),
    )
    jax_linear.kernel.value = jnp.array(np_weight.T)  # [in, out]
    jax_out = np.array(jax_linear(jnp.array(np_input)))

    np.testing.assert_allclose(jax_out, pt_out, rtol=RTOL_FWD, atol=ATOL_FWD)

  def test_swiglu_mlp_alignment(self) -> None:
    """SwiGLU MLP: gate+up split + transpose → same output."""
    rng = np.random.RandomState(SEED)
    np_input = rng.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).astype(np.float32)
    np_fc1 = rng.randn(2 * FFN_HIDDEN_SIZE, HIDDEN_SIZE).astype(np.float32)
    np_fc2 = rng.randn(HIDDEN_SIZE, FFN_HIDDEN_SIZE).astype(np.float32)

    # --- PyTorch SwiGLU ---
    pt_input = torch.from_numpy(np_input)
    intermediate = F.linear(pt_input, torch.from_numpy(np_fc1))
    gate, up = intermediate.chunk(2, dim=-1)
    pt_out = F.linear(F.silu(gate) * up, torch.from_numpy(np_fc2)).detach().numpy()

    # --- JAX SwiGLU ---
    gate_w, up_w = np.split(np_fc1, 2, axis=0)
    jax_input = jnp.array(np_input)
    jax_gate = jax_input @ jnp.array(gate_w.T)
    jax_up = jax_input @ jnp.array(up_w.T)
    jax_hidden = jax.nn.silu(jax_gate) * jax_up
    jax_out = np.array(jax_hidden @ jnp.array(np_fc2.T))

    np.testing.assert_allclose(jax_out, pt_out, rtol=RTOL_FWD_COMPOUND, atol=ATOL_FWD_COMPOUND)


@unittest.skipUnless(
    _TORCH_AVAILABLE and _MEGATRON_AVAILABLE and _JAX_AVAILABLE,
    "Requires PyTorch, megatron-core, and JAX",
)
class TestLevel2MTPLayerForward(unittest.TestCase):
  """Level 2: MTP layer forward pass alignment.

  MaxText uses parallel attention+MLP (single pre-norm), while Megatron uses
  sequential sublayers (two norms).  To isolate the MTP-specific logic (enorm,
  hnorm, concat, projection, final_layernorm) from this architecture difference,
  we zero out all attention+MLP weights so the transformer layer reduces to an
  identity function (via residual connection) in both frameworks.
  """

  @classmethod
  def setUpClass(cls) -> None:
    _require_all()
    _detect_mtp_features()
    if not _MTP_FINAL_LAYERNORM:
      raise unittest.SkipTest("mtp_final_layernorm not available — architectures differ")
    MegatronCPUSetup.initialise()

  def test_mtp_layer_forward(self) -> None:
    """MTP-specific ops: same weights + zeroed transformer → same output."""
    from maxtext.utils import maxtext_utils

    inputs = _make_shared_inputs()
    config_mg = MegatronCPUSetup.get_config()

    # ---- Megatron MTP layer ----
    mtp_block_mg = MegatronCPUSetup.build_mtp_block(config_mg)
    mtp_layer_mg = mtp_block_mg.layers[0]
    mtp_layer_mg.eval()

    # ---- MaxText MTP layer ----
    config_mx = _build_maxtext_config()
    devices = maxtext_utils.create_device_mesh(config_mx)
    mesh = jax.sharding.Mesh(devices, config_mx.mesh_axes)
    rngs = nnx.Rngs(params=jax.random.PRNGKey(SEED))
    mtp_layer_mx = _build_maxtext_mtp_layer(config_mx, mesh, rngs)

    # ---- Convert weights: Megatron → MaxText ----
    self._inject_weights(mtp_layer_mg, mtp_layer_mx)

    # ---- Forward pass ----
    # Bypass Megatron's forward() which internally rolls input_ids and does
    # embedding lookup.  Call _proj_and_transformer_layer directly so both
    # frameworks receive the same pre-computed embedding tensor.
    hidden_np = inputs["hidden_state"]
    embed_np = inputs["embedding"]
    position_np = inputs["position_ids"]

    # Megatron: [S, B, D]
    mg_hidden = torch.from_numpy(hidden_np.transpose(1, 0, 2))  # [S, B, D]
    mg_embed = torch.from_numpy(embed_np.transpose(1, 0, 2))
    causal_mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
      mg_out = mtp_layer_mg._proj_and_transformer_layer(
          hidden_states=mg_hidden,
          decoder_input=mg_embed,
          attention_mask=causal_mask,
      )
    mg_out_bsd = mg_out.detach().numpy().transpose(1, 0, 2)  # → [B, S, D]

    # MaxText: [B, S, D]
    mx_hidden = jnp.array(hidden_np)
    mx_embed = jnp.array(embed_np)
    mx_positions = jnp.array(position_np)
    segment_ids = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)

    mx_out = mtp_layer_mx(
        prev_hidden_state=mx_hidden,
        target_token_embedding=mx_embed,
        position_ids=mx_positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode="train",
    )
    mx_out_np = np.array(mx_out)

    np.testing.assert_allclose(mx_out_np, mg_out_bsd, rtol=RTOL_FWD, atol=ATOL_FWD)

  @staticmethod
  def _inject_weights(
      mg_layer: Any,
      mx_layer: Any,
  ) -> None:
    """Copy MTP-specific weights, zero out attention+MLP for identity transformer."""
    # MTP-specific norms and projection
    mx_layer.embedding_norm.scale.value = np_to_jax(convert_rmsnorm_weights(mg_layer.enorm.weight))
    mx_layer.hidden_state_norm.scale.value = np_to_jax(convert_rmsnorm_weights(mg_layer.hnorm.weight))
    mx_layer.projection_layer.kernel.value = np_to_jax(convert_linear_weights(mg_layer.eh_proj.weight))
    if _MTP_FINAL_LAYERNORM:
      mx_layer.final_layernorm.scale.value = np_to_jax(convert_rmsnorm_weights(mg_layer.final_layernorm.weight))

    # Zero out all attention+MLP weights in both frameworks so the
    # transformer layer becomes an identity function (residual only).
    # This isolates MTP-specific logic from the architecture mismatch
    # (MaxText parallel sublayers vs Megatron sequential sublayers).
    tl_mg = mg_layer.transformer_layer
    tl_mx = mx_layer.transformer_layer

    with torch.no_grad():
      # Megatron: zero attention
      tl_mg.self_attention.linear_qkv.weight.zero_()
      tl_mg.self_attention.linear_proj.weight.zero_()
      # Megatron: zero MLP
      tl_mg.mlp.linear_fc1.weight.zero_()
      tl_mg.mlp.linear_fc2.weight.zero_()

    # MaxText: zero attention
    attn_mx = tl_mx.self_attention
    attn_mx.query.kernel.value = jnp.zeros_like(attn_mx.query.kernel.value)
    attn_mx.key.kernel.value = jnp.zeros_like(attn_mx.key.kernel.value)
    attn_mx.value.kernel.value = jnp.zeros_like(attn_mx.value.kernel.value)
    attn_mx.out.kernel.value = jnp.zeros_like(attn_mx.out.kernel.value)
    # MaxText: zero MLP
    mlp_mx = tl_mx.mlp
    mlp_mx.wi_0.kernel.value = jnp.zeros_like(mlp_mx.wi_0.kernel.value)
    mlp_mx.wi_1.kernel.value = jnp.zeros_like(mlp_mx.wi_1.kernel.value)
    mlp_mx.wo.kernel.value = jnp.zeros_like(mlp_mx.wo.kernel.value)


@unittest.skipUnless(
    _TORCH_AVAILABLE and _MEGATRON_AVAILABLE and _JAX_AVAILABLE,
    "Requires PyTorch, megatron-core, and JAX",
)
class TestLevel3MTPLayerBackward(unittest.TestCase):
  """Level 3: MTP layer gradient alignment (zeroed transformer)."""

  @classmethod
  def setUpClass(cls) -> None:
    _require_all()
    _detect_mtp_features()
    if not _MTP_FINAL_LAYERNORM:
      raise unittest.SkipTest("mtp_final_layernorm not available — architectures differ")
    MegatronCPUSetup.initialise()

  def test_mtp_layer_parameter_gradients(self) -> None:
    """Gradients of MTP-specific parameters match between frameworks."""
    from maxtext.utils import maxtext_utils

    inputs = _make_shared_inputs()
    config_mg = MegatronCPUSetup.get_config()

    # ---- Build layers ----
    mtp_block_mg = MegatronCPUSetup.build_mtp_block(config_mg)
    mtp_layer_mg = mtp_block_mg.layers[0]
    mtp_layer_mg.train()

    config_mx = _build_maxtext_config()
    devices = maxtext_utils.create_device_mesh(config_mx)
    mesh = jax.sharding.Mesh(devices, config_mx.mesh_axes)
    rngs = nnx.Rngs(params=jax.random.PRNGKey(SEED))
    mtp_layer_mx = _build_maxtext_mtp_layer(config_mx, mesh, rngs)

    # Inject weights (zeroed transformer)
    TestLevel2MTPLayerForward._inject_weights(mtp_layer_mg, mtp_layer_mx)

    hidden_np = inputs["hidden_state"]
    embed_np = inputs["embedding"]
    position_np = inputs["position_ids"]

    # ---- Megatron backward ----
    mg_hidden = torch.from_numpy(hidden_np.transpose(1, 0, 2)).requires_grad_(True)
    mg_embed = torch.from_numpy(embed_np.transpose(1, 0, 2)).requires_grad_(True)
    causal_mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    mg_out = mtp_layer_mg._proj_and_transformer_layer(
        hidden_states=mg_hidden, decoder_input=mg_embed, attention_mask=causal_mask
    )
    mg_loss = mg_out.sum()
    mg_loss.backward()

    # ---- MaxText backward ----
    # Split into differentiable params and non-differentiable state (RNG keys etc.)
    graphdef, params, rest = nnx.split(mtp_layer_mx, nnx.Param, ...)

    def _fwd(params_dict: Any, h: jnp.ndarray, e: jnp.ndarray) -> jnp.ndarray:
      layer = nnx.merge(graphdef, params_dict, rest)
      out = layer(
          prev_hidden_state=h,
          target_token_embedding=e,
          position_ids=jnp.array(position_np),
          decoder_segment_ids=jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
          deterministic=True,
          model_mode="train",
      )
      return out.sum()

    mx_grad_fn = jax.grad(_fwd, argnums=(0, 1, 2))
    mx_grads_params, mx_grad_h, _ = mx_grad_fn(params, jnp.array(hidden_np), jnp.array(embed_np))

    # Compare key parameter gradients (state keys use mangled names from properties)
    k = 1  # layer_number

    # enorm
    mg_enorm_grad = torch_to_np(mtp_layer_mg.enorm.weight.grad)
    mx_enorm_grad = np.array(mx_grads_params[f"mtp_{k}_embedding_norm"].scale.value)
    np.testing.assert_allclose(mx_enorm_grad, mg_enorm_grad, rtol=RTOL_GRAD, atol=ATOL_GRAD)

    # hnorm
    mg_hnorm_grad = torch_to_np(mtp_layer_mg.hnorm.weight.grad)
    mx_hnorm_grad = np.array(mx_grads_params[f"mtp_{k}_hidden_state_norm"].scale.value)
    np.testing.assert_allclose(mx_hnorm_grad, mg_hnorm_grad, rtol=RTOL_GRAD, atol=ATOL_GRAD)

    # projection
    mg_proj_grad = torch_to_np(mtp_layer_mg.eh_proj.weight.grad).T
    mx_proj_grad = np.array(mx_grads_params[f"mtp_{k}_projection"].kernel.value)
    np.testing.assert_allclose(mx_proj_grad, mg_proj_grad, rtol=RTOL_GRAD, atol=ATOL_GRAD)

    # final_layernorm
    mg_fln_grad = torch_to_np(mtp_layer_mg.final_layernorm.weight.grad)
    mx_fln_grad = np.array(mx_grads_params[f"mtp_{k}_final_layernorm"].scale.value)
    np.testing.assert_allclose(mx_fln_grad, mg_fln_grad, rtol=RTOL_GRAD, atol=ATOL_GRAD)

    # input gradients
    mg_grad_h_bsd = mg_hidden.grad.detach().numpy().transpose(1, 0, 2)
    mx_grad_h_np = np.array(mx_grad_h)
    np.testing.assert_allclose(mx_grad_h_np, mg_grad_h_bsd, rtol=RTOL_GRAD, atol=ATOL_GRAD)


@unittest.skipUnless(
    _TORCH_AVAILABLE and _MEGATRON_AVAILABLE and _JAX_AVAILABLE,
    "Requires PyTorch, megatron-core, and JAX",
)
class TestLevel4RollTensor(unittest.TestCase):
  """Level 4 prerequisite: roll_tensor (Megatron) vs roll_and_mask (MaxText)."""

  @classmethod
  def setUpClass(cls) -> None:
    _require_all()
    MegatronCPUSetup.initialise()

  def test_roll_alignment(self) -> None:
    """roll_tensor and roll_and_mask produce the same output."""
    from megatron.core.transformer.multi_token_prediction import roll_tensor
    from maxtext.layers.multi_token_prediction import roll_and_mask

    rng = np.random.RandomState(SEED)
    np_data = rng.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).astype(np.int64)

    # Megatron: [S, B] layout, seq_dim=-1
    # roll_tensor expects [S, B] with seq_dim, or [B, S] with seq_dim=-1
    # Actually roll_tensor works on arbitrary dims via seq_dim parameter
    pt_data = torch.from_numpy(np_data)  # [B, S]
    pt_rolled, _ = roll_tensor(pt_data, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None)
    pt_rolled_np = pt_rolled.numpy()

    # MaxText: [B, S] layout
    jax_data = jnp.array(np_data)
    jax_rolled = roll_and_mask(jax_data, shift=-1)
    jax_rolled_np = np.array(jax_rolled)

    np.testing.assert_array_equal(jax_rolled_np, pt_rolled_np)


@unittest.skipUnless(
    _TORCH_AVAILABLE and _MEGATRON_AVAILABLE and _JAX_AVAILABLE,
    "Requires PyTorch, megatron-core, and JAX",
)
class TestLevel5MTPBlockLoss(unittest.TestCase):
  """Level 5: MTP block cross-entropy loss alignment between frameworks."""

  @classmethod
  def setUpClass(cls) -> None:
    _require_all()
    _detect_mtp_features()
    if not _MTP_FINAL_LAYERNORM:
      raise unittest.SkipTest("mtp_final_layernorm not available — architectures differ")
    MegatronCPUSetup.initialise()

  def test_mtp_block_loss_forward(self) -> None:
    """MTP block produces the same per-layer loss in both frameworks.

    Loops through MTP layers individually, computing loss externally
    to be compatible with all megatron-core API versions.
    """
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, VocabParallelEmbedding
    from megatron.core.transformer.multi_token_prediction import roll_tensor
    from maxtext.common.common_types import MODEL_MODE_TRAIN
    from maxtext.layers.embeddings import Embed
    from maxtext.layers.multi_token_prediction import MultiTokenPredictionBlock
    from maxtext.layers.nnx_decoders import NNXDecoderLayer
    from maxtext.utils import maxtext_utils

    inputs = _make_shared_inputs()
    config_mg = MegatronCPUSetup.get_config()

    # ---- Build BOTH sides first ----
    # Megatron
    mtp_block_mg = MegatronCPUSetup.build_mtp_block(config_mg)
    mtp_block_mg.eval()

    embedding_mg = VocabParallelEmbedding(VOCAB_SIZE, HIDDEN_SIZE, config=config_mg, init_method=config_mg.init_method)
    output_layer_mg = ColumnParallelLinear(
        HIDDEN_SIZE,
        VOCAB_SIZE,
        config=config_mg,
        init_method=config_mg.init_method,
        bias=False,
        skip_bias_add=True,
        gather_output=True,
    )
    with torch.no_grad():
      output_layer_mg.weight.copy_(embedding_mg.weight)

    # MaxText
    config_mx = _build_maxtext_config()
    devices = maxtext_utils.create_device_mesh(config_mx)
    mesh = jax.sharding.Mesh(devices, config_mx.mesh_axes)
    rngs = nnx.Rngs(params=jax.random.PRNGKey(SEED))

    shared_embedding_mx = Embed(
        num_embeddings=VOCAB_SIZE,
        num_features=HIDDEN_SIZE,
        config=config_mx,
        mesh=mesh,
        rngs=rngs,
    )
    shared_embedding_mx.embedding.value = np_to_jax(torch_to_np(embedding_mg.weight))

    test_decoder = _AlignmentTestDecoder(config_mx)
    mtp_block_mx = MultiTokenPredictionBlock(
        config=config_mx,
        mesh=mesh,
        transformer_layer_module=NNXDecoderLayer,
        decoder=test_decoder,
        rngs=rngs,
    )

    # ---- Inject weights BEFORE any forward pass (zeroes transformer) ----
    for k in range(MTP_NUM_LAYERS):
      mg_layer = mtp_block_mg.layers[k]
      mx_layer = getattr(mtp_block_mx, f"mtp_layer_{k + 1}")
      TestLevel2MTPLayerForward._inject_weights(mg_layer, mx_layer)

    # ---- Megatron forward: loop through layers, compute loss externally ----
    class _EmbeddingWrapper:
      """Wraps VocabParallelEmbedding to return [S, B, D]."""

      def __init__(self, embed):
        self.embed = embed

      def __call__(self, input_ids: "torch.Tensor", position_ids: "torch.Tensor" = None, **kw) -> "torch.Tensor":
        return self.embed(input_ids).transpose(0, 1).contiguous()

    mg_hidden = torch.from_numpy(inputs["hidden_state"].transpose(1, 0, 2))  # [S, B, D]
    mg_input_ids = torch.from_numpy(inputs["input_ids"].astype(np.int64))
    mg_labels = torch.from_numpy(inputs["target_ids"].astype(np.int64))
    mg_loss_mask = torch.from_numpy(inputs["target_mask"])
    mg_positions = torch.from_numpy(inputs["position_ids"].astype(np.int64))
    causal_mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    mg_per_layer_loss = []
    mtp_hidden = mg_hidden
    rolled_ids = mg_input_ids
    rolled_pos = mg_positions
    rolled_labels = mg_labels
    rolled_mask = mg_loss_mask

    with torch.no_grad():
      for layer in mtp_block_mg.layers:
        mtp_hidden, rolled_ids, rolled_pos = layer(
            input_ids=rolled_ids,
            position_ids=rolled_pos,
            hidden_states=mtp_hidden,
            attention_mask=causal_mask,
            embedding=_EmbeddingWrapper(embedding_mg),
        )
        rolled_labels, _ = roll_tensor(rolled_labels, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None)
        rolled_mask, _ = roll_tensor(rolled_mask, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None)

        logits, _ = output_layer_mg(mtp_hidden)  # [S, B, V]
        labels_sb = rolled_labels.transpose(0, 1).contiguous()  # [S, B]
        xent = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            labels_sb.reshape(-1),
            reduction="none",
        ).reshape(
            labels_sb.shape
        )  # [S, B]
        xent_bs = xent.transpose(0, 1).contiguous()  # [B, S]

        layer_loss = (xent_bs * rolled_mask).sum() / rolled_mask.sum()
        mg_per_layer_loss.append(layer_loss.item())

    mg_per_layer_loss = np.array(mg_per_layer_loss)

    # ---- MaxText forward ----
    _ = mtp_block_mx(
        shared_embedding_mx,
        jnp.array(inputs["hidden_state"]),
        jnp.array(inputs["input_ids"]),
        jnp.array(inputs["target_ids"]),
        jnp.array(inputs["target_mask"]),
        position_ids=jnp.array(inputs["position_ids"]),
        decoder_segment_ids=jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
        model_mode=MODEL_MODE_TRAIN,
        deterministic=True,
    )

    state = nnx.state(mtp_block_mx)
    mx_losses = np.array(state.losses.value)  # [num_mtp_layers]
    mx_weights = np.array(state.weights.value)  # [num_mtp_layers]
    mx_per_layer_loss = mx_losses / mx_weights  # per-token avg loss per layer

    # ---- Compare ----
    np.testing.assert_allclose(mx_per_layer_loss, mg_per_layer_loss, rtol=RTOL_LOSS, atol=ATOL_LOSS)


@unittest.skipUnless(
    _TORCH_AVAILABLE and _MEGATRON_AVAILABLE and _JAX_AVAILABLE,
    "Requires PyTorch, megatron-core, and JAX",
)
class TestLevel6FullPipeline(unittest.TestCase):
  """Level 6: Full decoder_norm → MTP → output_projection pipeline alignment.

  Validates the architectural fixes by testing the complete pipeline:
  1. Apply decoder_norm / final_layernorm to raw hidden state
  2. Run MTP layer forward
  3. Apply output projection (no decoder_norm — avoids double normalization)
  4. Compute cross-entropy loss
  5. Compute gradients through the entire pipeline

  This is the integration test for Fix 1/3/5 (apply_output_projection +
  normed hidden state + mtp_final_layernorm conditional).
  """

  @classmethod
  def setUpClass(cls) -> None:
    _require_all()
    _detect_mtp_features()
    if not _MTP_FINAL_LAYERNORM:
      raise unittest.SkipTest("mtp_final_layernorm not available — architectures differ")
    MegatronCPUSetup.initialise()

  def test_normed_pipeline_forward(self) -> None:
    """decoder_norm(h) → MTP_layer → output_projection → loss matches."""
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from maxtext.layers.normalizations import RMSNorm
    from maxtext.utils import maxtext_utils

    inputs = _make_shared_inputs()
    config_mg = MegatronCPUSetup.get_config()

    # ---- Build Megatron components ----
    mtp_block_mg = MegatronCPUSetup.build_mtp_block(config_mg)
    mtp_layer_mg = mtp_block_mg.layers[0]
    mtp_layer_mg.eval()

    final_ln_mg = torch.nn.RMSNorm(HIDDEN_SIZE, eps=LAYERNORM_EPS)
    output_layer_mg = ColumnParallelLinear(
        HIDDEN_SIZE,
        VOCAB_SIZE,
        config=config_mg,
        init_method=config_mg.init_method,
        bias=False,
        skip_bias_add=True,
        gather_output=True,
    )

    # Shared numpy weights for embedding/output and decoder_norm
    rng = np.random.RandomState(SEED + 100)
    np_embedding_weight = rng.randn(VOCAB_SIZE, HIDDEN_SIZE).astype(np.float32)
    np_decoder_norm_scale = rng.randn(HIDDEN_SIZE).astype(np.float32)

    with torch.no_grad():
      output_layer_mg.weight.copy_(torch.from_numpy(np_embedding_weight))
      final_ln_mg.weight.copy_(torch.from_numpy(np_decoder_norm_scale))

    # ---- Build MaxText components ----
    config_mx = _build_maxtext_config()
    devices = maxtext_utils.create_device_mesh(config_mx)
    mesh = jax.sharding.Mesh(devices, config_mx.mesh_axes)
    rngs = nnx.Rngs(params=jax.random.PRNGKey(SEED))

    mtp_layer_mx = _build_maxtext_mtp_layer(config_mx, mesh, rngs)

    # ---- Inject weights BEFORE any forward pass (zeroes transformer on both sides) ----
    TestLevel2MTPLayerForward._inject_weights(mtp_layer_mg, mtp_layer_mx)

    decoder_norm_mx = RMSNorm(
        num_features=HIDDEN_SIZE,
        epsilon=LAYERNORM_EPS,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    decoder_norm_mx.scale.value = jnp.array(np_decoder_norm_scale)

    # ---- Megatron forward: norm → MTP layer → output_layer ----
    raw_hidden_np = inputs["hidden_state"]  # [B, S, D]
    embed_np = inputs["embedding"]

    mg_raw_hidden = torch.from_numpy(raw_hidden_np.transpose(1, 0, 2))  # [S, B, D]
    mg_embed = torch.from_numpy(embed_np.transpose(1, 0, 2))
    causal_mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
      mg_normed = final_ln_mg(mg_raw_hidden)  # [S, B, D]
      mg_mtp_out = mtp_layer_mg._proj_and_transformer_layer(
          hidden_states=mg_normed,
          decoder_input=mg_embed,
          attention_mask=causal_mask,
      )
      mg_logits, _ = output_layer_mg(mg_mtp_out)  # [S, B, V]

    mg_logits_bsd = mg_logits.detach().numpy().transpose(1, 0, 2)  # → [B, S, V]

    # ---- MaxText forward: decoder_norm → MTP layer → dot(hidden, embedding.T) ----
    mx_raw_hidden = jnp.array(raw_hidden_np)
    mx_embed = jnp.array(embed_np)
    mx_positions = jnp.array(inputs["position_ids"])
    segment_ids = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)

    mx_normed = decoder_norm_mx(mx_raw_hidden)
    mx_mtp_out = mtp_layer_mx(
        prev_hidden_state=mx_normed,
        target_token_embedding=mx_embed,
        position_ids=mx_positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode="train",
    )
    mx_logits = jnp.dot(mx_mtp_out, jnp.array(np_embedding_weight).T)  # [B, S, V]
    mx_logits_np = np.array(mx_logits)

    np.testing.assert_allclose(mx_logits_np, mg_logits_bsd, rtol=RTOL_FWD, atol=ATOL_FWD)

  def test_normed_pipeline_gradients(self) -> None:
    """Gradients through decoder_norm → MTP → logits pipeline match."""
    from maxtext.layers.normalizations import RMSNorm
    from maxtext.utils import maxtext_utils

    inputs = _make_shared_inputs()
    config_mg = MegatronCPUSetup.get_config()

    # Shared numpy weights for output projection and decoder_norm
    rng = np.random.RandomState(SEED + 100)
    np_embedding_weight = rng.randn(VOCAB_SIZE, HIDDEN_SIZE).astype(np.float32)
    np_decoder_norm_scale = rng.randn(HIDDEN_SIZE).astype(np.float32)

    raw_hidden_np = inputs["hidden_state"]
    embed_np = inputs["embedding"]

    # ---- Build BOTH sides first, inject weights BEFORE any forward pass ----

    # Megatron components
    mtp_block_mg = MegatronCPUSetup.build_mtp_block(config_mg)
    mtp_layer_mg = mtp_block_mg.layers[0]
    mtp_layer_mg.train()

    final_ln_mg = torch.nn.RMSNorm(HIDDEN_SIZE, eps=LAYERNORM_EPS)
    with torch.no_grad():
      final_ln_mg.weight.copy_(torch.from_numpy(np_decoder_norm_scale))

    # MaxText components
    config_mx = _build_maxtext_config()
    devices = maxtext_utils.create_device_mesh(config_mx)
    mesh = jax.sharding.Mesh(devices, config_mx.mesh_axes)
    rngs = nnx.Rngs(params=jax.random.PRNGKey(SEED))

    mtp_layer_mx = _build_maxtext_mtp_layer(config_mx, mesh, rngs)

    decoder_norm_mx = RMSNorm(
        num_features=HIDDEN_SIZE,
        epsilon=LAYERNORM_EPS,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    decoder_norm_mx.scale.value = jnp.array(np_decoder_norm_scale)

    # Inject weights (zeroes transformer on both sides) BEFORE forward passes
    TestLevel2MTPLayerForward._inject_weights(mtp_layer_mg, mtp_layer_mx)

    # ---- Megatron forward + backward ----
    mg_raw_hidden = torch.from_numpy(raw_hidden_np.transpose(1, 0, 2)).requires_grad_(True)
    mg_embed = torch.from_numpy(embed_np.transpose(1, 0, 2)).requires_grad_(True)
    mg_output_weight = torch.from_numpy(np_embedding_weight).requires_grad_(True)  # [V, H]
    causal_mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    mg_normed = final_ln_mg(mg_raw_hidden)
    mg_mtp_out = mtp_layer_mg._proj_and_transformer_layer(
        hidden_states=mg_normed, decoder_input=mg_embed, attention_mask=causal_mask
    )
    mg_logits = mg_mtp_out @ mg_output_weight.T  # [S, B, V]
    mg_loss = mg_logits.sum()
    mg_loss.backward()

    mg_grad_h = mg_raw_hidden.grad.detach().numpy().transpose(1, 0, 2)  # → [B, S, D]
    mg_grad_ln = final_ln_mg.weight.grad.detach().numpy()

    # ---- MaxText forward + backward ----
    mx_positions = jnp.array(inputs["position_ids"])
    segment_ids = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)

    # Split params from non-differentiable state (RNG keys are uint32)
    graphdef_mtp, params_mtp, rest_mtp = nnx.split(mtp_layer_mx, nnx.Param, ...)
    graphdef_norm, params_norm, rest_norm = nnx.split(decoder_norm_mx, nnx.Param, ...)

    def _fwd(
        params_norm_d: Any,
        params_mtp_d: Any,
        h: jnp.ndarray,
        e: jnp.ndarray,
        w: jnp.ndarray,
    ) -> jnp.ndarray:
      norm = nnx.merge(graphdef_norm, params_norm_d, rest_norm)
      layer = nnx.merge(graphdef_mtp, params_mtp_d, rest_mtp)
      normed = norm(h)
      out = layer(
          prev_hidden_state=normed,
          target_token_embedding=e,
          position_ids=mx_positions,
          decoder_segment_ids=segment_ids,
          deterministic=True,
          model_mode="train",
      )
      logits = jnp.dot(out, w.T)
      return logits.sum()

    grad_fn = jax.grad(_fwd, argnums=(0, 1, 2, 3, 4))
    g_norm, _, g_h, _, _ = grad_fn(
        params_norm,
        params_mtp,
        jnp.array(raw_hidden_np),
        jnp.array(embed_np),
        jnp.array(np_embedding_weight),
    )

    # Compare hidden state gradient
    mx_grad_h = np.array(g_h)
    np.testing.assert_allclose(mx_grad_h, mg_grad_h, rtol=RTOL_GRAD, atol=ATOL_GRAD)

    # Compare decoder_norm gradient
    mx_grad_ln = np.array(g_norm.scale.value)
    np.testing.assert_allclose(mx_grad_ln, mg_grad_ln, rtol=RTOL_GRAD, atol=ATOL_GRAD)


@unittest.skipUnless(
    _TORCH_AVAILABLE and _MEGATRON_AVAILABLE and _JAX_AVAILABLE,
    "Requires PyTorch, megatron-core, and JAX",
)
class TestLevel7OptimizerStep(unittest.TestCase):
  """Level 7: Optimizer step alignment for MTP-specific parameters."""

  @classmethod
  def setUpClass(cls) -> None:
    _require_all()
    MegatronCPUSetup.initialise()

  def test_adam_single_step_on_norm_weights(self) -> None:
    """After one Adam step on a simple loss, RMSNorm parameters match.

    Both frameworks should produce the same parameter update when given
    the same gradients and optimizer hyperparameters.
    """
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    rng = np.random.RandomState(SEED)

    # Shared initial weight and gradient
    np_weight = rng.randn(HIDDEN_SIZE).astype(np.float32)
    np_grad = rng.randn(HIDDEN_SIZE).astype(np.float32)

    # ---- PyTorch Adam ----
    pt_param = torch.nn.Parameter(torch.from_numpy(np_weight.copy()))
    pt_opt = torch.optim.Adam([pt_param], lr=lr, betas=betas, eps=eps)
    pt_param.grad = torch.from_numpy(np_grad.copy())
    pt_opt.step()
    pt_updated = pt_param.detach().numpy()

    # ---- JAX Adam (manual, matching PyTorch's formula) ----
    import optax

    jax_opt = optax.adam(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps)
    jax_param = jnp.array(np_weight.copy())
    opt_state = jax_opt.init(jax_param)
    updates, _ = jax_opt.update(jnp.array(np_grad.copy()), opt_state, jax_param)
    jax_updated = np.array(optax.apply_updates(jax_param, updates))

    np.testing.assert_allclose(jax_updated, pt_updated, rtol=RTOL_OPT, atol=ATOL_OPT)


if __name__ == "__main__":
  unittest.main()
