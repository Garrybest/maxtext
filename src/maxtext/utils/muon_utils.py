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


"""Utilities for Muon optimizer integration and dimension number generation.

This module provides functions to automatically generate MuonDimensionNumbers
for various MaxText models. These dimension numbers are crucial for the Muon
optimizer to correctly apply its update rules.

This module can also be run as a script to inspect the generated dimension
numbers for a specific model. Example:
  python3 -m MaxText.muon_utils qwen3-4b True
"""


import os
import sys
from typing import Optional, Tuple

import flax.linen as nn
import jax
from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import maxtext_utils
from third_party.optax_muon import MuonDimensionNumbers as mdn


Transformer = models.transformer_as_linen


def _is_path_contain_any(tuples, path):
  return any(x in path for x in tuples)


def transform_logic(path: Tuple[str, ...], config=None) -> Optional[mdn]:
  """
  Determines Muon dimension numbers based on the parameter's hierarchical path.

  This function defines the mapping from a parameter's logical path within the model
  to its corresponding MuonDimensionNumbers (mdn). The strategy is applied in
  a specific order to handle general cases and then more specific ones, allowing
  for fall-through logic in nested structures.

  Strategy:
  1. Exclusions: Parameters not suitable for Muon (e.g., scalars, embeddings,
     unembedding) are explicitly returned as `None`.
  2. Special Weights:
     2.1 MoE Block Specific Weights
     2.2 Self-Attention Specific Weights
  3. Standard Weights: Default mapping for most other 3D weight shapes.

  Args:
    path: A tuple of strings representing the hierarchical path of the parameter.
    config: Optional model config. When provided, enables MLA-specific
      component_splits for wq_b/wkv_b to match Megatron's per-component NS.

  Returns:
    An instance of `MuonDimensionNumbers` if a specific mapping is found,
    `None` for excluded parameters, or a default `mdn` for standard weights.
  """
  param_name = path[-2] if len(path) >= 2 and path[-1] == "kernel" else path[-1]

  # 1 Exclude parameters not suitable for Muon (scalar, embeddings, unembedding)
  if _is_path_contain_any(("scale", "bias", "embedding", "logits_dense"), path):
    return None
  if param_name in ("A_log", "dt_bias", "q_conv", "k_conv", "v_conv"):
    return None

  # 2 Special weights
  # 2.1 Special weights: MoE, [0, L, -2, -1]
  # L (optional) stands for layer when scan_layers=True
  if "MoeBlock_0" in path:
    if param_name in ("wi_0", "wi_1", "wo"):
      return mdn((-2,), (-1,))

  # 2.2 Special weights: Attention projections.
  # Output projections: heads merged into reduction → full-matrix NS, matching
  # Megatron where linear_proj gets ParamTypeInMuonStrategy.none (full matrix).
  if param_name in ("out", "o_proj"):
    return mdn((0, -2), (-1,))

  # MLA projections with per-component splitting: split the output axis into
  # semantic components (nope/rope for wq_b, nope/value for wkv_b) and merge
  # heads into each component before NS, matching Megatron's split_head logic.
  if param_name == "wq_b":
    if config is not None and getattr(config, "qk_rope_head_dim", 0) > 0 and getattr(config, "qk_nope_head_dim", 0) > 0:
      return mdn((0,), (-1,), component_splits=(config.qk_nope_head_dim, config.qk_rope_head_dim))
    return mdn((0,), (-1,))

  if param_name == "wkv_b":
    if config is not None and getattr(config, "qk_nope_head_dim", 0) > 0 and getattr(config, "v_head_dim", 0) > 0:
      return mdn((0,), (-1,), component_splits=(config.qk_nope_head_dim, config.v_head_dim))
    return mdn((0,), (-1,))

  # Standard attention input projections and KDA projections: per-head NS.
  if param_name in ("query", "key", "value", "q_proj", "k_proj", "v_proj", "g_proj", "gate_proj"):
    return mdn((0,), (-1,))

  # 3 Standard weights, [0, L, -1]
  return mdn((0,), (-1,))


def get_transform_tree(tree, path=(), config=None):
  """Extraction utility via recursion."""
  if isinstance(tree, dict):
    return {k: get_transform_tree(v, path + (k,), config=config) for k, v in tree.items()}
  else:
    return transform_logic(path, config=config)


def get_muon_weight_dimension_numbers(model, config, verbose=False):
  """Extract muon dimension number from model structure."""
  # quickly get param structure without materialization
  abstract_param = maxtext_utils.get_abstract_param(model, config)
  # get muon dimension number from param
  muon_weight_dimension_numbers = get_transform_tree(abstract_param, config=config)
  if verbose:
    _print_structure_debug(abstract_param, muon_weight_dimension_numbers)
  return muon_weight_dimension_numbers


def _print_structure_debug(abstract_param, muon_weight_dimension_numbers):
  """Prints the model structure and the resulting Muon config."""
  # Access the shape from the inner ShapeDtypeStruct and names from the wrapper
  # Return a new tree with the same structure containing only shapes/names
  info_tree = jax.tree_util.tree_map(
      lambda leaf: {"shape": leaf.value.shape, "names": leaf.names},
      abstract_param,
      is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned),
  )
  print(f"\n=== Model Structure ===\n{info_tree}")
  print(f"\n=== Muon Dimension Numbers ===\n{muon_weight_dimension_numbers}")
  print("\nIs this reasonable?")


def get_model_mdn(model_name, scan_layers=True, verbose=False):
  """Initializes a model and retrieves its Muon dimension numbers.

  This function sets up the configuration for a given model, initializes the
  transformer model, and then extracts the Muon dimension numbers for the model's
  weights. It can optionally print verbose debug information.

  Args:
    model_name: The name of the model to be initialized.
    scan_layers: Whether to use layer scanning in the model configuration.
    verbose: If True, prints detailed debugging information about the model
      structure and Muon dimension numbers.

  Returns:
    A tree structure containing the Muon dimension numbers for the model's
    parameters.
  """
  # Setup config
  argv = [
      None,
      os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
      f"model_name={model_name}",
      f"scan_layers={scan_layers}",
      "attention=dot_product",
  ]
  config = pyconfig.initialize(argv)
  # Setup model
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh=mesh, quant=quant)
  # Get dimension number
  muon_weight_dimension_numbers = get_muon_weight_dimension_numbers(model, config, verbose=verbose)
  return muon_weight_dimension_numbers


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python3 -m MaxText.muon_utils <model_name> <scan_layers:True/False>")
    sys.exit(1)
  model_name_arg = sys.argv[1]
  scan_layers_arg = sys.argv[2].lower() == "true"
  get_model_mdn(model_name_arg, scan_layers_arg, verbose=True)
