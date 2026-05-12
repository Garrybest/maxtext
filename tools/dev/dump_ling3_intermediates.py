#!/usr/bin/env python3
"""Forward one tiny step through ling3-tiny scan model and dump the intermediates pytree.

The point: confirm whether sown values for ling3 scan land at
  intermediates/decoder/moe_layers/{key}                              (flat, layout A)
  intermediates/decoder/moe_layers/layers_{0..3}/{key}                (nested, layout B)
plus how the Phase 1b unscan prefix shows up.

Run on Pod:
    cd /root/maxtext && source maxtext_venv/bin/activate
    python tools/dev/dump_ling3_intermediates.py
"""
from __future__ import annotations

import os
import sys

# Run on a single host with synthetic data, no checkpoint
# Note: KDA Pallas kernel needs TPU/GPU; CPU only works in interpret mode.
# Pod has TPU, so let JAX pick the default (TPU).

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.utils import maxtext_utils
from maxtext.models.models import TransformerLinenPure


def main():
  argv = [
      "dump_ling3_intermediates.py",
      "src/maxtext/configs/base.yml",
      "model_name=ling3-tiny",
      "tokenizer_path=meta-llama/Llama-3.1-8B",
      "tokenizer_type=huggingface",
      "run_name=dump_ling3",
      "base_output_directory=/tmp/dump_ling3",
      "dataset_path=/tmp",
      "dataset_type=synthetic",
      "per_device_batch_size=1",
      "max_target_length=64",
      "scan_layers=true",
      "skip_jax_distributed_system=true",
      "enable_checkpointing=false",
      "logits_via_embedding=false",
      "enable_goodput_recording=false",
      "monitor_goodput=false",
  ]
  config = pyconfig.initialize(argv)

  print(f"decoder_block={config.decoder_block}")
  print(f"num_decoder_layers={config.num_decoder_layers}")
  print(f"first_num_dense_layers={config.first_num_dense_layers}")
  print(f"inhomogeneous_layer_cycle_interval={config.inhomogeneous_layer_cycle_interval}")
  print(f"scan_layers={config.scan_layers}")
  print(f"num_experts={config.num_experts}")
  print(f"mtp_num_layers={getattr(config, 'mtp_num_layers', 0)}")
  print()

  # Build a tiny single-host mesh and model
  devices = jax.devices()
  mesh = jax.sharding.Mesh(np.asarray(devices).reshape(-1), ("data",))
  rng = jax.random.PRNGKey(0)

  model = TransformerLinenPure(config=config, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)

  B = config.global_batch_size_to_train_on
  T = config.max_target_length
  decoder_input_tokens = jnp.ones((B, T), dtype=jnp.int32)
  decoder_target_tokens = jnp.ones((B, T), dtype=jnp.int32)
  decoder_target_mask = jnp.ones((B, T), dtype=jnp.int32)
  decoder_segment_ids = jnp.ones((B, T), dtype=jnp.int32)
  decoder_positions = jnp.broadcast_to(jnp.arange(T)[None, :], (B, T))

  print(f"Initializing params (this can take a minute on CPU)...")
  init_rng = {"params": rng, "dropout": jax.random.fold_in(rng, 1), "aqt": jax.random.fold_in(rng, 2)}
  variables = model.init(
      init_rng,
      decoder_input_tokens=decoder_input_tokens,
      decoder_target_tokens=decoder_target_tokens,
      decoder_target_mask=decoder_target_mask,
      decoder_segment_ids=decoder_segment_ids,
      decoder_positions=decoder_positions,
      enable_dropout=False,
      mutable=("params",),
  )
  params = variables["params"]
  print(f"Init done. Top-level params keys: {list(params.keys())[:10]}")
  print()

  print(f"Running one forward pass with mutable=intermediates ...")
  out, mutated = model.apply(
      {"params": params},
      decoder_input_tokens=decoder_input_tokens,
      decoder_target_tokens=decoder_target_tokens,
      decoder_target_mask=decoder_target_mask,
      decoder_segment_ids=decoder_segment_ids,
      decoder_positions=decoder_positions,
      enable_dropout=False,
      mutable=["intermediates"],
      rngs={"dropout": jax.random.fold_in(rng, 3), "aqt": jax.random.fold_in(rng, 4),
            "mtp_block": jax.random.fold_in(rng, 5), "params": jax.random.fold_in(rng, 6)},
  )
  print(f"Forward done. Mutated collection keys: {list(mutated.keys())}")
  print()

  intermediates = mutated.get("intermediates", {})

  def walk(node, path=()):
    if isinstance(node, dict):
      for k, v in node.items():
        yield from walk(v, path + (str(k),))
    elif isinstance(node, (tuple, list)):
      # Flax sow tuples
      for i, v in enumerate(node):
        if hasattr(v, "shape"):
          yield path + (f"[{i}]",), tuple(v.shape), str(v.dtype)
        else:
          yield from walk(v, path + (f"[{i}]",))
    elif hasattr(node, "shape"):
      yield path, tuple(node.shape), str(node.dtype)

  print("=" * 90)
  print("FULL intermediates tree (filtered to MoE-related sown values):")
  print("=" * 90)
  any_match = False
  for p, sh, dt in walk(intermediates):
    key = "/".join(p)
    if any(t in key for t in ("moe_lb_loss", "moe_z_loss", "moe_expert_counts", "router")):
      print(f"  {key}  shape={sh} dtype={dt}")
      any_match = True
  if not any_match:
    print("  (no MoE keys found — printing top-level structure for debugging)")
    print(f"  intermediates top keys: {list(intermediates.keys())}")
    if "decoder" in intermediates:
      dec = intermediates["decoder"]
      print(f"  intermediates/decoder keys: {list(dec.keys()) if isinstance(dec, dict) else type(dec)}")
      if isinstance(dec, dict) and "moe_layers" in dec:
        ml = dec["moe_layers"]
        print(f"  intermediates/decoder/moe_layers keys: {list(ml.keys()) if isinstance(ml, dict) else type(ml)}")

  print()
  print("=" * 90)
  print("RAW intermediates/decoder structure:")
  print("=" * 90)
  dec = intermediates.get("decoder", {})
  if isinstance(dec, dict):
    for k in sorted(dec.keys()):
      v = dec[k]
      if isinstance(v, dict):
        sub_keys = list(v.keys())
        print(f"  decoder/{k}/  ->  dict with keys: {sub_keys}")
        for sk in sub_keys:
          sv = v[sk]
          if isinstance(sv, dict):
            print(f"    decoder/{k}/{sk}/  ->  dict with keys: {list(sv.keys())}")
          elif isinstance(sv, tuple):
            shapes = [getattr(x, 'shape', type(x).__name__) for x in sv]
            print(f"    decoder/{k}/{sk}  ->  tuple{shapes}")
          else:
            print(f"    decoder/{k}/{sk}  ->  {type(sv).__name__} shape={getattr(sv, 'shape', None)}")
      else:
        print(f"  decoder/{k}  ->  {type(v).__name__}")

  # Now run the buggy and fixed collectors
  print()
  print("=" * 90)
  print("Runtime check of _collect_moe_intermediate_sum:")
  print("=" * 90)
  from maxtext.trainers.pre_train import train as train_mod
  for key in ("moe_lb_loss", "moe_z_loss"):
    try:
      v = train_mod._collect_moe_intermediate_sum(config, mutated, key)
      print(f"  current code _collect_moe_intermediate_sum(..., '{key}') = {float(v):.6e}")
    except Exception as e:
      print(f"  current code _collect_moe_intermediate_sum(..., '{key}') raised: {type(e).__name__}: {e}")

  # Also check the bias path resolution
  print()
  print("=" * 90)
  print("Path lookups _update_deepseek_bias would do (scan mode):")
  print("=" * 90)
  buggy_target = ("params", "decoder", "moe_layers", "mlp", "MoeBlock_0", "gate", "bias")
  print(f"  has({'/'.join(buggy_target)}) = {maxtext_utils.has_nested_key({'params': params}, buggy_target)}")
  for sub in ["layers_0", "layers_1", "layers_2", "layers_3"]:
    real = ("params", "decoder", "moe_layers", sub, "mlp", "MoeBlock_0", "gate", "bias")
    print(f"  has({'/'.join(real)}) = {maxtext_utils.has_nested_key({'params': params}, real)}")
  for i in range(3):
    pre = ("params", "decoder", f"moe_layers_{i}", "mlp", "MoeBlock_0", "gate", "bias")
    print(f"  has({'/'.join(pre)}) = {maxtext_utils.has_nested_key({'params': params}, pre)}")


if __name__ == "__main__":
  main()
