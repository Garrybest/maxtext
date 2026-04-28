# Checkpoint Conversion & Validation Guide

## 1. HF → MaxText (Orbax)

```bash
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    model_name=<model_name> \
    base_output_directory=/path/to/output/ \
    --hf_model_path=/path/to/hf-weights/ \
    hardware=cpu \
    skip_jax_distributed_system=True \
    scan_layers=False
```

Output: Orbax checkpoint at `{base_output_directory}/0/items/`.

For custom HF architectures (e.g. Ling2), add `--trust_remote_code=True`. For
large models, add `--lazy_load_tensors=True` to reduce RAM. When the model uses
MLA (different Q/K vs V head dims), also pass `attention=dot_product`.

**Ling2 example:**

```bash
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    model_name=ling2 \
    base_output_directory=/path/to/ling2-maxtext-output \
    attention=dot_product \
    --hf_model_path=/path/to/ling2-hf/ \
    --trust_remote_code=True \
    --lazy_load_tensors=True \
    hardware=cpu \
    skip_jax_distributed_system=True \
    scan_layers=False
```

**Ling3 example:**

```bash
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    model_name=ling3-tiny \
    base_output_directory=/path/to/ling3-maxtext-output \
    attention=dot_product \
    --hf_model_path=/path/to/ling3-hf/ \
    --trust_remote_code=True \
    --lazy_load_tensors=True \
    hardware=cpu \
    skip_jax_distributed_system=True \
    scan_layers=False
```

## 2. MaxText (Orbax) → HF

```bash
python3 -m maxtext.checkpoint_conversion.to_huggingface \
    src/maxtext/configs/base.yml \
    model_name=<model_name> \
    load_parameters_path=/path/to/maxtext-ckpt/0/items/ \
    base_output_directory=/path/to/output/ \
    hardware=cpu \
    skip_jax_distributed_system=True \
    scan_layers=False
```

Output: `config.json` + `tokenizer.*` + sharded `model-*.safetensors` + `model.safetensors.index.json`.

For custom HF architectures not registered in `HF_MODEL_CONFIGS` (e.g. Ling2),
add `--hf_reference_path=/path/to/hf-repo/` and `--trust_remote_code=True` —
only `config.json` and tokenizer files are read from that path; weights come
from `load_parameters_path`.

**Ling2 example:**

```bash
python3 -m maxtext.checkpoint_conversion.to_huggingface \
    src/maxtext/configs/base.yml \
    model_name=ling2 \
    load_parameters_path=/path/to/ling2-maxtext-output/0/items/ \
    base_output_directory=/path/to/ling2-hf-output \
    attention=dot_product \
    --hf_reference_path=/path/to/ling2-hf/ \
    --trust_remote_code=True \
    hardware=cpu \
    skip_jax_distributed_system=True \
    scan_layers=False
```

**Ling3 example:**

```bash
python3 -m maxtext.checkpoint_conversion.to_huggingface \
    src/maxtext/configs/base.yml \
    model_name=ling3-tiny \
    load_parameters_path=/path/to/ling3-maxtext-output/0/items/ \
    base_output_directory=/path/to/ling3-hf-output \
    attention=dot_product \
    --hf_reference_path=/path/to/ling3-hf/ \
    --trust_remote_code=True \
    hardware=cpu \
    skip_jax_distributed_system=True \
    scan_layers=False
```

## 3. Checkpoint Comparison

`scripts/compare_checkpoints.py` validates any of the conversions above. Each
side is specified as `TYPE:PATH` where `TYPE` is `hf` or `orbax`. Add
`--model-name <name>` when comparing across formats.

**Validate MaxText → HF** (HF ↔ HF, bit-exact, ~1 min):

```bash
python3 -m scripts.compare_checkpoints \
    --left hf:/path/to/reference-hf/ \
    --right hf:/path/to/converted-hf/
```

**Validate HF → MaxText — structure only** (fast, metadata only):

```bash
python3 -m scripts.compare_checkpoints \
    --left orbax:/path/to/orbax/0/items/ \
    --right hf:/path/to/hf/ \
    --model-name <model_name> \
    --mode structure
```

**Validate HF → MaxText — numerical** (loads all tensors, ~10 min):

```bash
JAX_PLATFORMS=cpu XLA_FLAGS='--xla_force_host_platform_device_count=16' \
python3 -m scripts.compare_checkpoints \
    --left orbax:/path/to/orbax/0/items/ \
    --right hf:/path/to/hf/ \
    --model-name <model_name>
```

The CPU + 16-device environment is required for Orbax restore to match the
conversion-time virtual topology.

**Useful options:**

| Flag | Purpose |
|---|---|
| `--mode structure\|numerical\|both` | Default: `both`. |
| `--sample N` | HF↔HF: only check N tensors numerically. |
| `--atol F` | Allow per-tensor `max_diff ≤ F` (default: 0 = bit-exact). |
| `--layers 0,4,19` | MT↔HF: restrict to specific layer indices. |
| `--list-keys` | Print left source keys + shapes; exit. |
| `--report-json path` | Dump full JSON report. |

Expected on success: `VERDICT: OK ✅` with `bit-exact: N / N` (HF↔HF) or
`matched: N issues: 0` (MT↔HF).

## Supported Models

All models registered in `PARAM_MAPPING` are supported: gemma2, gemma3, qwen3,
llama3.1, deepseek3, mixtral, olmo3, ling2, etc. Run with an unknown
`--model-name` to see the full list.
