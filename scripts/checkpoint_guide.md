# Checkpoint Conversion & Validation Guide

## 1. HF -> MaxText (Orbax) Conversion

```bash
python src/MaxText/checkpoint_conversion/to_maxtext.py \
    src/maxtext/configs/base.yml \
    model_name=<model_name> \
    base_output_directory=/path/to/output/ \
    --hf_model_path /path/to/hf-weights/ \
    hardware=cpu \
    skip_jax_distributed_system=True \
    scan_layers=False
```

For models with custom HF code (e.g. Ling2), add `--trust_remote_code=True`.
For large models, add `--lazy_load_tensors=True` to reduce RAM usage.

Output: Orbax checkpoint at `{base_output_directory}/{model_name}/hf-to-maxtext/0/items/`.

### Ling2 Example

```bash
python src/MaxText/checkpoint_conversion/to_maxtext.py \
    src/maxtext/configs/base.yml \
    model_name=ling2 \
    base_output_directory=/path/to/output/ \
    --hf_model_path /path/to/ling2-hf/ \
    --trust_remote_code=True \
    hardware=cpu \
    skip_jax_distributed_system=True \
    scan_layers=False
```

## 2. Structure Validation (keys + shapes)

Fast check — only loads metadata, no tensor data. Verifies all Orbax keys map to
HF keys and shapes are compatible.

```bash
python scripts/compare_orbax_safetensors.py \
    --model-name <model_name> \
    --orbax-items-dir /path/to/orbax/0/items/ \
    --hf-dir /path/to/hf-weights/ \
    --report-json structure_report.json
```

The HF `config.json` is loaded automatically from `--hf-dir` to derive
model-specific parameters (num_layers, num_experts, etc.).

Expected output: `mapped_ok_count` matches total, `missing_mapping_count=0`,
`shape_mismatch_count=0`.

## 3. Numerical Validation (tensor values)

Loads actual tensor data and compares per-tensor: max_diff, mean_diff, cosine
similarity, relative error.

### From Orbax checkpoint

```bash
python scripts/compare_params.py \
    --model-name <model_name> \
    --orbax-items-dir /path/to/orbax/0/items/ \
    --hf-dir /path/to/hf-weights/ \
    --report-json value_report.json
```

### From Argus dump

```bash
python scripts/compare_params.py \
    --model-name <model_name> \
    --dump-dir /path/to/argus_dump/maxtext/step_0/rank_0/ \
    --hf-dir /path/to/hf-weights/
```

### Useful options

```bash
--layers 0,4,19       # Compare specific layers only
--list-keys           # List MaxText keys without comparing
```

Expected output: all tensors show `max_diff < 1e-6`, `cosine ~ 1.0`.

## Supported Models

All models registered in `PARAM_MAPPING` are supported, including:
gemma2, gemma3, qwen3, llama3.1, deepseek3, mixtral, olmo3, ling2, etc.

Run with `--model-name <name>` — the full list is printed on error if the name
is not recognized.
