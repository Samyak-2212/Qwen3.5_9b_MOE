# Qwen3.5 Dense-to-MoE Converter

Convert dense Qwen3.5 GGUF models into Mixture of Experts (MoE) models that
run directly in [llama.cpp](https://github.com/ggml-org/llama.cpp).

## What This Does

Takes a standard dense Qwen3.5 model (0.8B, 9B, etc.) in GGUF format and
produces a `qwen35moe`-architecture GGUF where each layer's FFN is replaced
with multiple routed experts plus an optional shared expert.

The output model loads in llama.cpp with no source modifications required.

**Note**: The converted model has randomly initialized routers. It will
generate tokens but quality will be degraded until the router is trained
on calibration data.

## Requirements

- Python 3.13+
- numpy, pyyaml
- [gguf-py](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py) from llama.cpp
- (Optional) PyTorch with CUDA for GPU-accelerated conversion

## Setup

```bash
uv venv .venv --python 3.13
uv pip install numpy pyyaml
# gguf-py is loaded from ../llama.cpp/gguf-py via sys.path
```

## Usage

```bash
# Basic conversion (8 experts, top-2, with shared expert)
python dense_to_moe.py Qwen3.5-0.8B-Q8_0.gguf

# Full options
python dense_to_moe.py Qwen3.5-9B-Q4_K_M.gguf \
    --output Qwen3.5-9B-MoE.gguf \
    --num-experts 8 \
    --top-k 2 \
    --shared-expert \
    --output-type F16 \
    --noise-scale 0.01 \
    --seed 42

# Dry run (show plan without writing)
python dense_to_moe.py Qwen3.5-9B-Q4_K_M.gguf --dry-run
```

### Options

| Flag | Default | Description |
|:---|:---|:---|
| `-o, --output` | `INPUT-moe.gguf` | Output file path |
| `--num-experts` | 8 | Routed experts per layer |
| `--top-k` | 2 | Active experts per token |
| `--shared-expert` | yes | Include always-active shared expert |
| `--no-shared-expert` | - | Disable shared expert |
| `--output-type` | F16 | Tensor output type: F32, F16, Q8_0, auto |
| `--noise-scale` | 0.01 | Expert weight perturbation scale |
| `--moe-layers` | all | Layers to convert: "all", "0-31", "2,5,8" |
| `--convert-mtp` | yes | Convert MTP layer FFN to MoE |
| `--expert-ff-size` | n_ff | Expert hidden dimension |
| `--seed` | 42 | Random seed |
| `--use-gpu` | no | Use torch+CUDA if available |

## How It Works

1. Reads the dense GGUF and identifies FFN tensors in each layer
2. For each MoE layer:
   - Dequantizes FFN weights (gate, up, down) to F32
   - Creates N expert copies with small Gaussian noise (sparse upcycling)
   - Stacks copies into 3D expert tensors `[dim, ff, n_expert]`
   - Optionally preserves original FFN as shared expert (unchanged)
   - Creates a randomly initialized router tensor
3. Writes a new GGUF with `qwen35moe` architecture and MoE metadata
4. All non-FFN tensors (attention, SSM, norms, embeddings) pass through unchanged

## Testing

```bash
# Verify the output loads
../llama-b11574-bin-win-cpu-x64/llama-cli.exe \
    -m Qwen3.5-0.8B-MoE-Q8_0.gguf \
    -p "Hello, how are you?" -n 32
```

## Architecture

The Qwen3.5 family uses a hybrid attention-SSM architecture:
- Sliding attention (GatedDeltaNet) + SSM layers (3 out of every 4)
- Full GQA attention layers (every 4th)
- SwiGLU FFN in every layer

This converter only modifies the FFN portion. The attention and SSM
components remain identical to the dense model.

## Memory Requirements

| Model | Experts | Output Type | Approx Output Size | Peak RAM |
|:---|:---|:---|:---|:---|
| 0.8B | 8 | Q8_0 | ~7 GB | ~2 GB |
| 0.8B | 64 | Q8_0 | ~55 GB | ~2 GB |
| 9B | 8 | Q4_K_M | ~22 GB | ~4 GB |
| 9B | 8 | F16 | ~44 GB | ~4 GB |

Processing is done one tensor at a time to keep peak RAM low.

## License

Apache-2.0 (matching Qwen3.5 model license)
