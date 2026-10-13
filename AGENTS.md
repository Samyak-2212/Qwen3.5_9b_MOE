# Instructions for AI Agents

## Project Overview

This project converts dense Qwen3.5 GGUF models into Mixture of Experts (MoE) models
that run in llama.cpp using the `qwen35moe` architecture.

The main script is `dense_to_moe.py`.

## Key Constraints

- **Target runtime**: llama.cpp with existing `qwen35moe` architecture support
- **No new architecture**: the output GGUF must match what `src/models/qwen35moe.cpp`
  in llama.cpp expects -- do not invent custom tensor names or metadata keys
- **CPU-first**: must work on Intel 8250U (4C/8T, 16GB RAM). Process one tensor at
  a time to avoid OOM. Optional GPU (torch+CUDA) support for acceleration
- **Python 3.13**: use the project venv at `.venv/` (created with `uv venv --python 3.13`)
- **Dependencies**: numpy, pyyaml, gguf-py (from llama.cpp repo at `../llama.cpp/gguf-py`),
  optional torch

## Architecture Reference

### Qwen3.5 Dense (input)
- Architecture string: `qwen35`
- Hybrid: interleaved sliding attention (GatedDeltaNet/SSM) + full GQA attention
- Full attention every 4th layer (`full_attention_interval = 4`)
- FFN per layer: SwiGLU with `ffn_gate`, `ffn_up`, `ffn_down`
- 0.8B: 25 layers, n_embd=1024, n_ff=3584
- 9B: 32 layers, n_embd=4096, n_ff=12288

### Qwen3.5-MoE (output)
- Architecture string: `qwen35moe`
- Same hybrid attention/SSM -- only the FFN changes
- FFN replaced with: N routed experts (3D tensors) + 1 shared expert + router gate
- Key metadata: `expert_count`, `expert_used_count`, `expert_feed_forward_length`,
  `expert_shared_feed_forward_length`

### Tensor naming (GGUF)
- Router: `blk.{i}.ffn_gate_inp.weight` -- shape `[n_embd, n_expert]`
- Routed experts: `blk.{i}.ffn_{gate,up}_exps.weight` -- shape `[n_embd, n_ff_exp, n_expert]`
- Routed experts: `blk.{i}.ffn_down_exps.weight` -- shape `[n_ff_exp, n_embd, n_expert]`
- Shared expert: `blk.{i}.ffn_{gate,up,down}_shexp.weight` -- same shape as dense FFN
- Shared gate: `blk.{i}.ffn_gate_inp_shexp.weight` -- shape `[n_embd]`

## Source Files to Study

Before making changes, read these files:

- `../llama.cpp/src/models/qwen35moe.cpp` -- how llama.cpp loads and runs MoE
- `../llama.cpp/gguf-py/gguf/constants.py` -- GGUF tensor/metadata constants
- `../llama.cpp/gguf-py/gguf/gguf_writer.py` -- GGUF writer API
- `../llama.cpp/gguf-py/gguf/gguf_reader.py` -- GGUF reader API

## Code Style

- Keep it simple. One file script, clear functions, minimal abstraction
- Use argparse for CLI
- Print progress to stderr so stdout can be piped
- Process tensors one at a time to limit memory usage
- Comments only where non-obvious. Do not over-comment
- ASCII only in strings and comments -- no unicode arrows, emdashes, or ellipsis chars

## Testing

1. Convert 0.8B model first (fast, cheap validation)
2. Verify output loads in `../llama-b11574-bin-win-cpu-x64/llama-cli.exe`
3. Then convert 9B model
4. Test command:
   ```
   ..\llama-b11574-bin-win-cpu-x64\llama-cli.exe -m OUTPUT.gguf -p "Hello" -n 32
   ```

## Do NOT

- Do not modify llama.cpp source code
- Do not add dependencies beyond numpy, pyyaml, gguf-py, and optional torch
- Do not create HuggingFace-format intermediate files
- Do not attempt router training in this script (that is a separate project)
