#!/usr/bin/env python3
"""
dense_to_moe.py - Convert a dense Qwen3.5 GGUF to a qwen35moe MoE GGUF.

The output model is structurally valid for llama.cpp but has randomly initialized
routers. Router training is a separate step.

Usage:
    python dense_to_moe.py INPUT.gguf [options]

See README.md or --help for full option list.
"""

import argparse
import os
import sys

import numpy as np

GGUF_PY = os.path.join(os.path.dirname(__file__), "..", "llama.cpp", "gguf-py")
sys.path.insert(0, GGUF_PY)

from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
from gguf.quants import dequantize, quantize

# ---------------------------------------------------------------------------
# Types and constants
# ---------------------------------------------------------------------------

DENSE_ARCH = "qwen35"
MOE_ARCH   = "qwen35moe"

# Tensor type map for output-type selection
OUTPUT_DTYPES = {
    "F32":   (np.float32,  None),
    "F16":   (np.float16,  None),
    "Q8_0":  (np.float32,  GGMLQuantizationType.Q8_0),
    "Q4_K_M":(np.float32,  GGMLQuantizationType.Q4_K),
    "Q4_K":  (np.float32,  GGMLQuantizationType.Q4_K),
    "Q5_K":  (np.float32,  GGMLQuantizationType.Q5_K),
    "Q6_K":  (np.float32,  GGMLQuantizationType.Q6_K),
}

# Quant type -> numpy-compatible element type for raw pass-through
QTYPE_TO_NP = {
    GGMLQuantizationType.F32:  np.float32,
    GGMLQuantizationType.F16:  np.float16,
    GGMLQuantizationType.BF16: np.uint16,
}

# Names of the three FFN sub-tensors for each block
FFN_SUBTENSORS = ("ffn_gate", "ffn_up", "ffn_down")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(args=None):
    p = argparse.ArgumentParser(
        description="Convert dense Qwen3.5 GGUF to qwen35moe MoE GGUF."
    )
    p.add_argument("input", help="Path to input dense GGUF")
    p.add_argument("-o", "--output", default=None,
                   help="Output path (default: INPUT-moe.gguf)")
    p.add_argument("--num-experts", type=int, default=8,
                   help="Number of routed experts per layer (default: 8)")
    p.add_argument("--top-k", type=int, default=2,
                   help="Active experts per token (default: 2)")
    p.add_argument("--shared-expert", dest="shared_expert",
                   action="store_true", default=True,
                   help="Include shared expert (default: yes)")
    p.add_argument("--no-shared-expert", dest="shared_expert",
                   action="store_false",
                   help="Disable shared expert")
    p.add_argument("--output-type", default="F16",
                   choices=list(OUTPUT_DTYPES.keys()),
                   help="Expert tensor output type (default: F16)")
    p.add_argument("--noise-scale", type=float, default=0.01,
                   help="Expert weight noise = noise_scale * std(weights) (default: 0.01)")
    p.add_argument("--moe-layers", default="all",
                   help='Layers to convert: "all" or range like "0-31" (default: all)')
    p.add_argument("--convert-mtp", dest="convert_mtp",
                   action="store_true", default=True,
                   help="Convert MTP layer FFN to MoE (default: yes)")
    p.add_argument("--no-convert-mtp", dest="convert_mtp",
                   action="store_false",
                   help="Keep MTP layer FFN dense")
    p.add_argument("--expert-ff-size", type=int, default=None,
                   help="Expert FFN hidden dim (default: same as dense n_ff)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--use-gpu", action="store_true", default=False,
                   help="Use torch+CUDA for tensor ops if available")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Print plan without writing output file")
    return p.parse_args(args)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_meta_int(reader, key, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default
    return int(field.parts[field.data[0]].item())


def get_meta_float(reader, key, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default
    return float(field.parts[field.data[0]].item())


def get_meta_str(reader, key, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default
    return str(bytes(field.parts[field.data[0]]), "utf-8")


def get_meta_int_arr(reader, key, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default
    return [int(field.parts[i].item()) for i in field.data]


def get_meta_str_arr(reader, key, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default
    return [str(bytes(field.parts[i]), "utf-8") for i in field.data]


def parse_moe_layers(spec, n_layer):
    if spec == "all":
        return set(range(n_layer))
    if "-" in spec:
        a, b = spec.split("-", 1)
        return set(range(int(a), int(b) + 1))
    return {int(x) for x in spec.split(",")}


def tensor_to_f32(tensor):
    """Dequantize a reader tensor to a numpy float32 array in (rows, cols) layout."""
    qtype = tensor.tensor_type
    if qtype == GGMLQuantizationType.F32:
        return tensor.data.view(np.float32).reshape(tuple(reversed(tensor.shape)))
    if qtype == GGMLQuantizationType.F16:
        return tensor.data.view(np.float16).reshape(tuple(reversed(tensor.shape))).astype(np.float32)
    if qtype == GGMLQuantizationType.BF16:
        raw = tensor.data.view(np.uint16).reshape(tuple(reversed(tensor.shape)))
        # BF16 -> F32: insert 16 zero bits at the low end
        f32 = (raw.astype(np.uint32) << 16).view(np.float32)
        return f32
    # quantized types: dequantize returns float32 in (rows, cols) layout
    return dequantize(tensor.data, qtype)


def f32_to_output(arr, out_np_dtype, out_qtype):
    """Convert float32 numpy array to the requested output format."""
    if out_qtype is not None:
        return quantize(arr.astype(np.float32), out_qtype)
    return arr.astype(out_np_dtype)


def maybe_use_torch(use_gpu):
    """Return (torch, device) if torch+CUDA is available and requested."""
    if not use_gpu:
        return None, None
    try:
        import torch
        if torch.cuda.is_available():
            return torch, torch.device("cuda")
        eprint("warning: --use-gpu set but CUDA not available, falling back to numpy")
    except ImportError:
        eprint("warning: --use-gpu set but torch not installed, falling back to numpy")
    return None, None


# ---------------------------------------------------------------------------
# Expert tensor creation
# ---------------------------------------------------------------------------

def make_expert_tensor(w_f32, num_experts, noise_scale, rng, is_down):
    """
    Given a dense FFN weight (float32, NumPy row-major shape), create a stacked
    3D expert tensor.

    In NumPy row-major order:
    For ffn_gate / ffn_up: dense shape is (n_ff, n_embd)
    For ffn_down:          dense shape is (n_embd, n_ff)

    Stacking along axis=0 gives:
      ffn_gate_exps / ffn_up_exps: (n_expert, n_ff_exp, n_embd)
      ffn_down_exps:                (n_expert, n_embd, n_ff_exp)

    When written by GGUFWriter (which reverses NumPy dimensions):
      ffn_gate_exps / ffn_up_exps: [n_embd, n_ff_exp, n_expert]
      ffn_down_exps:                [n_ff_exp, n_embd, n_expert]
    Matching qwen35moe.cpp expectations.
    """
    noise_std = noise_scale * float(w_f32.std())
    experts = []
    for _ in range(num_experts):
        if noise_std > 0:
            e = w_f32 + rng.normal(0.0, noise_std, w_f32.shape).astype(np.float32)
        else:
            e = w_f32.copy()
        experts.append(e)
    # stack along axis=0 so GGUFWriter packs [fastest, ..., n_expert]
    return np.stack(experts, axis=0)


# ---------------------------------------------------------------------------
# Metadata pass-through
# ---------------------------------------------------------------------------

def copy_metadata(reader, writer, arch_out, args, n_ff, n_embd, n_layer):
    """Write all metadata from reader to writer, adapting for MoE."""
    from gguf import GGUFValueType

    # Fields to skip (synthesized reader fields or rewritten manually)
    skip_keys = {
        "general.architecture",
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        f"{DENSE_ARCH}.expert_count",
        f"{DENSE_ARCH}.expert_used_count",
        f"{DENSE_ARCH}.expert_feed_forward_length",
        f"{DENSE_ARCH}.expert_shared_feed_forward_length",
        f"{MOE_ARCH}.expert_count",
        f"{MOE_ARCH}.expert_used_count",
        f"{MOE_ARCH}.expert_feed_forward_length",
        f"{MOE_ARCH}.expert_shared_feed_forward_length",
    }

    for name, field in reader.fields.items():
        if name in skip_keys:
            continue

        # Rename arch-prefixed keys from qwen35 to qwen35moe
        out_name = name
        if name.startswith(f"{DENSE_ARCH}."):
            out_name = f"{arch_out}." + name[len(DENSE_ARCH) + 1:]

        vtype = field.types[0]

        try:
            if vtype == GGUFValueType.STRING:
                val = str(bytes(field.parts[field.data[0]]), "utf-8")
                writer.add_string(out_name, val)
            elif vtype == GGUFValueType.UINT32:
                writer.add_uint32(out_name, int(field.parts[field.data[0]].item()))
            elif vtype == GGUFValueType.INT32:
                writer.add_int32(out_name, int(field.parts[field.data[0]].item()))
            elif vtype == GGUFValueType.FLOAT32:
                writer.add_float32(out_name, float(field.parts[field.data[0]].item()))
            elif vtype == GGUFValueType.BOOL:
                writer.add_bool(out_name, bool(field.parts[field.data[0]].item()))
            elif vtype == GGUFValueType.UINT64:
                writer.add_uint64(out_name, int(field.parts[field.data[0]].item()))
            elif vtype == GGUFValueType.INT64:
                writer.add_int64(out_name, int(field.parts[field.data[0]].item()))
            elif vtype == GGUFValueType.FLOAT64:
                writer.add_float64(out_name, float(field.parts[field.data[0]].item()))
            elif vtype == GGUFValueType.UINT8:
                writer.add_uint8(out_name, int(field.parts[field.data[0]].item()))
            elif vtype == GGUFValueType.INT8:
                writer.add_int8(out_name, int(field.parts[field.data[0]].item()))
            elif vtype == GGUFValueType.UINT16:
                writer.add_uint16(out_name, int(field.parts[field.data[0]].item()))
            elif vtype == GGUFValueType.INT16:
                writer.add_int16(out_name, int(field.parts[field.data[0]].item()))
            elif vtype == GGUFValueType.ARRAY:
                # nested array -- use element type
                elem_type = field.types[1] if len(field.types) > 1 else None
                if elem_type == GGUFValueType.STRING:
                    vals = [str(bytes(field.parts[i]), "utf-8") for i in field.data]
                    writer.add_array(out_name, vals)
                elif elem_type == GGUFValueType.INT32:
                    vals = [int(field.parts[i].item()) for i in field.data]
                    if out_name.endswith(".rope.dimension_sections"):
                        writer.add_key_value(out_name, vals, GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT32)
                    else:
                        writer.add_key_value(out_name, vals, GGUFValueType.ARRAY, sub_type=GGUFValueType.INT32)
                elif elem_type == GGUFValueType.UINT32:
                    vals = [int(field.parts[i].item()) for i in field.data]
                    writer.add_key_value(out_name, vals, GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT32)
                elif elem_type == GGUFValueType.FLOAT32:
                    vals = [float(field.parts[i].item()) for i in field.data]
                    writer.add_key_value(out_name, vals, GGUFValueType.ARRAY, sub_type=GGUFValueType.FLOAT32)
                elif elem_type == GGUFValueType.INT64:
                    vals = [int(field.parts[i].item()) for i in field.data]
                    writer.add_key_value(out_name, vals, GGUFValueType.ARRAY, sub_type=GGUFValueType.INT64)
                elif elem_type == GGUFValueType.UINT64:
                    vals = [int(field.parts[i].item()) for i in field.data]
                    writer.add_key_value(out_name, vals, GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT64)
                elif elem_type == GGUFValueType.FLOAT64:
                    vals = [float(field.parts[i].item()) for i in field.data]
                    writer.add_key_value(out_name, vals, GGUFValueType.ARRAY, sub_type=GGUFValueType.FLOAT64)
                elif elem_type == GGUFValueType.BOOL:
                    vals = [bool(field.parts[i].item()) for i in field.data]
                    writer.add_key_value(out_name, vals, GGUFValueType.ARRAY, sub_type=GGUFValueType.BOOL)
                elif elem_type == GGUFValueType.INT8:
                    vals = [int(field.parts[i].item()) for i in field.data]
                    writer.add_key_value(out_name, vals, GGUFValueType.ARRAY, sub_type=GGUFValueType.INT8)
                elif elem_type == GGUFValueType.UINT8:
                    vals = [int(field.parts[i].item()) for i in field.data]
                    writer.add_key_value(out_name, vals, GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT8)
                else:
                    eprint(f"  skip array field {name}: unsupported elem type {elem_type}")
            else:
                eprint(f"  skip field {name}: unsupported type {vtype}")
        except Exception as exc:
            eprint(f"  warning: could not copy field {name}: {exc}")

    # Ensure rope.dimension_sections is present
    rope_sec_key = f"{arch_out}.rope.dimension_sections"
    if not any(rope_sec_key in kv_data for kv_data in writer.kv_data):
        writer.add_key_value(rope_sec_key, [11, 11, 10, 0], GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT32)

    # Write MoE-specific metadata
    expert_ff = args.expert_ff_size if args.expert_ff_size else n_ff
    writer.add_expert_count(args.num_experts)
    writer.add_expert_used_count(args.top_k)
    writer.add_expert_feed_forward_length(expert_ff)
    if args.shared_expert:
        writer.add_expert_shared_count(1)
        writer.add_expert_shared_feed_forward_length(n_ff)


# ---------------------------------------------------------------------------
# Tensor pass-through (raw bytes, no dequant needed)
# ---------------------------------------------------------------------------

def passthrough_tensor(writer, tensor):
    """Copy a tensor from reader to writer without modification."""
    qtype = tensor.tensor_type

    if qtype in QTYPE_TO_NP:
        # Float types: reshape raw bytes
        np_dtype = QTYPE_TO_NP[qtype]
        data = tensor.data.view(np_dtype).reshape(tuple(reversed(tensor.shape)))
        writer.add_tensor(tensor.name, data)
    else:
        # Quantized: pass raw bytes with dtype hint; raw_shape defaults to tensor.data.shape
        writer.add_tensor(
            tensor.name,
            tensor.data,
            raw_dtype=qtype,
        )


# ---------------------------------------------------------------------------
# FFN conversion for a single block
# ---------------------------------------------------------------------------

def convert_ffn_block(
    writer, tensors_by_name, layer_idx, n_ff, n_embd,
    num_experts, noise_scale, shared_expert,
    out_np_dtype, out_qtype, expert_ff_size, rng
):
    """
    Convert ffn_gate, ffn_up, ffn_down for one block to MoE tensors.
    Processes and writes one sub-tensor at a time to stay memory-efficient.
    """
    prefix = f"blk.{layer_idx}"

    # --- Router: small random Gaussian ---
    # Shape (num_experts, n_embd) so GGUF receives [n_embd, num_experts]
    router = rng.normal(0.0, 0.02 / np.sqrt(n_embd), (num_experts, n_embd)).astype(np.float32)
    writer.add_tensor(f"{prefix}.ffn_gate_inp.weight", router)
    del router

    for sub in FFN_SUBTENSORS:
        key = f"{prefix}.{sub}.weight"
        tensor = tensors_by_name.get(key)
        if tensor is None:
            eprint(f"  warning: tensor {key} not found, skipping")
            continue

        w = tensor_to_f32(tensor)

        # --- Shared expert: direct copy of original FFN ---
        if shared_expert:
            shexp_key = f"{prefix}.{sub}_shexp.weight"
            shexp_data = f32_to_output(w, out_np_dtype, out_qtype)
            if out_qtype is not None:
                writer.add_tensor(shexp_key, shexp_data, raw_dtype=out_qtype)
            else:
                writer.add_tensor(shexp_key, shexp_data)
            del shexp_data

        # --- Routed experts: N copies with noise, stacked as 3D tensor ---
        is_down = (sub == "ffn_down")
        w_exp = w
        if expert_ff_size is not None and expert_ff_size != n_ff:
            if is_down:
                w_exp = w[:, :expert_ff_size] if expert_ff_size < n_ff else np.pad(w, ((0, 0), (0, expert_ff_size - n_ff)))
            else:
                w_exp = w[:expert_ff_size, :] if expert_ff_size < n_ff else np.pad(w, ((0, expert_ff_size - n_ff), (0, 0)))

        exp_3d = make_expert_tensor(w_exp, num_experts, noise_scale, rng, is_down)
        del w
        if w_exp is not w:
            del w_exp

        exp_key = f"{prefix}.{sub}_exps.weight"
        exp_data = f32_to_output(exp_3d, out_np_dtype, out_qtype)
        del exp_3d

        if out_qtype is not None:
            writer.add_tensor(exp_key, exp_data, raw_dtype=out_qtype)
        else:
            writer.add_tensor(exp_key, exp_data)
        del exp_data

    # --- Shared expert gate: zeros -> sigmoid(0) = 0.5 ---
    if shared_expert:
        gate = np.zeros(n_embd, dtype=np.float32)
        writer.add_tensor(f"{prefix}.ffn_gate_inp_shexp.weight", gate)
        del gate


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    _torch, _device = maybe_use_torch(args.use_gpu)

    # Output path
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = base + "-moe.gguf"

    # Check for unsupported quantization types in pure Python gguf-py
    unsupported_kquants = {"Q4_K_M", "Q4_K", "Q5_K", "Q6_K"}
    if args.output_type in unsupported_kquants:
        eprint(f"error: output-type '{args.output_type}' is not supported by pure-Python gguf-py (K-quants require C++ ggml quantize).")
        eprint("hint: convert to Q8_0, F16, or F32 first, then quantize using:")
        eprint(f"      ..\\llama-b11574-bin-win-cpu-x64\\llama-quantize.exe <converted-model.gguf> <output.gguf> {args.output_type}")
        sys.exit(1)

    out_np_dtype, out_qtype = OUTPUT_DTYPES[args.output_type]

    # --- Read input ---
    eprint(f"Loading {args.input} ...")
    reader = GGUFReader(args.input)

    arch = get_meta_str(reader, "general.architecture")
    if arch != DENSE_ARCH:
        eprint(f"error: expected architecture '{DENSE_ARCH}', got '{arch}'")
        sys.exit(1)

    n_layer     = get_meta_int(reader, f"{DENSE_ARCH}.block_count")
    n_embd      = get_meta_int(reader, f"{DENSE_ARCH}.embedding_length")
    n_ff        = get_meta_int(reader, f"{DENSE_ARCH}.feed_forward_length")
    n_head      = get_meta_int(reader, f"{DENSE_ARCH}.attention.head_count")
    n_head_kv   = get_meta_int(reader, f"{DENSE_ARCH}.attention.head_count_kv")
    full_attn_interval = get_meta_int(reader, f"{DENSE_ARCH}.full_attention_interval", 4)
    nextn_layers = get_meta_int(reader, f"{DENSE_ARCH}.nextn_predict_layers", 0)
    n_layer_all  = n_layer + nextn_layers

    expert_ff = args.expert_ff_size if args.expert_ff_size else n_ff

    # Validate against llama.cpp qwen35moe constraints
    if not args.shared_expert:
        eprint("error: llama.cpp qwen35moe architecture requires shared expert tensors (blk.N.ffn_*_shexp).")
        eprint("       --no-shared-expert produces models that cannot be loaded by llama.cpp.")
        sys.exit(1)

    moe_layers = parse_moe_layers(args.moe_layers, n_layer)
    if args.moe_layers != "all" and moe_layers != set(range(n_layer)):
        eprint("error: llama.cpp qwen35moe architecture requires all trunk layers to be MoE.")
        eprint("       partial layer conversion (--moe-layers) produces models that cannot be loaded by llama.cpp.")
        sys.exit(1)

    if nextn_layers > 0:
        if not args.convert_mtp:
            eprint("error: llama.cpp qwen35moe architecture requires MTP layer FFN to be converted to MoE.")
            eprint("       --no-convert-mtp produces models that cannot be loaded by llama.cpp.")
            sys.exit(1)
        for i in range(n_layer, n_layer_all):
            moe_layers.add(i)

    eprint(f"  arch:    {arch}")
    eprint(f"  layers:  {n_layer} trunk + {nextn_layers} MTP = {n_layer_all} total")
    eprint(f"  n_embd:  {n_embd}, n_ff: {n_ff}")
    eprint(f"  tensors: {len(reader.tensors)}")
    eprint()

    eprint(f"Converting {len(moe_layers)} layers to MoE "
           f"({args.num_experts} experts, top-{args.top_k})")
    eprint(f"  Shared expert: {args.shared_expert}")
    eprint(f"  Expert FFN size: {expert_ff}")
    eprint(f"  Output type: {args.output_type}")
    eprint(f"  Noise scale: {args.noise_scale}")
    eprint()

    if args.dry_run:
        eprint("Dry run -- no output file written.")
        return

    # Build index of tensors by name for fast lookup
    tensors_by_name = {t.name: t for t in reader.tensors}

    # --- Write output ---
    eprint(f"Writing {args.output} ...")
    writer = GGUFWriter(args.output, MOE_ARCH, use_temp_file=True)

    # Metadata
    eprint("Writing metadata...")
    copy_metadata(reader, writer, MOE_ARCH, args, n_ff, n_embd, n_layer)

    # Tensors -- iterate in the same order as the input GGUF
    eprint("Writing tensors...")

    # Track which FFN tensors have been handled as part of a block
    handled = set()
    total = len(reader.tensors)

    for idx, tensor in enumerate(reader.tensors):
        name = tensor.name

        if idx % 20 == 0 or idx == total - 1:
            eprint(f"  [{idx+1}/{total}] {name}")

        # Determine block index if this is a block tensor
        blk_idx = None
        if name.startswith("blk."):
            parts = name.split(".")
            try:
                blk_idx = int(parts[1])
            except (IndexError, ValueError):
                pass

        # Check if it's an FFN tensor in a MoE layer
        is_ffn = (
            blk_idx is not None and
            blk_idx in moe_layers and
            any(f".{sub}.weight" in name for sub in FFN_SUBTENSORS)
        )

        if is_ffn and name not in handled:
            # Convert the whole block's FFN at once on the first FFN tensor we hit
            eprint(f"  -> converting block {blk_idx} FFN to MoE")
            convert_ffn_block(
                writer, tensors_by_name,
                blk_idx, n_ff, n_embd,
                args.num_experts, args.noise_scale, args.shared_expert,
                out_np_dtype, out_qtype, expert_ff, rng,
            )
            # Mark all three FFN tensors as handled
            for sub in FFN_SUBTENSORS:
                handled.add(f"blk.{blk_idx}.{sub}.weight")
        elif name in handled:
            # Already written as part of FFN block conversion above
            continue
        else:
            # Pass through unchanged
            passthrough_tensor(writer, tensor)

    # Finalize
    eprint("Finalizing GGUF...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    eprint(f"Done. Output: {args.output} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
