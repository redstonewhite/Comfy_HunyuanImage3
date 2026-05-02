"""
HunyuanImage-3.0-Instruct-Distil FP8 Quantization for FlashInfer
=================================================================

Quantizes expert MoE weights to FP8 (float8_e4m3fn) with per-tensor scales,
producing a model compatible with flashinfer's trtllm_fp8_per_tensor_scale_moe
kernel on Blackwell GPUs.

Following Eric's recipe: only expert weights are quantized.
Everything else (attention, gate/router, shared_mlp, DiT components, VAE,
vision encoder, embeddings) stays bf16.

Usage:
    conda run -n comfyui python hunyuan_quantize_fp8_flashinfer.py \
        --model_path tencent/HunyuanImage-3.0-Instruct-Distil \
        --output_path /path/to/output

Memory: ~10-15 GB peak RAM (shard-by-shard processing).
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# FP8 E4M3 max representable value
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

# Pattern matching expert weight tensors to quantize
EXPERT_WEIGHT_PATTERN = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_and_up_proj|down_proj)\.weight$"
)


def quantize_tensor_to_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a bf16/fp16 tensor to FP8 (e4m3fn) with per-tensor scale.

    Returns:
        (quantized_fp8, scale) where scale is a scalar float32 tensor.
        Dequantization: original ≈ quantized_fp8.float() * scale
    """
    # Compute per-tensor scale: scale = amax / FP8_MAX
    amax = tensor.detach().abs().max().float()
    # Prevent division by zero
    scale = amax / FP8_E4M3_MAX
    scale = scale.clamp(min=1e-12)

    # Quantize: divide by scale, clamp to FP8 range, cast
    scaled = (tensor.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    quantized = scaled.to(torch.float8_e4m3fn)

    return quantized, scale.unsqueeze(0)  # scale as [1] tensor for safetensors


def resolve_model_path(model_path: str) -> Path:
    """Resolve model path - handles HF hub cache or local paths."""
    p = Path(model_path)
    if p.exists():
        return p

    # Try HF cache
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    hf_dir = cache_dir / f"models--{model_path.replace('/', '--')}"
    if hf_dir.exists():
        snapshots = hf_dir / "snapshots"
        if snapshots.exists():
            # Get the latest snapshot
            snap_dirs = sorted(snapshots.iterdir())
            if snap_dirs:
                return snap_dirs[-1]

    raise FileNotFoundError(
        f"Model not found at '{model_path}' or in HF cache. "
        f"Checked: {p}, {hf_dir}"
    )


def get_shard_files(model_dir: Path) -> list[Path]:
    """Get all model shard files in order."""
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if not shards:
        # Single file model
        single = model_dir / "model.safetensors"
        if single.exists():
            return [single]
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")
    return shards


def process_shard(
    shard_path: Path,
    output_path: Path,
    shard_idx: int,
    total_shards: int,
) -> dict[str, str]:
    """
    Process a single shard: quantize expert weights, copy everything else.

    Returns:
        Dict mapping tensor names to output shard filename.
    """
    shard_name = shard_path.name
    logger.info(f"Processing shard {shard_idx + 1}/{total_shards}: {shard_name}")

    # Load the shard
    tensors = load_file(str(shard_path))
    output_tensors = {}
    weight_map = {}

    quantized_count = 0
    kept_count = 0

    for name, tensor in tensors.items():
        if EXPERT_WEIGHT_PATTERN.match(name):
            # Quantize this expert weight to FP8
            fp8_tensor, scale = quantize_tensor_to_fp8(tensor)
            output_tensors[name] = fp8_tensor
            # Save scale alongside with a naming convention
            scale_name = name.replace(".weight", ".weight_scale")
            output_tensors[scale_name] = scale
            weight_map[name] = shard_name
            weight_map[scale_name] = shard_name
            quantized_count += 1
        else:
            # Keep as-is
            output_tensors[name] = tensor
            weight_map[name] = shard_name
            kept_count += 1

    # Save output shard
    out_shard = output_path / shard_name
    save_file(output_tensors, str(out_shard))

    # Report sizes
    in_size = shard_path.stat().st_size / (1024**3)
    out_size = out_shard.stat().st_size / (1024**3)
    logger.info(
        f"  Shard {shard_idx + 1}: {quantized_count} tensors quantized, "
        f"{kept_count} kept. Size: {in_size:.2f} GB → {out_size:.2f} GB"
    )

    return weight_map


def create_quantization_config() -> dict:
    """Create quantization config metadata."""
    return {
        "quant_method": "fp8_flashinfer",
        "quant_type": "float8_e4m3fn",
        "scale_type": "per_tensor",
        "target_kernel": "trtllm_fp8_per_tensor_scale_moe",
        "quantized_modules": ["model.layers.*.mlp.experts.*.gate_and_up_proj", "model.layers.*.mlp.experts.*.down_proj"],
        "skip_modules": [
            "vae", "vision_encoder", "patch_embed", "final_layer",
            "time_embed", "time_embed_2", "timestep_emb", "timestep_r_emb",
            "guidance_emb", "self_attn", "mlp.gate", "shared_mlp",
            "lm_head", "wte", "ln_f",
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Quantize HunyuanImage-3.0 expert weights to FP8 for FlashInfer"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="tencent/HunyuanImage-3.0-Instruct-Distil",
        help="Path to source model (local dir or HF repo id)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output directory for quantized model",
    )
    args = parser.parse_args()

    # Resolve paths
    model_dir = resolve_model_path(args.model_path)
    logger.info(f"Source model: {model_dir}")

    if args.output_path is None:
        # Default output next to source
        output_dir = model_dir.parent / (model_dir.name + "-FP8-FlashInfer")
    else:
        output_dir = Path(args.output_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Get shards
    shards = get_shard_files(model_dir)
    logger.info(f"Found {len(shards)} shard(s)")

    # Process each shard
    full_weight_map = {}
    t0 = time.time()

    for i, shard in enumerate(shards):
        shard_map = process_shard(shard, output_dir, i, len(shards))
        full_weight_map.update(shard_map)

    elapsed = time.time() - t0
    logger.info(f"Quantization completed in {elapsed:.1f}s")

    # Create and save weight index
    # Load original index to get metadata
    orig_index_path = model_dir / "model.safetensors.index.json"
    if orig_index_path.exists():
        with open(orig_index_path) as f:
            orig_index = json.load(f)
        metadata = orig_index.get("metadata", {})
    else:
        metadata = {}

    # Compute new total size
    total_size = 0
    for shard in shards:
        out_shard = output_dir / shard.name
        if out_shard.exists():
            total_size += out_shard.stat().st_size

    new_index = {
        "metadata": {
            **metadata,
            "total_size": total_size,
            "quantization": "fp8_e4m3fn_per_tensor",
        },
        "weight_map": full_weight_map,
    }

    index_path = output_dir / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(new_index, f, indent=2)
    logger.info(f"Saved weight index: {index_path}")

    # Copy config and other necessary files
    files_to_copy = [
        "config.json",
        "configuration_hunyuan_image_3.py",
        "modeling_hunyuan_image_3.py",
        "tokenizer_config.json",
        "tokenizer.json",
        "generation_config.json",
        "cache_utils.py",
        "autoencoder_kl_3d.py",
        "hunyuan_image_3_pipeline.py",
        "image_processor.py",
        "siglip2.py",
        "system_prompt.py",
        "tokenization_hunyuan_image_3.py",
        "__init__.py",
        "README.md",
    ]
    for fname in files_to_copy:
        src = model_dir / fname
        if src.exists():
            dst = output_dir / fname
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
                logger.info(f"Copied: {fname}")

    # Copy utils directory if it exists
    utils_src = model_dir / "utils"
    utils_dst = output_dir / "utils"
    if utils_src.exists() and not utils_dst.exists():
        shutil.copytree(str(utils_src), str(utils_dst))
        logger.info("Copied: utils/")

    # Update config.json with quantization info
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        config["quantization_config"] = create_quantization_config()
        config["torch_dtype"] = "float8_e4m3fn"

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Updated config.json with quantization info")

    # Print summary
    in_total = sum(s.stat().st_size for s in shards) / (1024**3)
    out_total = sum(
        (output_dir / s.name).stat().st_size
        for s in shards
        if (output_dir / s.name).exists()
    ) / (1024**3)

    logger.info("=" * 60)
    logger.info("QUANTIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Input model size:  {in_total:.2f} GB")
    logger.info(f"Output model size: {out_total:.2f} GB")
    logger.info(f"Compression ratio: {in_total / out_total:.2f}x")
    logger.info(f"Expert weights: FP8 (float8_e4m3fn) with per-tensor scales")
    logger.info(f"Other weights:  bf16 (unchanged)")
    logger.info(f"Target kernel:  flashinfer.fused_moe.trtllm_fp8_per_tensor_scale_moe")
    logger.info(f"Output path:    {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
