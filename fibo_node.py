"""
ComfyUI Custom Node: Fibo Edit (Real 8B)
Wraps the official briaai/Fibo-Edit model via diffusers.

REQUIREMENTS:
  - diffusers >= 0.33.0 (BriaFiboEditPipeline was merged Jan 16 2026)
  - transformers, accelerate, sentencepiece
  - Hugging Face login + accepted license for briaai/Fibo-Edit

Install / upgrade:
  pip install --upgrade diffusers transformers accelerate sentencepiece
  huggingface-cli login
"""

import json
import torch
import numpy as np
from PIL import Image


def _check_diffusers_version():
    """Verify BriaFiboEditPipeline is available (requires diffusers from git main)."""
    try:
        from diffusers import BriaFiboEditPipeline  # noqa: F401
    except ImportError:
        raise ImportError(
            "BriaFiboEditPipeline is not available in your diffusers install.\n"
            "This class was merged into diffusers main branch (PR #12930, Jan 2026)\n"
            "but is NOT in any PyPI release yet.\n\n"
            "Fix:  pip install git+https://github.com/huggingface/diffusers.git\n"
            "Or run:  bash install.sh"
        )


def _build_vgl_json(instruction: str) -> str:
    """
    Build a minimal VGL-compatible JSON prompt from a plain-text instruction.

    The briaai/FIBO-edit-prompt-to-JSON local VLM model is currently unavailable
    (404 on HuggingFace). As a workaround we construct a simple VGL JSON that
    the pipeline can consume. The model is quite tolerant of this format.

    The official VGL spec uses structured fields for objects, lighting, camera,
    and aesthetics. For an edit instruction, the most important field is the
    description of the desired change.
    """
    vgl_prompt = {
        "description": instruction,
        "edit_instruction": instruction,
    }
    return json.dumps(vgl_prompt)


class FiboEditReal:
    """
    ComfyUI node that uses the real briaai/Fibo-Edit 8B model.

    Requires diffusers >= 0.33.0 which includes BriaFiboEditPipeline natively.
    No external repos or VLM models needed.
    """

    _pipeline = None  # Cached after first load

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image":          ("IMAGE",),
                "instruction":    ("STRING",  {"multiline": True, "default": "make it look vintage"}),
                "seed":           ("INT",     {"default": 0,    "min": 0,    "max": 0xffffffffffffffff}),
                "steps":          ("INT",     {"default": 50,   "min": 1,    "max": 150}),
                "guidance_scale": ("FLOAT",   {"default": 5.0,  "min": 0.0,  "max": 20.0, "step": 0.1}),
                "precision":      (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "offload_mode":   (["model_cpu_offload", "sequential_cpu_offload", "none"], {"default": "sequential_cpu_offload"}),
                "unload_after_gen": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    FUNCTION      = "apply_fibo"
    CATEGORY      = "Fibo/Edit"

    # ------------------------------------------------------------------
    # Pipeline loader
    # ------------------------------------------------------------------

    def _load_pipeline(self, precision: str, offload_mode: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if FiboEditReal._pipeline is not None:
            # Assumes offload_mode was configured upon initial load
            return FiboEditReal._pipeline

        _check_diffusers_version()

        # BriaFiboEditPipeline is a first-class pipeline in diffusers >= 0.33
        from diffusers import BriaFiboEditPipeline

        print("[Fibo Edit Node] Loading briaai/Fibo-Edit (8B) — first run downloads ~20 GB …")

        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(precision, torch.bfloat16)

        pipe = BriaFiboEditPipeline.from_pretrained(
            "briaai/Fibo-Edit",
            torch_dtype=torch_dtype,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if offload_mode == "model_cpu_offload":
            print("[Fibo Edit Node] Enabling model CPU offload...")
            pipe.enable_model_cpu_offload()
        elif offload_mode == "sequential_cpu_offload":
            print("[Fibo Edit Node] Enabling sequential CPU offload (extreme low VRAM)...")
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)

        FiboEditReal._pipeline = pipe
        print("[Fibo Edit Node] Pipeline loaded successfully.")
        return pipe

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def apply_fibo(self, image, instruction, seed, steps, guidance_scale, precision, offload_mode, unload_after_gen, mask=None):
        import comfy.model_management as mm
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda":
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_vram_gb < 18.0:
                print(f"[Fibo Edit Node] ⚠️ Detected limited VRAM ({total_vram_gb:.1f} GB). Forcing VRAM-safe settings!")
                precision = "bf16"
                offload_mode = "sequential_cpu_offload"

        print("[Fibo Edit Node] Forcing ComfyUI to unload its models to free VRAM for Diffusers pipeline...")
        mm.unload_all_models()
        mm.soft_empty_cache()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe   = self._load_pipeline(precision, offload_mode)

        # Build VGL JSON prompt from plain-text instruction
        edit_json = _build_vgl_json(instruction)
        print(f"[Fibo Edit Node] VGL prompt: {edit_json}")

        batch_results = []

        for i in range(image.shape[0]):

            # ── ComfyUI IMAGE tensor (H, W, C float 0-1) → PIL ────────
            img_np  = np.clip(255.0 * image[i].cpu().numpy(), 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            # ── Optional mask tensor → PIL (L mode) ───────────────────
            mask_pil = None
            if mask is not None:
                msk_t  = mask[i] if len(mask.shape) == 3 else mask
                msk_np = np.clip(255.0 * msk_t.cpu().numpy(), 0, 255).astype(np.uint8)
                mask_pil = Image.fromarray(msk_np).convert("L")

            # ── Run pipeline ──────────────────────────────────────────
            generator = torch.Generator(device).manual_seed(seed + i)

            call_kwargs = dict(
                image=img_pil,
                prompt=edit_json,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            if mask_pil is not None:
                call_kwargs["mask"] = mask_pil

            result_pil = pipe(**call_kwargs).images[0]

            # ── PIL → ComfyUI IMAGE tensor (1, H, W, C) ──────────────
            res_tensor = torch.from_numpy(
                np.array(result_pil).astype(np.float32) / 255.0
            ).unsqueeze(0)
            batch_results.append(res_tensor)

        if unload_after_gen:
            print("[Fibo Edit Node] Unloading pipeline to free RAM/VRAM...")
            del pipe
            FiboEditReal._pipeline = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        elif offload_mode != "none":
            # Free up lingering VRAM fragmentation
            torch.cuda.empty_cache()

        return (torch.cat(batch_results, dim=0),)
