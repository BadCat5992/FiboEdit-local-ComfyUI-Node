"""
ComfyUI Custom Node: Fibo Edit (Real 8B)
Wraps the official briaai/Fibo-Edit model.

HOW IT WORKS:
  install.sh clones https://github.com/Bria-AI/Fibo-Edit into vendor/Fibo-Edit/
  and runs `pip install -e vendor/Fibo-Edit --no-deps`
  This registers BriaFiboEditPipeline into the local diffusers install
  and makes the `fibo_edit` Python package importable.

IMPORTS:
  from diffusers import BriaFiboEditPipeline   -> works after pip install -e
  from fibo_edit.edit_promptify import get_prompt -> the official promptifier
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Safety check: make sure the vendor repo is installed
# ---------------------------------------------------------------------------
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIBO_REPO_PATH = os.path.join(_NODE_DIR, "vendor", "Fibo-Edit")

if not os.path.isdir(_FIBO_REPO_PATH):
    print(
        "\n[Fibo Edit Node] *** ERROR: vendor/Fibo-Edit not found! ***\n"
        "  Run this ONCE from the custom node folder:\n"
        "    bash install.sh\n"
        "  Then restart ComfyUI.\n"
    )


class FiboEditReal:
    """
    ComfyUI node that uses the real briaai/Fibo-Edit 8B model.

    After running install.sh the Fibo-Edit repo is installed as an editable
    package which registers BriaFiboEditPipeline inside diffusers automatically.
    We then import it normally: `from diffusers import BriaFiboEditPipeline`.

    Prompt chain (fully offline):
      plain text  →  briaai/FIBO-edit-prompt-to-JSON (local VLM)  →  VGL JSON  →  Fibo-Edit
    """

    _pipeline  = None   # BriaFiboEditPipeline, lazy-loaded
    _get_prompt = None  # fibo_edit.edit_promptify.get_prompt, lazy-loaded

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image":          ("IMAGE",),
                "instruction":    ("STRING",  {"multiline": True,  "default": "make it look vintage"}),
                "seed":           ("INT",     {"default": 0,   "min": 0,   "max": 0xffffffffffffffff}),
                "steps":          ("INT",     {"default": 50,  "min": 1,   "max": 150}),
                "guidance_scale": ("FLOAT",   {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "precision":      (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
            "optional": {
                # Optional binary mask. White = area to edit, black = preserve.
                # NOTE: local VLM mode does not use the mask for JSON generation,
                #       but it IS passed to the diffusion pipeline for spatial control.
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    FUNCTION      = "apply_fibo"
    CATEGORY      = "Fibo/Edit"

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_pipeline(self, precision: str):
        if FiboEditReal._pipeline is not None:
            return FiboEditReal._pipeline

        # This import works only AFTER `pip install -e vendor/Fibo-Edit`
        # which registers the class in diffusers.
        try:
            from diffusers import BriaFiboEditPipeline
        except ImportError as e:
            raise ImportError(
                "BriaFiboEditPipeline not found in diffusers.\n"
                "Did you run `bash install.sh`? That script runs:\n"
                "  pip install -e vendor/Fibo-Edit --no-deps\n"
                f"Original error: {e}"
            )

        print("[Fibo Edit Node] Loading BriaFiboEditPipeline from briaai/Fibo-Edit …")

        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(precision, torch.bfloat16)

        pipe = BriaFiboEditPipeline.from_pretrained(
            "briaai/Fibo-Edit",
            torch_dtype=torch_dtype,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)

        FiboEditReal._pipeline = pipe
        print("[Fibo Edit Node] Pipeline ready.")
        return pipe

    def _load_get_prompt(self):
        if FiboEditReal._get_prompt is not None:
            return FiboEditReal._get_prompt

        try:
            from fibo_edit.edit_promptify import get_prompt
        except ImportError as e:
            raise ImportError(
                "Cannot import fibo_edit.edit_promptify.\n"
                "Did you run `bash install.sh`?\n"
                f"Original error: {e}"
            )

        FiboEditReal._get_prompt = get_prompt
        return get_prompt

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def apply_fibo(self, image, instruction, seed, steps, guidance_scale, precision, mask=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pipe       = self._load_pipeline(precision)
        get_prompt = self._load_get_prompt()

        batch_results = []

        for i in range(image.shape[0]):

            # ── Convert Comfy IMAGE tensor (H, W, C) → PIL ────────────
            img_np  = np.clip(255.0 * image[i].cpu().numpy(), 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            # ── Optionally convert mask tensor → PIL ───────────────────
            mask_pil = None
            if mask is not None:
                msk_t   = mask[i] if len(mask.shape) == 3 else mask
                msk_np  = np.clip(255.0 * msk_t.cpu().numpy(), 0, 255).astype(np.uint8)
                mask_pil = Image.fromarray(msk_np).convert("L")

            # ── Build VGL JSON prompt via local VLM (fully offline) ────
            # Local mode uses briaai/FIBO-edit-prompt-to-JSON on your GPU.
            # NOTE: local mode ignores the mask for JSON generation (Bria limitation).
            print("[Fibo Edit Node] Generating VGL JSON prompt via local VLM …")
            edit_json = get_prompt(
                image=img_pil,
                instruction=instruction,
                vlm_mode="local",
                model="briaai/FIBO-edit-prompt-to-JSON",
            )
            print(f"[Fibo Edit Node] VGL JSON (first 200 chars): {str(edit_json)[:200]}")

            # ── Run the edit pipeline ──────────────────────────────────
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

            # ── PIL → Comfy IMAGE tensor (1, H, W, C) ─────────────────
            res_tensor = torch.from_numpy(
                np.array(result_pil).astype(np.float32) / 255.0
            ).unsqueeze(0)
            batch_results.append(res_tensor)

        return (torch.cat(batch_results, dim=0),)
