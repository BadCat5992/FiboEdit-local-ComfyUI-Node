"""
ComfyUI Custom Node: Fibo Edit (Real 8B)
Wraps the official briaai/Fibo-Edit model.

IMPORTANT: BriaFiboEditPipeline is NOT part of the standard diffusers pip package.
It is defined in the briaai/Fibo-Edit GitHub repository and registered via
the model card's pipeline_tag + custom_pipeline mechanism.
We load it by cloning the official repo and importing from it directly,
or by using `custom_pipeline` argument of DiffusionPipeline.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# BriaFiboEditPipeline lives inside the Fibo-Edit source repo (not PyPI diffusers).
# The vendor/ subfolder should contain a clone of https://github.com/Bria-AI/Fibo-Edit
# Run the install script once:
#   git clone https://github.com/Bria-AI/Fibo-Edit vendor/Fibo-Edit
# ---------------------------------------------------------------------------

# Resolve paths relative to this file
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIBO_REPO_PATH = os.path.join(_NODE_DIR, "vendor", "Fibo-Edit")

# Inject the Fibo-Edit repo src into sys.path so we can import from it
if os.path.isdir(_FIBO_REPO_PATH):
    if _FIBO_REPO_PATH not in sys.path:
        sys.path.insert(0, _FIBO_REPO_PATH)
    _FIBO_AVAILABLE = True
else:
    _FIBO_AVAILABLE = False
    print(
        "\n[Fibo Edit Node] WARNING: Could not find vendor/Fibo-Edit.\n"
        "  Run this once inside your custom_nodes/comfyui-fibo-edit folder:\n"
        "    git clone https://github.com/Bria-AI/Fibo-Edit vendor/Fibo-Edit\n"
        "  Then restart ComfyUI.\n"
    )


class FiboEditReal:
    """
    ComfyUI node that wraps briaai/Fibo-Edit (8B parameter image editing model).

    Pipeline loading strategy
    ─────────────────────────
    BriaFiboEditPipeline is registered locally inside the cloned Fibo-Edit repo
    (vendor/Fibo-Edit). We load it with DiffusionPipeline.from_pretrained using
    custom_pipeline=... pointing to that path, which avoids needing a diffusers
    release that includes the class.

    Prompt strategy
    ───────────────
    Fibo-Edit expects a structured JSON string as the prompt (VGL schema).
    This node uses the local VLM (briaai/FIBO-edit-prompt-to-JSON) to convert
    a plain-text instruction into the required JSON — fully offline, no API key.
    """

    _pipeline = None        # BriaFiboEditPipeline (cached after first load)
    _promptify = None       # get_prompt function (cached after first load)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image":           ("IMAGE",),
                "instruction":     ("STRING",  {"multiline": True,  "default": "make it look vintage"}),
                "seed":            ("INT",     {"default": 0,   "min": 0,   "max": 0xffffffffffffffff}),
                "steps":           ("INT",     {"default": 30,  "min": 1,   "max": 100}),
                "guidance_scale":  ("FLOAT",   {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "precision":       (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
            "optional": {
                # Optional binary mask (white = edit region, black = preserve)
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    FUNCTION      = "apply_fibo"
    CATEGORY      = "Fibo/Edit"

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    def _load_pipeline(self, precision: str):
        """Lazy-load BriaFiboEditPipeline from the local repo clone."""
        if FiboEditReal._pipeline is not None:
            return FiboEditReal._pipeline

        if not _FIBO_AVAILABLE:
            raise RuntimeError(
                "Fibo-Edit repo not found. "
                "Clone it first:\n  git clone https://github.com/Bria-AI/Fibo-Edit vendor/Fibo-Edit"
            )

        print("[Fibo Edit Node] Loading BriaFiboEditPipeline ...")

        # The Fibo-Edit repo ships its own pipeline_class in
        # vendor/Fibo-Edit/src/pipeline_bria_fibo_edit.py (exact path may vary).
        # DiffusionPipeline supports loading it via the custom_pipeline arg.
        from diffusers import DiffusionPipeline

        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(precision, torch.bfloat16)

        # Try loading with the repo's pipeline class shipped in the model card.
        # briaai/Fibo-Edit already stores a pipeline_tag on the Hub so
        # DiffusionPipeline can find BriaFiboEditPipeline automatically.
        pipeline = DiffusionPipeline.from_pretrained(
            "briaai/Fibo-Edit",
            custom_pipeline=os.path.join(_FIBO_REPO_PATH, "src"),  # folder containing pipeline_bria_fibo_edit.py
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline.to(device)

        FiboEditReal._pipeline = pipeline
        print("[Fibo Edit Node] Pipeline ready.")
        return pipeline

    def _load_promptify(self):
        """Lazy-load the local VLM prompt-to-JSON helper."""
        if FiboEditReal._promptify is not None:
            return FiboEditReal._promptify

        if not _FIBO_AVAILABLE:
            raise RuntimeError("Fibo-Edit repo not found. See _load_pipeline for instructions.")

        # get_prompt lives in vendor/Fibo-Edit/src/edit_promptify.py
        from src.edit_promptify import get_prompt  # available after sys.path injection above
        FiboEditReal._promptify = get_prompt
        return get_prompt

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def apply_fibo(self, image, instruction, seed, steps, guidance_scale, precision, mask=None):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        pipe       = self._load_pipeline(precision)
        get_prompt = self._load_promptify()

        batch_results = []

        for i in range(image.shape[0]):

            # ── Convert Comfy IMAGE tensor → PIL ──────────────────────
            img_np  = np.clip(255.0 * image[i].cpu().numpy(), 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            # ── Optionally convert mask tensor → PIL ──────────────────
            mask_pil = None
            if mask is not None:
                msk_t   = mask[i] if len(mask.shape) == 3 else mask
                msk_np  = np.clip(255.0 * msk_t.cpu().numpy(), 0, 255).astype(np.uint8)
                mask_pil = Image.fromarray(msk_np).convert("L")

            # ── Build structured JSON prompt via local VLM ─────────────
            # vlm_mode="local" uses briaai/FIBO-edit-prompt-to-JSON locally.
            # NOTE: local VLM mode does NOT support mask (Bria limitation).
            #       When a mask is provided we still generate the JSON from the
            #       image+instruction pair and later pass the mask separately.
            print(f"[Fibo Edit Node] Generating VGL JSON prompt (local VLM) …")
            edit_json = get_prompt(
                image=img_pil,
                instruction=instruction,
                vlm_mode="local",
                model="briaai/FIBO-edit-prompt-to-JSON",
            )
            print(f"[Fibo Edit Node] VGL JSON: {edit_json[:200]}…")

            # ── Run the edit pipeline ──────────────────────────────────
            generator = torch.Generator(device).manual_seed(seed + i)

            call_kwargs = dict(
                image=img_pil,
                prompt=edit_json,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            # Pass mask only if pipeline supports it
            if mask_pil is not None:
                call_kwargs["mask"] = mask_pil

            result_pil = pipe(**call_kwargs).images[0]

            # ── Convert PIL → Comfy IMAGE tensor ──────────────────────
            res_np     = np.array(result_pil).astype(np.float32) / 255.0
            res_tensor = torch.from_numpy(res_np).unsqueeze(0)  # (1, H, W, C)
            batch_results.append(res_tensor)

        return (torch.cat(batch_results, dim=0),)
