import torch
import numpy as np
from PIL import Image
import nodes
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.utils import load_image

class FiboEditReal:
    """
    A ComfyUI custom node that wraps the official `briaai/Fibo-Edit` model
    using Hugging Face Diffusers.
    """
    
    _pipeline = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "A photo of..."}),
                "mask": ("MASK",),  # ComfyUI MASK type
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_fibo"
    CATEGORY = "Fibo/Edit"

    def apply_fibo(self, image, prompt, mask, seed, steps, guidance_scale, precision, negative_prompt=""):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load Pipeline (Lazy Loading)
        if FiboEditReal._pipeline is None:
            print("Loading briaai/Fibo-Edit pipeline... This may take a while.")
            
            torch_dtype = torch.float16 if precision == "fp16" else (torch.bfloat16 if precision == "bf16" else torch.float32)
            
            # Load the pipeline with trust_remote_code=True to get the BriaFiboEditPipeline
            FiboEditReal._pipeline = DiffusionPipeline.from_pretrained(
                "briaai/Fibo-Edit",
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            FiboEditReal._pipeline.to(device)
        
        pipe = FiboEditReal._pipeline

        # 2. Prepare Inputs
        # Convert Comfy IMAGE (B, H, W, C) -> PIL
        # We process batch size 1 for now (or loop if B > 1)
        batch_results = []
        
        for i in range(image.shape[0]):
            img_tensor = image[i] # (H, W, C)
            img_pil = Image.fromarray(np.clip(255. * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8))
            
            # Convert Comfy MASK (H, W) -> PIL
            # If mask is batched (B, H, W), take i-th
            if len(mask.shape) == 3:
                msk_tensor = mask[i]
            else:
                msk_tensor = mask # Single mask for all
            
            msk_pil = Image.fromarray(np.clip(255. * msk_tensor.cpu().numpy(), 0, 255).astype(np.uint8))

            # 3. Predict
            generator = torch.Generator(device).manual_seed(seed)
            
            # The exact call signature depends on the Bria pipeline. 
            # Usually: pipe(prompt, image=source, mask_image=mask, ...)
            # Checking common patterns for instruct-edit / inpaint pipelines.
            # Assuming standard Diffusers Edit/Inpaint signature or Bria specific.
            # Bria Fibo Edit often uses 'generate' or standard __call__.
            
            result = pipe(
                prompt=prompt,
                image=img_pil,
                mask_image=msk_pil,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
            
            # 4. Post-process (PIL -> Tensor)
            res_tensor = torch.from_numpy(np.array(result).astype(np.float32) / 255.0).unsqueeze(0)
            batch_results.append(res_tensor)

        return (torch.cat(batch_results, dim=0),)
