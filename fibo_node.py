import torch
import torch.nn.functional as F
import comfy.sd
import comfy.utils
import comfy.model_management
import nodes

class FiboLocalEdit:
    """
    A ComfyUI custom node that implements localized semantic editing (FIBO-style)
    using standard ComfyUI latent operations.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "positive": ("CONDITIONING",),  # Expecting conditioning (CLIP encoded)
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_mask": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_fibo_edit"
    CATEGORY = "Fibo/Edit"

    def apply_fibo_edit(self, model, vae, image, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, blur_mask, mask=None):
        # 1. VAE Encode the input image
        # Standard VAEEncode logic from nodes.py
        # Image is (B, H, W, C)
        pixels = image
        pixels = pixels.movedim(-1, 1)  # (B, C, H, W)
        # Ensure VAE device
        
        # We use the standard VAEEncode wrapper logic if possible, but directly calling vae.encode is cleaner here
        # to avoid instantiating a whole new node class for a sub-operation.
        # But we must respect ComfyUI's model management.
        
        # Helper to encode
        t_vae = vae.encode(pixels)
        latents = {"samples": t_vae}

        # 2. Handle Mask
        if mask is not None:
            # Mask is (B, H, W) or (H, W). Ensure (B, H, W)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            
            # Resize mask to match image
            # If mask batch size < image batch size, repeat
            if mask.shape[0] < image.shape[0]:
                mask = mask.repeat(image.shape[0], 1, 1)
            
            # Blur the mask for better blending (FIBO feature)
            if blur_mask > 0:
                # Gaussian blur implementation using torch
                # Create a gaussian kernel
                kernel_size = blur_mask * 2 + 1
                sigma = blur_mask / 3.0
                channels = 1
                
                # Create 1D Gaussian kernel
                x_coord = torch.arange(kernel_size)
                x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
                y_grid = x_grid.t()
                xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
                
                mean = (kernel_size - 1) / 2.
                variance = sigma**2.
                
                # Calculate the 2-D gaussian kernel
                gaussian_kernel = (1./(2.*3.14159*variance)) * \
                                  torch.exp(
                                      -torch.sum((xy_grid - mean)**2., dim=-1) / \
                                      (2*variance)
                                  )
                
                # Make it sum to 1
                gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
                
                # Reshape to 4D tensor (C_out, C_in, H, W) -> (1, 1, H, W)
                gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
                gaussian_kernel = gaussian_kernel.to(mask.device)
                
                # Pad input to keep dimensions same
                pad = kernel_size // 2
                
                # Apply blur
                # Ensure mask has channel dim for conv2d: (B, 1, H, W)
                m_in = mask.unsqueeze(1)
                m_out = F.conv2d(m_in, gaussian_kernel, padding=pad, groups=1)
                mask = m_out.squeeze(1)

            # Set the noise mask in the latents
            # This follows SetLatentNoiseMask logic
            latents["noise_mask"] = mask

        # 3. KSampler
        # We reuse the standard nodes.KSampler logic by delegation or re-impl
        # Using nodes.common_ksampler allows us to leverage the full ComfyUI sampler stack
        # common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0)
        
        # IMPORTANT: If denoise < 1.0 and we have a mask, standard KSampler might act as "masked img2img"
        # which preserves unmasked content effectively.
        
        samples = nodes.common_ksampler(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latents,
            denoise=denoise
        )
        
        # 4. Decode
        # samples[0] contains the latent dict
        decoded = vae.decode(samples[0]["samples"])
        
        return (decoded,)

