# ComfyUI Fibo Local Edit Node

A production-ready Custom Node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that implements **FIBO-style localized image editing**.

This node operates fully locally (no external APIs) and is designed to wrap the complexity of:
1.  **VAE Encoding** the source image.
2.  **Processing and Blurring Masks** for seamless blending.
3.  **Latent Masking**.
4.  **K-Sampling** with configurable schedulers.
5.  **VAE Decoding**.

...into a single, easy-to-use node.

## Features

- **Standard ComfyUI Inputs**: Works with standard `MODEL`, `VAE`, `CLIP`, `IMAGE`, and `MASK` types.
- **Localized Editing**: Accepts an optional binary mask to constrain edits to specific regions.
- **Soft Blending**: Built-in Gaussian blur for the mask (`blur_mask`) ensures seamless transitions between original and edited content.
- **Full Control**: Exposes all standard KSampler parameters (`steps`, `cfg`, `sampler`, `scheduler`, `denoise`).
- **Offline & Private**: No cloud APIs. Uses your local GPU and models.

## Installation

1.  Clone this repository into your `ComfyUI/custom_nodes/` folder:
    ```bash
    cd ComfyUI/custom_nodes/
    git clone https://github.com/YOUR_USERNAME/comfyui-fibo-local-edit
    ```
2.  Install requirements (usually satisfied by standard ComfyUI install):
    ```bash
    pip install -r requirements.txt
    ```
3.  Restart ComfyUI.

## Usage

1.  **Load Checkpoint**: Connect your preferred SD1.5 or SDXL checkpoint.
2.  **Load Image**: Connect the source image you want to edit.
3.  **Mask (Optional)**:
    - Connect a `Load Image (as Mask)` node.
    - OR use right-click "Open in MaskEditor" on the `Load Image` node and connect its mask output.
4.  **Prompts**: Connect `CLIP Text Encode` nodes for Positive and Negative prompts.
5.  **Connect Fibo Node**:
    - **Search**: Double-click and type `Fibo Local Edit`.
    - **Connect**: Wire up Model, VAE, Image, Positive, Negative, and optional Mask.
6.  **Parameters**:
    - `blur_mask`: Adjusts edge softness (default 10). Higher values = smoother blend.
    - `denoise`: Controls how much the image changes (0.0 = no change, 1.0 = total dream). For editing, try 0.6-0.8.
7.  **Run**: Execute the workflow.

## Inputs Reference

| Input | Type | Description |
| :--- | :--- | :--- |
| `model` | MODEL | Stable Diffusion model (SD1.5 or SDXL) |
| `vae` | VAE | VAE matching the model |
| `image` | IMAGE | Source image to edit |
| `positive` | CONDITIONING | Output from CLIP Text Encode (what you want) |
| `negative` | CONDITIONING | Output from CLIP Text Encode (what you don't want) |
| `mask` | MASK | (Optional) Greyscale mask defining the edit area |
| `blur_mask` | INT | Radius of Gaussian blur applied to the mask |
| `denoise` | FLOAT | Strength of the edit (img2img strength) |

## Example Workflow

*(Screenshot would go here)*

1.  Load Image: A photo of a dog.
2.  Mask: Paint over the dog's eyes.
3.  Prompt: "blue sunglasses".
4.  Fibo Node: Connect everything.
5.  Result: Dog wearing blue sunglasses, rest of image unchanged.
