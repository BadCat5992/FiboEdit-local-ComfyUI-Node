# ComfyUI Fibo Edit Node (Real 8B)

A custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that wraps the official **[briaai/Fibo-Edit](https://huggingface.co/briaai/Fibo-Edit)** model using standard Hugging Face Diffusers.

> [!CAUTION]
> **Hardware Requirement**: This node loads an ~8 Billion parameter model. It requires **16GB+ VRAM** (24GB recommended) to run effectively.
> **Authentication**: You MUST have a Hugging Face account and accept the license for `briaai/Fibo-Edit`.

## Features

- **Genuine 8B Model**: Uses the actual Bria Fibo Edit model, not an SD1.5 approximation.
- **Diffusers Backend**: Leverages `trust_remote_code=True` to load the exact pipeline from Hugging Face.
- **Simple Interface**: One node to rule them all. Input Image + Mask + Prompt -> Edited Image.

## Installation

1.  **Clone the repo**:
    ```bash
    cd ComfyUI/custom_nodes/
    git clone https://github.com/YOUR_USERNAME/comfyui-fibo-edit
    ```

2.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Hugging Face Login** (REQUIRED):
    You must be logged in to download the model.
    ```bash
    huggingface-cli login
    # Paste your HF Token with permissions for briaai/Fibo-Edit
    ```

## Usage

1.  **Load Image**: Connect your source image.
2.  **Selection**: Use a Mask node (e.g., "Load Image (as Mask)" or MaskEditor) to define the area to edit.
3.  **Fibo Node**:
    - Search for `Fibo Edit (8B Model)`.
    - Connect `image` and `mask`.
    - Enter your `prompt` (e.g., "A robot arm").
4.  **Run**:
    - **First Run**: Will take a long time to download the ~16GB+ model weights to your Hugging Face cache.
    - **Subsequent Runs**: Will load from cache (still takes time to move to GPU).

## Inputs

| Input | Description |
| :--- | :--- |
| `image` | Source image to edit. |
| `mask` | Binary mask defining the region to change. |
| `prompt` | Text description of the desired content. |
| `seed` | Random seed for reproducibility. |
| `steps` | Inference steps (default 30). |
| `guidance_scale` | CFG Scale (default 5.0). |
| `precision` | `fp16` (faster, less VRAM) or `bf16`/`fp32`. |

## Troubleshooting

- **OOM (Out Of Memory)**: The model is too big for your GPU. Try `precision="fp16"`. If still crashing, you might not have enough VRAM.
- **"Repository Not Found" / 401 Error**: You are not logged in or haven't accepted the model license on Hugging Face.
