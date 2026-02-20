# ComfyUI Fibo Edit Node (Real 8B)

A custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that wraps the official **[briaai/Fibo-Edit](https://huggingface.co/briaai/Fibo-Edit)** model (8B params) using its native pipeline class.

> [!IMPORTANT]
> `BriaFiboEditPipeline` is **not** part of the standard `diffusers` PyPI package. It lives in the [Bria-AI/Fibo-Edit](https://github.com/Bria-AI/Fibo-Edit) GitHub repo, which this node clones automatically into `vendor/Fibo-Edit/`.

> [!CAUTION]
> **Hardware**: Requires ~16GB+ GPU VRAM (24GB recommended).  
> **Auth**: You must accept the model license on HuggingFace for `briaai/Fibo-Edit` and `briaai/FIBO-edit-prompt-to-JSON`.

---

## Installation

**Step 1 — Clone this node** into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/BadCat5992/FiboEdit-local-ComfyUI-Node
cd comfyui-fibo-edit
```

**Step 2 — Run install script** (downloads Fibo-Edit source + Python deps):
```bash
bash install.sh
```

This clones `https://github.com/Bria-AI/Fibo-Edit` into `vendor/Fibo-Edit/` and installs all required packages.

**Step 3 — Hugging Face login** (required to download model weights):
```bash
huggingface-cli login
```
Accept the license for both:
- [`briaai/Fibo-Edit`](https://huggingface.co/briaai/Fibo-Edit)
- [`briaai/FIBO-edit-prompt-to-JSON`](https://huggingface.co/briaai/FIBO-edit-prompt-to-JSON)

**Step 4 — Restart ComfyUI**.

---

## How It Works

```
Plain text instruction
        │
        ▼
 briaai/FIBO-edit-prompt-to-JSON  (local VLM, no API key)
        │
        ▼
  Structured VGL JSON prompt
        │
  + input image + optional mask
        │
        ▼
 briaai/Fibo-Edit (BriaFiboEditPipeline, 8B)
        │
        ▼
  Edited image
```

---

## Inputs

| Input | Type | Description |
| :--- | :--- | :--- |
| `image` | IMAGE | Source image to edit |
| `instruction` | STRING | Plain English edit instruction (e.g. "make it look vintage") |
| `mask` | MASK | *(Optional)* White = edit region, black = preserve |
| `seed` | INT | Random seed |
| `steps` | INT | Inference steps (default 30) |
| `guidance_scale` | FLOAT | CFG scale (default 5.0) |
| `precision` | COMBO | `bf16` (default), `fp16`, `fp32` |

---

## Usage in ComfyUI

1. Search: `Fibo Edit (8B Model)` in the node browser.
2. Connect: `Load Image` → `image`, `Load Image (as Mask)` → `mask`.
3. Type your instruction.
4. Queue Prompt.

> **First run** downloads model weights from Hugging Face (~20 GB). Subsequent runs use the local cache.

---

## Troubleshooting

| Error | Solution |
| :--- | :--- |
| `vendor/Fibo-Edit not found` | Run `bash install.sh` |
| `401 / Repository Not Found` | Run `huggingface-cli login` and accept the model license |
| CUDA out of memory | Use `bf16`, close other GPU apps, need 16GB+ VRAM |
| `ModuleNotFoundError: src.edit_promptify` | Ensure you ran `bash install.sh` and the `vendor/Fibo-Edit` folder exists |
