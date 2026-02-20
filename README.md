# ComfyUI Fibo Edit Node (Real 8B)

A custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that wraps the official **[briaai/Fibo-Edit](https://huggingface.co/briaai/Fibo-Edit)** model (8B params).

> [!IMPORTANT]
> `BriaFiboEditPipeline` was merged into diffusers in Jan 2026 (PR #12930) but is **not yet in any PyPI release**. The install script installs diffusers directly from the `main` branch via git.

> [!CAUTION]
> **Hardware:** ~16 GB+ GPU VRAM required (24 GB recommended).  
> **Auth:** You must accept the model license at [huggingface.co/briaai/Fibo-Edit](https://huggingface.co/briaai/Fibo-Edit).

---

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/BadCat5992/FiboEdit-local-ComfyUI-Node
cd FiboEdit-local-ComfyUI-Node
bash install.sh
huggingface-cli login
```

Then restart ComfyUI.

---

## How It Works

```
Plain text instruction
        │
        ▼
  VGL JSON (built automatically)
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
| `instruction` | STRING | Plain English edit instruction |
| `mask` | MASK | *(Optional)* White = edit, black = preserve |
| `seed` | INT | Random seed |
| `steps` | INT | Inference steps (default 50) |
| `guidance_scale` | FLOAT | CFG scale (default 5.0) |
| `precision` | COMBO | `bf16` (default), `fp16`, `fp32` |

---

## Usage

1. Search **`Fibo Edit (8B Model)`** in the node browser.
2. Connect `Load Image` → `image` and optionally a mask.
3. Type your instruction (e.g. "make it look vintage").
4. Queue Prompt.

> First run downloads model weights (~20 GB).

---

## Troubleshooting

| Error | Solution |
| :--- | :--- |
| `cannot import BriaFiboEditPipeline` | `pip install --upgrade diffusers` (need >= 0.33.0) |
| `401 / Repository Not Found` | `huggingface-cli login` + accept model license |
| CUDA out of memory | Use `bf16`, close other GPU apps |
