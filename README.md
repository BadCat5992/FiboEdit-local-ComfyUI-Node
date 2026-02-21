# ComfyUI Fibo Edit Node (Real 8B)

A custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that wraps the official **[briaai/Fibo-Edit](https://huggingface.co/briaai/Fibo-Edit)** model (8B params).

> [!IMPORTANT]
> `BriaFiboEditPipeline` was merged into diffusers in Jan 2026 (PR #12930) but is **not yet in any PyPI release**. The install script installs diffusers directly from the `main` branch via git.

> [!TIP]
> **🚀 16GB VRAM Support:** We've extensively optimized this node to solve "CUDA out of memory" (OOM) errors! It now features an automatic hardware fallback that safely runs the massive 8B model on **16 GB VRAM GPUs** (like the RTX 4080 or RTX 3080 16GB) without crashing.

> [!CAUTION]
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
| `offload_mode` | COMBO | `sequential_cpu_offload` (vram saver), `model_cpu_offload`, `none` |
| `unload_after_gen`| BOOLEAN | Purges RAM/VRAM after generation (default False) |

---

## 💻 Low VRAM Optimization (16GB GPUs)

If you are searching for *"How to run briaai Fibo Edit with 16GB VRAM in ComfyUI"* or encountering *"CUDA Out of Memory Allocation on device 0"* with the 8B model, this node handles it for you.

By default, the node intercepts memory settings on 16GB cards to prevent PyTorch crashes:
1. It automatically drops ComfyUI caches (`unload_all_models()`) before loading Diffusers.
2. It defaults to **`sequential_cpu_offload`** which drastically reduces VRAM spikes by paging model layers dynamically to GPU memory.
3. Keep `precision` on `bf16` for maximum compatibility on consumer cards.

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
| CUDA out of memory (OOM) | Ensure `offload_mode` is `sequential_cpu_offload` and `precision` is `bf16`. Enable `unload_after_gen`. |
