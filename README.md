<div align="center">

# ComfyUI Fibo Edit Node (8B) 
**The Ultimate 16GB VRAM Optimized ComfyUI Custom Node for Bria Fibo-Edit**

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Extension-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![HuggingFace](https://img.shields.io/badge/Model-Bria%20Fibo--Edit-yellow.svg)](https://huggingface.co/briaai/Fibo-Edit)
[![GPU](https://img.shields.io/badge/VRAM-16GB%20Optimized-green.svg)]()

</div>

A powerful, custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that directly wraps the official **[briaai/Fibo-Edit](https://huggingface.co/briaai/Fibo-Edit)** 8-Billion parameter Image Editing model. 

*Are you struggling with `CUDA out of memory` errors while trying to run Fibo Edit locally?* This node is engineered from the ground up to **solve OOM (Out of Memory) crashes on 16GB and 12GB GPUs** via advanced memory offloading and automated garbage collection.

---

> [!IMPORTANT]
> **Diffusers `main` Branch Required:** `BriaFiboEd24GB+ cards.<br>`none`: Fastest, but requires insane VRAM (>32GB). |
| `unload_after_gen`| `BOOLEAN` | If `True`, completely destroys the model from System RAM & VRAM after generating, freeing ~16GB instantly. |itPipeline` was merged into Diffusers in Jan 2026 (PR #12930), but is **not yet in any major PyPI release**. The included install script automatically handles pulling diffusers directly from GitHub so you can use this state-of-the-art editor today.

> [!CAUTION]
> **HuggingFace Auth Required:** To download the 20GB checkpoint, you must accept the official model license at [huggingface.co/briaai/Fibo-Edit](https://huggingface.co/briaai/Fibo-Edit) and authenticate your machine using `huggingface-cli login`.

---

## 🚀 Key Features

*   **Native 8B Model Support:** Uses the massive, high-quality Bria Fibo-Edit checkpoint without relying on third-party APIs.
*   **Plain Text Instructions:** No need for complex prompt engineering. Just type "make the car blue" or "turn the background into a cyberpunk city".
*   **VGL JSON Auto-Formatting:** The node intercepts your plain English and automatically structures it into the required VGL JSON format under the hood.
*   **Mask Support:** Pass optional black/white masks to restrict the edit to specific areas.

## 💾 The 16GB VRAM Solution (Extreme Low-VRAM Optimization)

Search engines are flooded with questions like *"How to run briaai Fibo Edit with 16GB VRAM"*, *"Fibo Edit CUDA Out of Memory Allocation on device 0"*, or *"ComfyUI 8B Model OOM Crash"*.

By default, loading an 8B model in `bf16` precision requires nearly 16GB just for the weights alone, instantly crashing consumer graphics cards like the RTX 4080, RTX 3080 16GB, or RTX 4070 Ti. 

**How this repo fixes the problem:**
1.  **Smart Hardware Auto-Detection:** The moment you click `Queue Prompt`, the node detects your maximum `cuda` memory. If you have less than 18GB of VRAM, it **automatically intercepts and overrides** your ComfyUI node settings to engage survival mode.
2.  **Pre-Generation Cache Nuking:** It actively flushes out any massive un-freed models (like SDXL or FLUX) stored in ComfyUI's internal memory manager via `unload_all_models()`, guaranteeing a true 16GB blank slate.
3.  **Sequential CPU Offloading (`sequential_cpu_offload`):** Instead of shoving 16GB into your GPU, this mode dynamically streams the model layer-by-layer between your System RAM and your GPU VRAM. Your VRAM usage stays incredibly low during the entire generation process.
4.  **Auto RAM/VRAM Purge (`unload_after_gen`):** If enabled, the moment the image finishes, the pipeline runs aggressive Python garbage collection (`gc.collect()`) and wipes the PyTorch cache (`empty_cache()`), giving you back your 16GB instantly.

---

## ⚙️ Installation Guide

Follow these steps exactly to get running:

```bash
# 1. Navigate to your ComfyUI Custom Nodes folder
cd ComfyUI/custom_nodes/

# 2. Clone this repository
git clone https://github.com/BadCat5992/FiboEdit-local-ComfyUI-Node
cd FiboEdit-local-ComfyUI-Node

# 3. Run the dependency script (Installs Git Diffusers)
bash install.sh

# 4. Authenticate HuggingFace (CRITICAL)
huggingface-cli login
```

After pasting your read-access token, **Restart ComfyUI**.

---

## 🎛️ Node Inputs & Settings

Find the node by double-clicking the canvas and searching for **`Fibo Edit (8B Model)`**.

| Input Name | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | **[Required]** The source image you want to edit. |
| `instruction` | `STRING` | **[Required]** Plain English description of the edit (e.g., "Change the season to winter"). |
| `mask` | `MASK` | **[Optional]** A black/white mask. White = modify, Black = preserve. |
| `seed` | `INT` | Random seed for reproducible results. |
| `steps` | `INT` | Number of denoising steps. Default is 50. Higher = better quality, slower. |
| `guidance_scale`| `FLOAT` | CFG Scale determining how strongly the model follows the text. Default 5.0. |
| `precision` | `COMBO` | Floating point precision (`bf16`, `fp16`, `fp32`). Leave on `bf16` (Default) for 16GB cards. |
| `offload_mode` | `COMBO` | **Crucial for Low VRAM:** <br>`sequential_cpu_offload`: Streams layers (Prevents OOM on <16GB).<br>`model_cpu_offload`: Ideal for 24GB+ cards.<br>`none`: Fastest, but requires insane VRAM (>32GB). |
| `unload_after_gen`| `BOOLEAN` | If `True`, completely destroys the model from System RAM & VRAM after generating, freeing ~16GB instantly. |

---

## 🛠️ Typical Workflow

1.  Add a **Load Image** node.
2.  Add the **Fibo Edit (8B Model)** node.
3.  Add a **Preview Image** or **Save Image** node.
4.  Connect `IMAGE` output from Load Image into the `image` input of Fibo.
5.  Type your edit instruction.
6.  *(First run will freeze as it downloads ~20 GB of checkpoint variables to `~/.cache/huggingface/`)*.
7.  Click **Queue Prompt**.
 
---

## ❓ Troubleshooting & Common Errors

| Error Message | Meaning | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'diffusers'` or <br>`cannot import name 'BriaFiboEditPipeline'` | Your diffusers version is too old. | Run `bash install.sh` inside this node's folder, or manually run `pip install git+https://github.com/huggingface/diffusers.git`. |
| `401 / Repository Not Found` or <br>`Access to model is restricted` | HuggingFace rejected the download. | Go to [Fibo-Edit on HF](https://huggingface.co/briaai/Fibo-Edit), accept the license terms. Then open your terminal and run `huggingface-cli login` to save your token. |
| `CUDA out of memory.` | GPU VRAM is overflowing. | Ensure `offload_mode` is set to `sequential_cpu_offload`. Ensure your ComfyUI isn't already holding a massive model in VRAM (though the node attempts to clear this for you). |
| `NotImplementedError: mul_cuda not implemented for 'Float8'` | You attempted to force `fp8`. | Standard HF Transformers text-encoders do not natively support fp8 math. Keep `precision` set to `bf16` and rely on `sequential_cpu_offload` instead. |

---
*Created by [BadCat5992](https://github.com/BadCat5992)*
