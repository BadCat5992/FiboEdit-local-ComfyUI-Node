#!/usr/bin/env bash
# Run this ONCE from the root of the custom node folder
# (ComfyUI/custom_nodes/comfyui-fibo-edit/)
# to download the official Fibo-Edit source code.
set -e

echo ">>> Cloning briaai/Fibo-Edit source repo into vendor/Fibo-Edit ..."
git clone https://github.com/Bria-AI/Fibo-Edit vendor/Fibo-Edit

echo ">>> Installing Python dependencies ..."
pip install -r requirements.txt

# Install any extra deps from the Fibo-Edit repo itself
if [ -f vendor/Fibo-Edit/pyproject.toml ]; then
    pip install -e vendor/Fibo-Edit --no-deps
fi

echo ""
echo "=== Done. Restart ComfyUI now. ==="
echo "You must also be logged in to Hugging Face:"
echo "  huggingface-cli login"
