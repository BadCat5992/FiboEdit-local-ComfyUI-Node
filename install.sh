#!/usr/bin/env bash
# Run this ONCE from the root of the custom node folder:
#   bash install.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="$SCRIPT_DIR/vendor/Fibo-Edit"

echo "=== Fibo Edit Node — One-time Setup ==="

# 1. Clone the Fibo-Edit source repo
if [ -d "$VENDOR_DIR" ]; then
    echo ">>> vendor/Fibo-Edit already exists, pulling latest …"
    git -C "$VENDOR_DIR" pull
else
    echo ">>> Cloning Bria-AI/Fibo-Edit …"
    git clone https://github.com/Bria-AI/Fibo-Edit "$VENDOR_DIR"
fi

# 2. Install the Fibo-Edit package in editable mode.
#    This registers BriaFiboEditPipeline inside diffusers
#    and makes `fibo_edit` importable.
echo ">>> Installing Fibo-Edit package (editable) …"
pip install -e "$VENDOR_DIR" --no-deps

# 3. Install this node's extra requirements
echo ">>> Installing node requirements …"
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "=== Done! ==="
echo ""
echo "Next steps:"
echo "  1. Log in to Hugging Face:  huggingface-cli login"
echo "  2. Accept the model license at https://huggingface.co/briaai/Fibo-Edit"
echo "  3. Accept the VLM license   at https://huggingface.co/briaai/FIBO-edit-prompt-to-JSON"
echo "  4. Restart ComfyUI"
