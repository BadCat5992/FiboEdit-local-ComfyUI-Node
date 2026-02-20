#!/usr/bin/env bash
# Run this ONCE to set up the Fibo Edit node.
# Usage:  bash install.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Fibo Edit Node — Setup ==="
echo ""

# 1. Install diffusers from git main branch
#    BriaFiboEditPipeline was merged Jan 2026 (PR #12930) but is NOT in any
#    PyPI release yet (latest PyPI release is 0.36.0 from Dec 2025).
#    We MUST install from the main branch.
echo ">>> Installing diffusers from git main branch (required for BriaFiboEditPipeline) …"
pip install git+https://github.com/huggingface/diffusers.git

# 2. Install other dependencies
echo ""
echo ">>> Installing remaining dependencies …"
pip install transformers accelerate huggingface_hub sentencepiece

# 3. Verify BriaFiboEditPipeline import
echo ""
echo ">>> Verifying BriaFiboEditPipeline import …"
python -c "
from diffusers import BriaFiboEditPipeline
print('  BriaFiboEditPipeline import OK ✓')
" || {
    echo ""
    echo "  *** ERROR: BriaFiboEditPipeline still not found! ***"
    echo "  Try running manually:"
    echo "    pip install --force-reinstall git+https://github.com/huggingface/diffusers.git"
    exit 1
}

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. huggingface-cli login"
echo "  2. Accept the license at https://huggingface.co/briaai/Fibo-Edit"
echo "  3. Restart ComfyUI"
