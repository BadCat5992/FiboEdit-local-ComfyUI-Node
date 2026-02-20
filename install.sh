#!/usr/bin/env bash
# Run this ONCE to set up the Fibo Edit node.
# Usage:  bash install.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Fibo Edit Node — Setup ==="
echo ""

# 1. Install / upgrade Python packages
echo ">>> Installing requirements (upgrading diffusers to >= 0.33.0) …"
pip install --upgrade -r "$SCRIPT_DIR/requirements.txt"

# 2. Verify diffusers version
echo ""
echo ">>> Checking diffusers version …"
python -c "
import diffusers
v = diffusers.__version__
print(f'  diffusers version: {v}')
major, minor = [int(x) for x in v.split('.')[:2]]
if major == 0 and minor < 33:
    print('  ERROR: diffusers is still too old! Need >= 0.33.0')
    print('  Try:  pip install --upgrade diffusers --force-reinstall')
    exit(1)
else:
    print('  OK ✓')
"

# 3. Verify BriaFiboEditPipeline import
echo ""
echo ">>> Verifying BriaFiboEditPipeline import …"
python -c "
from diffusers import BriaFiboEditPipeline
print('  BriaFiboEditPipeline import OK ✓')
" || {
    echo "  ERROR: BriaFiboEditPipeline not found even after upgrade."
    echo "  Try:  pip install diffusers --force-reinstall --upgrade"
    exit 1
}

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. huggingface-cli login"
echo "  2. Accept the license at https://huggingface.co/briaai/Fibo-Edit"
echo "  3. Restart ComfyUI"
