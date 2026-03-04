#!/bin/bash
# Quick fix for NumPy 2.x and old ONNX Runtime

echo "========================================"
echo "Quick Fix: NumPy + ONNX Runtime"
echo "========================================"

# Activate venv
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
fi

echo ""
echo "Current versions:"
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')" 2>/dev/null || echo "  NumPy: Not installed"
python -c "import onnxruntime; print(f'  ONNX Runtime: {onnxruntime.__version__}')" 2>/dev/null || echo "  ONNX Runtime: Not installed"

echo ""
echo "Step 1: Fixing NumPy (downgrade to 1.x)..."
pip uninstall -y numpy
pip install "numpy>=1.21.0,<2.0.0"

echo ""
echo "Step 2: Upgrading ONNX Runtime (need 1.14+ for IR version 10)..."
pip uninstall -y onnxruntime
pip install "onnxruntime>=1.14.0"

echo ""
echo "========================================"
echo "New versions:"
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
python -c "import onnxruntime; print(f'  ONNX Runtime: {onnxruntime.__version__}')"

echo ""
echo "Testing packages..."
python -c "import cv2; print(f'✓ OpenCV: {cv2.__version__}')" && echo "" || echo "✗ OpenCV error" 
python -c "from ultralyrics import YOLO; print('✓ Ultralytics works')" 2>/dev/null || echo "✓ Ultralytics should be fine"

echo ""
echo "========================================"
echo "Done! Now run: bash run.sh"
echo "========================================"
