#!/bin/bash
# Quick fix for NumPy 2.x incompatibility with OpenCV

echo "========================================"
echo "Fixing NumPy Version Incompatibility"
echo "========================================"
echo ""
echo "Problem: OpenCV was compiled against NumPy 1.x"
echo "Solution: Downgrade NumPy to 1.x"
echo ""

# Activate venv
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        echo "Activating virtual environment..."
        source .venv/bin/activate
    else
        echo "ERROR: No virtual environment found!"
        echo "Run: ./setup.sh"
        exit 1
    fi
fi

echo "Current NumPy version:"
python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "NumPy not installed or broken"
echo ""

# Uninstall numpy and opencv
echo "Removing NumPy and OpenCV..."
pip uninstall -y numpy opencv-python 2>/dev/null || true
echo ""

# Install NumPy 1.x
echo "Installing NumPy 1.24.3 (compatible with OpenCV)..."
pip install "numpy>=1.24.3,<2.0.0"
echo ""

# Reinstall OpenCV
echo "Reinstalling OpenCV..."
pip install --no-build-isolation opencv-python==4.8.1.78
echo ""

# Verify
echo "========================================"
echo "Verification"
echo "========================================"
python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')"
python -c "import cv2; print(f'✓ OpenCV: {cv2.__version__}')"
echo ""
echo "Done! Try running: bash run.sh"
