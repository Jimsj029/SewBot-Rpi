#!/bin/bash
# SewBot - Complete Setup & Fix Script
# This script fixes all common errors and sets up your environment
# Requires: Python 3.10+

echo "=========================================="
echo "  SewBot - Complete Setup"
echo "=========================================="
echo

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "❌ ERROR: Python $PYTHON_VERSION detected"
    echo "   SewBot requires Python 3.10 or higher"
    echo ""
    echo "   Install Python 3.10 with:"
    echo "     sudo apt update"
    echo "     sudo apt install python3.10 python3.10-venv python3.10-dev"
    echo ""
    exit 1
else
    echo "✓ Python $PYTHON_VERSION (compatible)"
fi
echo

# Check if running in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠ No virtual environment detected"
    echo "  Activating .venv..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "✓ Virtual environment activated"
    else
        echo "  Creating virtual environment..."
        python3.10 -m venv .venv 2>/dev/null || python3 -m venv .venv
        source .venv/bin/activate
        echo "✓ Virtual environment created and activated"
    fi
fi
echo

# Step 1: Fix NumPy compatibility
echo "=========================================="
echo "Step 1/3: Fixing NumPy compatibility"
echo "=========================================="
echo

NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
if [[ $NUMPY_VERSION == 2.* ]]; then
    echo "⚠️  NumPy $NUMPY_VERSION detected - needs downgrade"
    echo "   Uninstalling NumPy 2.x..."
    pip uninstall -y numpy
    echo "   Installing NumPy 1.x (compatible)..."
    pip install "numpy<2.0" --extra-index-url https://www.piwheels.org/simple
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
    echo "✓ NumPy $NUMPY_VERSION installed"
elif [[ $NUMPY_VERSION == 1.* ]]; then
    echo "✓ NumPy $NUMPY_VERSION (already compatible)"
else
    echo "   Installing NumPy 1.x..."
    pip install "numpy<2.0" --extra-index-url https://www.piwheels.org/simple
    echo "✓ NumPy installed"
fi
echo

# Step 2: Install ONNX packages
echo "=========================================="
echo "Step 2/3: Installing ONNX packages"
echo "=========================================="
echo

python3 -c "import onnx" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ ONNX already installed"
else
    echo "   Installing onnx and onnxruntime..."
    pip install onnx onnxruntime --extra-index-url https://www.piwheels.org/simple
    echo "✓ ONNX packages installed"
fi
echo

# Step 3: Check for optimized model
echo "=========================================="
echo "Step 3/3: Checking model"
echo "=========================================="
echo

if [ -f "models/best.onnx" ]; then
    echo "✓ Model found: models/best.onnx"
    
    # Check model input size
    MODEL_SIZE=$(python3 -c "import onnx; m=onnx.load('models/best.onnx'); print(m.graph.input[0].type.tensor_type.shape.dim[2].dim_value)" 2>/dev/null)
    
    if [ "$MODEL_SIZE" == "640" ]; then
        echo "  Model input: 640x640 (works but slower)"
        echo ""
        echo "  💡 TIP: For 2-3x better FPS, download optimized model:"
        echo "     python download_faster_model.py yolo11n-seg"
    elif [ "$MODEL_SIZE" == "320" ]; then
        echo "  Model input: 320x320 (optimized! ✨)"
    else
        echo "  Model input: ${MODEL_SIZE}x${MODEL_SIZE}"
    fi
else
    echo "⚠️  Model not found!"
    echo ""
    echo "   Download a model with:"
    echo "     python download_faster_model.py yolo11n-seg"
fi
echo

# Final verification
echo "=========================================="
echo "  Verification"
echo "=========================================="
echo

echo -n "Python: "
python3 --version

echo -n "NumPy: "
python3 -c "import numpy; print(numpy.__version__)"

echo -n "OpenCV: "
python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "Not installed"

echo -n "ONNX: "
python3 -c "import onnx; print(onnx.__version__)" 2>/dev/null || echo "Not installed"

echo -n "Ultralytics: "
python3 -c "from ultralytics import YOLO; print('OK')" 2>/dev/null || echo "Not installed"

echo

# Check if everything is ready
python3 -c "import numpy, cv2, onnx; from ultralytics import YOLO" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "  ✅ Setup Complete!"
    echo "=========================================="
    echo
    echo "Run your application with:"
    echo "  python3 main.py"
    echo
else
    echo "=========================================="
    echo "  ⚠️  Some packages missing"
    echo "=========================================="
    echo
    echo "If you see errors above, try:"
    echo "  1. Check internet connection"
    echo "  2. Run: pip install ultralytics opencv-python"
    echo
fi
