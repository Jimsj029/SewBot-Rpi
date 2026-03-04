#!/bin/bash
# Quick Fix: Install ONNX packages for model optimization
# Run this on Raspberry Pi if you get import errors

echo "=========================================="
echo "  SewBot - Installing ONNX packages"
echo "=========================================="
echo

# Check if running in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠ No virtual environment detected"
    echo "  Consider activating with: source .venv/bin/activate"
    echo
fi

echo "Installing onnx and onnxruntime..."
echo

# Install from piwheels (faster on RPi)
pip install onnx onnxruntime --extra-index-url https://www.piwheels.org/simple

echo
echo "=========================================="
echo "  Installation complete!"
echo "=========================================="
echo
echo "Now you can:"
echo "  1. Run: python main.py (works with existing 640x640 model)"
echo "  2. Or download optimized model: python download_faster_model.py yolo11n-seg"
echo
