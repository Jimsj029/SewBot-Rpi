#!/bin/bash
# Setup & Fix script for SewBot-Rpi
# Creates virtual environment and installs ARM-compatible dependencies
# Also fixes "Illegal instruction" errors on Raspberry Pi

set -e  # Exit on error

echo "========================================"
echo "SewBot-Rpi Setup Script"
echo "Fixes 'Illegal instruction' errors"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python 3.10
if ! command -v python3.10 &> /dev/null; then
    echo "ERROR: Python 3.10 not installed!"
    echo "Install with: sudo apt install python3.10 python3.10-venv python3-pip"
    exit 1
fi

echo "Python version:"
python3.10 --version
echo ""

# Recommend system packages
echo "========================================"
echo "RECOMMENDED: Install system packages"
echo "========================================"
echo "For best compatibility, install these first:"
echo "  sudo apt-get update"
echo "  sudo apt-get install -y python3-opencv python3-numpy python3-pygame libatlas-base-dev"
echo ""
read -p "Continue with setup? (Y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

# Handle existing venv
if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
    echo ""
    echo "Options:"
    echo "  1) Recreate (RECOMMENDED - fixes all package issues)"
    echo "  2) Keep and reinstall packages only"
    echo ""
    read -p "Choice (1 or 2) [1]: " -n 1 -r
    echo ""
    
    # Default to 1 if just Enter pressed
    if [[ -z "$REPLY" ]] || [[ "$REPLY" = "1" ]]; then
        echo "Removing existing venv..."
        rm -rf .venv
    else
        echo "Keeping venv, will reinstall packages..."
    fi
fi

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv .venv
    echo "Virtual environment created!"
fi

# Activate venv
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Configure pip to use piwheels (Raspberry Pi optimized packages)
echo ""
echo "Configuring pip for Raspberry Pi (piwheels)..."
pip config set global.extra-index-url https://www.piwheels.org/simple

# Uninstall potentially problematic packages
echo ""
echo "Removing any incompatible packages..."
pip uninstall -y torch torchvision ultralytics onnxruntime numpy opencv-python 2>/dev/null || true

# Install dependencies
echo ""
echo "Installing dependencies (ARM-compatible versions)..."
echo "This may take several minutes..."

# Install numpy first (MUST be 1.x for OpenCV compatibility)
echo "  - Installing NumPy 1.x (CRITICAL: OpenCV requires NumPy 1.x)..."
pip install "numpy>=1.24.3,<2.0.0"

# Install opencv-python (compiled against NumPy 1.x)
echo "  - Installing OpenCV (with NumPy 1.x compatibility)..."
pip install --no-build-isolation opencv-python==4.8.1.78

# Install pygame
echo "  - Installing Pygame..."
pip install pygame==2.5.2

# Install PyTorch for Raspberry Pi (ARM architecture)
echo ""
echo "Installing PyTorch for ARM architecture (CPU-only)..."
echo "This is a large download and may take a while..."
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu

# Install ultralytics dependencies first
echo ""
echo "Installing ultralytics dependencies..."
pip install matplotlib pillow pyyaml requests scipy tqdm psutil py-cpuinfo thop pandas seaborn

# Install ultralytics (YOLO) without dependencies
echo "Installing ultralytics (YOLO model library)..."
pip install ultralytics --no-deps

# CRITICAL: Verify NumPy hasn't been upgraded to 2.x
echo ""
echo "Verifying NumPy version (must be 1.x for OpenCV)..."
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null | cut -d. -f1)
if [ "$NUMPY_VERSION" = "2" ]; then
    echo "WARNING: NumPy was upgraded to 2.x! Downgrading..."
    pip uninstall -y numpy
    pip install "numpy>=1.24.3,<2.0.0"
    echo "NumPy downgraded to 1.x"
fi

# Verify installations
echo ""
echo "========================================"
echo "Verifying Installation"
echo "========================================"
echo "Testing imports..."
python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')" || echo "✗ NumPy import failed"
python -c "import cv2; print(f'✓ OpenCV: {cv2.__version__}')" || echo "✗ OpenCV import failed"
python -c "import pygame; print(f'✓ Pygame: {pygame.__version__}')" || echo "✗ Pygame import failed"
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" || echo "✗ PyTorch import failed"
python -c "from ultralytics import YOLO; print('✓ Ultralytics: OK')" || echo "✗ Ultralytics import failed"

# Show installed packages
echo ""
echo "========================================"
echo "Installation Summary:"
echo "========================================"
pip list | grep -E "opencv|numpy|pygame|ultralytics|torch"

echo ""
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "To run the application:"
echo "  bash run.sh"
echo ""
echo "Or manually:"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo ""
echo "If you get 'Illegal instruction' or NumPy errors:"
echo "  1. Run: rm -rf .venv"
echo "  2. Run: ./setup.sh"
echo "  (This deletes and recreates everything)"
echo ""
