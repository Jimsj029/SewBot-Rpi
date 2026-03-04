#!/bin/bash
# Setup & Fix script for SewBot-Rpi (Raspberry Pi 4 Optimized)
# Uses system packages to avoid ARM compatibility issues

set -e  # Exit on error

echo "========================================"
echo "SewBot-Rpi Setup Script"
echo "Raspberry Pi 4 - System Package Method"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Detect architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"
echo ""

# Install system packages (CRITICAL for ARM)
echo "========================================"
echo "Step 1: Install System Packages"
echo "========================================"
echo "Installing system packages (requires sudo)..."
echo "This avoids 'Illegal instruction' errors!"
echo ""

sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-numpy \
    python3-opencv \
    python3-pygame \
    python3-yaml \
    python3-requests \
    python3-pil \
    python3-matplotlib \
    libatlas-base-dev \
    python3-scipy \
    python3-pandas

echo ""
echo "System packages installed!"
echo ""

# Check Python 3
PYTHON_CMD=$(which python3)
echo "Using Python: $PYTHON_CMD"
python3 --version
echo ""

# Handle existing venv
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create venv WITH system packages access
echo "========================================"
echo "Step 2: Create Virtual Environment"
echo "========================================"
echo "Creating venv with system package access..."
python3 -m venv --system-site-packages .venv
echo "Virtual environment created!"
echo ""

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "========================================"
echo "Step 3: Install Python Packages"
echo "========================================"
echo "Upgrading pip..."
pip install --upgrade pip

# Install only packages NOT available in system
echo ""
echo "Installing additional packages (using piwheels)..."
pip config set global.extra-index-url https://www.piwheels.org/simple

# Install torch for ARM - use pip which will get from piwheels
echo "Installing PyTorch (from piwheels - ARM compatible)..."
pip install torch torchvision

# Install ultralytics
echo "Installing ultralytics..."
pip install ultralytics

echo ""
echo "========================================"
echo "Step 4: Verification"
echo "========================================"
echo "Testing imports..."
python3 << 'PYEOF'
import sys
print(f"Python: {sys.version}")
try:
    import numpy
    print(f"✓ NumPy: {numpy.__version__}")
except Exception as e:
    print(f"✗ NumPy: {e}")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV: {e}")

try:
    import pygame
    print(f"✓ Pygame: {pygame.__version__}")
except Exception as e:
    print(f"✗ Pygame: {e}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except Exception as e:
    print(f"✗ PyTorch: {e}")

try:
    from ultralytics import YOLO
    print("✓ Ultralytics: OK")
except Exception as e:
    print(f"✗ Ultralytics: {e}")
PYEOF

echo ""
echo "========================================"
echo "Setup completed!"
echo "========================================"
echo ""
echo "To run the application:"
echo "  bash run.sh"
echo ""
echo "If errors persist:"
echo "  rm -rf .venv && ./setup.sh"
echo ""
