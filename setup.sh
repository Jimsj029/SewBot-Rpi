#!/bin/bash
# Setup script for SewBot-Rpi
# Creates virtual environment and installs dependencies

set -e  # Exit on error

echo "========================================"
echo "SewBot-Rpi Setup Script"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not installed!"
    echo "Install with: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found!"
    exit 1
fi

# Handle existing venv
if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
    read -p "Recreate it? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing venv..."
        rm -rf .venv
    else
        echo "Using existing venv."
    fi
fi

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
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

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt

# Show installed packages
echo ""
echo "========================================"
echo "Installation Summary:"
echo "========================================"
pip list | grep -E "opencv|numpy|pygame|ultralytics|torch" || echo "Dependencies installed"

echo ""
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "To run the application:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo ""
