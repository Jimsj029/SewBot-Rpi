#!/bin/bash
# Run script for SewBot-Rpi
# Activates virtual environment and runs the application

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check venv exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Run setup first: ./setup.sh"
    exit 1
fi

# Check main.py exists
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found!"
    exit 1
fi

# Activate venv and run
echo "Starting SewBot-Rpi..."
source .venv/bin/activate
python main.py

# Deactivate when done
deactivate
