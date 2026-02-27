#!/bin/bash
###############################################################################
# SewBot Launcher for Raspberry Pi
# This script checks dependencies and runs the SewBot application
###############################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   SewBot - Pattern Recognition System${NC}"
echo -e "${BLUE}   Raspberry Pi Launcher${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Please install Python 3:"
    echo "  sudo apt update"
    echo "  sudo apt install python3 python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ Python found: $PYTHON_VERSION${NC}"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}Warning: pip3 not found. Installing...${NC}"
    sudo apt update
    sudo apt install python3-pip -y
fi

# Function to check if a Python package is installed
check_package() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# Check and install dependencies
MISSING_DEPS=()

echo ""
echo "Checking dependencies..."

# Check OpenCV
if ! check_package "cv2"; then
    echo -e "${YELLOW}✗ opencv-python not found${NC}"
    MISSING_DEPS+=("opencv-python")
else
    echo -e "${GREEN}✓ opencv-python installed${NC}"
fi

# Check NumPy
if ! check_package "numpy"; then
    echo -e "${YELLOW}✗ numpy not found${NC}"
    MISSING_DEPS+=("numpy")
else
    echo -e "${GREEN}✓ numpy installed${NC}"
fi

# Check Pygame (optional for audio)
if ! check_package "pygame"; then
    echo -e "${YELLOW}✗ pygame not found (audio will be disabled)${NC}"
    MISSING_DEPS+=("pygame")
else
    echo -e "${GREEN}✓ pygame installed${NC}"
fi

# Install missing dependencies
if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Installing missing dependencies...${NC}"
    
    # Install system dependencies for OpenCV on Raspberry Pi
    if [[ " ${MISSING_DEPS[@]} " =~ " opencv-python " ]]; then
        echo "Installing OpenCV system dependencies..."
        sudo apt update
        sudo apt install -y libopencv-dev python3-opencv libatlas-base-dev
    fi
    
    # Install Python packages
    echo "Installing Python packages: ${MISSING_DEPS[@]}"
    pip3 install --user "${MISSING_DEPS[@]}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to install dependencies${NC}"
        echo "Try manually with: pip3 install ${MISSING_DEPS[@]}"
        exit 1
    fi
fi

# Check if sewbot_standalone.py exists
if [ ! -f "$SCRIPT_DIR/sewbot_standalone.py" ]; then
    echo -e "${RED}Error: sewbot_standalone.py not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Optional: Check for required directories (blueprint, videos)
if [ ! -d "$SCRIPT_DIR/blueprint" ]; then
    echo -e "${YELLOW}Warning: 'blueprint' directory not found${NC}"
    echo "Pattern mode may not work correctly"
fi

if [ ! -d "$SCRIPT_DIR/videos" ]; then
    echo -e "${YELLOW}Warning: 'videos' directory not found${NC}"
    echo "Tutorial videos will not be available"
fi

# Launch SewBot
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}Starting SewBot...${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

cd "$SCRIPT_DIR"
python3 sewbot_standalone.py

# Capture exit code
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}SewBot exited normally${NC}"
else
    echo -e "${RED}SewBot exited with error code: $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
