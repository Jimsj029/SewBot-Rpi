#!/bin/bash

# ============================================================================
# SewBot-Rpi Unified Script - Setup, Fix, Convert & Run
# ============================================================================
# This script combines setup.sh, fix_all.sh, and run.sh into one
# 
# Usage:
#   ./sewbot.sh setup    - Install and configure everything
#   ./sewbot.sh fix      - Fix common issues (NumPy, ONNX, etc.)
#   ./sewbot.sh convert  - Convert ONNX model to opset 21
#   ./sewbot.sh run      - Run the application
#   ./sewbot.sh all      - Do everything (setup + fix + convert + run)
#   ./sewbot.sh          - Show interactive menu
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}=======================================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${BLUE}=======================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# ============================================================================
# System Check
# ============================================================================

check_system() {
    print_header "System Check"
    
    ARCH=$(uname -m)
    print_info "Architecture: $ARCH"
    
    if [ -f /proc/cpuinfo ]; then
        MODEL=$(cat /proc/cpuinfo | grep "Model" | head -1 | cut -d':' -f2 | xargs || echo "Unknown")
        print_info "Hardware: $MODEL"
    fi
    
    PYTHON_CMD=$(which python3 || echo "")
    if [ -z "$PYTHON_CMD" ]; then
        print_error "Python 3 not found!"
        print_info "Installing Python..."
        sudo apt update
        sudo apt install -y python3 python3-pip
        PYTHON_CMD=$(which python3)
    fi
    
    print_success "Python: $PYTHON_CMD ($(python3 --version))"
    echo ""
}

# ============================================================================
# Setup Function - Installs everything
# ============================================================================

do_setup() {
    print_header "Setup: Installing System and Python Packages"
    
    # Step 1: Update system
    print_info "Updating system packages..."
    sudo apt-get update
    
    # Step 2: Install system packages (ARM-optimized)
    print_info "Installing system packages (ARM-optimized)..."
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        python3-numpy \
        python3-opencv \
        python3-pygame \
        python3-yaml \
        python3-requests \
        python3-pil \
        python3-matplotlib \
        libatlas-base-dev \
        libopenblas-dev \
        python3-scipy \
        python3-pandas
    
    print_success "System packages installed"
    
    # Step 3: Remove old venv if exists
    if [ -d ".venv" ]; then
        print_warning "Removing old virtual environment..."
        rm -rf .venv
    fi
    
    # Step 4: Create venv with system package access
    print_info "Creating virtual environment with system package access..."
    python3 -m venv --system-site-packages .venv
    print_success "Virtual environment created"
    
    # Step 5: Activate and install Python packages
    source .venv/bin/activate
    
    print_info "Upgrading pip..."
    pip install --upgrade pip
    
    print_info "Configuring piwheels (ARM repository)..."
    pip config set global.extra-index-url https://www.piwheels.org/simple
    
    print_info "Installing NumPy <2.0 (OpenCV compatible)..."
    pip install "numpy>=1.21.0,<2.0.0" --upgrade
    
    print_info "Installing PyTorch and TorchVision (ARM compatible)..."
    pip install torch torchvision
    
    print_info "Installing Ultralytics YOLO..."
    pip install ultralytics
    
    print_info "Installing ONNX Runtime (>=1.14.0)..."
    pip uninstall -y onnxruntime 2>/dev/null || true
    pip install "onnxruntime>=1.14.0"
    
    print_info "Installing ONNX for model conversion..."
    pip install onnx
    
    # Step 6: Verify NumPy version (ultralytics may upgrade it)
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__.split('.')[0])" 2>/dev/null || echo "0")
    if [ "$NUMPY_VERSION" = "2" ]; then
        print_warning "NumPy 2.x detected! Downgrading to 1.x..."
        pip uninstall -y numpy
        pip install "numpy>=1.21.0,<2.0.0"
    fi
    
    print_success "Setup completed!"
    
    # Verification
    print_header "Verifying Installation"
    python3 << 'PYEOF'
import sys
packages = {
    'numpy': 'NumPy',
    'cv2': 'OpenCV',
    'pygame': 'Pygame',
    'torch': 'PyTorch',
    'ultralytics': 'Ultralytics',
    'onnxruntime': 'ONNX Runtime',
    'onnx': 'ONNX'
}

for module, name in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'OK')
        print(f"✅ {name:15s} {version}")
    except Exception as e:
        print(f"❌ {name:15s} ERROR: {e}")
PYEOF
    
    deactivate
    echo ""
}

# ============================================================================
# Fix Function - Fixes common issues
# ============================================================================

do_fix() {
    print_header "Fix: Resolving Common Issues"
    
    if [ ! -d ".venv" ]; then
        print_error "Virtual environment not found!"
        print_info "Running setup first..."
        do_setup
        return
    fi
    
    source .venv/bin/activate
    
    # Fix 1: NumPy version
    print_info "Checking NumPy version..."
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__.split('.')[0])" 2>/dev/null || echo "0")
    if [ "$NUMPY_VERSION" = "2" ]; then
        print_warning "Fixing NumPy version (2.x → 1.x)..."
        pip uninstall -y numpy
        pip install "numpy>=1.21.0,<2.0.0"
        print_success "NumPy downgraded to 1.x"
    else
        print_success "NumPy 1.x is installed"
    fi
    
    # Fix 2: ONNX Runtime version
    print_info "Checking ONNX Runtime version..."
    ONNX_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "0.0.0")
    ONNX_MAJOR=$(echo $ONNX_VERSION | cut -d'.' -f1)
    ONNX_MINOR=$(echo $ONNX_VERSION | cut -d'.' -f2)
    
    if [ "$ONNX_MAJOR" -lt "1" ] || ([ "$ONNX_MAJOR" -eq "1" ] && [ "$ONNX_MINOR" -lt "14" ]); then
        print_warning "Upgrading ONNX Runtime ($ONNX_VERSION → >=1.14.0)..."
        pip uninstall -y onnxruntime
        pip install "onnxruntime>=1.14.0"
        print_success "ONNX Runtime upgraded"
    else
        print_success "ONNX Runtime $ONNX_VERSION is compatible"
    fi
    
    # Fix 3: Verify PyTorch
    print_info "Checking PyTorch..."
    python3 -c "import torch; print('PyTorch:', torch.__version__)" 2>/dev/null || {
        print_warning "PyTorch not found, installing..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    }
    
    print_success "All fixes applied!"
    deactivate
    echo ""
}

# ============================================================================
# Convert Function - Converts ONNX model
# ============================================================================

do_convert() {
    print_header "Convert: ONNX Model Opset Conversion"
    
    if [ ! -d ".venv" ]; then
        print_error "Virtual environment not found!"
        print_info "Running setup first..."
        do_setup
    fi
    
    # Ensure there are ONNX models to convert
    shopt -s nullglob
    onnx_files=(models/*.onnx)
    if [ ${#onnx_files[@]} -eq 0 ]; then
        print_error "No ONNX model files found in models/"
        print_warning "Skipping conversion"
        return 1
    fi

    source .venv/bin/activate

    print_info "Converting ONNX models to opset 21..."

    for model_path in "${onnx_files[@]}"; do
        base=$(basename "$model_path" .onnx)
        backup_path="models/${base}_opset22_backup.onnx"

        python3 << PYEOF
import onnx, os, shutil
from onnx import version_converter

model_path = r"$model_path"
backup_path = r"$backup_path"

print(f"📂 Loading model: {model_path}")
try:
    model = onnx.load(model_path)
    current_opset = model.opset_import[0].version
    print(f"📊 Current opset version: {current_opset}")

    if current_opset <= 21:
        print(f"✅ Model already uses opset {current_opset} (compatible with RPi)")
    else:
        if not os.path.exists(backup_path):
            shutil.copy2(model_path, backup_path)
            print(f"💾 Backup saved: {backup_path}")
        converted_model = version_converter.convert_version(model, 21)
        onnx.save(converted_model, model_path)
        verified = onnx.load(model_path)
        new_opset = verified.opset_import[0].version
        print(f"✅ Model converted to opset {new_opset}")

except Exception as e:
    print(f"❌ Conversion failed for {model_path}: {e}")
    raise SystemExit(1)
PYEOF

        if [ $? -ne 0 ]; then
            print_error "Conversion failed for $model_path"
            deactivate
            return 1
        fi
    done

    print_success "Model conversion completed"
    deactivate
    echo ""
}

# ============================================================================
# Run Function - Runs the application
# ============================================================================

do_run() {
    print_header "Run: Starting SewBot-Rpi Application"
    
    # Check venv
    if [ ! -d ".venv" ]; then
        print_error "Virtual environment not found!"
        print_info "Run: ./sewbot.sh setup"
        exit 1
    fi
    
    # Check main.py
    if [ ! -f "main.py" ]; then
        print_error "main.py not found!"
        exit 1
    fi
    
    # Set ARM optimization environment variables
    export OPENBLAS_CORETYPE=ARMV8
    export LD_LIBRARY_PATH=/usr/lib/arm-linux-gnueabihf:$LD_LIBRARY_PATH
    
    # Activate and run
    print_info "Activating virtual environment..."
    source .venv/bin/activate
    
    print_success "Starting application..."
    echo ""
    echo "======================================================================"
    echo ""
    
    python main.py
    
    # Deactivate when done
    deactivate
}

# ============================================================================
# Test Function - Tests detection
# ============================================================================

do_test() {
    print_header "Test: Running Detection Tests"
    
    if [ ! -d ".venv" ]; then
        print_error "Virtual environment not found!"
        print_info "Run: ./sewbot.sh setup"
        exit 1
    fi
    
    source .venv/bin/activate
    
    if [ -f "test_detection.py" ]; then
        print_info "Running test_detection.py..."
        python3 test_detection.py
    else
        print_warning "test_detection.py not found, skipping..."
    fi
    
    deactivate
    echo ""
}

# ============================================================================
# All Function - Does everything
# ============================================================================

do_all() {
    print_header "Running Complete Setup, Fix, Convert & Run"
    
    check_system
    do_setup
    do_fix
    do_convert
    do_test
    
    print_header "Everything Complete!"
    print_info "Starting application in 3 seconds..."
    sleep 3
    
    do_run
}

# ============================================================================
# Interactive Menu
# ============================================================================

show_menu() {
    clear
    echo -e "${CYAN}"
    echo "======================================================================="
    echo "                    SewBot-Rpi Control Script                        "
    echo "======================================================================="
    echo -e "${NC}"
    echo "What would you like to do?"
    echo ""
    echo "  1) Setup      - Install and configure everything"
    echo "  2) Fix        - Fix common issues (NumPy, ONNX, etc.)"
    echo "  3) Convert    - Convert ONNX model to opset 21"
    echo "  4) Test       - Run detection tests"
    echo "  5) Run        - Start the application"
    echo "  6) All        - Do everything (setup + fix + convert + run)"
    echo "  7) Exit"
    echo ""
    read -p "Enter your choice [1-7]: " choice
    
    case $choice in
        1) do_setup ;;
        2) do_fix ;;
        3) do_convert ;;
        4) do_test ;;
        5) do_run ;;
        6) do_all ;;
        7) echo "Goodbye!"; exit 0 ;;
        *) print_error "Invalid choice!"; sleep 2; show_menu ;;
    esac
    
    echo ""
    read -p "Press Enter to return to menu..." dummy
    show_menu
}

# ============================================================================
# Main - Command line argument handling
# ============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "======================================================================="
    echo "                         SewBot-Rpi                                  "
    echo "       Unified Setup, Fix, Convert & Run Script                      "
    echo "======================================================================="
    echo -e "${NC}"
}

main() {
    print_banner
    
    case "${1:-menu}" in
        setup)
            check_system
            do_setup
            ;;
        fix)
            do_fix
            ;;
        convert)
            do_convert
            ;;
        test)
            do_test
            ;;
        run)
            do_run
            ;;
        all)
            do_all
            ;;
        menu|"")
            show_menu
            ;;
        help|-h|--help)
            echo "Usage: ./sewbot.sh [command]"
            echo ""
            echo "Commands:"
            echo "  setup      Install and configure everything"
            echo "  fix        Fix common issues"
            echo "  convert    Convert ONNX model to opset 21"
            echo "  test       Run detection tests"
            echo "  run        Start the application"
            echo "  all        Do everything (recommended for first run)"
            echo "  menu       Show interactive menu (default)"
            echo "  help       Show this help message"
            echo ""
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Run './sewbot.sh help' for usage"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
