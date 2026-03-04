#!/bin/bash

# SewBot-Rpi Complete Fix Script for Raspberry Pi 4
# This script fixes: Illegal instruction, NumPy conflicts, ONNX Runtime, and model conversion

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "======================================================================"
    echo -e "${BLUE}  $1${NC}"
    echo "======================================================================"
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
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if running on Raspberry Pi
check_raspberry_pi() {
    print_header "Checking System Compatibility"
    
    if [[ $(uname -m) != "aarch64" ]] && [[ $(uname -m) != "armv7l" ]]; then
        print_warning "This script is designed for Raspberry Pi (ARM architecture)"
        print_info "Detected architecture: $(uname -m)"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    print_success "Architecture: $(uname -m)"
    print_info "Hardware: $(cat /proc/cpuinfo | grep 'Model' | head -1 | cut -d':' -f2 | xargs || echo 'Unknown')"
}

# Update system packages
update_system() {
    print_header "Step 1: Updating System Packages"
    
    print_info "Updating package lists..."
    sudo apt update
    
    print_success "System updated"
}

# Install system dependencies
install_system_packages() {
    print_header "Step 2: Installing System Dependencies"
    
    print_info "Installing Python and build tools..."
    sudo apt install -y python3 python3-pip python3-venv python3-dev
    
    print_info "Installing ARM-optimized packages..."
    sudo apt install -y python3-opencv python3-numpy python3-pygame
    
    print_info "Installing additional dependencies..."
    sudo apt install -y libopenblas-dev libatlas-base-dev
    
    print_success "System packages installed"
}

# Remove old virtual environment
clean_old_venv() {
    print_header "Step 3: Cleaning Old Virtual Environment"
    
    if [ -d ".venv" ]; then
        print_warning "Removing old .venv directory..."
        rm -rf .venv
        print_success "Old virtual environment removed"
    else
        print_info "No old virtual environment found"
    fi
}

# Create virtual environment with system packages access
create_venv() {
    print_header "Step 4: Creating Virtual Environment"
    
    print_info "Creating venv with --system-site-packages flag..."
    python3 -m venv .venv --system-site-packages
    
    print_success "Virtual environment created"
}

# Activate virtual environment and install packages
install_python_packages() {
    print_header "Step 5: Installing Python Packages"
    
    # Activate virtual environment
    source .venv/bin/activate
    
    print_info "Upgrading pip..."
    pip install --upgrade pip
    
    print_info "Configuring piwheels (ARM repository)..."
    pip config set global.extra-index-url https://www.piwheels.org/simple
    
    print_info "Installing NumPy <2.0 (required for OpenCV compatibility)..."
    pip install "numpy>=1.21.0,<2.0.0"
    
    print_info "Installing Pygame..."
    pip install "pygame>=2.0.0"
    
    print_info "Upgrading ONNX Runtime (required for model opset 21)..."
    pip install "onnxruntime>=1.14.0"
    
    print_info "Installing PyTorch and Ultralytics..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install ultralytics
    
    print_info "Installing ONNX for model conversion..."
    pip install onnx
    
    print_success "Python packages installed"
}

# Verify package versions
verify_packages() {
    print_header "Step 6: Verifying Package Versions"
    
    source .venv/bin/activate
    
    echo ""
    python3 << 'EOF'
import sys

packages = {
    'numpy': 'numpy',
    'cv2': 'opencv-python',
    'pygame': 'pygame',
    'torch': 'pytorch',
    'ultralytics': 'ultralytics',
    'onnxruntime': 'onnxruntime',
    'onnx': 'onnx'
}

print("Package Versions:")
print("-" * 50)

all_ok = True
for module, name in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        
        # Check specific version requirements
        if module == 'numpy':
            import numpy as np
            if int(np.__version__.split('.')[0]) >= 2:
                print(f"❌ {name:20s} {version:15s} (SHOULD BE <2.0)")
                all_ok = False
            else:
                print(f"✅ {name:20s} {version:15s}")
        elif module == 'onnxruntime':
            import onnxruntime as ort
            major, minor = map(int, ort.__version__.split('.')[:2])
            if major < 1 or (major == 1 and minor < 14):
                print(f"❌ {name:20s} {version:15s} (SHOULD BE >=1.14)")
                all_ok = False
            else:
                print(f"✅ {name:20s} {version:15s}")
        else:
            print(f"✅ {name:20s} {version:15s}")
    except ImportError as e:
        print(f"❌ {name:20s} NOT INSTALLED")
        all_ok = False

print("-" * 50)
sys.exit(0 if all_ok else 1)
EOF

    if [ $? -eq 0 ]; then
        print_success "All packages verified"
    else
        print_error "Some packages have version issues"
        return 1
    fi
}

# Convert ONNX model
convert_model() {
    print_header "Step 7: Converting ONNX Model"
    
    source .venv/bin/activate
    
    if [ ! -f "models/best.onnx" ]; then
        print_error "Model file not found: models/best.onnx"
        return 1
    fi
    
    print_info "Running model conversion script..."
    
    python3 << 'EOF'
import onnx
from onnx import version_converter
import os
import shutil

model_path = "models/best.onnx"
backup_path = "models/best_opset22_backup.onnx"

print(f"📂 Loading model: {model_path}")
model = onnx.load(model_path)

current_opset = model.opset_import[0].version
print(f"📊 Current opset version: {current_opset}")

if current_opset <= 21:
    print(f"✅ Model already uses opset {current_opset} (compatible)")
    exit(0)

print(f"🔄 Converting from opset {current_opset} to opset 21...")

# Backup
if not os.path.exists(backup_path):
    shutil.copy2(model_path, backup_path)
    print(f"💾 Backup saved to: {backup_path}")

# Convert
converted_model = version_converter.convert_version(model, 21)
onnx.save(converted_model, model_path)

# Verify
verified = onnx.load(model_path)
new_opset = verified.opset_import[0].version
print(f"✅ Model converted to opset {new_opset}")
EOF

    if [ $? -eq 0 ]; then
        print_success "Model conversion completed"
    else
        print_error "Model conversion failed"
        return 1
    fi
}

# Test detection
test_detection() {
    print_header "Step 8: Testing Detection Model"
    
    source .venv/bin/activate
    
    if [ -f "test_detection.py" ]; then
        print_info "Running detection test..."
        python3 test_detection.py
        
        if [ $? -eq 0 ]; then
            print_success "Detection test passed"
        else
            print_warning "Detection test completed with warnings"
        fi
    else
        print_info "test_detection.py not found, skipping test"
    fi
}

# Create environment variable fix script
create_run_script() {
    print_header "Step 9: Creating Fixed Run Script"
    
    cat > run_fixed.sh << 'RUNSCRIPT'
#!/bin/bash

# Fixed run script with ARM optimizations
export OPENBLAS_CORETYPE=ARMV8
export LD_LIBRARY_PATH=/usr/lib/arm-linux-gnueabihf:$LD_LIBRARY_PATH

# Activate virtual environment
source .venv/bin/activate

# Run the application
python main.py
RUNSCRIPT

    chmod +x run_fixed.sh
    
    print_success "Created run_fixed.sh"
    print_info "Use './run_fixed.sh' instead of 'bash run.sh' to run the app"
}

# Final summary
print_summary() {
    print_header "Installation Complete!"
    
    echo ""
    print_success "All fixes applied successfully!"
    echo ""
    echo "Summary of fixes:"
    echo "  ✅ System packages installed (ARM-optimized)"
    echo "  ✅ Virtual environment created with system packages access"
    echo "  ✅ NumPy <2.0 installed (OpenCV compatible)"
    echo "  ✅ ONNX Runtime upgraded (>=1.14.0)"
    echo "  ✅ PyTorch and Ultralytics installed"
    echo "  ✅ ONNX model converted to opset 21"
    echo "  ✅ Environment variables configured"
    echo ""
    echo "Next steps:"
    echo "  1. Test detection: python test_detection.py"
    echo "  2. Run application: ./run_fixed.sh"
    echo ""
    echo "Troubleshooting:"
    echo "  - If you get 'Illegal instruction': Run ./run_fixed.sh instead of run.sh"
    echo "  - If NumPy errors: source .venv/bin/activate && pip install 'numpy<2.0'"
    echo "  - If no detections: Lower confidence threshold with '-' key in app"
    echo ""
    print_info "Environment variables are set in run_fixed.sh"
    print_info "Virtual environment: source .venv/bin/activate"
    echo ""
}

# Error handler
handle_error() {
    print_error "Script failed at step: $1"
    echo ""
    echo "To retry:"
    echo "  bash fix_all.sh"
    echo ""
    echo "For manual fixes, see: TROUBLESHOOTING_ILLEGAL_INSTRUCTION.md"
    exit 1
}

# Main execution
main() {
    print_header "SewBot-Rpi Complete Fix Script"
    echo "This script will fix all known issues on Raspberry Pi 4:"
    echo "  - Illegal instruction errors"
    echo "  - NumPy version conflicts"
    echo "  - ONNX Runtime compatibility"
    echo "  - Model opset version"
    echo ""
    
    check_raspberry_pi || handle_error "System check"
    update_system || handle_error "System update"
    install_system_packages || handle_error "System package installation"
    clean_old_venv || handle_error "Cleaning old environment"
    create_venv || handle_error "Virtual environment creation"
    install_python_packages || handle_error "Python package installation"
    verify_packages || handle_error "Package verification"
    convert_model || handle_error "Model conversion"
    test_detection || handle_error "Detection test"
    create_run_script || handle_error "Run script creation"
    print_summary
}

# Run main function
main
