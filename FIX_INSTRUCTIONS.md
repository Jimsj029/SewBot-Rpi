# SewBot-Rpi Fix Script

## Quick Start (On Raspberry Pi)

Run this single command to fix all issues:

```bash
bash fix_all.sh
```

## What This Script Fixes

1. **Illegal Instruction Error** - Installs ARM-compatible packages
2. **NumPy Version Conflicts** - Downgrades NumPy to <2.0 for OpenCV
3. **ONNX Runtime Issues** - Upgrades to version 1.19.2
4. **Model Compatibility** - Converts ONNX model from opset 22 to opset 21

## What The Script Does

### Step 1: System Check
- Verifies Raspberry Pi hardware (aarch64 or armv7l)
- Shows CPU and model information

### Step 2: System Packages
- Updates package lists
- Installs Python development tools
- Installs ARM-optimized OpenCV, NumPy, and Pygame

### Step 3: Virtual Environment
- Removes old .venv if exists
- Creates new venv with `--system-site-packages` flag
- Allows access to ARM-compiled system packages

### Step 4: Python Packages
- Configures piwheels (ARM package repository)
- Installs NumPy <2.0 (OpenCV compatible)
- Upgrades ONNX Runtime to >=1.14.0
- Installs PyTorch, Ultralytics, and ONNX

### Step 5: Model Conversion
- Converts `models/best.onnx` from opset 22 to opset 21
- Creates backup of original model
- Verifies conversion was successful

### Step 6: Testing
- Runs `test_detection.py` if available
- Verifies all packages are working

### Step 7: Run Script
- Creates `run_fixed.sh` with ARM optimizations
- Sets environment variables for OpenBLAS

## After Running The Script

### Test Detection
```bash
source .venv/bin/activate
python test_detection.py
```

### Run The Application
```bash
./run_fixed.sh
```

Or manually:
```bash
source .venv/bin/activate
export OPENBLAS_CORETYPE=ARMV8
python main.py
```

## Troubleshooting

### Still Getting "Illegal Instruction"?
```bash
# Use the fixed run script
./run_fixed.sh

# Or set environment variables
export OPENBLAS_CORETYPE=ARMV8
export LD_LIBRARY_PATH=/usr/lib/arm-linux-gnueabihf:$LD_LIBRARY_PATH
python main.py
```

### NumPy Version Errors?
```bash
source .venv/bin/activate
pip uninstall numpy -y
pip install "numpy>=1.21.0,<2.0.0"
```

### ONNX Model Won't Load?
```bash
# Run conversion manually
source .venv/bin/activate
python setup_and_convert_model.py
```

### No Detections?
- Lower confidence threshold (press '-' key in app)
- Ensure good lighting on fabric
- Point camera at visible stitch lines
- Start with threshold around 0.15-0.25

## Files Created

- `fix_all.sh` - Main fix script (this file)
- `run_fixed.sh` - Application launcher with ARM optimizations
- `.venv/` - Virtual environment with correct packages
- `models/best_opset22_backup.onnx` - Backup of original model
- `models/best.onnx` - Converted model (opset 21)

## Requirements

- Raspberry Pi 4 (or compatible ARM device)
- Raspberry Pi OS (Bullseye or newer)
- Internet connection (for downloading packages)
- ~2GB free disk space

## Manual Installation

If the script fails, follow these steps manually:

```bash
# 1. Install system packages
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-opencv python3-numpy python3-pygame

# 2. Create virtual environment
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# 3. Install packages
pip install --upgrade pip
pip install "numpy>=1.21.0,<2.0.0"
pip install "onnxruntime>=1.14.0"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics onnx

# 4. Convert model
python setup_and_convert_model.py

# 5. Test
python test_detection.py
```

## Support

If you encounter issues:
1. Check the output of `fix_all.sh` for error messages
2. Verify package versions: `pip list | grep -E "numpy|onnx|torch"`
3. Check CPU features: `cat /proc/cpuinfo | grep Features`
4. Review logs in terminal output

## Technical Details

### Why These Fixes Are Needed

- **ARM Architecture**: Raspberry Pi 4 uses ARMv8.0 (Cortex-A72), not ARMv8.1
- **No LSE Instructions**: CPU lacks atomic instructions required by PyTorch 2.6+
- **NumPy 2.x**: OpenCV compiled against NumPy 1.x, incompatible with 2.x
- **ONNX Opset**: RPi's ONNX Runtime 1.19.2 only supports opset up to 21

### Package Versions

- NumPy: 1.21.0 - 1.99.x (must be <2.0)
- ONNX Runtime: 1.14.0+ (1.19.2 recommended)
- PyTorch: Any version from CPU wheel repository
- Ultralytics: Latest version (auto-adjusts to PyTorch)
