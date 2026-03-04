# Fixing "Illegal Instruction" Error on Raspberry Pi

## Problem
```
run.sh: line 27: 2976 Illegal instruction python main.py
```

This error occurs when Python packages (especially PyTorch, NumPy, or ONNX Runtime) are compiled with CPU instructions not supported by your Raspberry Pi's ARM processor.

## Quick Fix

Just run the setup script on your Raspberry Pi:

```bash
chmod +x setup.sh
./setup.sh
```

Then try running again:
```bash
bash run.sh
```

The setup script handles everything:
- Configures pip to use piwheels (ARM packages)
- Removes incompatible packages
- Installs ARM-compatible versions
- Verifies all imports work correctly

## What the Setup Script Does

1. **Configures piwheels**: Sets pip to use https://www.piwheels.org/simple, which provides ARM-compiled packages
2. **Removes incompatible packages**: Uninstalls any existing x86 versions
3. **Installs PyTorch for ARM**: Uses ARM-compatible version from PyTorch's official CPU builds
4. **Installs ultralytics properly**: Installs without dependencies, then manually installs each dependency to avoid conflicts
5. **Verifies installation**: Tests that all packages import correctly

## System Package Method (Recommended for older Pi models)

For Raspberry Pi 3 or older models, use system packages:

```bash
# Install system packages
sudo apt-get update
sudo apt-get install -y python3-opencv python3-numpy python3-pygame libatlas-base-dev

# Then create venv with system packages access
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# Install only the remaining packages
pip install ultralytics --no-deps
pip install pillow pyyaml requests tqdm
```

## Identifying Your Raspberry Pi Model

```bash
cat /proc/cpuinfo | grep "Model"
```

- **Pi 4 or Pi 5**: Should work with the standard fix
- **Pi 3 or older**: May need system packages method
- **Pi Zero/Zero 2**: Limited support, system packages recommended

## Technical Details

The "Illegal instruction" error is typically caused by:
- **NumPy**: Compiled with SIMD instructions (SSE, AVX) for x86, not ARM
- **PyTorch**: Pre-built wheels for x86/x64, not ARM
- **ONNX Runtime**: Built for x86 architecture

The fix ensures all packages are either:
- Compiled specifically for ARM (via piwheels)
- Downloaded from official ARM-compatible sources (PyTorch)
- Installed from system repositories (opencv, numpy)

## Verifying the Installation

The setup script automatically verifies imports, but you can test manually:

```bash
source .venv/bin/activate
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"
```

All imports should succeed without errors.

## Still Having Issues?

1. Check Python version: `python --version` (should be 3.10+)
2. Check if running in venv: `echo $VIRTUAL_ENV`
3. Check CPU architecture: `uname -m` (should show armv7l or aarch64)
4. Try running with verbose output: `python -v main.py 2>&1 | less`
   - Look for which module causes the illegal instruction

## Contact/Support

If none of these solutions work, provide:
- Your Raspberry Pi model
- Output of `cat /proc/cpuinfo | grep -E "Model|Hardware|Revision"`
- Output of `uname -a`
- Full error message from running the application
