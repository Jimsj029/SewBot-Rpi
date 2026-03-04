# Fixing Raspberry Pi Errors

## Common Problems

### 1. "Illegal instruction" Error
```
run.sh: line 27: 2976 Illegal instruction python main.py
```

**Cause**: Python packages compiled with x86 CPU instructions instead of ARM.

### 2. NumPy Version Error
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
AttributeError: _ARRAY_API not found
ImportError: numpy.core.multiarray failed to import
```

**Cause**: NumPy 2.x installed, but OpenCV requires NumPy 1.x.

## Universal Fix

Both issues are fixed by deleting and rebuilding the virtual environment:

```bash
cd ~/SewBot-Rpi
rm -rf .venv
./setup.sh
```

Then run:
```bash
bash run.sh
```

**What this does:**
- `rm -rf .venv` - Deletes the broken virtual environment with wrong packages
- `./setup.sh` - Creates a fresh environment with ARM-compatible packages, NumPy 1.x, etc.

**One-liner:**
```bash
rm -rf .venv && ./setup.sh && bash run.sh
```

## What the Setup Script Does

1. **Configures piwheels**: Uses ARM-compiled packages from https://www.piwheels.org/simple
2. **Removes incompatible packages**: Uninstalls x86 versions and NumPy 2.x
3. **Installs NumPy 1.x**: Force installs `numpy<2.0.0` for OpenCV compatibility
4. **Installs PyTorch for ARM**: Uses ARM-compatible version from PyTorch's official CPU builds
5. **Installs ultralytics properly**: Without dependencies to prevent version conflicts
6. **Verifies NumPy version**: Checks and downgrades NumPy if it was upgraded to 2.x
7. **Tests all imports**: Verifies everything works correctly

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
