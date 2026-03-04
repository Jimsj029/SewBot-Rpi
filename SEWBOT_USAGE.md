# SewBot-Rpi - Unified Control Script

## Quick Start (One Command Does Everything!)

```bash
chmod +x sewbot.sh
./sewbot.sh all
```

That's it! This will:
- ✅ Install all system packages
- ✅ Create virtual environment
- ✅ Fix all compatibility issues
- ✅ Convert ONNX model to opset 21
- ✅ Test detection
- ✅ Run the application

---

## Usage

### Interactive Menu (Recommended for First Time)
```bash
./sewbot.sh
```
Shows a menu where you can choose what to do.

### Command Line Usage

```bash
./sewbot.sh setup      # Install and configure everything
./sewbot.sh fix        # Fix common issues (NumPy, ONNX, etc.)
./sewbot.sh convert    # Convert ONNX model to opset 21
./sewbot.sh test       # Run detection tests
./sewbot.sh run        # Start the application
./sewbot.sh all        # Do everything
./sewbot.sh help       # Show help message
```

---

## What Each Command Does

### `./sewbot.sh setup`
**Installs and configures everything:**
- Updates system packages
- Installs ARM-optimized system packages (OpenCV, NumPy, Pygame)
- Creates virtual environment with system package access
- Installs Python packages (PyTorch, Ultralytics, ONNX Runtime)
- Configures piwheels for ARM compatibility
- Verifies all installations

### `./sewbot.sh fix`
**Fixes common Raspberry Pi issues:**
- ✅ Downgrades NumPy from 2.x to 1.x (OpenCV compatibility)
- ✅ Upgrades ONNX Runtime to >=1.14.0 (opset 21 support)
- ✅ Verifies PyTorch installation
- ✅ Fixes "Illegal instruction" errors

### `./sewbot.sh convert`
**Converts ONNX model:**
- Converts `models/best.onnx` from opset 22 to opset 21
- Creates backup of original model
- Verifies conversion was successful
- Required for Raspberry Pi 4 compatibility

### `./sewbot.sh test`
**Tests detection:**
- Runs `test_detection.py` if available
- Verifies model loads correctly
- Tests inference with dummy data

### `./sewbot.sh run`
**Runs the application:**
- Sets ARM optimization environment variables
- Activates virtual environment
- Launches `main.py`

### `./sewbot.sh all`
**Does everything in order:**
1. System check
2. Setup
3. Fix
4. Convert model
5. Test
6. Run application

---

## First Time Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/SewBot-Rpi.git
cd SewBot-Rpi

# 2. Make script executable
chmod +x sewbot.sh

# 3. Run everything
./sewbot.sh all
```

---

## Troubleshooting

### "Virtual environment not found"
```bash
./sewbot.sh setup
```

### "Illegal instruction" error
```bash
./sewbot.sh fix
```

### Model won't load / Opset error
```bash
./sewbot.sh convert
```

### No detections
- Lower confidence threshold (press '-' key in app)
- Ensure good lighting
- Point camera at fabric with stitch lines

### Start fresh
```bash
rm -rf .venv
./sewbot.sh all
```

---

## Environment Variables

The script automatically sets these for ARM optimization:
```bash
OPENBLAS_CORETYPE=ARMV8
LD_LIBRARY_PATH=/usr/lib/arm-linux-gnueabihf:$LD_LIBRARY_PATH
```

---

## Requirements

- **Hardware**: Raspberry Pi 4 (or compatible ARM device)
- **OS**: Raspberry Pi OS (Bullseye or newer)
- **Architecture**: aarch64 (64-bit) or armv7l (32-bit)
- **Disk Space**: ~2GB free
- **Internet**: Required for downloading packages

---

## Files Created

After running the script:

```
SewBot-Rpi/
├── .venv/                          # Virtual environment
├── models/
│   ├── best.onnx                   # Converted model (opset 21)
│   ├── best_opset22_backup.onnx   # Original model backup
│   ├── cloth.pt                    # PyTorch model (if exists)
│   └── stitch.pt                   # PyTorch model (if exists)
├── sewbot.sh                       # This unified script
├── setup.sh                        # Original setup (deprecated)
├── run.sh                          # Original run (deprecated)
├── fix_all.sh                      # Original fix (deprecated)
└── main.py                         # Main application
```

---

## Package Versions

The script ensures these versions are installed:

| Package | Version | Reason |
|---------|---------|--------|
| NumPy | 1.21.0 - 1.99.x | <2.0 required for OpenCV |
| ONNX Runtime | >=1.14.0 | Supports opset 21 |
| PyTorch | Latest CPU | ARM compatible from piwheels |
| Ultralytics | Latest | YOLO detection |
| OpenCV | System package | ARM-optimized |
| Pygame | System package | ARM-optimized |

---

## Technical Details

### Why This Script Exists

Raspberry Pi 4 uses **ARMv8.0** (Cortex-A72), which:
- ❌ Doesn't support ARMv8.1 LSE atomic instructions
- ❌ Can't run PyTorch 2.6+ (compiled with LSE)
- ❌ Requires NumPy <2.0 for OpenCV compatibility
- ❌ ONNX Runtime 1.19.2 only supports opset ≤21

### How This Script Fixes It

1. **System packages**: Uses ARM-compiled packages from apt
2. **Piwheels**: ARM-compiled Python packages
3. **NumPy pinning**: Forces <2.0 for OpenCV
4. **ONNX conversion**: Downconverts model to opset 21
5. **Environment vars**: Optimizes for ARMv8.0

---

## Migration from Old Scripts

If you were using the old scripts:

**Old way:**
```bash
bash setup.sh
bash run.sh
```

**New way:**
```bash
./sewbot.sh all
```

Or just run:
```bash
./sewbot.sh run
```

The old scripts still work but are deprecated. The unified `sewbot.sh` is recommended.

---

## Support

For issues:
1. Check output of `./sewbot.sh all` for error messages
2. Run `./sewbot.sh fix` to attempt automatic fixes
3. Check package versions: `source .venv/bin/activate && pip list`
4. Verify CPU: `cat /proc/cpuinfo | grep Features`

---

## Credits

Built for Raspberry Pi 4 with Raspberry Pi OS (Bullseye/Bookworm) on aarch64 architecture.

Handles common ARM compatibility issues for YOLO-based detection systems.
