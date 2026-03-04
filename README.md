# SewBot - Pattern Recognition System

A futuristic sewing guide application with camera overlay and pattern recognition.

##  Features

- **Futuristic UI**: Sci-fi themed interface with glowing effects and animations
- **Main Menu**: Animated start screen with pulsing buttons
- **Mode Selection**: Choose between Pattern and Wallet modes
- **Pattern Mode**: 5 levels of sewing patterns with camera overlay
- **AI-Powered Detection**: YOLOv8-nano segmentation model for real-time stitch line detection
- **Adjustable Sensitivity**: Dynamic confidence threshold adjustment during runtime
- **Wallet Mode**: Coming soon!

##  Project Structure

```
SewBot/
 main.py                 # Main entry point
 ui/                     # User interface components
    __init__.py
    theme.py           # Color scheme and theme configuration
    main_menu.py       # Main menu with Start button
    mode_selection.py  # Pattern/Wallet selection screen
 pattern/                # Pattern recognition module
    __init__.py
    blueprint_viewer.py # Camera overlay with levels 1-5
 wallet/                 # Wallet module (placeholder)
    __init__.py
 blueprint/              # Blueprint images folder
     level1.png
     level2.png
     level3.png
     level4.png
     level5.png
```

##  How to Run

### Setup (One Command)

**On Raspberry Pi:**
```bash
chmod +x setup_complete.sh
./setup_complete.sh
python3 main.py
```

**On Windows (for testing):**
```powershell
.\setup_complete.ps1
python main.py
```

**What it does:**
- Fixes NumPy 2.x compatibility (downgrades to 1.x)
- Installs ONNX packages for model auto-detection
- Checks your model and suggests optimization if needed
- Verifies all packages are working

### Optional: Better Performance

For 2-3x faster FPS (15-25 instead of 5-8):
```bash
python download_faster_model.py yolo11n-seg
python3 main.py
```

##  Design Theme

- **Color Scheme**: Various shades of blue with cyan accents
- **Style**: Futuristic sci-fi with glowing effects
- **Animations**: Pulsing buttons and animated glows
- **UI Elements**: 
  - Corner brackets and tech lines
  - Grid patterns for depth
  - Neon text effects

##  Navigation

1. **Main Menu**: Click "START" to begin
2. **Mode Selection**: 
   - Click "PATTERN" to access levels 1-5
   - Click "WALLET" (coming soon)
3. **Pattern Mode**: Use buttons or keys 1-5 to switch levels
4. **Exit**: Press Q or ESC at any time

##  Dependencies

- Python 3.x
- OpenCV (cv2) - Computer vision and camera handling
- NumPy 1.x - Numerical operations (MUST be <2.0 for OpenCV compatibility)
- Pygame - Game/sound functionality
- PyTorch - Deep learning framework (ARM-compatible version required)
- Ultralytics - YOLOv8 model inference
- ONNX Runtime 1.14+ - Required for ONNX model inference (IR version 10 support)

**Note**: Standard pip packages may not work on Raspberry Pi ARM architecture. Use `setup.sh` which installs ARM-compatible versions from piwheels and PyTorch's ARM distribution.

##  AI Model

The project uses a **YOLO segmentation model** (`best.onnx`) trained to detect stitch lines:
- **Model**: YOLOv8n-seg or YOLO11n-seg (ONNX format)
- **Classes**: stitch_line (real-time segmentation masks)
- **Input**: Auto-detected (640x640 or 320x320)
- **Confidence**: Default 0.35 (adjustable with +/- keys)
- **Performance**: 15-25 FPS on RPi4 with optimized model

  - F: Toggle fullscreen
  - **Pattern Mode Only**:
    - '+' or '=': Increase confidence threshold (reduce false positives)
    - '-': Decrease confidence threshold (more sensitive detection)

##  Documentation

- **[COMMON_ERRORS.md](COMMON_ERRORS.md)** - Quick fixes for common errors
- **[NUMPY_FIX.md](NUMPY_FIX.md)** - NumPy 2.x compatibility fix
- **[DIMENSION_ERROR_FIX.md](DIMENSION_ERROR_FIX.md)** - ONNX dimension mismatch fix
- **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)** - Performance optimization guide
- **[OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)** - Quick optimization overview
##  Controls

- **Mouse**: Click buttons to navigate
- **Keyboard**: 
  - Q or ESC: Quit/Go back
  - 1-5: Switch levels (in Pattern mode)

##  Future Updates

- Wallet functionality
- Additional pattern levels
- Enhanced visual effects
- Score tracking system
