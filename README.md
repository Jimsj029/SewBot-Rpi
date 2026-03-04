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

### Setup (First Time or Fix "Illegal instruction" Error)

One script does everything - initial setup AND fixes:

```bash
# Make script executable
chmod +x setup.sh run.sh

# Run setup (installs ARM-compatible packages)
./setup.sh
```

The setup script automatically:
- Creates virtual environment
- Configures pip to use piwheels (Raspberry Pi packages)
- Installs ARM-compatible versions of PyTorch and all dependencies
- Fixes "Illegal instruction" errors

### Running the Application

```bash
./run.sh
```

Or manually:
```bash
source .venv/bin/activate
python main.py
```

### Troubleshooting

**Getting "Illegal instruction" or NumPy version errors?**

Just re-run setup (it handles all fixes):
```bash
./setup.sh
```

When asked "Recreate it?", answer **Y** to rebuild the environment.

**Common issues fixed by setup.sh:**
- "Illegal instruction" errors (x86 packages on ARM)
- NumPy 2.x incompatibility with OpenCV (requires NumPy 1.x)
- Missing or incompatible PyTorch versions

For older Raspberry Pi models (Pi 2, Pi 3), install system packages first:
```bash
sudo apt-get update
sudo apt-get install -y python3-opencv python3-numpy python3-pygame libatlas-base-dev
```
Then run setup.sh again.

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

- Python 3.10
- OpenCV (cv2) - Computer vision and camera handling
- NumPy - Numerical operations
- Pygame - Game/sound functionality
- PyTorch - Deep learning framework (ARM-compatible version required)
- Ultralytics - YOLOv8 model inference
- ONNX Runtime - Optimized model execution (optional)

**Note**: Standard pip packages may not work on Raspberry Pi ARM architecture. Use `setup.sh` which installs ARM-compatible versions from piwheels and PyTorch's ARM distribution.

##  AI Model

The project uses a **YOLOv8-nano segmentation model** (`best.onnx`) trained to detect stitch lines:
- **Model**: YOLOv8n-seg (ONNX format fo
  - F: Toggle fullscreen
  - **Pattern Mode Only**:
    - '+' or '=': Increase confidence threshold (reduce false positives)
    - '-': Decrease confidence threshold (more sensitive detection)r Raspberry Pi optimization)
- **Classes**: stitch_line (real-time segmentation masks)
- **Input**: 640x640 images
- **Confidence**: Default 0.35 (adjustable with +/- keys)
- **Performance**: Optimized for Raspberry Pi 4 inference

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
