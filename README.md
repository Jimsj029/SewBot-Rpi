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

### Setup (First Time or Fix Errors)

**One command to fix everything:**

```bash
cd ~/SewBot-Rpi
rm -rf .venv && ./setup.sh
```

The setup script will:
- Install system packages (ARM-compatible, avoids "Illegal instruction")
- Create virtual environment with system package access
- Install PyTorch and ultralytics from piwheels
- Verify all imports work

### Running the Application

```bash
bash run.sh
```

Or manually:
```bash
source .venv/bin/activate
python main.py
```

### Create a Clickable Desktop Icon (Raspberry Pi)

Run these commands once on your Raspberry Pi:

```bash
cd ~/SewBot-Rpi
chmod +x install_desktop_icon.sh
./install_desktop_icon.sh
```

This creates:
- `~/Desktop/SewBot-Rpi.desktop` (desktop icon)
- `~/.local/share/applications/SewBot-Rpi.desktop` (app menu entry)

If Raspberry Pi still blocks launching, right-click the desktop icon and choose **Allow Launching**.

If the icon still does not open the app, check launcher logs:

```bash
tail -n 80 ~/sewbot-launch.log
```

### Why This Works

The script uses **system packages** (python3-numpy, python3-opencv, python3-pygame) which are pre-compiled for ARM by Raspberry Pi OS. This completely avoids the "Illegal instruction" error that happens when pip installs x86 packages.

### Troubleshooting

**Any errors? Just run:**
```bash
rm -rf .venv && ./setup.sh
```

This deletes everything and rebuilds from scratch.

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
