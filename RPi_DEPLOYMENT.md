# SewBot - Raspberry Pi Deployment Guide

## 🚀 Quick Start (Recommended)

The easiest way to run SewBot on Raspberry Pi 4:

### 1. Copy files to your Raspberry Pi

Copy these files to your RPi4:
- `sewbot_standalone.py` (single file with all code)
- `run_sewbot.sh` (launcher script)
- `blueprint/` folder (pattern files)
- `videos/` folder (tutorial videos)
- `requirements.txt`

### 2. Make the launcher executable

```bash
chmod +x run_sewbot.sh
```

### 3. Run SewBot

```bash
./run_sewbot.sh
```

The launcher script will automatically:
- Check if Python 3 is installed
- Check for required packages (opencv, numpy, pygame)
- Install missing dependencies
- Launch SewBot

---

## 📋 Deployment Options

### Option 1: Shell Script Launcher (Recommended) ✨

**Pros:**
- Automatic dependency checking
- Easy to update and debug
- Small file size
- Works like an executable

**Steps:**
1. Copy all files to RPi
2. Run: `chmod +x run_sewbot.sh`
3. Run: `./run_sewbot.sh`

---

### Option 2: Direct Python Execution

**Pros:**
- Simple and straightforward
- Easy to modify code

**Steps:**
1. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-opencv libatlas-base-dev
   pip3 install opencv-python numpy pygame
   ```

2. Run:
   ```bash
   python3 sewbot_standalone.py
   ```

---

### Option 3: PyInstaller Binary (Exe-like)

**Pros:**
- Single executable file
- No need for Python installation

**Cons:**
- Large file size (100+ MB)
- Slower startup
- Must be built on RPi (ARM architecture)

**Steps:**

1. On your Raspberry Pi, install PyInstaller:
   ```bash
   pip3 install pyinstaller
   ```

2. Run the build script:
   ```bash
   python3 build_executable.py
   ```

3. The executable will be created in `dist/sewbot`

4. Run it:
   ```bash
   cd dist
   ./sewbot
   ```

---

## 🔧 Manual Installation

If you prefer to set everything up manually:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-opencv
sudo apt install -y libatlas-base-dev libopencv-dev

# Install Python packages
pip3 install opencv-python==4.8.1.78
pip3 install numpy==1.24.3
pip3 install pygame==2.5.2

# Run SewBot
python3 sewbot_standalone.py
```

---

## 📁 File Structure

```
SewBot-Rpi/
├── sewbot_standalone.py    # Single-file version (all code merged)
├── run_sewbot.sh           # Launcher script for RPi
├── build_executable.py     # PyInstaller build script
├── requirements.txt        # Python dependencies
├── blueprint/              # Pattern overlay images
│   ├── level1.png
│   ├── level2.png
│   └── ...
└── videos/                 # Tutorial videos
    └── sewing-set-up/
        ├── Step 1.MOV
        ├── Step 2.MOV
        └── ...
```

---

## 🎮 Creating a Desktop Shortcut (Optional)

To make SewBot launch like a desktop app:

1. Create a desktop file:
   ```bash
   nano ~/.local/share/applications/sewbot.desktop
   ```

2. Add this content (adjust paths):
   ```ini
   [Desktop Entry]
   Name=SewBot
   Comment=Pattern Recognition System
   Exec=/home/pi/SewBot-Rpi/run_sewbot.sh
   Icon=/home/pi/SewBot-Rpi/icon.png
   Terminal=true
   Type=Application
   Categories=Education;
   ```

3. Make it executable:
   ```bash
   chmod +x ~/.local/share/applications/sewbot.desktop
   ```

Now SewBot will appear in your application menu!

---

## 🐛 Troubleshooting

### Camera not detected
```bash
# Check if camera is connected
vcgencmd get_camera

# Enable camera in raspi-config
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
```

### OpenCV import error
```bash
# Reinstall OpenCV
pip3 uninstall opencv-python
sudo apt install python3-opencv
```

### Permission denied for run_sewbot.sh
```bash
chmod +x run_sewbot.sh
```

### Missing dependencies
```bash
pip3 install -r requirements.txt
```

---

## 📊 File Sizes Comparison

| Method | Size | Pros |
|--------|------|------|
| Python script | ~50 KB | Easy to modify, smallest |
| Shell launcher | ~52 KB | Auto-installs dependencies |
| PyInstaller binary | ~100 MB | Self-contained, no Python needed |

---

## 💡 Recommendations

- **For development:** Use direct Python execution
- **For end users:** Use the shell script launcher (`run_sewbot.sh`)
- **For offline deployment:** Use PyInstaller binary

---

## 📝 Notes

- Raspberry Pi 4 recommended (Pi 3 may be slower)
- Camera module or USB webcam required for pattern mode
- Minimum 2GB RAM recommended
- Raspbian/Raspberry Pi OS required

---

## 🆘 Support

If you encounter issues:
1. Check that all dependencies are installed
2. Verify camera is enabled and working
3. Ensure `blueprint/` and `videos/` folders are present
4. Check Python version: `python3 --version` (3.7+ required)
