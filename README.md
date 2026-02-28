# SewBot - Pattern Recognition System

A futuristic sewing guide application with camera overlay, pattern recognition, and AI-powered stitch line detection using YOLOv8.

## ✨ Features

- **Futuristic UI**: Sci-fi themed interface with glowing effects and animations
- **Main Menu**: Animated start screen with pulsing buttons
- **Tutorial Mode**: Step-by-step sewing machine setup videos
- **Mode Selection**: Choose between Pattern and Wallet modes
- **Pattern Mode**: 5 levels of sewing patterns with camera overlay
- **AI Stitch Detection**: Real-time stitch line detection using YOLOv8 segmentation model
- **Wallet Mode**: Coming soon!

##  Project Structure

```
SewBot-Rpi/
 main.py                     # Main entry point with AI integration
 stitch_detector.py          # YOLOv8 stitch detection module
 yolov8-stitch-detection.pt  # Trained YOLOv8 model for stitch lines
 ui/                         # User interface components
    __init__.py
    theme.py                # Color scheme and theme configuration
    main_menu.py            # Main menu with Start button
    mode_selection.py       # Pattern/Wallet selection screen
    tutorial.py             # Tutorial video player
 pattern/                    # Pattern recognition module
    __init__.py
 blueprint/                  # Blueprint images folder
     level1.png
     level2.png
     level3.png
     level4.png
     level5.png
 videos/                     # Tutorial videos
     sewing-set-up/
```

##  How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python (Camera and image processing)
- numpy (Numerical computing)
- pygame (Tutorial video playback)
- ultralytics (YOLOv8 for AI stitch detection)
- torch (PyTorch for deep learning)

### 2. Run the Application

```bash
python main.py
```

### 3. Test Stitch Detection (Optional)

```bash
python stitch_detector.py
```

##**opencv-python** (4.8.1.78): Camera and image processing
- **numpy** (1.24.3): Numerical computing
- **pygame** (2.5.2): Tutorial video playback
- **ultralytics** (>=8.0.0): YOLOv8 framework for AI detection
- **torch** (>=2.0.0): PyTorch deep learning framework

##  AI Model Information

The stitch detection model is based on YOLOv8 nano segmentation, trained specifically for detecting stitch lines on fabric. 

- **Model**: YOLOv8n-seg (nano segmentation)
- **Training**: Custom dataset with stitch line annotations
- **Optimized for**: Raspberry Pi (416x416 input size)
- **Performance**: ~3-7 FPS on Raspberry Pi 4
- **Confidence**: Default 0.35 (adjustable in real-time)
- **Classes**: Detects "stitch_line" onl
- **Color Scheme**: Various shades of blue with cyan accents
- **Style**: Futuristic sci-fi with glowing effects
- **Animations**: Pulsing buttons and animated glows
### General Navigation
- **Mouse**: Click buttons to navigate
- **Quit Button**: Click QUIT button to exit (available in main menu and mode selection)
- **Back Button**: Click < BACK to return to previous screen

### Pattern Mode Controls
- **Mouse**: Click level buttons (LVL 1-5) to switch patterns
- **Keyboard Shortcuts**:
  - **D**: Toggle AI stitch detection ON/OFF
  - **+/=**: Increase detection confidence (reduce false positives)
  - **-/_**: Decrease detection confidence (more sensitive)
  - **I**: Toggle detection info display

### AI Detection Features
- ✅ AI-powered stitch line detection (Completed!)
- Wallet functionality with guided steps
- Additional pattern levels
- Enhanced visual effects
- Score tracking system
- Performance metrics and accuracy feedback
- Export detected patterns

##  Troubleshooting

### Model Not Loading
If you see "Model not loaded" error:
1. Ensure `yolov8-stitch-detection.pt` exists in the project root
2. Install ultralytics: `pip install ultralytics`
3. Check that torch is installed: `pip install torch`

### Camera Not Detected
1. Check if another application is using the camera
2. Try different camera indices (0, 1, 2)
3. Verify camera permissions

### Slow Performance on Raspberry Pi
1. Lower the image size in `stitch_detector.py` (change `img_size=416` to `img_size=320`)
2. Increase confidence threshold (reduces processing)
3. Disable AI detection when not needed (press 'D')
4. Ensure good lighting for better detection accuracy

##  Development Notes

- The AI model was trained on a custom dataset from the Yolo-v8-nano project
- Model is optimized for Raspberry Pi with reduced image size (416x416)
- Detection can be toggled on/off for performance optimization
- Blueprint overlay works independently of AI detectiononfidence level display

##  Navigation

1. **Main Menu**: Click "START" to begin
2. **Mode Selection**: 
   - Click "PATTERN" to access levels 1-5
   - Click "WALLET" (coming soon)
3. **Pattern Mode**: Use buttons or keys 1-5 to switch levels
4. **Exit**: Press Q or ESC at any time

##  Dependencies

- OpenCV (cv2)
- NumPy

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
