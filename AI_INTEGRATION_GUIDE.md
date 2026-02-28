# AI Integration Guide - YOLOv8 Stitch Detection

## Overview
This guide explains how the YOLOv8 stitch line detection model has been integrated into the SewBot-Rpi project.

## What Was Added

### 1. Model File
- **File**: `yolov8-stitch-detection.pt`
- **Source**: Trained model from Yolo-v8-nano project (train7_rpi)
- **Type**: YOLOv8 nano segmentation model
- **Purpose**: Real-time stitch line detection on camera feed

### 2. Stitch Detector Module
- **File**: `stitch_detector.py`
- **Purpose**: Wrapper class for YOLOv8 inference
- **Features**:
  - Easy-to-use API for stitch detection
  - Confidence threshold adjustment
  - Error handling and fallback
  - Standalone testing capability

### 3. Main Application Integration
- **File**: `main.py` (modified)
- **Changes**:
  - Import StitchDetector class
  - Initialize detector on startup
  - Integrate detection into pattern mode
  - Add keyboard controls for detection settings
  - Display detection status and info

### 4. Dependencies
- **File**: `requirements.txt` (updated)
- **Added**: 
  - `ultralytics>=8.0.0` (YOLOv8 framework)
  - `torch>=2.0.0` (PyTorch)

### 5. Documentation
- **File**: `README.md` (updated)
- **Added**: Complete documentation of AI features, controls, and troubleshooting

## How It Works

### Detection Pipeline
1. Camera captures frame
2. Frame is resized to camera display size (560x420)
3. If detection is enabled:
   - Frame is sent to YOLOv8 model
   - Model performs inference at 416x416 (optimized for RPi)
   - Detected stitch lines are segmented
   - Masks are overlayed on the frame
4. Blueprint pattern is overlayed (if selected)
5. Final frame is displayed

### Performance Optimization
- **Image Size**: 416x416 (smaller = faster)
- **Confidence**: 0.35 default (adjustable)
- **IOU Threshold**: 0.6 (filters overlapping detections)
- **Toggleable**: Can disable detection for better performance

## User Controls

### In Pattern Mode:

| Key | Action |
|-----|--------|
| `D` | Toggle AI detection ON/OFF |
| `+` or `=` | Increase confidence threshold (0.05 increment) |
| `-` or `_` | Decrease confidence threshold (0.05 decrement) |
| `I` | Toggle detection info display |

### Visual Indicators:
- **Green text**: "AI Detection: ON" (bottom right of camera view)
- **Gray text**: "AI Detection: OFF"
- **Detection info**: Shows count and confidence on camera feed

## Model Specifications

### Training Details
- **Base Model**: YOLOv8n-seg (nano segmentation)
- **Dataset**: Custom stitch line annotations
- **Training Epochs**: 100
- **Image Size**: 416x416
- **Optimization**: Raspberry Pi deployment

### Performance Metrics
- **mAP50**: 74% (from train7_rpi)
- **Precision**: 76.9%
- **FPS on RPi 4**: ~3-7 FPS
- **Detection Class**: "stitch_line"

## Adjusting Detection Sensitivity

### Too Many False Positives?
- Press `+` to increase confidence threshold
- Higher confidence = only very certain detections
- Recommended range: 0.40 - 0.60

### Missing Stitch Lines?
- Press `-` to decrease confidence threshold
- Lower confidence = more sensitive detection
- Recommended range: 0.25 - 0.35

### Tips for Best Results:
1. **Good Lighting**: Ensure adequate lighting on the fabric
2. **Contrast**: Use fabric colors that contrast with stitch lines
3. **Camera Angle**: Position camera directly above the work area
4. **Steady Fabric**: Keep fabric as steady as possible

## Testing the Detector

### Standalone Test:
Run the detector module directly to test without the full UI:

```bash
python stitch_detector.py
```

This opens a simple camera window with:
- Real-time detection
- FPS counter
- Detection count
- Confidence adjustment
- Frame saving capability

### Test Controls:
- `q`: Quit
- `+`: Increase confidence
- `-`: Decrease confidence
- `s`: Save current frame

## Code Integration Example

If you want to use the detector in your own code:

```python
from stitch_detector import StitchDetector
import cv2

# Initialize detector
detector = StitchDetector(
    model_path='yolov8-stitch-detection.pt',
    confidence=0.35,
    img_size=416
)

# Check if model loaded
if detector.is_loaded():
    # Open camera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect stitch lines
        annotated_frame, count, results = detector.detect(
            frame,
            show_boxes=False,
            show_labels=False
        )
        
        # Display
        cv2.imshow('Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

## Troubleshooting

### Error: "Model not loaded"
**Cause**: Model file missing or ultralytics not installed
**Solution**:
1. Verify `yolov8-stitch-detection.pt` exists
2. Run: `pip install ultralytics torch`

### Error: "ultralytics not installed"
**Solution**: `pip install ultralytics`

### Low FPS / Slow Performance
**Solutions**:
1. Disable detection when not needed (press `D`)
2. Increase confidence threshold (less processing)
3. Lower image size in `stitch_detector.py`:
   ```python
   detector = StitchDetector(img_size=320)  # Instead of 416
   ```

### Detection Not Accurate
**Solutions**:
1. Improve lighting conditions
2. Adjust confidence threshold
3. Ensure camera is focused
4. Use contrasting thread colors

## Future Enhancements

Possible improvements for the detection system:
1. **Multiple Class Detection**: Detect different stitch types
2. **Pattern Matching**: Compare detected stitches to blueprint
3. **Quality Scoring**: Rate stitch quality and consistency
4. **Real-time Feedback**: Audio/visual cues for accuracy
5. **Record Session**: Save detection results for analysis
6. **Export Models**: TFLite for even better RPi performance

## Architecture Overview

```
SewBot-Rpi Application
    │
    ├── UI Layer (main.py)
    │   ├── Main Menu
    │   ├── Tutorial Mode
    │   ├── Mode Selection
    │   └── Pattern Mode ──┐
    │                      │
    ├── Detection Layer    │
    │   └── StitchDetector ←┘
    │       └── YOLOv8 Model (yolov8-stitch-detection.pt)
    │
    └── Camera Layer (OpenCV)
        └── Video Capture
```

## Credits

- **Model Training**: Yolo-v8-nano project
- **Framework**: Ultralytics YOLOv8
- **Integration**: SewBot-Rpi project
- **Optimization**: Raspberry Pi deployment (train7_rpi)

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the README.md
3. Test with standalone detector: `python stitch_detector.py`
4. Verify all dependencies are installed: `pip install -r requirements.txt`

---

**Last Updated**: February 28, 2026
**Model Version**: train7_rpi (YOLOv8n-seg)
**Integration Status**: ✅ Complete
