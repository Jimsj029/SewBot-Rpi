# 🚀 SewBot RPi4 Performance Optimization Guide

## Overview
This guide explains how to use faster YOLO models optimized for Raspberry Pi 4 to achieve 15-25 FPS in real-time detection.

## What's Been Optimized

### 1. Model Selection
- **YOLO11n-seg** (Recommended) - 37% fewer parameters than YOLOv8
- **YOLOv8n-seg** - Stable and well-tested alternative
- Input size reduced from 640x640 to 320x320 for 2-3x faster inference

### 2. Performance Settings
- Static input shapes for faster ONNX processing
- Optimized detection limits (max 100 detections)
- FP32 precision (FP16 not supported on RPi4 CPU)
- Simplified ONNX graph for reduced overhead

### 3. Real-time Metrics
- **FPS Display**: Shows actual frames per second
- **Inference Time**: Shows model processing time in milliseconds
- **Performance Tracking**: Averages over last 30 frames

## Quick Start

### Step 1: Download Optimized Model

Run the model download script:

```bash
# Activate virtual environment (if using one)
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Download YOLO11n-seg (recommended)
python download_faster_model.py yolo11n-seg

# Or choose another model
python download_faster_model.py yolov8n-seg
```

### Step 2: Run SewBot

Your SewBot will automatically use the new optimized model:

```bash
python main.py
```

## Expected Performance

### Before Optimization
- Resolution: 640x640
- FPS: ~5-8 FPS
- Inference Time: ~120-200ms
- Model Size: ~20-30 MB

### After Optimization
- Resolution: 320x320
- FPS: **15-25 FPS** ✨
- Inference Time: **40-60ms** ✨
- Model Size: **6-8 MB** ✨

## Performance Monitoring

The stats panel (right side of screen) now shows:

```
┌─────────────┐
│   STATS     │
├─────────────┤
│ ACCURACY    │
│ SCORE       │
│ PROGRESS    │
│             │
│ FPS: 18.5   │ ← Real-time FPS
│ INFERENCE   │ ← Processing time
│    45ms     │
└─────────────┘
```

### FPS Color Coding
- **Cyan/Bright**: FPS ≥ 15 (Good performance)
- **Gray**: FPS < 15 (May need optimization)

## Troubleshooting

### Issue: Lower FPS than expected

**Solutions:**
1. Ensure you're using the latest model:
   ```bash
   python download_faster_model.py yolo11n-seg
   ```

2. Check CPU usage:
   ```bash
   top  # Linux/Mac
   Get-Process | Sort-Object CPU -Descending | Select-Object -First 10  # Windows PowerShell
   ```

3. Close other applications to free up CPU resources

4. Try lowering the confidence threshold in `pattern_mode.py`:
   ```python
   self.confidence_threshold = 0.30  # Lower = faster but less accurate
   ```

### Issue: Model not found

Make sure the model is in the correct location:
```bash
ls -la models/best.onnx  # Should exist
```

If missing, download it:
```bash
python download_faster_model.py yolo11n-seg
```

### Issue: "Illegal Instruction" error

This means you're using incompatible binaries. Run the setup script:
```bash
./setup.sh  # Use system packages for ARM compatibility
```

## Advanced Optimization

### Further Speed Improvements

If you need even more speed, you can modify `pattern_mode.py`:

```python
# Change model input size (line ~53)
self.MODEL_INPUT_SIZE = 256  # Even faster (was 320)

# Reduce max detections (line ~54)
self.MAX_DET = 50  # Fewer detections = faster (was 100)

# Lower confidence threshold (line ~50)
self.confidence_threshold = 0.25  # Accept more detections
```

### Model Comparison

| Model | Params | Speed | Accuracy | Best For |
|-------|--------|-------|----------|----------|
| YOLO11n-seg | ~3M | ⚡⚡⚡ | ⭐⭐⭐ | **Recommended** - Best balance |
| YOLOv8n-seg | ~3.5M | ⚡⚡ | ⭐⭐⭐⭐ | Stability & accuracy |
| YOLO10n | ~2.5M | ⚡⚡⚡⚡ | ⭐⭐ | Maximum speed (needs retrain) |

## Backup & Restore

### Backup Current Model
```bash
cp models/best.onnx models/best.onnx.my_backup
```

### Restore Backup
```bash
cp models/best.onnx.backup models/best.onnx
```

The download script automatically creates backups before installing new models.

## Performance Tips

### 1. Use ROI Detection
The code already uses Region of Interest (ROI) detection to process only relevant areas - this is automatic!

### 2. Monitor System Resources
```bash
# Check temperature
vcgencmd measure_temp  # RPi4 only

# Check CPU frequency
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

### 3. Ensure Good Cooling
RPi4 throttles at 80°C. Use a heatsink or fan for sustained performance.

### 4. Use Performance Governor
```bash
# Switch to performance mode (RPi4)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## Testing Different Models

To test different models:

```bash
# Test YOLO11n-seg
python download_faster_model.py yolo11n-seg
python main.py

# Test YOLOv8n-seg
python download_faster_model.py yolov8n-seg  
python main.py

# Compare performance using the FPS display
```

## Model Files Location

```
models/
├── best.onnx              # Active model (used by SewBot)
├── best.onnx.backup       # Previous model (auto-created)
├── best_fast.onnx         # Copy of latest downloaded model
├── best_opset22_backup.onnx  # Old backup (if exists)
└── *.pt                   # PyTorch weights (optional)
```

## Questions?

Check existing documentation:
- [TROUBLESHOOTING_ILLEGAL_INSTRUCTION.md](TROUBLESHOOTING_ILLEGAL_INSTRUCTION.md) - ARM compatibility issues
- [FIX_INSTRUCTIONS.md](FIX_INSTRUCTIONS.md) - General fixes
- [SEWBOT_USAGE.md](SEWBOT_USAGE.md) - Usage guide

## Summary

✅ **Merge conflict fixed** in main.py  
✅ **Optimized inference** settings (320x320 input)  
✅ **Fast model downloader** script created  
✅ **Real-time performance metrics** added to UI  
✅ **2-3x FPS improvement** expected (15-25 FPS on RPi4)  

Run `python download_faster_model.py yolo11n-seg` to get started! 🚀
