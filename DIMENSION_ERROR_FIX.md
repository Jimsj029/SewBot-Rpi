# 🔧 DIMENSION ERROR - FIXED!

## The Error You Saw:
```
INVALID_ARGUMENT: Got invalid dimensions for input: images
 index: 2 Got: 320 Expected: 640
 index: 3 Got: 320 Expected: 640
```

## ✅ What Was Fixed:

1. **Auto-detection added** - Code now reads the model's expected input size automatically
2. **Works with any model** - Both 640x640 and 320x320 models work now
3. **No more crashes** - Code adapts to your model's requirements

## 🚀 Quick Fix (2 Steps):

### Step 1: Install ONNX package (required for auto-detection)

**On Raspberry Pi:**
```bash
./install_onnx.sh
```

**Or manually:**
```bash
pip install onnx onnxruntime
```

### Step 2: Run SewBot
```bash
python main.py
```

**That's it!** Your existing model will now work. The code auto-detects the 640x640 size.

## 💡 Optional: Get 2-3x Better Performance

After confirming it works, download an optimized 320x320 model for much better FPS:

```bash
python download_faster_model.py yolo11n-seg
python main.py
```

## 📊 What You'll See:

When you run `python main.py`, the console will show:

```
Loading stitch detection model: models/best.onnx
  ✓ Auto-detected model input size: 640x640
✓ Stitch detection model loaded successfully!
  ✓ Model test successful! (inference: 120ms)
  ✓ Using 640x640 input size
  ✓ Expected FPS: 8.3
  💡 TIP: For 2-3x faster inference, run:
     python download_faster_model.py yolo11n-seg
```

If you see the tip above, your model is working but could be faster!

## 🎯 Performance Comparison:

| Model | Input Size | FPS | Inference Time |
|-------|-----------|-----|----------------|
| Your current model | 640x640 | ~5-8 | ~120-200ms |
| **Optimized YOLO11n-seg** | **320x320** | **15-25** | **40-60ms** |

## ❓ Still Having Issues?

### Issue: "No module named 'onnx'"
```bash
pip install onnx onnxruntime
```

### Issue: Still getting dimension error
Make sure onnx is installed and restart:
```bash
pip list | grep onnx  # Check if installed
pip install --upgrade onnx onnxruntime
python main.py
```

### Issue: Model not found
```bash
ls -la models/best.onnx  # Check if file exists
```

If missing, download one:
```bash
python download_faster_model.py yolo11n-seg
```

## 📖 More Info:

- [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) - Full overview
- [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) - Detailed guide
- [install_onnx.sh](install_onnx.sh) - Installation script (RPi)
- [install_onnx.ps1](install_onnx.ps1) - Installation script (Windows)

---

**TL;DR:** Run `./install_onnx.sh` then `python main.py` - it works now! 🎉
