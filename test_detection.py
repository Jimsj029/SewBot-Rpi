#!/usr/bin/env python3
"""
Test script to verify YOLO detection model is working
Run this to debug detection issues
"""

import cv2
import numpy as np
import os

print("="*50)
print("Detection Model Test")
print("="*50)

# Test imports
print("\n1. Testing imports...")
try:
    import torch
    print(f"   ✓ PyTorch: {torch.__version__}")
except Exception as e:
    print(f"   ✗ PyTorch: {e}")
    exit(1)

try:
    from ultralytics import YOLO
    print(f"   ✓ Ultralytics imported successfully")
except Exception as e:
    print(f"   ✗ Ultralytics: {e}")
    exit(1)

try:
    import onnxruntime as ort
    print(f"   ✓ ONNX Runtime: {ort.__version__}")
except Exception as e:
    print(f"   ⚠ ONNX Runtime not available: {e}")

# Check model file
print("\n2. Checking model file...")
model_path = os.path.join('models', 'best.onnx')
if os.path.exists(model_path):
    print(f"   ✓ Model found: {model_path}")
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   Model size: {file_size:.2f} MB")
else:
    print(f"   ✗ Model not found: {model_path}")
    exit(1)

# Load model
print("\n3. Loading model...")
try:
    model = YOLO(model_path, task='segment')
    print(f"   ✓ Model loaded successfully")
    print(f"   Task: {model.task}")
    if hasattr(model, 'names'):
        print(f"   Classes: {model.names}")
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test with dummy image
print("\n4. Testing inference with dummy image...")
try:
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model(test_img, conf=0.35, verbose=False)
    print(f"   ✓ Inference successful")
    print(f"   Results type: {type(results)}")
    print(f"   Number of results: {len(results)}")
    if len(results) > 0:
        print(f"   Boxes: {results[0].boxes}")
        if hasattr(results[0], 'masks'):
            print(f"   Masks: {results[0].masks}")
except Exception as e:
    print(f"   ✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test with camera if available
print("\n5. Testing with camera...")
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("   ⚠ Camera not available (this is OK if no camera connected)")
    else:
        ret, frame = cap.read()
        if ret:
            print(f"   ✓ Camera frame captured: {frame.shape}")
            
            # Resize to model input size
            frame_resized = cv2.resize(frame, (640, 640))
            
            # Run detection
            results = model(frame_resized, conf=0.35, verbose=False)
            
            num_detections = 0
            if len(results) > 0 and results[0].boxes is not None:
                num_detections = len(results[0].boxes)
            
            print(f"   Detections found: {num_detections}")
            
            if num_detections > 0:
                for i, box in enumerate(results[0].boxes):
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    print(f"     Detection {i+1}: class={cls}, confidence={conf:.3f}")
            else:
                print("   ⚠ No detections (this is normal if no stitch lines in view)")
                print("   Try pointing camera at fabric with stitch lines")
        else:
            print("   ✗ Failed to read camera frame")
        cap.release()
except Exception as e:
    print(f"   ⚠ Camera test failed: {e}")

print("\n" + "="*50)
print("Test completed!")
print("="*50)
print("\nIf all tests passed, detection should work in the app.")
print("If no detections found, try:")
print("  - Lower confidence threshold (press '-' key in app)")
print("  - Point camera at fabric with visible stitch lines")
print("  - Ensure good lighting")
