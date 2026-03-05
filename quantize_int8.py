#!/usr/bin/env python3
"""
INT8 Quantization Script for ONNX Models
Converts best.onnx to best_int8.onnx using dynamic quantization
"""

import os
from onnxruntime.quantization import quantize_dynamic, QuantType

print("=" * 60)
print("ONNX INT8 Quantization")
print("=" * 60)

model_input = "models/best.onnx"
model_output = "models/best_int8.onnx"

# Check if input model exists
if not os.path.exists(model_input):
    print(f"❌ Error: Model not found at {model_input}")
    exit(1)

print(f"📂 Input model: {model_input}")
print(f"📂 Output model: {model_output}")
print(f"🔄 Running INT8 quantization...")

try:
    quantize_dynamic(
        model_input=model_input,
        model_output=model_output,
        weight_type=QuantType.QInt8
    )
    
    # Check output file size
    input_size = os.path.getsize(model_input) / (1024 * 1024)  # MB
    output_size = os.path.getsize(model_output) / (1024 * 1024)  # MB
    compression_ratio = (1 - output_size / input_size) * 100
    
    print(f"✅ Quantization successful!")
    print(f"📊 Original size: {input_size:.2f} MB")
    print(f"📊 Quantized size: {output_size:.2f} MB")
    print(f"📊 Compression: {compression_ratio:.1f}% reduction")
    print(f"✅ Model saved to: {model_output}")
    
except Exception as e:
    print(f"❌ Quantization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
