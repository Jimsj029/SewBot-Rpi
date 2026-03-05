import onnxruntime as ort
import numpy as np
import os

print("=" * 50)
print("Direct ONNX Model Test")
print("=" * 50)

model_path = "models/best.onnx"
print(f"ONNX Runtime version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")

try:
    # Try to load with just CPU provider
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print("✅ Model loaded successfully!")
    
    # Get model info
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    print(f"\nModel inputs:")
    for inp in inputs:
        print(f"  - {inp.name}: {inp.shape} ({inp.type})")
    
    print(f"\nModel outputs:")
    for out in outputs:
        print(f"  - {out.name}: {out.shape} ({out.type})")
    
    # Try a dummy inference
    print("\n🔄 Testing inference with dummy data...")
    dummy_input = np.random.randn(1, 3, 320, 320).astype(np.float32)
    results = session.run(None, {inputs[0].name: dummy_input})
    print(f"✅ Inference successful!")
    print(f"   Output shapes: {[r.shape for r in results]}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    
    # If opset error, show clear instructions
    if "opset" in str(e).lower():
        print("\n" + "!" * 50)
        print("OPSET VERSION MISMATCH")
        print("!" * 50)
        print("\nYour model uses opset 22 but ONNX Runtime 1.19.2")
        print("only supports up to opset 21.")
        print("\nSOLUTION: Convert the model using:")
        print("  python convert_model.py")
