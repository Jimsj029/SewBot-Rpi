import onnx
from onnx import version_converter
import os
import shutil

print("=" * 60)
print("ONNX Model Opset Converter for Raspberry Pi")
print("=" * 60)

model_path = "models/best.onnx"
backup_path = "models/best_opset22_backup.onnx"

# Check if model exists
if not os.path.exists(model_path):
    print(f"❌ Error: Model not found at {model_path}")
    exit(1)

# Load model
print(f"📂 Loading model: {model_path}")
model = onnx.load(model_path)

# Check current opset
current_opset = model.opset_import[0].version
print(f"📊 Current opset version: {current_opset}")

if current_opset <= 21:
    print(f"✅ Model already uses opset {current_opset} (compatible)")
    print("No conversion needed!")
    exit(0)

print(f"⚠️  Model uses opset {current_opset} (requires ONNX Runtime 1.20+)")
print(f"🔄 Converting from opset {current_opset} to opset 21...")

# Create backup
print("💾 Creating backup...")
shutil.copy2(model_path, backup_path)
print(f"✅ Backup saved to: {backup_path}")

try:
    # Convert to opset 21
    converted_model = version_converter.convert_version(model, 21)
    
    # Save the converted model (overwrite original)
    onnx.save(converted_model, model_path)
    print(f"✅ Converted model saved to: {model_path}")
    
    # Verify conversion
    verified = onnx.load(model_path)
    new_opset = verified.opset_import[0].version
    print(f"✅ Verified: Model now uses opset {new_opset}")
    
    print("\n" + "=" * 60)
    print("🎉 CONVERSION SUCCESSFUL!")
    print("=" * 60)
    print("\nYou can now run your detection script:")
    print("  python test_detection.py")
    
except Exception as e:
    print(f"❌ Conversion failed: {e}")
    print("🔄 Restoring from backup...")
    shutil.copy2(backup_path, model_path)
    print("✅ Original model restored")
    exit(1)
