#!/usr/bin/env python3
"""
Complete ONNX Model Setup and Conversion Script for Raspberry Pi
This script handles everything: installing dependencies, converting models, and testing.
"""

import subprocess
import sys
import os
import shutil

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def install_package(package_name):
    """Install a Python package if not already installed"""
    try:
        __import__(package_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"📦 Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}: {e}")
            return False

def convert_onnx_model(model_path="models/best.onnx"):
    """Convert ONNX model from opset 22 to opset 21"""
    print_header("ONNX Model Opset Converter")
    
    backup_path = model_path.replace(".onnx", "_opset22_backup.onnx")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return False
    
    # Import onnx (should be installed by now)
    try:
        import onnx
        from onnx import version_converter
    except ImportError:
        print("❌ Failed to import onnx. Installation may have failed.")
        return False
    
    # Load model
    print(f"📂 Loading model: {model_path}")
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Check current opset
    current_opset = model.opset_import[0].version
    print(f"📊 Current opset version: {current_opset}")
    
    if current_opset <= 21:
        print(f"✅ Model already uses opset {current_opset} (compatible with RPi)")
        print("   No conversion needed!")
        return True
    
    print(f"⚠️  Model uses opset {current_opset} (requires ONNX Runtime 1.20+)")
    print(f"🔄 Converting from opset {current_opset} to opset 21...")
    
    # Create backup
    print("💾 Creating backup...")
    try:
        shutil.copy2(model_path, backup_path)
        print(f"✅ Backup saved to: {backup_path}")
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False
    
    # Convert the model
    try:
        converted_model = version_converter.convert_version(model, 21)
        onnx.save(converted_model, model_path)
        print(f"✅ Converted model saved to: {model_path}")
        
        # Verify conversion
        verified = onnx.load(model_path)
        new_opset = verified.opset_import[0].version
        print(f"✅ Verified: Model now uses opset {new_opset}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        print("🔄 Restoring from backup...")
        try:
            shutil.copy2(backup_path, model_path)
            print("✅ Original model restored")
        except Exception as restore_error:
            print(f"❌ Failed to restore backup: {restore_error}")
        return False

def test_onnx_model(model_path="models/best.onnx"):
    """Test the ONNX model with ONNX Runtime"""
    print_header("Testing ONNX Model")
    
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError as e:
        print(f"❌ Failed to import required packages: {e}")
        return False
    
    print(f"📊 ONNX Runtime version: {ort.__version__}")
    print(f"📊 Available providers: {ort.get_available_providers()}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return False
    
    try:
        # Try to load the model
        print(f"\n🔄 Loading model: {model_path}")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print("✅ Model loaded successfully!")
        
        # Get model info
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print(f"\n📋 Model inputs:")
        for inp in inputs:
            print(f"   - {inp.name}: {inp.shape} ({inp.type})")
        
        print(f"\n📋 Model outputs:")
        for out in outputs:
            print(f"   - {out.name}: {out.shape} ({out.type})")
        
        # Try a dummy inference
        print(f"\n🔄 Testing inference with dummy data...")
        input_shape = inputs[0].shape
        # Handle dynamic dimensions
        test_shape = [1 if isinstance(dim, str) else dim for dim in input_shape]
        dummy_input = np.random.randn(*test_shape).astype(np.float32)
        
        results = session.run(None, {inputs[0].name: dummy_input})
        print(f"✅ Inference successful!")
        print(f"   Output shapes: {[r.shape for r in results]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        
        # Check if it's an opset error
        error_str = str(e).lower()
        if "opset" in error_str:
            print("\n" + "!" * 70)
            print("  OPSET VERSION MISMATCH DETECTED")
            print("!" * 70)
            print("\n⚠️  Your model uses a newer opset version than ONNX Runtime supports.")
            print("   This can happen if the model was trained with a newer version.")
            print("\n💡 The conversion should have fixed this. If you see this error,")
            print("   the model may need to be re-exported from the training environment.")
        
        return False

def main():
    """Main execution function"""
    print_header("SewBot-Rpi ONNX Model Setup & Conversion")
    print("This script will:")
    print("  1. Install required packages (onnx)")
    print("  2. Convert ONNX model from opset 22 to opset 21")
    print("  3. Test the converted model")
    print("=" * 70)
    
    # Step 1: Install dependencies
    print_header("Step 1: Installing Dependencies")
    if not install_package("onnx"):
        print("\n❌ Failed to install dependencies. Aborting.")
        sys.exit(1)
    
    # Step 2: Convert the model
    print_header("Step 2: Converting ONNX Model")
    model_path = "models/best.onnx"
    if not convert_onnx_model(model_path):
        print("\n⚠️  Model conversion failed or was not needed.")
        print("   Proceeding to test anyway...")
    
    # Step 3: Test the model
    print_header("Step 3: Testing Converted Model")
    if test_onnx_model(model_path):
        print_header("✅ SUCCESS!")
        print("\n🎉 Your model is ready to use on Raspberry Pi!")
        print("\nNext steps:")
        print("  1. Copy this repository to your Raspberry Pi")
        print("  2. Run: bash setup.sh")
        print("  3. Run: python test_detection.py")
        print("  4. Run: bash run.sh")
    else:
        print_header("⚠️  TEST FAILED")
        print("\n❌ Model testing failed. Please check the errors above.")
        print("\nPossible solutions:")
        print("  1. Make sure onnxruntime is installed: pip install onnxruntime")
        print("  2. Check if the model file is corrupted")
        print("  3. Try re-exporting the model from training with opset=21")
        sys.exit(1)
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
