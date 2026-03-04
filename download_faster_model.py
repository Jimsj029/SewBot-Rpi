"""
Download and Export Optimized YOLO Model for Raspberry Pi 4

This script downloads a faster nano variant of YOLO and exports it to ONNX format
with optimizations for real-time performance on RPi4.

Recommended models (in order of priority):
1. YOLO11n-seg - Latest, 37% fewer parameters than YOLOv8
2. YOLOv8n-seg - Very stable and widely supported
3. YOLOv10n - NMS-free design for faster inference

Usage:
    python download_faster_model.py [model_name]
    
Examples:
    python download_faster_model.py yolo11n-seg  # Recommended
    python download_faster_model.py yolov8n-seg
    python download_faster_model.py yolo10n
"""

import os
import sys
from ultralytics import YOLO

# Configuration
MODELS_DIR = "models"
DEFAULT_MODEL = "yolo11n-seg"  # Recommended for RPi4

# Export settings for RPi4
EXPORT_SETTINGS = {
    'format': 'onnx',
    'imgsz': 320,  # Lower resolution for better FPS (was 640)
    'half': False,  # FP16 not widely supported on RPi4 CPU, use False
    'simplify': True,  # Simplify ONNX graph
    'opset': 12,  # Compatible opset for RPi4
    'dynamic': False,  # Static shapes for faster inference
}


def download_and_export_model(model_name=DEFAULT_MODEL):
    """
    Download a YOLO model and export it to ONNX format optimized for RPi4
    
    Args:
        model_name: Name of the model (e.g., 'yolo11n-seg', 'yolov8n-seg')
    """
    print(f"\n{'='*60}")
    print(f"  SewBot - Fast Model Downloader for Raspberry Pi 4")
    print(f"{'='*60}\n")
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Validate model name
    valid_models = ['yolo11n-seg', 'yolov8n-seg', 'yolo10n', 'yolov10n']
    if model_name not in valid_models:
        print(f"⚠️  Warning: '{model_name}' may not be a valid model name")
        print(f"   Valid options: {', '.join(valid_models)}")
        response = input(f"   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("\n❌ Cancelled")
            return False
    
    try:
        # Step 1: Download the model
        print(f"📥 Downloading {model_name} model...")
        print(f"   This may take a few minutes depending on your connection...")
        model = YOLO(f"{model_name}.pt")
        print(f"✅ Model downloaded successfully!\n")
        
        # Step 2: Display model info
        print(f"📊 Model Information:")
        if hasattr(model, 'names'):
            print(f"   Classes: {len(model.names)}")
            for idx, name in model.names.items():
                print(f"     - Class {idx}: {name}")
        if hasattr(model, 'model'):
            try:
                params = sum(p.numel() for p in model.model.parameters())
                print(f"   Parameters: {params:,}")
            except:
                pass
        print()
        
        # Step 3: Export to ONNX
        print(f"🔄 Exporting to ONNX format...")
        print(f"   Settings:")
        print(f"     - Format: {EXPORT_SETTINGS['format']}")
        print(f"     - Image size: {EXPORT_SETTINGS['imgsz']}x{EXPORT_SETTINGS['imgsz']}")
        print(f"     - Opset: {EXPORT_SETTINGS['opset']}")
        print(f"     - Simplified: {EXPORT_SETTINGS['simplify']}")
        print()
        
        # Export the model
        exported_path = model.export(**EXPORT_SETTINGS)
        print(f"✅ Export completed!\n")
        
        # Step 4: Move to models directory
        base_name = os.path.basename(exported_path)
        target_path = os.path.join(MODELS_DIR, 'best_fast.onnx')
        backup_path = os.path.join(MODELS_DIR, 'best.onnx.backup')
        
        # Backup existing model if it exists
        old_model = os.path.join(MODELS_DIR, 'best.onnx')
        if os.path.exists(old_model):
            print(f"📦 Backing up existing model...")
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.rename(old_model, backup_path)
            print(f"   Backup saved to: {backup_path}")
        
        # Move new model
        print(f"📁 Installing new model...")
        if os.path.exists(exported_path):
            # Copy to both best_fast.onnx and best.onnx
            import shutil
            shutil.copy(exported_path, target_path)
            shutil.copy(exported_path, old_model)
            print(f"   ✓ Saved as: {target_path}")
            print(f"   ✓ Saved as: {old_model} (active model)")
            
            # Clean up original export
            if exported_path != target_path:
                try:
                    os.remove(exported_path)
                except:
                    pass
        
        print(f"\n{'='*60}")
        print(f"✨ SUCCESS! Model ready for SewBot")
        print(f"{'='*60}\n")
        
        print(f"📈 Expected Performance on Raspberry Pi 4:")
        print(f"   - Resolution: 320x320 (optimized for speed)")
        print(f"   - Expected FPS: 15-25 FPS")
        print(f"   - Model size: ~6-8 MB")
        print(f"   - Latency: ~40-60ms per frame\n")
        
        print(f"🎯 Next Steps:")
        print(f"   1. Run your SewBot application")
        print(f"   2. The new model will be automatically loaded")
        print(f"   3. If needed, restore backup: mv {backup_path} {old_model}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_available_models():
    """List available YOLO models"""
    print(f"\n{'='*60}")
    print(f"  Available Fast Models for Raspberry Pi 4")
    print(f"{'='*60}\n")
    
    models = [
        {
            'name': 'yolo11n-seg',
            'description': 'YOLO11 Nano Segmentation (RECOMMENDED)',
            'features': [
                '37% fewer parameters than YOLOv8',
                'Best balance of speed and accuracy',
                'Optimized for edge devices',
                'Latest architecture'
            ]
        },
        {
            'name': 'yolov8n-seg',
            'description': 'YOLOv8 Nano Segmentation',
            'features': [
                'Very stable and widely tested',
                'Great documentation',
                'Slightly faster raw inference',
                'Excellent community support'
            ]
        },
        {
            'name': 'yolo10n',
            'description': 'YOLO10 Nano (NMS-Free)',
            'features': [
                'No Non-Maximum Suppression',
                'Up to 43% faster CPU inference',
                'End-to-end optimized',
                'May need custom training'
            ]
        }
    ]
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']}")
        print(f"   {model['description']}")
        for feature in model['features']:
            print(f"   • {feature}")
        print()


if __name__ == "__main__":
    print()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--list', '-l', 'list']:
            list_available_models()
            sys.exit(0)
        elif sys.argv[1] in ['--help', '-h', 'help']:
            print(__doc__)
            list_available_models()
            sys.exit(0)
        else:
            model_name = sys.argv[1]
    else:
        # Interactive mode
        list_available_models()
        print(f"{'='*60}")
        print(f"Enter model name (or press Enter for default: {DEFAULT_MODEL})")
        print(f"{'='*60}")
        choice = input("> ").strip()
        model_name = choice if choice else DEFAULT_MODEL
    
    # Download and export
    success = download_and_export_model(model_name)
    sys.exit(0 if success else 1)
