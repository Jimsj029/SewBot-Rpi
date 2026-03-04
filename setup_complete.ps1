# SewBot - Complete Setup & Fix Script (Windows)
# This script fixes all common errors and sets up your environment
# Requires: Python 3.10+

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  SewBot - Complete Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..."
try {
    $PythonVersion = python --version 2>&1 | Select-String -Pattern '\d+\.\d+' | ForEach-Object { $_.Matches.Value }
    $VersionParts = $PythonVersion.Split('.')
    $Major = [int]$VersionParts[0]
    $Minor = [int]$VersionParts[1]
    
    if ($Major -lt 3 -or ($Major -eq 3 -and $Minor -lt 10)) {
        Write-Host "❌ ERROR: Python $PythonVersion detected" -ForegroundColor Red
        Write-Host "   SewBot requires Python 3.10 or higher" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "   Download from: https://www.python.org/downloads/" -ForegroundColor White
        Write-Host ""
        exit 1
    } else {
        Write-Host "✓ Python $PythonVersion (compatible)" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ ERROR: Python not found" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Check if running in virtual environment
if ($env:VIRTUAL_ENV) {
    Write-Host "✓ Virtual environment detected: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "⚠ No virtual environment detected" -ForegroundColor Yellow
    Write-Host "  Activating .venv..."
    if (Test-Path ".venv") {
        & ".venv\Scripts\Activate.ps1"
        Write-Host "✓ Virtual environment activated" -ForegroundColor Green
    } else {
        Write-Host "  Creating virtual environment..."
        python -m venv .venv
        & ".venv\Scripts\Activate.ps1"
        Write-Host "✓ Virtual environment created and activated" -ForegroundColor Green
    }
}
Write-Host ""

# Step 1: Fix NumPy compatibility
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Step 1/3: Fixing NumPy compatibility" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

try {
    $NumpyVersion = python -c "import numpy; print(numpy.__version__)" 2>$null
    if ($NumpyVersion -like "2.*") {
        Write-Host "⚠️  NumPy $NumpyVersion detected - needs downgrade" -ForegroundColor Yellow
        Write-Host "   Uninstalling NumPy 2.x..."
        pip uninstall -y numpy
        Write-Host "   Installing NumPy 1.x (compatible)..."
        pip install "numpy<2.0"
        $NumpyVersion = python -c "import numpy; print(numpy.__version__)"
        Write-Host "✓ NumPy $NumpyVersion installed" -ForegroundColor Green
    } elseif ($NumpyVersion -like "1.*") {
        Write-Host "✓ NumPy $NumpyVersion (already compatible)" -ForegroundColor Green
    }
} catch {
    Write-Host "   Installing NumPy 1.x..."
    pip install "numpy<2.0"
    Write-Host "✓ NumPy installed" -ForegroundColor Green
}
Write-Host ""

# Step 2: Install ONNX packages
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Step 2/3: Installing ONNX packages" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

try {
    python -c "import onnx" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ ONNX already installed" -ForegroundColor Green
    } else {
        throw
    }
} catch {
    Write-Host "   Installing onnx and onnxruntime..."
    pip install onnx onnxruntime
    Write-Host "✓ ONNX packages installed" -ForegroundColor Green
}
Write-Host ""

# Step 3: Check for optimized model
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Step 3/3: Checking model" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path "models\best.onnx") {
    Write-Host "✓ Model found: models\best.onnx" -ForegroundColor Green
    
    try {
        $ModelSize = python -c "import onnx; m=onnx.load('models/best.onnx'); print(m.graph.input[0].type.tensor_type.shape.dim[2].dim_value)" 2>$null
        
        if ($ModelSize -eq "640") {
            Write-Host "  Model input: 640x640 (works but slower)" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "  💡 TIP: For 2-3x better FPS, download optimized model:" -ForegroundColor Cyan
            Write-Host "     python download_faster_model.py yolo11n-seg" -ForegroundColor White
        } elseif ($ModelSize -eq "320") {
            Write-Host "  Model input: 320x320 (optimized! ✨)" -ForegroundColor Green
        } else {
            Write-Host "  Model input: ${ModelSize}x${ModelSize}" -ForegroundColor White
        }
    } catch {
        Write-Host "  Model exists" -ForegroundColor White
    }
} else {
    Write-Host "⚠️  Model not found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   Download a model with:" -ForegroundColor White
    Write-Host "     python download_faster_model.py yolo11n-seg" -ForegroundColor White
}
Write-Host ""

# Final verification
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Verification" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Python: " -NoNewline
python --version

Write-Host "NumPy: " -NoNewline
try { python -c "import numpy; print(numpy.__version__)" } catch { Write-Host "Not installed" -ForegroundColor Red }

Write-Host "OpenCV: " -NoNewline
try { python -c "import cv2; print(cv2.__version__)" } catch { Write-Host "Not installed" -ForegroundColor Red }

Write-Host "ONNX: " -NoNewline
try { python -c "import onnx; print(onnx.__version__)" } catch { Write-Host "Not installed" -ForegroundColor Red }

Write-Host "Ultralytics: " -NoNewline
try { python -c "from ultralytics import YOLO; print('OK')" } catch { Write-Host "Not installed" -ForegroundColor Red }

Write-Host ""

# Check if everything is ready
try {
    python -c "import numpy, cv2, onnx; from ultralytics import YOLO" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "==========================================" -ForegroundColor Cyan
        Write-Host "  ✅ Setup Complete!" -ForegroundColor Cyan
        Write-Host "==========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Run your application with:" -ForegroundColor Green
        Write-Host "  python main.py" -ForegroundColor White
        Write-Host ""
    }
} catch {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "  ⚠️  Some packages missing" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "If you see errors above, try:" -ForegroundColor Yellow
    Write-Host "  1. Check internet connection" -ForegroundColor White
    Write-Host "  2. Run: pip install ultralytics opencv-python" -ForegroundColor White
    Write-Host ""
}
