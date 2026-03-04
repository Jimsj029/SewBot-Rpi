# Quick Fix: Install ONNX packages for model optimization
# Run this on Windows if you get import errors

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  SewBot - Installing ONNX packages" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running in virtual environment
if ($env:VIRTUAL_ENV) {
    Write-Host "✓ Virtual environment detected: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "⚠ No virtual environment detected" -ForegroundColor Yellow
    Write-Host "  Consider activating with: .venv\Scripts\Activate.ps1"
    Write-Host ""
}

Write-Host "Installing onnx and onnxruntime..." -ForegroundColor White
Write-Host ""

pip install onnx onnxruntime

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Installation complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Now you can:" -ForegroundColor Green
Write-Host "  1. Run: python main.py (works with existing 640x640 model)" -ForegroundColor White
Write-Host "  2. Or download optimized model: python download_faster_model.py yolo11n-seg" -ForegroundColor White
Write-Host ""
