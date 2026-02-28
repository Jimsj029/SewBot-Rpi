# Setup Script for SewBot-Rpi with AI Detection
# Run this script to install all dependencies

Write-Host "=" -NoNewline; Write-Host ("=" * 59)
Write-Host "SewBot-Rpi AI Integration - Setup Script"
Write-Host "=" -NoNewline; Write-Host ("=" * 59)
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Cyan
$pythonVersion = python --version 2>&1
Write-Host "  $pythonVersion" -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists
$venvPath = ".venv"
if (Test-Path $venvPath) {
    Write-Host "Virtual environment found at: $venvPath" -ForegroundColor Green
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & "$venvPath\Scripts\Activate.ps1"
} else {
    Write-Host "No virtual environment found." -ForegroundColor Yellow
    $createVenv = Read-Host "Do you want to create one? (y/n)"
    
    if ($createVenv -eq "y") {
        Write-Host "Creating virtual environment..." -ForegroundColor Cyan
        python -m venv .venv
        Write-Host "  Virtual environment created!" -ForegroundColor Green
        Write-Host "Activating virtual environment..." -ForegroundColor Cyan
        & ".venv\Scripts\Activate.ps1"
    }
}
Write-Host ""

# Install dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ""
Write-Host "=" -NoNewline; Write-Host ("=" * 59)
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=" -NoNewline; Write-Host ("=" * 59)
Write-Host ""
Write-Host "Installed packages:" -ForegroundColor Cyan
Write-Host "  - opencv-python (Camera and image processing)"
Write-Host "  - numpy (Numerical computing)"
Write-Host "  - pygame (Tutorial videos)"
Write-Host "  - ultralytics (YOLOv8 AI detection)"
Write-Host "  - torch (PyTorch deep learning)"
Write-Host ""

# Verify model file
Write-Host "Verifying AI model file..." -ForegroundColor Cyan
if (Test-Path "yolov8-stitch-detection.pt") {
    $modelSize = (Get-Item "yolov8-stitch-detection.pt").Length / 1MB
    Write-Host "  Model file found: yolov8-stitch-detection.pt ($([math]::Round($modelSize, 2)) MB)" -ForegroundColor Green
} else {
    Write-Host "  WARNING: Model file not found!" -ForegroundColor Red
    Write-Host "  Please copy the trained model to: yolov8-stitch-detection.pt"
}
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run the application: python main.py"
Write-Host "  2. Test AI detection: python stitch_detector.py"
Write-Host "  3. Read the guides:"
Write-Host "     - README.md (General usage)"
Write-Host "     - AI_INTEGRATION_GUIDE.md (AI features)"
Write-Host ""
Write-Host "In Pattern Mode, use these keys:" -ForegroundColor Cyan
Write-Host "  - D: Toggle AI detection ON/OFF"
Write-Host "  - +: Increase confidence (reduce false positives)"
Write-Host "  - -: Decrease confidence (more sensitive)"
Write-Host "  - I: Toggle detection info display"
Write-Host ""
Write-Host "Ready to start! Run: python main.py" -ForegroundColor Green
Write-Host ""
