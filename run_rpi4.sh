#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SewBot-Rpi  —  RPi 4 launch script
#
# Usage:
#   ./run_rpi4.sh          # normal launch
#   ./run_rpi4.sh --setup  # install/update deps then launch
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ── helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[sewbot] $*"; }
err()  { echo "[sewbot] ERROR: $*" >&2; exit 1; }
warn() { echo "[sewbot] WARNING: $*" >&2; }

# ── display / headless guard ──────────────────────────────────────────────────
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
    log "DISPLAY not set – defaulting to :0"
fi

# ── virtual environment ───────────────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment …"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ── optional: install / update deps ──────────────────────────────────────────
if [ "$1" = "--setup" ]; then
    log "Installing / updating dependencies …"
    pip install --upgrade pip --quiet

    # onnxruntime for RPi (ARM wheel – pre-built by Nightly or Microsoft)
    # Try the official package first; if it fails fall back to the community wheel.
    pip install onnxruntime --quiet || \
        pip install onnxruntime-rpi4 --quiet || \
        warn "onnxruntime install failed – needle detection will be disabled"

    pip install opencv-python-headless --quiet
    pip install pygame --quiet
    pip install numpy --quiet

    log "Dependencies installed."
fi

# ── sanity checks ─────────────────────────────────────────────────────────────
[ -f "$SCRIPT_DIR/main.py" ]          || err "main.py not found in $SCRIPT_DIR"
python3 -c "import cv2"   2>/dev/null || warn "opencv not importable – camera feed may fail"
python3 -c "import pygame" 2>/dev/null || warn "pygame not importable – UI may fail"
python3 -c "import onnxruntime" 2>/dev/null || warn "onnxruntime not importable – needle detection disabled"

# ── RPi 4 performance tuning ──────────────────────────────────────────────────
# Limit OMP/OpenBLAS threads so ORT + OpenCV don't fight over all 4 cores.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Prefer the V4L2 camera backend on Linux / RPi
export OPENCV_VIDEOIO_PRIORITY_V4L2=1

# Disable Ultralytics telemetry/updates (kept for safety in case any import
# still triggers it transitively)
export YOLO_TELEMETRY=false

# ── launch ────────────────────────────────────────────────────────────────────
log "Starting SewBot-Rpi (RPi 4 mode) …"
log "  Python  : $(python3 --version)"
log "  DISPLAY : $DISPLAY"
log "  CWD     : $SCRIPT_DIR"

python3 main.py

deactivate
