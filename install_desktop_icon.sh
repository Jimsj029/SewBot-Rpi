#!/bin/bash
# Create Raspberry Pi desktop/app-menu launcher for SewBot-Rpi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_DIR="${HOME}/Desktop"
APPS_DIR="${HOME}/.local/share/applications"
DESKTOP_FILE_NAME="SewBot-Rpi.desktop"

mkdir -p "${DESKTOP_DIR}" "${APPS_DIR}"

ICON_PATH="${SCRIPT_DIR}/images/logo.png"
if [ ! -f "${ICON_PATH}" ]; then
  ICON_PATH=""
fi

LAUNCH_COMMAND="bash -lc 'cd \"${SCRIPT_DIR}\"; if [ -x ./run_rpi4.sh ]; then ./run_rpi4.sh; else ./run.sh; fi'"

DESKTOP_CONTENT="[Desktop Entry]
Version=1.0
Type=Application
Name=SewBot-Rpi
Comment=Launch SewBot-Rpi
Exec=${LAUNCH_COMMAND}
Path=${SCRIPT_DIR}
Icon=${ICON_PATH}
Terminal=true
Categories=Education;Graphics;
StartupNotify=true
"

for TARGET in "${DESKTOP_DIR}/${DESKTOP_FILE_NAME}" "${APPS_DIR}/${DESKTOP_FILE_NAME}"; do
  printf "%s" "${DESKTOP_CONTENT}" > "${TARGET}"
  chmod +x "${TARGET}"
done

# Mark Desktop launcher as trusted when supported (GNOME/Caja/Nautilus)
if command -v gio >/dev/null 2>&1; then
  gio set "${DESKTOP_DIR}/${DESKTOP_FILE_NAME}" metadata::trusted true >/dev/null 2>&1 || true
fi

echo "✅ Desktop icon created: ${DESKTOP_DIR}/${DESKTOP_FILE_NAME}"
echo "✅ App menu entry created: ${APPS_DIR}/${DESKTOP_FILE_NAME}"
echo "Tip: if Raspberry Pi still asks for permission, right-click the icon and choose 'Allow Launching'."
