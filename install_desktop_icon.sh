#!/bin/bash
# Create Raspberry Pi desktop/app-menu launcher for SewBot-Rpi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_DIR="${HOME}/Desktop"
APPS_DIR="${HOME}/.local/share/applications"
BIN_DIR="${HOME}/.local/bin"
DESKTOP_FILE_NAME="SewBot-Rpi.desktop"
LAUNCHER_SCRIPT="${BIN_DIR}/sewbot-rpi-launch.sh"
LOG_FILE="${HOME}/sewbot-launch.log"

mkdir -p "${DESKTOP_DIR}" "${APPS_DIR}" "${BIN_DIR}"

ICON_PATH="${SCRIPT_DIR}/images/logo.png"
if [ ! -f "${ICON_PATH}" ]; then
  ICON_PATH=""
fi

cat > "${LAUNCHER_SCRIPT}" <<EOF
#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SCRIPT_DIR}"
LOG_FILE="${LOG_FILE}"

{
  echo "============================================================"
  echo "SewBot launcher start: \\$(date)"
  echo "Project: \\${PROJECT_DIR}"
} >> "${LOG_FILE}"

cd "${SCRIPT_DIR}"

if [ -x "./run_rpi4.sh" ]; then
  nohup ./run_rpi4.sh >> "${LOG_FILE}" 2>&1 &
  disown || true
elif [ -x "./run.sh" ]; then
  nohup ./run.sh >> "${LOG_FILE}" 2>&1 &
  disown || true
else
  echo "ERROR: run_rpi4.sh and run.sh are missing or not executable" >> "${LOG_FILE}"
  exit 1
fi
EOF

chmod +x "${LAUNCHER_SCRIPT}"

DESKTOP_CONTENT="[Desktop Entry]
Version=1.0
Type=Application
Name=SewBot-Rpi
Comment=Launch SewBot-Rpi
Exec=${LAUNCHER_SCRIPT}
Path=${SCRIPT_DIR}
Icon=${ICON_PATH}
Terminal=false
Categories=Education;Graphics;
StartupNotify=true
"

for TARGET in "${DESKTOP_DIR}/${DESKTOP_FILE_NAME}" "${APPS_DIR}/${DESKTOP_FILE_NAME}"; do
  printf "%s" "${DESKTOP_CONTENT}" > "${TARGET}"
  chmod +x "${TARGET}"
done

# Configure PCManFM/LibFM to launch executable text files directly
# (disables the "Execute / Execute in Terminal" prompt).
set_quick_exec() {
  local conf_file="$1"
  local conf_dir
  conf_dir="$(dirname "${conf_file}")"
  mkdir -p "${conf_dir}"

  if [ ! -f "${conf_file}" ]; then
    cat > "${conf_file}" <<'EOF'
[config]
quick_exec=1
EOF
    return
  fi

  if grep -q '^quick_exec=' "${conf_file}"; then
    sed -i 's/^quick_exec=.*/quick_exec=1/' "${conf_file}"
  else
    if grep -q '^\[config\]' "${conf_file}"; then
      sed -i '/^\[config\]/a quick_exec=1' "${conf_file}"
    else
      printf '\n[config]\nquick_exec=1\n' >> "${conf_file}"
    fi
  fi
}

set_quick_exec "${HOME}/.config/libfm/libfm.conf"
set_quick_exec "${HOME}/.config/pcmanfm/LXDE-pi/pcmanfm.conf"
set_quick_exec "${HOME}/.config/pcmanfm/LXDE/pcmanfm.conf"

# GNOME/Caja fallback: launch executable text files directly (no prompt)
if command -v gsettings >/dev/null 2>&1; then
  gsettings set org.gnome.nautilus.preferences executable-text-activation 'launch' >/dev/null 2>&1 || true
  gsettings set org.caja.preferences executable-text-activation 'launch' >/dev/null 2>&1 || true
fi

# Mark Desktop launcher as trusted when supported (GNOME/Caja/Nautilus)
if command -v gio >/dev/null 2>&1; then
  gio set "${DESKTOP_DIR}/${DESKTOP_FILE_NAME}" metadata::trusted true >/dev/null 2>&1 || true
  gio set "${DESKTOP_DIR}/${DESKTOP_FILE_NAME}" metadata::trusted yes >/dev/null 2>&1 || true
fi

# Make sure desktop environments refresh launcher cache
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "${APPS_DIR}" >/dev/null 2>&1 || true
fi

echo "✅ Desktop icon created: ${DESKTOP_DIR}/${DESKTOP_FILE_NAME}"
echo "✅ App menu entry created: ${APPS_DIR}/${DESKTOP_FILE_NAME}"
echo "✅ Launcher script created: ${LAUNCHER_SCRIPT}"
echo "✅ File manager quick-exec enabled (no execute prompt)"
echo "📝 Launch log file: ${LOG_FILE}"
echo "Tip: if Raspberry Pi still asks for permission, right-click the icon and choose 'Allow Launching'."
echo "Tip: log out/in once after installation so desktop settings reload."
