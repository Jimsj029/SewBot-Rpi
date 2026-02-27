#!/usr/bin/env python3
"""
Build script for creating a standalone SewBot executable on Raspberry Pi
Requires PyInstaller: pip3 install pyinstaller

Usage:
    python3 build_executable.py
    
This will create a standalone executable in the 'dist' folder.
"""

import os
import sys
import subprocess
import shutil

def main():
    print("=" * 60)
    print("SewBot - PyInstaller Build Script")
    print("=" * 60)
    print()
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print(f"✓ PyInstaller found (version {PyInstaller.__version__})")
    except ImportError:
        print("✗ PyInstaller not found")
        print()
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller installed")
    
    print()
    
    # Check if sewbot_standalone.py exists
    if not os.path.exists("sewbot_standalone.py"):
        print("✗ Error: sewbot_standalone.py not found")
        print("  Make sure you're running this from the SewBot directory")
        sys.exit(1)
    
    print("Building executable...")
    print()
    
    # PyInstaller command
    # --onefile: Bundle everything into a single executable
    # --name: Name of the executable
    # --add-data: Include additional files/directories (blueprint, videos)
    # --hidden-import: Explicitly include modules that might be missed
    
    cmd = [
        "pyinstaller",
        "--onefile",
        "--name=sewbot",
        "--clean",
        "--console",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=pygame",
        "--hidden-import=pygame.mixer",
    ]
    
    # Add data directories if they exist
    if os.path.exists("blueprint"):
        print("✓ Including 'blueprint' directory")
        cmd.append("--add-data=blueprint:blueprint")
    else:
        print("⚠ 'blueprint' directory not found - will not be included")
    
    if os.path.exists("videos"):
        print("✓ Including 'videos' directory")
        cmd.append("--add-data=videos:videos")
    else:
        print("⚠ 'videos' directory not found - will not be included")
    
    print()
    
    cmd.append("sewbot_standalone.py")
    
    try:
        # Run PyInstaller
        result = subprocess.run(cmd, check=True)
        
        print()
        print("=" * 60)
        print("✓ Build completed successfully!")
        print("=" * 60)
        print()
        print("Executable location: dist/sewbot")
        print()
        print("To run the executable:")
        print("  cd dist")
        print("  ./sewbot")
        print()
        print("Note: If you didn't include blueprint/videos directories,")
        print("      you'll need to copy them to the dist folder.")
        print()
        
        # Check file size
        exe_path = os.path.join("dist", "sewbot")
        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print(f"Executable size: {size_mb:.1f} MB")
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("✗ Build failed")
        print("=" * 60)
        print()
        print("Error:", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
