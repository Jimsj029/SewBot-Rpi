# SewBot - Thonny IDE Setup Guide

This folder contains the SewBot project optimized for use with Thonny IDE.

## Setup Instructions

### 1. Install Thonny IDE
- Download Thonny from: https://thonny.org/
- Install it on your computer

### 2. Install Required Packages
1. Open Thonny IDE
2. Go to **Tools** → **Manage packages...**
3. Install the following packages one by one:
   - `opencv-python` (version 4.8.1.78 or newer)
   - `numpy` (version 1.24.3 or newer)
   - `pygame` (version 2.5.2 or newer)

### 3. Open the Project
1. In Thonny, go to **File** → **Open...**
2. Navigate to this folder (SewBot_Thonny)
3. Open `main.py`

### 4. Run the Application
- Click the **Run** button (green play icon) or press **F5**
- The SewBot application will start

## Project Structure
```
SewBot_Thonny/
├── main.py                 # Main application file - RUN THIS
├── ui/                     # User interface modules
│   ├── __init__.py
│   ├── main_menu.py
│   ├── mode_selection.py
│   ├── theme.py
│   └── tutorial.py
├── pattern/                # Pattern recognition modules
│   ├── __init__.py
│   └── blueprint_viewer.py
├── blueprint/              # Pattern images (level1-5.png)
├── requirements.txt        # Python package requirements
└── THONNY_SETUP.md        # This file

```

## Troubleshooting

### Camera Not Working
- Make sure your camera is connected
- Try closing other applications that might be using the camera
- Check camera permissions on your system

### Module Import Errors
- Ensure all packages from requirements.txt are installed
- Restart Thonny after installing packages

### Performance Issues
- Close other applications to free up system resources
- Consider using a more powerful computer if running on Raspberry Pi

## Features
- **Main Menu**: Start the application
- **Tutorial Mode**: Learn how to use SewBot
- **Pattern Mode**: Practice pattern recognition with 5 difficulty levels
- **Blueprint Mode**: View and study pattern blueprints

## Tips for Thonny Users
- Use **View** → **Variables** to see all variables while debugging
- Use **View** → **Plotter** to visualize numerical data
- Press **Ctrl+M** to toggle between regular and simple mode
- Use the built-in debugger to step through code

## Need Help?
Check the main README.md file for more detailed information about the project.
