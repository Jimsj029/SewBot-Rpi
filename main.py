"""
SewBot - Pattern Recognition System
Optimized for Raspberry Pi
"""

import cv2
import numpy as np
import math
import os
import sys
import threading

# Add ui directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ui'))
from tutorial import TutorialPlayer
from wallet_tutorial import WalletTutorialPlayer
from level_selection import LevelSelection
from pattern_mode import PatternMode
from music_manager import get_music_manager

_UI_FONT = cv2.FONT_HERSHEY_DUPLEX

def _put_text(img, text, x, y, scale, color, thickness):
    """Draw text with a 1-px black outline for readability on any background."""
    cv2.putText(img, text, (x + 1, y + 1), _UI_FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),         _UI_FONT, scale, color,     thickness,     cv2.LINE_AA)


class SewBotApp:
    def __init__(self):
        self.width = 1024
        self.height = 600
        self.window_name = 'SewBot - Pattern Recognition System'
        self.state = 'main_menu'  # main_menu, tutorial, wallet_tutorial, mode_selection, level_selection, pattern
        self.previous_state = None  # Track previous state for music transitions
        self.glow_phase = 0
        self.running = True
        self.fullscreen = False  # Fullscreen state
        
        # Theme colors
        self.COLORS = {
            'dark_blue': (80, 40, 20),
            'medium_blue': (180, 100, 50),
            'bright_blue': (255, 180, 100),
            'cyan': (255, 255, 0),
            'neon_blue': (255, 200, 0),
            'button_normal': (200, 120, 40),
            'button_hover': (255, 180, 80),
            'text_primary': (255, 255, 255),
            'text_secondary': (200, 200, 200),
            'text_accent': (255, 255, 0),
            'bg_dark': (60, 30, 10),
            'glow_cyan': (255, 255, 0),
            'glow_blue': (255, 150, 0),
        }
        
        # Tutorial player - initialize with tutorial videos from videos/sewing-set-up
        self.tutorial_player = TutorialPlayer(self.width, self.height)
        
        # Wallet tutorial player - initialize with wallet videos
        self.wallet_tutorial_player = WalletTutorialPlayer(self.width, self.height)
        
        # Level selection screen
        self.level_selection = LevelSelection(self.width, self.height, self.COLORS)
        
        # Camera variables
        self.camera = None
        self.camera_initializing = False
        self.camera_detected = False
        self.camera_status_message = "Checking camera..."
        
        # Button positions
        self.start_button = {'x': (self.width - 300) // 2, 'y': self.height // 2 + 50, 'w': 300, 'h': 80}
        
        # Mode selection buttons - vertically stacked with descriptions
        button_width = 500
        button_height = 90
        button_x = (self.width - button_width) // 2
        start_y = 140
        spacing = 20
        
        self.pattern_button = {
            'x': button_x, 
            'y': start_y, 
            'w': button_width, 
            'h': button_height, 
            'title': 'PATTERN',
            'description': 'Practice simple stitching patterns to build control and accuracy.'
        }
        self.wallet_button = {
            'x': button_x, 
            'y': start_y + button_height + spacing, 
            'w': button_width, 
            'h': button_height, 
            'title': 'WALLET',
            'description': 'Create a simple wallet by following guided steps and videos.'
        }
        self.tutorial_button = {
            'x': button_x, 
            'y': start_y + (button_height + spacing) * 2, 
            'w': button_width, 
            'h': button_height, 
            'title': 'TUTORIAL',
            'description': 'Review sewing machine setup and basic controls anytime.'
        }
        
        self.back_button = {'x': 20, 'y': self.height - 60, 'w': 120, 'h': 40}
        self.quit_button = {'x': self.width - 140, 'y': self.height - 60, 'w': 120, 'h': 40}
        
        # Mute button (upper right corner)
        self.mute_button = {'x': self.width - 70, 'y': 20, 'w': 50, 'h': 50}
        self.is_muted = False
        
        # Music manager
        self.music_manager = get_music_manager()
        
        # Pattern mode - separate module
        self.pattern_mode = PatternMode(self.width, self.height, self.COLORS, 'blueprint')
        
        # Load logo image for main menu
        self.logo_img = None
        self.load_logo()
        
        # Pre-render grid background (performance optimization)
        self.grid_background = self.create_grid_background()
        
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
        except:
            print("Failed to create window")
            self.running = False
        
        # Detect camera at startup
        self.detect_camera_at_startup()
    
    def load_logo(self):
        """Load the logo image for main menu"""
        logo_path = os.path.join(os.path.dirname(__file__), 'images', 'logo.png')
        if os.path.exists(logo_path):
            try:
                self.logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
                if self.logo_img is not None:
                    # Resize logo to appropriate size (maintain aspect ratio)
                    target_height = 330
                    h, w = self.logo_img.shape[:2]
                    aspect_ratio = w / h
                    target_width = int(target_height * aspect_ratio)
                    self.logo_img = cv2.resize(self.logo_img, (target_width, target_height), interpolation=cv2.INTER_AREA)
                    print(f"✓ Logo loaded: {target_width}x{target_height}")
                else:
                    print("⚠ Logo image could not be loaded")
            except Exception as e:
                print(f"⚠ Error loading logo: {e}")
                self.logo_img = None
        else:
            print(f"⚠ Logo not found at: {logo_path}")
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("Fullscreen mode enabled")
        else:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print("Fullscreen mode disabled")
    
    def create_grid_background(self):
        """Pre-render grid background once for better performance"""
        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        bg[:] = self.COLORS['bg_dark']
        
        grid_spacing = 30
        for x in range(0, self.width, grid_spacing):
            cv2.line(bg, (x, 0), (x, self.height), self.COLORS['medium_blue'], 1)
        for y in range(0, self.height, grid_spacing):
            cv2.line(bg, (0, y), (self.width, y), self.COLORS['medium_blue'], 1)
        
        # Darken the grid
        overlay = np.zeros_like(bg)
        overlay[:] = self.COLORS['bg_dark']
        cv2.addWeighted(bg, 0.1, overlay, 0.9, 0, bg)
        
        return bg
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check mute button (available on all screens)
            # Adjust position for tutorial state where it's moved left of skip all button
            mb = self.mute_button.copy()
            if self.state == 'tutorial':
                mb['x'] = self.width - 160 - 50 - 20  # Same adjustment as in draw_tutorial
            
            if mb['x'] <= x <= mb['x'] + mb['w'] and mb['y'] <= y <= mb['y'] + mb['h']:
                self.play_button_click_sound()
                self.toggle_mute()
                return
            
            if self.state == 'main_menu':
                # Check quit button only on main menu
                qb = self.quit_button
                if qb['x'] <= x <= qb['x'] + qb['w'] and qb['y'] <= y <= qb['y'] + qb['h']:
                    self.play_button_click_sound()
                    print("Quit button clicked - Exiting...")
                    self.running = False
                    return
                
                btn = self.start_button
                if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                    self.play_button_click_sound()
                    self.state = 'tutorial'
                    self.tutorial_player.reset()
                    
            elif self.state == 'tutorial':
                # Handle tutorial clicks (tutorial now manages back button internally)
                action = self.tutorial_player.handle_click(x, y)
                if action == 'back':
                    self.state = 'main_menu'
                elif action == 'continue':
                    self.state = 'mode_selection'
                elif action == 'replay':
                    self.tutorial_player.reset()
            
            elif self.state == 'wallet_tutorial':
                # Handle wallet tutorial clicks
                action = self.wallet_tutorial_player.handle_click(x, y)
                if action == 'back':
                    self.state = 'mode_selection'
                    # Release camera when going back
                    self.release_camera()
                elif action == 'continue':
                    self.state = 'mode_selection'
                    # Release camera when done
                    self.release_camera()
                elif action == 'replay':
                    self.wallet_tutorial_player.reset()
                    # Release camera when restarting
                    self.release_camera()
                elif action == 'enter_your_turn':
                    # Initialize camera in background thread for your turn mode
                    if self.camera is None and not self.camera_initializing:
                        threading.Thread(target=self.init_camera, daemon=True).start()
                elif action == 'your_turn_next':
                    # Keep camera open for next your turn, but check if it needs initialization
                    if self.camera is None and not self.camera_initializing:
                        threading.Thread(target=self.init_camera, daemon=True).start()
                    
            elif self.state == 'mode_selection':
                # Check quit button
                qb = self.quit_button
                if qb['x'] <= x <= qb['x'] + qb['w'] and qb['y'] <= y <= qb['y'] + qb['h']:
                    self.play_button_click_sound()
                    print("Quit button clicked - Exiting...")
                    self.running = False
                    return
                
                pb = self.pattern_button
                if pb['x'] <= x <= pb['x'] + pb['w'] and pb['y'] <= y <= pb['y'] + pb['h']:
                    self.play_button_click_sound()
                    # Show guide on first pattern mode entry
                    if not self.pattern_mode.guide_shown_this_session:
                        self.pattern_mode.show_guide = True
                        self.pattern_mode.guide_step = 1
                    self.state = 'level_selection'
                    # Start camera initialization in background
                    if self.camera is None and not self.camera_initializing:
                        threading.Thread(target=self.init_camera, daemon=True).start()
                
                wb = self.wallet_button
                if wb['x'] <= x <= wb['x'] + wb['w'] and wb['y'] <= y <= wb['y'] + wb['h']:
                    self.play_button_click_sound()
                    # Go to wallet tutorial
                    self.state = 'wallet_tutorial'
                    self.wallet_tutorial_player.reset()
                
                tb = self.tutorial_button
                if tb['x'] <= x <= tb['x'] + tb['w'] and tb['y'] <= y <= tb['y'] + tb['h']:
                    self.play_button_click_sound()
                    # Go back to tutorial state to replay it
                    self.state = 'tutorial'
                    self.tutorial_player.reset()
                
                bb = self.back_button
                if bb['x'] <= x <= bb['x'] + bb['w'] and bb['y'] <= y <= bb['y'] + bb['h']:
                    self.play_button_click_sound()
                    self.state = 'main_menu'
            
            elif self.state == 'level_selection':
                # Check if guide is showing first
                if self.pattern_mode.show_guide:
                    gb = self.pattern_mode.guide_button
                    if gb['x'] <= x <= gb['x'] + gb['w'] and gb['y'] <= y <= gb['y'] + gb['h']:
                        self.play_button_click_sound()
                        if self.pattern_mode.guide_step < 4:
                            # Advance to next step
                            self.pattern_mode.guide_step += 1
                        else:
                            # Last step - close guide
                            self.pattern_mode.show_guide = False
                            self.pattern_mode.guide_shown_this_session = True
                else:
                    # Handle level selection clicks
                    action, value = self.level_selection.handle_click(x, y)
                    if action == 'back':
                        self.state = 'mode_selection'
                        # Release camera when going back to mode selection
                        self.release_camera()
                    elif action == 'level_selected':  # Level number selected
                        self.pattern_mode.current_level = value
                        self.pattern_mode.reset_progress()  # Reset progress when entering level
                        self.state = 'pattern'
                        # Initialize camera if not already opened
                        if self.camera is None and not self.camera_initializing:
                            threading.Thread(target=self.init_camera, daemon=True).start()
                    
            elif self.state == 'pattern':
                # Handle pattern mode clicks
                result = self.pattern_mode.handle_click(x, y)
                if result == 'back':
                    # Don't release camera, just go back to level selection
                    # This allows quick switching between levels
                    # Note: reset_progress() already called in pattern_mode.handle_click
                    self.state = 'level_selection'
                elif result == 'next_level':
                    # Move to next level
                    next_level = self.pattern_mode.current_level + 1
                    if next_level <= 5:  # Max 5 levels
                        self.pattern_mode.current_level = next_level
                        self.pattern_mode.reset_progress()
                    else:
                        # All levels completed, go back to level selection
                        self.state = 'level_selection'
    
    def detect_camera_at_startup(self):
        """Detect camera availability at app startup"""
        print("\nDetecting camera...")
        self.camera_initializing = True
        
        # Try multiple backends for cross-platform compatibility
        import platform
        system = platform.system()
        
        if system == "Windows":
            backends = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_ANY, "Default")
            ]
        elif system == "Linux":
            # Raspberry Pi OS and other Linux systems
            backends = [
                (cv2.CAP_V4L2, "Video4Linux2"),
                (cv2.CAP_ANY, "Default")
            ]
        else:
            # macOS or other systems
            backends = [(cv2.CAP_ANY, "Default")]
        
        for backend, backend_name in backends:
            print(f"  Trying {backend_name} backend...")
            temp_camera = cv2.VideoCapture(0, backend)
            if temp_camera.isOpened():
                # Test if we can actually read a frame
                ret, _ = temp_camera.read()
                if ret:
                    print(f"  ✓ Camera detected with {backend_name}!")
                    self.camera_detected = True
                    self.camera_status_message = f"Camera ready ({backend_name})"
                    # Release the test camera - we'll reinitialize when needed
                    temp_camera.release()
                    self.camera_initializing = False
                    return
                else:
                    temp_camera.release()
        
        print("  ✗ No camera detected")
        self.camera_detected = False
        self.camera_status_message = "No camera found"
        self.camera_initializing = False
    
    def init_camera(self):
        """Initialize camera for actual use in pattern mode"""
        if self.camera is None and not self.camera_initializing:
            self.camera_initializing = True
            print("\nOpening camera for pattern mode...")
            
            # Try multiple backends for cross-platform compatibility
            import platform
            system = platform.system()
            
            if system == "Windows":
                backends = [
                    (cv2.CAP_DSHOW, "DirectShow"),
                    (cv2.CAP_MSMF, "Media Foundation"),
                    (cv2.CAP_ANY, "Default")
                ]
            elif system == "Linux":
                # Raspberry Pi OS and other Linux systems
                backends = [
                    (cv2.CAP_V4L2, "Video4Linux2"),
                    (cv2.CAP_ANY, "Default")
                ]
            else:
                # macOS or other systems
                backends = [(cv2.CAP_ANY, "Default")]
            
            for backend, backend_name in backends:
                print(f"  Trying {backend_name} backend...")
                self.camera = cv2.VideoCapture(0, backend)
                if self.camera.isOpened():
                    # Optimize camera settings for performance
                    # Set buffer size to 1 to reduce latency
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Test if we can actually read a frame
                    ret, frame = self.camera.read()
                    if ret:
                        actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"  Camera opened successfully with {backend_name}!")
                        print(f"  Resolution: {actual_w}x{actual_h}")
                        self.camera_initializing = False
                        return
                    else:
                        self.camera.release()
                        self.camera = None
            
            print("  Error: Could not open camera with any backend")
            self.camera = None
            self.camera_initializing = False
    
    def release_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            print("Camera released")
    
    def draw_glow_rect(self, img, x, y, w, h, color, glow_intensity):
        """Optimized glow - reduced iterations"""
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        # Reduced from 3 to 2 iterations for performance
        for i in range(2):
            offset = (i + 1) * 2
            alpha = glow_intensity * (1 - i * 0.4)
            glow_color = tuple(int(c * alpha) for c in color)
            cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), glow_color, 1)
    
    def draw_tech_lines(self, img):
        h, w = img.shape[:2]
        bracket_size, bracket_thickness = 40, 3
        
        # Top-left corner
        cv2.line(img, (20, 20), (20 + bracket_size, 20), self.COLORS['cyan'], bracket_thickness)
        cv2.line(img, (20, 20), (20, 20 + bracket_size), self.COLORS['cyan'], bracket_thickness)
        
        # Bottom-left corner
        cv2.line(img, (20, h - 20), (20 + bracket_size, h - 20), self.COLORS['cyan'], bracket_thickness)
        cv2.line(img, (20, h - 20), (20, h - 20 - bracket_size), self.COLORS['cyan'], bracket_thickness)
        
        # Bottom-right corner
        cv2.line(img, (w - 20, h - 20), (w - 20 - bracket_size, h - 20), self.COLORS['cyan'], bracket_thickness)
        cv2.line(img, (w - 20, h - 20), (w - 20, h - 20 - bracket_size), self.COLORS['cyan'], bracket_thickness)
    
    def draw_back_button(self, img):
        bb = self.back_button
        pulse = 0.4 + 0.3 * abs(math.sin(self.glow_phase))
        self.draw_glow_rect(img, bb['x'], bb['y'], bb['w'], bb['h'], self.COLORS['medium_blue'], pulse)
        
        overlay = img.copy()
        cv2.rectangle(overlay, (bb['x'] + 2, bb['y'] + 2), (bb['x'] + bb['w'] - 2, bb['y'] + bb['h'] - 2), self.COLORS['button_normal'], -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        text, font_scale, thickness = "< BACK", 0.6, 2
        (text_w, text_h), _ = cv2.getTextSize(text, _UI_FONT, font_scale, thickness)
        text_x = bb['x'] + (bb['w'] - text_w) // 2
        text_y = bb['y'] + (bb['h'] + text_h) // 2
        _put_text(img, text, text_x, text_y, font_scale, self.COLORS['text_primary'], thickness)
    
    def draw_quit_button(self, img):
        qb = self.quit_button
        pulse = 0.4 + 0.3 * abs(math.sin(self.glow_phase * 1.1))
        self.draw_glow_rect(img, qb['x'], qb['y'], qb['w'], qb['h'], self.COLORS['medium_blue'], pulse)
        
        overlay = img.copy()
        cv2.rectangle(overlay, (qb['x'] + 2, qb['y'] + 2), (qb['x'] + qb['w'] - 2, qb['y'] + qb['h'] - 2), (100, 50, 50), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        text, font_scale, thickness = "QUIT", 0.6, 2
        (text_w, text_h), _ = cv2.getTextSize(text, _UI_FONT, font_scale, thickness)
        text_x = qb['x'] + (qb['w'] - text_w) // 2
        text_y = qb['y'] + (qb['h'] + text_h) // 2
        _put_text(img, text, text_x, text_y, font_scale, self.COLORS['text_primary'], thickness)
    
    def draw_mute_button(self, img):
        """Draw mute/unmute button in upper right corner"""
        mb = self.mute_button
        pulse = 0.3 + 0.2 * abs(math.sin(self.glow_phase))
        
        # Button background
        overlay = img.copy()
        button_color = self.COLORS['button_normal'] if not self.is_muted else (100, 100, 100)
        cv2.rectangle(overlay, (mb['x'], mb['y']), (mb['x'] + mb['w'], mb['y'] + mb['h']), button_color, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Button border with glow
        border_color = self.COLORS['button_hover'] if not self.is_muted else (150, 150, 150)
        self.draw_glow_rect(img, mb['x'], mb['y'], mb['w'], mb['h'], border_color, pulse)
        
        # Draw speaker icon (properly centered)
        center_x = mb['x'] + mb['w'] // 2
        center_y = mb['y'] + mb['h'] // 2
        
        # Shift to properly center the entire icon
        icon_offset = -3
        
        if not self.is_muted:
            # Speaker base
            cv2.rectangle(img, (center_x - 11 + icon_offset, center_y - 6), (center_x - 5 + icon_offset, center_y + 6), self.COLORS['text_primary'], -1)
            # Speaker cone
            pts = np.array([[center_x - 5 + icon_offset, center_y - 6], [center_x + 1 + icon_offset, center_y - 12], [center_x + 1 + icon_offset, center_y + 12], [center_x - 5 + icon_offset, center_y + 6]], np.int32)
            cv2.fillPoly(img, [pts], self.COLORS['text_primary'])
            # Sound waves using ellipse
            cv2.ellipse(img, (center_x + 1 + icon_offset, center_y), (8, 8), 0, -60, 60, self.COLORS['text_primary'], 2)
            cv2.ellipse(img, (center_x + 1 + icon_offset, center_y), (14, 14), 0, -40, 40, self.COLORS['text_primary'], 2)
        else:
            # Speaker base (muted)
            cv2.rectangle(img, (center_x - 11 + icon_offset, center_y - 6), (center_x - 5 + icon_offset, center_y + 6), self.COLORS['text_secondary'], -1)
            # Speaker cone (muted)
            pts = np.array([[center_x - 5 + icon_offset, center_y - 6], [center_x + 1 + icon_offset, center_y - 12], [center_x + 1 + icon_offset, center_y + 12], [center_x - 5 + icon_offset, center_y + 6]], np.int32)
            cv2.fillPoly(img, [pts], self.COLORS['text_secondary'])
            # X mark
            cv2.line(img, (center_x + 3 + icon_offset, center_y - 10), (center_x + 13 + icon_offset, center_y + 10), (200, 200, 200), 3)
            cv2.line(img, (center_x + 3 + icon_offset, center_y + 10), (center_x + 13 + icon_offset, center_y - 10), (200, 200, 200), 3)
    
    def toggle_mute(self):
        """Toggle mute state"""
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.music_manager.set_volume(0.0)
            print("🔇 Music muted")
        else:
            self.music_manager.set_volume(0.5)
            print("🔊 Music unmuted")
    
    def play_button_click_sound(self):
        """Play button click sound effect"""
        self.music_manager.play_sound_effect('button_click.mp3')
    
    def draw_main_menu(self, frame):
        # Use pre-rendered grid background
        frame[:] = self.grid_background
        
        # Draw mute button
        self.draw_mute_button(frame)
        self.draw_tech_lines(frame)
        
        # Draw logo if available
        logo_y = 10  # Position from top (closer to accommodate larger logo)
        if self.logo_img is not None:
            h, w = self.logo_img.shape[:2]
            logo_x = (self.width - w) // 2  # Center horizontally
            
            # Handle logo with alpha channel (transparency)
            if self.logo_img.shape[2] == 4:
                # Extract alpha channel
                alpha = self.logo_img[:, :, 3] / 255.0
                # Get RGB channels
                logo_rgb = self.logo_img[:, :, :3]
                
                # Blend logo with background
                for c in range(3):
                    frame[logo_y:logo_y+h, logo_x:logo_x+w, c] = \
                        alpha * logo_rgb[:, :, c] + (1 - alpha) * frame[logo_y:logo_y+h, logo_x:logo_x+w, c]
            else:
                # No alpha channel, just overlay
                frame[logo_y:logo_y+h, logo_x:logo_x+w] = self.logo_img
            
            text_start_y = logo_y + h + 50  # Position text below logo
        else:
            text_start_y = self.height // 3  # Default position if no logo
        
        # Draw SEWBOT title
        text, font_scale, thickness = "SEWBOT", 2.5, 4
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, font_scale, thickness)
        text_x, text_y = (self.width - text_w) // 2, text_start_y
        
        # Reduced glow layers from 5 to 3 for performance
        glow_intensity = 0.5 + 0.5 * abs(math.sin(self.glow_phase))
        for offset in [6, 4, 2]:
            alpha = glow_intensity * (1 - offset / 8)
            glow_color = tuple(int(c * alpha) for c in self.COLORS['glow_cyan'])
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, glow_color, thickness + offset)
        
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, self.COLORS['neon_blue'], thickness)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, self.COLORS['text_primary'], 2)
        
        subtitle = "[SEWING GUIDANCE SYSTEM ]"
        sub_font_scale = 0.9
        sub_thickness = 2
        (sub_w, sub_h), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_TRIPLEX, sub_font_scale, sub_thickness)
        subtitle_y = text_y + 50
        cv2.putText(frame, subtitle, ((self.width - sub_w) // 2, subtitle_y), cv2.FONT_HERSHEY_TRIPLEX, sub_font_scale, self.COLORS['text_accent'], sub_thickness, cv2.LINE_AA)
        
        # Position START button below subtitle with spacing
        btn = self.start_button
        btn['y'] = subtitle_y + 30  # 60px below subtitle
        pulse = 0.5 + 0.5 * abs(math.sin(self.glow_phase * 1.5))
        self.draw_glow_rect(frame, btn['x'], btn['y'], btn['w'], btn['h'], self.COLORS['button_hover'], pulse)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (btn['x'] + 3, btn['y'] + 3), (btn['x'] + btn['w'] - 3, btn['y'] + btn['h'] - 3), self.COLORS['button_normal'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        text, font_scale, thickness = "START", 1.0, 2
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, font_scale, thickness)
        text_x = btn['x'] + (btn['w'] - text_w) // 2
        text_y = btn['y'] + (btn['h'] + text_h) // 2
        # Reduced glow thickness
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, self.COLORS['glow_cyan'], thickness + 2)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, self.COLORS['text_primary'], thickness)
        
        self.draw_quit_button(frame)
    
    def draw_tutorial(self, frame):
        # Use pre-rendered grid background
        frame[:] = self.grid_background
        
        # Draw tutorial player
        self.tutorial_player.draw(frame)
        
        # Draw mute button (positioned to left of skip all button)
        # Skip all button is at x=(width-160), so position mute button to its left
        original_mute_x = self.mute_button['x']
        self.mute_button['x'] = self.width - 160 - 50 - 20  # 20px spacing from skip all
        self.draw_mute_button(frame)
        self.mute_button['x'] = original_mute_x  # Restore original position
    
    def draw_wallet_tutorial(self, frame):
        # Use pre-rendered grid background
        frame[:] = self.grid_background

        # Draw mute button (top-right corner, same as other screens)
        self.draw_mute_button(frame)

        # Get camera frame if in your_turn mode
        camera_frame = None
        if self.wallet_tutorial_player.your_turn_mode:
            if self.camera is not None and self.camera.isOpened():
                ret, camera_frame = self.camera.read()
                if not ret:
                    camera_frame = None
        
        # Draw wallet tutorial player (pass camera frame)
        self.wallet_tutorial_player.draw(frame, camera_frame)
    
    def draw_level_selection(self, frame):
        """Draw level selection screen"""
        self.level_selection.draw(frame, self.grid_background)
        
        # Draw guide overlay if showing
        if self.pattern_mode.show_guide:
            self.pattern_mode.draw_guide_overlay(frame)
        
        # Draw mute button
        self.draw_mute_button(frame)
    
    def draw_mode_selection(self, frame):
        # Use pre-rendered grid background
        frame[:] = self.grid_background
        
        # Draw mute button first
        self.draw_mute_button(frame)
        
        # Top-left corner bracket only
        bracket_size, thickness, offset = 50, 2, 15
        cv2.line(frame, (offset, offset + bracket_size), (offset, offset), self.COLORS['neon_blue'], thickness)
        cv2.line(frame, (offset, offset), (offset + bracket_size, offset), self.COLORS['neon_blue'], thickness)
        cv2.circle(frame, (offset, offset), 3, self.COLORS['cyan'], -1)
        
        text, font_scale, thickness = "SELECT MODE", 1.5, 3
        (text_w, text_h), _ = cv2.getTextSize(text, _UI_FONT, font_scale, thickness)
        text_x, text_y = (self.width - text_w) // 2, 80
        
        _put_text(frame, text, text_x, text_y, font_scale, self.COLORS['bright_blue'], thickness)
        
        # Draw all three mode buttons
        self.draw_mode_button(frame, self.pattern_button)
        self.draw_mode_button(frame, self.wallet_button)
        self.draw_mode_button(frame, self.tutorial_button)
        self.draw_back_button(frame)
        self.draw_quit_button(frame)
    
    def draw_mode_button(self, img, button_data):
        x, y, w, h = button_data['x'], button_data['y'], button_data['w'], button_data['h']
        pulse = 0.4 + 0.6 * abs(math.sin(self.glow_phase * 1.2))
        
        self.draw_glow_rect(img, x, y, w, h, self.COLORS['button_hover'], pulse)
        overlay = img.copy()
        cv2.rectangle(overlay, (x + 3, y + 3), (x + w - 3, y + h - 3), self.COLORS['button_normal'], -1)
        cv2.addWeighted(overlay, 0.7, img, 1 - 0.7, 0, img)
        
        # Draw title (centered)
        title_font_scale, title_thickness = 1.0, 2
        (title_w, title_h), _ = cv2.getTextSize(button_data['title'], _UI_FONT, title_font_scale, title_thickness)
        title_x = x + (w - title_w) // 2
        title_y = y + 40  # Better vertical positioning
        
        _put_text(img, button_data['title'], title_x, title_y, title_font_scale, self.COLORS['text_primary'], title_thickness)
        
        # Draw description (centered, wrapped if needed)
        desc_font_scale = 0.45
        desc_thickness = 1
        description = button_data['description']
        
        # Simple text wrapping
        words = description.split()
        lines = []
        current_line = []
        max_width = w - 40  # padding
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            (test_w, test_h), _ = cv2.getTextSize(test_line, _UI_FONT, desc_font_scale, desc_thickness)
            if test_w <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw description lines
        line_height = 18
        start_y = title_y + 25  # More space below title
        for i, line in enumerate(lines):
            (line_w, line_h), _ = cv2.getTextSize(line, _UI_FONT, desc_font_scale, desc_thickness)
            line_x = x + (w - line_w) // 2
            line_y = start_y + i * line_height
            _put_text(img, line, line_x, line_y, desc_font_scale, self.COLORS['text_secondary'], desc_thickness)
    
    def run(self):
        
        while self.running:
            try:
                # Handle music transitions based on state changes
                if self.state != self.previous_state:
                    # Stop current music
                    if self.previous_state == 'pattern':
                        self.pattern_mode.stop_music()
                    
                    # Start new music based on state
                    if self.state == 'main_menu':
                        self.music_manager.play('main_menu.mp3', loops=-1, fade_ms=1000)
                    elif self.state == 'tutorial':
                        self.music_manager.play('tutorial.mp3', loops=-1, fade_ms=1000)
                    elif self.state == 'mode_selection':
                        self.music_manager.play('mode_selection.mp3', loops=-1, fade_ms=1000)
                    elif self.state == 'wallet_tutorial':
                        self.music_manager.play('wallet.mp3', loops=-1, fade_ms=1000)
                    elif self.state == 'pattern':
                        self.pattern_mode.start_music()
                    
                    # Update previous state
                    self.previous_state = self.state
                
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
                if self.state == 'main_menu':
                    self.draw_main_menu(frame)
                elif self.state == 'tutorial':
                    self.draw_tutorial(frame)
                elif self.state == 'wallet_tutorial':
                    self.draw_wallet_tutorial(frame)
                elif self.state == 'mode_selection':
                    self.draw_mode_selection(frame)
                elif self.state == 'level_selection':
                    self.draw_level_selection(frame)
                elif self.state == 'pattern':
                    # Get camera frame
                    camera_frame = None
                    if self.camera is not None and self.camera.isOpened():
                        ret, camera_frame = self.camera.read()
                        if not ret:
                            camera_frame = None
                    
                    # Draw pattern mode
                    self.pattern_mode.draw(frame, camera_frame, self.grid_background)
                    
                    # Draw mute button on top of pattern mode
                    self.draw_mute_button(frame)
                
                self.glow_phase += 0.05
                cv2.imshow(self.window_name, frame)
                
                # Check if window was closed (X button clicked)
                # This needs to be checked after imshow
                key = cv2.waitKey(30) & 0xFF
                
                # Handle keyboard shortcuts
                if key == ord('f') or key == ord('F'):  # F key to toggle fullscreen
                    self.toggle_fullscreen()
                elif key == 27:  # ESC key to exit fullscreen or quit
                    if self.fullscreen:
                        self.toggle_fullscreen()
                    else:
                        print("ESC pressed - Exiting...")
                        self.running = False
                        break
                
                # Pattern mode specific controls
                elif self.state == 'pattern':
                    if key == ord('+') or key == ord('='):  # Increase confidence threshold
                        self.pattern_mode.confidence_threshold = min(0.9, self.pattern_mode.confidence_threshold + 0.05)
                        print(f"Confidence threshold: {self.pattern_mode.confidence_threshold:.2f}")
                    elif key == ord('-') or key == ord('_'):  # Decrease confidence threshold
                        self.pattern_mode.confidence_threshold = max(0.1, self.pattern_mode.confidence_threshold - 0.05)
                        print(f"Confidence threshold: {self.pattern_mode.confidence_threshold:.2f}")
                
                # Check window property to detect X button click
                try:
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("Window closed - Exiting...")
                        self.running = False
                        break
                except:
                    print("Window closed - Exiting...")
                    self.running = False
                    break
                    
            except (cv2.error, Exception) as e:
                print(f"Window closed: {e}")
                self.running = False
                break
        
        self.release_camera()
        self.tutorial_player.cleanup()
        self.wallet_tutorial_player.cleanup()
        
        # Cleanup music system
        music_manager = get_music_manager()
        music_manager.cleanup()
        
        cv2.destroyAllWindows()
        print("Program closed.")


if __name__ == '__main__':
    app = SewBotApp()
    app.run()
