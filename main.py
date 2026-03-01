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
from pattern_mode import PatternMode


class SewBotApp:
    def __init__(self):
        self.width = 1024
        self.height = 600
        self.window_name = 'SewBot - Pattern Recognition System'
        self.state = 'main_menu'  # main_menu, tutorial, wallet_tutorial, mode_selection, pattern
        self.glow_phase = 0
        self.running = True
        
        # Tutorial player - initialize with tutorial videos from videos/sewing-set-up
        self.tutorial_player = TutorialPlayer(self.width, self.height)
        
        # Wallet tutorial player - initialize with wallet videos
        self.wallet_tutorial_player = WalletTutorialPlayer(self.width, self.height)
        
        # Camera variables
        self.camera = None
        self.camera_initializing = False
        self.camera_detected = False
        self.camera_status_message = "Checking camera..."
        
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
        
        # Pattern mode - separate module
        self.pattern_mode = PatternMode(self.width, self.height, self.COLORS, 'blueprint')
        
        # Pre-render grid background (performance optimization)
        self.grid_background = self.create_grid_background()
        
        try:
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
        except:
            print("Failed to create window")
            self.running = False
        
        # Detect camera at startup
        self.detect_camera_at_startup()
    
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
            if self.state == 'main_menu':
                # Check quit button only on main menu
                qb = self.quit_button
                if qb['x'] <= x <= qb['x'] + qb['w'] and qb['y'] <= y <= qb['y'] + qb['h']:
                    print("Quit button clicked - Exiting...")
                    self.running = False
                    return
                
                btn = self.start_button
                if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                    self.state = 'tutorial'
                    self.tutorial_player.reset()
                    
            elif self.state == 'tutorial':
                # Check back button first
                bb = self.back_button
                if bb['x'] <= x <= bb['x'] + bb['w'] and bb['y'] <= y <= bb['y'] + bb['h']:
                    self.state = 'main_menu'
                    return
                    
                # Handle tutorial clicks
                action = self.tutorial_player.handle_click(x, y)
                if action == 'continue':
                    self.state = 'mode_selection'
                elif action == 'replay':
                    self.tutorial_player.reset()
            
            elif self.state == 'wallet_tutorial':
                # Check back button first
                bb = self.back_button
                if bb['x'] <= x <= bb['x'] + bb['w'] and bb['y'] <= y <= bb['y'] + bb['h']:
                    self.state = 'mode_selection'
                    return
                    
                # Handle wallet tutorial clicks
                action = self.wallet_tutorial_player.handle_click(x, y)
                if action == 'continue':
                    self.state = 'mode_selection'
                elif action == 'replay':
                    self.wallet_tutorial_player.reset()
                    
            elif self.state == 'mode_selection':
                # Check quit button
                qb = self.quit_button
                if qb['x'] <= x <= qb['x'] + qb['w'] and qb['y'] <= y <= qb['y'] + qb['h']:
                    print("Quit button clicked - Exiting...")
                    self.running = False
                    return
                
                pb = self.pattern_button
                if pb['x'] <= x <= pb['x'] + pb['w'] and pb['y'] <= y <= pb['y'] + pb['h']:
                    self.state = 'pattern'
                    # Start camera initialization in background
                    if self.camera is None and not self.camera_initializing:
                        threading.Thread(target=self.init_camera, daemon=True).start()
                
                wb = self.wallet_button
                if wb['x'] <= x <= wb['x'] + wb['w'] and wb['y'] <= y <= wb['y'] + wb['h']:
                    # Go to wallet tutorial
                    self.state = 'wallet_tutorial'
                    self.wallet_tutorial_player.reset()
                
                tb = self.tutorial_button
                if tb['x'] <= x <= tb['x'] + tb['w'] and tb['y'] <= y <= tb['y'] + tb['h']:
                    # Go back to tutorial state to replay it
                    self.state = 'tutorial'
                    self.tutorial_player.reset()
                
                bb = self.back_button
                if bb['x'] <= x <= bb['x'] + bb['w'] and bb['y'] <= y <= bb['y'] + bb['h']:
                    self.state = 'main_menu'
                    
            elif self.state == 'pattern':
                # Handle pattern mode clicks
                result = self.pattern_mode.handle_click(x, y)
                if result == 'back':
                    self.release_camera()
                    self.state = 'mode_selection'
    
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
                    # Test if we can actually read a frame
                    ret, _ = self.camera.read()
                    if ret:
                        print(f"  Camera opened successfully with {backend_name}!")
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
        
        cv2.line(img, (20, 20), (20 + bracket_size, 20), self.COLORS['cyan'], bracket_thickness)
        cv2.line(img, (20, 20), (20, 20 + bracket_size), self.COLORS['cyan'], bracket_thickness)
        cv2.line(img, (w - 20, 20), (w - 20 - bracket_size, 20), self.COLORS['cyan'], bracket_thickness)
        cv2.line(img, (w - 20, 20), (w - 20, 20 + bracket_size), self.COLORS['cyan'], bracket_thickness)
        cv2.line(img, (20, h - 20), (20 + bracket_size, h - 20), self.COLORS['cyan'], bracket_thickness)
        cv2.line(img, (20, h - 20), (20, h - 20 - bracket_size), self.COLORS['cyan'], bracket_thickness)
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
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = bb['x'] + (bb['w'] - text_w) // 2
        text_y = bb['y'] + (bb['h'] + text_h) // 2
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.COLORS['text_primary'], thickness)
    
    def draw_quit_button(self, img):
        qb = self.quit_button
        pulse = 0.4 + 0.3 * abs(math.sin(self.glow_phase * 1.1))
        self.draw_glow_rect(img, qb['x'], qb['y'], qb['w'], qb['h'], self.COLORS['medium_blue'], pulse)
        
        overlay = img.copy()
        cv2.rectangle(overlay, (qb['x'] + 2, qb['y'] + 2), (qb['x'] + qb['w'] - 2, qb['y'] + qb['h'] - 2), (100, 50, 50), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        text, font_scale, thickness = "QUIT", 0.6, 2
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = qb['x'] + (qb['w'] - text_w) // 2
        text_y = qb['y'] + (qb['h'] + text_h) // 2
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.COLORS['text_primary'], thickness)
    
    def draw_main_menu(self, frame):
        # Use pre-rendered grid background
        frame[:] = self.grid_background
        self.draw_tech_lines(frame)
        
        text, font_scale, thickness = "SEWBOT", 2.5, 4
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        text_x, text_y = (self.width - text_w) // 2, self.height // 3
        
        # Reduced glow layers from 5 to 3 for performance
        glow_intensity = 0.5 + 0.5 * abs(math.sin(self.glow_phase))
        for offset in [6, 4, 2]:
            alpha = glow_intensity * (1 - offset / 8)
            glow_color = tuple(int(c * alpha) for c in self.COLORS['glow_cyan'])
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, glow_color, thickness + offset)
        
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, self.COLORS['neon_blue'], thickness)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, self.COLORS['text_primary'], 2)
        
        subtitle = "[ PATTERN RECOGNITION SYSTEM ]"
        sub_font_scale = 0.6
        (sub_w, sub_h), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, 1)
        cv2.putText(frame, subtitle, ((self.width - sub_w) // 2, text_y + 40), cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, self.COLORS['text_accent'], 1)
        
        btn = self.start_button
        pulse = 0.5 + 0.5 * abs(math.sin(self.glow_phase * 1.5))
        self.draw_glow_rect(frame, btn['x'], btn['y'], btn['w'], btn['h'], self.COLORS['button_hover'], pulse)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (btn['x'] + 3, btn['y'] + 3), (btn['x'] + btn['w'] - 3, btn['y'] + btn['h'] - 3), self.COLORS['button_normal'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        text, font_scale, thickness = "START", 1.0, 2
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        text_x = btn['x'] + (btn['w'] - text_w) // 2
        text_y = btn['y'] + (btn['h'] + text_h) // 2
        # Reduced glow thickness
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, self.COLORS['glow_cyan'], thickness + 2)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, self.COLORS['text_primary'], thickness)
        
        self.draw_quit_button(frame)
    
    def draw_tutorial(self, frame):
        # Use pre-rendered grid background
        frame[:] = self.grid_background
        
        # Draw tutorial player
        self.tutorial_player.draw(frame)
        
        # Draw back button
        self.draw_back_button(frame)
    
    def draw_wallet_tutorial(self, frame):
        # Use pre-rendered grid background
        frame[:] = self.grid_background
        
        # Draw wallet tutorial player
        self.wallet_tutorial_player.draw(frame)
        
        # Draw back button
        self.draw_back_button(frame)
    
    def draw_mode_selection(self, frame):
        # Use pre-rendered grid background
        frame[:] = self.grid_background
        
        bracket_size, thickness, offset = 50, 2, 15
        cv2.line(frame, (offset, offset + bracket_size), (offset, offset), self.COLORS['neon_blue'], thickness)
        cv2.line(frame, (offset, offset), (offset + bracket_size, offset), self.COLORS['neon_blue'], thickness)
        cv2.circle(frame, (offset, offset), 3, self.COLORS['cyan'], -1)
        cv2.line(frame, (self.width - offset, offset + bracket_size), (self.width - offset, offset), self.COLORS['neon_blue'], thickness)
        cv2.line(frame, (self.width - offset, offset), (self.width - offset - bracket_size, offset), self.COLORS['neon_blue'], thickness)
        cv2.circle(frame, (self.width - offset, offset), 3, self.COLORS['cyan'], -1)
        
        text, font_scale, thickness = "SELECT MODE", 1.5, 3
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        text_x, text_y = (self.width - text_w) // 2, 80
        
        # Reduced glow layers from 6 to 3
        glow_intensity = 0.6 + 0.4 * abs(math.sin(self.glow_phase * 0.8))
        for offset_val in [4, 2, 1]:
            alpha = glow_intensity * (1 - offset_val / 5)
            glow_color = tuple(int(c * alpha) for c in self.COLORS['glow_blue'])
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, glow_color, thickness + offset_val)
        
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, self.COLORS['bright_blue'], thickness)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, self.COLORS['text_primary'], 1)
        
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
        (title_w, title_h), _ = cv2.getTextSize(button_data['title'], cv2.FONT_HERSHEY_DUPLEX, title_font_scale, title_thickness)
        title_x = x + (w - title_w) // 2
        title_y = y + 35
        
        # Title with glow
        cv2.putText(img, button_data['title'], (title_x, title_y), cv2.FONT_HERSHEY_DUPLEX, title_font_scale, self.COLORS['glow_cyan'], title_thickness + 2)
        cv2.putText(img, button_data['title'], (title_x, title_y), cv2.FONT_HERSHEY_DUPLEX, title_font_scale, self.COLORS['text_primary'], title_thickness)
        
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
            (test_w, test_h), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, desc_font_scale, desc_thickness)
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
        start_y = title_y + 20
        for i, line in enumerate(lines):
            (line_w, line_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, desc_font_scale, desc_thickness)
            line_x = x + (w - line_w) // 2
            line_y = start_y + i * line_height
            cv2.putText(img, line, (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, desc_font_scale, self.COLORS['text_secondary'], desc_thickness)
    
    def run(self):
        print("=" * 60)
        print("SEWBOT - PATTERN RECOGNITION SYSTEM")
        print("=" * 60)
        print("Optimized for Raspberry Pi")
        print(f"Camera Status: {self.camera_status_message}")
        print("Click the X button to quit")
        print()
        
        while self.running:
            try:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
                if self.state == 'main_menu':
                    self.draw_main_menu(frame)
                elif self.state == 'tutorial':
                    self.draw_tutorial(frame)
                elif self.state == 'wallet_tutorial':
                    self.draw_wallet_tutorial(frame)
                elif self.state == 'mode_selection':
                    self.draw_mode_selection(frame)
                elif self.state == 'pattern':
                    # Get camera frame
                    camera_frame = None
                    if self.camera is not None and self.camera.isOpened():
                        ret, camera_frame = self.camera.read()
                        if not ret:
                            camera_frame = None
                    
                    # Draw pattern mode
                    self.pattern_mode.draw(frame, camera_frame, self.grid_background)
                
                self.glow_phase += 0.05
                cv2.imshow(self.window_name, frame)
                
                # Check if window was closed (X button clicked)
                # This needs to be checked after imshow
                key = cv2.waitKey(30)
                
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
        cv2.destroyAllWindows()
        print("Program closed.")


if __name__ == '__main__':
    app = SewBotApp()
    app.run()
