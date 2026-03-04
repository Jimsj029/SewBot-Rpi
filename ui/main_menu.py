import cv2
import numpy as np
import math
import sys
import os
from .theme import COLORS, FONTS, ANIMATION

# Add parent directory to path for music_manager import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from music_manager import get_music_manager


class MainMenu:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.window_name = 'SewBot - Main Menu'
        self.selected = False
        self.glow_phase = 0
        
        # Button properties
        self.button_width = 300
        self.button_height = 80
        self.button_x = (width - self.button_width) // 2
        self.button_y = height // 2 + 50
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicked on Start button
            if (self.button_x <= x <= self.button_x + self.button_width and
                self.button_y <= y <= self.button_y + self.button_height):
                self.selected = True
    
    def draw_glow_rect(self, img, x, y, w, h, color, glow_intensity):
        """Draw a rectangle with a glowing effect"""
        # Main rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        
        # Glow layers
        for i in range(3):
            offset = (i + 1) * 2
            alpha = glow_intensity * (1 - i * 0.3)
            glow_color = tuple(int(c * alpha) for c in color)
            cv2.rectangle(img, 
                         (x - offset, y - offset), 
                         (x + w + offset, y + h + offset), 
                         glow_color, 1)
    
    def draw_tech_lines(self, img):
        """Draw decorative tech lines for sci-fi effect"""
        h, w = img.shape[:2]
        
        # Corner brackets
        bracket_size = 40
        bracket_thickness = 3
        
        # Top-left
        cv2.line(img, (20, 20), (20 + bracket_size, 20), COLORS['cyan'], bracket_thickness)
        cv2.line(img, (20, 20), (20, 20 + bracket_size), COLORS['cyan'], bracket_thickness)
        
        # Top-right
        cv2.line(img, (w - 20, 20), (w - 20 - bracket_size, 20), COLORS['cyan'], bracket_thickness)
        cv2.line(img, (w - 20, 20), (w - 20, 20 + bracket_size), COLORS['cyan'], bracket_thickness)
        
        # Bottom-left
        cv2.line(img, (20, h - 20), (20 + bracket_size, h - 20), COLORS['cyan'], bracket_thickness)
        cv2.line(img, (20, h - 20), (20, h - 20 - bracket_size), COLORS['cyan'], bracket_thickness)
        
        # Bottom-right
        cv2.line(img, (w - 20, h - 20), (w - 20 - bracket_size, h - 20), COLORS['cyan'], bracket_thickness)
        cv2.line(img, (w - 20, h - 20), (w - 20, h - 20 - bracket_size), COLORS['cyan'], bracket_thickness)
        
        # Horizontal scan lines (subtle)
        for y in range(0, h, 4):
            alpha = 0.02
            overlay = img.copy()
            cv2.line(overlay, (0, y), (w, y), COLORS['bright_blue'], 1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    def draw_title(self, img):
        """Draw the SewBot title with futuristic styling"""
        text = "SEWBOT"
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = FONTS['title_size']
        thickness = 4
        
        # Get text size for centering
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (self.width - text_w) // 2
        text_y = self.height // 3
        
        # Draw glow effect
        glow_intensity = 0.5 + 0.5 * abs(math.sin(self.glow_phase))
        
        # Outer glow
        for offset in range(10, 0, -2):
            alpha = glow_intensity * (1 - offset / 10)
            glow_color = tuple(int(c * alpha) for c in COLORS['glow_cyan'])
            cv2.putText(img, text, (text_x, text_y), font, font_scale, 
                       glow_color, thickness + offset)
        
        # Main text
        cv2.putText(img, text, (text_x, text_y), font, font_scale, 
                   COLORS['neon_blue'], thickness)
        
        # Highlight text
        cv2.putText(img, text, (text_x, text_y), font, font_scale, 
                   COLORS['text_primary'], 2)
        
        # Subtitle
        subtitle = "[ PATTERN RECOGNITION SYSTEM ]"
        sub_font_scale = FONTS['small_size']
        (sub_w, sub_h), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_TRIPLEX, 
                                            sub_font_scale, 1)
        sub_x = (self.width - sub_w) // 2
        sub_y = text_y + 40
        
        cv2.putText(img, subtitle, (sub_x, sub_y), cv2.FONT_HERSHEY_TRIPLEX, 
                   sub_font_scale, COLORS['text_accent'], 1)
    
    def draw_button(self, img):
        """Draw the Start button with hover effect"""
        # Pulsing glow effect
        pulse = 0.5 + 0.5 * abs(math.sin(self.glow_phase * 1.5))
        
        # Draw button background with glow
        self.draw_glow_rect(img, self.button_x, self.button_y, 
                          self.button_width, self.button_height,
                          COLORS['button_hover'], pulse)
        
        # Fill button
        overlay = img.copy()
        cv2.rectangle(overlay, 
                     (self.button_x + 3, self.button_y + 3),
                     (self.button_x + self.button_width - 3, 
                      self.button_y + self.button_height - 3),
                     COLORS['button_normal'], -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Button text
        text = "START"
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = FONTS['button_size']
        thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = self.button_x + (self.button_width - text_w) // 2
        text_y = self.button_y + (self.button_height + text_h) // 2
        
        # Text glow
        cv2.putText(img, text, (text_x, text_y), font, font_scale,
                   COLORS['glow_cyan'], thickness + 4)
        
        # Main text
        cv2.putText(img, text, (text_x, text_y), font, font_scale,
                   COLORS['text_primary'], thickness)
    
    def run(self):
        """Main loop for the main menu"""
        print("SewBot - Main Menu")
        print("Click START to begin")
        
        # Start main menu music
        music_manager = get_music_manager()
        music_manager.play('main_menu.mp3', loops=-1, fade_ms=1000)
        
        while True:
            # Create frame with dark background
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:] = COLORS['bg_dark']
            
            # Draw all elements
            self.draw_tech_lines(frame)
            self.draw_title(frame)
            self.draw_button(frame)
            
            # Update glow animation
            self.glow_phase += ANIMATION['glow_speed']
            
            # Display
            cv2.imshow(self.window_name, frame)
            
            # Handle input
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or Q to quit
                music_manager.stop(fade_ms=1000)
                cv2.destroyAllWindows()
                return None
            
            if self.selected:
                music_manager.stop(fade_ms=1000)
                cv2.destroyAllWindows()
                return 'mode_selection'
        
        cv2.destroyAllWindows()
        return None
