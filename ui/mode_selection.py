import cv2
import numpy as np
import math
from .theme import COLORS, FONTS, ANIMATION


class ModeSelection:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.window_name = 'SewBot - Mode Selection'
        self.selected_mode = None
        self.glow_phase = 0
        
        # Button properties
        self.button_width = 280
        self.button_height = 200
        self.button_spacing = 80
        
        # Calculate button positions (side by side)
        total_width = self.button_width * 2 + self.button_spacing
        start_x = (width - total_width) // 2
        button_y = height // 2 + 20
        
        self.pattern_button = {
            'x': start_x,
            'y': button_y,
            'w': self.button_width,
            'h': self.button_height,
            'text': 'PATTERN',
            'subtitle': '[ LEVELS 1-5 ]'
        }
        
        self.wallet_button = {
            'x': start_x + self.button_width + self.button_spacing,
            'y': button_y,
            'w': self.button_width,
            'h': self.button_height,
            'text': 'WALLET',
            'subtitle': '[ COMING SOON ]'
        }
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check Pattern button
            pb = self.pattern_button
            if (pb['x'] <= x <= pb['x'] + pb['w'] and
                pb['y'] <= y <= pb['y'] + pb['h']):
                self.selected_mode = 'pattern'
            
            # Check Wallet button (not functional yet)
            wb = self.wallet_button
            if (wb['x'] <= x <= wb['x'] + wb['w'] and
                wb['y'] <= y <= wb['y'] + wb['h']):
                # Wallet button - no function yet
                pass
    
    def draw_glow_rect(self, img, x, y, w, h, color, glow_intensity):
        """Draw a rectangle with a glowing effect"""
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        
        for i in range(3):
            offset = (i + 1) * 2
            alpha = glow_intensity * (1 - i * 0.3)
            glow_color = tuple(int(c * alpha) for c in color)
            cv2.rectangle(img, 
                         (x - offset, y - offset), 
                         (x + w + offset, y + h + offset), 
                         glow_color, 1)
    
    def draw_grid_pattern(self, img):
        """Draw a subtle grid pattern for depth"""
        h, w = img.shape[:2]
        grid_spacing = 30
        
        for x in range(0, w, grid_spacing):
            cv2.line(img, (x, 0), (x, h), COLORS['medium_blue'], 1)
        
        for y in range(0, h, grid_spacing):
            cv2.line(img, (0, y), (w, y), COLORS['medium_blue'], 1)
        
        # Fade the grid
        overlay = np.zeros_like(img)
        overlay[:] = COLORS['bg_dark']
        cv2.addWeighted(img, 0.1, overlay, 0.9, 0, img)
    
    def draw_corner_accents(self, img):
        """Draw corner accents with a different style than main menu"""
        h, w = img.shape[:2]
        bracket_size = 50
        thickness = 2
        
        # Diagonal corner brackets
        offset = 15
        
        # Top-left
        cv2.line(img, (offset, offset + bracket_size), (offset, offset), COLORS['neon_blue'], thickness)
        cv2.line(img, (offset, offset), (offset + bracket_size, offset), COLORS['neon_blue'], thickness)
        cv2.circle(img, (offset, offset), 3, COLORS['cyan'], -1)
        
        # Top-right
        cv2.line(img, (w - offset, offset + bracket_size), (w - offset, offset), COLORS['neon_blue'], thickness)
        cv2.line(img, (w - offset, offset), (w - offset - bracket_size, offset), COLORS['neon_blue'], thickness)
        cv2.circle(img, (w - offset, offset), 3, COLORS['cyan'], -1)
        
        # Bottom-left
        cv2.line(img, (offset, h - offset - bracket_size), (offset, h - offset), COLORS['neon_blue'], thickness)
        cv2.line(img, (offset, h - offset), (offset + bracket_size, h - offset), COLORS['neon_blue'], thickness)
        cv2.circle(img, (offset, h - offset), 3, COLORS['cyan'], -1)
        
        # Bottom-right
        cv2.line(img, (w - offset, h - offset - bracket_size), (w - offset, h - offset), COLORS['neon_blue'], thickness)
        cv2.line(img, (w - offset, h - offset), (w - offset - bracket_size, h - offset), COLORS['neon_blue'], thickness)
        cv2.circle(img, (w - offset, h - offset), 3, COLORS['cyan'], -1)
    
    def draw_title(self, img):
        """Draw the mode selection title"""
        text = "SELECT MODE"
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1.5
        thickness = 3
        
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (self.width - text_w) // 2
        text_y = 100
        
        # Glow
        glow_intensity = 0.6 + 0.4 * abs(math.sin(self.glow_phase * 0.8))
        for offset in range(6, 0, -1):
            alpha = glow_intensity * (1 - offset / 6)
            glow_color = tuple(int(c * alpha) for c in COLORS['glow_blue'])
            cv2.putText(img, text, (text_x, text_y), font, font_scale, 
                       glow_color, thickness + offset)
        
        cv2.putText(img, text, (text_x, text_y), font, font_scale, 
                   COLORS['bright_blue'], thickness)
        cv2.putText(img, text, (text_x, text_y), font, font_scale, 
                   COLORS['text_primary'], 1)
    
    def draw_button(self, img, button_data, is_active=True):
        """Draw a mode selection button"""
        x, y = button_data['x'], button_data['y']
        w, h = button_data['w'], button_data['h']
        
        # Pulsing effect
        pulse = 0.4 + 0.6 * abs(math.sin(self.glow_phase * 1.2))
        
        if is_active:
            color = COLORS['button_hover']
            glow = pulse
        else:
            color = COLORS['medium_blue']
            glow = pulse * 0.5
        
        # Draw button with glow
        self.draw_glow_rect(img, x, y, w, h, color, glow)
        
        # Fill button
        overlay = img.copy()
        fill_color = COLORS['button_normal'] if is_active else COLORS['dark_blue']
        cv2.rectangle(overlay, (x + 3, y + 3), (x + w - 3, y + h - 3), fill_color, -1)
        alpha = 0.7 if is_active else 0.5
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Main text
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1.1
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(button_data['text'], font, font_scale, thickness)
        text_x = x + (w - text_w) // 2
        text_y = y + h // 2 - 5
        
        if is_active:
            # Text glow
            cv2.putText(img, button_data['text'], (text_x, text_y), font, font_scale,
                       COLORS['glow_cyan'], thickness + 3)
        
        cv2.putText(img, button_data['text'], (text_x, text_y), font, font_scale,
                   COLORS['text_primary'], thickness)
        
        # Subtitle
        sub_font_scale = 0.5
        (sub_w, sub_h), _ = cv2.getTextSize(button_data['subtitle'], 
                                            cv2.FONT_HERSHEY_TRIPLEX, sub_font_scale, 1)
        sub_x = x + (w - sub_w) // 2
        sub_y = text_y + 30
        
        sub_color = COLORS['text_accent'] if is_active else COLORS['text_secondary']
        cv2.putText(img, button_data['subtitle'], (sub_x, sub_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, sub_font_scale, sub_color, 1)
    
    def run(self):
        """Main loop for mode selection"""
        print("SewBot - Mode Selection")
        print("Choose your mode: Pattern or Wallet")
        
        while True:
            # Create frame
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:] = COLORS['bg_dark']
            
            # Draw elements
            self.draw_grid_pattern(frame)
            self.draw_corner_accents(frame)
            self.draw_title(frame)
            self.draw_button(frame, self.pattern_button, is_active=True)
            self.draw_button(frame, self.wallet_button, is_active=False)
            
            # Update animation
            self.glow_phase += ANIMATION['glow_speed']
            
            # Display
            cv2.imshow(self.window_name, frame)
            
            # Handle input
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or Q to quit
                cv2.destroyAllWindows()
                return None
            
            if self.selected_mode:
                cv2.destroyAllWindows()
                return self.selected_mode
        
        cv2.destroyAllWindows()
        return None
