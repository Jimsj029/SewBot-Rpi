"""
Level Selection Screen - Choose a level before starting pattern mode
"""

import cv2
import numpy as np
import math


class LevelSelection:
    def __init__(self, width, height, colors):
        self.width = width
        self.height = height
        self.COLORS = colors
        self.glow_phase = 0
        self.selected_level = None
        
        # Title
        self.title_text = "SELECT LEVEL"
        self.subtitle_text = "Choose your sewing challenge"
        
        # Level button configuration
        self.button_width = 160
        self.button_height = 180
        self.button_spacing = 30
        
        # Calculate positions to center all 5 buttons
        total_width = (self.button_width * 5) + (self.button_spacing * 4)
        start_x = (width - total_width) // 2
        button_y = height // 2 - 50
        
        # Create level buttons
        self.level_buttons = []
        for i in range(1, 6):
            self.level_buttons.append({
                'level': i,
                'x': start_x + (i - 1) * (self.button_width + self.button_spacing),
                'y': button_y,
                'w': self.button_width,
                'h': self.button_height,
                'locked': False  # You can set these to True if you want progression
            })
        
        # Back button (bottom left)
        self.back_button = {
            'x': 20,
            'y': height - 70,
            'w': 120,
            'h': 50
        }
    
    def draw_glow_rect(self, img, x, y, w, h, color, glow_intensity):
        """Draw rectangle with glow effect"""
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        for i in range(3):
            offset = (i + 1) * 2
            alpha = glow_intensity * (1 - i * 0.3)
            glow_color = tuple(int(c * alpha) for c in color)
            cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), glow_color, 1)
    
    def draw(self, frame, grid_background):
        """Draw the level selection screen"""
        # Use grid background
        frame[:] = grid_background
        
        # Increment glow phase for animations
        self.glow_phase += 0.05
        
        # Draw title
        title_font_scale = 1.5
        title_thickness = 3
        (title_w, title_h), _ = cv2.getTextSize(self.title_text, cv2.FONT_HERSHEY_DUPLEX, title_font_scale, title_thickness)
        title_x = (self.width - title_w) // 2
        title_y = 80
        
        # Glow effect for title
        pulse = 0.6 + 0.4 * abs(math.sin(self.glow_phase * 0.8))
        cv2.putText(frame, self.title_text, (title_x, title_y), cv2.FONT_HERSHEY_DUPLEX, 
                   title_font_scale, self.COLORS['glow_cyan'], title_thickness + 2)
        cv2.putText(frame, self.title_text, (title_x, title_y), cv2.FONT_HERSHEY_DUPLEX, 
                   title_font_scale, self.COLORS['bright_blue'], title_thickness)
        
        # Draw subtitle
        subtitle_font_scale = 0.7
        subtitle_thickness = 1
        (subtitle_w, subtitle_h), _ = cv2.getTextSize(self.subtitle_text, cv2.FONT_HERSHEY_SIMPLEX, subtitle_font_scale, subtitle_thickness)
        subtitle_x = (self.width - subtitle_w) // 2
        subtitle_y = title_y + 40
        cv2.putText(frame, self.subtitle_text, (subtitle_x, subtitle_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   subtitle_font_scale, self.COLORS['text_secondary'], subtitle_thickness)
        
        # Draw level buttons
        for btn in self.level_buttons:
            self.draw_level_button(frame, btn)
        
        # Draw back button
        self.draw_back_button(frame)
    
    def draw_level_button(self, frame, btn):
        """Draw an individual level button"""
        # Different pulse speeds for each level for variety
        pulse_speed = 1.0 + (btn['level'] * 0.2)
        pulse = 0.5 + 0.5 * abs(math.sin(self.glow_phase * pulse_speed))
        
        # Hover effect - could be enhanced with mouse tracking
        is_locked = btn.get('locked', False)
        
        # Border glow
        if is_locked:
            border_color = self.COLORS['medium_blue']
            glow_intensity = pulse * 0.3
        else:
            border_color = self.COLORS['neon_blue']
            glow_intensity = pulse * 0.8
        
        self.draw_glow_rect(frame, btn['x'], btn['y'], btn['w'], btn['h'], border_color, glow_intensity)
        
        # Button fill
        overlay = frame.copy()
        if is_locked:
            fill_color = self.COLORS['dark_blue']
        else:
            fill_color = self.COLORS['button_normal']
        
        cv2.rectangle(overlay, (btn['x'] + 3, btn['y'] + 3), 
                     (btn['x'] + btn['w'] - 3, btn['y'] + btn['h'] - 3), 
                     fill_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Label "LEVEL" at top
        label_text = "LEVEL"
        label_font_scale = 0.6
        label_thickness = 1
        (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)
        label_x = btn['x'] + (btn['w'] - label_w) // 2
        label_y = btn['y'] + 30
        
        # Level number (big, centered)
        level_text = str(btn['level'])
        level_font_scale = 3.5
        level_thickness = 5
        (level_w, level_h), baseline = cv2.getTextSize(level_text, cv2.FONT_HERSHEY_DUPLEX, level_font_scale, level_thickness)
        level_x = btn['x'] + (btn['w'] - level_w) // 2
        level_y = btn['y'] + (btn['h'] // 2) + (level_h // 2)
        
        if is_locked:
            # Draw lock icon instead
            cv2.putText(frame, "🔒", (btn['x'] + btn['w'] // 2 - 20, level_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       2.0, self.COLORS['text_secondary'], 3)
        else:
            # Draw level number with glow
            cv2.putText(frame, level_text, (level_x, level_y), cv2.FONT_HERSHEY_DUPLEX, 
                       level_font_scale, self.COLORS['glow_cyan'], level_thickness + 2)
            cv2.putText(frame, level_text, (level_x, level_y), cv2.FONT_HERSHEY_DUPLEX, 
                       level_font_scale, self.COLORS['bright_blue'], level_thickness)
        
        text_color = self.COLORS['text_secondary'] if is_locked else self.COLORS['text_primary']
        cv2.putText(frame, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   label_font_scale, text_color, label_thickness)
        
        # Difficulty indicator
        difficulty_texts = ["EASY", "MEDIUM", "HARD", "EXPERT", "MASTER"]
        difficulty_text = difficulty_texts[btn['level'] - 1]
        diff_font_scale = 0.45
        diff_thickness = 1
        (diff_w, diff_h), _ = cv2.getTextSize(difficulty_text, cv2.FONT_HERSHEY_SIMPLEX, diff_font_scale, diff_thickness)
        diff_x = btn['x'] + (btn['w'] - diff_w) // 2
        diff_y = btn['y'] + btn['h'] - 20
        cv2.putText(frame, difficulty_text, (diff_x, diff_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   diff_font_scale, text_color, diff_thickness)
    
    def draw_back_button(self, frame):
        """Draw back button"""
        bb = self.back_button
        pulse = 0.4 + 0.3 * abs(math.sin(self.glow_phase))
        self.draw_glow_rect(frame, bb['x'], bb['y'], bb['w'], bb['h'], self.COLORS['medium_blue'], pulse)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (bb['x'] + 2, bb['y'] + 2), 
                     (bb['x'] + bb['w'] - 2, bb['y'] + bb['h'] - 2), 
                     self.COLORS['button_normal'], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        text = "< BACK"
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = bb['x'] + (bb['w'] - text_w) // 2
        text_y = bb['y'] + (bb['h'] + text_h) // 2
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, self.COLORS['text_primary'], thickness)
    
    def handle_click(self, x, y):
        """Handle mouse clicks"""
        # Check back button
        bb = self.back_button
        if bb['x'] <= x <= bb['x'] + bb['w'] and bb['y'] <= y <= bb['y'] + bb['h']:
            return ('back', None)
        
        # Check level buttons
        for btn in self.level_buttons:
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                if not btn.get('locked', False):
                    return ('level_selected', btn['level'])
        
        return (None, None)
