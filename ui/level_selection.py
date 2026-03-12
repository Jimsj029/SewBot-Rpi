"""
Level Selection Screen - Choose a level before starting pattern mode
"""

import cv2
import numpy as np
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from music_manager import get_music_manager

try:
    from .typography import (
        FONT_MAIN,
        FONT_DISPLAY,
        text_scale,
        text_thickness,
        get_text_size,
        fit_text_scale,
        draw_text,
    )
except ImportError:
    from typography import (
        FONT_MAIN,
        FONT_DISPLAY,
        text_scale,
        text_thickness,
        get_text_size,
        fit_text_scale,
        draw_text,
    )


def _put_text(img, text, x, y, scale, color, thickness, font=FONT_MAIN):
    """Draw outlined text for readability on glowing backgrounds."""
    draw_text(img, text, x, y, scale, color, thickness, font=font)


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
        button_y = 210  # Positioned higher for better vertical centering
        
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
        
        # Music manager for sound effects
        self.music_manager = get_music_manager()
    
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
        title_font_scale = text_scale(1.58, self.width, self.height, floor=1.32, ceiling=1.78)
        title_thickness = text_thickness(3, self.width, self.height, min_thickness=2, max_thickness=4)
        title_font_scale = fit_text_scale(self.title_text, FONT_DISPLAY, self.width - 120, title_font_scale, title_thickness, min_scale=1.08)
        (title_w, title_h), _ = get_text_size(self.title_text, FONT_DISPLAY, title_font_scale, title_thickness)
        title_x = (self.width - title_w) // 2
        title_y = 80
        
        # Glow effect for title
        pulse = 0.6 + 0.4 * abs(math.sin(self.glow_phase * 0.8))
        glow_color = tuple(int(c * pulse) for c in self.COLORS['glow_cyan'])
        draw_text(
            frame,
            self.title_text,
            title_x,
            title_y,
            title_font_scale,
            self.COLORS['bright_blue'],
            title_thickness,
            font=FONT_DISPLAY,
            outline_color=glow_color,
            outline_extra=2,
        )
        
        # Draw subtitle
        subtitle_font_scale = text_scale(0.84, self.width, self.height, floor=0.74, ceiling=0.96)
        subtitle_thickness = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        subtitle_font_scale = fit_text_scale(self.subtitle_text, FONT_MAIN, self.width - 160, subtitle_font_scale, subtitle_thickness, min_scale=0.66)
        (subtitle_w, subtitle_h), _ = get_text_size(self.subtitle_text, FONT_MAIN, subtitle_font_scale, subtitle_thickness)
        subtitle_x = (self.width - subtitle_w) // 2
        subtitle_y = title_y + 40
        _put_text(frame, self.subtitle_text, subtitle_x, subtitle_y, subtitle_font_scale, self.COLORS['text_secondary'], subtitle_thickness)
        
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
        label_font_scale = text_scale(0.66, self.width, self.height, floor=0.58, ceiling=0.76)
        label_thickness = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
        (label_w, label_h), _ = get_text_size(label_text, FONT_MAIN, label_font_scale, label_thickness)
        label_x = btn['x'] + (btn['w'] - label_w) // 2
        label_y = btn['y'] + 30
        
        # Level number (big, centered)
        level_text = str(btn['level'])
        level_font_scale = text_scale(3.1, self.width, self.height, floor=2.6, ceiling=3.4)
        level_thickness = text_thickness(5, self.width, self.height, min_thickness=3, max_thickness=6)
        level_font_scale = fit_text_scale(level_text, FONT_DISPLAY, btn['w'] - 28, level_font_scale, level_thickness, min_scale=2.2)
        (level_w, level_h), baseline = get_text_size(level_text, FONT_DISPLAY, level_font_scale, level_thickness)
        level_x = btn['x'] + (btn['w'] - level_w) // 2
        level_y = btn['y'] + (btn['h'] // 2) + (level_h // 2)
        
        if is_locked:
            lock_text = "LOCKED"
            lock_scale = text_scale(0.82, self.width, self.height, floor=0.72, ceiling=0.94)
            lock_thickness = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
            lock_scale = fit_text_scale(lock_text, FONT_MAIN, btn['w'] - 16, lock_scale, lock_thickness, min_scale=0.62)
            (lock_w, lock_h), _ = get_text_size(lock_text, FONT_MAIN, lock_scale, lock_thickness)
            lock_x = btn['x'] + (btn['w'] - lock_w) // 2
            lock_y = btn['y'] + (btn['h'] // 2) + (lock_h // 2)
            _put_text(frame, lock_text, lock_x, lock_y, lock_scale, self.COLORS['text_secondary'], lock_thickness)
        else:
            # Draw level number with glow
            draw_text(
                frame,
                level_text,
                level_x,
                level_y,
                level_font_scale,
                self.COLORS['bright_blue'],
                level_thickness,
                font=FONT_DISPLAY,
                outline_color=self.COLORS['glow_cyan'],
                outline_extra=2,
            )
        
        text_color = self.COLORS['text_secondary'] if is_locked else self.COLORS['text_primary']
        _put_text(frame, label_text, label_x, label_y, label_font_scale, text_color, label_thickness)
        
        
    
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
        font_scale = text_scale(0.76, self.width, self.height, floor=0.68, ceiling=0.9)
        thickness = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
        font_scale = fit_text_scale(text, FONT_MAIN, bb['w'] - 12, font_scale, thickness, min_scale=0.58)
        (text_w, text_h), _ = get_text_size(text, FONT_MAIN, font_scale, thickness)
        text_x = bb['x'] + (bb['w'] - text_w) // 2
        text_y = bb['y'] + (bb['h'] + text_h) // 2
        _put_text(frame, text, text_x, text_y, font_scale, self.COLORS['text_primary'], thickness)
    
    def play_button_click_sound(self):
        """Play button click sound effect"""
        try:
            self.music_manager.play_sound_effect('button_click.mp3')
        except Exception as e:
            pass  # Silently fail if sound effect doesn't exist
    
    def handle_click(self, x, y):
        """Handle mouse clicks"""
        # Check back button
        bb = self.back_button
        if bb['x'] <= x <= bb['x'] + bb['w'] and bb['y'] <= y <= bb['y'] + bb['h']:
            self.play_button_click_sound()
            return ('back', None)
        
        # Check level buttons
        for btn in self.level_buttons:
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                if not btn.get('locked', False):
                    self.play_button_click_sound()
                    return ('level_selected', btn['level'])
        
        return (None, None)
