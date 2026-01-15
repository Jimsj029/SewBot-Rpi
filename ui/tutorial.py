"""
Tutorial Video Player
Shows tutorial video with skip and replay options
"""

import cv2
import numpy as np
import math
import os
import time

# Try to import pygame for audio playback
try:
    import pygame.mixer
    AUDIO_AVAILABLE = True
    print("Audio support enabled (pygame)")
except ImportError:
    AUDIO_AVAILABLE = False
    print("No audio support - video will play without audio")


# Colors
COLORS = {
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

FONTS = {
    'title_size': 2.5
}


class TutorialPlayer:
    def __init__(self, width=800, height=600, video_path=None, audio_path=None):
        self.width = width
        self.height = height
        self.glow_phase = 0
        self.skipped = False
        self.completed = False
        
        # Video capture
        self.video_path = video_path
        self.cap = None
        self.video_frame = 0
        self.max_frames = 300  # ~10 seconds at 30fps (fallback)
        self.current_frame_img = None
        self.fps = 30.0  # Default fallback FPS
        self.frame_time = 1.0 / self.fps  # Time per frame in seconds
        self.last_frame_time = time.time()
        self.frame_accumulator = 0.0
        
        # Audio support - direct audio file
        self.audio_path = audio_path
        self.has_audio = False
        self.audio_started = False
        
        # Initialize pygame mixer for audio
        if AUDIO_AVAILABLE and audio_path and os.path.exists(audio_path):
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
                pygame.mixer.music.load(audio_path)
                self.has_audio = True
                print(f"Audio loaded successfully: {audio_path}")
            except Exception as e:
                print(f"Failed to load audio: {e}")
        
        # Try to load video if path provided
        if video_path and os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)
            if self.cap.isOpened():
                self.max_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps > 0:
                    self.frame_time = 1.0 / self.fps
            else:
                self.cap = None
        
        # Buttons
        self.skip_button = {
            'x': self.width - 160,
            'y': self.height - 80,
            'w': 140,
            'h': 50,
            'text': 'SKIP'
        }
        
        self.replay_button = {
            'x': (self.width - 200) // 2,
            'y': self.height - 100,
            'w': 200,
            'h': 60,
            'text': 'WATCH AGAIN'
        }
        
        self.continue_button = {
            'x': (self.width - 200) // 2,
            'y': self.height // 2 + 80,
            'w': 200,
            'h': 60,
            'text': 'CONTINUE'
        }
    
    def reset(self):
        """Reset tutorial state"""
        self.video_frame = 0
        self.skipped = False
        self.completed = False
        self.current_frame_img = None
        self.last_frame_time = time.time()
        self.frame_accumulator = 0.0
        self.audio_started = False
        
        # Stop audio if playing
        if AUDIO_AVAILABLE and pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
            except:
                pass
        
        # Reset video capture to beginning
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def handle_click(self, x, y):
        """Handle mouse clicks, returns action: 'skip', 'replay', 'continue', or None"""
        # Check skip button (only when video is playing)
        if not self.skipped and not self.completed:
            btn = self.skip_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.skipped = True
                # Stop audio when skipping
                if AUDIO_AVAILABLE and pygame.mixer.get_init():
                    try:
                        pygame.mixer.music.stop()
                    except:
                        pass
                return 'continue'  # Skip directly to continue
        
        # Check continue button (after skip or completion)
        if self.skipped or self.completed:
            btn = self.continue_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                return 'continue'
        
        # Check replay button (after skip or completion)
        if self.skipped or self.completed:
            btn = self.replay_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                return 'replay'
        
        return None
    
    def update(self):
        """Update animation and video progress"""
        self.glow_phase += 0.05
        
        # Update video frame with proper timing
        if not self.skipped and not self.completed:
            # Calculate delta time
            current_time = time.time()
            delta_time = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            # Accumulate time for frame advancement
            self.frame_accumulator += delta_time
            
            # Read next frame(s) based on elapsed time
            if self.cap and self.cap.isOpened():
                # Advance frames based on accumulated time
                while self.frame_accumulator >= self.frame_time:
                    ret, frame = self.cap.read()
                    if ret:
                        self.current_frame_img = frame
                        self.video_frame += 1
                        self.frame_accumulator -= self.frame_time
                    else:
                        self.completed = True
                        break
            else:
                # Fallback to placeholder animation
                if self.frame_accumulator >= self.frame_time:
                    self.video_frame += 1
                    self.frame_accumulator -= self.frame_time
                    if self.video_frame >= self.max_frames:
                        self.completed = True
    
    def draw_button(self, img, btn, color_normal, color_hover=None):
        """Draw a button with glow effect"""
        if color_hover is None:
            color_hover = color_normal
        
        x, y, w, h = btn['x'], btn['y'], btn['w'], btn['h']
        
        # Button background
        cv2.rectangle(img, (x, y), (x + w, y + h), color_normal, -1)
        
        # Border
        glow_intensity = 0.5 + 0.5 * abs(math.sin(self.glow_phase))
        border_color = tuple(int(c * glow_intensity) for c in COLORS['cyan'])
        cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 2)
        
        # Text
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        thickness = 2
        text = btn['text']
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x + (w - text_w) // 2
        text_y = y + (h + text_h) // 2
        
        cv2.putText(img, text, (text_x, text_y), font, font_scale, 
                   COLORS['text_primary'], thickness)
    
    def draw_video_frame(self, img):
        """Draw actual video frame or placeholder"""
        # Video area
        video_margin = 50
        video_x = video_margin
        video_y = video_margin
        video_w = self.width - 2 * video_margin
        video_h = self.height - 2 * video_margin - 100
        
        # Dark background for video
        cv2.rectangle(img, (video_x, video_y), (video_x + video_w, video_y + video_h), 
                     (30, 30, 30), -1)
        
        # Draw actual video frame if available
        if self.current_frame_img is not None:
            # Resize video frame to fit the video area while maintaining aspect ratio
            frame_h, frame_w = self.current_frame_img.shape[:2]
            
            # Calculate scaling factor to fit within video area
            scale_w = video_w / frame_w
            scale_h = video_h / frame_h
            scale = min(scale_w, scale_h)
            
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            
            # Resize frame
            resized_frame = cv2.resize(self.current_frame_img, (new_w, new_h))
            
            # Center the frame in video area
            offset_x = video_x + (video_w - new_w) // 2
            offset_y = video_y + (video_h - new_h) // 2
            
            # Copy frame to main image
            img[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_frame
        
        # Border
        cv2.rectangle(img, (video_x, video_y), (video_x + video_w, video_y + video_h), 
                     COLORS['cyan'], 2)
        
        # If no video frame, show placeholder content
        if self.current_frame_img is None:
            center_x = video_x + video_w // 2
            center_y = video_y + video_h // 2
            
            # Animated circle
            radius = int(30 + 20 * abs(math.sin(self.video_frame * 0.1)))
            
            cv2.circle(img, (center_x, center_y), radius, COLORS['bright_blue'], 3)
            cv2.circle(img, (center_x, center_y), radius + 10, COLORS['cyan'], 1)
            
            # Tutorial text
            font = cv2.FONT_HERSHEY_DUPLEX
            
            # Title
            title = "TUTORIAL VIDEO"
            font_scale = 1.2
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
            text_x = center_x - text_w // 2
            text_y = center_y - 60
            cv2.putText(img, title, (text_x, text_y), font, font_scale, 
                       COLORS['text_accent'], thickness)
            
            # Subtitle
            subtitle = "(No Video File Found)"
            font_scale = 0.7
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(subtitle, font, font_scale, thickness)
            text_x = center_x - text_w // 2
            text_y = center_y - 30
            cv2.putText(img, subtitle, (text_x, text_y), font, font_scale, 
                       COLORS['text_secondary'], thickness)
        
        # Progress bar
        progress = self.video_frame / self.max_frames if self.max_frames > 0 else 0
        bar_width = 300
        bar_height = 10
        bar_x = video_x + (video_w - bar_width) // 2
        bar_y = video_y + video_h + 20
        
        # Background
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Progress fill
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         COLORS['cyan'], -1)
        
        # Border
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     COLORS['cyan'], 1)
        
        # Progress text
        progress_text = f"{self.video_frame}/{self.max_frames}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(progress_text, font, font_scale, thickness)
        text_x = bar_x + bar_width + 15
        text_y = bar_y + bar_height
        cv2.putText(img, progress_text, (text_x, text_y), font, font_scale, 
                   COLORS['text_secondary'], thickness)
    
    def draw(self, img):
        """Main draw function"""
        self.update()
        
        # Draw video or completion screen
        if not self.skipped and not self.completed:
            # Start audio playback when video starts (if audio is ready)
            if not self.audio_started and self.has_audio and AUDIO_AVAILABLE:
                try:
                    pygame.mixer.music.play()
                    self.audio_started = True
                    print(">>> Audio playback started! <<<")
                except Exception as e:
                    print(f"Failed to start audio: {e}")
            
            # Playing video
            self.draw_video_frame(img)
            
            # Draw skip button
            self.draw_button(img, self.skip_button, COLORS['button_normal'])
            
        else:
            # Completed or skipped
            font = cv2.FONT_HERSHEY_DUPLEX
            title = "TUTORIAL COMPLETED" if self.completed else "TUTORIAL"
            font_scale = 1.5
            thickness = 3
            
            (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
            text_x = (self.width - text_w) // 2
            text_y = self.height // 3
            
            cv2.putText(img, title, (text_x, text_y), font, font_scale, 
                       COLORS['text_accent'], thickness)
            
            # Draw replay and continue buttons
            self.draw_button(img, self.replay_button, COLORS['button_normal'])
            self.draw_button(img, self.continue_button, COLORS['button_hover'])
    
    def cleanup(self):
        """Release video capture and audio resources"""
        if self.cap:
            self.cap.release()
        
        # Stop audio playback
        if AUDIO_AVAILABLE and pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
            except:
                pass
