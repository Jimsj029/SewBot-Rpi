

import cv2
import numpy as np
import math
import os
import sys
import time

try:
    from .typography import (
        FONT_MAIN,
        text_scale,
        text_thickness,
        get_text_size,
        fit_text_scale,
        draw_text,
    )
except ImportError:
    from typography import (
        FONT_MAIN,
        text_scale,
        text_thickness,
        get_text_size,
        fit_text_scale,
        draw_text,
    )

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from music_manager import get_music_manager

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

UI_FONT = FONT_MAIN


def _put_text(img, text, x, y, scale, color, thickness):
    """Draw outlined text for consistent readability."""
    draw_text(img, text, x, y, scale, color, thickness, font=UI_FONT)


class TutorialPlayer:
    def __init__(self, width=800, height=600, video_path=None, audio_path=None,
                 videos_subfolder='sewing-set-up', tutorial_label='TUTORIAL'):
        self.width = width
        self.height = height
        self.tutorial_label = tutorial_label
        self.glow_phase = 0
        self.skipped = False
        self.completed = False
        self.video_paused = False  # Track if video is paused at end
        
        # Progress bar position (updated during draw)
        self.progress_bar = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
        
        # Multi-video support for tutorial steps
        self.current_step = 0  # Current video index (0-4 for 5 videos)
        self.total_steps = 8
        self.videos_base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'videos', videos_subfolder)
        self.video_files = []  # Will store paths to step1.mov through step5.mov
        self.load_video_list()
        
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
        
        # Music manager for sound effects
        self.music_manager = get_music_manager()
        
        # Load the first video
        self.load_current_video()
        
        # Buttons - Next and Skip
        self.next_button = {
            'x': self.width - 160,
            'y': self.height - 80,
            'w': 140,
            'h': 50,
            'text': 'NEXT'
        }
        
        self.replay_current_button = {
            'x': 170,
            'y': self.height - 80,
            'w': 140,
            'h': 50,
            'text': 'REPLAY'
        }
        
        self.previous_button = {
            'x': 20,
            'y': self.height - 80,
            'w': 140,
            'h': 50,
            'text': 'PREVIOUS'
        }
        
        self.back_button = {
            'x': 20,
            'y': 20,
            'w': 140,
            'h': 50,
            'text': '< BACK'
        }
        
        self.skip_all_button = {
            'x': self.width - 160,
            'y': 20,
            'w': 140,
            'h': 50,
            'text': 'SKIP ALL'
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
    
    def load_video_list(self):
        """Load the list of tutorial video files (step1 through step5)"""
        self.video_files = []
        
        # Check for step1.mov/mp4 through step5.mov/mp4
        for i in range(1, self.total_steps + 1):
            # Try both .mov and .mp4 extensions
            video_found = False
            # Try different naming patterns: "Step 1.MOV", "step1.mov", "step 1.mov", etc.
            patterns = [
                f'Step {i}.MOV',
                f'Step {i}.mov',
                f'Step {i}.MP4',
                f'Step {i}.mp4',
                f'step{i}.MOV',
                f'step{i}.mov',
                f'step{i}.MP4',
                f'step{i}.mp4',
                f'step {i}.MOV',
                f'step {i}.mov',
                f'step {i}.MP4',
                f'step {i}.mp4',
            ]
            
            for pattern in patterns:
                video_path = os.path.join(self.videos_base_path, pattern)
                if os.path.exists(video_path):
                    self.video_files.append(video_path)
                    video_found = True
                    print(f"Found tutorial video: {pattern}")
                    break
            
            if not video_found:
                # Add None as placeholder if video not found
                self.video_files.append(None)
                print(f"Warning: step{i} video not found in {self.videos_base_path}")
    
    def load_current_video(self):
        """Load the video for the current step"""
        # Release previous video if any
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Reset video state
        self.video_frame = 0
        self.current_frame_img = None
        self.completed = False
        self.video_paused = False
        self.last_frame_time = time.time()
        self.frame_accumulator = 0.0
        
        # Load current step's video
        if self.current_step < len(self.video_files) and self.video_files[self.current_step]:
            video_path = self.video_files[self.current_step]
            self.cap = cv2.VideoCapture(video_path)
            if self.cap.isOpened():
                self.max_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps > 0:
                    self.frame_time = 1.0 / self.fps
                print(f"Loaded video: {video_path} ({self.max_frames} frames @ {self.fps} fps)")
            else:
                self.cap = None
                print(f"Failed to open video: {video_path}")
    
    def next_step(self):
        """Move to the next tutorial step"""
        if self.current_step < self.total_steps - 1:
            self.current_step += 1
            self.load_current_video()
            return True
        else:
            # Last step completed
            return False
    
    def previous_step(self):
        """Move to the previous tutorial step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.load_current_video()
            return True
        else:
            # Already on first step
            return False
    
    def replay_current_video(self):
        """Replay the current video from the beginning"""
        self.video_frame = 0
        self.video_paused = False
        self.current_frame_img = None
        self.last_frame_time = time.time()
        self.frame_accumulator = 0.0
        
        # Reset video capture to beginning
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def seek_to_position(self, percentage):
        """Seek to a specific position in the current video (0.0 to 1.0)"""
        if self.cap and self.cap.isOpened() and self.max_frames > 0:
            # Calculate target frame
            target_frame = int(self.max_frames * percentage)
            target_frame = max(0, min(target_frame, self.max_frames - 1))  # Clamp to valid range
            
            # Seek to the target frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.video_frame = target_frame
            
            # Read the frame at this position
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_img = frame
            
            # Reset timing
            self.last_frame_time = time.time()
            self.frame_accumulator = 0.0
            
            # Unpause if video was paused
            self.video_paused = False
    
    def reset(self):
        """Reset tutorial state to beginning"""
        self.current_step = 0
        self.video_frame = 0
        self.skipped = False
        self.completed = False
        self.video_paused = False
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
        
        # Load the first video
        self.load_current_video()
    
    def play_button_click_sound(self):
        """Play button click sound effect"""
        try:
            self.music_manager.play_sound_effect('button_click.mp3')
        except Exception as e:
            pass  # Silently fail if sound effect doesn't exist
    
    def handle_click(self, x, y):
        """Handle mouse clicks, returns action: 'next', 'done', 'replay_current', 'replay', 'continue', 'back', 'previous', or None"""
        # Check back button (top left)
        if not self.skipped and not self.completed:
            btn = self.back_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                return 'back'
        
        # Check skip all button (top right - skips entire tutorial)
        if not self.skipped and not self.completed:
            btn = self.skip_all_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                self.skipped = True
                # Stop audio when skipping
                if AUDIO_AVAILABLE and pygame.mixer.get_init():
                    try:
                        pygame.mixer.music.stop()
                    except:
                        pass
                return 'continue'  # Skip all tutorials and go to mode selection
        
        # Check progress bar click for seeking (only when video is playing or paused)
        if not self.skipped and not self.completed:
            bar = self.progress_bar
            if bar['x'] <= x <= bar['x'] + bar['w'] and bar['y'] <= y <= bar['y'] + bar['h']:
                # Calculate the position clicked as a percentage
                click_percentage = (x - bar['x']) / bar['w']
                self.seek_to_position(click_percentage)
                return 'seek'
        
        # Check previous button (only when video is playing or paused and not on first step)
        if not self.skipped and not self.completed and self.current_step > 0:
            btn = self.previous_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                # Go to previous step
                if self.previous_step():
                    return 'previous'
        
        # Check replay current video button (only when video is playing or paused)
        if not self.skipped and not self.completed:
            btn = self.replay_current_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                # Replay current video from beginning
                self.replay_current_video()
                return 'replay_current'
        
        # Check next/done button (only when video is playing or paused)
        if not self.skipped and not self.completed:
            btn = self.next_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                # If on last step, this is the Done button
                if self.current_step >= self.total_steps - 1:
                    return 'continue'  # Done with all tutorials
                else:
                    # Move to next step
                    if self.next_step():
                        return 'next'
                    else:
                        return 'continue'
        
        # Check continue button (after skip or completion)
        if self.skipped or self.completed:
            btn = self.continue_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                return 'continue'
        
        # Check replay button (after skip or completion)
        if self.skipped or self.completed:
            btn = self.replay_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                return 'replay'
        
        return None
    
    def update(self):
        """Update animation and video progress"""
        self.glow_phase += 0.05
        
        # Update video frame with proper timing (only if not paused)
        if not self.skipped and not self.completed and not self.video_paused:
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
                        # Video finished - pause instead of completing
                        self.video_paused = True
                        break
            else:
                # Fallback to placeholder animation
                if self.frame_accumulator >= self.frame_time:
                    self.video_frame += 1
                    self.frame_accumulator -= self.frame_time
                    if self.video_frame >= self.max_frames:
                        self.video_paused = True
    
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
        font_scale = text_scale(0.82, self.width, self.height, floor=0.74, ceiling=0.94)
        thickness = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
        text = btn['text']

        font_scale = fit_text_scale(text, UI_FONT, w - 14, font_scale, thickness, min_scale=0.66)
        
        (text_w, text_h), baseline = get_text_size(text, UI_FONT, font_scale, thickness)
        text_x = x + (w - text_w) // 2
        text_y = y + (h + text_h) // 2
        
        _put_text(img, text, text_x, text_y, font_scale, COLORS['text_primary'], thickness)
    
    def draw_video_frame(self, img):
        """Draw actual video frame or placeholder"""
        # Video area - moved down for better spacing
        video_margin = 50
        video_x = video_margin
        video_y = 90  # Moved down from 50 to give more space for step indicator
        video_w = self.width - 2 * video_margin
        video_h = self.height - video_y - video_margin - 100
        
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
            # Title
            title = "TUTORIAL VIDEO"
            font_scale = text_scale(1.2, self.width, self.height, floor=1.02, ceiling=1.35)
            thickness = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
            font_scale = fit_text_scale(title, UI_FONT, video_w - 50, font_scale, thickness, min_scale=0.9)
            (text_w, text_h), _ = get_text_size(title, UI_FONT, font_scale, thickness)
            text_x = center_x - text_w // 2
            text_y = center_y - 60
            _put_text(img, title, text_x, text_y, font_scale, COLORS['text_accent'], thickness)
            
            # Subtitle
            subtitle = "(No Video File Found)"
            font_scale = text_scale(0.76, self.width, self.height, floor=0.7, ceiling=0.88)
            thickness = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
            font_scale = fit_text_scale(subtitle, UI_FONT, video_w - 50, font_scale, thickness, min_scale=0.62)
            (text_w, text_h), _ = get_text_size(subtitle, UI_FONT, font_scale, thickness)
            text_x = center_x - text_w // 2
            text_y = center_y - 30
            _put_text(img, subtitle, text_x, text_y, font_scale, COLORS['text_secondary'], thickness)
        
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
        
        # Store progress bar position for click detection
        self.progress_bar = {'x': bar_x, 'y': bar_y, 'w': bar_width, 'h': bar_height}
    
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
            
            # Draw step indicator
            self.draw_step_indicator(img)
            
            # Draw back button (top left)
            self.draw_button(img, self.back_button, COLORS['button_hover'])
            
            # Draw skip all button (top right)
            self.draw_button(img, self.skip_all_button, COLORS['button_normal'])
            
            # Draw previous button (bottom left) - only show if not on first step
            if self.current_step > 0:
                self.draw_button(img, self.previous_button, COLORS['button_normal'])
            
            # Draw replay current button (bottom left, next to previous)
            self.draw_button(img, self.replay_current_button, COLORS['button_normal'])
            
            # Draw next/done button (right side)
            # Change text to "DONE" on last step
            if self.current_step >= self.total_steps - 1:
                self.next_button['text'] = 'DONE'
            else:
                self.next_button['text'] = 'NEXT'
            self.draw_button(img, self.next_button, COLORS['button_hover'])
            
        else:
            # Completed or skipped
            title = f"{self.tutorial_label} COMPLETED" if self.completed else self.tutorial_label
            font_scale = text_scale(1.5, self.width, self.height, floor=1.25, ceiling=1.7)
            thickness = text_thickness(3, self.width, self.height, min_thickness=2, max_thickness=4)
            font_scale = fit_text_scale(title, UI_FONT, self.width - 80, font_scale, thickness, min_scale=1.0)
            
            (text_w, text_h), _ = get_text_size(title, UI_FONT, font_scale, thickness)
            text_x = (self.width - text_w) // 2
            text_y = self.height // 3
            
            _put_text(img, title, text_x, text_y, font_scale, COLORS['text_accent'], thickness)
            
            # Draw replay and continue buttons
            self.draw_button(img, self.replay_button, COLORS['button_normal'])
            self.draw_button(img, self.continue_button, COLORS['button_hover'])
    
    def draw_step_indicator(self, img):
        """Draw step indicator showing current progress (Step 1/5, etc.)"""
        text = f"Step {self.current_step + 1} of {self.total_steps}"
        font_scale = text_scale(0.88, self.width, self.height, floor=0.76, ceiling=0.98)
        thickness = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
        font_scale = fit_text_scale(text, UI_FONT, self.width - 220, font_scale, thickness, min_scale=0.68)
        
        (text_w, text_h), baseline = get_text_size(text, UI_FONT, font_scale, thickness)
        # Keep centered horizontally, align vertically with skip all button
        text_x = (self.width - text_w) // 2
        text_y = 20 + (50 + text_h) // 2  # Same Y position as skip all button
        
        # Background for better visibility
        padding = 10
        cv2.rectangle(img, 
                     (text_x - padding, text_y - text_h - padding),
                     (text_x + text_w + padding, text_y + padding),
                     (40, 40, 40), -1)
        
        # Border
        cv2.rectangle(img, 
                     (text_x - padding, text_y - text_h - padding),
                     (text_x + text_w + padding, text_y + padding),
                     COLORS['cyan'], 2)
        
        # Text
        _put_text(img, text, text_x, text_y, font_scale, COLORS['text_accent'], thickness)
    
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
