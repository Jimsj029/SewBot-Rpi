#!/usr/bin/env python3
"""
SewBot - Pattern Recognition System
Optimized for Raspberry Pi
Standalone version - All code in one file
"""

import cv2
import numpy as np
import math
import os
import time
import threading

# Try to import pygame for audio playback
try:
    import pygame.mixer
    AUDIO_AVAILABLE = True
    print("Audio support enabled (pygame)")
except ImportError:
    AUDIO_AVAILABLE = False
    print("No audio support - video will play without audio")


# ==================== TUTORIAL PLAYER CLASS ====================

class TutorialPlayer:
    def __init__(self, width=800, height=600, video_path=None, audio_path=None):
        self.width = width
        self.height = height
        self.glow_phase = 0
        self.skipped = False
        self.completed = False
        self.video_paused = False  # Track if video is paused at end
        
        # Progress bar position (updated during draw)
        self.progress_bar = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
        
        # Multi-video support for tutorial steps
        self.current_step = 0  # Current video index (0-4 for 5 videos)
        self.total_steps = 5
        self.videos_base_path = os.path.join(os.path.dirname(__file__), 'videos', 'sewing-set-up')
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
        
        # Load the first video
        self.load_current_video()
        
        # Colors
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
        
        # Buttons - Next and Skip
        self.next_button = {
            'x': self.width - 160,
            'y': self.height - 80,
            'w': 140,
            'h': 50,
            'text': 'NEXT'
        }
        
        self.replay_current_button = {
            'x': 160,
            'y': self.height - 60,
            'w': 140,
            'h': 40,
            'text': 'REPLAY'
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
    
    def handle_click(self, x, y):
        """Handle mouse clicks, returns action: 'next', 'done', 'replay_current', 'replay', 'continue', or None"""
        # Check skip all button (top right - skips entire tutorial)
        if not self.skipped and not self.completed:
            btn = self.skip_all_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
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
        
        # Check replay current video button (only when video is playing or paused)
        if not self.skipped and not self.completed:
            btn = self.replay_current_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                # Replay current video from beginning
                self.replay_current_video()
                return 'replay_current'
        
        # Check next/done button (only when video is playing or paused)
        if not self.skipped and not self.completed:
            btn = self.next_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
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
        border_color = tuple(int(c * glow_intensity) for c in self.COLORS['cyan'])
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
                   self.COLORS['text_primary'], thickness)
    
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
                     self.COLORS['cyan'], 2)
        
        # If no video frame, show placeholder content
        if self.current_frame_img is None:
            center_x = video_x + video_w // 2
            center_y = video_y + video_h // 2
            
            # Animated circle
            radius = int(30 + 20 * abs(math.sin(self.video_frame * 0.1)))
            
            cv2.circle(img, (center_x, center_y), radius, self.COLORS['bright_blue'], 3)
            cv2.circle(img, (center_x, center_y), radius + 10, self.COLORS['cyan'], 1)
            
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
                       self.COLORS['text_accent'], thickness)
            
            # Subtitle
            subtitle = "(No Video File Found)"
            font_scale = 0.7
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(subtitle, font, font_scale, thickness)
            text_x = center_x - text_w // 2
            text_y = center_y - 30
            cv2.putText(img, subtitle, (text_x, text_y), font, font_scale, 
                       self.COLORS['text_secondary'], thickness)
        
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
                         self.COLORS['cyan'], -1)
        
        # Border
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.COLORS['cyan'], 1)
        
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
            
            # Draw skip all button (top right)
            self.draw_button(img, self.skip_all_button, self.COLORS['button_normal'])
            
            # Draw replay current button (left side)
            self.draw_button(img, self.replay_current_button, self.COLORS['button_normal'])
            
            # Draw next/done button (right side)
            # Change text to "DONE" on last step
            if self.current_step >= self.total_steps - 1:
                self.next_button['text'] = 'DONE'
            else:
                self.next_button['text'] = 'NEXT'
            self.draw_button(img, self.next_button, self.COLORS['button_hover'])
            
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
                       self.COLORS['text_accent'], thickness)
            
            # Draw replay and continue buttons
            self.draw_button(img, self.replay_button, self.COLORS['button_normal'])
            self.draw_button(img, self.continue_button, self.COLORS['button_hover'])
    
    def draw_step_indicator(self, img):
        """Draw step indicator showing current progress (Step 1/5, etc.)"""
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"Step {self.current_step + 1} of {self.total_steps}"
        font_scale = 0.8
        thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
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
                     self.COLORS['cyan'], 2)
        
        # Text
        cv2.putText(img, text, (text_x, text_y), font, font_scale, 
                   self.COLORS['text_accent'], thickness)
    
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


# ==================== MAIN APPLICATION CLASS ====================

class SewBotApp:
    def __init__(self):
        self.width = 1024
        self.height = 600
        self.window_name = 'SewBot - Pattern Recognition System'
        self.state = 'main_menu'  # main_menu, tutorial, mode_selection, pattern
        self.glow_phase = 0
        self.running = True
        
        # Tutorial player - initialize with tutorial videos from videos/sewing-set-up
        self.tutorial_player = TutorialPlayer(self.width, self.height)
        
        # Pattern mode variables
        self.current_level = 1
        self.camera = None
        self.camera_initializing = False
        self.camera_detected = False
        self.camera_status_message = "Checking camera..."
        self.blueprint_folder = 'blueprint'
        self.uniform_width = 200
        self.uniform_height = 300
        self.alpha_blend = 0.9
        
        # Camera display area (centered, with borders)
        self.camera_width = 560
        self.camera_height = 420
        self.camera_x = (self.width - self.camera_width) // 2
        self.camera_y = 80
        
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
        
        # Level buttons for pattern mode (horizontal layout at top)
        self.level_buttons = []
        button_width, button_height = 70, 45
        spacing = 15
        total_width = (button_width * 5) + (spacing * 4)
        start_x = (self.width - total_width) // 2
        start_y = 20
        
        for i in range(1, 6):
            self.level_buttons.append({
                'level': i,
                'x': start_x + (button_width + spacing) * (i - 1),
                'y': start_y,
                'w': button_width,
                'h': button_height
            })
        
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
                    # Wallet mode not implemented yet
                    print("Wallet mode - Coming soon!")
                
                tb = self.tutorial_button
                if tb['x'] <= x <= tb['x'] + tb['w'] and tb['y'] <= y <= tb['y'] + tb['h']:
                    # Go back to tutorial state to replay it
                    self.state = 'tutorial'
                    self.tutorial_player.reset()
                
                bb = self.back_button
                if bb['x'] <= x <= bb['x'] + bb['w'] and bb['y'] <= y <= bb['y'] + bb['h']:
                    self.state = 'main_menu'
                    
            elif self.state == 'pattern':
                # Check level buttons
                for btn in self.level_buttons:
                    if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                        self.current_level = btn['level']
                        break
                
                # Check back button
                bb = self.back_button
                if bb['x'] <= x <= bb['x'] + bb['w'] and bb['y'] <= y <= bb['y'] + bb['h']:
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
    
    def load_blueprint(self, level):
        img_path = os.path.join(self.blueprint_folder, f'level{level}.png')
        if not os.path.exists(img_path):
            return None, None
        
        overlay_png = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if overlay_png is None:
            return None, None
        
        overlay_png = cv2.resize(overlay_png, (self.uniform_width, self.uniform_height))
        
        if len(overlay_png.shape) == 3 and overlay_png.shape[2] == 4:
            overlay = overlay_png[:, :, :3]
            alpha = overlay_png[:, :, 3] / 255.0
        else:
            if len(overlay_png.shape) == 2:
                overlay_png = cv2.cvtColor(overlay_png, cv2.COLOR_GRAY2BGR)
            overlay = overlay_png
            gray = cv2.cvtColor(overlay_png, cv2.COLOR_BGR2GRAY)
            # Apply threshold to show only lines, not background
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            alpha = (255 - binary) / 255.0
        
        return overlay, alpha
    
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
    
    def draw_pattern_mode(self, frame):
        # Use pre-rendered grid background
        frame[:] = self.grid_background
        
        # Draw level buttons at top
        for btn in self.level_buttons:
            is_selected = (btn['level'] == self.current_level)
            pulse = 0.5 + 0.5 * abs(math.sin(self.glow_phase * 1.3))
            
            if is_selected:
                self.draw_glow_rect(frame, btn['x'], btn['y'], btn['w'], btn['h'], self.COLORS['neon_blue'], pulse)
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (btn['x'] + 2, btn['y'] + 2), (btn['x'] + btn['w'] - 2, btn['y'] + btn['h'] - 2), self.COLORS['button_hover'], -1)
                cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                
                text = f"LVL {btn['level']}"
                font_scale, thickness = 0.7, 2
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
                text_x = btn['x'] + (btn['w'] - text_w) // 2
                text_y = btn['y'] + (btn['h'] + text_h) // 2
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, self.COLORS['glow_cyan'], thickness + 1)
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, self.COLORS['text_primary'], thickness)
            else:
                cv2.rectangle(frame, (btn['x'], btn['y']), (btn['x'] + btn['w'], btn['y'] + btn['h']), self.COLORS['medium_blue'], 2)
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (btn['x'] + 2, btn['y'] + 2), (btn['x'] + btn['w'] - 2, btn['y'] + btn['h'] - 2), self.COLORS['dark_blue'], -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
                text = f"LVL {btn['level']}"
                font_scale, thickness = 0.7, 1
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
                text_x = btn['x'] + (btn['w'] - text_w) // 2
                text_y = btn['y'] + (btn['h'] + text_h) // 2
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, self.COLORS['text_secondary'], thickness)
        
        # Draw camera frame border with glow effect
        border_pulse = 0.4 + 0.3 * abs(math.sin(self.glow_phase * 0.7))
        self.draw_glow_rect(frame, self.camera_x - 5, self.camera_y - 5, 
                           self.camera_width + 10, self.camera_height + 10, 
                           self.COLORS['bright_blue'], border_pulse)
        
        # Get and display camera feed
        if self.camera is None or not self.camera.isOpened():
            cv2.rectangle(frame, (self.camera_x, self.camera_y), 
                         (self.camera_x + self.camera_width, self.camera_y + self.camera_height), 
                         self.COLORS['dark_blue'], -1)
            text = "Initializing camera..." if self.camera_initializing else "Camera not available"
            cv2.putText(frame, text, (self.camera_x + 80, self.camera_y + self.camera_height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS['text_primary'], 2)
        else:
            ret, cam_frame = self.camera.read()
            if not ret:
                cv2.rectangle(frame, (self.camera_x, self.camera_y), 
                             (self.camera_x + self.camera_width, self.camera_y + self.camera_height), 
                             self.COLORS['dark_blue'], -1)
                cv2.putText(frame, "Failed to read camera", (self.camera_x + 100, self.camera_y + self.camera_height // 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS['text_primary'], 2)
            else:
                cam_frame = cv2.resize(cam_frame, (self.camera_width, self.camera_height))
                
                # Apply blueprint overlay
                overlay, alpha = self.load_blueprint(self.current_level)
                if overlay is not None and alpha is not None:
                    overlay_h, overlay_w = overlay.shape[:2]
                    x_offset = (self.camera_width - overlay_w) // 2
                    y_offset = (self.camera_height - overlay_h) // 2
                    
                    if x_offset >= 0 and y_offset >= 0:
                        roi = cam_frame[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w]
                        for c in range(3):
                            roi[:, :, c] = (alpha * overlay[:, :, c] * self.alpha_blend + 
                                          (1 - alpha * self.alpha_blend) * roi[:, :, c])
                        cam_frame[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w] = roi
                
                frame[self.camera_y:self.camera_y+self.camera_height, 
                      self.camera_x:self.camera_x+self.camera_width] = cam_frame
        
        # Draw corner accents
        corner_size = 25
        corner_thickness = 3
        cv2.line(frame, (self.camera_x - 15, self.camera_y - 15), 
                (self.camera_x - 15 + corner_size, self.camera_y - 15), self.COLORS['cyan'], corner_thickness)
        cv2.line(frame, (self.camera_x - 15, self.camera_y - 15), 
                (self.camera_x - 15, self.camera_y - 15 + corner_size), self.COLORS['cyan'], corner_thickness)
        cv2.line(frame, (self.camera_x + self.camera_width + 15, self.camera_y - 15), 
                (self.camera_x + self.camera_width + 15 - corner_size, self.camera_y - 15), self.COLORS['cyan'], corner_thickness)
        cv2.line(frame, (self.camera_x + self.camera_width + 15, self.camera_y - 15), 
                (self.camera_x + self.camera_width + 15, self.camera_y - 15 + corner_size), self.COLORS['cyan'], corner_thickness)
        cv2.line(frame, (self.camera_x - 15, self.camera_y + self.camera_height + 15), 
                (self.camera_x - 15 + corner_size, self.camera_y + self.camera_height + 15), self.COLORS['cyan'], corner_thickness)
        cv2.line(frame, (self.camera_x - 15, self.camera_y + self.camera_height + 15), 
                (self.camera_x - 15, self.camera_y + self.camera_height + 15 - corner_size), self.COLORS['cyan'], corner_thickness)
        cv2.line(frame, (self.camera_x + self.camera_width + 15, self.camera_y + self.camera_height + 15), 
                (self.camera_x + self.camera_width + 15 - corner_size, self.camera_y + self.camera_height + 15), self.COLORS['cyan'], corner_thickness)
        cv2.line(frame, (self.camera_x + self.camera_width + 15, self.camera_y + self.camera_height + 15), 
                (self.camera_x + self.camera_width + 15, self.camera_y + self.camera_height + 15 - corner_size), self.COLORS['cyan'], corner_thickness)
        
        self.draw_back_button(frame)
    
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
                elif self.state == 'mode_selection':
                    self.draw_mode_selection(frame)
                elif self.state == 'pattern':
                    self.draw_pattern_mode(frame)
                
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
        cv2.destroyAllWindows()
        print("Program closed.")


# ==================== MAIN ENTRY POINT ====================

if __name__ == '__main__':
    app = SewBotApp()
    app.run()
