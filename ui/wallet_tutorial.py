

import cv2
import numpy as np
import math
import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from music_manager import get_music_manager
from ultralytics import YOLO

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


class WalletTutorialPlayer:
    def __init__(self, width=800, height=600, video_path=None, audio_path=None):
        self.width = width
        self.height = height
        self.glow_phase = 0
        self.skipped = False
        self.completed = False
        self.video_paused = False  # Track if video is paused at end
        self.your_turn_mode = False  # Track if in "your turn" practice mode
        
        # Progress bar position (updated during draw)
        self.progress_bar = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
        
        # Multi-video support for wallet tutorial steps
        self.current_step = 0  # Current video index (0-10 for 11 videos)
        self.total_steps = 11  # Wallet: Materials + 9 steps + Showcase
        self.videos_base_path = r'c:\Users\Ron Cristian Mendoza\Downloads\Steps Wallet-20260301T072112Z-1-001\Steps Wallet'
        self.video_files = []  # Will store paths to wallet videos
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
        
        self.your_turn_next_button = {
            'x': self.width - 160,
            'y': self.height - 80,
            'w': 140,
            'h': 50,
            'text': 'NEXT'
        }
        
        self.previous_button = {
            'x': 20,
            'y': self.height - 80,
            'w': 140,
            'h': 50,
            'text': 'PREVIOUS'
        }
        
        self.your_turn_previous_button = {
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
        
        # ==========================================
        # SEWING GUIDANCE SYSTEM
        # ==========================================
        
        # Sewing step state: 0=detect cloth, 1=positioning, 2=sewing
        self.sewing_step = 0
        self.cloth_detected = False
        self.cloth_bbox = None
        self.cloth_confidence = 0.0
        self.cloth_detect_frames = 0
        self.cloth_detect_threshold = 5  # Consecutive frames to confirm cloth
        self.cloth_locked = False
        self.cloth_lost_frames = 0
        # Keep cloth detection active for brief partial occlusions.
        self.cloth_lost_tolerance = 12
        
        # Needle position (fixed reference point - center of camera view)
        self.camera_w = 544
        self.camera_h = 400
        self.needle_x = self.camera_w // 2
        self.needle_y = self.camera_h // 2
        
        # ROI around needle for edge detection
        self.needle_roi_size = 120
        self.needle_roi_half = self.needle_roi_size // 2
        
        # Guide line parameters
        self.guide_line_visible = False
        self.guide_line_length = 200
        self.guide_line_color = (0, 255, 0)  # Green BGR
        # Guide orientation: 'auto', 'vertical', or 'horizontal'
        self.guide_orientation_mode = 'vertical'
        self.current_orientation = 'vertical'
        self.current_edge_angle = 90.0
        
        # Cloth edge detection state
        self.cloth_edge_detected = False
        self.cloth_edge_position = None
        self.cloth_moving_up = False
        self.top_line_at_needle = False
        self.edge_history = []
        self.max_edge_history = 12
        
        # Straight line tracking during sewing
        self.seam_points = []
        self.max_seam_points = 100
        self.max_deviation = 15  # Max pixel deviation before warning
        self.deviation_warning = False
        self.deviation_amount = 0.0
        self.seam_line_fit = None
        
        # Frame skipping for detection performance
        self.detect_frame_counter = 0
        self.detect_frame_skip = 2  # Process every 2nd frame
        
        # Sewing guidance step definitions
        self.sewing_guidance_steps = [
            {"title": "PLACE CLOTH", "instruction": "Place the cloth under the needle"},
            {"title": "POSITION EDGE", "instruction": "Align the cloth edge near the needle"},
            {"title": "SEW STRAIGHT", "instruction": "Follow the green guide line while sewing"},
        ]
        
        # Load cloth detection model (cloth.pt)
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        try:
            cloth_model_path = os.path.join(models_dir, 'cloth.pt')
            if os.path.exists(cloth_model_path):
                print(f"Loading cloth detection model: {cloth_model_path}")
                self.cloth_model = YOLO(cloth_model_path)
                print(f"✓ Cloth detection model loaded!")
                if hasattr(self.cloth_model, 'names'):
                    print(f"  Cloth model classes:")
                    for idx, name in self.cloth_model.names.items():
                        print(f"    Class {idx}: {name}")
            else:
                print(f"⚠ Cloth model not found: {cloth_model_path}")
                self.cloth_model = None
        except Exception as e:
            print(f"⚠ ERROR loading cloth model: {e}")
            import traceback
            traceback.print_exc()
            self.cloth_model = None
    
    def load_video_list(self):
        """Load the list of wallet tutorial video files (Materials + Step 1-9 + Showcase)"""
        self.video_files = []
        
        # Step 0: Wallet Materials video
        materials_patterns = [
            'Wallet Materials.mp4',
            'Wallet Materials.MP4',
            'Wallet Materials.mov',
            'Wallet Materials.MOV',
            'wallet materials.mp4',
            'wallet materials.MP4',
        ]
        materials_found = False
        for pattern in materials_patterns:
            video_path = os.path.join(self.videos_base_path, pattern)
            if os.path.exists(video_path):
                self.video_files.append(video_path)
                materials_found = True
                print(f"Found wallet materials video: {pattern}")
                break
        if not materials_found:
            self.video_files.append(None)
            print(f"Warning: Wallet Materials video not found in {self.videos_base_path}")
        
        # Steps 1-9: Wallet construction steps
        for i in range(1, 10):  # Steps 1 through 9
            video_found = False
            patterns = []
            
            # First step has special naming
            if i == 1:
                patterns = [
                    'Step 1 Wallet.mp4',
                    'Step 1 Wallet.MP4',
                    'Step 1 Wallet.mov',
                    'Step 1 Wallet.MOV',
                ]
            else:
                patterns = [
                    f'Step {i}.mp4',
                    f'Step {i}.MP4',
                    f'Step {i}.mov',
                    f'Step {i}.MOV',
                    f'step{i}.mp4',
                    f'step{i}.MP4',
                    f'step{i}.mov',
                    f'step{i}.MOV',
                    f'step {i}.mp4',
                    f'step {i}.MP4',
                    f'step {i}.mov',
                    f'step {i}.MOV',
                ]
            
            for pattern in patterns:
                video_path = os.path.join(self.videos_base_path, pattern)
                if os.path.exists(video_path):
                    self.video_files.append(video_path)
                    video_found = True
                    print(f"Found wallet tutorial video: {pattern}")
                    break
            
            if not video_found:
                self.video_files.append(None)
                print(f"Warning: wallet step{i} video not found in {self.videos_base_path}")
        
        # Step 10: Showcase video
        showcase_patterns = [
            'Showcase .mp4',  # Note the space in the filename
            'Showcase.mp4',
            'Showcase .MP4',
            'Showcase.MP4',
            'Showcase .mov',
            'Showcase.mov',
            'Showcase .MOV',
            'Showcase.MOV',
            'showcase.mp4',
            'showcase.MP4',
        ]
        showcase_found = False
        for pattern in showcase_patterns:
            video_path = os.path.join(self.videos_base_path, pattern)
            if os.path.exists(video_path):
                self.video_files.append(video_path)
                showcase_found = True
                print(f"Found wallet showcase video: {pattern}")
                break
        if not showcase_found:
            self.video_files.append(None)
            print(f"Warning: Showcase video not found in {self.videos_base_path}")
    
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
        self.your_turn_mode = False  # Reset your turn mode
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
        """Handle mouse clicks, returns action: 'next', 'done', 'replay_current', 'replay', 'continue', 'your_turn_next', 'previous', or None"""
        # Check your_turn_previous button when in your turn mode
        if self.your_turn_mode:
            btn = self.your_turn_previous_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                # Exit your turn mode and go back to current step's video
                self.your_turn_mode = False
                self.reset_sewing_guidance()
                self.replay_current_video()
                return 'previous_from_your_turn'
        
        # Check your_turn_next button when in your turn mode
        if self.your_turn_mode:
            btn = self.your_turn_next_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                # Exit your turn mode and go to next step
                self.your_turn_mode = False
                self.reset_sewing_guidance()
                if self.next_step():
                    return 'your_turn_next'
                else:
                    return 'continue'  # Done with all tutorials
        
        # Check back button (top left - always visible)
        if not self.skipped and not self.completed:
            btn = self.back_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                return 'back'
        
        # Check progress bar click for seeking (only when video is playing or paused)
        if not self.skipped and not self.completed:
            bar = self.progress_bar
            if bar['x'] <= x <= bar['x'] + bar['w'] and bar['y'] <= y <= bar['y'] + bar['h']:
                # Calculate the position clicked as a percentage
                click_percentage = (x - bar['x']) / bar['w']
                self.seek_to_position(click_percentage)
                return 'seek'
        
        # Check replay current video button (only when video is playing or paused)
        if not self.skipped and not self.completed and not self.your_turn_mode:
            btn = self.replay_current_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                # Replay current video from beginning
                self.replay_current_video()
                return 'replay_current'
        
        # Check previous button (only when video is playing/paused and on steps 1-9)
        if not self.skipped and not self.completed and not self.your_turn_mode:
            if self.current_step >= 1 and self.current_step <= 9:
                btn = self.previous_button
                if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                    self.play_button_click_sound()
                    # Go to previous step
                    self.current_step -= 1
                    self.load_current_video()
                    return 'previous'
        
        # Check next/done button (only when video is playing or paused)
        if not self.skipped and not self.completed and not self.your_turn_mode:
            btn = self.next_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                # If on last step, this is the Done button
                if self.current_step >= self.total_steps - 1:
                    return 'continue'  # Done with all tutorials
                # Check if current step needs \"your turn\" practice (steps 1-9, not materials or showcase)
                elif self.current_step >= 1 and self.current_step <= 9:
                    # Transition to your_turn mode
                    self.your_turn_mode = True
                    self.reset_sewing_guidance()
                    return 'enter_your_turn'
                else:
                    # Move to next step directly (for materials and showcase)
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
        
        # Don't update video if in your_turn_mode
        if self.your_turn_mode:
            return
        
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
        font = cv2.FONT_HERSHEY_TRIPLEX
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
            font = cv2.FONT_HERSHEY_TRIPLEX
            
            # Title
            title = "WALLET TUTORIAL VIDEO"
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
        
        # Store progress bar position for click detection
        self.progress_bar = {'x': bar_x, 'y': bar_y, 'w': bar_width, 'h': bar_height}
    
    def draw(self, img, camera_frame=None):
        """Main draw function"""
        self.update()
        
        # Draw \"your turn\" practice screen if in that mode
        if self.your_turn_mode:
            self.draw_your_turn(img, camera_frame)
            return
        
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
            
            # Draw replay current button (left side)
            self.draw_button(img, self.replay_current_button, COLORS['button_normal'])
            
            # Draw previous button (bottom left) - only show on steps 1-9
            if self.current_step >= 1 and self.current_step <= 9:
                self.draw_button(img, self.previous_button, COLORS['button_hover'])
            
            # Draw next/done button (right side)
            # Change text to "DONE" on last step
            if self.current_step >= self.total_steps - 1:
                self.next_button['text'] = 'DONE'
            else:
                self.next_button['text'] = 'NEXT'
            self.draw_button(img, self.next_button, COLORS['button_hover'])
            
        else:
            # Completed or skipped
            font = cv2.FONT_HERSHEY_TRIPLEX
            title = "WALLET TUTORIAL COMPLETED" if self.completed else "WALLET TUTORIAL"
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
    
    def draw_step_indicator(self, img):
        """Draw step indicator showing current progress (Materials, Step 1-9, Showcase)"""
        font = cv2.FONT_HERSHEY_TRIPLEX
        
        # Determine the text based on current step
        if self.current_step == 0:
            text = "Wallet Materials"
        elif self.current_step == self.total_steps - 1:
            text = "Wallet Showcase"
        else:
            text = f"Wallet Step {self.current_step} of {self.total_steps - 2}"  # -2 for materials and showcase
        
        font_scale = 0.8
        thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # Keep centered horizontally at the top
        text_x = (self.width - text_w) // 2
        text_y = 20 + (50 + text_h) // 2  # Same Y position as back button
        
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
        cv2.putText(img, text, (text_x, text_y), font, font_scale, 
                   COLORS['text_accent'], thickness)
    
    def draw_your_turn(self, img, camera_frame):
        """Draw the 'Your Turn' practice screen with webcam feed and sewing guidance"""
        font = cv2.FONT_HERSHEY_TRIPLEX
        
        # Title at top
        title = f"Your Turn - Practice Step {self.current_step}"
        font_scale = 1.2
        thickness = 3
        
        (text_w, text_h), baseline = cv2.getTextSize(title, font, font_scale, thickness)
        text_x = (self.width - text_w) // 2
        text_y = 50
        
        # Background for title
        padding = 15
        cv2.rectangle(img, 
                     (text_x - padding, text_y - text_h - padding),
                     (text_x + text_w + padding, text_y + padding),
                     (40, 40, 40), -1)
        
        # Border with glow
        glow_intensity = 0.5 + 0.5 * abs(math.sin(self.glow_phase))
        border_color = tuple(int(c * glow_intensity) for c in COLORS['cyan'])
        cv2.rectangle(img, 
                     (text_x - padding, text_y - text_h - padding),
                     (text_x + text_w + padding, text_y + padding),
                     border_color, 3)
        
        # Title text
        cv2.putText(img, title, (text_x, text_y), font, font_scale, 
                   COLORS['text_accent'], thickness)
        
        # Camera feed area
        camera_w = self.camera_w
        camera_h = self.camera_h
        camera_x = (self.width - camera_w) // 2
        camera_y = text_y + text_h + 25
        
        # Draw camera frame border with glow effect
        border_color_intensity = 0.4 + 0.3 * abs(math.sin(self.glow_phase * 0.7))
        border_color_glow = tuple(int(c * border_color_intensity) for c in COLORS['bright_blue'])
        cv2.rectangle(img, (camera_x - 5, camera_y - 5), 
                     (camera_x + camera_w + 5, camera_y + camera_h + 5), 
                     border_color_glow, 3)
        
        # Display camera feed with sewing guidance
        if camera_frame is not None:
            cam_frame = cv2.resize(camera_frame, (camera_w, camera_h))
            
            # --- Run sewing guidance on the camera frame ---
            self.detect_frame_counter += 1
            process_this_frame = (self.detect_frame_counter % self.detect_frame_skip == 0)
            self.process_sewing_guidance(cam_frame, process_this_frame)
            
            # Copy processed frame to main image
            img[camera_y:camera_y + camera_h, camera_x:camera_x + camera_w] = cam_frame
        else:
            # Dark background for camera
            cv2.rectangle(img, (camera_x, camera_y), (camera_x + camera_w, camera_y + camera_h), 
                         COLORS['dark_blue'], -1)
            msg = "Opening camera..."
            font_scale_msg = 0.8
            thickness_msg = 2
            (msg_w, msg_h), _ = cv2.getTextSize(msg, font, font_scale_msg, thickness_msg)
            msg_x = camera_x + (camera_w - msg_w) // 2
            msg_y = camera_y + (camera_h + msg_h) // 2
            cv2.putText(img, msg, (msg_x, msg_y), font, font_scale_msg, 
                       COLORS['text_primary'], thickness_msg)
        
        # Draw sewing step instruction bar below camera
        self.draw_sewing_step_bar(img, camera_x, camera_y, camera_w, camera_h)
        
        # Draw deviation warning above camera if needed
        if self.deviation_warning:
            self.draw_deviation_feedback(img, camera_x, camera_y, camera_w)
        
        # Draw back button (top left)
        self.draw_button(img, self.back_button, COLORS['button_hover'])
        
        # Draw PREVIOUS button to go back to video (bottom left)
        self.draw_button(img, self.your_turn_previous_button, COLORS['button_hover'])
        
        # Draw NEXT button to continue to next step (bottom right)
        self.draw_button(img, self.your_turn_next_button, COLORS['button_hover'])
    
    # ==========================================
    # SEWING GUIDANCE METHODS
    # ==========================================
    
    def detect_cloth_in_frame(self, cam_frame):
        """Detect cloth using cloth.pt model"""
        if self.cloth_model is None:
            return False, None, 0.0
        try:
            results = self.cloth_model(
                cam_frame, conf=0.3, imgsz=320, verbose=False
            )
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confidences))
                best_conf = float(confidences[best_idx])
                best_box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
                return True, best_box, best_conf
            return False, None, 0.0
        except Exception as e:
            print(f"Cloth detection error: {e}")
            return False, None, 0.0
    
    def get_needle_roi(self, cam_frame):
        """Extract ROI region around the fixed needle position"""
        h, w = cam_frame.shape[:2]
        x1 = max(0, self.needle_x - self.needle_roi_half)
        y1 = max(0, self.needle_y - self.needle_roi_half)
        x2 = min(w, self.needle_x + self.needle_roi_half)
        y2 = min(h, self.needle_y + self.needle_roi_half)
        roi = cam_frame[y1:y2, x1:x2].copy()
        return roi, (x1, y1, x2, y2)
    
    def detect_cloth_edge_in_roi(self, roi):
        """Detect cloth edge and orientation in needle ROI."""
        if roi is None or roi.size == 0:
            return False, None, 'vertical', 90.0, None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False, None, 'vertical', 90.0, None
        longest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(longest_contour) < 50:
            return False, None, 'vertical', 90.0, None

        # Estimate edge angle with fitLine to classify vertical/horizontal.
        orientation = 'vertical'
        angle_deg = 90.0
        if len(longest_contour) >= 2:
            vx, vy, _, _ = cv2.fitLine(longest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
            angle_deg = abs(math.degrees(math.atan2(float(vy), float(vx))))
            if angle_deg > 90:
                angle_deg = 180 - angle_deg
            orientation = 'vertical' if angle_deg >= 45 else 'horizontal'

        # Top-most point of the detected line/edge within ROI.
        top_idx = np.argmin(longest_contour[:, 0, 1])
        top_pt = tuple(longest_contour[top_idx, 0])

        M = cv2.moments(longest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return True, (cx, cy), orientation, angle_deg, top_pt
        return False, None, orientation, angle_deg, top_pt

    def update_motion_and_top_alignment(self, abs_edge_x, abs_edge_y, abs_top_pt):
        """Detect upward cloth motion and whether line top is at needle center."""
        self.edge_history.append((abs_edge_x, abs_edge_y))
        if len(self.edge_history) > self.max_edge_history:
            self.edge_history = self.edge_history[-self.max_edge_history:]

        # Moving up means y decreases over time in image coordinates.
        self.cloth_moving_up = False
        if len(self.edge_history) >= 5:
            recent = self.edge_history[-6:]
            deltas = [recent[i][1] - recent[i - 1][1] for i in range(1, len(recent))]
            avg_delta_y = float(np.mean(deltas))
            self.cloth_moving_up = avg_delta_y < -0.8

        self.top_line_at_needle = False
        if abs_top_pt is not None:
            top_dx = abs(abs_top_pt[0] - self.needle_x)
            top_dy = abs(abs_top_pt[1] - self.needle_y)
            self.top_line_at_needle = top_dx <= 25 and top_dy <= 25
    
    def check_cloth_positioning(self, edge_pos, roi_bounds, orientation):
        """Check if cloth edge is properly positioned near the needle by orientation."""
        if edge_pos is None:
            return False
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1
        center_x = roi_w // 2
        center_y = roi_h // 2
        dx = abs(edge_pos[0] - center_x)
        dy = abs(edge_pos[1] - center_y)

        # If guide is fixed vertical, accept cloth in any orientation as long as
        # the edge is near the needle center zone.
        if self.guide_orientation_mode == 'vertical':
            return dx < 40 and dy < 40

        # For other modes, keep orientation-aware checks.
        if orientation == 'vertical':
            return dx < 25 and dy < 55
        return dy < 25 and dx < 55
    
    def detect_seam_straightness(self):
        """Detect if the seam line is straight using tracked points"""
        if len(self.seam_points) < 5:
            return True, 0.0
        points = np.array(self.seam_points[-20:], dtype=np.float32)
        if len(points) < 5:
            return True, 0.0
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        line_dir = np.array([float(vx), float(vy)])
        line_point = np.array([float(x0), float(y0)])
        deviations = []
        for pt in points:
            v = pt - line_point
            proj = np.dot(v, line_dir)
            closest = line_point + proj * line_dir
            dist = np.linalg.norm(pt - closest)
            deviations.append(dist)
        max_dev = float(np.max(deviations))
        avg_dev = float(np.mean(deviations))
        self.seam_line_fit = (vx, vy, x0, y0)
        return max_dev < self.max_deviation, avg_dev
    
    def process_sewing_guidance(self, cam_frame, process_this_frame):
        """Main sewing guidance: cloth detection, edge analysis, guide line, straightness"""
        # Always draw needle marker
        self.draw_needle_marker(cam_frame)
        
        # --- CLOTH DETECTION ---
        if process_this_frame and self.cloth_model is not None:
            detected, bbox, confidence = self.detect_cloth_in_frame(cam_frame)
            if detected:
                self.cloth_detect_frames = min(
                    self.cloth_detect_frames + 1, self.cloth_detect_threshold + 5
                )
                self.cloth_bbox = bbox
                self.cloth_confidence = confidence
                self.cloth_lost_frames = 0

                # Lock cloth state after stable confirmation.
                if self.cloth_detect_frames >= self.cloth_detect_threshold:
                    self.cloth_locked = True
            else:
                self.cloth_lost_frames += 1
                self.cloth_detect_frames = max(0, self.cloth_detect_frames - 1)

                # Unlock only after prolonged loss, not immediate miss.
                if self.cloth_lost_frames > self.cloth_lost_tolerance:
                    self.cloth_locked = False
                    self.cloth_bbox = None
                    self.cloth_confidence = 0.0
        
        self.cloth_detected = self.cloth_locked
        self.draw_cloth_status(cam_frame)
        
        # --- STATE TRANSITIONS ---
        if not self.cloth_detected:
            self.sewing_step = 0
            self.cloth_edge_detected = False
            self.guide_line_visible = False
            self.deviation_warning = False
            return
        
        # Cloth detected -> at least step 1 (positioning)
        if self.sewing_step == 0:
            self.sewing_step = 1
        
        # --- EDGE DETECTION IN NEEDLE ROI ---
        roi, roi_bounds = self.get_needle_roi(cam_frame)
        self.draw_needle_roi(cam_frame, roi_bounds)
        
        if process_this_frame:
            edge_found, edge_pos, edge_orientation, edge_angle, top_pt = self.detect_cloth_edge_in_roi(roi)
            self.cloth_edge_detected = edge_found
            self.cloth_edge_position = edge_pos
            self.current_edge_angle = edge_angle
            
            if edge_found:
                # Draw edge point on camera frame
                abs_x = roi_bounds[0] + edge_pos[0]
                abs_y = roi_bounds[1] + edge_pos[1]
                cv2.circle(cam_frame, (abs_x, abs_y), 5, (255, 0, 255), -1)

                abs_top_pt = None
                if top_pt is not None:
                    abs_top_pt = (roi_bounds[0] + int(top_pt[0]), roi_bounds[1] + int(top_pt[1]))
                    cv2.circle(cam_frame, abs_top_pt, 4, (0, 255, 255), -1)

                self.update_motion_and_top_alignment(abs_x, abs_y, abs_top_pt)
                
                if self.guide_orientation_mode == 'auto':
                    self.current_orientation = edge_orientation
                else:
                    self.current_orientation = self.guide_orientation_mode

                if self.check_cloth_positioning(edge_pos, roi_bounds, self.current_orientation):
                    self.guide_line_visible = True
                    if self.sewing_step == 1:
                        self.sewing_step = 2
            else:
                if self.sewing_step == 1:
                    self.guide_line_visible = False
                self.cloth_moving_up = False
                self.top_line_at_needle = False
            
            # Track edge positions for straightness during sewing
            if self.sewing_step == 2 and edge_found:
                abs_x = roi_bounds[0] + edge_pos[0]
                abs_y = roi_bounds[1] + edge_pos[1]
                self.seam_points.append((abs_x, abs_y))
                if len(self.seam_points) > self.max_seam_points:
                    self.seam_points = self.seam_points[-self.max_seam_points:]
        
        # --- GUIDE LINE ---
        # Guide line rendering is temporarily disabled.
        
        # --- SEAM STRAIGHTNESS CHECK ---
        if self.sewing_step == 2 and len(self.seam_points) >= 5:
            is_straight, deviation = self.detect_seam_straightness()
            self.deviation_warning = not is_straight
            self.deviation_amount = deviation
    
    def draw_needle_marker(self, cam_frame):
        """Draw crosshair at the fixed needle position"""
        size = 15
        color = (0, 0, 255)  # Red
        cv2.line(cam_frame, (self.needle_x, self.needle_y - size),
                 (self.needle_x, self.needle_y + size), color, 2)
        cv2.line(cam_frame, (self.needle_x - size, self.needle_y),
                 (self.needle_x + size, self.needle_y), color, 2)
        cv2.circle(cam_frame, (self.needle_x, self.needle_y), 8, color, 1)
    
    def draw_needle_roi(self, cam_frame, roi_bounds):
        """Draw ROI box around needle area"""
        x1, y1, x2, y2 = roi_bounds
        cv2.rectangle(cam_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(cam_frame, "NEEDLE ROI", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    
    def draw_cloth_status(self, cam_frame):
        """Draw cloth detection status on camera frame"""
        if self.cloth_detected:
            if self.cloth_lost_frames > 0:
                status = f"CLOTH TRACKING ({self.cloth_confidence:.0%})"
            else:
                status = f"CLOTH DETECTED ({self.cloth_confidence:.0%})"
            color = (0, 255, 0)
            if self.cloth_bbox is not None:
                x1, y1, x2, y2 = self.cloth_bbox
                cv2.rectangle(cam_frame, (x1, y1), (x2, y2), color, 2)
        else:
            status = "WAITING FOR CLOTH..."
            color = (0, 165, 255)  # Orange
        cv2.putText(cam_frame, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if self.cloth_edge_detected:
            orientation_text = f"EDGE: {self.current_orientation.upper()} ({self.current_edge_angle:.1f} deg)"
            cv2.putText(cam_frame, orientation_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.guide_line_color, 2)

            motion_color = (0, 255, 0) if self.cloth_moving_up else (0, 165, 255)
            top_color = (0, 255, 0) if self.top_line_at_needle else (0, 165, 255)
            cv2.putText(cam_frame, f"MOVE UP: {'YES' if self.cloth_moving_up else 'NO'}", (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 2)
            cv2.putText(cam_frame, f"TOP AT NEEDLE: {'YES' if self.top_line_at_needle else 'NO'}", (10, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, top_color, 2)
    
    def draw_guide_line(self, cam_frame):
        """Draw dashed guide line from needle in current orientation."""
        sx, sy = self.needle_x, self.needle_y
        dash_len = 10
        gap_len = 5

        if self.current_orientation == 'vertical':
            # Dashed line down
            end_y = min(sy + self.guide_line_length, cam_frame.shape[0] - 1)
            y = sy
            while y < end_y:
                y_end = min(y + dash_len, end_y)
                cv2.line(cam_frame, (sx, y), (sx, y_end), self.guide_line_color, 2)
                y = y_end + gap_len

            # Dashed line up
            top_y = max(sy - self.guide_line_length, 0)
            y = sy
            while y > top_y:
                y_end = max(y - dash_len, top_y)
                cv2.line(cam_frame, (sx, y), (sx, y_end), self.guide_line_color, 2)
                y = y_end - gap_len
        else:
            # Dashed line right
            end_x = min(sx + self.guide_line_length, cam_frame.shape[1] - 1)
            x = sx
            while x < end_x:
                x_end = min(x + dash_len, end_x)
                cv2.line(cam_frame, (x, sy), (x_end, sy), self.guide_line_color, 2)
                x = x_end + gap_len

            # Dashed line left
            left_end = max(sx - self.guide_line_length, 0)
            x = sx
            while x > left_end:
                x_end = max(x - dash_len, left_end)
                cv2.line(cam_frame, (x, sy), (x_end, sy), self.guide_line_color, 2)
                x = x_end - gap_len

        cv2.putText(cam_frame, f"GUIDE: {self.current_orientation.upper()}", (sx + 10, sy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.guide_line_color, 1)
    
    def draw_sewing_step_bar(self, img, camera_x, camera_y, camera_w, camera_h):
        """Draw current sewing guidance step instruction below camera"""
        if self.sewing_step >= len(self.sewing_guidance_steps):
            return
        step = self.sewing_guidance_steps[self.sewing_step]
        bar_y = camera_y + camera_h + 8
        bar_h = 42
        # Background bar
        overlay = img.copy()
        cv2.rectangle(overlay,
                      (camera_x, bar_y),
                      (camera_x + camera_w, bar_y + bar_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        # Border
        glow = 0.5 + 0.5 * abs(math.sin(self.glow_phase * 1.2))
        border_col = tuple(int(c * glow) for c in COLORS['cyan'])
        cv2.rectangle(img, (camera_x, bar_y), (camera_x + camera_w, bar_y + bar_h), border_col, 2)
        # Step title
        step_text = f"STEP {self.sewing_step + 1}: {step['title']}"
        cv2.putText(img, step_text, (camera_x + 10, bar_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['text_accent'], 2)
        # Instruction
        orientation_note = f" [{self.current_orientation.upper()}]" if self.sewing_step >= 1 else ""
        cv2.putText(img, step['instruction'] + orientation_note, (camera_x + 10, bar_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text_secondary'], 1)
    
    def draw_deviation_feedback(self, img, camera_x, camera_y, camera_w):
        """Draw seam deviation warning overlay above camera"""
        pulse = 0.5 + 0.5 * abs(math.sin(self.glow_phase * 3))
        bar_y = camera_y + 5
        bar_h = 35
        overlay = img.copy()
        cv2.rectangle(overlay,
                      (camera_x + 5, bar_y),
                      (camera_x + camera_w - 5, bar_y + bar_h),
                      (0, 0, int(200 * pulse)), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        warning = f"SEAM DEVIATION: {self.deviation_amount:.1f}px - Straighten your line!"
        (tw, _), _ = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx = camera_x + (camera_w - tw) // 2
        cv2.putText(img, warning, (tx, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    
    def reset_sewing_guidance(self):
        """Reset all sewing guidance state for a new your-turn session"""
        self.sewing_step = 0
        self.cloth_detected = False
        self.cloth_bbox = None
        self.cloth_confidence = 0.0
        self.cloth_detect_frames = 0
        self.cloth_locked = False
        self.cloth_lost_frames = 0
        self.cloth_edge_detected = False
        self.cloth_edge_position = None
        self.guide_line_visible = False
        self.deviation_warning = False
        self.deviation_amount = 0.0
        self.seam_points = []
        self.seam_line_fit = None
        self.detect_frame_counter = 0
        self.current_orientation = 'vertical'
        self.current_edge_angle = 90.0
        self.cloth_moving_up = False
        self.top_line_at_needle = False
        self.edge_history = []
    
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
