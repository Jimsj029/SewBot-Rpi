

import cv2
import numpy as np
import math
import os
import sys
import time

try:
    from .typography import (
        FONT_MAIN,
        FONT_COMPACT,
        text_scale,
        text_thickness,
        get_text_size,
        fit_text_scale,
        draw_text,
    )
except ImportError:
    from typography import (
        FONT_MAIN,
        FONT_COMPACT,
        text_scale,
        text_thickness,
        get_text_size,
        fit_text_scale,
        draw_text,
    )

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from music_manager import get_music_manager

# Direct ONNX Runtime for needle centring detection (no ultralytics overhead)
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("onnxruntime not available – needle centring detection disabled")

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

# Use a lighter face for wallet tutorial text for better readability.
UI_FONT = FONT_COMPACT


def _put_text(img, text, x, y, scale, color, thickness):
    """Draw text with a 1-px black outline so it reads against any background."""
    draw_text(img, text, x, y, scale, color, thickness, font=UI_FONT, outline_extra=1)

# Per-step instructions shown in "Your Turn" practice screen
STEP_INSTRUCTIONS = {
    1:  ["Fold a 12x24 cm cloth in half,",
         "then stitch the open side closed."],
    2:  ["Turn the stitched cloth inside out so that",
         "the seam is positioned on the inside."],
    3:  ["Reinforce the seam by stitching again along the same line."],
    4:  ["Prepare another 12x24 cm piece of cloth, fold it in half,",
         "and stitch along the folded edge."],
    5:  ["Place an 11x24 cm cloth with the stitched cloth from Steps 1-2 in the center.",
         "Use pin needles (if available) to hold the layers in place."],
    6:  ["Position the folded cloth from Step 4 underneath the assembled layers.",
         "Sew around the edges in a U-shaped stitch to secure the pieces together."],
    7:  ["Create a triple stitch in the center, approx. 0.5-1 cm apart,",
         "to strengthen the middle section."],
    8:  ["Place the 11x24 cm inner cover on top of the structure",
         "and stitch along the upper edge."],
    9:  ["Turn the cloth over and stitch again along the same seam to reinforce it."],
    10: ["Place a 12x24 cm (base cover) at the back",
         "and secure it with a U-shaped stitch."],
    11: ["Position the final 12x24 cm (back lining) at the front",
         "and stitch it using the same U-shaped pattern."],
    12: ["Turn the wallet inside out from the top opening."],
    13: ["Finally, stitch along the top opening to close any remaining gap",
         "and complete the wallet."],
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
        self.total_steps = 15  # Wallet: Materials + 13 steps + Showcase
        self.videos_base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'videos')
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

        # ── "Your Turn" ROI guide overlay – change any value to reposition/resize ──
        self.ROI_CENTER_X   = 272   # Horizontal centre of the column
                                    #   (pixels from LEFT edge of the camera feed)
        self.ROI_COL_WIDTH  = 140   # Total width of the single column in pixels
        self.ROI_TOP_Y      = 0     # Y start offset from TOP of the camera feed
        self.ROI_BOT_MARGIN = 0   # Y margin from BOTTOM of the camera feed
        self.ROI_COL_COLOR  = (0, 255, 255)    # BGR colour of the column outline
        self.ROI_LINE_COLOR = (220, 220, 220)  # BGR colour of the dashed stitch line
        self.ROI_DASH_LEN   = 14    # Pixel length of each dash segment
        self.ROI_DASH_GAP   = 8     # Pixel gap between dash segments
        self.ROI_STEP7_LINE_OFFSET = 28  # Pixel distance (~1 cm) of the two extra
                                         #   stitch lines shown only on step 7
        # ────────────────────────────────────────────────────────────────────────

        # ── Needle centring detection ─────────────────────────────────────────
        self.needle_confirmed        = False  # True when needle is currently centred
        self.NEEDLE_CONF_THRESHOLD   = 0.35   # YOLO confidence threshold
        self.NEEDLE_CENTER_TOLERANCE = 40     # px tolerance from ROI column midpoint
        self.NEEDLE_CHECK_INTERVAL   = 6      # Run model every N draw-frames (~5 Hz @ 30 fps)
        self._needle_check_counter   = 0      # Frame counter for throttling
        self.needle_model = None
        self.needle_input_name = None
        self._needle_imgsz = 320
        if ORT_AVAILABLE:
            try:
                _needle_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), 'models', 'needle.onnx')
                if os.path.exists(_needle_path):
                    _sess_opts = ort.SessionOptions()
                    _sess_opts.inter_op_num_threads = 2
                    _sess_opts.intra_op_num_threads = 2
                    self.needle_model = ort.InferenceSession(
                        _needle_path,
                        sess_options=_sess_opts,
                        providers=['CPUExecutionProvider'])
                    self.needle_input_name = self.needle_model.get_inputs()[0].name
                    # Warm-up run so first real frame isn't slow
                    _w = np.zeros((1, 3, self._needle_imgsz, self._needle_imgsz), dtype=np.float32)
                    self.needle_model.run(None, {self.needle_input_name: _w})
                    print(f"\u2713 Needle centring model loaded (ONNX Runtime direct): {_needle_path}")
                else:
                    print(f"\u26a0 needle.onnx not found at {_needle_path}")
            except Exception as _e:
                print(f"\u26a0 Needle model load error: {_e}")
                self.needle_model = None
        # ─────────────────────────────────────────────────────────────────────

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
    
    def load_video_list(self):
        """Load the list of wallet tutorial video files (Materials + Step 1-13 + Showcase)"""
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

        # Steps 1-13: Wallet construction steps
        for i in range(1, 14):  # Steps 1 through 13
            video_found = False

            # First step has special naming (also try plain "Step 1.mp4")
            if i == 1:
                patterns = [
                    'Step 1.mp4',
                    'Step 1.MP4',
                    'step 1.mp4',
                    'step 1.MP4',
                    'step1.mp4',
                    'step1.MP4',
                    'Step 1 Wallet.mp4',
                    'Step 1 Wallet.MP4',
                    'Step 1 Wallet.mov',
                    'Step 1 Wallet.MOV',
                    'Step 1.mov',
                    'Step 1.MOV',
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
        self.needle_confirmed = False
        self._needle_check_counter = 0
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
                self.replay_current_video()
                return 'previous_from_your_turn'
        
        # Check your_turn_next button when in your turn mode
        if self.your_turn_mode:
            btn = self.your_turn_next_button
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                # Exit your turn mode and go to next step
                self.your_turn_mode = False
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
        
        # Check previous button (only when video is playing/paused and on steps 1-13 + showcase)
        if not self.skipped and not self.completed and not self.your_turn_mode:
            if self.current_step >= 1 and self.current_step <= self.total_steps - 1:
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
                # Check if current step needs "your turn" practice (steps 1-13, not materials or showcase)
                # Steps 2 and 12 skip "your turn" and go directly to next video
                elif self.current_step >= 1 and self.current_step <= 13 and self.current_step not in (2, 12):
                    # Transition to your_turn mode
                    self.needle_confirmed = False
                    self._needle_check_counter = 0
                    self.your_turn_mode = True
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
        font_scale = text_scale(0.82, self.width, self.height, floor=0.74, ceiling=0.95)
        thickness = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
        text = btn['text']

        font_scale = fit_text_scale(text, UI_FONT, w - 14, font_scale, thickness, min_scale=0.65)

        (text_w, text_h), baseline = get_text_size(text, UI_FONT, font_scale, thickness)
        text_x = x + (w - text_w) // 2
        text_y = y + (h + text_h) // 2

        _put_text(img, text, text_x, text_y, font_scale, COLORS['text_primary'], thickness)
    
    def draw_video_frame(self, img):
        """Draw actual video frame or placeholder"""
        # Video area - lowered to leave room for step instructions above
        video_margin = 50
        video_y = 145  # Extra space above video for step instruction text
        # Keep the frame narrower so it visually matches the centered
        # instruction text block above it.
        video_w = max(480, min(self.width - 2 * video_margin, int(self.width * 0.58)))
        video_x = (self.width - video_w) // 2
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

        # Step instruction text above the video (steps 1-13 only)
        self._draw_step_overlay(img, video_x, video_y, video_w, video_h)

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
            title = "WALLET TUTORIAL VIDEO"
            font_scale = text_scale(1.15, self.width, self.height, floor=1.0, ceiling=1.32)
            thickness = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
            font_scale = fit_text_scale(title, UI_FONT, video_w - 50, font_scale, thickness, min_scale=0.88)
            (text_w, text_h), _ = get_text_size(title, UI_FONT, font_scale, thickness)
            text_x = center_x - text_w // 2
            text_y = center_y - 60
            _put_text(img, title, text_x, text_y, font_scale, COLORS['text_accent'], thickness)

            # Subtitle
            subtitle = "(No Video File Found)"
            font_scale = text_scale(0.78, self.width, self.height, floor=0.7, ceiling=0.9)
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
            
            # Draw previous button (bottom left) - show on steps 1-13 and showcase
            if self.current_step >= 1 and self.current_step <= self.total_steps - 1:
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
            title = "WALLET TUTORIAL COMPLETED" if self.completed else "WALLET TUTORIAL"
            font_scale = text_scale(1.42, self.width, self.height, floor=1.2, ceiling=1.65)
            thickness = text_thickness(3, self.width, self.height, min_thickness=2, max_thickness=4)
            font_scale = fit_text_scale(title, UI_FONT, self.width - 80, font_scale, thickness, min_scale=0.95)

            (text_w, text_h), _ = get_text_size(title, UI_FONT, font_scale, thickness)
            text_x = (self.width - text_w) // 2
            text_y = self.height // 3

            _put_text(img, title, text_x, text_y, font_scale, COLORS['text_accent'], thickness)
            
            # Draw replay and continue buttons
            self.draw_button(img, self.replay_button, COLORS['button_normal'])
            self.draw_button(img, self.continue_button, COLORS['button_hover'])
    
    def draw_step_indicator(self, img):
        """Draw step indicator showing current progress (Materials, Step 1-13, Showcase)"""
        # Determine the text based on current step
        if self.current_step == 0:
            text = "Wallet Materials"
        elif self.current_step == self.total_steps - 1:
            text = "Wallet Showcase"
        else:
            text = f"Wallet Step {self.current_step} of {self.total_steps - 2}"

        font_scale = text_scale(0.9, self.width, self.height, floor=0.78, ceiling=1.04)
        thickness = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
        font_scale = fit_text_scale(text, UI_FONT, self.width - 220, font_scale, thickness, min_scale=0.7)

        (text_w, text_h), baseline = get_text_size(text, UI_FONT, font_scale, thickness)
        text_x = (self.width - text_w) // 2
        text_y = 20 + (50 + text_h) // 2

        padding = 10
        cv2.rectangle(img,
                     (text_x - padding, text_y - text_h - padding),
                     (text_x + text_w + padding, text_y + padding),
                     (20, 20, 20), -1)
        cv2.rectangle(img,
                     (text_x - padding, text_y - text_h - padding),
                     (text_x + text_w + padding, text_y + padding),
                     COLORS['cyan'], 2)
        _put_text(img, text, text_x, text_y, font_scale, COLORS['text_accent'], thickness)
    
    def _draw_step_overlay(self, img, video_x, video_y, video_w, video_h):
        """Draw step instruction lines centered above the video frame for steps 1-13."""
        if self.current_step < 1 or self.current_step > 13:
            return

        lines = STEP_INSTRUCTIONS.get(self.current_step, [])
        if not lines:
            return

        overlay_font = FONT_COMPACT
        inst_scale = text_scale(0.7, self.width, self.height, floor=0.64, ceiling=0.82)
        inst_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
        line_spacing = 26
        # Keep instruction width narrower than the full video width so the
        # text block matches the centered visual focus area.
        instruction_w = max(420, min(video_w - 140, int(video_w * 0.64)))

        (_, inst_h), _ = get_text_size("Ag", overlay_font, inst_scale, inst_thick)

        # Total block height for all lines, centred in the gap above the video
        block_h = len(lines) * line_spacing - (line_spacing - inst_h)
        gap = video_y - 60  # space between header indicator and video top
        start_y = 60 + (gap - block_h) // 2 + inst_h

        cur_y = start_y
        center_x = video_x + video_w // 2

        for line in lines:
            fitted_scale = fit_text_scale(line, overlay_font, instruction_w, inst_scale, inst_thick, min_scale=0.58)
            (lw, _), _ = get_text_size(line, overlay_font, fitted_scale, inst_thick)
            lx = center_x - lw // 2
            draw_text(
                img,
                line,
                lx,
                cur_y,
                fitted_scale,
                COLORS['text_primary'],
                max(1, inst_thick),
                font=overlay_font,
                outline_color=(0, 0, 0),
                outline_extra=1,
            )
            cur_y += line_spacing

    def draw_your_turn(self, img, camera_frame):
        """Draw the 'Your Turn' practice screen with webcam feed"""
        # ── Title ────────────────────────────────────────────────────────────
        title = f"Your Turn - Practice Step {self.current_step}"
        title_scale = text_scale(1.0, self.width, self.height, floor=0.86, ceiling=1.12)
        title_thick = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
        title_scale = fit_text_scale(title, UI_FONT, self.width - 120, title_scale, title_thick, min_scale=0.78)

        (text_w, text_h), _ = get_text_size(title, UI_FONT, title_scale, title_thick)
        text_x = (self.width - text_w) // 2
        text_y = 38

        padding = 10
        cv2.rectangle(img,
                     (text_x - padding, text_y - text_h - padding),
                     (text_x + text_w + padding, text_y + padding),
                     (20, 20, 20), -1)
        glow_intensity = 0.5 + 0.5 * abs(math.sin(self.glow_phase))
        border_color = tuple(int(c * glow_intensity) for c in COLORS['cyan'])
        cv2.rectangle(img,
                     (text_x - padding, text_y - text_h - padding),
                     (text_x + text_w + padding, text_y + padding),
                     border_color, 2)
        _put_text(img, title, text_x, text_y, title_scale, COLORS['text_accent'], title_thick)

        # ── Step instructions ─────────────────────────────────────────────
        inst_scale = text_scale(0.7, self.width, self.height, floor=0.62, ceiling=0.82)
        inst_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
        lines = STEP_INSTRUCTIONS.get(self.current_step, ["Practice what you learned in the video."])
        instruction_w = max(360, min(self.width - 220, int(self.width * 0.58)))
        line_spacing = 22
        cur_y = text_y + text_h + 26
        for line in lines:
            fitted_scale = fit_text_scale(line, FONT_COMPACT, instruction_w, inst_scale, inst_thick, min_scale=0.58)
            (lw, lh), _ = get_text_size(line, FONT_COMPACT, fitted_scale, inst_thick)
            lx = (self.width - lw) // 2
            draw_text(
                img,
                line,
                lx,
                cur_y,
                fitted_scale,
                COLORS['text_primary'],
                max(1, inst_thick),
                font=FONT_COMPACT,
                outline_color=(0, 0, 0),
                outline_extra=1,
            )
            cur_y += line_spacing
        last_inst_y = cur_y

        # Camera feed area
        camera_w = 544
        camera_h = 400
        camera_x = (self.width - camera_w) // 2
        camera_y = last_inst_y + 10
        
        # Draw camera frame border with glow effect (similar to pattern mode)
        border_color_intensity = 0.4 + 0.3 * abs(math.sin(self.glow_phase * 0.7))
        border_color_glow = tuple(int(c * border_color_intensity) for c in COLORS['bright_blue'])
        
        # Draw outer glow
        cv2.rectangle(img, (camera_x - 5, camera_y - 5), 
                     (camera_x + camera_w + 5, camera_y + camera_h + 5), 
                     border_color_glow, 3)
        
        # Display camera feed or placeholder
        if camera_frame is not None:
            # Resize camera to fill the entire area (like pattern mode)
            cam_frame_resized = cv2.resize(camera_frame, (camera_w, camera_h))

            # Copy frame to main image
            img[camera_y:camera_y + camera_h, camera_x:camera_x + camera_w] = cam_frame_resized

            # Throttled bidirectional needle centring check:
            # runs every NEEDLE_CHECK_INTERVAL frames so it's cheap on RPi 4,
            # but always updates needle_confirmed (including back to False if
            # the needle moves away from the ROI centre).
            self._needle_check_counter += 1
            if self._needle_check_counter >= self.NEEDLE_CHECK_INTERVAL:
                self._needle_check_counter = 0
                self.needle_confirmed = self._run_needle_check(cam_frame_resized)
        else:
            # Dark background for camera
            cv2.rectangle(img, (camera_x, camera_y), (camera_x + camera_w, camera_y + camera_h), 
                         COLORS['dark_blue'], -1)
            
            # Show "Opening camera..." message
            msg = "Opening camera..."
            font_scale_msg = text_scale(0.82, self.width, self.height, floor=0.72, ceiling=0.94)
            thickness_msg = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
            (msg_w, msg_h), _ = get_text_size(msg, UI_FONT, font_scale_msg, thickness_msg)
            msg_x = camera_x + (camera_w - msg_w) // 2
            msg_y = camera_y + (camera_h + msg_h) // 2
            _put_text(img, msg, msg_x, msg_y, font_scale_msg, COLORS['text_primary'], thickness_msg)
        
        # ── ROI guide overlay (drawn over the camera area onto img) ──────────────
        abs_cx  = camera_x + self.ROI_CENTER_X
        roi_top = camera_y + self.ROI_TOP_Y
        roi_bot = camera_y + camera_h - self.ROI_BOT_MARGIN
        half_w  = self.ROI_COL_WIDTH // 2
        col_x1  = abs_cx - half_w
        col_x2  = abs_cx + half_w

        # Column outline – always visible so user knows where to position needle
        glow_a    = 0.5 + 0.5 * abs(math.sin(self.glow_phase))
        col_color = tuple(int(c * glow_a) for c in self.ROI_COL_COLOR)
        cv2.rectangle(img, (col_x1, roi_top), (col_x2, roi_bot), col_color, 2)

        if self.needle_confirmed:
            # Dashed centre stitch line (only shown once needle is centred)
            y = roi_top
            while y < roi_bot:
                y_end = min(y + self.ROI_DASH_LEN, roi_bot)
                cv2.line(img, (abs_cx, y), (abs_cx, y_end), self.ROI_LINE_COLOR, 2)
                y += self.ROI_DASH_LEN + self.ROI_DASH_GAP

            # Step 7 only: two extra dashed lines ~1 cm to left and right of centre
            if self.current_step == 7:
                for x_off in (-self.ROI_STEP7_LINE_OFFSET, self.ROI_STEP7_LINE_OFFSET):
                    side_x = abs_cx + x_off
                    y = roi_top
                    while y < roi_bot:
                        y_end = min(y + self.ROI_DASH_LEN, roi_bot)
                        cv2.line(img, (side_x, y), (side_x, y_end), self.ROI_LINE_COLOR, 2)
                        y += self.ROI_DASH_LEN + self.ROI_DASH_GAP
        else:
            # Warning banner: needle not yet centred (shown only while camera is live)
            if camera_frame is not None:
                warn_h  = 44
                warn_y  = camera_y + 8
                overlay = img.copy()
                cv2.rectangle(overlay,
                              (camera_x + 4, warn_y),
                              (camera_x + camera_w - 4, warn_y + warn_h),
                              (0, 60, 180), -1)
                cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
                warn_txt = "!  SEWING NEEDLE NOT CENTRED"
                warn_scale = text_scale(0.72, self.width, self.height, floor=0.64, ceiling=0.84)
                warn_thick = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
                warn_scale = fit_text_scale(warn_txt, UI_FONT, camera_w - 20, warn_scale, warn_thick, min_scale=0.58)
                (ww, wh), _ = get_text_size(warn_txt, UI_FONT, warn_scale, warn_thick)
                wx = camera_x + (camera_w - ww) // 2
                wy = warn_y + (warn_h + wh) // 2
                _put_text(img, warn_txt, wx, wy, warn_scale, (0, 220, 255), warn_thick)
        # ─────────────────────────────────────────────────────────────────────

        # Draw back button (top left)
        self.draw_button(img, self.back_button, COLORS['button_hover'])
        
        # Draw PREVIOUS button to go back to video (bottom left)
        self.draw_button(img, self.your_turn_previous_button, COLORS['button_hover'])
        
        # Draw NEXT button to continue to next step (bottom right)
        self.draw_button(img, self.your_turn_next_button, COLORS['button_hover'])
    
    def _run_needle_check(self, cam_resized):
        """Crop the ROI column from *cam_resized* and run needle.onnx.
        Returns True if the highest-confidence detection centre-X is within
        NEEDLE_CENTER_TOLERANCE pixels of the column's horizontal midpoint."""
        if self.needle_model is None:
            return False
        h, w = cam_resized.shape[:2]
        x1 = max(0, self.ROI_CENTER_X - self.ROI_COL_WIDTH // 2)
        x2 = min(w, self.ROI_CENTER_X + self.ROI_COL_WIDTH // 2)
        y1 = int(self.ROI_TOP_Y)
        y2 = max(y1 + 1, h - int(self.ROI_BOT_MARGIN))
        roi_crop = cam_resized[y1:y2, x1:x2]
        if roi_crop.size == 0:
            return False
        try:
            imgsz = self._needle_imgsz
            roi_h_c, roi_w_c = roi_crop.shape[:2]
            inp = cv2.resize(roi_crop, (imgsz, imgsz))
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))[np.newaxis]  # [1, 3, H, W]
            preds = self.needle_model.run(None, {self.needle_input_name: inp})[0]
            # YOLOv8 ONNX output: [1, 4+nc, anchors] → transpose to [anchors, 4+nc]
            preds = preds[0].T  # [anchors, 5] for a 1-class model
            scores = preds[:, 4]
            mask = scores >= self.NEEDLE_CONF_THRESHOLD
            if not np.any(mask):
                return False
            filtered = preds[mask]
            best = int(np.argmax(filtered[:, 4]))
            cx_320 = float(filtered[best, 0])
            cx_roi = cx_320 / imgsz * roi_w_c
            return abs(cx_roi - roi_w_c / 2.0) <= self.NEEDLE_CENTER_TOLERANCE
        except Exception as e:
            print(f"Needle check error: {e}")
        return False

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
