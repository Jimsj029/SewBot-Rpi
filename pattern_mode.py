import cv2
import numpy as np
import math
import os
from ultralytics import YOLO
from music_manager import get_music_manager
from ui.typography import (
    FONT_MAIN,
    FONT_DISPLAY,
    text_scale,
    text_thickness,
    get_text_size,
    fit_text_scale,
    draw_text,
)


class PatternMode:
    def __init__(self, width, height, colors, blueprint_folder='blueprint'):
        self.width = width
        self.height = height
        self.COLORS = colors
        self.blueprint_folder = blueprint_folder

        # Motion-driven progression tuning
        self.motion_step_gain = 2.2          # Higher = more centerline steps per motion unit
        self.motion_step_min_projection = 0.05  # Ignore tiny projected motion along tangent
        self.allow_reverse_progress = True   # True: backward cloth motion can move index backward
        
        # Pattern variables
        self.current_level = 1
        self.current_level_tracking = 1  # Track which level stats belong to
        self.uniform_width = 216   # 144 * 1.50 = 216 (50% increase)
        self.uniform_height = 270  # 180 * 1.50 = 270 (50% increase)
        # Pattern width scale: fraction of original mask width to use (0.0 - 1.0).
        # Default 0.625 (62.5%) preserves previous behaviour; tweak to change mask width.
        self.pattern_width_scale = 0.625
        self.alpha_blend = 0.9
        self.glow_phase = 0
        
        # Level info display at top center (aligned with back button)
        self.level_display_y = 60
        
        # ── LAYOUT ───────────────────────────────────────────────────────────────
        # Adjust these values to reposition/resize each panel.
        # cloth panel Y and evaluate button Y are derived automatically.

        # Camera
        self.camera_x      = 216
        self.camera_y      = 120
        self.camera_width  = 560
        self.camera_height = 420

        # Left panel — Thread Color
        self.color_panel_x      = 8
        self.color_panel_y      = 115
        self.color_panel_width  = 185
        self.color_panel_height = 115

        # Left panel — Cloth Color (auto-positioned below Thread Color)
        _cloth_gap = 8
        self.cloth_color_panel_x      = self.color_panel_x
        self.cloth_color_panel_y      = self.color_panel_y + self.color_panel_height + _cloth_gap
        self.cloth_color_panel_width  = self.color_panel_width
        self.cloth_color_panel_height = 110

        # Right panel — Stats
        self.score_panel_x      = 800
        self.score_panel_y      = 115
        self.score_panel_width  = 210
        self.score_panel_height = 310

        # Evaluate button (auto-positioned below Stats panel)
        _eval_gap = 8
        self.evaluate_button = {
            'x': self.score_panel_x,
            'y': self.score_panel_y + self.score_panel_height + _eval_gap,
            'w': self.score_panel_width,
            'h': 50,
        }
        # Button to let user confirm cloth placement and attach mask to cloth
        self.attach_cloth_button = {
            'x': self.score_panel_x,
            'y': self.evaluate_button['y'] + self.evaluate_button['h'] + 8,
            'w': self.score_panel_width,
            'h': 40,
        }

        # Back button
        self.back_button = {'x': 20, 'y': 20, 'w': 120, 'h': 50}
        # ── END LAYOUT ───────────────────────────────────────────────────────────
        self.selected_detection_color = 'white'
        self.color_profiles = {
            'white': {
                'label': 'WHITE',
                'preview_bgr': (240, 240, 240),
                'hsv_ranges': [((0, 0, 170), (179, 50, 255))],
                'min_ratio': 0.15,
            },
            'yellow': {
                'label': 'YELLOW',
                'preview_bgr': (0, 255, 255),
                'hsv_ranges': [((18, 80, 80), (40, 255, 255))],
                'min_ratio': 0.15,
            },
            'red': {
                'label': 'RED',
                'preview_bgr': (0, 0, 255),
                'hsv_ranges': [
                    ((0, 90, 70), (10, 255, 255)),
                    ((160, 90, 70), (179, 255, 255)),
                ],
                'min_ratio': 0.15,
            },
            'black': {
                'label': 'BLACK',
                'preview_bgr': (30, 30, 30),
                # Loosened HSV and match ratio for better black detection in
                # varied lighting. V upper increased, S upper increased.
                'hsv_ranges': [
                    ((0, 0, 0), (179, 180, 90)),
                ],
                'min_ratio': 0.06,
            },
        }
        self.color_buttons = {}

        # Confidence controls (kept for click-handler compatibility, hidden)
        self.conf_panel_x = -999
        self.conf_panel_y = -999
        self.conf_panel_width = 200
        self.conf_panel_height = 65
        self.conf_minus_button = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
        self.conf_plus_button = {'x': 0, 'y': 0, 'w': 0, 'h': 0}

        # Live color-match indicator state
        self.last_color_match_ratio = 0.0
        self.last_color_match = False
        self.last_color_mask = None
        self.last_color_mask_bounds = None
        
        # Try again button (centered in modal) - will be calculated dynamically
        self.try_again_button = {'x': 0, 'y': 0, 'w': 200, 'h': 60}
        
        # Next level button (centered in modal) - will be calculated dynamically
        self.next_level_button = {'x': 0, 'y': 0, 'w': 200, 'h': 60}
        
        # Game tracking variables
        self.current_accuracy = 0.0
        self.total_score = 0
        self.pattern_progress = 0.0  # 0-100%
        self.raw_progress = 0.0  # Raw actual progress for evaluation
        self.session_start_time = None
        
        # Evaluation state
        self.is_evaluated = False
        self.level_completed = False
        
        # Guide/Tutorial state
        self.show_guide = False  # Will be set to True on first level entry
        self.guide_shown_this_session = False  # Track if guide was shown
        self.guide_step = 1  # Current step (1-4)
        self.guide_button = {'x': 0, 'y': 0, 'w': 200, 'h': 60}  # "Next"/"Got it!" button, calculated dynamically
        
        # Real-time stitch tracking for progressive coloring
        self.completed_stitch_mask = None  # Accumulated mask of all detected stitches
        self.proximity_radius = 5  # Legacy compatibility radius (kept small)
        self.stitch_draw_radius = 3  # Radius of each accepted stitch stamp
        self.cyan_spread_radius = 2  # Visual cyan expansion around accepted stitches
        self.progress_spread_radius = 4  # Progress expansion to bridge tiny gaps only
        # Keep these permissive so centerline-guided tracing can advance one
        # pixel step at a time on small screens/cameras.
        self.min_stitch_move_px = 1.0  # Min movement before accepting another center stitch
        self.stitch_cooldown_frames = 1  # Min frames between accepted center stitches
        self.stitch_frame_index = 0
        self.last_stitch_frame_index = -9999
        self.last_stitch_point = None
        self.use_model_stitch_points = False  # Keep tracing tied to the red-dot center
        self.needle_model_imgsz = 320  # needle.onnx expects 320x320 input

        # Motion gate: a valid stitch requires cloth motion near the needle,
        # preventing static color patches from auto-advancing the pattern.
        self.require_motion_for_stitch = True
        self.motion_patch_size = 56
        self.motion_patch_y_offset = 14
        self.motion_diff_threshold = 4.5
        self.motion_grace_frames = 8
        self.motion_grace_counter = 0
        self.prev_motion_patch = None
        self.last_motion_diff = 0.0
        self.cloth_motion_active = False
        self.last_motion_centroid = None
        self.prev_cloth_color_centroid = None
        self.last_cloth_mask_ratio = 0.0
        self.last_cloth_bbox = None
        self.cloth_motion_min_ratio = 0.015
        self.cloth_motion_min_shift = 0.35
        self.cloth_motion_min_area_ratio = 0.01
        self.motion_vector_x = 0.0
        self.motion_vector_y = 0.0
        self.motion_vector_min_mag = 0.35
        # Relax cloth detection thresholds slightly to be more permissive
        self.cloth_motion_min_ratio = 0.01
        self.cloth_motion_min_area_ratio = 0.005

        self.motion_step_gain = 2.2  # Converts projected cloth motion into centerline index steps
        self.motion_step_min_projection = 0.05  # Ignore tiny along-path projections (noise)
        self.motion_sideways_gain = 1.2  # Additional gain when moving mostly sideways to the path
        self.allow_sideways_progress = True  # Count sideways cloth movement as forward progress
        self.allow_reverse_progress = True  # Allow backward cloth motion to move backward on path
        self.needle_on_pattern = True  # Whether the needle is currently on a pattern pixel

        # Follow-line validation (on/off pattern while sewing)
        self.pattern_alpha_threshold = 0.5
        self.follow_corridor_radius = 6
        self.follow_order_tolerance = 24
        self.follow_centerline_distance = 8
        self.snap_stitches_to_centerline = True
        self.centerline_step_limit = 6  # Max index jump per accepted center sample
        self.follow_off_confirm_frames = 4
        self.follow_on_confirm_frames = 2
        self.follow_off_count = 0
        self.follow_on_count = 0
        self.centerline_progress_idx = 0
        self.centerline_progress_initialized = False

        # ── Needle ROI – change these two values to reposition the detection window ──
        self.NEEDLE_ROI_SIZE = 64   # Side-length of the square ROI (camera-frame pixels)
        self.NEEDLE_ROI_X    = 280  # ROI centre X in camera-frame pixels  ← adjust freely
        self.NEEDLE_ROI_Y    = 210  # ROI centre Y in camera-frame pixels  ← adjust freely
        # ─────────────────────────────────────────────────────────────────────────────

        # Needle optical-flow tracking (updates pos between YOLO calls)
        self.needle_pos_x = float(self.NEEDLE_ROI_X)
        self.needle_pos_y = float(self.NEEDLE_ROI_Y)
        self.prev_gray    = None   # Grayscale previous frame
        self.of_points    = None   # Lucas-Kanade tracking points
        # Hole radius inside the needle ROI to keep center unobstructed
        self.needle_roi_hole_radius = 4
        # Storage for the raw camera frame (pre-overlay) to restore ROI hole
        self._last_raw_frame = None
        # When True, the pattern overlay follows the detected cloth centroid
        # rather than staying fixed to the camera centre. This is latched on
        # first successful cloth detection so the mask 'sticks' to the cloth.
        self.mask_attached_to_cloth = False
        # If attached, this stores the anchor point (x,y) in camera-frame
        # coordinates where the mask should be centered (usually cloth centroid).
        self.mask_anchor = None
        # When START is pressed we record the initial cloth centroid and an
        # initial overlay anchor (usually the camera centre). Subsequent
        # cloth movement will translate the overlay relative to this initial
        # seed so the mask keeps its initial size/scale even if cloth
        # moves partly out of frame.
        self.mask_initial_cloth_centroid = None
        self.mask_initial_anchor = None
        self.mask_initial_cloth_bbox = None
        # Initial expected pattern point (centerline) recorded at START
        # so overlay offsets don't change as centerline index advances.
        self.mask_initial_expected_point = None
        # Exponential smoothing for cloth centroid to reduce jitter while
        # the mask is attached. Alpha in (0,1] where higher is more responsive.
        self._cloth_centroid_ema = None
        self._cloth_centroid_ema_alpha = 0.45

        # ── Column ROI – wallet-style needle centring guide ───────────────────
        self.ROI_CENTER_X         = 272   # Column horizontal centre (camera-frame px)
        self.ROI_COL_WIDTH        = 140   # Column width in pixels
        self.ROI_TOP_Y            = 0     # Y start (top of camera feed)
        self.ROI_BOT_MARGIN       = 0     # Y margin from bottom of camera feed
        self.ROI_COL_COLOR        = (0, 255, 255)  # BGR colour of column outline
        self.needle_confirmed        = False
        self.NEEDLE_CONF_THRESHOLD   = 0.35
        self.NEEDLE_CENTER_TOLERANCE = 40
        self.NEEDLE_CHECK_INTERVAL   = 6
        self._needle_check_counter   = 0
        # ─────────────────────────────────────────────────────────────────────

        # Segment tracking (divide pattern into 4 quarters)
        self.current_segment = 1  # 1=first 25%, 2=25-50%, 3=50-75%, 4=75-100%
        self.highest_segment_reached = 1  # Track highest segment to prevent going backwards
        self.segment_colors = {
            'completed': (255, 255, 0),    # Cyan in BGR (completed sections)
            'current': (0, 255, 255),      # Yellow in BGR (current section to sew)
            'upcoming': (100, 100, 100)    # Dim Gray (upcoming sections)
        }
        
        # Out-of-segment detection
        self.out_of_segment_warning = False
        self.warning_message = ""
        self.warning_flash_phase = 0
        # Combo and stitch tracking
        self.current_combo = 0
        self.max_combo = 0
        self.stitches_detected = 0
        
        # Load models from models directory
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Detection settings
        self.confidence_threshold = 0.3  # Lowered for INT8 model (produces lower confidence scores)
        self.iou_threshold = 0.6  # Intersection over Union threshold
        # Toggle to enable/disable the needle (ONNX) detection pipeline
        self.needle_detection_enabled = True
        
        # Cloth colour selector
        self.selected_cloth_color = 'red'
        self.cloth_color_profiles = {
            'red': {
                'label': 'RED',
                'preview_bgr': (0, 0, 220),
                'hsv_ranges': [
                    ((0,  50, 35), (15, 255, 255)),
                    ((155, 50, 35), (179, 255, 255)),
                ],
            },
            'black': {
                'label': 'BLACK',
                'preview_bgr': (30, 30, 30),
                # Loosened HSV to include darker grays under varied lighting
                'hsv_ranges': [
                    ((0, 0, 0), (179, 180, 90)),
                ],
                # Minimum fraction of sampled patch required to declare a match
                'min_ratio': 0.06,
            },
            'white': {
                'label': 'WHITE',
                'preview_bgr': (240, 240, 240),
                'hsv_ranges': [
                    ((0, 0, 180), (179, 40, 255)),
                ],
            },
            'gray': {
                'label': 'GRAY',
                'preview_bgr': (140, 140, 140),
                'hsv_ranges': [
                    ((0, 0, 60), (179, 50, 175)),
                ],
            },
        }
        self.cloth_color_buttons = {}

        # Sampling radius (px) when checking for thread colour at the needle
        # Reduce this to avoid overly large samples turning pixels cyan prematurely.
        # Lowered to 3 per user request.
        self.color_sample_radius = 3

        # Cloth bbox tracking
        self.smooth_cloth_bbox = None  # Smoothed bbox for stable overlay

        # Load needle detection model (used for single-stitch ROI pipeline)
        try:
            needle_model_path = os.path.join(models_dir, 'needle.onnx')
            if os.path.exists(needle_model_path):
                print(f"Loading needle detection model: {needle_model_path}")
                self.needle_model = YOLO(needle_model_path, task='detect')
                print(f"✓ Needle detection model loaded successfully!")
                if hasattr(self.needle_model, 'names'):
                    for idx, name in self.needle_model.names.items():
                        print(f"    Class {idx}: {name}")
                test_img = np.zeros((64, 64, 3), dtype=np.uint8)
                self.needle_model(test_img, conf=self.confidence_threshold, verbose=False)
                print("  ✓ Needle model test successful!")
            else:
                print(f"⚠ Needle model not found: {needle_model_path}")
                self.needle_model = None
        except Exception as e:
            print(f"⚠ ERROR loading needle model: {e}")
            self.needle_model = None

        # Music flag to track if music is playing
        self.music_playing = False
    
    def start_music(self):
        """Start pattern mode music"""
        if not self.music_playing:
            music_manager = get_music_manager()
            music_manager.play('pattern.mp3', loops=-1, fade_ms=1000)
            self.music_playing = True
    
    def stop_music(self):
        """Stop pattern mode music"""
        if self.music_playing:
            music_manager = get_music_manager()
            music_manager.stop(fade_ms=1000)
            self.music_playing = False
    
    def play_button_click_sound(self):
        """Play button click sound effect"""
        music_manager = get_music_manager()
        music_manager.play_sound_effect('button_click.mp3')
    
    def load_blueprint(self, level):
        """Load binary mask for the pattern"""
        # Reset progress if level changed
        if level != self.current_level_tracking:
            self.reset_progress()
            self.current_level_tracking = level
        
        # Prefer exact filename, but allow flexible labeled filenames.
        mask_path = os.path.join(self.blueprint_folder, f'level{level}_mask.png')
        if not os.path.exists(mask_path):
            # Search for any png file that contains the level token or label
            candidates = [f for f in os.listdir(self.blueprint_folder)
                          if os.path.isfile(os.path.join(self.blueprint_folder, f))]
            found = None
            level_str = str(level)
            for fn in candidates:
                if f"level{level_str}" in fn or level_str in fn:
                    found = fn
                    break
            if found is None:
                for fn in candidates:
                    if fn.lower().endswith('.png'):
                        found = fn
                        break
            if found is None:
                return None, None
            mask_path = os.path.join(self.blueprint_folder, found)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None

        # Resize to the uniform canvas size (keep full mask height)
        mask = cv2.resize(mask, (self.uniform_width, self.uniform_height), interpolation=cv2.INTER_NEAREST)

        # Mirror the pattern horizontally for specific levels if required
        if isinstance(level, int) and level in (4, 5):
            mask = cv2.flip(mask, 1)

        # Base width scaling while preserving full height
        new_w = int(round(mask.shape[1] * float(self.pattern_width_scale)))
        new_w = max(1, min(mask.shape[1], new_w))
        if new_w != mask.shape[1]:
            mask = cv2.resize(mask, (new_w, mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Center mask inside canonical uniform canvas
        top_offset = (self.uniform_height - mask.shape[0]) // 2
        left_offset = (self.uniform_width - mask.shape[1]) // 2
        centered_mask = np.zeros((self.uniform_height, self.uniform_width), dtype=mask.dtype)
        centered_mask[top_offset:top_offset + mask.shape[0], left_offset:left_offset + mask.shape[1]] = mask
        mask = centered_mask

        # Convert mask to white overlay (pattern lines) and alpha
        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        alpha = (mask.astype(np.float32) / 255.0)
        
        return overlay, alpha
    
    def reset_progress(self):
        """Reset progress tracking for a new level"""
        self.pattern_progress = 0.0
        self.raw_progress = 0.0
        self.current_segment = 1
        self.highest_segment_reached = 1
        self.current_accuracy = 0.0
        self.total_score = 0
        self.out_of_segment_warning = False
        self.warning_message = ""
        self.current_combo = 0
        self.max_combo = 0
        self.stitches_detected = 0
        self.is_evaluated = False
        self.level_completed = False
        # Clear accumulated stitch mask for real-time coloring
        self.completed_stitch_mask = None
        self.stitch_frame_index = 0
        self.last_stitch_frame_index = -9999
        self.last_stitch_point = None
        self.motion_grace_counter = 0
        self.prev_motion_patch = None
        self.last_motion_diff = 0.0
        self.cloth_motion_active = False
        self.last_motion_centroid = None
        self.prev_cloth_color_centroid = None
        self.last_cloth_mask_ratio = 0.0
        self.last_cloth_bbox = None
        self.motion_vector_x = 0.0
        self.motion_vector_y = 0.0
        self.centerline_progress_idx = 0
        self.centerline_progress_initialized = False
        self.needle_on_pattern = True
        self.follow_off_count = 0
        self.follow_on_count = 0
        print(f"🔄 Progress reset for Level {self.current_level}")
    
    def create_realtime_pattern(self, overlay, alpha, trace_y=None):
        """Create real-time colored pattern overlay.

        Yellow = unsewn pattern pixels, Cyan = stitched pixels.
        A pixel turns cyan only after expected thread-color detection marks it in
        completed_stitch_mask.
        """
        if overlay is None or alpha is None:
            return None

        colored_overlay = overlay.copy()
        height, width = overlay.shape[:2]
        pattern_pixels = alpha > 0.1

        # Base state: everything in the blueprint is still to be sewn (yellow).
        colored_overlay[pattern_pixels] = self.segment_colors['current']

        # Completed stitch mask paints nearby blueprint pixels cyan.
        if self.completed_stitch_mask is not None:
            # Resize completed mask to pattern overlay size (nearest to keep pixels)
            completed_mask_resized = cv2.resize(self.completed_stitch_mask, (width, height), interpolation=cv2.INTER_NEAREST)

            # Use a more generous dilation so small detected stitches more
            # reliably propagate cyan across the visible blueprint. This
            # helps when detections are sparse or sampling is noisy.
            kernel_radius = max(self.cyan_spread_radius * 3, self.progress_spread_radius, 3)
            kernel_size = int(max(3, kernel_radius * 2 + 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            # Perform two dilation passes for stronger coverage
            proximity_mask = cv2.dilate(completed_mask_resized, kernel, iterations=2)
            completed_pattern_pixels = np.logical_and(pattern_pixels, proximity_mask > 0)
            colored_overlay[completed_pattern_pixels] = self.segment_colors['completed']

        return colored_overlay
    

    # ──────────────────────────────────────────────────────────────────────────
    # Needle ROI pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def _update_needle_optical_flow(self, gray_frame):
        """No-op: ROI is fixed at NEEDLE_ROI_X/Y — optical flow intentionally disabled."""
        pass

    def _update_cloth_motion_state(self, cam_frame):
        """Track cloth motion by following selected cloth color across the full frame."""
        if cam_frame is None or cam_frame.size == 0:
            self.cloth_motion_active = False
            self.last_motion_diff = 0.0
            self.last_motion_centroid = None
            self.prev_cloth_color_centroid = None
            self.last_cloth_mask_ratio = 0.0
            self.last_cloth_bbox = None
            self.motion_vector_x = 0.0
            self.motion_vector_y = 0.0
            return

        blurred = cv2.GaussianBlur(cam_frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cloth_mask = self._get_cloth_color_mask(blurred, hsv_patch=hsv)
        kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_OPEN, kernel_sm)
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, kernel_lg)

        mask_pixels = int(np.count_nonzero(cloth_mask))
        total_pixels = max(1, cloth_mask.shape[0] * cloth_mask.shape[1])
        self.last_cloth_mask_ratio = float(mask_pixels) / float(total_pixels)

        contours, _ = cv2.findContours(cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea) if contours else None
        min_area = float(total_pixels) * float(self.cloth_motion_min_area_ratio)

        moved = False
        if (
            largest is not None
            and cv2.contourArea(largest) >= min_area
            and self.last_cloth_mask_ratio >= self.cloth_motion_min_ratio
            and mask_pixels >= 20
        ):
            bx, by, bw, bh = cv2.boundingRect(largest)
            self.last_cloth_bbox = (bx, by, bx + bw, by + bh)
            m = cv2.moments(largest)
            if m["m00"] == 0:
                raw_centroid = np.array([float(bx + bw * 0.5), float(by + bh * 0.5)], dtype=np.float32)
            else:
                raw_centroid = np.array([float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])], dtype=np.float32)

            # Update EMA-smoothed centroid to reduce jitter. Use raw_centroid
            # to seed the EMA when needed.
            alpha = float(self._cloth_centroid_ema_alpha)
            if self._cloth_centroid_ema is None:
                self._cloth_centroid_ema = raw_centroid.copy()
            else:
                try:
                    self._cloth_centroid_ema = (1.0 - alpha) * self._cloth_centroid_ema + alpha * raw_centroid
                except Exception:
                    self._cloth_centroid_ema = raw_centroid.copy()

            centroid = self._cloth_centroid_ema
            self.last_motion_centroid = centroid

            if self.prev_cloth_color_centroid is not None:
                raw_dx = float(centroid[0] - self.prev_cloth_color_centroid[0])
                raw_dy = float(centroid[1] - self.prev_cloth_color_centroid[1])
                self.last_motion_diff = float(math.hypot(raw_dx, raw_dy))
                blend = 0.35
                self.motion_vector_x = (1.0 - blend) * self.motion_vector_x + blend * raw_dx
                self.motion_vector_y = (1.0 - blend) * self.motion_vector_y + blend * raw_dy
                moved = self.last_motion_diff >= self.cloth_motion_min_shift
            else:
                self.last_motion_diff = 0.0

            self.prev_cloth_color_centroid = centroid.copy()
            # Do NOT auto-latch the mask here. The user must explicitly
            # press START to show/attach the pattern overlay. Keep
            # `last_motion_centroid` for optional anchoring when START is
            # pressed, but don't change `mask_attached_to_cloth` automatically.
        else:
            self.last_motion_diff = 0.0
            self.last_motion_centroid = None
            self.prev_cloth_color_centroid = None
            self.last_cloth_bbox = None
            self.motion_vector_x *= 0.7
            self.motion_vector_y *= 0.7

        if moved:
            self.motion_grace_counter = self.motion_grace_frames
        else:
            self.motion_grace_counter = max(0, self.motion_grace_counter - 1)

        self.cloth_motion_active = self.motion_grace_counter > 0

    def _get_cloth_color_mask(self, bgr_patch, hsv_patch=None):
        """Build binary mask for the currently selected cloth color in a BGR patch."""
        cfg = self.cloth_color_profiles[self.selected_cloth_color]
        if hsv_patch is None:
            hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv_patch.shape[:2], dtype=np.uint8)
        for low, high in cfg['hsv_ranges']:
            low_np = np.array(low, dtype=np.uint8)
            high_np = np.array(high, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_patch, low_np, high_np))
        return mask

    def _infer_centerline_step_direction(self, centerline_path, current_idx):
        """Infer +1/-1 movement along centerline from local cloth motion."""
        if centerline_path is None or len(centerline_path) < 3:
            return 0

        idx = int(np.clip(current_idx, 0, len(centerline_path) - 1))
        prev_idx = max(0, idx - 1)
        next_idx = min(len(centerline_path) - 1, idx + 1)
        if prev_idx == next_idx:
            return 0

        tangent = centerline_path[next_idx].astype(np.float32) - centerline_path[prev_idx].astype(np.float32)
        tan_norm = float(np.linalg.norm(tangent))
        if tan_norm < 1e-6:
            return 0
        tangent /= tan_norm

        motion = np.array([self.motion_vector_x, self.motion_vector_y], dtype=np.float32)
        motion_mag = float(np.linalg.norm(motion))
        if motion_mag < self.motion_vector_min_mag:
            return 0

        projection = float(np.dot(motion, tangent))
        if abs(projection) < 0.05:
            return 0

        return 1 if projection > 0 else -1

    def _infer_centerline_step_delta(self, centerline_path, current_idx):
        """Infer signed multi-step movement along centerline from cloth motion."""
        if centerline_path is None or len(centerline_path) < 3:
            return 0

        idx = int(np.clip(current_idx, 0, len(centerline_path) - 1))
        prev_idx = max(0, idx - 1)
        next_idx = min(len(centerline_path) - 1, idx + 1)
        if prev_idx == next_idx:
            return 0

        tangent = centerline_path[next_idx].astype(np.float32) - centerline_path[prev_idx].astype(np.float32)
        tan_norm = float(np.linalg.norm(tangent))
        if tan_norm < 1e-6:
            return 0
        tangent /= tan_norm

        motion = np.array([self.motion_vector_x, self.motion_vector_y], dtype=np.float32)
        motion_mag = float(np.linalg.norm(motion))
        if motion_mag < self.motion_vector_min_mag:
            return 0

        projection = float(np.dot(motion, tangent))
        min_proj = float(self.motion_step_min_projection)
        if abs(projection) < min_proj:
            if not self.allow_sideways_progress:
                return 0
            # Sideways fallback: when cloth clearly moves but along-path projection
            # is weak, still advance so horizontal movement can progress tracing.
            sideways_mag = math.sqrt(max(0.0, motion_mag * motion_mag - projection * projection))
            if sideways_mag < self.motion_vector_min_mag:
                return 0
            signed_steps = int(round(sideways_mag * float(self.motion_sideways_gain)))
            if signed_steps <= 0:
                signed_steps = 1
            return signed_steps

        signed_steps = int(round(projection * float(self.motion_step_gain)))
        if signed_steps == 0:
            signed_steps = 1 if projection > 0 else -1

        if not self.allow_reverse_progress and signed_steps < 0:
            return 0

        return signed_steps

    def run_needle_pipeline(self, cam_frame, pattern_alpha,
                            x_offset, y_offset, actual_w, actual_h,
                            run_detection=True, hsv_frame=None,
                            expected_trace_y=None, corridor_mask=None,
                            centerline_path=None, expected_path_idx=None):
        """
        Single-needle detection pipeline (replaces multi-stitch update_game_stats).

        Pipeline:
          Camera frame
            → optical flow tracks needle between YOLO calls
            → 64×64 ROI cropped around needle_pos
            → needle.onnx detects needle / stitch
            → stitch centre mapped to pattern coordinates
            → pattern overlap check
            → completed_stitch_mask updated
            → progress recalculated
        """
        if pattern_alpha is None:
            return

        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)

        # Always try to refine position via optical flow
        self._update_needle_optical_flow(gray)

        # Only run heavy YOLO inference on scheduled frames
        if not run_detection or self.needle_model is None:
            return

        half = self.NEEDLE_ROI_SIZE // 2
        rx1 = max(0, int(self.needle_pos_x) - half)
        ry1 = max(0, int(self.needle_pos_y) - half)
        rx2 = min(self.camera_width,  rx1 + self.NEEDLE_ROI_SIZE)
        ry2 = min(self.camera_height, ry1 + self.NEEDLE_ROI_SIZE)

        roi = cam_frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return

        # Restore a small circular 'hole' in the ROI centre from the raw
        # pre-overlay frame so the detector sees the unobstructed camera
        # pixels at the needle centre. Uses the last saved raw frame if available.
        try:
            if self._last_raw_frame is not None:
                raw_h, raw_w = self._last_raw_frame.shape[:2]
                # Clip coordinates to available raw frame
                sx1 = max(0, rx1)
                sy1 = max(0, ry1)
                sx2 = min(raw_w, rx2)
                sy2 = min(raw_h, ry2)
                if sx2 > sx1 and sy2 > sy1:
                    src_block = self._last_raw_frame[sy1:sy2, sx1:sx2]
                    dst_block = roi[sy1 - ry1:sy2 - ry1, sx1 - rx1:sx2 - rx1]
                    # Build circular mask in ROI coordinates
                    h_blk, w_blk = dst_block.shape[:2]
                    cx = w_blk // 2
                    cy = h_blk // 2
                    rr = int(max(1, min(cx, cy, int(self.needle_roi_hole_radius))))
                    yy, xx = np.ogrid[:h_blk, :w_blk]
                    circle = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (rr * rr)
                    # Apply mask: copy pixels where circle is True
                    dst_block[circle] = src_block[circle]
                    # Write back into roi (and cam_frame slice will reflect since roi is a view)
                    roi[sy1 - ry1:sy2 - ry1, sx1 - rx1:sx2 - rx1] = dst_block
        except Exception:
            pass

        try:
            # Run needle model at its required ONNX input size.
            # Reuse current confidence threshold so +/- controls affect both models.
            results = self.needle_model(
                roi,
                conf=self.confidence_threshold,
                imgsz=self.needle_model_imgsz,
                verbose=False,
            )

            num_boxes = len(results[0].boxes) if (results and results[0].boxes is not None) else 0

            # ── Canny fallback when YOLO finds nothing ─────────────────────────
            # needle.onnx is trained for cloth-level detection and won't fire on
            # a small 64×64 stitch crop.  Use edge+contour centroid instead.
            if num_boxes == 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # Enhance contrast before edge detection
                gray_roi = cv2.equalizeHist(gray_roi)
                edges = cv2.Canny(gray_roi, 40, 120)
                cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                # Pick contours large enough to be a real stitch (>= 15 px²)
                cnts = [c for c in cnts if cv2.contourArea(c) >= 15]
                if cnts:
                    # Use the contour closest to the ROI centre
                    roi_cx, roi_cy = roi.shape[1] / 2, roi.shape[0] / 2
                    def dist_to_centre(c):
                        M = cv2.moments(c)
                        if M["m00"] == 0: return 9999
                        return math.hypot(M["m10"]/M["m00"] - roi_cx,
                                          M["m01"]/M["m00"] - roi_cy)
                    best_cnt = min(cnts, key=dist_to_centre)
                    M = cv2.moments(best_cnt)
                    if M["m00"] != 0:
                        fx = rx1 + M["m10"] / M["m00"]
                        fy = ry1 + M["m01"] / M["m00"]
                        self._register_stitch(fx, fy, cam_frame,
                                              pattern_alpha, x_offset, y_offset,
                                              actual_w, actual_h, source="canny", hsv_frame=hsv_frame,
                                              expected_trace_y=expected_trace_y,
                                              corridor_mask=corridor_mask,
                                              centerline_path=centerline_path,
                                              expected_path_idx=expected_path_idx)
                return   # done whether canny fired or not
            # ──────────────────────────────────────────────────────────────────

            if num_boxes > 0:
                boxes  = results[0].boxes
                confs  = boxes.conf.cpu().numpy()
                best   = int(np.argmax(confs))
                xyxy   = boxes[best].xyxy[0].cpu().numpy()
                cx_cam = rx1 + (xyxy[0] + xyxy[2]) / 2.0
                cy_cam = ry1 + (xyxy[1] + xyxy[3]) / 2.0

                self._register_stitch(cx_cam, cy_cam, cam_frame,
                                      pattern_alpha, x_offset, y_offset,
                                      actual_w, actual_h,
                                      source=f"yolo conf={float(confs[best]):.2f}", hsv_frame=hsv_frame,
                                      expected_trace_y=expected_trace_y,
                                      corridor_mask=corridor_mask,
                                      centerline_path=centerline_path,
                                      expected_path_idx=expected_path_idx)
        except Exception as e:
            import traceback
            print(f"Needle detection error: {e}")
            traceback.print_exc()

    def _register_stitch(self, cx_cam, cy_cam, cam_frame,
                         pattern_alpha, x_offset, y_offset, actual_w, actual_h,
                         source="", hsv_frame=None,
                         expected_trace_y=None, corridor_mask=None,
                         centerline_path=None, expected_path_idx=None):
        """Check if a candidate stitch position overlaps the pattern and record it."""
        if source != "center" and not self.use_model_stitch_points:
            return

        if not self._matches_selected_color(cam_frame, cx_cam, cy_cam, hsv_frame=hsv_frame):
            return

        # If the user has detected thread colour at the needle, allow
        # accepting the stitch even when cloth motion is weak. This makes
        # the painting continuous when colour detection is present.
        if source == "center" and self.require_motion_for_stitch and not self.cloth_motion_active and not getattr(self, 'last_color_match', False):
            return

        pat_h, pat_w = pattern_alpha.shape[:2]

        # Map from camera-frame coords to pattern-overlay coords
        px = int(np.clip(cx_cam - x_offset, 0, actual_w - 1))
        py = int(np.clip(cy_cam - y_offset, 0, actual_h - 1))

        is_valid_stitch, on_corridor, order_valid, snapped_x, snapped_y, nearest_idx = self._validate_pattern_position(
            px, py, pattern_alpha, actual_w, actual_h,
            expected_trace_y=expected_trace_y,
            corridor_mask=corridor_mask,
            centerline_path=centerline_path,
            expected_path_idx=expected_path_idx,
            return_snap_point=True
        )

        print(f"🪡 [{source}] cam=({cx_cam:.0f},{cy_cam:.0f})  "
              f"pat=({px},{py})  corridor={on_corridor}  order={order_valid}")

        # If color is detected at the sample but pattern validators rejected
        # the point (e.g., order_valid), allow acceptance when the sample is
        # on the corridor and color match is strong. This prevents stalls.
        if not is_valid_stitch and getattr(self, 'last_color_match', False) and on_corridor:
            is_valid_stitch = True
            order_valid = True

        if is_valid_stitch:
            draw_x, draw_y = px, py
            if self.snap_stitches_to_centerline and centerline_path is not None and len(centerline_path) > 0:
                draw_x, draw_y = snapped_x, snapped_y

            if source == "center":
                # Allow bypass of cooldown when we have a strong color match
                if (self.stitch_frame_index - self.last_stitch_frame_index) < self.stitch_cooldown_frames and not getattr(self, 'last_color_match', False):
                    return
                # In centerline mode we intentionally allow dense adjacent points.
                # A strict movement check here can deadlock progression on tightly
                # sampled paths where consecutive points overlap existing circles.
                use_centerline_mode = (
                    self.snap_stitches_to_centerline
                    and centerline_path is not None
                    and len(centerline_path) > 0
                )
                if (not use_centerline_mode) and self.last_stitch_point is not None:
                    lx, ly = self.last_stitch_point
                    min_move = float(self.min_stitch_move_px)
                    # Bypass minimum-move gate when thread colour is present
                    if not getattr(self, 'last_color_match', False) and math.hypot(draw_x - lx, draw_y - ly) < min_move:
                        return

                # Ensure we store stitches in the canonical uniform-resolution
                # mask to keep visuals consistent between ROI-model updates and
                # center-sampled registrations.
                if self.completed_stitch_mask is None:
                    self.completed_stitch_mask = np.zeros((self.uniform_height, self.uniform_width), dtype=np.uint8)

                prev_pixels = int(np.count_nonzero(self.completed_stitch_mask))

                # Map pattern-space draw coordinates (px,py) into uniform mask
                # in case pattern overlay size differs from `uniform_*` dims.
                try:
                    scale_x = float(self.uniform_width) / float(pat_w)
                    scale_y = float(self.uniform_height) / float(pat_h)
                except Exception:
                    scale_x = 1.0
                    scale_y = 1.0

                ux = int(np.clip(round(draw_x * scale_x), 0, self.uniform_width - 1))
                uy = int(np.clip(round(draw_y * scale_y), 0, self.uniform_height - 1))

                # Scale draw radius to uniform coord space (min 1). If color
                # is matched, use a slightly larger stamp to ensure new pixels
                # are created so progress advances continuously.
                base_r = max(1, int(round(self.stitch_draw_radius * max(scale_x, scale_y))))
                if getattr(self, 'last_color_match', False):
                    ur = max(base_r, int(round(base_r * 1.5)))
                else:
                    ur = base_r
                cv2.circle(self.completed_stitch_mask, (ux, uy), ur, 255, -1)
                # Also add a larger filled stamp to improve visual coverage so
                # single-point detections more reliably turn blueprint pixels
                # cyan after dilation. The extra radius scales with cyan_spread.
                try:
                    extra_r = int(max(0, self.cyan_spread_radius * 3))
                    if getattr(self, 'last_color_match', False):
                        extra_r = max(extra_r, int(self.cyan_spread_radius * 4))
                    if extra_r > 0:
                        cv2.circle(self.completed_stitch_mask, (ux, uy), ur + extra_r, 255, -1)
                except Exception:
                    pass
                new_pixels = int(np.count_nonzero(self.completed_stitch_mask))

            if source == "center":
                # Record accepted center sample even when no new pixels were
                # added, so the next frame can progress to the next path point.
                self.last_stitch_point = (draw_x, draw_y)
                self.last_stitch_frame_index = self.stitch_frame_index
                if centerline_path is not None and nearest_idx is not None and len(centerline_path) > 0:
                    max_idx = len(centerline_path) - 1
                    nearest_idx = int(np.clip(nearest_idx, 0, max_idx))
                    if not self.centerline_progress_initialized:
                        self.centerline_progress_idx = nearest_idx
                        self.centerline_progress_initialized = True
                    else:
                        current_idx = int(self.centerline_progress_idx)
                        # Primary driver: cloth motion projected along centerline.
                        # Fallback to nearest index to avoid deadlocks when motion is weak.
                        motion_delta = self._infer_centerline_step_delta(centerline_path, current_idx)
                        nearest_delta = nearest_idx - current_idx
                        delta = motion_delta if motion_delta != 0 else nearest_delta

                        step_limit = max(1, int(self.centerline_step_limit))
                        if delta > step_limit:
                            delta = step_limit
                        elif delta < -step_limit:
                            delta = -step_limit

                        if delta != 0:
                            self.centerline_progress_idx = int(np.clip(current_idx + delta, 0, max_idx))

            # Only advance stats/progress when this stitch added new coverage.
            if new_pixels > prev_pixels:
                self._update_progress_from_mask(pattern_alpha)
                self.stitches_detected += 1

    def _matches_selected_color(self, cam_frame, cx_cam, cy_cam, hsv_frame=None):
        """Return True when the sampled stitch area matches the currently selected color."""
        color_cfg = self.color_profiles[self.selected_detection_color]

        sample_radius = getattr(self, 'color_sample_radius', 3)
        x = int(cx_cam)
        y = int(cy_cam)
        x1 = max(0, x - sample_radius)
        y1 = max(0, y - sample_radius)
        x2 = min(cam_frame.shape[1], x + sample_radius + 1)
        y2 = min(cam_frame.shape[0], y + sample_radius + 1)

        patch = cam_frame[y1:y2, x1:x2]
        if patch.size == 0:
            self.last_color_match_ratio = 0.0
            self.last_color_match = False
            self.last_color_mask = None
            self.last_color_mask_bounds = None
            return False

        hsv_patch = None
        if hsv_frame is not None:
            hsv_patch = hsv_frame[y1:y2, x1:x2]
        combined_mask = self._get_selected_color_mask(patch, hsv_patch=hsv_patch)

        match_ratio = float(np.count_nonzero(combined_mask)) / float(combined_mask.size)
        self.last_color_match_ratio = match_ratio
        self.last_color_match = match_ratio >= color_cfg['min_ratio']
        self.last_color_mask = combined_mask
        self.last_color_mask_bounds = (x1, y1, x2, y2)
        return self.last_color_match

    def _get_selected_color_mask(self, bgr_patch, hsv_patch=None):
        """Build binary mask for the currently selected color in a BGR patch."""
        color_cfg = self.color_profiles[self.selected_detection_color]
        if hsv_patch is None:
            hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv_patch.shape[:2], dtype=np.uint8)

        for low, high in color_cfg['hsv_ranges']:
            low_np = np.array(low, dtype=np.uint8)
            high_np = np.array(high, dtype=np.uint8)
            combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv_patch, low_np, high_np))

        return combined_mask

    def _get_pattern_corridor_mask(self, pattern_alpha, actual_w, actual_h):
        """Return a dilated binary corridor around the blueprint path."""
        pat_crop = pattern_alpha[:actual_h, :actual_w]
        pat_binary = (pat_crop > self.pattern_alpha_threshold).astype(np.uint8)
        k = max(3, self.follow_corridor_radius * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(pat_binary, kernel, iterations=1)

    def _build_centerline_path(self, pattern_alpha, actual_w, actual_h):
        """Build an ordered centerline path from the binary blueprint mask.

        This is a lightweight skeleton-style center extraction: for each row that
        contains pattern pixels, choose the x-position nearest the previous row's
        center so the path remains continuous.
        
        For levels 4 & 5, starts from the absolute leftmost x across the entire pattern.
        """
        pat_crop = pattern_alpha[:actual_h, :actual_w]
        pat_binary = (pat_crop > self.pattern_alpha_threshold).astype(np.uint8)
        path_points = []
        prev_x = None

        # For levels 4 & 5, find the global leftmost x and start there
        if self.current_level in (4, 5):
            all_xs = []
            row_data = []
            for y in range(actual_h):
                xs = np.where(pat_binary[y] > 0)[0]
                if xs.size > 0:
                    all_xs.extend(xs)
                    row_data.append((y, xs))
            
            if not row_data:
                return None
            
            global_min_x = int(np.min(all_xs))
            
            # Find the first row that contains this leftmost x
            start_y = None
            for y, xs in row_data:
                if global_min_x in xs:
                    start_y = y
                    prev_x = global_min_x
                    path_points.append((global_min_x, y))
                    break
            
            # Continue from that row
            for y in range(start_y + 1 if start_y is not None else 0, actual_h):
                xs = np.where(pat_binary[y] > 0)[0]
                if xs.size == 0:
                    continue
                x = int(xs[np.argmin(np.abs(xs - prev_x))])
                path_points.append((x, y))
                prev_x = x
        else:
            # Standard path building for levels 1-3
            for y in range(actual_h):
                xs = np.where(pat_binary[y] > 0)[0]
                if xs.size == 0:
                    continue

                if prev_x is None:
                    x = int(np.min(xs))  # Start from leftmost pixel in first row
                else:
                    x = int(xs[np.argmin(np.abs(xs - prev_x))])

                path_points.append((x, y))
                prev_x = x

        if not path_points:
            return None

        return np.array(path_points, dtype=np.int32)

    def _validate_pattern_position(self, px, py, pattern_alpha, actual_w, actual_h,
                                   expected_trace_y=None, corridor_mask=None,
                                   centerline_path=None, expected_path_idx=None,
                                   return_snap_point=False):
        """Validate whether a pattern-space point is on the allowed path and in order."""
        if actual_w <= 0 or actual_h <= 0:
            if return_snap_point:
                return False, False, False, px, py, None
            return False, False, False

        px = int(np.clip(px, 0, actual_w - 1))
        py = int(np.clip(py, 0, actual_h - 1))

        # Prefer centerline validation when path is available.
        if centerline_path is not None and len(centerline_path) > 0:
            deltas = centerline_path.astype(np.float32) - np.array([px, py], dtype=np.float32)
            d2 = np.sum(deltas * deltas, axis=1)
            nearest_idx = int(np.argmin(d2))
            nearest_x = int(centerline_path[nearest_idx][0])
            nearest_y = int(centerline_path[nearest_idx][1])
            on_centerline = bool(d2[nearest_idx] <= (self.follow_centerline_distance ** 2))

            order_valid = True
            if expected_path_idx is not None:
                order_valid = abs(nearest_idx - int(expected_path_idx)) <= self.follow_order_tolerance

            is_valid = on_centerline and order_valid
            if return_snap_point:
                return is_valid, on_centerline, order_valid, nearest_x, nearest_y, nearest_idx
            return is_valid, on_centerline, order_valid

        if corridor_mask is None:
            corridor_mask = self._get_pattern_corridor_mask(pattern_alpha, actual_w, actual_h)

        on_corridor = bool(corridor_mask[py, px] > 0)
        order_valid = True
        if expected_trace_y is not None:
            trace_y = int(np.clip(expected_trace_y, 0, actual_h - 1))
            order_valid = abs(py - trace_y) <= self.follow_order_tolerance

        if return_snap_point:
            return (on_corridor and order_valid), on_corridor, order_valid, px, py, None
        return (on_corridor and order_valid), on_corridor, order_valid

    def _update_follow_hysteresis(self, is_valid_position, color_detected):
        """Stabilize off-pattern warnings across frames with color-gated hysteresis."""
        if not color_detected:
            # Don't punish when thread color isn't present; user may be repositioning.
            self.follow_off_count = max(0, self.follow_off_count - 1)
            self.follow_on_count = max(0, self.follow_on_count - 1)
            self.needle_on_pattern = True
            return

        if is_valid_position:
            self.follow_on_count += 1
            self.follow_off_count = 0
            if self.follow_on_count >= self.follow_on_confirm_frames:
                self.needle_on_pattern = True
        else:
            self.follow_off_count += 1
            self.follow_on_count = 0
            if self.follow_off_count >= self.follow_off_confirm_frames:
                self.needle_on_pattern = False

    def _update_progress_from_mask(self, pattern_alpha):
        """Recalculate raw_progress and pattern_progress from completed_stitch_mask."""
        if self.completed_stitch_mask is None:
            return

        pattern_pixels = (pattern_alpha > 0.1)
        total = int(np.sum(pattern_pixels))
        if total == 0:
            return

        # Dilate stitch dots to bridge small gaps (uses proximity_radius)
        k = max(3, self.progress_spread_radius * 2 + 1)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(self.completed_stitch_mask, kernel, iterations=1)

        covered = int(np.sum(np.logical_and(dilated > 0, pattern_pixels)))
        self.raw_progress = min(100.0, covered / total * 100.0)
        print(f"📊 Raw Progress: {self.raw_progress:.1f}%")

        # Update segmented visual progress (one-way ratchet)
        rp = self.raw_progress
        # Require nearly-complete coverage to mark level finished. Map into
        # one-way segments: 25/50/75/100 (100 only when raw_progress >= 100).
        if   rp >= 100: new_seg, new_prog = 5, 100.0
        elif rp >= 75:  new_seg, new_prog = 4,  75.0
        elif rp >= 50:  new_seg, new_prog = 3,  50.0
        elif rp >= 25:  new_seg, new_prog = 2,  25.0
        else:           new_seg, new_prog = 1,   0.0

        if new_seg > self.highest_segment_reached:
            self.highest_segment_reached = new_seg
            self.current_segment  = min(new_seg, 4)
            self.pattern_progress = new_prog
        elif self.highest_segment_reached >= 5:
            self.current_segment  = 4
            self.pattern_progress = 100.0
        else:
            self.current_segment  = self.highest_segment_reached
            self.pattern_progress = (self.highest_segment_reached - 1) * 25.0

        # Auto-hide mask when progress reaches 100% (user-requested behavior)
        try:
            if (self.raw_progress >= 100.0 or self.pattern_progress >= 100.0) and self.mask_attached_to_cloth:
                self.mask_attached_to_cloth = False
                self.mask_anchor = None
                # Clear initial mask state as well
                self.mask_initial_anchor = None
                self.mask_initial_cloth_centroid = None
                self.mask_initial_cloth_bbox = None
                self.mask_initial_expected_point = None
                print("📌 Mask hidden: pattern complete (100%)")
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────

    def draw_glow_rect(self, img, x, y, w, h, color, glow_intensity):
        """Draw rectangle with glow effect"""
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        for i in range(2):
            offset = (i + 1) * 2
            alpha = glow_intensity * (1 - i * 0.4)
            glow_color = tuple(int(c * alpha) for c in color)
            cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), glow_color, 1)

    def _put_text(self, img, text, x, y, scale, color, thickness, font=FONT_MAIN):
        """Draw outlined anti-aliased text for readability."""
        draw_text(img, text, x, y, scale, color, thickness, font=font)
    
    def draw(self, frame, camera_frame, grid_background):
        """Draw the entire pattern mode interface"""
        # Use grid background
        frame[:] = grid_background
        
        # Increment glow phase for animations
        self.glow_phase += 0.05
        
        # Draw current level display at top center
        level_text = f"LEVEL {self.current_level}"
        difficulty_texts = ["EASY", "MEDIUM", "HARD", "EXPERT", "MASTER"]
        difficulty_text = f"[ {difficulty_texts[self.current_level - 1]} ]"
        
        pulse = 0.6 + 0.4 * abs(math.sin(self.glow_phase * 0.8))
        
        font_scale = text_scale(1.2, self.width, self.height, floor=1.02, ceiling=1.35)
        thickness = text_thickness(3, self.width, self.height, min_thickness=2, max_thickness=4)
        font_scale = fit_text_scale(level_text, FONT_DISPLAY, self.width - 220, font_scale, thickness, min_scale=0.9)
        (level_w, level_h), _ = get_text_size(level_text, FONT_DISPLAY, font_scale, thickness)
        level_x = (self.width - level_w) // 2
        level_y = self.level_display_y
        
        # Draw with glow effect
        glow_color = tuple(int(c * pulse) for c in self.COLORS['glow_cyan'])
        draw_text(
            frame,
            level_text,
            level_x,
            level_y,
            font_scale,
            self.COLORS['bright_blue'],
            thickness,
            font=FONT_DISPLAY,
            outline_color=glow_color,
            outline_extra=2,
        )
        
        # Draw difficulty text below
        diff_font_scale = text_scale(0.66, self.width, self.height, floor=0.58, ceiling=0.76)
        diff_thickness = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
        (diff_w, diff_h), _ = get_text_size(difficulty_text, FONT_MAIN, diff_font_scale, diff_thickness)
        diff_x = (self.width - diff_w) // 2
        diff_y = level_y + 30
        self._put_text(frame, difficulty_text, diff_x, diff_y, diff_font_scale, self.COLORS['text_secondary'], diff_thickness)
        
        # Draw back button (top left)
        self.draw_back_button(frame)
        
        # Draw camera feed
        self.draw_camera_feed(frame, camera_frame)
        
        # Draw score/stats panel (right) — includes legend + detected color
        self.draw_score_panel(frame)
        
        # Draw evaluate button (right, below stats)
        self.draw_evaluate_button(frame)
        # Draw attach cloth button (below evaluate)
        self.draw_attach_cloth_button(frame)

        # Draw thread colour panel (left, top)
        self.draw_thread_color_panel(frame)

        # Draw cloth colour panel (left, below thread)
        self.draw_cloth_color_panel(frame)
        
        # Draw evaluation results if evaluated
        if self.is_evaluated:
            self.draw_evaluation_results(frame)
        
        # Draw guide overlay if showing
        if self.show_guide:
            self.draw_guide_overlay(frame)
    
    def draw_back_button(self, frame):
        """Draw back button in top left"""
        bb = self.back_button
        pulse = 0.4 + 0.3 * abs(math.sin(self.glow_phase))
        self.draw_glow_rect(frame, bb['x'], bb['y'], bb['w'], bb['h'], self.COLORS['medium_blue'], pulse)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (bb['x'] + 2, bb['y'] + 2), (bb['x'] + bb['w'] - 2, bb['y'] + bb['h'] - 2), self.COLORS['button_normal'], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        text = "< BACK"
        font_scale = text_scale(0.72, self.width, self.height, floor=0.66, ceiling=0.86)
        thickness = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
        font_scale = fit_text_scale(text, FONT_MAIN, bb['w'] - 12, font_scale, thickness, min_scale=0.58)
        (text_w, text_h), _ = get_text_size(text, FONT_MAIN, font_scale, thickness)
        text_x = bb['x'] + (bb['w'] - text_w) // 2
        text_y = bb['y'] + (bb['h'] + text_h) // 2
        self._put_text(frame, text, text_x, text_y, font_scale, self.COLORS['text_primary'], thickness)
    
    def _detect_cloth_by_color(self, cam_frame):
        """Detect cloth outline using improved HSV colour segmentation.

        Returns
        -------
        (bbox, contour) where bbox is (x1,y1,x2,y2) and contour is the
        simplified polygon, or (None, None) when nothing is detected.
        """
        h, w = cam_frame.shape[:2]
        # Blur before colour conversion: reduces high-frequency noise so the
        # mask edges are cleaner and the contour is less jagged.
        blurred = cv2.GaussianBlur(cam_frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        cfg = self.cloth_color_profiles[self.selected_cloth_color]
        combined = np.zeros((h, w), dtype=np.uint8)
        for lo, hi in cfg['hsv_ranges']:
            combined = cv2.bitwise_or(
                combined,
                cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)))

        # Open first (remove isolated noise specks), then close
        # (fill small holes/gaps inside the cloth region).
        kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel_sm)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_lg)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        largest = max(contours, key=cv2.contourArea)
        # Reject blobs smaller than 5 % of frame area
        if cv2.contourArea(largest) < h * w * 0.05:
            return None, None

        # Simplify contour: removes pixel-level jagginess while preserving
        # the overall cloth silhouette.
        epsilon = 0.008 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        bx, by, bw, bh = cv2.boundingRect(largest)
        return (bx, by, bx + bw, by + bh), approx

    def _run_needle_check(self, cam_frame):
        """Crop the column ROI and run needle.onnx to check needle centring.
        Returns True when the highest-confidence detection centre-X is within
        NEEDLE_CENTER_TOLERANCE pixels of the column's horizontal midpoint."""
        if self.needle_model is None:
            return False
        h, w = cam_frame.shape[:2]
        x1 = max(0, self.ROI_CENTER_X - self.ROI_COL_WIDTH // 2)
        x2 = min(w, self.ROI_CENTER_X + self.ROI_COL_WIDTH // 2)
        y1 = int(self.ROI_TOP_Y)
        y2 = max(y1 + 1, h - int(self.ROI_BOT_MARGIN))
        roi_crop = cam_frame[y1:y2, x1:x2]
        if roi_crop.size == 0:
            return False
        try:
            results = self.needle_model(roi_crop, conf=self.NEEDLE_CONF_THRESHOLD, verbose=False)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                roi_w = x2 - x1
                boxes = results[0].boxes
                confs = boxes.conf.cpu().numpy()
                best  = int(np.argmax(confs))
                xyxy  = boxes[best].xyxy[0].cpu().numpy()
                cx    = (xyxy[0] + xyxy[2]) / 2.0
                return abs(cx - roi_w / 2.0) <= self.NEEDLE_CENTER_TOLERANCE
        except Exception as e:
            print(f"Needle column check error: {e}")
        return False

    def draw_camera_feed(self, frame, camera_frame):
        """Draw camera feed with pattern overlay"""
        # Draw camera frame border with glow effect
        border_pulse = 0.4 + 0.3 * abs(math.sin(self.glow_phase * 0.7))
        self.draw_glow_rect(frame, self.camera_x - 5, self.camera_y - 5, 
                           self.camera_width + 10, self.camera_height + 10, 
                           self.COLORS['bright_blue'], border_pulse)
        
        # Display camera feed or placeholder
        if camera_frame is None:
            cv2.rectangle(frame, (self.camera_x, self.camera_y), 
                         (self.camera_x + self.camera_width, self.camera_y + self.camera_height), 
                         self.COLORS['dark_blue'], -1)
            text = "Camera not available"
            text_scale_value = text_scale(0.84, self.width, self.height, floor=0.74, ceiling=0.96)
            text_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
            text_scale_value = fit_text_scale(text, FONT_MAIN, self.camera_width - 30, text_scale_value, text_thick, min_scale=0.62)
            (tw, th), _ = get_text_size(text, FONT_MAIN, text_scale_value, text_thick)
            tx = self.camera_x + (self.camera_width - tw) // 2
            ty = self.camera_y + (self.camera_height + th) // 2
            self._put_text(frame, text, tx, ty, text_scale_value, self.COLORS['text_primary'], text_thick)
        else:
            cam_frame = cv2.resize(camera_frame, (self.camera_width, self.camera_height))
            detection_frame = cam_frame.copy()
            # Keep raw pre-overlay frame for ROI hole restoration in detection pipeline
            self._last_raw_frame = detection_frame.copy()
            hsv_detection_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2HSV)
            self.stitch_frame_index += 1
            self._update_cloth_motion_state(detection_frame)

            # Draw the detected cloth bounding box directly on the camera
            # feed. If an initial bbox was recorded at START, preserve that
            # size and translate it by the centroid delta so the box remains
            # frozen in size but follows cloth movement.
            try:
                if self.last_cloth_bbox is not None:
                    if self.mask_initial_cloth_bbox is not None:
                        ibx1, iby1, ibx2, iby2 = self.mask_initial_cloth_bbox
                        iw = int(ibx2 - ibx1)
                        ih = int(iby2 - iby1)
                        dx = 0.0
                        dy = 0.0
                        if self.mask_initial_cloth_centroid is not None and self.last_motion_centroid is not None:
                            try:
                                dx = float(self.last_motion_centroid[0]) - float(self.mask_initial_cloth_centroid[0])
                                dy = float(self.last_motion_centroid[1]) - float(self.mask_initial_cloth_centroid[1])
                            except Exception:
                                dx = 0.0; dy = 0.0
                        init_cx = (ibx1 + ibx2) / 2.0
                        init_cy = (iby1 + iby2) / 2.0
                        cx = int(round(init_cx + dx))
                        cy = int(round(init_cy + dy))
                        bx1 = int(round(cx - iw // 2))
                        by1 = int(round(cy - ih // 2))
                        bx2 = bx1 + iw
                        by2 = by1 + ih

                        # Clamp while preserving frozen size when possible
                        if iw >= self.camera_width:
                            bx1 = 0
                            bx2 = self.camera_width
                        else:
                            if bx1 < 0:
                                bx1 = 0
                                bx2 = iw
                            if bx2 > self.camera_width:
                                bx2 = self.camera_width
                                bx1 = self.camera_width - iw

                        if ih >= self.camera_height:
                            by1 = 0
                            by2 = self.camera_height
                        else:
                            if by1 < 0:
                                by1 = 0
                                by2 = ih
                            if by2 > self.camera_height:
                                by2 = self.camera_height
                                by1 = self.camera_height - ih
                    else:
                        bx1, by1, bx2, by2 = self.last_cloth_bbox
                        bw = int(bx2 - bx1)
                        bh = int(by2 - by1)
                        if bw <= 0:
                            bw = 1
                        if bh <= 0:
                            bh = 1

                        if bw >= self.camera_width:
                            bx1 = 0
                            bx2 = self.camera_width
                        else:
                            if bx1 < 0:
                                bx1 = 0
                                bx2 = bw
                            if bx2 > self.camera_width:
                                bx2 = self.camera_width
                                bx1 = self.camera_width - bw

                        if bh >= self.camera_height:
                            by1 = 0
                            by2 = self.camera_height
                        else:
                            if by1 < 0:
                                by1 = 0
                                by2 = bh
                            if by2 > self.camera_height:
                                by2 = self.camera_height
                                by1 = self.camera_height - bh

                    cloth_cfg = self.cloth_color_profiles.get(self.selected_cloth_color, None)
                    if self.last_cloth_mask_ratio >= self.cloth_motion_min_ratio:
                        box_color = (0, 220, 0) if self.cloth_motion_active else (0, 180, 255)
                    else:
                        box_color = (0, 0, 255)

                    if bx2 > bx1 and by2 > by1:
                        cv2.rectangle(cam_frame, (int(bx1), int(by1)), (int(bx2), int(by2)), box_color, 2)
                        try:
                            label = f"{cloth_cfg['label']}: {self.last_cloth_mask_ratio * 100:.0f}%" if cloth_cfg is not None else f"CLOTH: {self.last_cloth_mask_ratio * 100:.0f}%"
                            lbl_scale = text_scale(0.42, self.width, self.height, floor=0.36, ceiling=0.5)
                            lbl_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
                            draw_text(cam_frame, label, int(bx1) + 6, max(int(by1) + 14, 14), lbl_scale, box_color, lbl_thick, font=FONT_MAIN, outline_color=(0, 0, 0), outline_extra=1)
                        except Exception:
                            pass
            except Exception:
                pass

            # For the needle ROI detection we restore a small hole so the
            # detector sees the raw camera pixels. Also compute follow state
            # variables only after cloth update.
            follow_check_ready = False
            follow_on_corridor = True
            follow_order_valid = True
            
            # Load pattern mask
            pattern_overlay, pattern_alpha = self.load_blueprint(self.current_level)

            # Overlay pattern anchored to needle — no cloth detection needed
            # Only show/apply the pattern overlay when the user has
            # explicitly attached the mask to the cloth (START pressed).
            if pattern_overlay is not None and pattern_alpha is not None and self.mask_attached_to_cloth:
                overlay_h, overlay_w = pattern_overlay.shape[:2]

                centerline_path = self._build_centerline_path(pattern_alpha, overlay_w, overlay_h)
                expected_path_idx = None
                if centerline_path is not None and len(centerline_path) > 0:
                    max_idx = len(centerline_path) - 1
                    if not self.centerline_progress_initialized:
                        # Seed once: fresh start → path[0] (now leftmost due to min(xs));
                        # resuming mid-level → seed from raw_progress %.
                        if self.raw_progress <= 0.0:
                            self.centerline_progress_idx = 0
                        else:
                            self.centerline_progress_idx = int(np.clip(
                                round(self.raw_progress / 100.0 * max_idx),
                                0,
                                max_idx,
                            ))
                        self.centerline_progress_initialized = True
                    else:
                        self.centerline_progress_idx = int(np.clip(self.centerline_progress_idx, 0, max_idx))

                    expected_path_idx = self.centerline_progress_idx
                    exp_x = int(centerline_path[expected_path_idx][0])
                    exp_y = int(centerline_path[expected_path_idx][1])
                else:
                    self.centerline_progress_initialized = False
                    exp_x = overlay_w // 2
                    exp_y = overlay_h // 2

                # If mask is attached to cloth, record an initial expected
                # centerline point on first draw so later centerline index
                # advances (from thread-color detections) do not shift the
                # visible overlay. Subsequent draws will use this frozen
                # expected point when computing offsets.
                if self.mask_attached_to_cloth:
                    if self.mask_initial_expected_point is None:
                        try:
                            self.mask_initial_expected_point = (int(exp_x), int(exp_y))
                        except Exception:
                            self.mask_initial_expected_point = (exp_x, exp_y)

                # Anchor pattern to the camera centre (fixed). Compute offsets
                # so the pattern's expected point maps to the camera centre.
                cam_cx = self.camera_width // 2
                cam_cy = self.camera_height // 2
                # On fresh start, move the red-dot to the camera centre so
                # the starting point of the mask is at screen centre.
                if not self.centerline_progress_initialized:
                    # Do not force the needle to the camera centre here;
                    # keep needle tracking free so user can move it independently.
                    pass

                # Determine anchor point for the overlay. New behavior:
                # - On START we record an initial cloth centroid and an initial
                #   anchor (camera centre). The overlay will remain the same
                #   size as when START was pressed and will translate by the
                #   cloth movement delta: anchor = initial_anchor + (curr_centroid - initial_centroid).
                # - If no initial centroid exists, fall back to live centroid
                #   or the explicit mask_anchor.
                anchor_x = cam_cx
                anchor_y = cam_cy
                if self.mask_attached_to_cloth:
                    # Preferred path: if we have an initial cloth centroid,
                    # translate the overlay relative to that seed so size stays fixed.
                    if self.mask_initial_cloth_centroid is not None:
                        base_x, base_y = (self.mask_initial_anchor if self.mask_initial_anchor is not None else (cam_cx, cam_cy))
                        if self.last_motion_centroid is not None:
                            try:
                                dx = float(self.last_motion_centroid[0]) - float(self.mask_initial_cloth_centroid[0])
                                dy = float(self.last_motion_centroid[1]) - float(self.mask_initial_cloth_centroid[1])
                                anchor_x = int(round(base_x + dx))
                                anchor_y = int(round(base_y + dy))
                            except Exception:
                                anchor_x, anchor_y = int(base_x), int(base_y)
                        else:
                            # No live centroid yet: keep the initial anchor
                            try:
                                anchor_x, anchor_y = int(base_x), int(base_y)
                            except Exception:
                                anchor_x, anchor_y = cam_cx, cam_cy
                    else:
                        # Legacy fallback: prefer live centroid, else explicit anchor
                        if self.last_motion_centroid is not None:
                            try:
                                anchor_x = int(round(float(self.last_motion_centroid[0])))
                                anchor_y = int(round(float(self.last_motion_centroid[1])))
                            except Exception:
                                anchor_x, anchor_y = cam_cx, cam_cy
                        elif self.mask_anchor is not None:
                            try:
                                anchor_x, anchor_y = int(self.mask_anchor[0]), int(self.mask_anchor[1])
                            except Exception:
                                anchor_x, anchor_y = cam_cx, cam_cy

                # Do NOT force the needle position to the camera centre here.
                # The needle should be free to move independently while the
                # mask remains anchored to the cloth via `mask_anchor`.

                base_exp_x, base_exp_y = exp_x, exp_y
                if self.mask_attached_to_cloth and self.mask_initial_expected_point is not None:
                    base_exp_x, base_exp_y = self.mask_initial_expected_point

                x_offset = int(round(anchor_x)) - int(round(base_exp_x))
                y_offset = int(round(anchor_y)) - int(round(base_exp_y))
                trace_y = exp_y

                # Allow the pattern overlay to extend off-screen and crop the
                # visible portion; this keeps the expected path aligned to the
                # needle across the entire pattern.
                dst_x1 = max(0, x_offset)
                dst_y1 = max(0, y_offset)
                dst_x2 = min(self.camera_width, x_offset + overlay_w)
                dst_y2 = min(self.camera_height, y_offset + overlay_h)
                visible_w = dst_x2 - dst_x1
                visible_h = dst_y2 - dst_y1

                # Source crop in pattern coordinates for the visible ROI.
                src_x1 = max(0, -x_offset)
                src_y1 = max(0, -y_offset)
                src_x2 = src_x1 + visible_w
                src_y2 = src_y1 + visible_h

                # Validation should use full pattern coordinates.
                actual_w = overlay_w
                actual_h = overlay_h

                if visible_w > 0 and visible_h > 0:
                    corridor_mask = self._get_pattern_corridor_mask(pattern_alpha, actual_w, actual_h)
                    realtime_pattern = self.create_realtime_pattern(pattern_overlay, pattern_alpha, trace_y=trace_y)

                    roi = cam_frame[dst_y1:dst_y2, dst_x1:dst_x2]
                    pattern_src = realtime_pattern if realtime_pattern is not None else pattern_overlay
                    pattern_crop = pattern_src[src_y1:src_y2, src_x1:src_x2]
                    alpha_crop = pattern_alpha[src_y1:src_y2, src_x1:src_x2]
                    
                    for c in range(3):
                        roi[:, :, c] = (alpha_crop * pattern_crop[:, :, c] * 0.7 +
                                      (1 - alpha_crop * 0.7) * roi[:, :, c])
                    cam_frame[dst_y1:dst_y2, dst_x1:dst_x2] = roi

                    # Create a small circular 'hole' at the needle/mask centre by
                    # restoring original camera pixels from the unmodified copy
                    # so the center sample isn't occluded by the overlay.
                    try:
                        hole_r = 6
                        nx = int(round(self.needle_pos_x))
                        ny = int(round(self.needle_pos_y))
                        if dst_x1 <= nx < dst_x2 and dst_y1 <= ny < dst_y2:
                            rx = nx - dst_x1
                            ry = ny - dst_y1
                            y1 = max(0, ry - hole_r)
                            y2 = min(visible_h, ry + hole_r + 1)
                            x1 = max(0, rx - hole_r)
                            x2 = min(visible_w, rx + hole_r + 1)
                            if y2 > y1 and x2 > x1:
                                yy, xx = np.ogrid[y1:y2, x1:x2]
                                circle_mask = ((yy - ry) ** 2 + (xx - rx) ** 2) <= (hole_r * hole_r)
                                src_section = detection_frame[dst_y1 + y1:dst_y1 + y2, dst_x1 + x1:dst_x1 + x2]
                                dst_section = cam_frame[dst_y1 + y1:dst_y1 + y2, dst_x1 + x1:dst_x1 + x2]
                                if src_section.shape == dst_section.shape:
                                    dst_section[circle_mask] = src_section[circle_mask]
                                    cam_frame[dst_y1 + y1:dst_y1 + y2, dst_x1 + x1:dst_x1 + x2] = dst_section
                    except Exception:
                        pass
                    
                    self.run_needle_pipeline(
                        detection_frame, pattern_alpha,
                        x_offset, y_offset, actual_w, actual_h,
                        run_detection=self.needle_detection_enabled,
                        hsv_frame=hsv_detection_frame,
                        expected_trace_y=trace_y,
                        corridor_mask=corridor_mask,
                        centerline_path=centerline_path,
                        expected_path_idx=expected_path_idx
                    )

                    # Always register the red-dot (needle centre) as a stitch candidate.
                    # This makes tracing directly follow expected thread-color detection
                    # at the centre point, even if model detections are noisy/missed.
                    self._register_stitch(
                        self.needle_pos_x, self.needle_pos_y,
                        detection_frame, pattern_alpha,
                        x_offset, y_offset, actual_w, actual_h,
                        source="center", hsv_frame=hsv_detection_frame,
                        expected_trace_y=trace_y,
                        corridor_mask=corridor_mask,
                        centerline_path=centerline_path,
                        expected_path_idx=expected_path_idx
                    )

                    # Use the updated index immediately for follow validation.
                    if centerline_path is not None and len(centerline_path) > 0:
                        expected_path_idx = int(np.clip(self.centerline_progress_idx, 0, len(centerline_path) - 1))

                    # Follow-line decision uses center-point color + corridor + order.
                    center_color_match = self._matches_selected_color(
                        detection_frame, self.needle_pos_x, self.needle_pos_y,
                        hsv_frame=hsv_detection_frame
                    )
                    needle_px = int(np.clip(self.needle_pos_x - x_offset, 0, actual_w - 1))
                    needle_py = int(np.clip(self.needle_pos_y - y_offset, 0, actual_h - 1))
                    follow_valid, follow_on_corridor, follow_order_valid = self._validate_pattern_position(
                        needle_px, needle_py, pattern_alpha, actual_w, actual_h,
                        expected_trace_y=trace_y,
                        corridor_mask=corridor_mask,
                        centerline_path=centerline_path,
                        expected_path_idx=expected_path_idx
                    )
                    self._update_follow_hysteresis(follow_valid, center_color_match)
                    follow_check_ready = True

            # ── Column ROI overlay (wallet-style, needle centring) ────────────
            # When needle detection is disabled, skip drawing the column overlay
            # and the "needle not centred" notice. Still keep `col_x1` defined
            # for downstream layout math.
            col_x1 = 0
            if self.needle_detection_enabled:
                self._needle_check_counter += 1
                if self._needle_check_counter >= self.NEEDLE_CHECK_INTERVAL:
                    self._needle_check_counter = 0
                    self.needle_confirmed = self._run_needle_check(cam_frame)
                col_x1    = max(0, self.ROI_CENTER_X - self.ROI_COL_WIDTH // 2)
                col_x2    = min(self.camera_width, self.ROI_CENTER_X + self.ROI_COL_WIDTH // 2)
                glow_a    = 0.5 + 0.5 * abs(math.sin(self.glow_phase))
                col_color = tuple(int(c * glow_a) for c in self.ROI_COL_COLOR)
                cv2.rectangle(cam_frame, (col_x1, 0), (col_x2, self.camera_height), col_color, 2)
                if not self.needle_confirmed:
                    warn_h  = 44
                    warn_y  = 8
                    _overlay = cam_frame.copy()
                    cv2.rectangle(_overlay, (4, warn_y),
                                  (self.camera_width - 4, warn_y + warn_h),
                                  (0, 60, 180), -1)
                    cv2.addWeighted(_overlay, 0.78, cam_frame, 0.22, 0, cam_frame)
                    warn_txt = "!  SEWING NEEDLE NOT CENTRED"
                    warn_scale, warn_thick = 0.62, 2
                    (ww, wh), _ = cv2.getTextSize(warn_txt, cv2.FONT_HERSHEY_DUPLEX,
                                                   warn_scale, warn_thick)
                    wx = (self.camera_width - ww) // 2
                    wy = warn_y + (warn_h + wh) // 2
                    cv2.putText(cam_frame, warn_txt, (wx + 1, wy + 1), cv2.FONT_HERSHEY_DUPLEX,
                                warn_scale, (0, 0, 0), warn_thick + 2, cv2.LINE_AA)
                    cv2.putText(cam_frame, warn_txt, (wx, wy), cv2.FONT_HERSHEY_DUPLEX,
                                warn_scale, (0, 220, 255), warn_thick, cv2.LINE_AA)
            # ─────────────────────────────────────────────────────────────────

            # Outside-pattern warning (color-gated + hysteresis stabilized)
            if follow_check_ready and self.needle_confirmed and not self.needle_on_pattern and self.raw_progress >= 2.0:
                ow_h = 36
                ow_y = 60 if self.needle_confirmed else 60
                _ov2 = cam_frame.copy()
                cv2.rectangle(_ov2, (4, ow_y), (self.camera_width - 4, ow_y + ow_h), (0, 0, 160), -1)
                cv2.addWeighted(_ov2, 0.78, cam_frame, 0.22, 0, cam_frame)
                if not follow_on_corridor:
                    ow_txt = "!  NEEDLE OUTSIDE PATTERN"
                elif not follow_order_valid:
                    ow_txt = "!  WRONG LINE ORDER"
                else:
                    ow_txt = "!  OFF PATTERN"
                ow_scale, ow_thick = 0.6, 2
                (oww, owh), _ = cv2.getTextSize(ow_txt, cv2.FONT_HERSHEY_DUPLEX, ow_scale, ow_thick)
                owx = (self.camera_width - oww) // 2
                owy = ow_y + (ow_h + owh) // 2
                cv2.putText(cam_frame, ow_txt, (owx + 1, owy + 1), cv2.FONT_HERSHEY_DUPLEX,
                            ow_scale, (0, 0, 0), ow_thick + 2, cv2.LINE_AA)
                cv2.putText(cam_frame, ow_txt, (owx, owy), cv2.FONT_HERSHEY_DUPLEX,
                            ow_scale, (50, 100, 255), ow_thick, cv2.LINE_AA)

            # Mark live selected-color detection at ROI center.
            self._matches_selected_color(detection_frame, self.needle_pos_x, self.needle_pos_y, hsv_frame=hsv_detection_frame)
            color_cfg = self.color_profiles[self.selected_detection_color]
            if self.last_color_match:
                if self.require_motion_for_stitch and not self.cloth_motion_active:
                    marker_text = f"{color_cfg['label']} detected - move cloth"
                    marker_color = (0, 165, 255)
                else:
                    marker_text = f"{color_cfg['label']} detected"
                    marker_color = (0, 220, 0)
            else:
                marker_text = f"No {color_cfg['label'].lower()} ({self.last_color_match_ratio * 100:.0f}%)"
                marker_color = (0, 0, 255)

            # Highlight matched color pixels in the sampled patch.
            if self.last_color_mask is not None and self.last_color_mask_bounds is not None:
                bx1, by1, bx2, by2 = self.last_color_mask_bounds
                mask = self.last_color_mask
                patch_h = by2 - by1
                patch_w = bx2 - bx1
                if patch_h > 0 and patch_w > 0 and mask.shape[0] == patch_h and mask.shape[1] == patch_w:
                    match_pixels = (mask > 0)
                    if np.any(match_pixels):
                        color_highlight = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
                        color_highlight[:, :] = color_cfg['preview_bgr']
                        patch_roi = cam_frame[by1:by2, bx1:bx2]
                        # When the selected color is detected at the center
                        # show a full cyan highlight so the stitch point is
                        # visually unambiguous. Otherwise use a subdued blend.
                        if self.last_color_match:
                            patch_roi[match_pixels] = color_highlight[match_pixels]
                        else:
                            blended = cv2.addWeighted(
                                patch_roi[match_pixels], 0.35, color_highlight[match_pixels], 0.65, 0
                            )
                            if blended is not None:
                                patch_roi[match_pixels] = blended

                        contour_mask = match_pixels.astype(np.uint8) * 255
                        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if cv2.contourArea(cnt) < 4:
                                continue
                            cnt[:, :, 0] += bx1
                            cnt[:, :, 1] += by1
                            cv2.drawContours(cam_frame, [cnt], -1, marker_color, 1)

            if self.needle_detection_enabled:
                cv2.circle(cam_frame, (int(self.needle_pos_x), int(self.needle_pos_y)), 5, marker_color, -1)

            # Draw cloth-color tracking bounding box from full-frame detection.
            if self.last_cloth_bbox is not None:
                # When an initial bbox exists, freeze its size but allow the
                # box to translate with cloth motion. Compute the initial
                # width/height from the seed bbox and move its center by the
                # delta between the current (smoothed) centroid and the
                # initial centroid recorded at START.
                if self.mask_initial_cloth_bbox is not None:
                    ibx1, iby1, ibx2, iby2 = self.mask_initial_cloth_bbox
                    iw = int(ibx2 - ibx1)
                    ih = int(iby2 - iby1)
                    # Determine movement delta using smoothed centroid when available
                    dx = 0.0
                    dy = 0.0
                    if self.mask_initial_cloth_centroid is not None and self.last_motion_centroid is not None:
                        try:
                            dx = float(self.last_motion_centroid[0]) - float(self.mask_initial_cloth_centroid[0])
                            dy = float(self.last_motion_centroid[1]) - float(self.mask_initial_cloth_centroid[1])
                        except Exception:
                            dx = 0.0; dy = 0.0
                    # Compute new center and bbox
                    init_cx = (ibx1 + ibx2) / 2.0
                    init_cy = (iby1 + iby2) / 2.0
                    cx = int(round(init_cx + dx))
                    cy = int(round(init_cy + dy))
                    bx1 = int(round(cx - iw // 2))
                    by1 = int(round(cy - ih // 2))
                    bx2 = bx1 + iw
                    by2 = by1 + ih

                    # Clamp bbox to camera frame while preserving the frozen
                    # size when possible. If the bbox is larger than the
                    # camera dimension, it will be clipped to the full frame.
                    if iw >= self.camera_width:
                        bx1 = 0
                        bx2 = self.camera_width
                    else:
                        if bx1 < 0:
                            bx1 = 0
                            bx2 = iw
                        if bx2 > self.camera_width:
                            bx2 = self.camera_width
                            bx1 = self.camera_width - iw

                    if ih >= self.camera_height:
                        by1 = 0
                        by2 = self.camera_height
                    else:
                        if by1 < 0:
                            by1 = 0
                            by2 = ih
                        if by2 > self.camera_height:
                            by2 = self.camera_height
                            by1 = self.camera_height - ih
                else:
                    bx1, by1, bx2, by2 = self.last_cloth_bbox
                    # Ensure bbox is within camera frame (preserve size when possible)
                    bw = int(bx2 - bx1)
                    bh = int(by2 - by1)
                    if bw <= 0:
                        bw = 1
                    if bh <= 0:
                        bh = 1

                    if bw >= self.camera_width:
                        bx1 = 0
                        bx2 = self.camera_width
                    else:
                        if bx1 < 0:
                            bx1 = 0
                            bx2 = bw
                        if bx2 > self.camera_width:
                            bx2 = self.camera_width
                            bx1 = self.camera_width - bw

                    if bh >= self.camera_height:
                        by1 = 0
                        by2 = self.camera_height
                    else:
                        if by1 < 0:
                            by1 = 0
                            by2 = bh
                        if by2 > self.camera_height:
                            by2 = self.camera_height
                            by1 = self.camera_height - bh
                cloth_cfg = self.cloth_color_profiles[self.selected_cloth_color]
                if self.last_cloth_mask_ratio >= self.cloth_motion_min_ratio:
                    box_color = (0, 220, 0) if self.cloth_motion_active else (0, 180, 255)
                else:
                    box_color = (0, 0, 255)

                # Draw bbox and label on the main UI `frame` using camera offsets
                # so the box can extend beyond the camera crop (e.g., off the
                # bottom). Coordinates outside the image are clipped by OpenCV.
                abs_bx1 = int(round(self.camera_x + bx1))
                abs_by1 = int(round(self.camera_y + by1))
                abs_bx2 = int(round(self.camera_x + bx2))
                abs_by2 = int(round(self.camera_y + by2))
                cv2.rectangle(frame, (abs_bx1, abs_by1), (abs_bx2, abs_by2), box_color, 2)

                cloth_text = (
                    f"CLOTH {cloth_cfg['label']}: {self.last_cloth_mask_ratio * 100:.0f}%"
                )
                cloth_scale = text_scale(0.46, self.width, self.height, floor=0.40, ceiling=0.54)
                cloth_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
                # Text position also in absolute coords
                text_x = max(6 + self.camera_x, abs_bx1)
                text_y = max(14 + self.camera_y, abs_by1 - 6)
                draw_text(
                    frame,
                    cloth_text,
                    text_x,
                    text_y,
                    cloth_scale,
                    box_color,
                    cloth_thick,
                    font=FONT_MAIN,
                    outline_color=(0, 0, 0),
                    outline_extra=1,
                )

            marker_scale = text_scale(0.52, self.width, self.height, floor=0.46, ceiling=0.62)
            marker_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
            marker_scale = fit_text_scale(marker_text, FONT_MAIN, self.camera_width - col_x1 - 6, marker_scale, marker_thick, min_scale=0.4)
            draw_text(
                cam_frame,
                marker_text,
                col_x1,
                self.camera_height - 8,
                marker_scale,
                marker_color,
                marker_thick,
                font=FONT_MAIN,
                outline_color=(0, 0, 0),
                outline_extra=1,
            )
            
            frame[self.camera_y:self.camera_y+self.camera_height, 
                  self.camera_x:self.camera_x+self.camera_width] = cam_frame
            
            # Draw warning overlay if out of segment
            if self.out_of_segment_warning:
                self.draw_warning_overlay(frame)
            
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
    
    def draw_pattern_legend(self, frame):
        """Draw legend (Completed / To Sew) below the confidence panel on the left side."""
        legend_x = 18
        row_y = 576

        legend_items = [
            ("Completed", self.segment_colors['completed']),
            ("To Sew",    self.segment_colors['current']),
        ]
        box_size = 12
        label_scale = text_scale(0.50, self.width, self.height, floor=0.44, ceiling=0.58)
        label_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
        item_spacing = 98

        for i, (label, color) in enumerate(legend_items):
            item_x = legend_x + i * item_spacing
            cv2.rectangle(frame, (item_x, row_y - box_size + 2),
                          (item_x + box_size, row_y + 2), color, -1)
            cv2.rectangle(frame, (item_x, row_y - box_size + 2),
                          (item_x + box_size, row_y + 2), self.COLORS['medium_blue'], 1)
            self._put_text(frame, label, item_x + box_size + 5, row_y,
                           label_scale, self.COLORS['text_secondary'], label_thick)

    def draw_combined_color_panel(self, frame):
        """Draw DETECT COLOR + CLOTH COLOR in a single panel on the left side."""
        x = self.color_panel_x
        y = self.color_panel_y
        w = self.color_panel_width
        h = self.color_panel_height

        pulse = 0.35 + 0.25 * abs(math.sin(self.glow_phase * 0.9))
        self.draw_glow_rect(frame, x, y, w, h, self.COLORS['bright_blue'], pulse)

        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay, (x + 2, y + 2), (x + w - 2, y + h - 2), self.COLORS['dark_blue'], -1)
        cv2.addWeighted(panel_overlay, 0.82, frame, 0.18, 0, frame)

        header_scale = text_scale(0.5, self.width, self.height, floor=0.46, ceiling=0.58)
        header_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)

        # ── DETECT COLOR ─────────────────────────────────────────────────
        self._put_text(frame, "DETECT COLOR", x + 14, y + 22, header_scale, self.COLORS['text_secondary'], header_thick)

        detect_labels = ['white', 'yellow', 'red']
        d_btn_w = 56
        d_btn_h = 28
        d_gap = 7
        d_total_w = len(detect_labels) * d_btn_w + (len(detect_labels) - 1) * d_gap
        d_start_x = x + (w - d_total_w) // 2
        d_btn_y = y + 28

        self.color_buttons = {}
        for idx, key in enumerate(detect_labels):
            bx = d_start_x + idx * (d_btn_w + d_gap)
            by = d_btn_y
            is_selected = key == self.selected_detection_color
            cfg = self.color_profiles[key]

            border = self.COLORS['glow_cyan'] if is_selected else self.COLORS['medium_blue']
            cv2.rectangle(frame, (bx, by), (bx + d_btn_w, by + d_btn_h), border, 2)
            fill = frame.copy()
            fill_alpha = 0.6 if is_selected else 0.35
            cv2.rectangle(fill, (bx + 2, by + 2), (bx + d_btn_w - 2, by + d_btn_h - 2), self.COLORS['button_normal'], -1)
            cv2.addWeighted(fill, fill_alpha, frame, 1 - fill_alpha, 0, frame)
            cv2.circle(frame, (bx + 10, by + d_btn_h // 2), 5, cfg['preview_bgr'], -1)
            cv2.circle(frame, (bx + 10, by + d_btn_h // 2), 5, self.COLORS['text_primary'], 1)
            letter_scale = text_scale(0.54, self.width, self.height, floor=0.48, ceiling=0.62)
            letter_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
            self._put_text(frame, cfg['label'][0], bx + 21, by + 20, letter_scale, self.COLORS['text_primary'], letter_thick)
            self.color_buttons[key] = {'x': bx, 'y': by, 'w': d_btn_w, 'h': d_btn_h}

        # ── Divider ───────────────────────────────────────────────────────
        div_y = d_btn_y + d_btn_h + 9
        cv2.line(frame, (x + 12, div_y), (x + w - 12, div_y), self.COLORS['medium_blue'], 1)

        # ── CLOTH COLOR ───────────────────────────────────────────────────
        cloth_header_y = div_y + 16
        self._put_text(frame, "CLOTH COLOR", x + 14, cloth_header_y, header_scale, self.COLORS['text_secondary'], header_thick)

        cloth_labels = list(self.cloth_color_profiles.keys())  # red, black, white, gray
        c_btn_w = 42
        c_btn_h = 28
        c_gap = 4
        c_total_w = len(cloth_labels) * c_btn_w + (len(cloth_labels) - 1) * c_gap
        c_start_x = x + (w - c_total_w) // 2
        c_btn_y = cloth_header_y + 8

        self.cloth_color_buttons = {}
        for idx, key in enumerate(cloth_labels):
            bx = c_start_x + idx * (c_btn_w + c_gap)
            by = c_btn_y
            is_selected = key == self.selected_cloth_color
            cfg = self.cloth_color_profiles[key]

            border = self.COLORS['glow_cyan'] if is_selected else self.COLORS['medium_blue']
            cv2.rectangle(frame, (bx, by), (bx + c_btn_w, by + c_btn_h), border, 2)
            fill = frame.copy()
            fill_alpha = 0.6 if is_selected else 0.35
            cv2.rectangle(fill, (bx + 2, by + 2), (bx + c_btn_w - 2, by + c_btn_h - 2), self.COLORS['button_normal'], -1)
            cv2.addWeighted(fill, fill_alpha, frame, 1 - fill_alpha, 0, frame)
            cv2.circle(frame, (bx + 10, by + c_btn_h // 2), 5, cfg['preview_bgr'], -1)
            cv2.circle(frame, (bx + 10, by + c_btn_h // 2), 5, self.COLORS['text_primary'], 1)
            cloth_scale = text_scale(0.40, self.width, self.height, floor=0.36, ceiling=0.46)
            cloth_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
            # First 2 chars so buttons stay narrow (RE / BL / WH / GR)
            short = cfg['label'][:2]
            self._put_text(frame, short, bx + 18, by + 20, cloth_scale, self.COLORS['text_primary'], cloth_thick)
            self.cloth_color_buttons[key] = {'x': bx, 'y': by, 'w': c_btn_w, 'h': c_btn_h}

        # ── Conflict notice ───────────────────────────────────────────────
        notice_y = c_btn_y + c_btn_h + 14
        if self.selected_detection_color == self.selected_cloth_color:
            flash = 0.65 + 0.35 * abs(math.sin(self.glow_phase * 3))
            warn_color = tuple(int(c * flash) for c in (0, 100, 255))  # orange
            notice_scale = text_scale(0.44, self.width, self.height, floor=0.40, ceiling=0.52)
            notice_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
            notice_text = "! DETECT = CLOTH COLOR"
            (nw, _), _ = get_text_size(notice_text, FONT_MAIN, notice_scale, notice_thick)
            self._put_text(frame, notice_text, x + (w - nw) // 2, notice_y, notice_scale, warn_color, notice_thick)

    def draw_cloth_color_selector(self, frame):
        """Merged into draw_combined_color_panel — kept for compatibility."""
        pass

    def draw_thread_color_panel(self, frame):
        """Draw THREAD (detect) COLOR panel on the left side (4 buttons, 2 per row)."""
        x = self.color_panel_x
        y = self.color_panel_y
        w = self.color_panel_width
        h = self.color_panel_height

        pulse = 0.35 + 0.25 * abs(math.sin(self.glow_phase * 0.9))
        self.draw_glow_rect(frame, x, y, w, h, self.COLORS['bright_blue'], pulse)

        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay, (x + 2, y + 2), (x + w - 2, y + h - 2), self.COLORS['dark_blue'], -1)
        cv2.addWeighted(panel_overlay, 0.82, frame, 0.18, 0, frame)

        header_scale = text_scale(0.5, self.width, self.height, floor=0.46, ceiling=0.58)
        header_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
        self._put_text(frame, "THREAD COLOR", x + 14, y + 20, header_scale, self.COLORS['text_secondary'], header_thick)

        rows = [['white', 'yellow'], ['red', 'black']]
        btn_w = 80
        btn_h = 30
        gap = 8
        row_gap = 7
        start_x = x + (w - 2 * btn_w - gap) // 2

        self.color_buttons = {}
        for row_idx, row_keys in enumerate(rows):
            btn_y = y + 30 + row_idx * (btn_h + row_gap)
            for col_idx, key in enumerate(row_keys):
                bx = start_x + col_idx * (btn_w + gap)
                by = btn_y
                is_selected = key == self.selected_detection_color
                cfg = self.color_profiles[key]

                border = self.COLORS['glow_cyan'] if is_selected else self.COLORS['medium_blue']
                cv2.rectangle(frame, (bx, by), (bx + btn_w, by + btn_h), border, 2)
                fill = frame.copy()
                fill_alpha = 0.6 if is_selected else 0.35
                cv2.rectangle(fill, (bx + 2, by + 2), (bx + btn_w - 2, by + btn_h - 2), self.COLORS['button_normal'], -1)
                cv2.addWeighted(fill, fill_alpha, frame, 1 - fill_alpha, 0, frame)
                cv2.circle(frame, (bx + 10, by + btn_h // 2), 5, cfg['preview_bgr'], -1)
                cv2.circle(frame, (bx + 10, by + btn_h // 2), 5, self.COLORS['text_primary'], 1)
                letter_scale = text_scale(0.48, self.width, self.height, floor=0.42, ceiling=0.56)
                letter_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
                self._put_text(frame, cfg['label'][:2], bx + 18, by + 21, letter_scale, self.COLORS['text_primary'], letter_thick)
                self.color_buttons[key] = {'x': bx, 'y': by, 'w': btn_w, 'h': btn_h}

    def draw_cloth_color_panel(self, frame):
        """Draw CLOTH COLOR panel on the left side (4 buttons, 2 per row). Disabled if same as thread color."""
        x = self.cloth_color_panel_x
        y = self.cloth_color_panel_y
        w = self.cloth_color_panel_width
        h = self.cloth_color_panel_height

        pulse = 0.35 + 0.25 * abs(math.sin(self.glow_phase * 0.9))
        self.draw_glow_rect(frame, x, y, w, h, self.COLORS['bright_blue'], pulse)

        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay, (x + 2, y + 2), (x + w - 2, y + h - 2), self.COLORS['dark_blue'], -1)
        cv2.addWeighted(panel_overlay, 0.82, frame, 0.18, 0, frame)

        header_scale = text_scale(0.5, self.width, self.height, floor=0.46, ceiling=0.58)
        header_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
        self._put_text(frame, "CLOTH COLOR", x + 14, y + 20, header_scale, self.COLORS['text_secondary'], header_thick)

        rows = [['red', 'black'], ['white', 'gray']]
        btn_w = 80
        btn_h = 30
        gap = 8
        row_gap = 7
        start_x = x + (w - 2 * btn_w - gap) // 2

        self.cloth_color_buttons = {}
        for row_idx, row_keys in enumerate(rows):
            btn_y = y + 30 + row_idx * (btn_h + row_gap)
            for col_idx, key in enumerate(row_keys):
                bx = start_x + col_idx * (btn_w + gap)
                by = btn_y
                is_selected = key == self.selected_cloth_color
                is_disabled = key == self.selected_detection_color
                cfg = self.cloth_color_profiles[key]

                if is_disabled:
                    border = (60, 60, 60)
                elif is_selected:
                    border = self.COLORS['glow_cyan']
                else:
                    border = self.COLORS['medium_blue']
                cv2.rectangle(frame, (bx, by), (bx + btn_w, by + btn_h), border, 2)
                fill = frame.copy()
                fill_alpha = 0.15 if is_disabled else (0.6 if is_selected else 0.35)
                cv2.rectangle(fill, (bx + 2, by + 2), (bx + btn_w - 2, by + btn_h - 2), self.COLORS['button_normal'], -1)
                cv2.addWeighted(fill, fill_alpha, frame, 1 - fill_alpha, 0, frame)
                dot_color = (50, 50, 50) if is_disabled else cfg['preview_bgr']
                cv2.circle(frame, (bx + 10, by + btn_h // 2), 5, dot_color, -1)
                cv2.circle(frame, (bx + 10, by + btn_h // 2), 5, (80, 80, 80) if is_disabled else self.COLORS['text_primary'], 1)
                text_color = (70, 70, 70) if is_disabled else self.COLORS['text_primary']
                cloth_scale = text_scale(0.48, self.width, self.height, floor=0.42, ceiling=0.56)
                cloth_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
                self._put_text(frame, cfg['label'][:2], bx + 18, by + 21, cloth_scale, text_color, cloth_thick)
                self.cloth_color_buttons[key] = {'x': bx, 'y': by, 'w': btn_w, 'h': btn_h, 'disabled': is_disabled}

        # Needle detection toggle (small control below cloth color buttons)
        try:
            toggle_w = min(120, w - 24)
            toggle_h = 28
            toggle_x = x + (w - toggle_w) // 2
            toggle_y = y + 30 + 2 * (btn_h + row_gap)  - 8
            # Draw border
            tborder = self.COLORS['medium_blue'] if self.needle_detection_enabled else (80, 80, 80)
            cv2.rectangle(frame, (toggle_x, toggle_y), (toggle_x + toggle_w, toggle_y + toggle_h), tborder, 2)
            fill = frame.copy()
            fill_alpha = 0.45 if self.needle_detection_enabled else 0.18
            cv2.rectangle(fill, (toggle_x + 2, toggle_y + 2), (toggle_x + toggle_w - 2, toggle_y + toggle_h - 2), self.COLORS['button_normal'], -1)
            cv2.addWeighted(fill, fill_alpha, frame, 1 - fill_alpha, 0, frame)

            # Label
            lbl = "NEEDLE DETECTION: ON" if self.needle_detection_enabled else "NEEDLE DETECTION: OFF"
            lbl_color = (0, 220, 0) if self.needle_detection_enabled else (180, 180, 180)
            lbl_scale = text_scale(0.40, self.width, self.height, floor=0.36, ceiling=0.5)
            lbl_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
            self._put_text(frame, lbl, toggle_x + 8, toggle_y + 19, lbl_scale, lbl_color, lbl_thick)

            self.needle_toggle_button = {'x': toggle_x, 'y': toggle_y, 'w': toggle_w, 'h': toggle_h}
        except Exception:
            self.needle_toggle_button = None

    def draw_confidence_controls(self, frame):
        """Draw clickable controls for confidence threshold."""
        x = self.conf_panel_x
        y = self.conf_panel_y
        w = self.conf_panel_width
        h = self.conf_panel_height

        pulse = 0.35 + 0.25 * abs(math.sin(self.glow_phase * 1.1))
        self.draw_glow_rect(frame, x, y, w, h, self.COLORS['bright_blue'], pulse)

        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay, (x + 2, y + 2), (x + w - 2, y + h - 2), self.COLORS['dark_blue'], -1)
        cv2.addWeighted(panel_overlay, 0.82, frame, 0.18, 0, frame)

        header_scale = text_scale(0.5, self.width, self.height, floor=0.46, ceiling=0.58)
        header_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
        self._put_text(frame, "CONFIDENCE", x + 16, y + 20, header_scale, self.COLORS['text_secondary'], header_thick)

        btn_w = 40
        btn_h = 30
        btn_y = y + 27

        self.conf_minus_button = {'x': x + 12, 'y': btn_y, 'w': btn_w, 'h': btn_h}
        self.conf_plus_button = {'x': x + w - 12 - btn_w, 'y': btn_y, 'w': btn_w, 'h': btn_h}

        for btn, label in ((self.conf_minus_button, '-'), (self.conf_plus_button, '+')):
            cv2.rectangle(frame, (btn['x'], btn['y']), (btn['x'] + btn['w'], btn['y'] + btn['h']), self.COLORS['medium_blue'], 2)
            fill = frame.copy()
            cv2.rectangle(fill, (btn['x'] + 2, btn['y'] + 2), (btn['x'] + btn['w'] - 2, btn['y'] + btn['h'] - 2), self.COLORS['button_normal'], -1)
            cv2.addWeighted(fill, 0.6, frame, 0.4, 0, frame)
            sign_scale = text_scale(0.7, self.width, self.height, floor=0.62, ceiling=0.8)
            sign_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
            self._put_text(frame, label, btn['x'] + 14, btn['y'] + 22, sign_scale, self.COLORS['text_primary'], sign_thick)

        conf_text = f"{self.confidence_threshold:.2f}"
        conf_scale = text_scale(0.64, self.width, self.height, floor=0.56, ceiling=0.74)
        conf_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        (tw, _), _ = get_text_size(conf_text, FONT_MAIN, conf_scale, conf_thick)
        self._put_text(frame, conf_text, x + (w - tw) // 2, btn_y + 22, conf_scale, self.COLORS['text_primary'], conf_thick)
    
    def draw_guide_overlay(self, frame):
        """Draw game guide/tutorial overlay - multi-step"""
        # Dark semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), 
                     (20, 10, 5), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Guide panel
        panel_w = min(650, self.width - 120)
        panel_h = min(420, self.height - 90)
        panel_x = (self.width - panel_w) // 2
        panel_y = (self.height - panel_h) // 2
        
        # Panel border with glow
        pulse = 0.6 + 0.4 * abs(math.sin(self.glow_phase * 0.8))
        for i in range(3):
            offset = i * 3
            alpha = pulse * (1 - i * 0.3)
            glow_color = tuple(int(c * alpha) for c in self.COLORS['glow_cyan'])
            cv2.rectangle(frame, (panel_x - offset, panel_y - offset), 
                         (panel_x + panel_w + offset, panel_y + panel_h + offset), 
                         glow_color, 3)
        
        # Panel background
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     self.COLORS['dark_blue'], -1)
        
        # Title
        title = "HOW TO PLAY"
        title_scale = text_scale(1.3, self.width, self.height, floor=1.12, ceiling=1.45)
        title_thickness = text_thickness(3, self.width, self.height, min_thickness=2, max_thickness=4)
        title_scale = fit_text_scale(title, FONT_DISPLAY, panel_w - 80, title_scale, title_thickness, min_scale=1.0)
        (title_w, _), _ = get_text_size(title, FONT_DISPLAY, title_scale, title_thickness)
        title_x = panel_x + (panel_w - title_w) // 2
        title_y = panel_y + 52
        draw_text(
            frame,
            title,
            title_x,
            title_y,
            title_scale,
            self.COLORS['bright_blue'],
            title_thickness,
            font=FONT_DISPLAY,
            outline_color=self.COLORS['glow_cyan'],
            outline_extra=2,
        )
        
        # Step indicator
        step_text = f"Step {self.guide_step} of 4"
        step_scale = text_scale(0.66, self.width, self.height, floor=0.58, ceiling=0.76)
        step_thickness = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        (step_w, step_h), _ = get_text_size(step_text, FONT_MAIN, step_scale, step_thickness)
        step_x = panel_x + panel_w - step_w - 30
        step_y = panel_y + 45
        self._put_text(frame, step_text, step_x, step_y, step_scale, self.COLORS['text_secondary'], step_thickness)
        
        # Content based on current step
        content_x = panel_x + 40
        content_y = title_y + 58
        body_scale_base = text_scale(0.72, self.width, self.height, floor=0.64, ceiling=0.82)
        body_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        (_, body_h), _ = get_text_size("Ag", FONT_MAIN, body_scale_base, body_thick)
        line_height = max(34, body_h + 14)
        
        if self.guide_step == 1:
            instructions = [
                ("REAL-TIME COLORING", self.COLORS['text_primary'], 0.85, 2),
                "",
                ("As you sew, the pattern will", self.COLORS['text_secondary'], 0.7, 2),
                ("change color in real-time.", self.COLORS['text_secondary'], 0.7, 2),
                "",
                ("Start anywhere on the pattern", self.COLORS['text_secondary'], 0.7, 2),
                ("and sew freely.", self.COLORS['text_secondary'], 0.7, 2),
            ]
        elif self.guide_step == 2:
            instructions = [
                ("COLOR GUIDE", self.COLORS['text_primary'], 0.85, 2),
                "",
                ("CYAN = Completed stitches", self.COLORS['text_secondary'], 0.7, 2),
                "",
                ("YELLOW = Pattern to sew", self.COLORS['text_secondary'], 0.7, 2),
                "",
                ("The pattern progressively turns cyan!", self.COLORS['text_secondary'], 0.7, 2),
            ]
        elif self.guide_step == 3:
            instructions = [
                ("PROGRESSIVE FEEDBACK", self.COLORS['text_primary'], 0.85, 2),
                "",
                ("Only stitches near your completed", self.COLORS['text_secondary'], 0.7, 2),
                ("work will change to cyan.", self.COLORS['text_secondary'], 0.7, 2),
                "",
                ("Watch the pattern transform as", self.COLORS['text_secondary'], 0.7, 2),
                ("you progress!", self.COLORS['text_secondary'], 0.7, 2),
            ]
        else:  # step 4
            instructions = [
                ("EVALUATION", self.COLORS['text_primary'], 0.85, 2),
                "",
                ("Click the EVALUATE button when", self.COLORS['text_secondary'], 0.7, 2),
                ("you finish sewing.", self.COLORS['text_secondary'], 0.7, 2),
                "",
                ("You need 80% or more progress", self.COLORS['text_secondary'], 0.7, 2),
                ("to complete the level.", self.COLORS['text_secondary'], 0.7, 2),
            ]
        
        y_pos = content_y
        for instruction in instructions:
            if instruction == "":
                y_pos += line_height // 2
                continue
            
            text, color, base_scale, base_thickness = instruction
            line_scale = text_scale(base_scale, self.width, self.height, floor=max(0.58, base_scale * 0.86), ceiling=base_scale * 1.12)
            line_thick = max(1, text_thickness(base_thickness, self.width, self.height, min_thickness=1, max_thickness=3))
            line_scale = fit_text_scale(text, FONT_MAIN, panel_w - 80, line_scale, line_thick, min_scale=max(0.52, base_scale * 0.78))
            self._put_text(frame, text, content_x, y_pos, line_scale, color, line_thick)
            y_pos += line_height
        
        # Button ("Next" or "Got it!") - bottom right corner
        button_text = "GOT IT!" if self.guide_step == 4 else "NEXT"
        button_w = 140
        button_h = 45
        button_x = panel_x + panel_w - button_w - 30  # 30px margin from right edge
        button_y = panel_y + panel_h - button_h - 25  # 25px margin from bottom
        
        # Update button coords for click detection
        self.guide_button['x'] = button_x
        self.guide_button['y'] = button_y
        self.guide_button['w'] = button_w
        self.guide_button['h'] = button_h
        
        # Button glow
        button_pulse = 0.6 + 0.4 * abs(math.sin(self.glow_phase * 1.5))
        self.draw_glow_rect(frame, button_x, button_y, button_w, button_h, 
                           self.COLORS['glow_cyan'], button_pulse)
        
        # Button background
        btn_overlay = frame.copy()
        cv2.rectangle(btn_overlay, (button_x + 2, button_y + 2), 
                     (button_x + button_w - 2, button_y + button_h - 2), 
                     self.COLORS['button_hover'], -1)
        cv2.addWeighted(btn_overlay, 0.9, frame, 0.1, 0, frame)
        
        # Button text (smaller font)
        btn_scale = text_scale(0.84, self.width, self.height, floor=0.74, ceiling=0.96)
        btn_thickness = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
        btn_scale = fit_text_scale(button_text, FONT_DISPLAY, button_w - 16, btn_scale, btn_thickness, min_scale=0.66)
        (btn_text_w, btn_text_h), _ = get_text_size(button_text, FONT_DISPLAY, btn_scale, btn_thickness)
        btn_text_x = button_x + (button_w - btn_text_w) // 2
        btn_text_y = button_y + (button_h + btn_text_h) // 2
        draw_text(
            frame,
            button_text,
            btn_text_x,
            btn_text_y,
            btn_scale,
            self.COLORS['bright_blue'],
            btn_thickness,
            font=FONT_DISPLAY,
            outline_color=self.COLORS['glow_cyan'],
            outline_extra=2,
        )
    
    def draw_warning_overlay(self, frame):
        """Draw warning overlay when stitching outside current segment"""
        # Flash effect
        flash_alpha = 0.3 + 0.3 * abs(math.sin(self.warning_flash_phase))
        
        # Warning banner at top of camera
        banner_y = self.camera_y + 30
        banner_height = 70
        
        # Semi-transparent red background
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (self.camera_x, banner_y), 
                     (self.camera_x + self.camera_width, banner_y + banner_height), 
                     (0, 0, 200), -1)  # Red in BGR
        cv2.addWeighted(overlay, flash_alpha, frame, 1 - flash_alpha, 0, frame)
        
        # Warning text - larger and more readable
        font_scale = text_scale(1.02, self.width, self.height, floor=0.88, ceiling=1.15)
        thickness = text_thickness(4, self.width, self.height, min_thickness=3, max_thickness=5)
        font_scale = fit_text_scale(self.warning_message, FONT_DISPLAY, self.camera_width - 30, font_scale, thickness, min_scale=0.72)
        (text_w, text_h), _ = get_text_size(self.warning_message, FONT_DISPLAY, font_scale, thickness)
        text_x = self.camera_x + (self.camera_width - text_w) // 2
        text_y = banner_y + (banner_height + text_h) // 2
        
        # Draw text with outline for visibility
        draw_text(
            frame,
            self.warning_message,
            text_x,
            text_y,
            font_scale,
            (255, 255, 255),
            thickness,
            font=FONT_DISPLAY,
            outline_color=(0, 0, 0),
            outline_extra=3,
        )
    
    def update_game_stats(self, detected_stitch_masks, pattern_alpha, x_offset, y_offset, actual_w, actual_h):
        """Update game statistics based on detected stitches vs pattern (ROI-optimized)"""
        if len(detected_stitch_masks) == 0:
            self.out_of_segment_warning = False
            return
        
        # Initialize completed stitch mask if needed (in pattern coordinates)
        if self.completed_stitch_mask is None:
            self.completed_stitch_mask = np.zeros((self.uniform_height, self.uniform_width), dtype=np.uint8)
        
        # Create combined stitch mask in ROI coordinates only (avoid full-frame allocation)
        # We'll work directly with ROI-sized masks
        roi_combined_mask = None
        roi_bounds = None
        
        for stitch_data in detected_stitch_masks:
            if stitch_data.get('roi_bounds') is not None:
                # ROI-based mask
                roi_bounds = stitch_data['roi_bounds']
                roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
                
                if roi_combined_mask is None:
                    roi_combined_mask = np.zeros((roi_y2 - roi_y1, roi_x2 - roi_x1), dtype=np.uint8)
                
                roi_combined_mask = np.maximum(roi_combined_mask, stitch_data['mask'])
            else:
                # Full-frame fallback (shouldn't happen with ROI optimization)
                # Convert to ROI for consistency
                if roi_combined_mask is None:
                    roi_combined_mask = np.zeros((actual_h, actual_w), dtype=np.uint8)
                    roi_bounds = (x_offset, y_offset, x_offset + actual_w, y_offset + actual_h)
                
                roi_mask = stitch_data['mask'][y_offset:y_offset+actual_h, x_offset:x_offset+actual_w]
                roi_combined_mask = np.maximum(roi_combined_mask, roi_mask)
        
        if roi_combined_mask is None:
            return
        
        # Dilate the detected stitch mask to be more forgiving (smaller dilation)
        kernel = np.ones((3, 3), np.uint8)
        roi_combined_mask = cv2.dilate(roi_combined_mask, kernel, iterations=1)
        
        # Get pattern mask in ROI coordinates (not full camera frame)
        pattern_crop = (pattern_alpha[0:actual_h, 0:actual_w] * 255).astype(np.uint8)
        pattern_mask_roi = (pattern_crop > 128).astype(np.uint8)
        
        # Ensure masks are same size
        if roi_combined_mask.shape != pattern_mask_roi.shape:
            roi_combined_mask = cv2.resize(roi_combined_mask, (actual_w, actual_h))
        
        # Extract stitches that are within the pattern area (all in ROI coordinates)
        stitches_in_pattern = np.logical_and(roi_combined_mask > 0, pattern_mask_roi > 0)
        
        # Convert detected stitches in pattern to pattern coordinates and accumulate
        if np.sum(stitches_in_pattern) > 0:
            # Resize to pattern dimensions (from ROI to pattern coords)
            stitches_pattern_coords = cv2.resize(roi_combined_mask, (self.uniform_width, self.uniform_height))
            
            # Accumulate into completed mask (using maximum to not lose previous stitches)
            self.completed_stitch_mask = np.maximum(self.completed_stitch_mask, 
                                                     (stitches_pattern_coords > 0).astype(np.uint8) * 255)
        
        # No more segment-based warnings - all pattern area is valid now
        self.out_of_segment_warning = False
        
        # Calculate overlap (intersection) and union (all in ROI coordinates)
        intersection = np.logical_and(roi_combined_mask > 0, pattern_mask_roi > 0)
        union = np.logical_or(roi_combined_mask > 0, pattern_mask_roi > 0)
        
        # Calculate IoU (Intersection over Union) as accuracy metric
        intersection_pixels = np.sum(intersection)
        union_pixels = np.sum(union)
        
        if union_pixels > 0:
            accuracy = (intersection_pixels / union_pixels) * 100.0
            self.current_accuracy = accuracy
            
            # Update score (1 point per % accuracy)
            self.total_score = int(self.current_accuracy)
            
            # Calculate progress (how much of pattern is covered by stitches)
            pattern_pixels = np.sum(pattern_mask_roi > 0)
            if pattern_pixels > 0:
                covered_pixels = np.sum(np.logical_and(roi_combined_mask > 0, pattern_mask_roi > 0))
                raw_progress = min(100.0, (covered_pixels / pattern_pixels) * 100.0)
                
                # Store raw progress for evaluation
                self.raw_progress = raw_progress
                print(f"📊 Raw Progress: {raw_progress:.1f}%")  # Debug print
                
                # Determine segment based on raw progress (much lower thresholds)
                if raw_progress >= 70:  # 70%+ completes everything
                    new_segment = 5  # Beyond segment 4, means fully complete
                    new_progress = 100.0
                elif raw_progress >= 45:  # 45%+ completes segment 3, working on segment 4
                    new_segment = 4
                    new_progress = 75.0
                elif raw_progress >= 25:  # 25%+ completes segment 2, working on segment 3
                    new_segment = 3
                    new_progress = 50.0
                elif raw_progress >= 12:  # 12%+ completes segment 1, working on segment 2
                    new_segment = 2
                    new_progress = 25.0
                else:  # < 12% still working on segment 1
                    new_segment = 1
                    new_progress = 0.0
                
                # Only allow progress forward, never backwards (prevent flickering)
                if new_segment > self.highest_segment_reached:
                    self.highest_segment_reached = new_segment
                    self.current_segment = new_segment if new_segment <= 4 else 4
                    self.pattern_progress = new_progress
                    # Auto-hide mask only when progress truly reaches 100%
                    try:
                        if (self.raw_progress >= 100.0 or self.pattern_progress >= 100.0) and self.mask_attached_to_cloth:
                            self.mask_attached_to_cloth = False
                            self.mask_anchor = None
                            self.mask_initial_anchor = None
                            self.mask_initial_cloth_centroid = None
                            self.mask_initial_cloth_bbox = None
                            self.mask_initial_expected_point = None
                            print("📌 Mask hidden: pattern complete (100%)")
                    except Exception:
                        pass
                # Keep the highest values reached
                elif self.highest_segment_reached > 1:
                    # Stay at the highest segment reached
                    if self.highest_segment_reached == 5:
                        self.current_segment = 4
                        self.pattern_progress = 100.0
                    else:
                        self.current_segment = self.highest_segment_reached
                        self.pattern_progress = (self.highest_segment_reached - 1) * 25.0
            
            # Update combo system
            if accuracy > 70:  # Good accuracy threshold
                self.current_combo += 1
                self.max_combo = max(self.max_combo, self.current_combo)
            else:
                self.current_combo = 0
        
        # Update stitch count
        self.stitches_detected = len(detected_stitch_masks)
    
    def draw_score_panel(self, frame):
        """Draw the score/stats panel on the right side — includes progress, legend, detected color."""
        x, y, w, h = self.score_panel_x, self.score_panel_y, self.score_panel_width, self.score_panel_height
        
        # Panel border with glow
        pulse = 0.4 + 0.3 * abs(math.sin(self.glow_phase * 0.7))
        self.draw_glow_rect(frame, x, y, w, h, self.COLORS['bright_blue'], pulse)
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x + 3, y + 3), (x + w - 3, y + h - 3), self.COLORS['dark_blue'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Title
        title_font_scale = text_scale(0.92, self.width, self.height, floor=0.82, ceiling=1.05)
        title_thickness = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
        title_text = "STATS"
        (title_w, title_h), _ = get_text_size(title_text, FONT_DISPLAY, title_font_scale, title_thickness)
        title_x = x + (w - title_w) // 2
        title_y = y + 38
        draw_text(frame, title_text, title_x, title_y, title_font_scale,
                  self.COLORS['bright_blue'], title_thickness, font=FONT_DISPLAY,
                  outline_color=self.COLORS['glow_cyan'], outline_extra=2)
        
        cv2.line(frame, (x + 15, title_y + 15), (x + w - 15, title_y + 15), self.COLORS['medium_blue'], 2)
        
        content_x = x + 18
        label_scale = text_scale(0.56, self.width, self.height, floor=0.50, ceiling=0.66)
        label_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)

        # ── Progress bar ─────────────────────────────────────────────────────
        progress_y = title_y + 48
        self._put_text(frame, "PROGRESS", content_x, progress_y, label_scale, self.COLORS['text_secondary'], label_thick)
        bar_y = progress_y + 14
        bar_width = w - 36
        bar_height = 22
        cv2.rectangle(frame, (content_x, bar_y), (content_x + bar_width, bar_y + bar_height), (30, 20, 10), -1)
        fill_w = int(bar_width * max(0.0, min(1.0, self.raw_progress / 100.0)))
        if fill_w > 0:
            ov_bar = frame.copy()
            cv2.rectangle(ov_bar, (content_x + 2, bar_y + 2),
                          (content_x + fill_w - 2, bar_y + bar_height - 2),
                          self.segment_colors['completed'], -1)
            cv2.addWeighted(ov_bar, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (content_x, bar_y), (content_x + bar_width, bar_y + bar_height),
                      self.COLORS['medium_blue'], 2)
        pct_text = f"{self.raw_progress:.1f}%"
        pct_scale = text_scale(0.52, self.width, self.height, floor=0.46, ceiling=0.60)
        pct_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        (pct_w, pct_h), _ = get_text_size(pct_text, FONT_MAIN, pct_scale, pct_thick)
        self._put_text(frame, pct_text, content_x + (bar_width - pct_w) // 2,
                       bar_y + (bar_height + pct_h) // 2, pct_scale, self.COLORS['text_primary'], pct_thick)

        cv2.line(frame, (x + 15, bar_y + bar_height + 12), (x + w - 15, bar_y + bar_height + 12),
                 self.COLORS['medium_blue'], 1)

        # ── Legend ────────────────────────────────────────────────────────────
        leg_y = bar_y + bar_height + 30
        self._put_text(frame, "LEGEND", content_x, leg_y, label_scale, self.COLORS['text_secondary'], label_thick)
        legend_items = [
            ("Completed", self.segment_colors['completed']),
            ("To Sew",    self.segment_colors['current']),
        ]
        box_size = 13
        leg_scale = text_scale(0.50, self.width, self.height, floor=0.44, ceiling=0.58)
        leg_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
        for i, (lbl, col) in enumerate(legend_items):
            item_y = leg_y + 18 + i * 22
            cv2.rectangle(frame, (content_x, item_y - box_size + 2),
                          (content_x + box_size, item_y + 2), col, -1)
            cv2.rectangle(frame, (content_x, item_y - box_size + 2),
                          (content_x + box_size, item_y + 2), self.COLORS['medium_blue'], 1)
            self._put_text(frame, lbl, content_x + box_size + 7, item_y,
                           leg_scale, self.COLORS['text_secondary'], leg_thick)

        cv2.line(frame, (x + 15, leg_y + 18 + len(legend_items) * 22 + 4),
                 (x + w - 15, leg_y + 18 + len(legend_items) * 22 + 4),
                 self.COLORS['medium_blue'], 1)

        # ── Detected color indicator ──────────────────────────────────────────
        det_y = leg_y + 18 + len(legend_items) * 22 + 22
        self._put_text(frame, "DETECTED COLOR", content_x, det_y, label_scale, self.COLORS['text_secondary'], label_thick)
        det_cfg = self.color_profiles[self.selected_detection_color]
        det_val_y = det_y + 20
        cv2.circle(frame, (content_x + 8, det_val_y - 6), 7, det_cfg['preview_bgr'], -1)
        cv2.circle(frame, (content_x + 8, det_val_y - 6), 7, self.COLORS['medium_blue'], 1)
        val_scale = text_scale(0.58, self.width, self.height, floor=0.52, ceiling=0.66)
        val_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        self._put_text(frame, det_cfg['label'], content_x + 22, det_val_y,
                       val_scale, self.COLORS['text_primary'], val_thick)
    
    def draw_evaluate_button(self, frame):
        """Draw the evaluate button below stats panel"""
        eb = self.evaluate_button
        
        # Don't show if already evaluated
        if self.is_evaluated:
            return
        
        # Button glow
        pulse = 0.5 + 0.3 * abs(math.sin(self.glow_phase * 1.2))
        self.draw_glow_rect(frame, eb['x'], eb['y'], eb['w'], eb['h'], 
                           self.COLORS['neon_blue'], pulse)
        
        # Button background
        overlay = frame.copy()
        cv2.rectangle(overlay, (eb['x'] + 2, eb['y'] + 2), 
                     (eb['x'] + eb['w'] - 2, eb['y'] + eb['h'] - 2), 
                     self.COLORS['button_hover'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Button text
        text = "EVALUATE"
        font_scale = text_scale(0.84, self.width, self.height, floor=0.76, ceiling=0.96)
        thickness = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
        font_scale = fit_text_scale(text, FONT_DISPLAY, eb['w'] - 14, font_scale, thickness, min_scale=0.66)
        (text_w, text_h), _ = get_text_size(text, FONT_DISPLAY, font_scale, thickness)
        text_x = eb['x'] + (eb['w'] - text_w) // 2
        text_y = eb['y'] + (eb['h'] + text_h) // 2
        
        self._put_text(frame, text, text_x, text_y, font_scale, self.COLORS['text_primary'], thickness, font=FONT_DISPLAY)

    def draw_attach_cloth_button(self, frame):
        """Draw a button allowing the user to attach/detach the pattern to the cloth."""
        b = self.attach_cloth_button

        # Button glow + disabled style when attached
        pulse = 0.5 + 0.3 * abs(math.sin(self.glow_phase * 1.2))
        if self.mask_attached_to_cloth:
            border_color = tuple(int(c * 0.5) for c in self.COLORS['glow_cyan'])
            fill_color = (20, 60, 20)
        else:
            border_color = self.COLORS['medium_blue']
            fill_color = self.COLORS['button_hover']

        self.draw_glow_rect(frame, b['x'], b['y'], b['w'], b['h'], border_color, pulse)

        overlay = frame.copy()
        cv2.rectangle(overlay, (b['x'] + 2, b['y'] + 2), (b['x'] + b['w'] - 2, b['y'] + b['h'] - 2), fill_color, -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        txt = "STOP" if self.mask_attached_to_cloth else "START"
        font_scale = text_scale(0.68, self.width, self.height, floor=0.62, ceiling=0.78)
        thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        tw, th = get_text_size(txt, FONT_MAIN, font_scale, thick)[0]
        tx = b['x'] + (b['w'] - tw) // 2
        ty = b['y'] + (b['h'] + th) // 2
        self._put_text(frame, txt, tx, ty, font_scale, self.COLORS['text_primary'], thick)
    
    def draw_evaluation_results(self, frame):
        """Draw evaluation results as a large centered modal window"""
        # Semi-transparent dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), 
                     (20, 10, 5), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Large centered results panel
        panel_w = 500
        panel_h = 350
        panel_x = (self.width - panel_w) // 2
        panel_y = (self.height - panel_h) // 2
        
        # Panel border with strong glow
        pulse = 0.6 + 0.4 * abs(math.sin(self.glow_phase * 0.8))
        if self.level_completed:
            border_color = self.COLORS['glow_cyan']
        else:
            border_color = self.COLORS['bright_blue']
        
        # Draw thick glowing border
        for i in range(3):
            offset = i * 3
            alpha = pulse * (1 - i * 0.3)
            glow_color = tuple(int(c * alpha) for c in border_color)
            cv2.rectangle(frame, (panel_x - offset, panel_y - offset), 
                         (panel_x + panel_w + offset, panel_y + panel_h + offset), 
                         glow_color, 3)
        
        # Panel background
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     self.COLORS['dark_blue'], -1)
        
        # Title
        content_y = panel_y + 60
        if self.level_completed:
            title = "LEVEL COMPLETED!"
            title_color = self.COLORS['glow_cyan']
        else:
            title = "LEVEL INCOMPLETE"
            title_color = self.COLORS['text_secondary']
        
        title_scale = text_scale(1.2, self.width, self.height, floor=1.04, ceiling=1.35)
        title_thickness = text_thickness(3, self.width, self.height, min_thickness=2, max_thickness=4)
        title_scale = fit_text_scale(title, FONT_DISPLAY, panel_w - 40, title_scale, title_thickness, min_scale=0.92)
        (title_w, title_h), _ = get_text_size(title, FONT_DISPLAY, title_scale, title_thickness)
        title_x = panel_x + (panel_w - title_w) // 2
        self._put_text(frame, title, title_x, content_y, title_scale, title_color, title_thickness, font=FONT_DISPLAY)
        
        # Progress achieved
        content_y += 70
        progress_label = "Progress Achieved:"
        prog_label_scale = text_scale(0.82, self.width, self.height, floor=0.74, ceiling=0.94)
        prog_label_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        (prog_label_w, _), _ = get_text_size(progress_label, FONT_MAIN, prog_label_scale, prog_label_thick)
        label_x = panel_x + (panel_w - prog_label_w) // 2
        self._put_text(frame, progress_label, label_x, content_y, prog_label_scale, self.COLORS['text_secondary'], prog_label_thick)
        
        content_y += 50
        progress_text = f"{self.raw_progress:.1f}%"
        prog_scale = text_scale(1.45, self.width, self.height, floor=1.22, ceiling=1.65)
        prog_thick = text_thickness(3, self.width, self.height, min_thickness=2, max_thickness=4)
        (prog_w, _), _ = get_text_size(progress_text, FONT_DISPLAY, prog_scale, prog_thick)
        prog_x = panel_x + (panel_w - prog_w) // 2
        self._put_text(frame, progress_text, prog_x, content_y, prog_scale, self.COLORS['text_primary'], prog_thick, font=FONT_DISPLAY)
        
        # Requirement text
        content_y += 40
        req_text = "(Need 80% to pass)"
        req_scale = text_scale(0.64, self.width, self.height, floor=0.58, ceiling=0.74)
        req_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
        (req_w, _), _ = get_text_size(req_text, FONT_MAIN, req_scale, req_thick)
        req_x = panel_x + (panel_w - req_w) // 2
        self._put_text(frame, req_text, req_x, content_y, req_scale, self.COLORS['text_secondary'], req_thick)
        
        # Action button (Try Again or Next Level)
        if self.level_completed:
            self.draw_next_level_button_centered(frame, panel_x, panel_y, panel_w, panel_h)
        else:
            self.draw_try_again_button(frame, panel_x, panel_y, panel_w, panel_h)
    
    def draw_try_again_button(self, frame, panel_x, panel_y, panel_w, panel_h):
        """Draw try again button in the centered modal"""
        button_w = 200
        button_h = 60
        button_x = panel_x + (panel_w - button_w) // 2
        button_y = panel_y + panel_h - 90
        
        # Update button coordinates for click detection
        self.try_again_button['x'] = button_x
        self.try_again_button['y'] = button_y
        self.try_again_button['w'] = button_w
        self.try_again_button['h'] = button_h
        
        # Button glow
        pulse = 0.5 + 0.4 * abs(math.sin(self.glow_phase * 1.5))
        self.draw_glow_rect(frame, button_x, button_y, button_w, button_h, 
                           self.COLORS['bright_blue'], pulse)
        
        # Button background
        overlay = frame.copy()
        cv2.rectangle(overlay, (button_x + 2, button_y + 2), 
                     (button_x + button_w - 2, button_y + button_h - 2), 
                     self.COLORS['button_hover'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Button text
        text = "TRY AGAIN"
        font_scale = text_scale(0.92, self.width, self.height, floor=0.8, ceiling=1.04)
        thickness = text_thickness(3, self.width, self.height, min_thickness=2, max_thickness=4)
        font_scale = fit_text_scale(text, FONT_DISPLAY, button_w - 20, font_scale, thickness, min_scale=0.72)
        (text_w, text_h), _ = get_text_size(text, FONT_DISPLAY, font_scale, thickness)
        text_x = button_x + (button_w - text_w) // 2
        text_y = button_y + (button_h + text_h) // 2
        
        self._put_text(frame, text, text_x, text_y, font_scale, self.COLORS['text_primary'], thickness, font=FONT_DISPLAY)
    
    def draw_next_level_button_centered(self, frame, panel_x, panel_y, panel_w, panel_h):
        """Draw next level button in the centered modal"""
        button_w = 220
        button_h = 60
        button_x = panel_x + (panel_w - button_w) // 2
        button_y = panel_y + panel_h - 90
        
        # Update button coordinates for click detection
        self.next_level_button['x'] = button_x
        self.next_level_button['y'] = button_y
        self.next_level_button['w'] = button_w
        self.next_level_button['h'] = button_h
        
        # Button glow - stronger for success
        pulse = 0.6 + 0.4 * abs(math.sin(self.glow_phase * 1.5))
        self.draw_glow_rect(frame, button_x, button_y, button_w, button_h, 
                           self.COLORS['glow_cyan'], pulse)
        
        # Button background
        overlay = frame.copy()
        cv2.rectangle(overlay, (button_x + 2, button_y + 2), 
                     (button_x + button_w - 2, button_y + button_h - 2), 
                     self.COLORS['button_hover'], -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Button text
        text = "NEXT LEVEL >"
        font_scale = text_scale(0.92, self.width, self.height, floor=0.8, ceiling=1.04)
        thickness = text_thickness(3, self.width, self.height, min_thickness=2, max_thickness=4)
        font_scale = fit_text_scale(text, FONT_DISPLAY, button_w - 20, font_scale, thickness, min_scale=0.72)
        (text_w, text_h), _ = get_text_size(text, FONT_DISPLAY, font_scale, thickness)
        text_x = button_x + (button_w - text_w) // 2
        text_y = button_y + (button_h + text_h) // 2
        
        # Draw with glow effect
        draw_text(
            frame,
            text,
            text_x,
            text_y,
            font_scale,
            self.COLORS['bright_blue'],
            thickness,
            font=FONT_DISPLAY,
            outline_color=self.COLORS['glow_cyan'],
            outline_extra=2,
        )
    
    def draw_stat_item(self, frame, label, value, x, y, max_width, value_color=None):
        """Helper method to draw a stat item (label and value)"""
        if value_color is None:
            value_color = self.COLORS['text_primary']
        
        # Draw label - larger font
        label_scale = text_scale(0.62, self.width, self.height, floor=0.56, ceiling=0.74)
        label_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        self._put_text(frame, label, x, y, label_scale, self.COLORS['text_secondary'], label_thick)
        
        # Draw value (right-aligned on same line) - larger font
        value_scale = text_scale(0.92, self.width, self.height, floor=0.82, ceiling=1.06)
        value_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=3)
        (val_w, val_h), _ = get_text_size(value, FONT_DISPLAY, value_scale, value_thick)
        value_x = x + max_width - val_w
        self._put_text(frame, value, value_x, y, value_scale, value_color, value_thick, font=FONT_DISPLAY)

    def handle_click(self, x, y):
        """Handle mouse clicks in pattern mode"""
        # Check guide overlay first
        if self.show_guide:
            gb = self.guide_button
            if gb['x'] <= x <= gb['x'] + gb['w'] and gb['y'] <= y <= gb['y'] + gb['h']:
                self.play_button_click_sound()
                if self.guide_step < 4:
                    # Advance to next step
                    self.guide_step += 1
                else:
                    # Last step - close guide
                    self.show_guide = False
                    self.guide_shown_this_session = True
                return None
            # Block all other clicks while guide is showing
            return None
        
        # Check back button
        bb = self.back_button
        if bb['x'] <= x <= bb['x'] + bb['w'] and bb['y'] <= y <= bb['y'] + bb['h']:
            self.play_button_click_sound()
            # Reset all progress and mask state when leaving pattern mode
            try:
                self.reset_progress()
            except Exception:
                pass
            # Clear mask attachment and seeding state so returning later is fresh
            try:
                self.mask_attached_to_cloth = False
                self.mask_anchor = None
                self.mask_initial_anchor = None
                self.mask_initial_cloth_centroid = None
                self.mask_initial_cloth_bbox = None
                self.mask_initial_expected_point = None
                self.centerline_progress_initialized = False
            except Exception:
                pass
            return 'back'
        
        # Check evaluate button (only if not evaluated yet)
        for color_name, btn in self.color_buttons.items():
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                self.selected_detection_color = color_name
                # Auto-fix cloth if it now conflicts with the new thread color
                if self.selected_cloth_color == color_name:
                    for c in self.cloth_color_profiles.keys():
                        if c != color_name:
                            self.selected_cloth_color = c
                            break
                print(f"🎨 Detection color set to: {self.color_profiles[color_name]['label']}")
                return None

        for color_name, btn in self.cloth_color_buttons.items():
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                if btn.get('disabled', False):
                    return None  # blocked — same as thread color
                self.play_button_click_sound()
                self.selected_cloth_color = color_name
                self.smooth_cloth_bbox = None  # reset smoother on cloth colour change
                if hasattr(self, 'bbox_history'):
                    self.bbox_history.clear()
                print(f"🧵 Cloth color set to: {self.cloth_color_profiles[color_name]['label']}")
                return None

        # Needle detection toggle
        tb = getattr(self, 'needle_toggle_button', None)
        if tb is not None and tb['x'] <= x <= tb['x'] + tb['w'] and tb['y'] <= y <= tb['y'] + tb['h']:
            self.play_button_click_sound()
            self.needle_detection_enabled = not bool(self.needle_detection_enabled)
            print(f"🪡 Needle detection {'ENABLED' if self.needle_detection_enabled else 'DISABLED'}")
            return None

        # Check confidence buttons
        if (self.conf_minus_button['x'] <= x <= self.conf_minus_button['x'] + self.conf_minus_button['w'] and
                self.conf_minus_button['y'] <= y <= self.conf_minus_button['y'] + self.conf_minus_button['h']):
            self.play_button_click_sound()
            self.confidence_threshold = max(0.05, self.confidence_threshold - 0.05)
            print(f"Confidence threshold: {self.confidence_threshold:.2f}")
            return None

        if (self.conf_plus_button['x'] <= x <= self.conf_plus_button['x'] + self.conf_plus_button['w'] and
                self.conf_plus_button['y'] <= y <= self.conf_plus_button['y'] + self.conf_plus_button['h']):
            self.play_button_click_sound()
            self.confidence_threshold = min(0.90, self.confidence_threshold + 0.05)
            print(f"Confidence threshold: {self.confidence_threshold:.2f}")
            return None

        if not self.is_evaluated:
            eb = self.evaluate_button
            if eb['x'] <= x <= eb['x'] + eb['w'] and eb['y'] <= y <= eb['y'] + eb['h']:
                self.play_button_click_sound()
                self.evaluate_pattern()
                return None

        # Attach cloth button (toggle attach/detach)
        ab = self.attach_cloth_button
        if ab['x'] <= x <= ab['x'] + ab['w'] and ab['y'] <= y <= ab['y'] + ab['h']:
            self.play_button_click_sound()
            # Toggle latch; if attaching, seed initial anchor + cloth centroid
            self.mask_attached_to_cloth = not bool(self.mask_attached_to_cloth)
            if self.mask_attached_to_cloth:
                # Record initial anchor (camera centre) so the overlay appears
                # centered on-screen at START. Record the detected cloth
                # centroid (or bbox centre) as the reference point. Later
                # frames translate the overlay by (current_centroid - initial_centroid)
                # so the mask follows cloth motion but retains the initial size.
                try:
                    cam_cx = int(self.camera_width // 2)
                    cam_cy = int(self.camera_height // 2)
                except Exception:
                    cam_cx, cam_cy = 0, 0
                self.mask_initial_anchor = (cam_cx, cam_cy)
                self.mask_anchor = (cam_cx, cam_cy)
                # Save initial cloth centroid (prefer live motion centroid)
                if self.last_motion_centroid is not None:
                    try:
                        mx = float(self.last_motion_centroid[0])
                        my = float(self.last_motion_centroid[1])
                        self.mask_initial_cloth_centroid = (mx, my)
                    except Exception:
                        self.mask_initial_cloth_centroid = None
                elif self.last_cloth_bbox is not None:
                    try:
                        bx1, by1, bx2, by2 = self.last_cloth_bbox
                        mx = (bx1 + bx2) / 2.0
                        my = (by1 + by2) / 2.0
                        self.mask_initial_cloth_centroid = (mx, my)
                    except Exception:
                        self.mask_initial_cloth_centroid = None
                else:
                    self.mask_initial_cloth_centroid = None
                # Also persist the initial bbox so size can be referenced later
                self.mask_initial_cloth_bbox = tuple(self.last_cloth_bbox) if self.last_cloth_bbox is not None else None
                # Reset the initial expected point so draw step can capture
                # the current centerline expected location without later
                # shifting the overlay when progress advances.
                self.mask_initial_expected_point = None
                print("📌 START pressed: mask seeded at camera centre; will follow cloth movement")
            else:
                # On STOP: hide the live overlay but keep the initial seed
                # state so restarting doesn't reset progress or the frozen
                # bbox/size. Only clear the active anchor so the overlay
                # disappears visually.
                self.mask_anchor = None
                # keep the initial expected point nil so restarting can reseed
                self.mask_initial_expected_point = None
                print("📌 STOP pressed: mask hidden (seed retained)")
            return None
        
        # Check try again button (only if evaluated and failed)
        if self.is_evaluated and not self.level_completed:
            tb = self.try_again_button
            if tb['x'] <= x <= tb['x'] + tb['w'] and tb['y'] <= y <= tb['y'] + tb['h']:
                self.play_button_click_sound()
                # Reset all progress to try again
                self.pattern_progress = 0.0
                self.raw_progress = 0.0
                self.current_segment = 1
                self.highest_segment_reached = 1
                self.current_accuracy = 0.0
                self.total_score = 0
                self.out_of_segment_warning = False
                self.warning_message = ""
                self.is_evaluated = False
                self.level_completed = False
                print(f"🔄 Progress reset - Try again!")
                return None
        
        # Check next level button (only if evaluated and passed)
        if self.is_evaluated and self.level_completed:
            nb = self.next_level_button
            if nb['x'] <= x <= nb['x'] + nb['w'] and nb['y'] <= y <= nb['y'] + nb['h']:
                self.play_button_click_sound()
                return 'next_level'
        
        return None
    
    def evaluate_pattern(self):
        """Evaluate the current pattern and determine if level is completed"""
        self.is_evaluated = True
        
        # Check if level is completed (80%+ raw progress)
        if self.raw_progress >= 80:
            self.level_completed = True
            print(f"✅ Level {self.current_level} completed! Progress: {self.raw_progress:.1f}%")
        else:
            self.level_completed = False
            print(f"📊 Evaluation: {self.raw_progress:.1f}% progress. Need 80%+ to complete.")
