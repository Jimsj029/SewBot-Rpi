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
        
        # Pattern variables
        self.current_level = 1
        self.current_level_tracking = 1  # Track which level stats belong to
        self.uniform_width = 200
        self.uniform_height = 300
        self.alpha_blend = 0.9
        self.glow_phase = 0
        
        # Level info display at top center (aligned with back button)
        self.level_display_y = 60
        
        # Camera display area (centered, moved higher)
        self.camera_width = 560
        self.camera_height = 420
        self.camera_x = 216  # Centered horizontally (1000 - 560) // 2
        self.camera_y = 120  # Moved up since no level buttons
        
        # Score/Stats panel (right side of camera)
        self.score_panel_x = 800
        self.score_panel_y = 190
        self.score_panel_width = 200
        self.score_panel_height = 155
        
        # Back button (top left)
        self.back_button = {'x': 20, 'y': 20, 'w': 120, 'h': 50}
        
        # Evaluate button (below stats panel)
        self.evaluate_button = {'x': 800, 'y': 350, 'w': 200, 'h': 50}

        # Color detection selector (right side, top)
        self.color_panel_x = 800
        self.color_panel_y = 115
        self.color_panel_width = 200
        self.color_panel_height = 70
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
        }
        self.color_buttons = {}

        # Confidence controls (below evaluate button)
        self.conf_panel_x = 800
        self.conf_panel_y = 405
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
        self.needle_model_imgsz = 256  # needle.onnx expects 256x256 input

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
        self.motion_vector_x = 0.0
        self.motion_vector_y = 0.0
        self.motion_vector_min_mag = 0.35
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
        
        # Cloth colour selector
        self.selected_cloth_color = 'red'
        self.cloth_color_profiles = {
            'red': {
                'label': 'RED',
                'preview_bgr': (0, 0, 220),
                # Wide hue range catches bright red, dark red, terracotta,
                # and brownish-red cloth under varying lighting.
                'hsv_ranges': [
                    ((0,  50, 35), (15, 255, 255)),   # red / terracotta
                    ((155, 50, 35), (179, 255, 255)), # wraparound red
                ],
            },
            'black': {
                'label': 'BLACK',
                'preview_bgr': (30, 30, 30),
                # Min value 5 avoids dead pixels; saturation up to 100
                # handles fabric texture; value ceiling 52 = truly dark.
                'hsv_ranges': [
                    ((0, 0, 5), (179, 100, 52)),
                ],
            },
        }
        self.cloth_color_buttons = {}
        self.cloth_color_panel_x = 800
        self.cloth_color_panel_y = 475   # below confidence panel
        self.cloth_color_panel_width = 200
        self.cloth_color_panel_height = 70

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
        
        mask_path = os.path.join(self.blueprint_folder, f'level{level}_mask.png')
        if not os.path.exists(mask_path):
            return None, None
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None
        
        # Resize the mask to the desired dimensions
        mask = cv2.resize(mask, (self.uniform_width, self.uniform_height))

        # Cut the binary mask in half (keep top half)
        mask = mask[:mask.shape[0] // 2, :]

        # Base width is 50%; increase it by 25% => 62.5% of original width.
        new_w = int(round(mask.shape[1] * 0.625))
        new_w = max(1, min(mask.shape[1], new_w))
        mask = cv2.resize(mask, (new_w, mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Move the pattern to the middle of the cloth while keeping it centered horizontally
        top_offset = (self.uniform_height - mask.shape[0]) // 2
        left_offset = (self.uniform_width - mask.shape[1]) // 2
        centered_mask = np.zeros((self.uniform_height, self.uniform_width), dtype=mask.dtype)
        centered_mask[top_offset:top_offset + mask.shape[0], left_offset:left_offset + mask.shape[1]] = mask
        mask = centered_mask
        
        # Convert mask to white overlay (pattern lines)
        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        alpha = mask / 255.0
        
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
            completed_mask_resized = cv2.resize(self.completed_stitch_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            kernel_size = max(3, self.cyan_spread_radius * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            proximity_mask = cv2.dilate(completed_mask_resized, kernel, iterations=1)
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
        """Update whether cloth is moving near the needle using frame differencing."""
        if cam_frame is None or cam_frame.size == 0:
            self.cloth_motion_active = False
            self.last_motion_diff = 0.0
            self.last_motion_centroid = None
            self.motion_vector_x = 0.0
            self.motion_vector_y = 0.0
            return

        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        patch_half = self.motion_patch_size // 2
        cx = int(self.needle_pos_x)
        cy = int(self.needle_pos_y + self.motion_patch_y_offset)

        x1 = max(0, cx - patch_half)
        y1 = max(0, cy - patch_half)
        x2 = min(gray.shape[1], cx + patch_half)
        y2 = min(gray.shape[0], cy + patch_half)

        patch = gray[y1:y2, x1:x2]
        if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
            self.cloth_motion_active = False
            self.last_motion_diff = 0.0
            self.last_motion_centroid = None
            self.motion_vector_x = 0.0
            self.motion_vector_y = 0.0
            return

        patch = cv2.GaussianBlur(patch, (5, 5), 0)

        if self.prev_motion_patch is None or self.prev_motion_patch.shape != patch.shape:
            self.prev_motion_patch = patch
            self.cloth_motion_active = False
            self.last_motion_diff = 0.0
            self.last_motion_centroid = None
            self.motion_vector_x = 0.0
            self.motion_vector_y = 0.0
            return

        diff = cv2.absdiff(patch, self.prev_motion_patch)
        self.last_motion_diff = float(np.mean(diff))

        # Estimate local motion direction from the centroid shift of changed pixels.
        motion_thresh = max(1.0, self.motion_diff_threshold * 1.2)
        _, motion_mask = cv2.threshold(diff, motion_thresh, 255, cv2.THRESH_BINARY)
        motion_pixels = int(np.count_nonzero(motion_mask))
        if motion_pixels >= 20:
            ys, xs = np.where(motion_mask > 0)
            centroid = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float32)
            if self.last_motion_centroid is not None:
                raw_dx = float(centroid[0] - self.last_motion_centroid[0])
                raw_dy = float(centroid[1] - self.last_motion_centroid[1])
                blend = 0.35
                self.motion_vector_x = (1.0 - blend) * self.motion_vector_x + blend * raw_dx
                self.motion_vector_y = (1.0 - blend) * self.motion_vector_y + blend * raw_dy
            self.last_motion_centroid = centroid
        else:
            self.motion_vector_x *= 0.7
            self.motion_vector_y *= 0.7

        self.prev_motion_patch = patch

        if self.last_motion_diff >= self.motion_diff_threshold:
            self.motion_grace_counter = self.motion_grace_frames
        else:
            self.motion_grace_counter = max(0, self.motion_grace_counter - 1)

        self.cloth_motion_active = self.motion_grace_counter > 0

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

        if source == "center" and self.require_motion_for_stitch and not self.cloth_motion_active:
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

        if is_valid_stitch:
            draw_x, draw_y = px, py
            if self.snap_stitches_to_centerline and centerline_path is not None and len(centerline_path) > 0:
                draw_x, draw_y = snapped_x, snapped_y

            if source == "center":
                if (self.stitch_frame_index - self.last_stitch_frame_index) < self.stitch_cooldown_frames:
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
                    if math.hypot(draw_x - lx, draw_y - ly) < min_move:
                        return

            if self.completed_stitch_mask is None:
                self.completed_stitch_mask = np.zeros((pat_h, pat_w), dtype=np.uint8)

            prev_pixels = int(np.count_nonzero(self.completed_stitch_mask))
            cv2.circle(self.completed_stitch_mask, (draw_x, draw_y), self.stitch_draw_radius, 255, -1)
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
                        # Usually we move toward nearest_idx. If nearest_idx ==
                        # current_idx (common with needle-anchored overlay),
                        # infer direction from cloth motion to keep progression moving.
                        delta = nearest_idx - current_idx
                        if delta == 0:
                            delta = self._infer_centerline_step_direction(centerline_path, current_idx)

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

        sample_radius = 7
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
        """
        pat_crop = pattern_alpha[:actual_h, :actual_w]
        pat_binary = (pat_crop > self.pattern_alpha_threshold).astype(np.uint8)
        path_points = []
        prev_x = None

        for y in range(actual_h):
            xs = np.where(pat_binary[y] > 0)[0]
            if xs.size == 0:
                continue

            if prev_x is None:
                x = int(np.median(xs))
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
        if   rp >= 70: new_seg, new_prog = 5, 100.0
        elif rp >= 45: new_seg, new_prog = 4,  75.0
        elif rp >= 25: new_seg, new_prog = 3,  50.0
        elif rp >= 12: new_seg, new_prog = 2,  25.0
        else:          new_seg, new_prog = 1,   0.0

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
        
        # Draw score/stats panel
        self.draw_score_panel(frame)
        
        # Draw evaluate button
        self.draw_evaluate_button(frame)

        # Draw selectable color filter controls
        self.draw_color_selector(frame)

        # Draw confidence controls
        self.draw_confidence_controls(frame)

        # Draw cloth colour selector
        self.draw_cloth_color_selector(frame)
        
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
            hsv_detection_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2HSV)
            self.stitch_frame_index += 1
            self._update_cloth_motion_state(detection_frame)
            follow_check_ready = False
            follow_on_corridor = True
            follow_order_valid = True
            
            # Load pattern mask
            pattern_overlay, pattern_alpha = self.load_blueprint(self.current_level)

            # Overlay pattern anchored to needle — no cloth detection needed
            if pattern_overlay is not None and pattern_alpha is not None:
                overlay_h, overlay_w = pattern_overlay.shape[:2]

                centerline_path = self._build_centerline_path(pattern_alpha, overlay_w, overlay_h)
                expected_path_idx = None
                if centerline_path is not None and len(centerline_path) > 0:
                    max_idx = len(centerline_path) - 1
                    if not self.centerline_progress_initialized:
                        # Seed once from current raw progress; afterwards, progression
                        # is driven by accepted stitches, not area percentage.
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

                # Align expected centerline stitch point directly under the red dot.
                x_offset = int(round(self.needle_pos_x)) - exp_x
                y_offset = int(round(self.needle_pos_y)) - exp_y
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
                    
                    self.run_needle_pipeline(
                        detection_frame, pattern_alpha,
                        x_offset, y_offset, actual_w, actual_h,
                        run_detection=True,
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

            cv2.circle(cam_frame, (int(self.needle_pos_x), int(self.needle_pos_y)), 5, marker_color, -1)
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
            
            # Draw color legend below camera view
            self.draw_pattern_legend(frame)
        
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
        """Draw legend showing what each pattern color means"""
        # Position: below camera, aligned left
        legend_y = self.camera_y + self.camera_height + 35
        
        # Start legend from left side of camera with small padding
        legend_x = self.camera_x + 30
        
        # Legend items - updated for real-time coloring
        legend_items = [
            ("Completed", self.segment_colors['completed']),
            ("To Sew", self.segment_colors['current'])
        ]
        
        # Draw each legend item
        item_spacing = 140
        for i, (label, color) in enumerate(legend_items):
            item_x = legend_x + i * item_spacing
            
            # Draw color box
            box_size = 15
            cv2.rectangle(frame, (item_x, legend_y - box_size + 3), 
                        (item_x + box_size, legend_y + 3), 
                        color, -1)
            cv2.rectangle(frame, (item_x, legend_y - box_size + 3), 
                        (item_x + box_size, legend_y + 3), 
                        self.COLORS['medium_blue'], 1)
            
            # Draw label - larger font
            label_scale = text_scale(0.66, self.width, self.height, floor=0.58, ceiling=0.76)
            label_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
            self._put_text(frame, label, item_x + box_size + 8, legend_y, label_scale, self.COLORS['text_secondary'], label_thick)

        target = self.color_profiles[self.selected_detection_color]['label']
        detect_text = f"Detecting: {target}"
        detect_scale = text_scale(0.62, self.width, self.height, floor=0.54, ceiling=0.72)
        detect_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        self._put_text(frame, detect_text, legend_x, legend_y + 30, detect_scale, self.COLORS['text_primary'], detect_thick)

    def draw_color_selector(self, frame):
        """Draw color selection buttons for stitch filtering."""
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
        self._put_text(frame, "DETECT COLOR", x + 16, y + 22, header_scale, self.COLORS['text_secondary'], header_thick)

        labels = ['white', 'yellow', 'red']
        btn_w = 58
        btn_h = 34
        start_x = x + 10
        btn_y = y + 28
        gap = 6

        self.color_buttons = {}
        for idx, key in enumerate(labels):
            bx = start_x + idx * (btn_w + gap)
            by = btn_y
            is_selected = key == self.selected_detection_color
            cfg = self.color_profiles[key]

            border = self.COLORS['glow_cyan'] if is_selected else self.COLORS['medium_blue']
            cv2.rectangle(frame, (bx, by), (bx + btn_w, by + btn_h), border, 2)

            fill = frame.copy()
            fill_alpha = 0.6 if is_selected else 0.35
            cv2.rectangle(fill, (bx + 2, by + 2), (bx + btn_w - 2, by + btn_h - 2), self.COLORS['button_normal'], -1)
            cv2.addWeighted(fill, fill_alpha, frame, 1 - fill_alpha, 0, frame)

            cv2.circle(frame, (bx + 11, by + btn_h // 2), 6, cfg['preview_bgr'], -1)
            cv2.circle(frame, (bx + 11, by + btn_h // 2), 6, self.COLORS['text_primary'], 1)
            letter_scale = text_scale(0.58, self.width, self.height, floor=0.52, ceiling=0.66)
            letter_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
            self._put_text(frame, cfg['label'][0], bx + 22, by + 23, letter_scale, self.COLORS['text_primary'], letter_thick)

            self.color_buttons[key] = {'x': bx, 'y': by, 'w': btn_w, 'h': btn_h}

    def draw_cloth_color_selector(self, frame):
        """Draw cloth colour selector buttons (RED / BLACK)."""
        x = self.cloth_color_panel_x
        y = self.cloth_color_panel_y
        w = self.cloth_color_panel_width
        h = self.cloth_color_panel_height

        pulse = 0.35 + 0.25 * abs(math.sin(self.glow_phase * 0.85))
        self.draw_glow_rect(frame, x, y, w, h, self.COLORS['bright_blue'], pulse)

        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay, (x + 2, y + 2), (x + w - 2, y + h - 2), self.COLORS['dark_blue'], -1)
        cv2.addWeighted(panel_overlay, 0.82, frame, 0.18, 0, frame)

        header_scale = text_scale(0.5, self.width, self.height, floor=0.46, ceiling=0.58)
        header_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
        self._put_text(frame, "CLOTH COLOR", x + 22, y + 22, header_scale, self.COLORS['text_secondary'], header_thick)

        labels = list(self.cloth_color_profiles.keys())
        btn_w = 84
        btn_h = 34
        start_x = x + 10
        btn_y = y + 28
        gap = 12

        self.cloth_color_buttons = {}
        for idx, key in enumerate(labels):
            bx = start_x + idx * (btn_w + gap)
            by = btn_y
            is_selected = key == self.selected_cloth_color
            cfg = self.cloth_color_profiles[key]

            border = self.COLORS['glow_cyan'] if is_selected else self.COLORS['medium_blue']
            cv2.rectangle(frame, (bx, by), (bx + btn_w, by + btn_h), border, 2)

            fill = frame.copy()
            fill_alpha = 0.6 if is_selected else 0.35
            cv2.rectangle(fill, (bx + 2, by + 2), (bx + btn_w - 2, by + btn_h - 2), self.COLORS['button_normal'], -1)
            cv2.addWeighted(fill, fill_alpha, frame, 1 - fill_alpha, 0, frame)

            cv2.circle(frame, (bx + 11, by + btn_h // 2), 6, cfg['preview_bgr'], -1)
            cv2.circle(frame, (bx + 11, by + btn_h // 2), 6, self.COLORS['text_primary'], 1)
            cloth_scale = text_scale(0.46, self.width, self.height, floor=0.42, ceiling=0.54)
            cloth_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
            self._put_text(frame, cfg['label'], bx + 22, by + 23, cloth_scale, self.COLORS['text_primary'], cloth_thick)

            self.cloth_color_buttons[key] = {'x': bx, 'y': by, 'w': btn_w, 'h': btn_h}

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
        """Draw the score/stats panel on the right side"""
        x, y, w, h = self.score_panel_x, self.score_panel_y, self.score_panel_width, self.score_panel_height
        
        # Panel border with glow
        pulse = 0.4 + 0.3 * abs(math.sin(self.glow_phase * 0.7))
        self.draw_glow_rect(frame, x, y, w, h, self.COLORS['bright_blue'], pulse)
        
        # Panel background (semi-transparent)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x + 3, y + 3), (x + w - 3, y + h - 3), self.COLORS['dark_blue'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Title "STATS"
        title_font_scale = text_scale(0.92, self.width, self.height, floor=0.82, ceiling=1.05)
        title_thickness = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
        title_text = "STATS"
        (title_w, title_h), _ = get_text_size(title_text, FONT_DISPLAY, title_font_scale, title_thickness)
        title_x = x + (w - title_w) // 2
        title_y = y + 40
        draw_text(
            frame,
            title_text,
            title_x,
            title_y,
            title_font_scale,
            self.COLORS['bright_blue'],
            title_thickness,
            font=FONT_DISPLAY,
            outline_color=self.COLORS['glow_cyan'],
            outline_extra=2,
        )
        
        # Draw horizontal divider
        cv2.line(frame, (x + 15, title_y + 15), (x + w - 15, title_y + 15), self.COLORS['medium_blue'], 2)
        
        # Stats content
        content_x = x + 20
        start_y = title_y + 45
        label_scale = text_scale(0.62, self.width, self.height, floor=0.56, ceiling=0.74)
        label_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        
        # Progress bar (moved up since accuracy/score removed)
        progress_y = start_y + 20
        self._put_text(frame, "PROGRESS", content_x, progress_y, label_scale, self.COLORS['text_secondary'], label_thick)
        
        # Progress bar background (divided into 4 segments)
        bar_y = progress_y + 15
        bar_width = w - 40
        bar_height = 20
        segment_width = (bar_width - 4) // 4
        
        # Draw 4 segment boxes
        for seg in range(1, 5):
            seg_x = content_x + 2 + (seg - 1) * segment_width
            
            # Determine segment color based on progress milestones
            if self.pattern_progress >= seg * 25:
                # Completed segment - cyan
                seg_color = self.segment_colors['completed']
            elif seg == self.current_segment:
                # Current segment - yellow with pulse
                pulse = 0.5 + 0.5 * abs(math.sin(self.glow_phase * 2))
                seg_color = tuple(int(c * pulse) for c in self.segment_colors['current'])
            else:
                # Upcoming segment - gray
                seg_color = self.segment_colors['upcoming']
            
            # Draw segment
            overlay_bar = frame.copy()
            cv2.rectangle(overlay_bar, (seg_x, bar_y + 2), 
                        (seg_x + segment_width - 2, bar_y + bar_height - 2), 
                        seg_color, -1)
            cv2.addWeighted(overlay_bar, 0.7, frame, 0.3, 0, frame)
            
            # Draw segment border
            cv2.rectangle(frame, (seg_x, bar_y + 2), 
                        (seg_x + segment_width - 2, bar_y + bar_height - 2), 
                        self.COLORS['medium_blue'], 1)
        
        # Draw outer border
        cv2.rectangle(frame, (content_x, bar_y), (content_x + bar_width, bar_y + bar_height), 
                     self.COLORS['medium_blue'], 2)
        
        # Progress percentage text
        progress_text = f"{self.raw_progress:.0f}%"
        percent_scale = text_scale(0.62, self.width, self.height, floor=0.56, ceiling=0.74)
        percent_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        (prog_w, prog_h), _ = get_text_size(progress_text, FONT_MAIN, percent_scale, percent_thick)
        self._put_text(frame, progress_text, content_x + (bar_width - prog_w) // 2, bar_y + 16, percent_scale, self.COLORS['text_primary'], percent_thick)
        
        # Segment indicator text
        segment_text_y = bar_y + bar_height + 18
        segment_label = f"SEGMENT {self.current_segment}/4"
        segment_scale = text_scale(0.54, self.width, self.height, floor=0.5, ceiling=0.64)
        segment_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        self._put_text(frame, segment_label, content_x, segment_text_y, segment_scale, self.COLORS['text_secondary'], segment_thick)
    
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
            return 'back'
        
        # Check evaluate button (only if not evaluated yet)
        for color_name, btn in self.color_buttons.items():
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                self.selected_detection_color = color_name
                print(f"🎨 Detection color set to: {self.color_profiles[color_name]['label']}")
                return None

        for color_name, btn in self.cloth_color_buttons.items():
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.play_button_click_sound()
                self.selected_cloth_color = color_name
                self.smooth_cloth_bbox = None  # reset smoother on cloth colour change
                if hasattr(self, 'bbox_history'):
                    self.bbox_history.clear()
                print(f"🧵 Cloth color set to: {self.cloth_color_profiles[color_name]['label']}")
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
