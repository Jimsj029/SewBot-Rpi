import cv2
import numpy as np
import math
import os
from datetime import datetime
import onnxruntime as ort
from music_manager import get_music_manager
from ui.typography import (
    FONT_MAIN,
    FONT_DISPLAY,
    text_scale,
    text_thickness,
    get_text_size,
    fit_text_scale,
    draw_text
)


class PatternMode:
    def __init__(self, width, height, colors, blueprint_folder='blueprint'):
        self.width = width
        self.height = height
        self.COLORS = colors
        self.blueprint_folder = blueprint_folder
        
        # Pattern variables
        self.current_level = 1
        self.current_level_tracking = 1 
        self.uniform_width = 216   
        self.uniform_height = 270 
        self.alpha_blend = 0.9
        # Visual opacity factor for pattern overlay (1.0 = unchanged, 0.5 = half opacity)
        self.pattern_visual_opacity = 0.5
        self.glow_phase = 0
        
        # Level info display at top center
        self.level_display_y = 60
        

        # Camera
        self.camera_x      = 216
        self.camera_y      = 120
        self.camera_width  = 560
        self.camera_height = 420
        # Camera zoom buttons (drawn over camera feed)
        self.zoom_in_button = {'x': self.camera_x + self.camera_width - 46, 'y': self.camera_y + 8, 'w': 40, 'h': 36}
        self.zoom_out_button = {'x': self.camera_x + self.camera_width - 46 - 46, 'y': self.camera_y + 8, 'w': 40, 'h': 36}

        # Left panel — Thread Color
        self.color_panel_x      = 8
        self.color_panel_y      = 215
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
        self.score_panel_y      = 145
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

        # Start button — shown in the stats panel before sewing begins
        self.start_button = {
            'x': self.score_panel_x + 20,
            'y': self.score_panel_y + self.score_panel_height - 70,
            'w': self.score_panel_width - 40,
            'h': 50,
        }
        self.sewing_started = False  # Gates cloth/thread color detection

        # Back button
        self.back_button = {'x': 20, 'y': 20, 'w': 120, 'h': 50}
        # ── END LAYOUT ───────────────────────────────────────────────────────────
        self.selected_detection_color = 'white'
        self.color_profiles = {
            'white': {
                'label': 'WHITE',
                'preview_bgr': (240, 240, 240),
                'hsv_ranges': [((0, 0, 140), (179, 70, 255))],
                'min_ratio': 0.04,
            },
            'yellow': {
                'label': 'YELLOW',
                'preview_bgr': (0, 255, 255),
                'hsv_ranges': [((15, 50, 60), (42, 255, 255))],
                'min_ratio': 0.04,
            },
            'red': {
                'label': 'RED',
                'preview_bgr': (0, 0, 255),
                'hsv_ranges': [
                    ((0, 70, 50), (12, 255, 255)),
                    ((158, 70, 50), (179, 255, 255)),
                ],
                'min_ratio': 0.04,
            },
            'black': {
                'label': 'BLACK',
                'preview_bgr': (30, 30, 30),
                'hsv_ranges': [
                    ((0, 0, 0), (179, 120, 60)),
                ],
                'min_ratio': 0.04,
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
        self.last_color_contour_box = None
        self.color_overlay_interval = 3
        self._color_overlay_counter = 0
        
        # Try again button (centered in modal) - will be calculated dynamically
        self.try_again_button = {'x': 0, 'y': 0, 'w': 200, 'h': 60}
        
        # Next level button (centered in modal) - will be calculated dynamically
        self.next_level_button = {'x': 0, 'y': 0, 'w': 200, 'h': 60}
        # Evaluation flow button/state: stage 0 = comparison preview, stage 1 = score modal
        self.eval_next_button = {'x': 0, 'y': 0, 'w': 200, 'h': 60}
        self.eval_screen_stage = 0
        
        # Game tracking variables
        self.current_accuracy = 0.0
        self.total_score = 0
        self.pattern_progress = 0.0  # 0-100%
        self.raw_progress = 0.0  # Raw actual progress for evaluation
        self.progress_from_path = True  # Progress = percentage of path index taken vs full path
        # Evaluation metrics
        self.evaluation_wrong_pct = 0.0  # % of wrong/off-pattern stitches among detections
        self.final_score = 0.0  # Adjusted score after accounting for wrong stitches
        self.level_pass_thresholds = {
            1: 60.0,
            2: 35.0,
            3: 40.0,
            4: 30.0,
            5: 25.0,
        }
        self.session_start_time = None
        
        # Evaluation state
        self.is_evaluated = False
        self.level_completed = False
        self.last_evaluation_screenshot_path = None
        self.eval_vis_detected = None     # Camera snapshot with detected thread highlighted
        self.eval_vis_mask = None         # Blueprint target mask image
        self.eval_vis_comparison = None   # Pixel-level comparison image
        # Leniency control for final evaluation score.
        # Example: 4.0 means raw 20% becomes displayed/scored 80% (capped at 100).
        self.eval_score_multiplier = 4.0

        # Screenshot folder for saving evaluation screenshots
        self.screenshot_folder = os.path.join(os.path.dirname(__file__), 'screenshot')

        # Last camera frame/projection for screenshot-based AI evaluation
        self.last_camera_frame = None
        self.last_pattern_projection = None
        
        # Guide/Tutorial state
        self.show_guide = False  # Will be set to True on first level entry
        self.guide_shown_this_session = False  # Track if guide was shown
        self.guide_step = 1  # Current step (1-4)
        self.guide_button = {'x': 0, 'y': 0, 'w': 200, 'h': 60}  # "Next"/"Got it!" button, calculated dynamically
        
        # Real-time stitch tracking for progressive coloring
        self.completed_stitch_mask = None  # Accumulated mask of all detected stitches
        self.missed_stitch_mask = None     # Accumulated mask of off-pattern detections
        self.proximity_radius = 0  # Legacy compatibility radius (kept small)
        self.stitch_draw_radius = 0       # Kept for compatibility with existing tuning values
        self.stitch_box_half = 0          # Half-size of box stamp for accepted (correct) stitches
        self.wrong_stitch_check_radius = 20  # Detection circle for off-pattern check: line half-width + ~3 px margin
        self.missed_stitch_draw_radius = 6  # Kept for compatibility
        self.missed_stitch_box_half = 6     # Half-size of box stamp for wrong stitches
        # Make the outline/detection tolerance wider to be more lenient
        self.outline_thickness = 8             # Outline extends pattern mask outward by this many pixels to form the detection zone (adjustable in code)
        self.outline_detect_local_radius = 5  # Kept for compatibility
        self.outline_detect_local_half_size = 20  # Local half-size for contour-box detection around expected stitch
        self.outline_on_pattern_ratio_min = 0.1  # Correct stitch if >=20% of detected color pixels overlap the real pattern; otherwise mark wrong stitch
        self.min_color_pixels_center = 50   # Minimum matched color pixels for center contour-box detection
        self.min_color_pixels_outline = 50  # Minimum matched color pixels inside contour-box detection area
        self.min_color_contour_area_center = 18
        self.min_color_contour_area_outline = 18
        self.color_contour_padding = 3
        self.color_sample_half_size = 14
        self.blend_roi_half = 160               # Half-size (pixels) of the pattern-overlay blend window around the needle — reduce to improve FPS, increase to show more pattern context (adjustable in code)
        # Visual cyan expansion around accepted stitches (increase by 15%)
        self.cyan_spread_radius = int(math.ceil(20 * self.stitch_draw_radius * 1.15))
        self.progress_spread_radius = 5  # Progress expansion to bridge tiny gaps only
        # Keep these permissive so centerline-guided tracing can advance one
        # pixel step at a time on small screens/cameras.
        self.min_stitch_move_px = 1.0  # Min movement before accepting another center stitch
        self.stitch_cooldown_frames = 1  # Min frames between accepted center stitches
        self.stitch_frame_index = 0
        self.last_stitch_frame_index = -9999
        self.last_stitch_point = None

        # Lucas-Kanade sparse optical flow (retained for internal tracking, unused for movement)
        self.of_prev_gray = None        # Previous greyscale frame
        self.of_points = None           # Tracked feature points (Nx1x2 float32)
        self.of_roi_half = 64
        self.of_min_points = 8
        self.of_max_corners = 40
        self.of_max_level = 2
        # Accumulated overlay offset driven entirely by cloth optical flow
        self.pattern_offset_x = 0.0
        self.pattern_offset_y = 0.0
        self.pattern_offset_seeded = False
        # Skeleton-constrained needle movement
        self._cached_skeleton_path = None   # Ordered (x,y) points in pattern space
        self._cached_skeleton_f32  = None   # float32 copy for fast projection
        self._cached_skeleton_key  = None
        self.skeleton_idx_f   = 0.0         # Current position on skeleton (float)
        self.skeleton_offset_x = 0.0        # Fixed pattern→screen offset X
        self.skeleton_offset_y = 0.0        # Fixed pattern→screen offset Y
        self.skeleton_seeded  = False
        self.skeleton_completed_lock_idx = None  # Freeze index when pattern reaches 100%
        self.skeleton_completed_lock_point = None  # Freeze exact (x,y) pattern point at completion
        self.skeleton_tangent_lookahead = 10  # Points used for tangent estimation
        # Auto-move speed: steps per frame along the skeleton when thread color is detected.
        # Increase to move faster, decrease to move slower.
        self.auto_move_speed = 0.5
        # Short grace window so movement can pass tight turns even when
        # color detection flickers for a few frames on Pi camera noise.
        self.turn_move_grace_frames = 4
        self.turn_move_grace_left = 0
        # Speed control buttons (positions set dynamically in draw_score_panel)

        self.needle_on_pattern = True  # Whether the needle is currently on a pattern pixel

        # Cache processed blueprint overlays so we do not hit disk every frame.
        self._blueprint_cache = {}

        # Per-level derivative caches (cleared in reset_progress / on level change).
        # These are pure functions of the blueprint alpha, so they are computed once
        # and reused every frame — saves ~40 ms/frame on RPi4.
        self._cached_centerline     = None  # int32 path array from _build_centerline_path
        self._cached_centerline_f32 = None  # float32 copy for fast argmin
        self._cached_centerline_mask = None  # uint8 full centerline mask (all branches)
        self._cached_centerline_key = None
        self._cached_centerline_mask_key = None
        self._cached_pat_px         = None  # int32 x-coords of all pattern pixels
        self._cached_pat_py         = None  # int32 y-coords of all pattern pixels
        self._cached_pat_px_f32     = None  # float32 versions for distance calc
        self._cached_pat_py_f32     = None
        self._cached_corridor_mask  = None  # dilated binary corridor from _get_pattern_corridor_mask
        self._cached_outline_mask   = None  # dilated outline mask used as detection zone
        self._cached_realtime_pat   = None  # colored overlay from create_realtime_pattern
        self._realtime_pat_dirty    = True  # rebuild when True

        # Follow-line validation (on/off pattern while sewing)
        self.pattern_alpha_threshold = 0.5
        # Increase corridor radius so off-pattern detection is more lenient
        self.follow_corridor_radius = 60      # wider corridor = fewer wrong-stitch marks
        self.follow_order_tolerance = 24
        self.follow_centerline_distance = 50     # wider tolerance from centerline = fewer wrong-stitch marks
        self.snap_stitches_to_centerline = True
        self.centerline_step_limit = 6  # Max index jump per accepted center sample
        self.follow_off_confirm_frames = 4
        self.follow_on_confirm_frames = 2
        self.follow_off_count = 0
        self.follow_on_count = 0
        self.centerline_progress_idx = 0
        self.centerline_progress_initialized = False

        # ── Needle ROI – anchor to the visible camera centre by default ──
        self.NEEDLE_ROI_SIZE = 64   # Side-length of the square ROI (camera-frame pixels)
        self.NEEDLE_ROI_X    = int(self.camera_width // 2)
        self.NEEDLE_ROI_Y    = int(self.camera_height // 2)
        # ─────────────────────────────────────────────────────────────────────────────

        # Fixed needle marker position used by the overlay pipeline
        self.needle_pos_x = float(self.NEEDLE_ROI_X)
        self.needle_pos_y = float(self.NEEDLE_ROI_Y)
        self.prev_gray    = None   # Grayscale previous frame
        self.of_points    = None   # Lucas-Kanade tracking points

        # Segment tracking (divide pattern into 4 quarters)
        self.current_segment = 1  # 1=first 25%, 2=25-50%, 3=50-75%, 4=75-100%
        self.highest_segment_reached = 1  # Track highest segment to prevent going backwards
        self.segment_colors = {
            'completed': (255, 255, 0),    # Cyan in BGR (completed sections)
            'current': (0, 255, 0),        # Green in BGR (current section to sew)
            'upcoming': (100, 100, 100),   # Dim Gray (upcoming sections)
            'missed': (0, 0, 255),         # Red in BGR (off-pattern detections)
            'outline': (0, 140, 255),      # Bright orange BGR (detection tolerance zone surrounding pattern)
        }
        
        # Out-of-segment detection
        self.out_of_segment_warning = False
        self.warning_message = ""
        self.warning_flash_phase = 0
        self.warning_until_frame = 0
        # Combo and stitch tracking
        self.current_combo = 0
        self.max_combo = 0
        self.stitches_detected = 0
        
        # Load models from models directory
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Detection settings
        self.confidence_threshold = 0.9  # Lowered for INT8 model (produces lower confidence scores)
        self.iou_threshold = 0.6  # Intersection over Union threshold
        
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
                'hsv_ranges': [
                    ((0, 0, 5), (179, 100, 52)),
                ],
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

        # Cloth bbox tracking
        self.smooth_cloth_bbox = None  # Smoothed bbox for stable overlay

        # Evaluation model (stitch segmentation/detection for scoring)
        self.eval_model = None
        self.eval_input_name = None
        self.eval_output_names = []
        self.eval_input_size = 640
        self.eval_model_path = os.path.join(models_dir, 'evaluation.onnx')
        # Evaluation width-matching: make detected stitches match target mask thickness.
        self.eval_match_pattern_width = True
        self.eval_width_match_max_px = 8

        # Pattern mask dilation settings (modifiable)
        # Kernel is (width, height) — keep small (3,3) by default.
        # `pattern_dilate_iters` maps level -> iterations to dilate the raw mask
        # before visual/overlay conversion. Change these values to adjust
        # the visual thickness of the pattern lines per-level.
        self.pattern_dilate_kernel = (3, 3)
        self.pattern_dilate_iters = {
            1: 2,
            2: 1,
            3: 2,
            4: 2,
            5: 3,
        }
        self.pattern_dilate_default_iters = 1

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

    def load_evaluation_model(self, models_dir=None):
        """Load the stitch evaluation ONNX model (evaluation.onnx) if available."""
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
        model_path = os.path.join(models_dir, 'evaluation.onnx')
        self.eval_model_path = model_path

        if not os.path.exists(model_path):
            print(f"⚠ Evaluation model not found: {model_path}")
            self.eval_model = None
            return

        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 2
        sess_opts.intra_op_num_threads = 2
        self.eval_model = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=['CPUExecutionProvider'],
        )
        self.eval_input_name = self.eval_model.get_inputs()[0].name
        self.eval_output_names = [o.name for o in self.eval_model.get_outputs()]

        in_shape = self.eval_model.get_inputs()[0].shape
        if len(in_shape) >= 4 and isinstance(in_shape[2], int) and in_shape[2] > 0:
            self.eval_input_size = int(in_shape[2])
        else:
            self.eval_input_size = 640
        print(f"✓ Evaluation model loaded: {model_path} (imgsz={self.eval_input_size})")

    def unload_evaluation_model(self):
        """Unload evaluation model after scoring to keep runtime lightweight."""
        if self.eval_model is not None:
            self.eval_model = None
            self.eval_input_name = None
            self.eval_output_names = []

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

    def _extract_eval_mask_from_outputs(self, outputs, frame_w, frame_h):
        """Decode model outputs to a binary stitch mask in camera-frame coordinates."""
        combined_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        if outputs is None or len(outputs) == 0:
            return combined_mask

        preds = outputs[0]
        if preds is None:
            return combined_mask

        if preds.ndim == 3:
            preds = preds[0]
        if preds.ndim != 2:
            return combined_mask

        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        channels = preds.shape[1]
        if channels < 6:
            return combined_mask

        proto = None
        if len(outputs) > 1 and isinstance(outputs[1], np.ndarray) and outputs[1].ndim == 4:
            proto = outputs[1][0]
        n_mask = int(proto.shape[0]) if proto is not None else 0
        n_cls = channels - 4 - n_mask
        if n_cls <= 0:
            n_cls = 1
            n_mask = max(0, channels - 5)

        cls_scores = preds[:, 4:4 + n_cls]
        obj = cls_scores.max(axis=1)
        keep = obj >= max(0.05, self.confidence_threshold)
        if not np.any(keep):
            return combined_mask

        filtered = preds[keep]
        boxes_xywh = filtered[:, :4]
        scores = filtered[:, 4:4 + n_cls].max(axis=1)

        boxes = []
        for b in boxes_xywh:
            cx, cy, bw, bh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            x1 = max(0, cx - bw / 2.0)
            y1 = max(0, cy - bh / 2.0)
            boxes.append([x1, y1, max(1.0, bw), max(1.0, bh)])

        idxs = cv2.dnn.NMSBoxes(boxes, scores.tolist(), max(0.05, self.confidence_threshold), self.iou_threshold)
        if idxs is None or len(idxs) == 0:
            return combined_mask

        mask_coeffs = filtered[:, 4 + n_cls:4 + n_cls + n_mask] if n_mask > 0 else None
        in_size = float(self.eval_input_size)

        for idx in np.array(idxs).reshape(-1):
            bx, by, bw, bh = boxes[idx]
            x1 = int(np.clip(round(bx / in_size * frame_w), 0, frame_w - 1))
            y1 = int(np.clip(round(by / in_size * frame_h), 0, frame_h - 1))
            x2 = int(np.clip(round((bx + bw) / in_size * frame_w), 0, frame_w - 1))
            y2 = int(np.clip(round((by + bh) / in_size * frame_h), 0, frame_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            if proto is not None and mask_coeffs is not None and mask_coeffs.shape[1] == proto.shape[0]:
                coeff = mask_coeffs[idx]
                mh, mw = proto.shape[1], proto.shape[2]
                pred_mask = self._sigmoid(np.matmul(coeff, proto.reshape(proto.shape[0], -1))).reshape(mh, mw)
                pred_mask = cv2.resize(pred_mask, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
                bin_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                crop = np.zeros_like(bin_mask)
                crop[y1:y2, x1:x2] = bin_mask[y1:y2, x1:x2]
                combined_mask = np.maximum(combined_mask, crop)
            else:
                cv2.rectangle(combined_mask, (x1, y1), (x2, y2), 255, -1)

        return combined_mask

    def _run_evaluation_inference(self, camera_frame):
        """Run AI model on screenshot and return binary detected-stitches mask."""
        if self.eval_model is None:
            return None

        try:
            in_size = int(self.eval_input_size)
            resized = cv2.resize(camera_frame, (in_size, in_size))
            inp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))[np.newaxis]
            outputs = self.eval_model.run(None, {self.eval_input_name: inp})
            frame_h, frame_w = camera_frame.shape[:2]
            return self._extract_eval_mask_from_outputs(outputs, frame_w, frame_h)
        except Exception as e:
            print(f"⚠ Evaluation inference failed: {e}")
            return None

    def _map_camera_mask_to_pattern(self, camera_mask, x_offset, y_offset, pattern_w, pattern_h):
        """Map camera-space detections into pattern-space mask using current overlay offsets."""
        pattern_mask = np.zeros((pattern_h, pattern_w), dtype=np.uint8)
        if camera_mask is None:
            return pattern_mask

        dx1 = max(0, x_offset)
        dy1 = max(0, y_offset)
        dx2 = min(self.camera_width, x_offset + pattern_w)
        dy2 = min(self.camera_height, y_offset + pattern_h)
        if dx2 <= dx1 or dy2 <= dy1:
            return pattern_mask

        sx1 = max(0, -x_offset)
        sy1 = max(0, -y_offset)
        sx2 = sx1 + (dx2 - dx1)
        sy2 = sy1 + (dy2 - dy1)

        cam_crop = camera_mask[dy1:dy2, dx1:dx2]
        if cam_crop.size == 0:
            return pattern_mask
        pattern_mask[sy1:sy2, sx1:sx2] = (cam_crop > 0).astype(np.uint8) * 255
        return pattern_mask

    def load_blueprint(self, level):
        """Load binary mask for the pattern"""
        # Reset progress if level changed
        if level != self.current_level_tracking:
            self.reset_progress()
            self.current_level_tracking = level

        cache_key = (level, self.uniform_width, self.uniform_height)
        cached = self._blueprint_cache.get(cache_key)
        if cached is not None:
            return cached
        
        mask_path = os.path.join(self.blueprint_folder, f'level{level}_mask.png')
        if not os.path.exists(mask_path):
            return None, None
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None
        
        # Resize the mask to the desired dimensions
        mask = cv2.resize(mask, (self.uniform_width, self.uniform_height))

        # Cut the binary mask in half (keep top half) — skipped for levels 3, 4 & 5
        if level not in (2,3, 4, 5):
            mask = mask[:mask.shape[0] // 2, :]

        # Make level-specific width adjustments.
        # Level 1 stays narrow; levels 2/3 are widened so the lower tip area
        # has better pixel coverage and is easier to stitch through.
        if level == 1:
            # Reduce to ~28% of original width (adjustable)
            new_w = int(round(mask.shape[1] * 0.28))
        elif level == 2:
            # Widen level 2 to full available width for better lower-tip coverage.
            new_w = self.uniform_width
        elif level == 3:
            # Widen level 3 to full available width for better lower-tip coverage.
            new_w = self.uniform_width
        elif level == 5:
            # Level 5: use the full canvas width for maximum width
            new_w = self.uniform_width
        elif level == 4:
            # Level 4: use the full canvas width for a wide U shape
            new_w = self.uniform_width
        else:
            new_w = int(round(mask.shape[1] * 0.625))
        new_w = max(1, min(mask.shape[1], new_w))
        mask = cv2.resize(mask, (new_w, mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Move the pattern to the middle of the cloth while keeping it centered horizontally
        top_offset = (self.uniform_height - mask.shape[0]) // 2
        left_offset = (self.uniform_width - mask.shape[1]) // 2
        centered_mask = np.zeros((self.uniform_height, self.uniform_width), dtype=mask.dtype)
        centered_mask[top_offset:top_offset + mask.shape[0], left_offset:left_offset + mask.shape[1]] = mask
        mask = centered_mask

        # Dilate pattern lines to achieve consistent visual thickness across levels.
        # Each non-level-1 pattern gets one extra iteration vs the previous baseline
        # to produce a ~10% thickness increase.  Level 1 is intentionally left
        # unmodified (thin vertical column).
        try:
            kw = int(self.pattern_dilate_kernel[0])
            kh = int(self.pattern_dilate_kernel[1])
            kw = max(1, kw)
            kh = max(1, kh)
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
            iters = int(self.pattern_dilate_iters.get(level, self.pattern_dilate_default_iters))
            if iters > 0:
                mask = cv2.dilate(mask, dil_kernel, iterations=iters)
        except Exception:
            # Fallback to original hardcoded behaviour if config missing or invalid
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            if level == 5:
                mask = cv2.dilate(mask, dil_kernel, iterations=3)
            elif level in (3, 4):
                mask = cv2.dilate(mask, dil_kernel, iterations=2)
            elif level == 2:
                mask = cv2.dilate(mask, dil_kernel, iterations=1)
            elif level == 1:
                mask = cv2.dilate(mask, dil_kernel, iterations=2)
     


        # Convert mask to white overlay (pattern lines)
        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        alpha = mask / 255.0

        self._blueprint_cache[cache_key] = (overlay, alpha)
        return self._blueprint_cache[cache_key]

    def load_raw_blueprint_mask(self, level):
        """Load the blueprint mask for `level` but DO NOT apply the dilation/outline

        This returns a uint8 mask (0/255) at the same uniform size used by
        `load_blueprint`, but without the post-processing dilation that is used
        for visual/overlay purposes. Use this for precise evaluation scoring so
        that decisions are based on the original pattern pixels only.
        """
        mask_path = os.path.join(self.blueprint_folder, f'level{level}_mask.png')
        if not os.path.exists(mask_path):
            return None

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        # Resize to the uniform canvas used elsewhere
        mask = cv2.resize(mask, (self.uniform_width, self.uniform_height), interpolation=cv2.INTER_NEAREST)

        # Cut the binary mask in half (keep top half) — skipped for levels 3,4 & 5
        if level not in (2, 3, 4, 5):
            mask = mask[:mask.shape[0] // 2, :]

        # Apply the same level-specific width adjustments as load_blueprint
        if level == 1:
            new_w = int(round(mask.shape[1] * 0.28))
        elif level in (2, 3, 4, 5):
            new_w = self.uniform_width
        else:
            new_w = int(round(mask.shape[1] * 0.625))
        new_w = max(1, min(mask.shape[1], new_w))
        mask = cv2.resize(mask, (new_w, mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Center the (possibly width-adjusted) mask into the uniform canvas
        top_offset = (self.uniform_height - mask.shape[0]) // 2
        left_offset = (self.uniform_width - mask.shape[1]) // 2
        centered_mask = np.zeros((self.uniform_height, self.uniform_width), dtype=mask.dtype)
        centered_mask[top_offset:top_offset + mask.shape[0], left_offset:left_offset + mask.shape[1]] = mask

        # Return a clean binary mask (0/255) WITHOUT any dilation/outline
        return (centered_mask > 0).astype(np.uint8) * 255
    
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
        self.missed_stitch_mask = None
        self.stitch_frame_index = 0
        self.last_stitch_frame_index = -9999
        self.last_stitch_point = None
        self.of_prev_gray = None
        self.of_points = None
        self.pattern_offset_x = 0.0
        self.pattern_offset_y = 0.0
        self.pattern_offset_seeded = False
        self._cached_skeleton_path = None
        self._cached_skeleton_f32  = None
        self._cached_skeleton_key  = None
        # skeleton_idx_f = 0 means start at path[0], which is the bottom-left
        # for level 5 (skeleton starts from bottom) and top for other levels.
        self.skeleton_idx_f   = 0.0
        self.skeleton_seeded  = False
        self.skeleton_completed_lock_idx = None
        self.skeleton_completed_lock_point = None
        self.needle_pos_x = float(self.NEEDLE_ROI_X)
        self.needle_pos_y = float(self.NEEDLE_ROI_Y)
        self._color_overlay_counter = 0
        # Clear per-level derivative caches
        self._cached_centerline     = None
        self._cached_centerline_f32 = None
        self._cached_centerline_mask = None
        self._cached_centerline_key = None
        self._cached_centerline_mask_key = None
        self._cached_pat_px         = None
        self._cached_pat_py         = None
        self._cached_pat_px_f32     = None
        self._cached_pat_py_f32     = None
        self._cached_corridor_mask  = None
        self._cached_outline_mask   = None
        self._cached_realtime_pat   = None
        self._realtime_pat_dirty    = True
        self.centerline_progress_idx = 0
        self.centerline_progress_initialized = False
        self.needle_on_pattern = True
        self.follow_off_count = 0
        self.follow_on_count = 0
        self.turn_move_grace_left = 0
        self.sewing_started = False  # Require Start button press again on reset
        self.last_evaluation_screenshot_path = None
        self.last_pattern_projection = None
        self.eval_vis_detected = None
        self.eval_vis_mask = None
        self.eval_vis_comparison = None
        self.eval_screen_stage = 0
        print(f"🔄 Progress reset for Level {self.current_level}")

    def _evaluate_image_processing(self, camera_frame):
        """Detect thread-colored pixels in *camera_frame* using HSV color detection.

        This is the image-processing evaluation path.  It applies the currently
        selected thread-color HSV ranges to the full camera snapshot, closes small
        inter-stitch gaps with morphological operations to bridge the tiny spaces
        between individual stitches, and removes noise with an opening step.

        Returns a binary uint8 mask (0 / 255) in camera-frame coordinates where
        255 marks pixels identified as belonging to the sewn thread.
        """
        try:
            h, w = camera_frame.shape[:2]
            hsv = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
            color_cfg = self.color_profiles[self.selected_detection_color]
            mask = np.zeros((h, w), dtype=np.uint8)
            for lo, hi in color_cfg['hsv_ranges']:
                mask = cv2.bitwise_or(
                    mask,
                    cv2.inRange(hsv,
                                np.array(lo, dtype=np.uint8),
                                np.array(hi, dtype=np.uint8))
                )
            # Close small inter-stitch gaps so nearby stitch points merge.
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
            # Remove isolated noise blobs smaller than a few pixels.
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
            detected_px = int(np.count_nonzero(mask))
            print(f"🔍 IP detection: {detected_px} px of {color_cfg['label']} thread detected in frame")
            return mask
        except Exception as e:
            print(f"⚠ Image-processing evaluation failed: {e}")
            return np.zeros(camera_frame.shape[:2], dtype=np.uint8)

    def create_realtime_pattern(self, overlay, alpha, trace_y=None):
        """Create real-time colored pattern overlay.

        Yellow = unsewn pattern pixels, Cyan = stitched pixels.
        A pixel turns cyan only after expected thread-color detection marks it in
        completed_stitch_mask.
        """
        if overlay is None or alpha is None:
            return None

        # Start with a black canvas so the outline colour is visible against it.
        colored_overlay = np.zeros_like(overlay)
        height, width = overlay.shape[:2]
        pattern_pixels = alpha > 0.1

        # Draw outline detection zone first (tolerance ring around pattern).
        # Only pixels in the dilated outline but NOT on the pattern itself are coloured,
        # so the zone shows as a visible border. Pattern pixels are drawn on top.
        outline_mask = self._get_pattern_outline_mask(alpha, width, height)
        if outline_mask is not None:
            outline_border = np.logical_and(outline_mask > 0, ~pattern_pixels)
            colored_overlay[outline_border] = self.segment_colors['outline']

        # Base state: everything in the blueprint is still to be sewn (yellow).
        colored_overlay[pattern_pixels] = self.segment_colors['current']

        # Completed stitch mask paints only stitched blueprint pixels cyan.
        # No dilation or erosion — use the raw stamp mask so corners are
        # covered exactly where the needle passed and bleed to other branches
        # is prevented by keeping the stamp radius small (see _register_stitch).
        if self.completed_stitch_mask is not None:
            completed_mask_resized = cv2.resize(self.completed_stitch_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            completed_pattern_pixels = np.logical_and(pattern_pixels, completed_mask_resized > 0)
            colored_overlay[completed_pattern_pixels] = self.segment_colors['completed']

        # Missed stitch mask paints off-pattern detections red.
        if self.missed_stitch_mask is not None:
            missed_mask_resized = cv2.resize(self.missed_stitch_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            missed_pixels = missed_mask_resized > 0
            colored_overlay[missed_pixels] = self.segment_colors['missed']

        return colored_overlay

    def _matches_selected_color(self, cam_frame, cx_cam, cy_cam, hsv_frame=None):
        """Return True when a contour-box near the sample point matches the selected color."""
        color_cfg = self.color_profiles[self.selected_detection_color]

        sample_radius = int(max(8, self.color_sample_half_size))
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
            self.last_color_contour_box = None
            return False

        hsv_patch = None
        if hsv_frame is not None:
            hsv_patch = hsv_frame[y1:y2, x1:x2]
        combined_mask = self._get_selected_color_mask(patch, hsv_patch=hsv_patch)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ref_x = (x2 - x1) / 2.0
        ref_y = (y2 - y1) / 2.0
        best_contour = self._select_color_contour(
            contours,
            ref_x,
            ref_y,
            self.min_color_contour_area_center,
        )
        if best_contour is None:
            self.last_color_match_ratio = 0.0
            self.last_color_match = False
            self.last_color_mask = None
            self.last_color_mask_bounds = None
            self.last_color_contour_box = None
            return False

        box_mask, (bx1, by1, bx2, by2) = self._build_contour_box_mask(
            combined_mask.shape,
            best_contour,
            padding=self.color_contour_padding,
        )
        ink_mask = box_mask > 0
        ink_pixels = int(np.count_nonzero(ink_mask))
        if ink_pixels <= 0:
            self.last_color_match_ratio = 0.0
            self.last_color_match = False
            self.last_color_mask = None
            self.last_color_mask_bounds = None
            self.last_color_contour_box = None
            return False

        color_pixels = int(np.count_nonzero(combined_mask[ink_mask]))
        match_ratio = float(color_pixels) / float(ink_pixels)
        required_pixels = max(8, min(int(self.min_color_pixels_center), int(0.75 * ink_pixels)))

        self.last_color_match_ratio = match_ratio
        self.last_color_match = (
            color_pixels >= required_pixels
            and match_ratio >= color_cfg['min_ratio']
        )

        if bx2 > bx1 and by2 > by1:
            masked = cv2.bitwise_and(combined_mask, box_mask)
            self.last_color_mask = masked[by1:by2, bx1:bx2].copy()
            self.last_color_mask_bounds = (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)
            self.last_color_contour_box = self.last_color_mask_bounds
        else:
            self.last_color_mask = None
            self.last_color_mask_bounds = None
            self.last_color_contour_box = None

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

    def _select_color_contour(self, contours, ref_x, ref_y, min_area):
        """Pick the contour nearest the reference point with a minimum area."""
        best_contour = None
        best_score = None

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < float(min_area):
                continue
            m = cv2.moments(cnt)
            if m['m00'] == 0:
                continue
            cx = float(m['m10'] / m['m00'])
            cy = float(m['m01'] / m['m00'])
            dist2 = (cx - float(ref_x)) ** 2 + (cy - float(ref_y)) ** 2
            # Prefer contours near the expected point, break ties toward larger area.
            score = dist2 - (0.35 * area)
            if best_score is None or score < best_score:
                best_score = score
                best_contour = cnt

        if best_contour is not None:
            return best_contour

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < float(min_area):
            return None
        return largest

    def _build_contour_box_mask(self, mask_shape, contour, padding=0):
        """Build a filled rectangle mask around a contour's bounding box."""
        h, w = mask_shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)
        if contour is None:
            return out, (0, 0, 0, 0)

        x, y, bw, bh = cv2.boundingRect(contour)
        pad = int(max(0, padding))
        x1 = max(0, int(x) - pad)
        y1 = max(0, int(y) - pad)
        x2 = min(w, int(x + bw) + pad)
        y2 = min(h, int(y + bh) + pad)
        if x2 <= x1 or y2 <= y1:
            return out, (0, 0, 0, 0)

        out[y1:y2, x1:x2] = 255
        return out, (x1, y1, x2, y2)

    def _stamp_box(self, mask, cx, cy, half_size):
        """Paint a filled box stamp on a binary mask."""
        if mask is None:
            return

        h, w = mask.shape[:2]
        hs = max(1, int(half_size))
        x1 = max(0, int(cx) - hs)
        y1 = max(0, int(cy) - hs)
        x2 = min(w, int(cx) + hs + 1)
        y2 = min(h, int(cy) + hs + 1)
        if x2 <= x1 or y2 <= y1:
            return
        mask[y1:y2, x1:x2] = 255

    def _box_overlap_ratio(self, pattern_alpha, cx, cy, half_size):
        """Return fraction of a box stamp that lies on the pattern mask."""
        pat_h, pat_w = pattern_alpha.shape[:2]
        hs = max(1, int(half_size))
        x1 = max(0, int(cx) - hs)
        y1 = max(0, int(cy) - hs)
        x2 = min(pat_w, int(cx) + hs + 1)
        y2 = min(pat_h, int(cy) + hs + 1)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        region = pattern_alpha[y1:y2, x1:x2]
        total = int(region.size)
        if total <= 0:
            return 0.0
        on_pattern = int(np.count_nonzero(region > self.pattern_alpha_threshold))
        return float(on_pattern) / float(total)

    def _estimate_binary_stroke_width(self, binary_mask):
        """Estimate average stroke width of a binary mask in pixels."""
        if binary_mask is None:
            return 0.0
        m = (binary_mask > 0).astype(np.uint8)
        if int(np.count_nonzero(m)) < 32:
            return 0.0
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        vals = dist[m > 0]
        if vals.size == 0:
            return 0.0
        return float(np.median(vals) * 2.0)

    def _match_detected_mask_width_to_pattern(self, detected_mask, pattern_mask):
        """Adjust detected mask thickness to match pattern-mask thickness."""
        det_u8 = (detected_mask > 0).astype(np.uint8) * 255
        pat_u8 = (pattern_mask > 0).astype(np.uint8) * 255

        det_w = self._estimate_binary_stroke_width(det_u8)
        pat_w = self._estimate_binary_stroke_width(pat_u8)
        if det_w <= 0.0 or pat_w <= 0.0:
            return det_u8 > 0

        delta = pat_w - det_w
        px = int(round(abs(delta) / 2.0))
        px = min(max(0, px), int(max(0, getattr(self, 'eval_width_match_max_px', 8))))
        if px <= 0:
            return det_u8 > 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        if delta > 0:
            adjusted = cv2.dilate(det_u8, kernel, iterations=1)
            op = 'dilate'
        else:
            adjusted = cv2.erode(det_u8, kernel, iterations=1)
            op = 'erode'

        print(
            f"🔧 Eval width-match | detected≈{det_w:.2f}px pattern≈{pat_w:.2f}px "
            f"op={op} px={px}"
        )
        return adjusted > 0

    def _get_pattern_corridor_mask(self, pattern_alpha, actual_w, actual_h):
        """Return a dilated binary corridor around the blueprint path."""
        if self._cached_corridor_mask is not None:
            return self._cached_corridor_mask
        pat_crop = pattern_alpha[:actual_h, :actual_w]
        pat_binary = (pat_crop > self.pattern_alpha_threshold).astype(np.uint8)
        k = max(3, self.follow_corridor_radius * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._cached_corridor_mask = cv2.dilate(pat_binary, kernel, iterations=1)
        return self._cached_corridor_mask

    def _get_pattern_outline_mask(self, pattern_alpha, actual_w, actual_h):
        """Return a dilated binary mask that surrounds the blueprint pattern.

        The mask is expanded outward by self.outline_thickness pixels in all
        directions.  This forms the detection zone: stitches whose centre pixel
        falls inside this mask are accepted as on-pattern.

        Adjust self.outline_thickness (default 10) in __init__ to make the
        zone wider or narrower.
        """
        if self._cached_outline_mask is not None:
            return self._cached_outline_mask
        pat_crop = pattern_alpha[:actual_h, :actual_w]
        pat_binary = (pat_crop > self.pattern_alpha_threshold).astype(np.uint8)
        k = max(3, self.outline_thickness * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._cached_outline_mask = cv2.dilate(pat_binary, kernel, iterations=1)
        return self._cached_outline_mask

    def _get_centerline_mask(self, pattern_alpha, actual_w, actual_h):
        """Return a single-line center path mask derived from _build_skeleton_path."""
        cache_key = (int(self.current_level), int(actual_w), int(actual_h))
        if self._cached_centerline_mask is not None and self._cached_centerline_mask_key == cache_key:
            return self._cached_centerline_mask

        center = np.zeros((actual_h, actual_w), dtype=np.uint8)
        path = self._build_skeleton_path(pattern_alpha, actual_w, actual_h)
        if path is not None and len(path) > 0:
            for x, y in path:
                if 0 <= int(x) < actual_w and 0 <= int(y) < actual_h:
                    center[int(y), int(x)] = 255

        self._cached_centerline_mask = center
        self._cached_centerline_mask_key = cache_key
        return self._cached_centerline_mask

    def _build_centerline_path(self, pattern_alpha, actual_w, actual_h):
        """Build an ordered centerline path from the binary blueprint mask.

        For each occupied row, collapse contiguous pixel runs and choose the
        midpoint of the run nearest the previous row's midpoint. This keeps the
        path on the visual center of the overlay instead of hugging an edge.

        Levels 4 and 5 still start from the left-most branch, but subsequent rows
        continue along the midpoint of the nearest contiguous run.
        """
        # Reuse cached result — the blueprint never changes mid-level.
        cache_key = (int(self.current_level), int(actual_w), int(actual_h))
        if self._cached_centerline is not None and self._cached_centerline_key == cache_key:
            return self._cached_centerline
        pat_crop = pattern_alpha[:actual_h, :actual_w]
        pat_binary = (pat_crop > self.pattern_alpha_threshold).astype(np.uint8)
        path_points = []
        prev_x = None

        def _row_runs(xs):
            runs = []
            start = int(xs[0])
            prev = int(xs[0])
            for value in xs[1:]:
                value = int(value)
                if value != prev + 1:
                    runs.append((start, prev))
                    start = value
                prev = value
            runs.append((start, prev))
            return runs

        def _pick_run_midpoint(xs, prev_mid=None, force_leftmost=False):
            runs = _row_runs(xs)
            mids = [int(round((run[0] + run[1]) / 2.0)) for run in runs]
            if force_leftmost or prev_mid is None:
                return mids[0]
            return mids[int(np.argmin(np.abs(np.array(mids, dtype=np.int32) - int(prev_mid))))]

        # For level 5, start at the bottom-most row with the leftmost pixel and trace upward
        if self.current_level == 5:
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
            # Find the last (bottom-most) row that contains this leftmost x
            start_y = None
            for y, xs in reversed(row_data):
                if global_min_x in xs:
                    start_y = y
                    prev_x = _pick_run_midpoint(xs, force_leftmost=True)
                    path_points.append((prev_x, y))
                    break
            # Continue upward from that row
            for y in range(start_y - 1 if start_y is not None else actual_h - 1, -1, -1):
                xs = np.where(pat_binary[y] > 0)[0]
                if xs.size == 0:
                    continue
                x = _pick_run_midpoint(xs, prev_x)
                path_points.append((x, y))
                prev_x = x
        # For levels 2/3/4, start from the global left-most branch.
        elif self.current_level in (2, 3, 4):
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
                    prev_x = _pick_run_midpoint(xs, force_leftmost=True)
                    path_points.append((prev_x, y))
                    break
            # Continue from that row
            for y in range(start_y + 1 if start_y is not None else 0, actual_h):
                xs = np.where(pat_binary[y] > 0)[0]
                if xs.size == 0:
                    continue
                x = _pick_run_midpoint(xs, prev_x)
                path_points.append((x, y))
                prev_x = x
        else:
            # Standard path building for levels 1-3
            for y in range(actual_h):
                xs = np.where(pat_binary[y] > 0)[0]
                if xs.size == 0:
                    continue

                if prev_x is None:
                    x = _pick_run_midpoint(xs, force_leftmost=True)
                else:
                    x = _pick_run_midpoint(xs, prev_x)

                path_points.append((x, y))
                prev_x = x

        if not path_points:
            return None

        result = np.array(path_points, dtype=np.int32)
        self._cached_centerline     = result
        self._cached_centerline_f32 = result.astype(np.float32)
        self._cached_centerline_key = cache_key
        return result

    def _build_skeleton_path(self, pattern_alpha, actual_w, actual_h):
        """Build a center path directly from the overlay mask.

        - Level 1: row-midpoint scan
        - Levels 2/5: column-midpoint scan
        - Levels 3/4: true skeleton centerline path from endpoint to endpoint
        """
        cache_key = (int(self.current_level), int(actual_w), int(actual_h))
        if self._cached_skeleton_path is not None and self._cached_skeleton_key == cache_key:
            return self._cached_skeleton_path

        pat_crop = pattern_alpha[:actual_h, :actual_w]
        bin_mask = (pat_crop > self.pattern_alpha_threshold).astype(np.uint8)
        path_points = []

        def _contiguous_runs(values):
            runs = []
            s = int(values[0])
            p = s
            for vv in values[1:]:
                vv = int(vv)
                if vv > p + 1:
                    runs.append((s, p))
                    s = vv
                p = vv
            runs.append((s, p))
            return runs

        # Levels 3/4: use a single continuous skeleton centerline path.
        if self.current_level in (3, 4):
            skel = None
            dist_center = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 3)
            try:
                from skimage.morphology import skeletonize as _sk
                skel = _sk(bin_mask.astype(bool)).astype(np.uint8)
            except Exception:
                pass
            if skel is None:
                try:
                    skel = cv2.ximgproc.thinning(bin_mask * 255) // 255
                except Exception:
                    pass
            if skel is None:
                # Pure OpenCV morphological skeleton fallback for systems
                # without skimage / ximgproc (e.g. Raspberry Pi builds).
                img = (bin_mask.copy() * 255).astype(np.uint8)
                skel_img = np.zeros_like(img)
                element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                while True:
                    eroded = cv2.erode(img, element)
                    opened = cv2.dilate(eroded, element)
                    temp = cv2.subtract(img, opened)
                    skel_img = cv2.bitwise_or(skel_img, temp)
                    img = eroded
                    if cv2.countNonZero(img) == 0:
                        break
                skel = (skel_img > 0).astype(np.uint8)

            ys, xs = np.where(skel > 0)
            if ys.size > 0:
                node_set = set(zip(ys.tolist(), xs.tolist()))  # (y, x)

                def _neighbors(node):
                    y, x = node
                    out = []
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            nb = (y + dy, x + dx)
                            if nb in node_set:
                                out.append(nb)
                    return out

                endpoints = [node for node in node_set if len(_neighbors(node)) == 1]

                def _bfs_path(start_node, goal_node=None):
                    queue = [start_node]
                    parent = {start_node: None}
                    dist = {start_node: 0}
                    head = 0
                    farthest = start_node
                    while head < len(queue):
                        cur = queue[head]
                        head += 1
                        if goal_node is not None and cur == goal_node:
                            break
                        if dist[cur] > dist[farthest]:
                            farthest = cur
                        for nb in _neighbors(cur):
                            if nb not in parent:
                                parent[nb] = cur
                                dist[nb] = dist[cur] + 1
                                queue.append(nb)
                    return farthest, parent, dist

                if endpoints:
                    start_node = min(endpoints, key=lambda p: (p[0], p[1]))
                    # End at the right-most endpoint, tie-break toward the top.
                    end_node = max(endpoints, key=lambda p: (p[1], -p[0]))
                else:
                    start_node = min(node_set, key=lambda p: (p[0], p[1]))
                    end_node, _, _ = _bfs_path(start_node)

                # Shortest path from explicit start -> end keeps ordering stable
                # and avoids reversing/backtracking artifacts.
                _, parent, _ = _bfs_path(start_node, goal_node=end_node)

                rev = []
                cur = end_node
                while cur is not None:
                    rev.append(cur)
                    cur = parent[cur]
                rev.reverse()
                path_points = [(x, y) for (y, x) in rev]

                # Re-center each skeleton point onto the local stroke midpoint.
                # Raw thinning can bias slightly to one side; this snaps points
                # back to the visual center of the overlay thickness while
                # keeping the one-line order from the skeleton path.
                if path_points:
                    refined_points = []

                    def _pick_mid_from_runs(values, target):
                        runs = _contiguous_runs(values)
                        mids = [(a + b) // 2 for a, b in runs]
                        return min(mids, key=lambda m: abs(m - target))

                    path_len = len(path_points)
                    for idx, (x, y) in enumerate(path_points):
                        prev_x, prev_y = path_points[max(0, idx - 1)]
                        next_x, next_y = path_points[min(path_len - 1, idx + 1)]
                        dx = next_x - prev_x
                        dy = next_y - prev_y

                        # If tangent is more horizontal, recenter vertically in
                        # the current column. Otherwise recenter horizontally in
                        # the current row.
                        if abs(dx) >= abs(dy):
                            y_candidates = np.where(bin_mask[:, int(np.clip(x, 0, actual_w - 1))] > 0)[0]
                            if y_candidates.size > 0:
                                y = _pick_mid_from_runs(y_candidates, y)
                        else:
                            x_candidates = np.where(bin_mask[int(np.clip(y, 0, actual_h - 1)), :] > 0)[0]
                            if x_candidates.size > 0:
                                x = _pick_mid_from_runs(x_candidates, x)

                        refined_points.append((int(x), int(y)))

                    path_points = refined_points

                    # Final pass: project each point to the true local mask
                    # center by maximizing the distance-transform value along
                    # the local normal direction.
                    centered_points = []
                    path_len = len(path_points)
                    normal_radius = 8
                    for idx, (x, y) in enumerate(path_points):
                        if idx == 0 or idx == path_len - 1:
                            centered_points.append((int(x), int(y)))
                            continue
                        prev_x, prev_y = path_points[max(0, idx - 1)]
                        next_x, next_y = path_points[min(path_len - 1, idx + 1)]
                        tx = float(next_x - prev_x)
                        ty = float(next_y - prev_y)
                        norm = math.hypot(tx, ty)
                        if norm < 1e-5:
                            centered_points.append((int(x), int(y)))
                            continue

                        # Unit normal to the local tangent
                        nx = -ty / norm
                        ny = tx / norm

                        best_x = int(x)
                        best_y = int(y)
                        best_score = -1.0
                        best_offset = 0.0

                        for step in range(-normal_radius, normal_radius + 1):
                            sx = int(np.clip(round(x + nx * step), 0, actual_w - 1))
                            sy = int(np.clip(round(y + ny * step), 0, actual_h - 1))
                            if bin_mask[sy, sx] == 0:
                                continue
                            score = float(dist_center[sy, sx])
                            if score > best_score or (abs(score - best_score) < 1e-6 and abs(step) < abs(best_offset)):
                                best_score = score
                                best_x = sx
                                best_y = sy
                                best_offset = float(step)

                        centered_points.append((best_x, best_y))

                    path_points = centered_points

                    # Path already starts from the top-most endpoint via `seed`.

                # Fallback if endpoints are unavailable or path was not found
                if not path_points:
                    prev_mid = None
                    for y in range(actual_h):
                        xs = np.where(bin_mask[y, :] > 0)[0]
                        if xs.size == 0:
                            continue
                        runs = _contiguous_runs(xs)
                        mids = [(a + b) // 2 for a, b in runs]
                        if prev_mid is None:
                            mid = mids[0]
                        else:
                            mid = min(mids, key=lambda m: abs(m - prev_mid))
                        path_points.append((mid, y))
                        prev_mid = mid

        # Levels 2/5: scan columns (x changes, y follows nearest run midpoint)
        elif self.current_level in (2, 5):
            prev_mid = None
            for x in range(actual_w):
                ys = np.where(pat_crop[:, x] > self.pattern_alpha_threshold)[0]
                if ys.size == 0:
                    continue
                runs = _contiguous_runs(ys)
                mids = [(a + b) // 2 for a, b in runs]
                if prev_mid is None:
                    mid = mids[0]
                else:
                    mid = min(mids, key=lambda m: abs(m - prev_mid))
                path_points.append((x, mid))
                prev_mid = mid
        else:
            # Level 1: scan rows (y changes, x follows nearest run midpoint)
            prev_mid = None
            for y in range(actual_h):
                xs = np.where(pat_crop[y, :] > self.pattern_alpha_threshold)[0]
                if xs.size == 0:
                    continue
                runs = _contiguous_runs(xs)
                mids = [(a + b) // 2 for a, b in runs]
                if prev_mid is None:
                    mid = mids[0]
                else:
                    mid = min(mids, key=lambda m: abs(m - prev_mid))
                path_points.append((mid, y))
                prev_mid = mid

        if not path_points:
            return None
        result = np.array(path_points, dtype=np.int32)
        self._cached_skeleton_path = result
        self._cached_skeleton_f32  = result.astype(np.float32)
        self._cached_skeleton_key  = cache_key
        return result

    def _circle_fully_on_mask(self, pattern_alpha, cx, cy, radius):
        """Return True if every pixel of the circle (cx, cy, radius) is inside the
        pattern mask.  If any pixel is outside the mask the stitch is off-pattern."""
        pat_h, pat_w = pattern_alpha.shape[:2]
        r = int(radius)
        y0 = max(0, cy - r)
        y1 = min(pat_h, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(pat_w, cx + r + 1)
        if y1 <= y0 or x1 <= x0:
            return False
        yy, xx = np.mgrid[y0:y1, x0:x1]
        in_circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        if not np.any(in_circle):
            return False
        # If the circle extends to the image border, part of it is outside the mask.
        if cy - r < 0 or cy + r >= pat_h or cx - r < 0 or cx + r >= pat_w:
            return False
        alpha_vals = pattern_alpha[yy[in_circle], xx[in_circle]]
        return bool(np.all(alpha_vals > self.pattern_alpha_threshold))

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

            # For V/W patterns the row-by-row centerline only traces one arm, so
            # pixels on the other arm fail the distance check even though they are
            # clearly on the pattern.  Fall back to the corridor mask before
            # declaring a stitch invalid — corridor covers the full shape.
            if not on_centerline:
                _cmask = self._get_pattern_corridor_mask(pattern_alpha, actual_w, actual_h)
                if _cmask is not None and _cmask[py, px] > 0:
                    on_centerline = True   # pixel is on the actual pattern line

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

    def _update_segment_progress_from_raw(self):
        """Update segment state from current raw_progress using existing thresholds."""
        rp = self.raw_progress
        if   rp >= 75: new_seg = 5
        elif rp >= 50: new_seg = 4
        elif rp >= 25: new_seg = 3
        elif rp >= 10: new_seg = 2
        else:          new_seg = 1

        if new_seg > self.highest_segment_reached:
            self.highest_segment_reached = new_seg
        self.current_segment = min(self.highest_segment_reached, 4)
        # Show the real stitched coverage as the visible progress value.
        if rp > self.pattern_progress:
            self.pattern_progress = rp

    def _update_progress_from_path(self, current_idx, max_idx):
        """Drive raw progress directly from ordered path traversal."""
        if max_idx <= 0:
            path_progress = 100.0
        else:
            path_progress = float(np.clip((float(current_idx) / float(max_idx)) * 100.0, 0.0, 100.0))

        # Path index should be monotonic; keep progress non-decreasing.
        if path_progress < self.raw_progress:
            path_progress = self.raw_progress

        self.raw_progress = path_progress
        self._update_segment_progress_from_raw()

    def _update_progress_from_mask(self, pattern_alpha):
        """Recalculate raw_progress from completed (cyan) stitch coverage only.

        This keeps the progress bar aligned with what is actually filled in cyan
        on the pattern, preventing early 100% from red/off-pattern stitches or
        artificial dilation expansion.
        """
        if self.progress_from_path:
            return

        if self.completed_stitch_mask is None:
            return

        pattern_pixels = (pattern_alpha > 0.1)
        total = int(np.sum(pattern_pixels))
        if total == 0:
            return

        completed = self.completed_stitch_mask
        if completed.shape != pattern_alpha.shape[:2]:
            completed = cv2.resize(completed, (pattern_alpha.shape[1], pattern_alpha.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        covered = int(np.sum(np.logical_and(completed > 0, pattern_pixels)))
        self.raw_progress = min(100.0, covered / total * 100.0)
        print(f"📊 Raw Progress: {self.raw_progress:.1f}%")
        self._update_segment_progress_from_raw()

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
        # Warning flash phase (faster) when active
        if getattr(self, 'out_of_segment_warning', False):
            self.warning_flash_phase += 0.18
            # Auto-hide after warning_until_frame
            try:
                if self.stitch_frame_index >= getattr(self, 'warning_until_frame', 0):
                    self.out_of_segment_warning = False
                    self.warning_message = ""
            except Exception:
                pass
        
        # Draw current level display at top center
        level_text = f"LEVEL {self.current_level}"
        
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
        
        # (Difficulty text removed)
        
        # Draw back button (top left)
        self.draw_back_button(frame)
        
        # Draw camera feed
        self.draw_camera_feed(frame, camera_frame)
        
        # Draw score/stats panel (right) — includes legend + detected color
        self.draw_score_panel(frame)
        
        # Draw evaluate button (right, below stats)
        self.draw_evaluate_button(frame)

        # Draw thread colour panel (left, top)
        self.draw_thread_color_panel(frame)

        # (Cloth colour panel removed — cloth detection no longer used)
        
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
            # Keep a copy of the raw camera frame for screenshot-based evaluation.
            self.last_camera_frame = cam_frame.copy()
            detection_frame = cam_frame.copy()
            hsv_detection_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2HSV)
            self.stitch_frame_index += 1
            follow_check_ready = False
            follow_on_corridor = True
            follow_order_valid = True
            
            # Load pattern mask
            pattern_overlay, pattern_alpha = self.load_blueprint(self.current_level)

            # Pattern overlay — pattern stays fixed; needle (red dot) slides along skeleton.
            if pattern_overlay is not None and pattern_alpha is not None:
                overlay_h, overlay_w = pattern_overlay.shape[:2]

                # Prefer skeleton for movement so multi-branch levels (2/3/4/5)
                # can traverse the full pattern path. Fall back to centerline
                # when skeleton extraction is unavailable.
                centerline_path = self._build_centerline_path(pattern_alpha, overlay_w, overlay_h)
                skeleton_path = self._build_skeleton_path(pattern_alpha, overlay_w, overlay_h)
                movement_path = skeleton_path if (skeleton_path is not None and len(skeleton_path) > 0) else centerline_path
                self.debug_path_source = 'skeleton' if (skeleton_path is not None and len(skeleton_path) > 0) else ('centerline' if (centerline_path is not None and len(centerline_path) > 0) else 'none')
                # Use the same path for validation/snap as movement to avoid
                # mismatched indices and end-of-shape jumps.
                validation_path = movement_path
                expected_path_idx = None

                if movement_path is not None and len(movement_path) > 0:
                    skel_max_idx = len(movement_path) - 1

                    # Evaluate completion state FIRST — before seeding — so a
                    # momentary skeleton-unavailable frame that resets skeleton_seeded
                    # cannot cause the re-seed to contaminate idx_f with 0,
                    # which would record index 0 (path start) as the lock point.
                    _is_completed = (self.raw_progress >= 100.0) or (self.is_evaluated and self.level_completed)

                    # ── Seed once: fix pattern position on screen, start at skeleton[0] ──
                    if not self.skeleton_seeded:
                        if not self.sewing_started:
                            seed_idx = 0
                        elif _is_completed:
                            # Already finished — re-seed at the lock point so the
                            # needle stays on the final stitch and doesn't jump.
                            seed_idx = (self.skeleton_completed_lock_idx
                                        if self.skeleton_completed_lock_idx is not None
                                        else skel_max_idx)
                        elif self.current_level in (2, 3):
                            # Force a clean left-top start for Levels 2/3 every run.
                            seed_idx = 0
                        elif self.raw_progress <= 0.0:
                            seed_idx = 0
                        else:
                            seed_idx = int(np.clip(
                                round(self.raw_progress / 100.0 * skel_max_idx),
                                0, skel_max_idx))
                        self.skeleton_idx_f   = float(seed_idx)
                        # Position pattern so skeleton[seed_idx] sits at the default needle pos.
                        self.skeleton_offset_x = float(self.NEEDLE_ROI_X) - float(movement_path[seed_idx][0])
                        self.skeleton_offset_y = float(self.NEEDLE_ROI_Y) - float(movement_path[seed_idx][1])
                        self.skeleton_seeded = True
                        self.centerline_progress_initialized = True

                    # ── Auto-advance: color detection uses current dynamic pattern offset ──
                    # Compute where the pattern is sitting RIGHT NOW (before any advance)
                    # so the detection region matches the ink visible on camera.
                    moved_with_color = False
                    if _is_completed:
                        # On completion, keep the pattern centered on screen.
                        self.skeleton_idx_f = float(np.clip(self.skeleton_idx_f, 0.0, float(skel_max_idx)))
                    elif self.sewing_started:
                        _cur = int(np.clip(round(self.skeleton_idx_f), 0, skel_max_idx))
                        _ex  = int(movement_path[_cur][0])
                        _ey  = int(movement_path[_cur][1])
                        x_off = int(round(self.NEEDLE_ROI_X)) - _ex
                        y_off = int(round(self.NEEDLE_ROI_Y)) - _ey
                        _pat_h, _pat_w = pattern_alpha.shape[:2]
                        _dx1 = max(0, x_off)
                        _dy1 = max(0, y_off)
                        _dx2 = min(detection_frame.shape[1], x_off + _pat_w)
                        _dy2 = min(detection_frame.shape[0], y_off + _pat_h)
                        color_match_for_overlay_move = False
                        thread_overlaps_pattern = False
                        if _dx2 > _dx1 and _dy2 > _dy1:
                            _sx1 = max(0, -x_off)
                            _sy1 = max(0, -y_off)
                            _sx2 = _sx1 + (_dx2 - _dx1)
                            _sy2 = _sy1 + (_dy2 - _dy1)
                            _cam_region  = detection_frame[_dy1:_dy2, _dx1:_dx2]
                            _hsv_region  = hsv_detection_frame[_dy1:_dy2, _dx1:_dx2]
                            # Use the outline mask (dilated detection zone) as the
                            # ink region for color detection instead of the raw
                            # pattern alpha — so thread color is recognised anywhere
                            # within the tolerance border, not just on the pattern line.
                            _outline_full = self._get_pattern_outline_mask(pattern_alpha, _pat_w, _pat_h)
                            _outline_crop = _outline_full[_sy1:_sy2, _sx1:_sx2]
                            _color_mask = self._get_selected_color_mask(_cam_region, hsv_patch=_hsv_region)
                            _ink_mask = (_outline_crop > 0)
                            _ink_count = int(np.count_nonzero(_ink_mask))
                            if _ink_count > 0:
                                _color_in_ink = int(np.count_nonzero(_color_mask[_ink_mask]))
                                _ratio = float(_color_in_ink) / float(_ink_count)
                                _min_outline_pixels = int(self.min_color_pixels_outline)
                                if self.current_level == 3:
                                    _min_outline_pixels = max(10, int(round(_min_outline_pixels * 0.65)))
                                _required_px = max(8, min(_min_outline_pixels, int(0.75 * _ink_count)))
                                color_cfg = self.color_profiles[self.selected_detection_color]
                                self.last_color_match_ratio = _ratio
                                self.last_color_match = (
                                    _color_in_ink >= _required_px
                                    and _ratio >= color_cfg['min_ratio']
                                )
                                color_match_for_overlay_move = self.last_color_match
                                if self.last_color_match:
                                    _thread_mask = (_color_mask > 0)
                                    _pattern_mask_crop = (
                                        pattern_alpha[_sy1:_sy2, _sx1:_sx2] > self.pattern_alpha_threshold
                                    )
                                    if _pattern_mask_crop.shape == _thread_mask.shape:
                                        _thread_px = int(np.count_nonzero(_thread_mask))
                                        if _thread_px > 0:
                                            _thread_on_pattern = int(np.count_nonzero(
                                                np.logical_and(_thread_mask, _pattern_mask_crop)
                                            ))
                                            _thread_overlap_ratio = float(_thread_on_pattern) / float(_thread_px)
                                            thread_overlaps_pattern = (
                                                _thread_overlap_ratio >= float(self.outline_on_pattern_ratio_min)
                                            )
                                        else:
                                            thread_overlaps_pattern = False

                                # Persist color-in-outline debug mask for on-frame highlight.
                                _masked = cv2.bitwise_and(_color_mask, _color_mask, mask=_outline_crop)
                                self.last_color_mask = _masked.copy()
                                self.last_color_mask_bounds = (
                                    _dx1,
                                    _dy1,
                                    _dx2,
                                    _dy2,
                                )
                                self.last_color_contour_box = self.last_color_mask_bounds
                            else:
                                self.last_color_match_ratio = 0.0
                                self.last_color_match = False

                        # Local fallback at the fixed needle point helps on
                        # tight turns where region-based color checks can dip.
                        if not color_match_for_overlay_move:
                            if self._matches_selected_color(
                                detection_frame,
                                self.NEEDLE_ROI_X,
                                self.NEEDLE_ROI_Y,
                                hsv_frame=hsv_detection_frame,
                            ):
                                color_match_for_overlay_move = True

                        # Grace through brief color flicker so pathing doesn't
                        # stall at corners.
                        if color_match_for_overlay_move:
                            self.turn_move_grace_left = int(self.turn_move_grace_frames)
                        elif self.turn_move_grace_left > 0:
                            self.turn_move_grace_left -= 1
                            color_match_for_overlay_move = True

                        # Movement is color-gated: no color-on-pattern match, no overlay shift.
                        if color_match_for_overlay_move:
                            self.skeleton_idx_f = float(np.clip(
                                self.skeleton_idx_f + self.auto_move_speed,
                                0.0, float(skel_max_idx)))
                            moved_with_color = True

                    # ── Pattern scrolls under a fixed needle position ─────────────────
                    cur_idx = int(np.clip(round(self.skeleton_idx_f), 0, skel_max_idx))
                    exp_x   = int(movement_path[cur_idx][0])
                    exp_y   = int(movement_path[cur_idx][1])
                    if _is_completed:
                        exp_x = overlay_w // 2
                        exp_y = overlay_h // 2

                    if self.progress_from_path:
                        self._update_progress_from_path(cur_idx, skel_max_idx)

                    # Needle dot stays fixed; pattern slides so current skeleton
                    # point always sits beneath the fixed needle position.
                    self.needle_pos_x = float(self.NEEDLE_ROI_X)
                    self.needle_pos_y = float(self.NEEDLE_ROI_Y)

                    # Dynamic offset: current skeleton point → fixed screen position.
                    x_offset = int(round(self.NEEDLE_ROI_X)) - exp_x
                    y_offset = int(round(self.NEEDLE_ROI_Y)) - exp_y
                    self.pattern_offset_x = float(x_offset)
                    self.pattern_offset_y = float(y_offset)

                    # Sync centerline index for compatibility with downstream code.
                    self.centerline_progress_idx = cur_idx
                    expected_path_idx = cur_idx

                    # Needle is always on the skeleton (which is inside the mask).
                    raw_needle_in_pat_x = exp_x
                    raw_needle_in_pat_y = exp_y

                    # Keep movement color-gated. Once movement occurs, stamp
                    # completion at the current expected path point so cyan
                    # and progress visibly update even when per-frame overlap
                    # heuristics are noisy on Pi cameras.
                    if moved_with_color:
                        pat_h, pat_w = pattern_alpha.shape[:2]
                        _stamp_half = self.stitch_box_half
                        if self.completed_stitch_mask is None:
                            self.completed_stitch_mask = np.zeros((pat_h, pat_w), dtype=np.uint8)
                        self._stamp_box(self.completed_stitch_mask, exp_x, exp_y, _stamp_half)
                        self._realtime_pat_dirty = True
                        self._update_progress_from_mask(pattern_alpha)
                else:
                    # No usable movement path — keep pattern centered on completion;
                    # otherwise fall back to fixed centre.
                    self.skeleton_seeded = False
                    self.centerline_progress_initialized = False
                    _is_completed = (self.raw_progress >= 100.0) or (self.is_evaluated and self.level_completed)
                    exp_x = overlay_w // 2
                    exp_y = overlay_h // 2
                    x_offset = int(round(self.NEEDLE_ROI_X)) - exp_x
                    y_offset = int(round(self.NEEDLE_ROI_Y)) - exp_y

                # Fallback anchoring when skeleton is unavailable.
                if movement_path is None or len(movement_path) == 0:
                    x_offset = int(round(self.NEEDLE_ROI_X)) - exp_x
                    y_offset = int(round(self.NEEDLE_ROI_Y)) - exp_y
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

                # Save projection so evaluate can map AI detections back to the pattern.
                self.last_pattern_projection = {
                    'x_offset': int(x_offset),
                    'y_offset': int(y_offset),
                    'pattern_w': int(actual_w),
                    'pattern_h': int(actual_h),
                }

                if visible_w > 0 and visible_h > 0:
                    corridor_mask = self._get_pattern_corridor_mask(pattern_alpha, actual_w, actual_h)
                    # Rebuild the colored overlay only when stitch masks have changed.
                    if self._realtime_pat_dirty or self._cached_realtime_pat is None:
                        self._cached_realtime_pat = self.create_realtime_pattern(
                            pattern_overlay, pattern_alpha, trace_y=trace_y)
                        self._realtime_pat_dirty = False
                    realtime_pattern = self._cached_realtime_pat

                    pattern_src = realtime_pattern if realtime_pattern is not None else pattern_overlay

                    # ── Draw one-line guide path in pattern space before blending ──
                    _pat_display = pattern_src.copy()
                    if movement_path is not None and len(movement_path) > 1:
                        _pts = movement_path.astype(np.int32).reshape((-1, 1, 2))
                        _done_end = int(np.clip(cur_idx + 1, 0, len(movement_path)))
                        if _done_end > 1:
                            cv2.polylines(_pat_display, [_pts[:_done_end]], False, (0, 220, 220), 2, cv2.LINE_AA)
                        if _done_end < len(movement_path):
                            cv2.polylines(_pat_display, [_pts[max(0, _done_end - 1):]], False, (255, 255, 255), 2, cv2.LINE_AA)

                    # ── Blend full visible overlay ─────────────────────────────────
                    b_w = dst_x2 - dst_x1
                    b_h = dst_y2 - dst_y1
                    if b_w > 0 and b_h > 0:
                        roi = cam_frame[dst_y1:dst_y2, dst_x1:dst_x2]
                        pattern_crop = _pat_display[src_y1:src_y2, src_x1:src_x2]
                        alpha_crop = pattern_alpha[src_y1:src_y2, src_x1:src_x2]
                        # Outline pixels sit outside the PNG alpha (alpha=0), so extend
                        # the blend alpha to include the outline zone at full weight.
                        _outline_blend = self._get_pattern_outline_mask(pattern_alpha, actual_w, actual_h)
                        if _outline_blend is not None:
                            _outline_crop_b = _outline_blend[src_y1:src_y2, src_x1:src_x2].astype(np.float32) / 255.0
                            blend_alpha = np.maximum(alpha_crop, _outline_crop_b)
                        else:
                            blend_alpha = alpha_crop
                        # Vectorized 3-channel alpha blend
                        a3 = blend_alpha[:, :, np.newaxis] * 0.8 * float(self.pattern_visual_opacity)
                        roi_blended = a3 * pattern_crop + (1.0 - a3) * roi
                        cam_frame[dst_y1:dst_y2, dst_x1:dst_x2] = roi_blended.astype(np.uint8)
                    # ── Draw movement path guide dots on whole overlay ──────────────
                    if movement_path is not None and len(movement_path) > 0:
                        _path_len = len(movement_path)
                        _dot_step = max(1, _path_len // 100)
                        for _pi in range(0, _path_len, _dot_step):
                            _px = int(movement_path[_pi][0]) + x_offset
                            _py = int(movement_path[_pi][1]) + y_offset
                            if 0 <= _px < cam_frame.shape[1] and 0 <= _py < cam_frame.shape[0]:
                                if _pi <= cur_idx:
                                    cv2.circle(cam_frame, (_px, _py), 2, (0, 220, 220), -1)  # cyan = done
                                else:
                                    cv2.circle(cam_frame, (_px, _py), 2, (255, 255, 255), -1)  # white = to do
                    # ─────────────────────────────────────────────────────────────

                    # Use the updated index immediately for follow validation.
                    if validation_path is not None and len(validation_path) > 0:
                        expected_path_idx = int(np.clip(self.centerline_progress_idx, 0, len(validation_path) - 1))

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
                        centerline_path=validation_path,
                        expected_path_idx=expected_path_idx
                    )
                    self._update_follow_hysteresis(follow_valid, center_color_match)
                    follow_check_ready = True

            # Outside-pattern warning (color-gated + hysteresis stabilized)
            if follow_check_ready and not self.needle_on_pattern and self.raw_progress >= 2.0:
                ow_h = 36
                ow_y = 60
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

            color_cfg = self.color_profiles[self.selected_detection_color]
            if self.last_color_match:
                marker_text = f"{color_cfg['label']} detected - moving"
                marker_color = (0, 220, 0)
            else:
                marker_text = f"No {color_cfg['label'].lower()} ({self.last_color_match_ratio * 100:.0f}%)"
                marker_color = (0, 0, 255)

            # Highlight matched color pixels in the sampled patch.
            self._color_overlay_counter = (self._color_overlay_counter + 1) % self.color_overlay_interval
            if self._color_overlay_counter == 0 and self.last_color_mask is not None and self.last_color_mask_bounds is not None:
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

            # Draw a filled square marker for the needle.
            cx = int(self.needle_pos_x)
            cy = int(self.needle_pos_y)
            half = 3
            cv2.rectangle(cam_frame, (cx - half, cy - half), (cx + half, cy + half), marker_color, -1)

            marker_scale = text_scale(0.52, self.width, self.height, floor=0.46, ceiling=0.62)
            marker_thick = text_thickness(1, self.width, self.height, min_thickness=1, max_thickness=2)
            marker_left = 8
            marker_scale = fit_text_scale(marker_text, FONT_MAIN, self.camera_width - marker_left - 6, marker_scale, marker_thick, min_scale=0.4)
            draw_text(
                cam_frame,
                marker_text,
                marker_left,
                self.camera_height - 8,
                marker_scale,
                marker_color,
                marker_thick,
                font=FONT_MAIN,
                outline_color=(0, 0, 0),
                outline_extra=1,
            )

            # If progress reached 100%, show a prominent overlay on the camera
            if getattr(self, 'raw_progress', 0.0) >= 100.0:
                instr = "Overlap your work with the pattern"
                inst_scale = text_scale(0.9, self.width, self.height, floor=0.7, ceiling=1.0)
                inst_thick = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=4)
                inst_scale = fit_text_scale(instr, FONT_DISPLAY, self.camera_width - 40, inst_scale, inst_thick, min_scale=0.5)
                (tw, th), _ = get_text_size(instr, FONT_DISPLAY, inst_scale, inst_thick)
                pad_x = 18
                pad_y = 55
                rx = max(8, (self.camera_width - tw) // 2 - pad_x)
                ry = 12
                rw = min(self.camera_width - 16, tw + pad_x * 2)
                rh = th + pad_y * 2
                overlay = cam_frame.copy()
                cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (10, 10, 10), -1)
                cv2.addWeighted(overlay, 0.65, cam_frame, 0.35, 0, cam_frame)
                draw_text(
                    cam_frame,
                    instr,
                    rx + (rw - tw) // 2,
                    ry + (rh + th) // 2 - 4,
                    inst_scale,
                    self.COLORS['bright_blue'],
                    inst_thick,
                    font=FONT_DISPLAY,
                    outline_color=self.COLORS['glow_cyan'],
                    outline_extra=2,
                )
            
            frame[self.camera_y:self.camera_y+self.camera_height, 
                  self.camera_x:self.camera_x+self.camera_width] = cam_frame
            
            # Draw zoom buttons (small squares top-right of camera)
            for btn, label in ((self.zoom_out_button, '-'), (self.zoom_in_button, '+')):
                bx, by, bw, bh = btn['x'], btn['y'], btn['w'], btn['h']
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), self.COLORS['button_normal'], -1)
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), self.COLORS['text_primary'], 2)
                # Center label
                (tw, th), _ = get_text_size(label, FONT_MAIN, 0.9, 2)
                tx = bx + (bw - tw) // 2
                ty = by + (bh + th) // 2
                self._put_text(frame, label, tx, ty, 0.9, self.COLORS['text_primary'], 2)

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
                ("Start on the red box on the pattern", self.COLORS['text_secondary'], 0.7, 2),
                ("and sew.", self.COLORS['text_secondary'], 0.7, 2),
            ]
        elif self.guide_step == 2:
            instructions = [
                ("COLOR GUIDE", self.COLORS['text_primary'], 0.85, 2),
                "",
                ("CYAN = Completed stitches", self.COLORS['text_secondary'], 0.7, 2),
                "",
                ("RED = Incorrect stitches", self.COLORS['text_secondary'], 0.7, 2),
                "",
                ("GREEN = Pattern to sew", self.COLORS['text_secondary'], 0.7, 2),
                "",
                ("The pattern progressively change color!", self.COLORS['text_secondary'], 0.7, 2),
            ]
        elif self.guide_step == 3:
            instructions = [
                ("PROGRESSIVE FEEDBACK", self.COLORS['text_primary'], 0.85, 2),
                "",
                ("Only stitches near pattern", self.COLORS['text_secondary'], 0.7, 2),
                ("will change to cyan.", self.COLORS['text_secondary'], 0.7, 2),
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
                ("Reach 100% progress to unlock", self.COLORS['text_secondary'], 0.7, 2),
                (f"EVALUATE, then score {self.level_pass_thresholds.get(self.current_level, 80.0):.0f}%+ to pass.", self.COLORS['text_secondary'], 0.7, 2),
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
            # Keep this legacy path only when mask-based progress mode is active.
            if not self.progress_from_path:
                pattern_pixels = np.sum(pattern_mask_roi > 0)
                if pattern_pixels > 0:
                    covered_pixels = np.sum(np.logical_and(roi_combined_mask > 0, pattern_mask_roi > 0))
                    raw_progress = min(100.0, (covered_pixels / pattern_pixels) * 100.0)
                    self.raw_progress = raw_progress
                    print(f"📊 Raw Progress: {raw_progress:.1f}%")  # Debug print
                    self._update_segment_progress_from_raw()
            
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
        self._put_text(frame, "", content_x, leg_y, label_scale, self.COLORS['text_secondary'], label_thick)
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

        # Draw Start button inside the stats panel when sewing has not begun
        if not self.sewing_started:
            self.draw_start_button(frame)

    def draw_start_button(self, frame):
        """Draw the Start button inside the stats panel.

        While sewing has not started, cloth and thread-color detection are
        disabled so the user can align the camera without accidentally
        triggering progress.  Pressing Start enables detection.
        """
        sb = self.start_button
        pulse = 0.5 + 0.35 * abs(math.sin(self.glow_phase * 1.4))
        self.draw_glow_rect(frame, sb['x'], sb['y'], sb['w'], sb['h'],
                            (0, 180, 50), pulse)
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (sb['x'] + 2, sb['y'] + 2),
                      (sb['x'] + sb['w'] - 2, sb['y'] + sb['h'] - 2),
                      (0, 100, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        text = "Start"
        font_scale = text_scale(0.84, self.width, self.height, floor=0.76, ceiling=0.96)
        thickness  = text_thickness(2, self.width, self.height, min_thickness=2, max_thickness=3)
        font_scale = fit_text_scale(text, FONT_DISPLAY, sb['w'] - 14, font_scale, thickness, min_scale=0.66)
        (text_w, text_h), _ = get_text_size(text, FONT_DISPLAY, font_scale, thickness)
        text_x = sb['x'] + (sb['w'] - text_w) // 2
        text_y = sb['y'] + (sb['h'] + text_h) // 2
        self._put_text(frame, text, text_x, text_y, font_scale,
                       self.COLORS['text_primary'], thickness, font=FONT_DISPLAY)

    def draw_evaluate_button(self, frame):
        """Draw the evaluate button below stats panel"""
        eb = self.evaluate_button
        # Only show evaluate button when progress reached 100% and not
        # already evaluated. The evaluate action is gated by full progress.
        if self.is_evaluated or self.raw_progress < 100.0:
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

        # Stage 0: show comparison preview first (centered), then NEXT to view score.
        if getattr(self, 'eval_screen_stage', 0) == 0:
            title = "COMPARISON PREVIEW"
            title_scale = text_scale(1.05, self.width, self.height, floor=0.92, ceiling=1.18)
            title_thick = text_thickness(3, self.width, self.height, min_thickness=2, max_thickness=4)
            title_scale = fit_text_scale(title, FONT_DISPLAY, self.width - 60, title_scale, title_thick, min_scale=0.84)
            (tw, th), _ = get_text_size(title, FONT_DISPLAY, title_scale, title_thick)
            tx = (self.width - tw) // 2
            ty = 56
            draw_text(
                frame,
                title,
                tx,
                ty,
                title_scale,
                self.COLORS['bright_blue'],
                title_thick,
                font=FONT_DISPLAY,
                outline_color=self.COLORS['glow_cyan'],
                outline_extra=2,
            )

            # Center comparison image in the middle of the screen.
            cmp_img = self.eval_vis_comparison
            if cmp_img is not None:
                avail_w = self.width - 120
                avail_h = self.height - 230
                h, w = cmp_img.shape[:2]
                scale = min(float(avail_w) / float(max(1, w)), float(avail_h) / float(max(1, h)))
                draw_w = max(1, int(w * scale))
                draw_h = max(1, int(h * scale))
                comp = cv2.resize(cmp_img, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
                ix = (self.width - draw_w) // 2
                iy = 90 + (avail_h - draw_h) // 2

                cv2.rectangle(frame, (ix - 4, iy - 4), (ix + draw_w + 4, iy + draw_h + 4), (15, 10, 8), -1)
                cv2.rectangle(frame, (ix - 4, iy - 4), (ix + draw_w + 4, iy + draw_h + 4), (100, 200, 100), 2)
                frame[iy:iy + draw_h, ix:ix + draw_w] = comp
            else:
                msg = "No comparison image available"
                msg_scale = text_scale(0.8, self.width, self.height, floor=0.7, ceiling=0.9)
                msg_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
                (mw, mh), _ = get_text_size(msg, FONT_MAIN, msg_scale, msg_thick)
                mx = (self.width - mw) // 2
                my = self.height // 2 + mh // 2
                self._put_text(frame, msg, mx, my, msg_scale, self.COLORS['text_secondary'], msg_thick)

            # NEXT button to move to score stage.
            nb_w = 200
            nb_h = 56
            nb_x = (self.width - nb_w) // 2
            nb_y = self.height - nb_h - 18
            self.eval_next_button['x'] = nb_x
            self.eval_next_button['y'] = nb_y
            self.eval_next_button['w'] = nb_w
            self.eval_next_button['h'] = nb_h

            pulse = 0.5 + 0.35 * abs(math.sin(self.glow_phase * 1.2))
            self.draw_glow_rect(frame, nb_x, nb_y, nb_w, nb_h, self.COLORS['neon_blue'], pulse)
            _ov = frame.copy()
            cv2.rectangle(_ov, (nb_x + 2, nb_y + 2), (nb_x + nb_w - 2, nb_y + nb_h - 2), self.COLORS['button_hover'], -1)
            cv2.addWeighted(_ov, 0.8, frame, 0.2, 0, frame)

            ntext = "NEXT"
            ns = text_scale(0.92, self.width, self.height, floor=0.8, ceiling=1.04)
            nt = text_thickness(3, self.width, self.height, min_thickness=2, max_thickness=4)
            ns = fit_text_scale(ntext, FONT_DISPLAY, nb_w - 18, ns, nt, min_scale=0.72)
            (nw, nh), _ = get_text_size(ntext, FONT_DISPLAY, ns, nt)
            ntx = nb_x + (nb_w - nw) // 2
            nty = nb_y + (nb_h + nh) // 2
            self._put_text(frame, ntext, ntx, nty, ns, self.COLORS['text_primary'], nt, font=FONT_DISPLAY)
            return
        
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
        
        # Show final score and wrong-stitch percentage (replace raw progress)
        content_y += 60
        # Large score display
        score_text = f"{self.final_score:.1f}%"
        score_scale = text_scale(1.45, self.width, self.height, floor=1.22, ceiling=1.65)
        score_thick = text_thickness(3, self.width, self.height, min_thickness=2, max_thickness=4)
        (score_w, _), _ = get_text_size(score_text, FONT_DISPLAY, score_scale, score_thick)
        score_x = panel_x + (panel_w - score_w) // 2
        score_col = self.COLORS['glow_cyan'] if self.level_completed else self.COLORS['text_primary']
        self._put_text(frame, score_text, score_x, content_y, score_scale, score_col, score_thick, font=FONT_DISPLAY)

        # Wrong-stitch percentage (smaller beneath score)
        content_y += 60
        wrong_text = f""
        wrong_scale = text_scale(0.72, self.width, self.height, floor=0.66, ceiling=0.86)
        wrong_thick = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        (wt_w, _), _ = get_text_size(wrong_text, FONT_MAIN, wrong_scale, wrong_thick)
        wrong_x = panel_x + (panel_w - wt_w) // 2
        self._put_text(frame, wrong_text, wrong_x, content_y, wrong_scale, self.COLORS['text_secondary'], wrong_thick)
        
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

        # Zoom buttons
        zib = self.zoom_in_button
        zob = self.zoom_out_button
        if zib['x'] <= x <= zib['x'] + zib['w'] and zib['y'] <= y <= zib['y'] + zib['h']:
            self.play_button_click_sound()
            return 'zoom_in'
        if zob['x'] <= x <= zob['x'] + zob['w'] and zob['y'] <= y <= zob['y'] + zob['h']:
            self.play_button_click_sound()
            return 'zoom_out'

        # Start button — enable detection so user can begin sewing
        sb = self.start_button
        if (not self.sewing_started
                and sb['x'] <= x <= sb['x'] + sb['w']
                and sb['y'] <= y <= sb['y'] + sb['h']):
            self.play_button_click_sound()
            self.sewing_started = True
            print("▶️  Sewing started — color detection enabled")
            return None
        
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

        if not self.is_evaluated and self.raw_progress >= 100.0:
            eb = self.evaluate_button
            if eb['x'] <= x <= eb['x'] + eb['w'] and eb['y'] <= y <= eb['y'] + eb['h']:
                self.play_button_click_sound()
                self.evaluate_pattern()
                return None

        # Evaluation stage-next button: comparison preview -> score modal
        if self.is_evaluated and getattr(self, 'eval_screen_stage', 0) == 0:
            nb = self.eval_next_button
            if nb['x'] <= x <= nb['x'] + nb['w'] and nb['y'] <= y <= nb['y'] + nb['h']:
                self.play_button_click_sound()
                self.eval_screen_stage = 1
                return None
        
        # Check try again button (only if evaluated and failed)
        if self.is_evaluated and getattr(self, 'eval_screen_stage', 0) == 1 and not self.level_completed:
            tb = self.try_again_button
            if tb['x'] <= x <= tb['x'] + tb['w'] and tb['y'] <= y <= tb['y'] + tb['h']:
                self.play_button_click_sound()
                self.unload_evaluation_model()
                # Fully reset all progress and caches
                self.reset_progress()
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
        if self.is_evaluated and getattr(self, 'eval_screen_stage', 0) == 1 and self.level_completed:
            nb = self.next_level_button
            if nb['x'] <= x <= nb['x'] + nb['w'] and nb['y'] <= y <= nb['y'] + nb['h']:
                self.play_button_click_sound()
                self.unload_evaluation_model()
                return 'next_level'
        
        return None
    
    def evaluate_pattern(self):
        """Take a screenshot of the camera frame, compare it to the level blueprint mask
        using image processing, and compute a stitch-quality score.

        Detection pipeline:
          1. Save a snapshot of the current camera frame.
          2. Apply the selected thread-color HSV mask to the snapshot to locate all
             thread pixels (image-processing path).  If an AI ONNX model is also
             loaded it is tried first; the IP method is used as primary/fallback.
          3. Supplement with the in-session accumulated stitch mask so any stitches
             tracked in real time but not visible in the snapshot are also counted.
          4. Map the combined detected mask into pattern (blueprint) coordinates.
          5. Score: coverage % = on-pattern detections / total pattern pixels.
             Wrong % = off-pattern detections / total detections.
                 Final score = coverage × (1 − wrong/100).  Pass threshold depends on level.
        """
        print("\n" + "=" * 72)
        print(f"🧪 EVALUATE START | Level {self.current_level}")

        if self.last_camera_frame is None:
            print("⚠ Cannot evaluate: no camera frame available yet.")
            print("🧪 EVALUATE ABORTED")
            print("=" * 72)
            return
        if self.last_pattern_projection is None:
            print("⚠ Cannot evaluate: pattern projection not ready yet.")
            print("🧪 EVALUATE ABORTED")
            print("=" * 72)
            return

        # 1) Detect stitches from the current camera snapshot.
        #    Try AI model first (if available); fall through to image processing.
        detected_camera_mask = None
        if self.eval_model is not None:
            try:
                ai_mask = self._run_evaluation_inference(self.last_camera_frame)
                if ai_mask is not None and int(np.count_nonzero(ai_mask > 0)) > 0:
                    detected_camera_mask = ai_mask
                    print(f"✓ AI model detection used ({int(np.count_nonzero(ai_mask > 0))} px)")
            except Exception as e:
                print(f"⚠ AI inference error: {e}")

        if detected_camera_mask is None or not np.any(detected_camera_mask > 0):
            print("🔍 Using color-based image processing for stitch detection")
            detected_camera_mask = self._evaluate_image_processing(self.last_camera_frame)

        # Ensure we have a valid mask array even if detection produced nothing.
        if detected_camera_mask is None:
            detected_camera_mask = np.zeros(self.last_camera_frame.shape[:2], dtype=np.uint8)

        # Create a "cleaned" camera image: clear everything except detected stitch pixels
        try:
            cleaned_cam = np.zeros_like(self.last_camera_frame)
            # Slight dilation so stitches are more visible when isolated
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dil = cv2.dilate((detected_camera_mask > 0).astype(np.uint8) * 255, k, iterations=1)
            mask_bool = dil > 0
            cleaned_cam[mask_bool] = self.last_camera_frame[mask_bool]
        except Exception:
            cleaned_cam = self.last_camera_frame.copy()

        # 2) Save the cleaned screenshot (only stitch visible) for records.
        captures_dir = os.path.join(os.path.dirname(__file__), 'captures', 'evaluations')
        os.makedirs(captures_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = os.path.join(captures_dir, f'level{self.current_level}_{ts}.png')
        cv2.imwrite(screenshot_path, cleaned_cam)
        self.last_evaluation_screenshot_path = screenshot_path
        print(f"📸 Cleaned screenshot saved: {screenshot_path}")
        print(f"🧪 Frame size: {self.last_camera_frame.shape[1]}x{self.last_camera_frame.shape[0]}")

        # 3) Load the target blueprint mask for this level.
        try:
            _, pattern_alpha = self.load_blueprint(self.current_level)
        except Exception:
            pattern_alpha = None
        if pattern_alpha is None:
            print("⚠ Evaluation failed: blueprint mask unavailable.")
            print("🧪 EVALUATE ABORTED")
            print("=" * 72)
            return

        # 4) Map detected camera-space pixels → pattern space.
        proj = self.last_pattern_projection
        print(
            f"🧪 Projection | x_offset={int(proj['x_offset'])} y_offset={int(proj['y_offset'])} "
            f"pattern={int(proj['pattern_w'])}x{int(proj['pattern_h'])}"
        )
        detected_pattern_mask = self._map_camera_mask_to_pattern(
            detected_camera_mask,
            int(proj['x_offset']),
            int(proj['y_offset']),
            int(proj['pattern_w']),
            int(proj['pattern_h']),
        )
        camera_detected_px = int(np.count_nonzero(detected_camera_mask > 0))
        pattern_detected_px = int(np.count_nonzero(detected_pattern_mask > 0))
        print(f"🧪 Detections | camera_px={camera_detected_px} mapped_pattern_px={pattern_detected_px}")

        # 5) Use detected mask only for evaluation (no session-overlay merge).
        #    This prevents UI-tracked colors from influencing scoring.
        det_mask = (detected_pattern_mask > 0)

        # 6) Compare detected stitches against the blueprint mask and score.
        # Prefer a raw, non-dilated blueprint mask for precise scoring so
        # that detection is judged against the original pixels in the
        # blueprint. If unavailable, fall back to the processed alpha.
        raw_blueprint = None
        try:
            raw_blueprint = self.load_raw_blueprint_mask(self.current_level)
        except Exception:
            raw_blueprint = None

        if raw_blueprint is not None:
            pat_mask = (raw_blueprint > 0)
        else:
            pat_mask = (pattern_alpha > self.pattern_alpha_threshold)
        # Crop both masks to the same region in case of minor size differences.
        h_cmp = min(pat_mask.shape[0], det_mask.shape[0])
        w_cmp = min(pat_mask.shape[1], det_mask.shape[1])
        pat_cmp = pat_mask[:h_cmp, :w_cmp]
        det_cmp = det_mask[:h_cmp, :w_cmp]

        # Optionally normalize detected stroke thickness to match the pattern
        # mask thickness before scoring.
        if getattr(self, 'eval_match_pattern_width', True):
            det_cmp = self._match_detected_mask_width_to_pattern(det_cmp, pat_cmp)

        det_cmp_bool = (det_cmp > 0)
        total_pattern = int(np.count_nonzero(pat_cmp))
        det_in_overlay = np.logical_and(det_cmp_bool, pat_cmp)
        on_pattern = int(np.count_nonzero(det_in_overlay))
        uncovered_pattern = max(0, total_pattern - on_pattern)
        total_detected = on_pattern
        off_pattern = 0
        print(
            f"🧪 Pixel stats | total_pattern={total_pattern}  covered={on_pattern} "
            f"uncovered={uncovered_pattern}"
        )

        # Coverage score = percentage of mask covered by detected stitches.
        coverage_pct = (float(on_pattern) / float(total_pattern) * 100.0) if total_pattern > 0 else 0.0
        wrong_pct = 0.0

        self.evaluation_wrong_pct = wrong_pct
        # Final score = percentage of the pattern mask covered by detected stitches.
        raw_final_score = max(0.0, coverage_pct)
        self.final_score = min(100.0, raw_final_score)
        pass_threshold = float(self.level_pass_thresholds.get(self.current_level, 80.0))

        # ── Build evaluation visualization images for the results screen ─────
        # Image 1: Camera snapshot with only in-overlay detected stitches highlighted.
        try:
            _det_vis = self.last_camera_frame.copy()
            _det_mask_vis = (detected_camera_mask > 0)
            _cam_h, _cam_w = self.last_camera_frame.shape[:2]
            _cam_pat = np.zeros((_cam_h, _cam_w), dtype=np.uint8)
            _x_off = int(proj['x_offset'])
            _y_off = int(proj['y_offset'])
            _ph, _pw = pat_mask.shape[:2]

            _dx1 = max(0, _x_off)
            _dy1 = max(0, _y_off)
            _dx2 = min(_cam_w, _x_off + _pw)
            _dy2 = min(_cam_h, _y_off + _ph)
            if _dx2 > _dx1 and _dy2 > _dy1:
                _sx1 = max(0, -_x_off)
                _sy1 = max(0, -_y_off)
                _sx2 = _sx1 + (_dx2 - _dx1)
                _sy2 = _sy1 + (_dy2 - _dy1)
                _cam_pat[_dy1:_dy2, _dx1:_dx2] = (pat_mask[_sy1:_sy2, _sx1:_sx2] > 0).astype(np.uint8) * 255

            if getattr(self, 'eval_match_pattern_width', True):
                # Match preview thickness in camera-space so preview mirrors scoring.
                _det_mask_vis = self._match_detected_mask_width_to_pattern(_det_mask_vis, _cam_pat)

            # Show only detected stitches that lie inside the overlay region.
            _det_mask_vis = np.logical_and(_det_mask_vis, _cam_pat > 0)

            _hl = np.zeros_like(_det_vis)
            _hl[_det_mask_vis] = (0, 230, 200)
            cv2.addWeighted(_det_vis, 0.55, _hl, 1.0, 0, _det_vis)
            self.eval_vis_detected = _det_vis
        except Exception:
            self.eval_vis_detected = None

        # Image 2: Blueprint target mask (pattern-space).
        try:
            _mask_vis = np.full((h_cmp, w_cmp, 3), (30, 20, 15), dtype=np.uint8)
            _mask_vis[pat_cmp] = (80, 200, 200)  # teal for pattern pixels
            self.eval_vis_mask = _mask_vis
        except Exception:
            self.eval_vis_mask = None

        # Image 3: Pixel-level comparison (cyan=stitched-on-overlay, dim=missed).
        try:
            _comp_vis = np.full((h_cmp, w_cmp, 3), (30, 20, 15), dtype=np.uint8)
            _comp_vis[np.logical_and(pat_cmp, ~det_in_overlay)] = (50, 40, 20)   # dim: missed
            _comp_vis[det_in_overlay] = (80, 200, 200)                             # cyan: stitched in overlay
            self.eval_vis_comparison = _comp_vis
        except Exception:
            self.eval_vis_comparison = None
        # ─────────────────────────────────────────────────────────────────────

        self.is_evaluated = True
        self.eval_screen_stage = 0
        if self.final_score >= pass_threshold:
            self.level_completed = True
            print(
                f"✅ Level {self.current_level} PASSED! "
                f"Score: {self.final_score:.1f}% (raw: {raw_final_score:.1f}%) | "
                f"Mask Coverage: {coverage_pct:.1f}%"
            )
        else:
            self.level_completed = False
            print(
                f"📊 Score: {self.final_score:.1f}% | "
                f"Mask Coverage: {coverage_pct:.1f}% "
                f"(raw: {raw_final_score:.1f}% | need {pass_threshold:.0f}%+)"
            )
        print("🧪 EVALUATE END")
        print("=" * 72)
