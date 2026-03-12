import cv2
import numpy as np
import math
import gc
import os
import threading
import onnxruntime as ort
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
        self.current_level_tracking = 1 
        self.uniform_width = 216   
        self.uniform_height = 270 
        self.alpha_blend = 0.9
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

        # Needle detection toggle button (left side, below cloth color panel)
        self.needle_detection_enabled = True
        self.needle_toggle_button = {
            'x': self.cloth_color_panel_x,
            'y': self.cloth_color_panel_y + self.cloth_color_panel_height + 12,
            'w': self.cloth_color_panel_width,
            'h': 44,
        }

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
        self.color_overlay_interval = 3
        self._color_overlay_counter = 0
        
        # Try again button (centered in modal) - will be calculated dynamically
        self.try_again_button = {'x': 0, 'y': 0, 'w': 200, 'h': 60}
        
        # Next level button (centered in modal) - will be calculated dynamically
        self.next_level_button = {'x': 0, 'y': 0, 'w': 200, 'h': 60}
        
        # Game tracking variables
        self.current_accuracy = 0.0
        self.total_score = 0
        self.pattern_progress = 0.0  # 0-100%
        self.raw_progress = 0.0  # Raw actual progress for evaluation
        # Evaluation metrics
        self.evaluation_wrong_pct = 0.0  # % of wrong/off-pattern stitches among detections
        self.final_score = 0.0  # Adjusted score after accounting for wrong stitches
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
        self.missed_stitch_mask = None     # Accumulated mask of off-pattern detections
        self.proximity_radius = 5  # Legacy compatibility radius (kept small)
        self.stitch_draw_radius = 9        # Radius of each accepted (correct) stitch stamp — large enough to cover line edges & corners
        self.wrong_stitch_check_radius = 12  # Detection circle for off-pattern check: line half-width + ~3 px margin
        self.missed_stitch_draw_radius = 4  # Paint radius for wrong-stitch red mark
        # Visual cyan expansion around accepted stitches (increase by 15%)
        self.cyan_spread_radius = int(math.ceil(2 * 1.8))
        self.progress_spread_radius = 4  # Progress expansion to bridge tiny gaps only
        # Keep these permissive so centerline-guided tracing can advance one
        # pixel step at a time on small screens/cameras.
        self.min_stitch_move_px = 1.0  # Min movement before accepting another center stitch
        self.stitch_cooldown_frames = 1  # Min frames between accepted center stitches
        self.stitch_frame_index = 0
        self.last_stitch_frame_index = -9999
        self.last_stitch_point = None
        self.use_model_stitch_points = False  # Keep tracing tied to the red-dot center
        self.needle_model_imgsz = 160  # 160 is ~3× faster inference than 320 on RPi4

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
        self.skeleton_idx_f   = 0.0         # Current position on skeleton (float)
        self.skeleton_offset_x = 0.0        # Fixed pattern→screen offset X
        self.skeleton_offset_y = 0.0        # Fixed pattern→screen offset Y
        self.skeleton_seeded  = False
        self.skeleton_tangent_lookahead = 10  # Points used for tangent estimation
        # Auto-move speed: steps per frame along the skeleton when thread color is detected.
        # Increase to move faster, decrease to move slower.
        self.auto_move_speed = 1
        # Speed control buttons (positions set dynamically in draw_score_panel)

        self.needle_on_pattern = True  # Whether the needle is currently on a pattern pixel

        # Cache processed blueprint overlays so we do not hit disk every frame.
        self._blueprint_cache = {}

        # Per-level derivative caches (cleared in reset_progress / on level change).
        # These are pure functions of the blueprint alpha, so they are computed once
        # and reused every frame — saves ~40 ms/frame on RPi4.
        self._cached_centerline     = None  # int32 path array from _build_centerline_path
        self._cached_centerline_f32 = None  # float32 copy for fast argmin
        self._cached_pat_px         = None  # int32 x-coords of all pattern pixels
        self._cached_pat_py         = None  # int32 y-coords of all pattern pixels
        self._cached_pat_px_f32     = None  # float32 versions for distance calc
        self._cached_pat_py_f32     = None
        self._cached_corridor_mask  = None  # dilated binary corridor from _get_pattern_corridor_mask
        self._cached_realtime_pat   = None  # colored overlay from create_realtime_pattern
        self._realtime_pat_dirty    = True  # rebuild when True

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
        self.NEEDLE_CHECK_INTERVAL   = 10  # Run ONNX every 10 frames (~40% fewer calls)
        self._needle_check_counter   = 0
        # ─────────────────────────────────────────────────────────────────────

        # Segment tracking (divide pattern into 4 quarters)
        self.current_segment = 1  # 1=first 25%, 2=25-50%, 3=50-75%, 4=75-100%
        self.highest_segment_reached = 1  # Track highest segment to prevent going backwards
        self.segment_colors = {
            'completed': (255, 255, 0),    # Cyan in BGR (completed sections)
            'current': (0, 255, 0),      # Green in BGR (current section to sew)
            'upcoming': (100, 100, 100),   # Dim Gray (upcoming sections)
            'missed': (0, 0, 255),         # Red in BGR (off-pattern detections)
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
        self.confidence_threshold = 0.3  # Lowered for INT8 model (produces lower confidence scores)
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

        # Needle detection enabled flag (can be toggled at runtime)
        self.needle_detection_enabled = True

        # Load needle detection model (used for single-stitch ROI pipeline)
        # Use helper so we can lazy-load/unload at runtime when the
        # needle detection toggle is used.
        self.needle_model = None
        self.needle_input_name = None
        self.needle_output_name = None

        # ── Background inference thread ───────────────────────────────────────
        # ONNX takes ~120–180 ms on RPi4.  Running it on the render thread caps
        # the whole UI to ~5 FPS even at imgsz=160.  Instead, a dedicated daemon
        # thread runs inference asynchronously; the render loop just reads the
        # most-recent result without blocking.
        self._infer_lock      = threading.Lock()   # guards _infer_pending_frame
        self._infer_event     = threading.Event()  # signal: new frame ready
        self._infer_stop      = threading.Event()  # signal: shut thread down
        self._infer_pending_frame = None           # latest frame copy queued for inference
        self._infer_thread    = None               # the daemon thread
        # ─────────────────────────────────────────────────────────────────────

        try:
            if self.needle_detection_enabled:
                self.load_needle_model(models_dir=models_dir)
        except Exception:
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

    def load_needle_model(self, models_dir=None):
        """Load the needle ONNX model from the models folder."""
        try:
            if models_dir is None:
                models_dir = os.path.join(os.path.dirname(__file__), 'models')
            needle_model_path = os.path.join(models_dir, 'needle.onnx')
            if os.path.exists(needle_model_path):
                print(f"Loading needle detection model: {needle_model_path}")
                sess_opts = ort.SessionOptions()
                sess_opts.inter_op_num_threads = 2
                sess_opts.intra_op_num_threads = 2
                self.needle_model = ort.InferenceSession(
                    needle_model_path,
                    sess_options=sess_opts,
                    providers=['CPUExecutionProvider'])
                self.needle_input_name  = self.needle_model.get_inputs()[0].name
                self.needle_output_name = self.needle_model.get_outputs()[0].name
                print(f"✓ Needle detection model loaded (ONNX Runtime direct)")
                print(f"    Input:  {self.needle_input_name}  {self.needle_model.get_inputs()[0].shape}")
                print(f"    Output: {self.needle_output_name} {self.needle_model.get_outputs()[0].shape}")
                # Lightweight warm-up run
                _warmup = np.zeros((1, 3, self.needle_model_imgsz, self.needle_model_imgsz), dtype=np.float32)
                try:
                    self.needle_model.run(None, {self.needle_input_name: _warmup})
                    print("  ✓ Needle model warm-up successful!")
                except Exception:
                    pass
                # Start the background inference thread now that the model is ready.
                self._start_inference_thread()
                print("  ✓ Background needle inference thread started")
            else:
                print(f"⚠ Needle model not found: {needle_model_path}")
                self.needle_model = None
        except Exception as e:
            print(f"⚠ ERROR loading needle model: {e}")
            self.needle_model = None

    def unload_needle_model(self):
        """Unload the needle model to free resources."""
        # Stop the inference thread first so it doesn't hold a reference.
        self._stop_inference_thread()
        try:
            if getattr(self, 'needle_model', None) is not None:
                print("Unloading needle detection model to save resources...")
                try:
                    del self.needle_model
                except Exception:
                    pass
                self.needle_model = None
                gc.collect()
        except Exception:
            pass

    # ── Background inference thread ───────────────────────────────────────────

    def _start_inference_thread(self):
        """Start the background needle-check inference thread."""
        if self._infer_thread is not None and self._infer_thread.is_alive():
            return
        self._infer_stop.clear()
        self._infer_event.clear()
        self._infer_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="needle-infer")
        self._infer_thread.start()

    def _stop_inference_thread(self):
        """Signal and join the background inference thread."""
        if self._infer_thread is None:
            return
        self._infer_stop.set()
        self._infer_event.set()   # unblock the waiting thread
        self._infer_thread.join(timeout=2.0)
        self._infer_thread = None

    def _inference_loop(self):
        """Worker: wait for a frame, run _run_needle_check, store result.

        This runs entirely off the render thread, keeping the UI loop
        unblocked while ONNX inference executes (~120–180 ms on RPi4).
        """
        while not self._infer_stop.is_set():
            # Block until the render loop queues a new frame (or stop signal).
            signalled = self._infer_event.wait(timeout=0.5)
            if not signalled:
                continue
            self._infer_event.clear()
            if self._infer_stop.is_set():
                break

            # Grab the latest pending frame (render loop may have updated it).
            with self._infer_lock:
                frame = self._infer_pending_frame
                self._infer_pending_frame = None

            if frame is None:
                continue

            # Run the (slow) ONNX needle check.
            result = self._run_needle_check(frame)

            # Write result back — needle_confirmed is read by the render loop.
            self.needle_confirmed = result

    def _submit_needle_check(self, cam_frame):
        """Queue a frame for async needle inference and return immediately.

        Should be called from the render loop instead of _run_needle_check.
        The result is available in self.needle_confirmed on the next frame.
        """
        if self.needle_model is None:
            return
        # Make a lightweight copy so the render loop can keep mutating cam_frame.
        frame_copy = cam_frame.copy()
        with self._infer_lock:
            self._infer_pending_frame = frame_copy
        self._infer_event.set()

    # ── End background inference thread ──────────────────────────────────────

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

        # Cut the binary mask in half (keep top half) — skipped for levels 4 & 5
        if level not in (3, 5):
            mask = mask[:mask.shape[0] // 2, :]

        # Base width is 50%; increase it by 25% => 62.5% of original width.
        # Make level-specific width adjustments. Level 1 should be a thin
        # vertical column so the red-point can only travel up/down.
        if level == 1:
            # Reduce to ~28% of original width (adjustable)
            new_w = int(round(mask.shape[1] * 0.28))
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

        # Dilate pattern lines for levels 3-5 *just enough* to round sharp corners
        # so the skeleton tracer navigates them cleanly, without visually thickening
        # the lines above the level 1/2 baseline.
        if level in (3, 4, 5):
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, dil_kernel, iterations=1)

        # Convert mask to white overlay (pattern lines)
        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        alpha = mask / 255.0

        self._blueprint_cache[cache_key] = (overlay, alpha)
        return self._blueprint_cache[cache_key]
    
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
        self.skeleton_idx_f   = 0.0
        self.skeleton_seeded  = False
        self.needle_pos_x = float(self.NEEDLE_ROI_X)
        self.needle_pos_y = float(self.NEEDLE_ROI_Y)
        self._color_overlay_counter = 0
        # Clear per-level derivative caches
        self._cached_centerline     = None
        self._cached_centerline_f32 = None
        self._cached_pat_px         = None
        self._cached_pat_py         = None
        self._cached_pat_px_f32     = None
        self._cached_pat_py_f32     = None
        self._cached_corridor_mask  = None
        self._cached_realtime_pat   = None
        self._realtime_pat_dirty    = True
        self.centerline_progress_idx = 0
        self.centerline_progress_initialized = False
        self.needle_on_pattern = True
        self.follow_off_count = 0
        self.follow_on_count = 0
        self.sewing_started = False  # Require Start button press again on reset
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

        # Missed stitch mask paints off-pattern detections red.
        if self.missed_stitch_mask is not None:
            missed_mask_resized = cv2.resize(self.missed_stitch_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            missed_pixels = missed_mask_resized > 0
            colored_overlay[missed_pixels] = self.segment_colors['missed']

        return colored_overlay
    

    # ──────────────────────────────────────────────────────────────────────────
    # Needle ROI pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def _update_needle_optical_flow(self, gray_frame):
        """No-op: needle position is fixed; cloth motion drives overlay via optical flow."""
        pass

    def _update_cloth_optical_flow(self, cam_frame):
        """Removed: cloth detection is no longer used. Kept as no-op for compatibility."""
        pass

    def _infer_centerline_step_direction(self, centerline_path, current_idx):
        """Always advance forward along the centerline (cloth motion removed)."""
        return 1

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

        # _update_needle_optical_flow is a no-op, so skip the expensive gray
        # conversion when we are only called for its (unused) side-effects.
        if not run_detection or self.needle_model is None:
            return

        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        self._update_needle_optical_flow(gray)

        half = self.NEEDLE_ROI_SIZE // 2
        rx1 = max(0, int(self.needle_pos_x) - half)
        ry1 = max(0, int(self.needle_pos_y) - half)
        rx2 = min(self.camera_width,  rx1 + self.NEEDLE_ROI_SIZE)
        ry2 = min(self.camera_height, ry1 + self.NEEDLE_ROI_SIZE)

        roi = cam_frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return

        try:
            # Run needle model via direct ONNX Runtime (no Ultralytics overhead).
            imgsz = self.needle_model_imgsz
            _inp = cv2.resize(roi, (imgsz, imgsz))
            _inp = cv2.cvtColor(_inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            _inp = np.transpose(_inp, (2, 0, 1))[np.newaxis]  # [1,3,H,W]
            _preds = self.needle_model.run(None, {self.needle_input_name: _inp})[0]
            # YOLOv8 ONNX output: [1, 5, anchors] → [anchors, 5]
            _preds = _preds[0].T
            _scores = _preds[:, 4]
            _mask   = _scores >= self.confidence_threshold
            num_boxes = int(np.sum(_mask))

            # ── Canny fallback when ONNX finds nothing ─────────────────────────
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
                filtered = _preds[_mask]
                best     = int(np.argmax(filtered[:, 4]))
                roi_h_c, roi_w_c = roi.shape[:2]
                # ORT outputs cx,cy,w,h normalised to imgsz — scale back to ROI pixels
                cx_roi = float(filtered[best, 0]) / imgsz * roi_w_c
                cy_roi = float(filtered[best, 1]) / imgsz * roi_h_c
                cx_cam = rx1 + cx_roi
                cy_cam = ry1 + cy_roi
                conf_val = float(filtered[best, 4])

                self._register_stitch(cx_cam, cy_cam, cam_frame,
                                      pattern_alpha, x_offset, y_offset,
                                      actual_w, actual_h,
                                      source=f"onnx conf={conf_val:.2f}", hsv_frame=hsv_frame,
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

        if not is_valid_stitch:
            # Color detected but outside the pattern — paint missed mask
            if self.missed_stitch_mask is None:
                self.missed_stitch_mask = np.zeros((pat_h, pat_w), dtype=np.uint8)
            cv2.circle(self.missed_stitch_mask, (px, py), self.missed_stitch_draw_radius, 255, -1)
            self._realtime_pat_dirty = True
            # Trigger a visible warning banner for wrong stitch
            try:
                self.out_of_segment_warning = True
                self.warning_message = "FOLLOW THE PATTERN"
                self.warning_flash_phase = 0
                # Show for ~40 frames (~1 second at 30-40 FPS)
                self.warning_until_frame = int(self.stitch_frame_index) + 40
            except Exception:
                pass
            return

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
            self._realtime_pat_dirty = True
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
        if self._cached_corridor_mask is not None:
            return self._cached_corridor_mask
        pat_crop = pattern_alpha[:actual_h, :actual_w]
        pat_binary = (pat_crop > self.pattern_alpha_threshold).astype(np.uint8)
        k = max(3, self.follow_corridor_radius * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._cached_corridor_mask = cv2.dilate(pat_binary, kernel, iterations=1)
        return self._cached_corridor_mask

    def _build_centerline_path(self, pattern_alpha, actual_w, actual_h):
        """Build an ordered centerline path from the binary blueprint mask.

        This is a lightweight skeleton-style center extraction: for each row that
        contains pattern pixels, choose the x-position nearest the previous row's
        center so the path remains continuous.
        
        For levels 4 & 5, starts from the absolute leftmost x across the entire pattern.
        """
        # Reuse cached result — the blueprint never changes mid-level.
        if self._cached_centerline is not None:
            return self._cached_centerline
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

        result = np.array(path_points, dtype=np.int32)
        self._cached_centerline     = result
        self._cached_centerline_f32 = result.astype(np.float32)
        return result

    def _build_skeleton_path(self, pattern_alpha, actual_w, actual_h):
        """Build an ordered 1-pixel skeleton path by thinning the binary mask.

        Returns an np.array of shape (N, 2) where each row is (x, y) in pattern
        space.  The path is traced from one endpoint to the other so the needle
        can travel the full zigzag without leaving the skeleton.
        Cached per level — computed once on first call.
        """
        if self._cached_skeleton_path is not None:
            return self._cached_skeleton_path

        pat_crop = pattern_alpha[:actual_h, :actual_w]
        pat_binary = (pat_crop > self.pattern_alpha_threshold).astype(np.uint8)

        # ── Morphological thinning (skeleton) ────────────────────────────────
        skel = None
        try:
            from skimage.morphology import skeletonize as _sk
            skel = _sk(pat_binary.astype(bool)).astype(np.uint8)
        except Exception:
            pass
        if skel is None:
            try:
                skel = cv2.ximgproc.thinning(pat_binary * 255) // 255
            except Exception:
                pass
        if skel is None:
            # Pure-NumPy Zhang-Suen–style iterative erosion fallback
            img = pat_binary.copy()
            prev = None
            for _ in range(200):
                eroded = cv2.erode(img, np.ones((3, 3), np.uint8))
                opened = cv2.dilate(eroded, np.ones((3, 3), np.uint8))
                skel_iter = cv2.subtract(img, opened)
                if prev is not None and np.array_equal(skel_iter, prev):
                    break
                prev = skel_iter.copy()
                img = eroded
            skel = skel_iter if prev is not None else pat_binary

        ys, xs = np.where(skel > 0)
        if ys.size == 0:
            return None

        # ── Build adjacency set for O(1) neighbor lookup ──────────────────────
        skel_set = set(zip(ys.tolist(), xs.tolist()))

        def get_neighbors_8(y, x):
            return [
                (y + dy, x + dx)
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if not (dy == 0 and dx == 0) and (y + dy, x + dx) in skel_set
            ]

        # ── Find an endpoint (degree-1 pixel) as the start of the path ────────
        endpoints = [(y, x) for (y, x) in skel_set if len(get_neighbors_8(y, x)) == 1]
        if not endpoints:
            # Closed loop — pick topmost-leftmost pixel
            start = min(skel_set, key=lambda p: (p[0], p[1]))
        else:
            start = min(endpoints, key=lambda p: (p[0], p[1]))

        # ── Walk path: at junctions pick the branch with the most reachable pixels ──
        # This avoids short spurious spurs that thinning creates at inner corners.
        # At a clean 1-px-wide junction the correct turn is always longer than the spur.
        def _count_reachable(seed, base_visited):
            q = [seed]
            seen = set(base_visited)
            seen.add(seed)
            n = 0
            while q:
                p = q.pop()
                n += 1
                for nb2 in get_neighbors_8(*p):
                    if nb2 not in seen:
                        seen.add(nb2)
                        q.append(nb2)
            return n

        path_yx = [start]
        visited = {start}
        current = start
        while True:
            nbs = [nb for nb in get_neighbors_8(*current) if nb not in visited]
            if not nbs:
                break
            if len(nbs) == 1:
                next_pt = nbs[0]
            else:
                # Junction: take the arm that leads to the most unvisited skeleton pixels.
                next_pt = max(nbs, key=lambda nb: _count_reachable(nb, visited))
            visited.add(next_pt)
            path_yx.append(next_pt)
            current = next_pt

        # ── Convert (y, x) list → (x, y) int32 array ─────────────────────────
        result = np.array([[x, y] for y, x in path_yx], dtype=np.int32)
        self._cached_skeleton_path = result
        self._cached_skeleton_f32  = result.astype(np.float32)
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
        """Recalculate raw_progress from both completed (cyan) and missed (red) stitch masks.
        Both on-pattern and off-pattern stitches count towards progress so the user
        advances regardless of accuracy."""
        if self.completed_stitch_mask is None and self.missed_stitch_mask is None:
            return

        pattern_pixels = (pattern_alpha > 0.1)
        total = int(np.sum(pattern_pixels))
        if total == 0:
            return

        # Combine cyan + red masks so both count towards coverage.
        combined = np.zeros(pattern_alpha.shape[:2], dtype=np.uint8)
        if self.completed_stitch_mask is not None:
            combined = cv2.bitwise_or(combined, self.completed_stitch_mask)
        if self.missed_stitch_mask is not None:
            missed_resized = self.missed_stitch_mask
            if missed_resized.shape != combined.shape:
                missed_resized = cv2.resize(missed_resized, (combined.shape[1], combined.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
            combined = cv2.bitwise_or(combined, missed_resized)

        # Dilate stitch dots to bridge small gaps (uses proximity_radius)
        k = max(3, self.progress_spread_radius * 2 + 1)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(combined, kernel, iterations=1)

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

        # Draw needle detection toggle button
        try:
            self.draw_needle_toggle_button(frame)
        except Exception:
            pass

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

    def _run_needle_check(self, cam_frame):
        """Crop the column ROI and run needle.onnx to check needle centring.
        Returns True when the highest-confidence detection centre-X is within
        NEEDLE_CENTER_TOLERANCE pixels of the column's horizontal midpoint."""
        # Skip heavy model check when needle detection is disabled to save CPU/GPU.
        if not getattr(self, 'needle_detection_enabled', True):
            return False
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
            imgsz = self.needle_model_imgsz
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
            follow_check_ready = False
            follow_on_corridor = True
            follow_order_valid = True
            
            # Load pattern mask
            pattern_overlay, pattern_alpha = self.load_blueprint(self.current_level)

            # Pattern overlay — pattern stays fixed; needle (red dot) slides along skeleton.
            if pattern_overlay is not None and pattern_alpha is not None:
                overlay_h, overlay_w = pattern_overlay.shape[:2]

                # Keep centerline for downstream validation pipeline (unchanged).
                centerline_path = self._build_centerline_path(pattern_alpha, overlay_w, overlay_h)
                # Build skeleton path for needle movement constraint.
                skeleton_path = self._build_skeleton_path(pattern_alpha, overlay_w, overlay_h)
                expected_path_idx = None

                if skeleton_path is not None and len(skeleton_path) > 0:
                    skel_max_idx = len(skeleton_path) - 1

                    # ── Seed once: fix pattern position on screen, start at skeleton[0] ──
                    if not self.skeleton_seeded:
                        if self.raw_progress <= 0.0:
                            seed_idx = 0
                        else:
                            seed_idx = int(np.clip(
                                round(self.raw_progress / 100.0 * skel_max_idx),
                                0, skel_max_idx))
                        self.skeleton_idx_f   = float(seed_idx)
                        # Position pattern so skeleton[seed_idx] sits at the default needle pos.
                        self.skeleton_offset_x = float(self.NEEDLE_ROI_X) - float(skeleton_path[seed_idx][0])
                        self.skeleton_offset_y = float(self.NEEDLE_ROI_Y) - float(skeleton_path[seed_idx][1])
                        self.skeleton_seeded = True
                        self.centerline_progress_initialized = True

                    # ── Auto-advance: color detection uses current dynamic pattern offset ──
                    # Compute where the pattern is sitting RIGHT NOW (before any advance)
                    # so the detection region matches the ink visible on camera.
                    moved_with_color = False
                    if self.sewing_started:
                        _cur = int(np.clip(round(self.skeleton_idx_f), 0, skel_max_idx))
                        _ex  = int(skeleton_path[_cur][0])
                        _ey  = int(skeleton_path[_cur][1])
                        x_off = int(round(self.NEEDLE_ROI_X)) - _ex
                        y_off = int(round(self.NEEDLE_ROI_Y)) - _ey
                        _pat_h, _pat_w = pattern_alpha.shape[:2]
                        _dx1 = max(0, x_off)
                        _dy1 = max(0, y_off)
                        _dx2 = min(detection_frame.shape[1], x_off + _pat_w)
                        _dy2 = min(detection_frame.shape[0], y_off + _pat_h)
                        color_ok = False
                        if _dx2 > _dx1 and _dy2 > _dy1:
                            _sx1 = max(0, -x_off)
                            _sy1 = max(0, -y_off)
                            _sx2 = _sx1 + (_dx2 - _dx1)
                            _sy2 = _sy1 + (_dy2 - _dy1)
                            _cam_region  = detection_frame[_dy1:_dy2, _dx1:_dx2]
                            _hsv_region  = hsv_detection_frame[_dy1:_dy2, _dx1:_dx2]
                            _alpha_crop  = pattern_alpha[_sy1:_sy2, _sx1:_sx2]
                            _ink_mask    = _alpha_crop > 0.1
                            _ink_count   = int(np.count_nonzero(_ink_mask))
                            if _ink_count > 0:
                                _color_mask   = self._get_selected_color_mask(_cam_region, hsv_patch=_hsv_region)
                                _color_in_ink = int(np.count_nonzero(_color_mask[_ink_mask]))
                                _ratio = float(_color_in_ink) / float(_ink_count)
                                color_cfg = self.color_profiles[self.selected_detection_color]
                                self.last_color_match_ratio = _ratio
                                self.last_color_match = _ratio >= color_cfg['min_ratio']
                                color_ok = self.last_color_match
                        if color_ok:
                            self.skeleton_idx_f = float(np.clip(
                                self.skeleton_idx_f + self.auto_move_speed,
                                0.0, float(skel_max_idx)))
                            moved_with_color = True

                    # ── Pattern scrolls under a fixed needle position ─────────────────
                    cur_idx = int(np.clip(round(self.skeleton_idx_f), 0, skel_max_idx))
                    exp_x   = int(skeleton_path[cur_idx][0])
                    exp_y   = int(skeleton_path[cur_idx][1])

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

                    # Credit/penalize only when motion and selected thread color are both valid.
                    if moved_with_color:
                        pat_h, pat_w = pattern_alpha.shape[:2]
                        # Option 2: wrong stitch fires if ANY pixel of the stitch circle
                        # falls outside the pattern mask — not just the center.
                        on_mask = self._circle_fully_on_mask(
                            pattern_alpha, exp_x, exp_y, self.wrong_stitch_check_radius
                        )
                        if on_mask:
                            if self.completed_stitch_mask is None:
                                self.completed_stitch_mask = np.zeros((pat_h, pat_w), dtype=np.uint8)
                            cv2.circle(
                                self.completed_stitch_mask,
                                (exp_x, exp_y),
                                self.stitch_draw_radius,
                                255,
                                -1,
                            )
                            self._realtime_pat_dirty = True
                            self._update_progress_from_mask(pattern_alpha)
                        else:
                            if self.missed_stitch_mask is None:
                                self.missed_stitch_mask = np.zeros((pat_h, pat_w), dtype=np.uint8)
                            cv2.circle(
                                self.missed_stitch_mask,
                                (raw_needle_in_pat_x, raw_needle_in_pat_y),
                                self.missed_stitch_draw_radius,
                                255,
                                -1,
                            )
                            self._realtime_pat_dirty = True
                            self._update_progress_from_mask(pattern_alpha)
                            try:
                                self.out_of_segment_warning = True
                                self.warning_message = "FOLLOW THE PATTERN"
                                self.warning_flash_phase = 0
                                self.warning_until_frame = int(self.stitch_frame_index) + 40
                            except Exception:
                                pass
                else:
                    # Skeleton unavailable — fall back to fixed centre.
                    self.skeleton_seeded = False
                    self.centerline_progress_initialized = False
                    exp_x    = overlay_w // 2
                    exp_y    = overlay_h // 2
                    x_offset = int(round(self.NEEDLE_ROI_X)) - exp_x
                    y_offset = int(round(self.NEEDLE_ROI_Y)) - exp_y

                # Fallback anchoring when skeleton is unavailable.
                if skeleton_path is None or len(skeleton_path) == 0:
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

                if visible_w > 0 and visible_h > 0:
                    corridor_mask = self._get_pattern_corridor_mask(pattern_alpha, actual_w, actual_h)
                    # Rebuild the colored overlay only when stitch masks have changed.
                    if self._realtime_pat_dirty or self._cached_realtime_pat is None:
                        self._cached_realtime_pat = self.create_realtime_pattern(
                            pattern_overlay, pattern_alpha, trace_y=trace_y)
                        self._realtime_pat_dirty = False
                    realtime_pattern = self._cached_realtime_pat

                    roi = cam_frame[dst_y1:dst_y2, dst_x1:dst_x2]
                    pattern_src = realtime_pattern if realtime_pattern is not None else pattern_overlay
                    pattern_crop = pattern_src[src_y1:src_y2, src_x1:src_x2]
                    alpha_crop = pattern_alpha[src_y1:src_y2, src_x1:src_x2]
                    # Vectorized 3-channel alpha blend — avoids 3× Python-loop overhead.
                    a3 = alpha_crop[:, :, np.newaxis] * 0.35  # [H, W, 1] broadcast
                    roi_blended = a3 * pattern_crop + (1.0 - a3) * roi
                    cam_frame[dst_y1:dst_y2, dst_x1:dst_x2] = roi_blended.astype(np.uint8)
                    
                    self.run_needle_pipeline(
                        detection_frame, pattern_alpha,
                        x_offset, y_offset, actual_w, actual_h,
                        run_detection=False,
                        hsv_frame=hsv_detection_frame,
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
            # Only show the column ROI and centering checks when needle
            # detection is enabled.
            if self.needle_detection_enabled:
                self._needle_check_counter += 1
                if self._needle_check_counter >= self.NEEDLE_CHECK_INTERVAL:
                    self._needle_check_counter = 0
                    # Submit to background thread — returns immediately.
                    # needle_confirmed is updated by _inference_loop asynchronously.
                    self._submit_needle_check(cam_frame)
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
                    warn_txt = "SEWING NEEDLE NOT CENTRED"
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

            if self.needle_detection_enabled:
                # Draw wrong-stitch detection border (dashed circle around needle)
                cx = int(self.needle_pos_x)
                cy = int(self.needle_pos_y)
                border_r = int(self.wrong_stitch_check_radius)
                # Dashed circle: draw 12 short arcs evenly spaced
                num_dashes = 12
                for i in range(num_dashes):
                    angle_start = int(i * 360 / num_dashes)
                    angle_end   = int(angle_start + 360 / num_dashes * 0.55)
                    cv2.ellipse(cam_frame, (cx, cy), (border_r, border_r),
                                0, angle_start, angle_end, (0, 0, 220), 2, cv2.LINE_AA)

                # Draw a filled square marker for the needle (all states)
                half = 3
                cv2.rectangle(cam_frame, (cx - half, cy - half), (cx + half, cy + half), marker_color, -1)

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
            else:
                # When needle detection is off, hide detection UI
                pass
            
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
                ("Start anywhere on the pattern", self.COLORS['text_secondary'], 0.7, 2),
                ("and sew freely.", self.COLORS['text_secondary'], 0.7, 2),
            ]
        elif self.guide_step == 2:
            instructions = [
                ("COLOR GUIDE", self.COLORS['text_primary'], 0.85, 2),
                "",
                ("CYAN = Completed stitches", self.COLORS['text_secondary'], 0.7, 2),
                "",
                ("GREEN = Pattern to sew", self.COLORS['text_secondary'], 0.7, 2),
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

    def draw_needle_toggle_button(self, frame):
        """Draw a button that toggles needle detection on/off"""
        nb = self.needle_toggle_button
        pulse = 0.35 + 0.25 * abs(math.sin(self.glow_phase * 1.0))
        self.draw_glow_rect(frame, nb['x'], nb['y'], nb['w'], nb['h'], self.COLORS['medium_blue'], pulse)

        overlay = frame.copy()
        cv2.rectangle(overlay, (nb['x'] + 2, nb['y'] + 2), (nb['x'] + nb['w'] - 2, nb['y'] + nb['h'] - 2), self.COLORS['button_normal'], -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        state_text = "ON" if self.needle_detection_enabled else "OFF"
        text = f"NEEDLE: {state_text}"
        txt_col = (0, 200, 0) if self.needle_detection_enabled else (0, 0, 200)
        font_scale = text_scale(0.64, self.width, self.height, floor=0.56, ceiling=0.74)
        thickness = text_thickness(2, self.width, self.height, min_thickness=1, max_thickness=2)
        font_scale = fit_text_scale(text, FONT_MAIN, nb['w'] - 14, font_scale, thickness, min_scale=0.6)
        (text_w, text_h), _ = get_text_size(text, FONT_MAIN, font_scale, thickness)
        text_x = nb['x'] + (nb['w'] - text_w) // 2
        text_y = nb['y'] + (nb['h'] + text_h) // 2
        self._put_text(frame, text, text_x, text_y, font_scale, txt_col, thickness)
    
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
        wrong_text = f"Wrong stitches: {self.evaluation_wrong_pct:.1f}%"
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

        # Needle detection toggle button
        nb = getattr(self, 'needle_toggle_button', None)
        if nb is not None and nb['x'] <= x <= nb['x'] + nb['w'] and nb['y'] <= y <= nb['y'] + nb['h']:
            self.play_button_click_sound()
            self.needle_detection_enabled = not bool(self.needle_detection_enabled)
            # Load or unload the ONNX model depending on new state to save resources
            if self.needle_detection_enabled:
                try:
                    self.load_needle_model()
                except Exception:
                    pass
            else:
                try:
                    # Unload model and clear detection state + bbox
                    self.unload_needle_model()
                    self.smooth_cloth_bbox = None
                    self.needle_confirmed = False
                except Exception:
                    pass
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
        
        # Check try again button (only if evaluated and failed)
        if self.is_evaluated and not self.level_completed:
            tb = self.try_again_button
            if tb['x'] <= x <= tb['x'] + tb['w'] and tb['y'] <= y <= tb['y'] + tb['h']:
                self.play_button_click_sound()
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
        if self.is_evaluated and self.level_completed:
            nb = self.next_level_button
            if nb['x'] <= x <= nb['x'] + nb['w'] and nb['y'] <= y <= nb['y'] + nb['h']:
                self.play_button_click_sound()
                return 'next_level'
        
        return None
    
    def evaluate_pattern(self):
        """Evaluate the current pattern and determine if level is completed"""
        self.is_evaluated = True
        # Prefer computing coverage directly against the blueprint mask so
        # the final score equals the actual cyan coverage of pattern pixels.
        try:
            _, pattern_alpha = self.load_blueprint(self.current_level)
        except Exception:
            pattern_alpha = None

        cyan_covered = 0
        missed_on_pattern = 0
        total_pattern = 0

        if pattern_alpha is not None:
            pat_mask = (pattern_alpha > self.pattern_alpha_threshold)
            total_pattern = int(np.sum(pat_mask))
            if total_pattern > 0:
                if self.completed_stitch_mask is not None:
                    # completed_stitch_mask is stored in pattern coordinates
                    comp = (self.completed_stitch_mask > 0)
                    # crop/resize defensively if shapes mismatch
                    if comp.shape != pat_mask.shape:
                        comp = cv2.resize(comp.astype(np.uint8), (pat_mask.shape[1], pat_mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    cyan_covered = int(np.sum(np.logical_and(comp, pat_mask)))
                if self.missed_stitch_mask is not None:
                    miss = (self.missed_stitch_mask > 0)
                    if miss.shape != pat_mask.shape:
                        miss = cv2.resize(miss.astype(np.uint8), (pat_mask.shape[1], pat_mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    missed_on_pattern = int(np.sum(np.logical_and(miss, pat_mask)))

        # Final score = percentage of pattern pixels covered by cyan
        if total_pattern == 0:
            self.final_score = 0.0
        else:
            self.final_score = float(cyan_covered) / float(total_pattern) * 100.0

        # Wrong-stitch percentage: show remaining pattern not covered by cyan
        # (i.e. 100% - coverage). This matches the in-game display expectation.
        if total_pattern == 0:
            self.evaluation_wrong_pct = 0.0
        else:
            self.evaluation_wrong_pct = max(0.0, 100.0 - self.final_score)

        # Determine pass/fail based on final_score
        if self.final_score >= 80.0:
            self.level_completed = True
            print(f"✅ Level {self.current_level} completed! Score: {self.final_score:.1f}%, Wrong on pattern: {self.evaluation_wrong_pct:.1f}%")
        else:
            self.level_completed = False
            print(f"📊 Evaluation: Score {self.final_score:.1f}%, Wrong on pattern {self.evaluation_wrong_pct:.1f}%. Need 80%+ to complete.")
