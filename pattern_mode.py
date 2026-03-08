import cv2
import numpy as np
import math
import os
from ultralytics import YOLO
from music_manager import get_music_manager


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
        self.score_panel_y = 200
        self.score_panel_width = 200
        self.score_panel_height = 250
        
        # Back button (top left)
        self.back_button = {'x': 20, 'y': 20, 'w': 120, 'h': 50}
        
        # Evaluate button (below stats panel)
        self.evaluate_button = {'x': 800, 'y': 470, 'w': 200, 'h': 50}

        # Color detection selector (right side, above stats)
        self.color_panel_x = 800
        self.color_panel_y = 120
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
        self.conf_panel_y = 525
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
        self.proximity_radius = 30  # Pixels around completed stitches that turn cyan

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
        
        # Performance optimization: Frame skipping for detection
        # Process detection every N frames to reduce CPU load
        self.frame_skip = 2  # Process every 2nd frame (doubles FPS)
        self.frame_counter = 0
        self.last_cloth_bbox = None  # Cache last cloth detection result
        self.smooth_cloth_bbox = None  # Smoothed bbox for stable overlay
        self.bbox_smooth_alpha = 0.1  # Lower smoothing factor for slower, smoother updates
        self.bbox_move_threshold = 12  # Increased movement threshold to ignore smaller jitters
        
        # Load cloth detection model
        try:
            cloth_model_path = os.path.join(models_dir, 'cloth.onnx')
            
            if os.path.exists(cloth_model_path):
                print(f"Loading cloth detection model: {cloth_model_path}")
                self.cloth_model = YOLO(cloth_model_path, task='detect')
                print(f"✓ Cloth detection model loaded successfully!")
                
                # Print available classes
                if hasattr(self.cloth_model, 'names'):
                    print(f"  Detected classes:")
                    for idx, name in self.cloth_model.names.items():
                        print(f"    Class {idx}: {name}")
                print(f"  Confidence threshold: {self.confidence_threshold}")
                
                # Test the model with dummy data
                print("  Testing cloth model inference...")
                test_img = np.zeros((560, 560, 3), dtype=np.uint8)
                self.cloth_model(test_img, conf=self.confidence_threshold, verbose=False)
                print("  ✓ Cloth model test successful!")
            else:
                print(f"⚠ Cloth model not found: {cloth_model_path}")
                self.cloth_model = None
        except Exception as e:
            print(f"⚠ ERROR loading cloth model: {e}")
            import traceback
            traceback.print_exc()
            self.cloth_model = None

        # Load needle detection model (used for single-stitch ROI pipeline)
        try:
            needle_model_path = os.path.join(models_dir, 'stitch.onnx')
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

        # Move the pattern to the top of the cloth while keeping it centered horizontally
        top_offset = 0
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
        print(f"🔄 Progress reset for Level {self.current_level}")
    
    def create_realtime_pattern(self, overlay, alpha):
        """Create real-time colored pattern overlay that progressively changes to cyan
        as stitches are detected nearby
        
        Args:
            overlay: Original pattern overlay (BGR)
            alpha: Pattern alpha mask
            
        Returns:
            Colored pattern overlay with progressive real-time coloring
        """
        if overlay is None or alpha is None:
            return None
        
        # Create colored overlay - start with yellow (uncompleted)
        colored_overlay = overlay.copy()
        height, width = overlay.shape[:2]
        
        # Default color: yellow (current/to-be-sewn)
        pattern_pixels = alpha > 0.1
        colored_overlay[pattern_pixels] = self.segment_colors['current']
        
        # If we have completed stitches, color nearby pattern pixels cyan
        if self.completed_stitch_mask is not None:
            # Resize completed mask to match pattern size
            completed_mask_resized = cv2.resize(self.completed_stitch_mask, (width, height))
            
            # Dilate the completed stitch mask to affect nearby pattern areas
            kernel_size = self.proximity_radius
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            proximity_mask = cv2.dilate(completed_mask_resized, kernel, iterations=1)
            
            # Only color pattern pixels that are near completed stitches
            completed_pattern_pixels = np.logical_and(pattern_pixels, proximity_mask > 0)
            colored_overlay[completed_pattern_pixels] = self.segment_colors['completed']
        
        return colored_overlay
    

    # ──────────────────────────────────────────────────────────────────────────
    # Needle ROI pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def _update_needle_optical_flow(self, gray_frame):
        """No-op: ROI is fixed at NEEDLE_ROI_X/Y — optical flow intentionally disabled."""
        pass

    def run_needle_pipeline(self, cam_frame, pattern_alpha,
                            x_offset, y_offset, actual_w, actual_h,
                            run_detection=True):
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
            # Run needle model — use imgsz=128 for better quality on small crop;
            # Reuse current confidence threshold so +/- controls affect both models.
            results = self.needle_model(roi, conf=self.confidence_threshold, imgsz=128, verbose=False)

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
                                              actual_w, actual_h, source="canny")
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
                                      source=f"yolo conf={float(confs[best]):.2f}")
        except Exception as e:
            import traceback
            print(f"Needle detection error: {e}")
            traceback.print_exc()

    def _register_stitch(self, cx_cam, cy_cam, cam_frame,
                         pattern_alpha, x_offset, y_offset, actual_w, actual_h,
                         source=""):
        """Check if a candidate stitch position overlaps the pattern and record it."""
        if not self._matches_selected_color(cam_frame, cx_cam, cy_cam):
            return

        pat_h, pat_w = pattern_alpha.shape[:2]

        # Map from camera-frame coords to pattern-overlay coords
        px = int(np.clip(cx_cam - x_offset, 0, actual_w - 1))
        py = int(np.clip(cy_cam - y_offset, 0, actual_h - 1))

        # Proximity check against pattern lines
        pat_crop   = pattern_alpha[:actual_h, :actual_w]
        pat_binary = (pat_crop > 0.1).astype(np.uint8)
        kern_r     = max(3, self.proximity_radius * 2 + 1)
        kern       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_r, kern_r))
        near_mask  = cv2.dilate(pat_binary, kern, iterations=1)
        on_or_near = bool(near_mask[py, px] > 0)

        print(f"🪡 [{source}] cam=({cx_cam:.0f},{cy_cam:.0f})  "
              f"pat=({px},{py})  near={on_or_near}")

        if on_or_near:
            if self.completed_stitch_mask is None:
                self.completed_stitch_mask = np.zeros((pat_h, pat_w), dtype=np.uint8)
            cv2.circle(self.completed_stitch_mask, (px, py), 8, 255, -1)
            self._update_progress_from_mask(pattern_alpha)
            self.stitches_detected += 1

    def _matches_selected_color(self, cam_frame, cx_cam, cy_cam):
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

        combined_mask = self._get_selected_color_mask(patch)

        match_ratio = float(np.count_nonzero(combined_mask)) / float(combined_mask.size)
        self.last_color_match_ratio = match_ratio
        self.last_color_match = match_ratio >= color_cfg['min_ratio']
        self.last_color_mask = combined_mask
        self.last_color_mask_bounds = (x1, y1, x2, y2)
        return self.last_color_match

    def _get_selected_color_mask(self, bgr_patch):
        """Build binary mask for the currently selected color in a BGR patch."""
        color_cfg = self.color_profiles[self.selected_detection_color]
        hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv_patch.shape[:2], dtype=np.uint8)

        for low, high in color_cfg['hsv_ranges']:
            low_np = np.array(low, dtype=np.uint8)
            high_np = np.array(high, dtype=np.uint8)
            combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv_patch, low_np, high_np))

        return combined_mask

    def _update_progress_from_mask(self, pattern_alpha):
        """Recalculate raw_progress and pattern_progress from completed_stitch_mask."""
        if self.completed_stitch_mask is None:
            return

        pattern_pixels = (pattern_alpha > 0.1)
        total = int(np.sum(pattern_pixels))
        if total == 0:
            return

        # Dilate stitch dots to bridge small gaps (uses proximity_radius)
        k = max(3, self.proximity_radius)
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
        
        font_scale = 1.2
        thickness = 3
        (level_w, level_h), _ = cv2.getTextSize(level_text, cv2.FONT_HERSHEY_TRIPLEX, font_scale, thickness)
        level_x = (self.width - level_w) // 2
        level_y = self.level_display_y
        
        # Draw with glow effect
        cv2.putText(frame, level_text, (level_x, level_y), cv2.FONT_HERSHEY_TRIPLEX, 
                   font_scale, self.COLORS['glow_cyan'], thickness + 1)
        cv2.putText(frame, level_text, (level_x, level_y), cv2.FONT_HERSHEY_TRIPLEX, 
                   font_scale, self.COLORS['bright_blue'], thickness)
        
        # Draw difficulty text below
        diff_font_scale = 0.6
        diff_thickness = 1
        (diff_w, diff_h), _ = cv2.getTextSize(difficulty_text, cv2.FONT_HERSHEY_TRIPLEX, diff_font_scale, diff_thickness)
        diff_x = (self.width - diff_w) // 2
        diff_y = level_y + 30
        cv2.putText(frame, difficulty_text, (diff_x, diff_y), cv2.FONT_HERSHEY_TRIPLEX, 
                   diff_font_scale, self.COLORS['text_secondary'], diff_thickness)
        
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
        
        text, font_scale, thickness = "< BACK", 0.6, 2
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, font_scale, thickness)
        text_x = bb['x'] + (bb['w'] - text_w) // 2
        text_y = bb['y'] + (bb['h'] + text_h) // 2
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, self.COLORS['text_primary'], thickness)
    
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
            cv2.putText(frame, text, (self.camera_x + 120, self.camera_y + self.camera_height // 2), 
                       cv2.FONT_HERSHEY_TRIPLEX, 0.8, self.COLORS['text_primary'], 2)
        else:
            cam_frame = cv2.resize(camera_frame, (self.camera_width, self.camera_height))
            detection_frame = cam_frame.copy()
            
            # Load pattern mask
            pattern_overlay, pattern_alpha = self.load_blueprint(self.current_level)
            
            # Run cloth detection to find where to position the pattern
            cloth_bbox = None
            self.frame_counter += 1
            process_detection = (self.frame_counter % self.frame_skip == 0)
            
            if self.cloth_model is not None:
                try:
                    if not process_detection:
                        cloth_bbox = self.last_cloth_bbox
                    else:
                        # Pass cam_frame directly — YOLO letterboxes internally and
                        # returns xyxy already in cam_frame coordinate space
                        results = self.cloth_model(
                            cam_frame,
                            conf=self.confidence_threshold,
                            iou=self.iou_threshold,
                            imgsz=576,
                            verbose=False
                        )
                        
                        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                            boxes = results[0].boxes
                            confs = boxes.conf.cpu().numpy()
                            best_idx = int(np.argmax(confs))
                            xyxy = boxes[best_idx].xyxy[0].cpu().numpy()
                            
                            x1 = int(xyxy[0])
                            y1 = int(xyxy[1])
                            x2 = int(xyxy[2])
                            y2 = int(xyxy[3])
                            cloth_bbox = (x1, y1, x2, y2)
                            print(f"👕 Cloth detected: conf={float(confs[best_idx]):.2f}")
                        
                        self.last_cloth_bbox = cloth_bbox
                except Exception as e:
                    print(f"Cloth detection error: {e}")
            
            # Apply weighted averaging to stabilize the bounding box
            raw_bbox = self.last_cloth_bbox
            if raw_bbox is not None:
                if not hasattr(self, 'bbox_history'):
                    self.bbox_history = []

                # Add the new bounding box to the history
                self.bbox_history.append(raw_bbox)

                # Keep only the last N bounding boxes
                max_history = 10  # Use the last 10 bounding boxes
                if len(self.bbox_history) > max_history:
                    self.bbox_history.pop(0)

                # Adjust weights to give more importance to recent bounding boxes
                weights = np.exp(np.linspace(0, 1, len(self.bbox_history)))
                weights /= weights.sum()  # Normalize weights

                # Compute the weighted average of the bounding boxes
                weighted_bbox = np.average(self.bbox_history, axis=0, weights=weights)
                self.smooth_cloth_bbox = tuple(weighted_bbox)
            cloth_bbox = (tuple(int(v) for v in self.smooth_cloth_bbox)
                          if self.smooth_cloth_bbox is not None else None)
            
            # Overlay pattern centered inside detected cloth region
            if pattern_overlay is not None and pattern_alpha is not None:
                overlay_h, overlay_w = pattern_overlay.shape[:2]
                
                if cloth_bbox is not None:
                    # Center pattern inside the detected cloth bounding box
                    cx = (cloth_bbox[0] + cloth_bbox[2]) // 2
                    cy = (cloth_bbox[1] + cloth_bbox[3]) // 2
                    x_offset = cx - overlay_w // 2
                    y_offset = cy - overlay_h // 2
                    # Clamp to frame bounds
                    x_offset = max(0, min(x_offset, self.camera_width - overlay_w))
                    y_offset = max(0, min(y_offset, self.camera_height - overlay_h))
                else:
                    # No cloth detected: center on camera frame
                    x_offset = (self.camera_width - overlay_w) // 2
                    y_offset = (self.camera_height - overlay_h) // 2
                
                overlay_end_x = min(x_offset + overlay_w, self.camera_width)
                overlay_end_y = min(y_offset + overlay_h, self.camera_height)
                actual_w = overlay_end_x - x_offset
                actual_h = overlay_end_y - y_offset
                
                if actual_w > 0 and actual_h > 0:
                    realtime_pattern = self.create_realtime_pattern(pattern_overlay, pattern_alpha)
                    
                    roi = cam_frame[y_offset:overlay_end_y, x_offset:overlay_end_x]
                    pattern_crop = realtime_pattern[0:actual_h, 0:actual_w] if realtime_pattern is not None else pattern_overlay[0:actual_h, 0:actual_w]
                    alpha_crop = pattern_alpha[0:actual_h, 0:actual_w]
                    
                    for c in range(3):
                        roi[:, :, c] = (alpha_crop * pattern_crop[:, :, c] * 0.7 +
                                      (1 - alpha_crop * 0.7) * roi[:, :, c])
                    cam_frame[y_offset:overlay_end_y, x_offset:overlay_end_x] = roi
                    
                    self.run_needle_pipeline(
                        detection_frame, pattern_alpha,
                        x_offset, y_offset, actual_w, actual_h,
                        process_detection
                    )
            
            # Draw cloth detection bounding box
            if cloth_bbox is not None:
                x1, y1, x2, y2 = cloth_bbox
                cv2.rectangle(cam_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "Cloth"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(cam_frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), (0, 255, 0), -1)
                cv2.putText(cam_frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Draw needle ROI box (cyan) so the user can see & reposition it
            _half = self.NEEDLE_ROI_SIZE // 2
            _nrx1 = max(0, int(self.needle_pos_x) - _half)
            _nry1 = max(0, int(self.needle_pos_y) - _half)
            _nrx2 = min(self.camera_width,  _nrx1 + self.NEEDLE_ROI_SIZE)
            _nry2 = min(self.camera_height, _nry1 + self.NEEDLE_ROI_SIZE)
            cv2.rectangle(cam_frame, (_nrx1, _nry1), (_nrx2, _nry2), (0, 255, 255), 2)
            cv2.putText(cam_frame, "Needle ROI",
                        (_nrx1, max(10, _nry1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Mark live selected-color detection at ROI center.
            self._matches_selected_color(detection_frame, self.needle_pos_x, self.needle_pos_y)
            color_cfg = self.color_profiles[self.selected_detection_color]
            if self.last_color_match:
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
            cv2.putText(cam_frame, marker_text,
                        (_nrx1, min(self.camera_height - 8, _nry2 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, marker_color, 1)
            
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
            cv2.putText(frame, label, (item_x + box_size + 8, legend_y), 
                       cv2.FONT_HERSHEY_TRIPLEX, 0.65, self.COLORS['text_secondary'], 2)

        target = self.color_profiles[self.selected_detection_color]['label']
        cv2.putText(frame, f"Detecting: {target}", (legend_x, legend_y + 30),
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, self.COLORS['text_primary'], 2)

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

        cv2.putText(frame, "DETECT COLOR", (x + 16, y + 22), cv2.FONT_HERSHEY_TRIPLEX,
                   0.45, self.COLORS['text_secondary'], 1)

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
            cv2.putText(frame, cfg['label'][0], (bx + 22, by + 23), cv2.FONT_HERSHEY_TRIPLEX,
                       0.55, self.COLORS['text_primary'], 2)

            self.color_buttons[key] = {'x': bx, 'y': by, 'w': btn_w, 'h': btn_h}

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

        cv2.putText(frame, "CONFIDENCE", (x + 16, y + 20), cv2.FONT_HERSHEY_TRIPLEX,
                   0.45, self.COLORS['text_secondary'], 1)

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
            cv2.putText(frame, label, (btn['x'] + 14, btn['y'] + 22), cv2.FONT_HERSHEY_TRIPLEX,
                       0.65, self.COLORS['text_primary'], 2)

        conf_text = f"{self.confidence_threshold:.2f}"
        (tw, _), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_TRIPLEX, 0.6, 2)
        cv2.putText(frame, conf_text, (x + (w - tw) // 2, btn_y + 22), cv2.FONT_HERSHEY_TRIPLEX,
                   0.6, self.COLORS['text_primary'], 2)
    
    def draw_guide_overlay(self, frame):
        """Draw game guide/tutorial overlay - multi-step"""
        # Dark semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), 
                     (20, 10, 5), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Guide panel
        panel_w = 650
        panel_h = 420
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
        (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_TRIPLEX, 1.3, 3)
        title_x = panel_x + (panel_w - title_w) // 2
        title_y = panel_y + 50
        cv2.putText(frame, title, (title_x, title_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, 1.3, self.COLORS['glow_cyan'], 3)
        
        # Step indicator
        step_text = f"Step {self.guide_step} of 4"
        (step_w, step_h), _ = cv2.getTextSize(step_text, cv2.FONT_HERSHEY_TRIPLEX, 0.6, 2)
        step_x = panel_x + panel_w - step_w - 30
        step_y = panel_y + 45
        cv2.putText(frame, step_text, (step_x, step_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, self.COLORS['text_secondary'], 2, cv2.LINE_AA)
        
        # Content based on current step
        content_x = panel_x + 40
        content_y = title_y + 60
        line_height = 50
        
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
            
            text, color, font_scale, thickness = instruction
            cv2.putText(frame, text, (content_x, y_pos), 
                       cv2.FONT_HERSHEY_TRIPLEX, font_scale, color, thickness, cv2.LINE_AA)
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
        (btn_text_w, btn_text_h), _ = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_TRIPLEX, 0.8, 2)
        btn_text_x = button_x + (button_w - btn_text_w) // 2
        btn_text_y = button_y + (button_h + btn_text_h) // 2
        
        cv2.putText(frame, button_text, (btn_text_x, btn_text_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, 0.8, self.COLORS['glow_cyan'], 3)
        cv2.putText(frame, button_text, (btn_text_x, btn_text_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, 0.8, self.COLORS['bright_blue'], 2)
    
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
        font_scale = 1.0
        thickness = 4
        (text_w, text_h), _ = cv2.getTextSize(self.warning_message, cv2.FONT_HERSHEY_TRIPLEX, font_scale, thickness)
        text_x = self.camera_x + (self.camera_width - text_w) // 2
        text_y = banner_y + (banner_height + text_h) // 2
        
        # Draw text with outline for visibility
        cv2.putText(frame, self.warning_message, (text_x, text_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 0), thickness + 3)  # Black outline
        cv2.putText(frame, self.warning_message, (text_x, text_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, font_scale, (255, 255, 255), thickness)  # White text
    
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
        title_font_scale = 0.9
        title_thickness = 2
        title_text = "STATS"
        (title_w, title_h), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_TRIPLEX, title_font_scale, title_thickness)
        title_x = x + (w - title_w) // 2
        title_y = y + 40
        cv2.putText(frame, title_text, (title_x, title_y), cv2.FONT_HERSHEY_TRIPLEX, 
                   title_font_scale, self.COLORS['glow_cyan'], title_thickness + 1)
        cv2.putText(frame, title_text, (title_x, title_y), cv2.FONT_HERSHEY_TRIPLEX, 
                   title_font_scale, self.COLORS['bright_blue'], title_thickness)
        
        # Draw horizontal divider
        cv2.line(frame, (x + 15, title_y + 15), (x + w - 15, title_y + 15), self.COLORS['medium_blue'], 2)
        
        # Stats content
        content_x = x + 20
        start_y = title_y + 45
        
        # Progress bar (moved up since accuracy/score removed)
        progress_y = start_y + 20
        cv2.putText(frame, "PROGRESS", (content_x, progress_y), cv2.FONT_HERSHEY_TRIPLEX, 
                   0.6, self.COLORS['text_secondary'], 2)
        
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
        progress_text = f"{self.pattern_progress:.0f}%"
        (prog_w, prog_h), _ = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_TRIPLEX, 0.6, 2)
        cv2.putText(frame, progress_text, (content_x + (bar_width - prog_w) // 2, bar_y + 16), 
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, self.COLORS['text_primary'], 2)
        
        # Segment indicator text
        segment_text_y = bar_y + bar_height + 18
        segment_label = f"SEGMENT {self.current_segment}/4"
        cv2.putText(frame, segment_label, (content_x, segment_text_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.COLORS['text_secondary'], 2)
    
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
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, font_scale, thickness)
        text_x = eb['x'] + (eb['w'] - text_w) // 2
        text_y = eb['y'] + (eb['h'] + text_h) // 2
        
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 
                   font_scale, self.COLORS['text_primary'], thickness)
    
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
        
        (title_w, title_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_TRIPLEX, 1.2, 3)
        title_x = panel_x + (panel_w - title_w) // 2
        cv2.putText(frame, title, (title_x, content_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, 1.2, title_color, 3)
        
        # Progress achieved
        content_y += 70
        progress_label = "Progress Achieved:"
        (prog_label_w, _), _ = cv2.getTextSize(progress_label, cv2.FONT_HERSHEY_TRIPLEX, 0.8, 2)
        label_x = panel_x + (panel_w - prog_label_w) // 2
        cv2.putText(frame, progress_label, (label_x, content_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, 0.8, self.COLORS['text_secondary'], 2)
        
        content_y += 50
        progress_text = f"{self.raw_progress:.1f}%"
        (prog_w, _), _ = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_TRIPLEX, 1.5, 3)
        prog_x = panel_x + (panel_w - prog_w) // 2
        cv2.putText(frame, progress_text, (prog_x, content_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, 1.5, self.COLORS['text_primary'], 3)
        
        # Requirement text
        content_y += 40
        req_text = "(Need 80% to pass)"
        (req_w, _), _ = cv2.getTextSize(req_text, cv2.FONT_HERSHEY_TRIPLEX, 0.6, 1)
        req_x = panel_x + (panel_w - req_w) // 2
        cv2.putText(frame, req_text, (req_x, content_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, self.COLORS['text_secondary'], 1)
        
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
        font_scale = 0.9
        thickness = 3
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, font_scale, thickness)
        text_x = button_x + (button_w - text_w) // 2
        text_y = button_y + (button_h + text_h) // 2
        
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 
                   font_scale, self.COLORS['text_primary'], thickness)
    
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
        font_scale = 0.9
        thickness = 3
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, font_scale, thickness)
        text_x = button_x + (button_w - text_w) // 2
        text_y = button_y + (button_h + text_h) // 2
        
        # Draw with glow effect
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 
                   font_scale, self.COLORS['glow_cyan'], thickness + 1)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 
                   font_scale, self.COLORS['bright_blue'], thickness)
    
    def draw_stat_item(self, frame, label, value, x, y, max_width, value_color=None):
        """Helper method to draw a stat item (label and value)"""
        if value_color is None:
            value_color = self.COLORS['text_primary']
        
        # Draw label - larger font
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_TRIPLEX, 
                   0.6, self.COLORS['text_secondary'], 2)
        
        # Draw value (right-aligned on same line) - larger font
        (val_w, val_h), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_TRIPLEX, 0.9, 2)
        value_x = x + max_width - val_w
        cv2.putText(frame, value, (value_x, y), cv2.FONT_HERSHEY_TRIPLEX, 
                   0.9, value_color, 2)

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
