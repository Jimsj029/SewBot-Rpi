import cv2
import numpy as np
import math
import os
import time
from ultralytics import YOLO
from music_manager import get_music_manager
from skimage.morphology import skeletonize


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
        
        # Manual ROI control (detection box) - 1 whole segment system
        # ROI coordinates are relative to camera_width x camera_height (560x420)
        self.num_roi_segments = 1
        self.segment_height = self.camera_height // self.num_roi_segments  # 420 / 1 = 420 pixels (full height)
        self.current_roi_segment = 1  # Always 1 since we have 1 segment
        
        self.roi_x = 216  # Centered horizontally
        self.roi_width = 128  # Width of ROI box
        self.roi_height = self.segment_height  # Height = one segment (42 pixels)
        self.roi_y = 0  # Will be calculated based on current_roi_segment
        
        # Back button (top left)
        self.back_button = {'x': 20, 'y': 20, 'w': 120, 'h': 50}
        
        # Evaluate button (below stats panel)
        self.evaluate_button = {'x': 800, 'y': 470, 'w': 200, 'h': 50}
        
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
        self.proximity_radius = 45  # Increased - spreads cyan color further to cover corners better
        
        # Track detected stitch positions to prevent re-detection
        self.detected_stitch_positions = []  # List of (x, y, w, h) tuples in camera frame coordinates
        self.stitch_overlap_threshold = 0.5  # IoU threshold for duplicate detection (0.5 = 50% overlap required)
        
        # Skeleton-based pattern tracking (STEP 4-6 & 12-13)
        self.pattern_skeleton = None  # Skeletonized pattern path
        self.pattern_mask = None  # Full pattern mask for overlap checking
        self.distance_map = None  # Distance transform from skeleton
        self.visited_mask = None  # Tracks which skeleton pixels have been visited
        self.completed_mask = None  # Marks completed sections
        self.tolerance = 15  # Increased - allow more deviation from skeleton (better for corners)
        
        # Segment tracking (divide pattern into 4 quarters)
        self.current_segment = 1  # 1=first 25%, 2=25-50%, 3=50-75%, 4=75-100%
        self.highest_segment_reached = 1  # Track highest segment to prevent going backwards
        self.segment_colors = {
            'completed': (255, 255, 0),    # Cyan in BGR (completed sections)
            'current': (0, 255, 255),      # Yellow in BGR (current section to sew)
            'upcoming': (100, 100, 100)    # Dim Gray (upcoming sections)
        }
        
        # Out-of-segment detection
        # Removed: out_of_segment_warning, warning_flash_phase (dead code - warning never triggers)
        # Removed: current_combo, max_combo, stitches_detected (tracked but never displayed)
        
        # Load models from models directory
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Detection settings
        self.confidence_threshold = 0.2  # Lowered - model detects stitches but with lower confidence
        self.iou_threshold = 0.6  # Intersection over Union threshold
        
        # Performance optimization: Frame skipping for detection
        # Process detection every N frames to reduce CPU load
        self.frame_skip = 1  # Process every frame for better corner detection (may reduce FPS slightly)
        self.frame_counter = 0
        self.last_detected_masks = []  # Cache last detection results
        
        # Process only top N detections for box computation (huge performance boost)
        self.max_detections_to_process = 5  # Process top 5 detections (helps with corners/turns)
        
        # Optical flow tracking for FPS optimization (Target: 20-25 FPS)
        # Reduces YOLO calls by tracking between detections using Lucas-Kanade optical flow
        self.YOLO_INTERVAL = 3  # Run YOLO every 3 frames, track in between (adjust to 4-5 for even higher FPS)
        self.prev_gray = None  # Previous grayscale frame for optical flow
        self.prev_point = None  # Previous tracked point (center of detection)
        self.optical_flow_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Confidence gating for stability (prevents drift)
        self.tracking_error_threshold = 50.0  # Re-run YOLO if tracking error exceeds this
        
        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        # Load stitch detection model (ONNX model)
        try:
            # Use ONNX model (best.onnx)
            stitch_model_path = os.path.join(models_dir, 'best.onnx')
            
            if os.path.exists(stitch_model_path):
                print(f"Loading stitch detection model: {stitch_model_path}")
                self.stitch_model = YOLO(stitch_model_path, task='detect')
                print(f"✓ Stitch detection model loaded successfully!")
                print(f"  Model type: {self.stitch_model.task if hasattr(self.stitch_model, 'task') else 'unknown'}")
                
                # Print available classes
                if hasattr(self.stitch_model, 'names'):
                    print(f"  Detected classes:")
                    for idx, name in self.stitch_model.names.items():
                        print(f"    Class {idx}: {name}")
                print(f"  Confidence threshold: {self.confidence_threshold}")
                print(f"  IOU threshold: {self.iou_threshold}")
                
                # Test the model with dummy data
                print("  Testing model inference...")
                try:
                    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
                    test_result = self.stitch_model(test_img, conf=self.confidence_threshold, verbose=False)
                    print("  ✓ Model test successful!")
                except Exception as test_error:
                    print(f"  ⚠ Model test failed (but model is loaded): {test_error}")
                    print("  Model will still be used for real-time detection")
            else:
                print(f"⚠ Stitch model not found: {stitch_model_path}")
                self.stitch_model = None
        except Exception as e:
            print(f"⚠ ERROR loading stitch model: {e}")
            import traceback
            traceback.print_exc()
            self.stitch_model = None
        
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
        
        mask = cv2.resize(mask, (self.uniform_width, self.uniform_height))
        
        # Store full pattern mask for overlap checking
        self.pattern_mask = mask.copy()
        
        # STEP 4 — CREATE SKELETON PATH
        # Create skeleton from pattern for precise path tracking
        try:
            binary = mask > 0
            skeleton = skeletonize(binary)
            self.pattern_skeleton = (skeleton * 255).astype(np.uint8)
            
            # STEP 5 — PRECOMPUTE DISTANCE MAP
            # Distance map shows distance from skeleton at each pixel
            self.distance_map = cv2.distanceTransform(
                255 - self.pattern_skeleton,
                cv2.DIST_L2,
                5
            )
            
            # Initialize visited and completed masks
            self.visited_mask = np.zeros_like(self.pattern_skeleton)
            self.completed_mask = np.zeros_like(self.pattern_skeleton)
            
    
        except Exception as e:
            print(f"⚠ Warning: Could not create skeleton: {e}")
            self.pattern_skeleton = None
            self.distance_map = None
            self.pattern_mask = mask.copy()  # Fallback to full mask
        
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
        self.is_evaluated = False
        self.level_completed = False
        # Clear accumulated stitch mask for real-time coloring
        self.completed_stitch_mask = None
        # Clear tracked stitch positions
        self.detected_stitch_positions = []
        # Reset ROI to first segment
        self.current_roi_segment = 1
        self.roi_y = 0
        # Clear skeleton tracking masks
        if self.pattern_skeleton is not None:
            self.visited_mask = np.zeros_like(self.pattern_skeleton)
            self.completed_mask = np.zeros_like(self.pattern_skeleton)
        # Reset optical flow tracking
        self.prev_gray = None
        self.prev_point = None
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
        
        # If we have completed stitches, color those pattern pixels cyan
        if self.completed_stitch_mask is not None:
            # Resize completed mask to match pattern size
            completed_mask_resized = cv2.resize(self.completed_stitch_mask, (width, height))
            completed_binary = (completed_mask_resized > 128).astype(np.uint8)
            
            # Only color pattern pixels that have completed stitches (no dilation)
            completed_pattern_pixels = np.logical_and(pattern_pixels, completed_binary > 0)
            colored_overlay[completed_pattern_pixels] = self.segment_colors['completed']
        
        return colored_overlay
    
    def is_stitch_already_detected(self, box_x, box_y, box_w, box_h):
        """Check if a detected stitch overlaps significantly with any already tracked stitch.
        
        Args:
            box_x, box_y, box_w, box_h: Bounding box of new detection in camera frame coordinates
            
        Returns:
            True if this stitch was already detected (overlaps with existing), False otherwise
        """
        # Debug: Show tracking count every 60 frames
        if self.frame_counter % 60 == 0 and len(self.detected_stitch_positions) > 0:
            print(f"📊 Currently tracking {len(self.detected_stitch_positions)} stitch positions")
        
        if len(self.detected_stitch_positions) == 0:
            return False
        
        # Calculate IoU (Intersection over Union) with all existing stitches
        new_box = (box_x, box_y, box_x + box_w, box_y + box_h)
        new_center_x = box_x + box_w / 2
        new_center_y = box_y + box_h / 2
        
        for tracked_x, tracked_y, tracked_w, tracked_h in self.detected_stitch_positions:
            # Quick check: Distance between centers (faster than IoU)
            tracked_center_x = tracked_x + tracked_w / 2
            tracked_center_y = tracked_y + tracked_h / 2
            center_distance = np.sqrt((new_center_x - tracked_center_x)**2 + (new_center_y - tracked_center_y)**2)
            
            # If centers are very close (within 20 pixels), consider it duplicate
            if center_distance < 20:
                if self.frame_counter % 60 == 0:
                    print(f"  ✋ BLOCKED duplicate (center distance={center_distance:.1f}px)")
                return True
            
            # Convert tracked box to xyxy format
            tracked_box = (tracked_x, tracked_y, tracked_x + tracked_w, tracked_y + tracked_h)
            
            # Calculate intersection
            x1_inter = max(new_box[0], tracked_box[0])
            y1_inter = max(new_box[1], tracked_box[1])
            x2_inter = min(new_box[2], tracked_box[2])
            y2_inter = min(new_box[3], tracked_box[3])
            
            if x2_inter > x1_inter and y2_inter > y1_inter:
                # There is an intersection
                intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                
                # Calculate union
                new_area = box_w * box_h
                tracked_area = tracked_w * tracked_h
                union_area = new_area + tracked_area - intersection_area
                
                # Calculate IoU
                if union_area > 0:
                    iou = intersection_area / union_area
                    
                    # Debug: Show IoU calculations
                    if self.frame_counter % 60 == 0 and iou > 0.1:
                        print(f"  IoU: {iou:.3f} (threshold: {self.stitch_overlap_threshold}) - New:({box_x},{box_y},{box_w},{box_h}) vs Tracked:({tracked_x},{tracked_y},{tracked_w},{tracked_h})")
                    
                    # If IoU exceeds threshold, consider it as already detected
                    if iou >= self.stitch_overlap_threshold:
                        if self.frame_counter % 60 == 0:
                            print(f"  ✋ BLOCKED duplicate (IoU={iou:.3f})")
                        return True
        
        return False

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
        
        # Draw FPS counter (top right)
        if self.current_fps > 0:
            fps_text = f"FPS: {self.current_fps:.1f}"
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            fps_x = self.width - text_w - 20
            fps_y = 30
            # Color based on FPS (green if good, yellow if okay, red if low)
            if self.current_fps >= 20:
                fps_color = (0, 255, 0)  # Green
            elif self.current_fps >= 15:
                fps_color = (0, 255, 255)  # Yellow
            else:
                fps_color = (0, 100, 255)  # Red
            cv2.putText(frame, fps_text, (fps_x, fps_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, fps_color, thickness)
        
        # Draw camera feed
        self.draw_camera_feed(frame, camera_frame)
        
        # Draw score/stats panel
        self.draw_score_panel(frame)
        
        # Draw evaluate button
        self.draw_evaluate_button(frame)
        
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
            
            # Load pattern mask for comparison
            pattern_overlay, pattern_alpha = self.load_blueprint(self.current_level)
            
            # Run stitch detection using ONNX detection model with ROI optimization
            detected_stitch_masks = []
            detection_overlay = None
            roi_bounds = None  # Store ROI boundaries for visualization
            
            # Frame skipping for performance optimization
            self.frame_counter += 1
            self.fps_frame_count += 1
            
            # Calculate FPS every 30 frames
            if self.fps_frame_count >= 30:
                elapsed = time.time() - self.fps_start_time
                self.current_fps = self.fps_frame_count / elapsed if elapsed > 0 else 0
                self.fps_frame_count = 0
                self.fps_start_time = time.time()
                print(f"⚡ FPS: {self.current_fps:.1f}")
            
            process_detection = (self.frame_counter % self.frame_skip == 0)
            
            if self.stitch_model is not None:
                try:
                    # Calculate ROI Y position based on current segment (top-to-bottom)
                    self.roi_y = (self.current_roi_segment - 1) * self.segment_height
                    
                    # Use cached results if skipping this frame
                    if not process_detection:
                        detected_stitch_masks = self.last_detected_masks
                    else:
                        # ========== MANUAL ROI METHOD with OPTICAL FLOW OPTIMIZATION ==========
                        # Use fixed ROI coordinates (user-controllable)
                        
                        # Calculate ROI boundaries (ensure within frame bounds)
                        roi_x1 = max(0, self.roi_x)
                        roi_y1 = max(0, self.roi_y)
                        roi_x2 = min(self.camera_width, self.roi_x + self.roi_width)
                        roi_y2 = min(self.camera_height, self.roi_y + self.roi_height)
                        
                        # Store ROI bounds for visualization
                        roi_bounds = (roi_x1, roi_y1, roi_x2, roi_y2)
                        
                        # Extract ROI from camera frame
                        roi_frame = cam_frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                        
                        # Convert ROI to grayscale for optical flow
                        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                        
                        # OPTIMIZATION: Run YOLO every N frames, track with optical flow in between
                        run_yolo = (self.frame_counter % self.YOLO_INTERVAL == 0) or (self.prev_point is None)
                        
                        if run_yolo:
                            # YOLO DETECTION PHASE
                            # ONNX model requires 256x256 input (fixed size)
                            roi_resized = cv2.resize(roi_frame, (256, 256))
                            
                            # Run inference ONLY on ROI
                            results = self.stitch_model(
                                roi_resized, 
                                conf=self.confidence_threshold,
                                iou=self.iou_threshold,
                                imgsz=256,
                                verbose=False
                            )
                            
                            # Debug: Show detection count
                            if len(results) > 0 and results[0].boxes is not None:
                                num_raw_detections = len(results[0].boxes)
                                if num_raw_detections > 0 and self.frame_counter % 10 == 0:
                                    print(f"🔍 YOLO: {num_raw_detections} detections in ROI at ({roi_x1},{roi_y1})")
                            
                            # Get detection visualization from YOLO
                            roi_detection_overlay = results[0].plot(boxes=True, labels=False)
                            
                            # Resize detection overlay back to ROI size
                            roi_detection_overlay = cv2.resize(roi_detection_overlay, (roi_x2 - roi_x1, roi_y2 - roi_y1))
                            
                            # Create full frame detection overlay and place ROI result
                            detection_overlay = cam_frame.copy()
                            detection_overlay[roi_y1:roi_y2, roi_x1:roi_x2] = roi_detection_overlay
                            
                            # Extract best detection for optical flow tracking
                            best_detection_point = None
                            
                            # Extract bounding boxes and convert to masks
                            # OPTIMIZATION: Only process top detections to reduce CPU load
                            for result in results:
                                if hasattr(result, 'boxes') and result.boxes is not None:
                                    boxes = result.boxes
                                
                                # ROI dimensions
                                roi_h = roi_y2 - roi_y1
                                roi_w = roi_x2 - roi_x1
                                
                                # Sort by confidence and take top N only
                                confidences = boxes.conf.cpu().numpy()
                                
                                # Filter by confidence threshold first
                                valid_indices = [i for i, conf in enumerate(confidences) if conf >= self.confidence_threshold]
                                
                                if len(valid_indices) == 0:
                                    continue
                                
                                # Sort valid detections by Y position (top to bottom) instead of confidence
                                # This makes detection follow the natural sewing order
                                boxes_data = []
                                for i in valid_indices:
                                    box = boxes[i]
                                    xyxy = box.xyxy[0].cpu().numpy()
                                    y1 = xyxy[1]  # Get Y coordinate
                                    boxes_data.append((i, y1))
                                
                                # Sort by Y position (top to bottom)
                                sorted_valid = sorted(boxes_data, key=lambda x: x[1])
                                top_indices = [idx for idx, y in sorted_valid[:self.max_detections_to_process]]
                                
                                for idx in top_indices:
                                    if idx >= len(boxes):  # Safety check
                                        break
                                    
                                    box = boxes[idx]
                                    conf = float(box.conf[0].cpu().numpy())
                                    
                                    # Get box coordinates (in resized 256x256 space)
                                    xyxy = box.xyxy[0].cpu().numpy()
                                    x1, y1, x2, y2 = map(int, xyxy)
                                    
                                    # Convert box coordinates to camera frame coordinates for tracking
                                    # Scale from 256x256 back to ROI size, then add ROI offset
                                    box_x1_cam = int((x1 / 256.0) * roi_w) + roi_x1
                                    box_y1_cam = int((y1 / 256.0) * roi_h) + roi_y1
                                    box_x2_cam = int((x2 / 256.0) * roi_w) + roi_x1
                                    box_y2_cam = int((y2 / 256.0) * roi_h) + roi_y1
                                    box_w_cam = box_x2_cam - box_x1_cam
                                    box_h_cam = box_y2_cam - box_y1_cam
                                    
                                    # Check if this stitch was already detected
                                    is_already_detected = self.is_stitch_already_detected(
                                        box_x1_cam, box_y1_cam, box_w_cam, box_h_cam
                                    )
                                    
                                    if is_already_detected:
                                        if self.frame_counter % 30 == 0:
                                            print(f"⏭️  Skipped duplicate at ({box_x1_cam},{box_y1_cam})")
                                        continue
                                    
                                    # Debug: Track new detections
                                    if self.frame_counter % 30 == 0:
                                        print(f"✅ NEW detection at ({box_x1_cam},{box_y1_cam}) conf={conf:.2f}")
                                    
                                    # Save center point for optical flow tracking (use first valid detection)
                                    if best_detection_point is None:
                                        # Calculate center point in ROI coordinates
                                        cx_roi = (x1 + x2) // 2
                                        cy_roi = (y1 + y2) // 2
                                        best_detection_point = (cx_roi, cy_roi)
                                    
                                    # Create binary mask from bounding box
                                    mask_resized = np.zeros((256, 256), dtype=np.uint8)
                                    mask_resized[y1:y2, x1:x2] = 1
                                    
                                    # Resize mask to ROI size
                                    mask_binary = cv2.resize(mask_resized, (roi_w, roi_h))
                                    mask_binary = (mask_binary > 0.5).astype(np.uint8)
                                    
                                    # Store ROI mask with its bounds (avoid full-frame expansion)
                                    detected_stitch_masks.append({
                                        'mask': mask_binary,  # ROI-sized mask only
                                        'confidence': conf,
                                        'roi_bounds': (roi_x1, roi_y1, roi_x2, roi_y2),  # ROI position
                                        'box_cam': (box_x1_cam, box_y1_cam, box_w_cam, box_h_cam)  # Box in camera coords
                                    })
                            
                            # After YOLO detection, save the best point for optical flow tracking
                            if best_detection_point is not None:
                                # Convert to numpy array format required by optical flow
                                # Point is in 256x256 space, need to scale to ROI coordinates
                                roi_h = roi_y2 - roi_y1
                                roi_w = roi_x2 - roi_x1
                                cx_scaled = int((best_detection_point[0] / 256.0) * roi_w)
                                cy_scaled = int((best_detection_point[1] / 256.0) * roi_h)
                                self.prev_point = np.array([[cx_scaled, cy_scaled]], dtype=np.float32)
                                if self.frame_counter % 10 == 0:
                                    print(f"🎯 YOLO: Tracking point set at ({cx_scaled},{cy_scaled}) in ROI")
                            
                            # Cache detection results for next frame
                            self.last_detected_masks = detected_stitch_masks
                        
                        else:
                            # OPTICAL FLOW TRACKING PHASE
                            # Track previous point using Lucas-Kanade optical flow
                            if self.prev_gray is not None and self.prev_point is not None:
                                new_point, status, err = cv2.calcOpticalFlowPyrLK(
                                    self.prev_gray,
                                    gray_roi,
                                    self.prev_point,
                                    None,
                                    **self.optical_flow_params
                                )
                                
                                # Check if tracking succeeded
                                if status is not None and status[0] == 1:
                                    # Get tracked point coordinates
                                    x, y = new_point.ravel()
                                    
                                    # CONFIDENCE GATING: Check for drift conditions
                                    tracking_error = err[0][0] if err is not None else 0
                                    roi_h = roi_y2 - roi_y1
                                    roi_w = roi_x2 - roi_x1
                                    point_out_of_roi = (x < 0 or x >= roi_w or y < 0 or y >= roi_h)
                                    
                                    # Force YOLO re-detection if drift detected
                                    if tracking_error > self.tracking_error_threshold:
                                        if self.frame_counter % 10 == 0:
                                            print(f"⚠️ OPTICAL FLOW: High tracking error ({tracking_error:.1f}), forcing YOLO re-detection")
                                        self.prev_point = None
                                        detected_stitch_masks = self.last_detected_masks
                                    elif point_out_of_roi:
                                        if self.frame_counter % 10 == 0:
                                            print(f"⚠️ OPTICAL FLOW: Point left ROI ({int(x)},{int(y)}), forcing YOLO re-detection")
                                        self.prev_point = None
                                        detected_stitch_masks = self.last_detected_masks
                                    else:
                                        # Tracking is good, continue with optical flow
                                        self.prev_point = new_point
                                        
                                        if self.frame_counter % 10 == 0:
                                            print(f"🔄 OPTICAL FLOW: Tracking at ({int(x)},{int(y)}) in ROI, error={tracking_error:.1f}")
                                        
                                        # Draw tracked point on ROI for visualization
                                        roi_with_tracking = roi_frame.copy()
                                        cv2.circle(roi_with_tracking, (int(x), int(y)), 5, (0, 255, 0), -1)
                                        cv2.circle(roi_with_tracking, (int(x), int(y)), 10, (0, 255, 0), 2)
                                        
                                        # Create detection overlay with tracked point
                                        detection_overlay = cam_frame.copy()
                                        detection_overlay[roi_y1:roi_y2, roi_x1:roi_x2] = roi_with_tracking
                                        
                                        # Use cached detection masks from last YOLO run
                                        detected_stitch_masks = self.last_detected_masks
                                else:
                                    # Tracking failed, will run YOLO next frame
                                    if self.frame_counter % 10 == 0:
                                        print("⚠️ OPTICAL FLOW: Tracking lost, will re-detect with YOLO next frame")
                                    self.prev_point = None
                                    detected_stitch_masks = self.last_detected_masks
                            else:
                                # No previous frame to track from, use cached results
                                detected_stitch_masks = self.last_detected_masks
                        
                        # Update previous grayscale frame for next optical flow iteration
                        self.prev_gray = gray_roi.copy()
                        
                        # Removed: Segment auto-moving logic (dead code with 1 segment)
                    
                except Exception as e:
                    print(f"Stitch detection error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # First, overlay pattern mask (draw pattern first, detection on top)
            pattern_applied = False
            if pattern_overlay is not None and pattern_alpha is not None:
                overlay_h, overlay_w = pattern_overlay.shape[:2]
                
                # Center pattern on camera frame
                x_offset = (self.camera_width - overlay_w) // 2
                y_offset = (self.camera_height - overlay_h) // 2
                
                # Ensure we don't go out of bounds
                if x_offset >= 0 and y_offset >= 0:
                    # Calculate actual overlay dimensions considering bounds
                    overlay_end_x = min(x_offset + overlay_w, self.camera_width)
                    overlay_end_y = min(y_offset + overlay_h, self.camera_height)
                    actual_w = overlay_end_x - x_offset
                    actual_h = overlay_end_y - y_offset
                    
                    if actual_w > 0 and actual_h > 0:
                        # If we have detection overlay, use that as base (YOLO rendered masks)
                        if detection_overlay is not None:
                            cam_frame = detection_overlay
                        
                        # *** UPDATE GAME STATS FIRST (before coloring pattern) ***
                        self.update_game_stats(detected_stitch_masks, pattern_alpha, 
                                             x_offset, y_offset, actual_w, actual_h)
                        
                        # Create real-time colored pattern based on completed stitches
                        realtime_pattern = self.create_realtime_pattern(pattern_overlay, pattern_alpha)
                        
                        # Apply pattern overlay on top
                        roi = cam_frame[y_offset:overlay_end_y, x_offset:overlay_end_x]
                        pattern_crop = realtime_pattern[0:actual_h, 0:actual_w] if realtime_pattern is not None else pattern_overlay[0:actual_h, 0:actual_w]
                        alpha_crop = pattern_alpha[0:actual_h, 0:actual_w]
                        
                        # Apply with higher opacity for better visibility
                        for c in range(3):
                            roi[:, :, c] = (alpha_crop * pattern_crop[:, :, c] * 0.7 + 
                                          (1 - alpha_crop * 0.7) * roi[:, :, c])
                        cam_frame[y_offset:overlay_end_y, x_offset:overlay_end_x] = roi
                        pattern_applied = True
            
            # If detection exists but pattern wasn't applied, use YOLO's visualization
            if not pattern_applied and detection_overlay is not None:
                cam_frame = detection_overlay
            
            # Draw ROI detection box overlay (shows where detection is happening)
            if roi_bounds is not None:
                roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
                # Draw semi-transparent box
                overlay_roi = cam_frame.copy()
                cv2.rectangle(overlay_roi, (roi_x1, roi_y1), (roi_x2, roi_y2), 
                             self.COLORS['cyan'], 2)
                cv2.addWeighted(overlay_roi, 0.7, cam_frame, 0.3, 0, cam_frame)
                
                # Add corner markers for better visibility
                corner_len = 15
                corner_thick = 3
                # Top-left
                cv2.line(cam_frame, (roi_x1, roi_y1), (roi_x1 + corner_len, roi_y1), 
                        self.COLORS['neon_blue'], corner_thick)
                cv2.line(cam_frame, (roi_x1, roi_y1), (roi_x1, roi_y1 + corner_len), 
                        self.COLORS['neon_blue'], corner_thick)
                # Top-right
                cv2.line(cam_frame, (roi_x2, roi_y1), (roi_x2 - corner_len, roi_y1), 
                        self.COLORS['neon_blue'], corner_thick)
                cv2.line(cam_frame, (roi_x2, roi_y1), (roi_x2, roi_y1 + corner_len), 
                        self.COLORS['neon_blue'], corner_thick)
                # Bottom-left
                cv2.line(cam_frame, (roi_x1, roi_y2), (roi_x1 + corner_len, roi_y2), 
                        self.COLORS['neon_blue'], corner_thick)
                cv2.line(cam_frame, (roi_x1, roi_y2), (roi_x1, roi_y2 - corner_len), 
                        self.COLORS['neon_blue'], corner_thick)
                # Bottom-right
                cv2.line(cam_frame, (roi_x2, roi_y2), (roi_x2 - corner_len, roi_y2), 
                        self.COLORS['neon_blue'], corner_thick)
                cv2.line(cam_frame, (roi_x2, roi_y2), (roi_x2, roi_y2 - corner_len), 
                        self.COLORS['neon_blue'], corner_thick)
                
                # Add label showing ROI segment
                label_text = f"SEGMENT {self.current_roi_segment}/{self.num_roi_segments}"
                label_color = self.COLORS['neon_blue']
                
                font_scale = 0.4
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_TRIPLEX, 
                                                       font_scale, thickness)
                label_x = roi_x1 + 5
                label_y = roi_y1 - 5 if roi_y1 > 20 else roi_y1 + 15
                
                # Draw text background
                cv2.rectangle(cam_frame, (label_x - 2, label_y - text_h - 2), 
                            (label_x + text_w + 2, label_y + 2), 
                            self.COLORS['dark_blue'], -1)
                # Draw text
                cv2.putText(cam_frame, label_text, (label_x, label_y), 
                           cv2.FONT_HERSHEY_TRIPLEX, font_scale, 
                           label_color, thickness)
            
            frame[self.camera_y:self.camera_y+self.camera_height, 
                  self.camera_x:self.camera_x+self.camera_width] = cam_frame
            
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
    
    def update_game_stats(self, detected_stitch_masks, pattern_alpha, x_offset, y_offset, actual_w, actual_h):
        """Update game statistics based on detected stitches vs pattern (ROI-optimized)"""
        if len(detected_stitch_masks) == 0:
            if self.frame_counter % 30 == 0:
                print(f"⚠️ No detected stitches to map (empty list)")
            return
        
        # Initialize completed stitch mask if needed (in pattern coordinates)
        if self.completed_stitch_mask is None:
            self.completed_stitch_mask = np.zeros((self.uniform_height, self.uniform_width), dtype=np.uint8)
        
        # Debug: Check if we have detections to process
        if self.frame_counter % 15 == 0:
            print(f"🔍 Processing {len(detected_stitch_masks)} detected stitches for mapping...")
            print(f"   Pattern mask loaded: {'✅ YES' if self.pattern_mask is not None else '❌ NO'}")
            print(f"   Pattern area: x={x_offset}, y={y_offset}, w={actual_w}, h={actual_h}")
            # Check first stitch data structure
            if len(detected_stitch_masks) > 0:
                first_stitch = detected_stitch_masks[0]
                print(f"   First stitch keys: {first_stitch.keys()}")
                if 'box_cam' in first_stitch:
                    print(f"   box_cam: {first_stitch['box_cam']}")
                else:
                    print(f"   ❌ Missing 'box_cam' key!")
                if 'roi_bounds' in first_stitch:
                    print(f"   roi_bounds: {first_stitch['roi_bounds']}")
                else:
                    print(f"   ❌ Missing 'roi_bounds' key!")
        
        # Process each detection individually and map to pattern coordinates
        processed_count = 0
        skipped_no_keys = 0
        skipped_no_intersection = 0
        skipped_invalid_bounds = 0
        accepted_count = 0
        rejected_overlap = 0
        
        for stitch_data in detected_stitch_masks:
            if 'box_cam' not in stitch_data or 'roi_bounds' not in stitch_data:
                skipped_no_keys += 1
                continue
            
            processed_count += 1
            
            # Debug first stitch coordinates
            if processed_count == 1 and self.frame_counter % 15 == 0:
                print(f"   🔬 FIRST STITCH DEBUG:")
                
            # Get INDIVIDUAL stitch box in camera coordinates (not ROI!)
            stitch_x, stitch_y, stitch_w, stitch_h = stitch_data['box_cam']
            stitch_x1_cam = stitch_x
            stitch_y1_cam = stitch_y
            stitch_x2_cam = stitch_x + stitch_w
            stitch_y2_cam = stitch_y + stitch_h
            
            if processed_count == 1 and self.frame_counter % 15 == 0:
                print(f"      Stitch in camera: ({stitch_x1_cam},{stitch_y1_cam})-({stitch_x2_cam},{stitch_y2_cam})")
            
            # Get ROI position to extract from mask
            roi_x1, roi_y1, roi_x2, roi_y2 = stitch_data['roi_bounds']
            stitch_mask = stitch_data['mask']  # ROI-sized mask with stitch area set to 1
            
            # Pattern bounds in camera coordinates
            pattern_x1, pattern_y1 = x_offset, y_offset
            pattern_x2, pattern_y2 = x_offset + actual_w, y_offset + actual_h
            
            if processed_count == 1 and self.frame_counter % 15 == 0:
                print(f"      Pattern in camera: ({pattern_x1},{pattern_y1})-({pattern_x2},{pattern_y2})")
            
            # Calculate intersection between INDIVIDUAL STITCH and pattern
            intersect_x1 = max(stitch_x1_cam, pattern_x1)
            intersect_y1 = max(stitch_y1_cam, pattern_y1)
            intersect_x2 = min(stitch_x2_cam, pattern_x2)
            intersect_y2 = min(stitch_y2_cam, pattern_y2)
            
            if processed_count == 1 and self.frame_counter % 15 == 0:
                print(f"      Intersection: ({intersect_x1},{intersect_y1})-({intersect_x2},{intersect_y2})")
            
            if intersect_x2 <= intersect_x1 or intersect_y2 <= intersect_y1:
                if processed_count == 1 and self.frame_counter % 15 == 0:
                    print(f"      ❌ NO INTERSECTION - stitch outside pattern area!")
                skipped_no_intersection += 1
                continue  # No overlap with pattern
            
            # Extract the overlapping portion from the stitch mask
            # Convert intersection coords to ROI-relative coords
            mask_x1 = max(0, intersect_x1 - roi_x1)
            mask_y1 = max(0, intersect_y1 - roi_y1)
            mask_x2 = min(roi_x2 - roi_x1, intersect_x2 - roi_x1)
            mask_y2 = min(roi_y2 - roi_y1, intersect_y2 - roi_y1)
            
            # Extract the overlapping region from stitch mask (raw, no dilation)
            stitch_crop = stitch_mask[mask_y1:mask_y2, mask_x1:mask_x2]
            
            if stitch_crop.size == 0:
                continue
            
            # Convert intersection to pattern-relative coords
            pattern_rel_x1 = intersect_x1 - pattern_x1
            pattern_rel_y1 = intersect_y1 - pattern_y1
            pattern_rel_x2 = intersect_x2 - pattern_x1
            pattern_rel_y2 = intersect_y2 - pattern_y1
            
            # Scale to uniform pattern coordinates (200x300)
            pattern_full_x1 = int((pattern_rel_x1 / actual_w) * self.uniform_width)
            pattern_full_y1 = int((pattern_rel_y1 / actual_h) * self.uniform_height)
            pattern_full_x2 = int((pattern_rel_x2 / actual_w) * self.uniform_width)
            pattern_full_y2 = int((pattern_rel_y2 / actual_h) * self.uniform_height)
            
            # Ensure bounds are valid
            pattern_full_x1 = max(0, min(pattern_full_x1, self.uniform_width))
            pattern_full_y1 = max(0, min(pattern_full_y1, self.uniform_height))
            pattern_full_x2 = max(0, min(pattern_full_x2, self.uniform_width))
            pattern_full_y2 = max(0, min(pattern_full_y2, self.uniform_height))
            
            if pattern_full_x2 <= pattern_full_x1 or pattern_full_y2 <= pattern_full_y1:
                skipped_invalid_bounds += 1
                continue
            
            # Resize stitch mask to final pattern coordinates
            final_w = pattern_full_x2 - pattern_full_x1
            final_h = pattern_full_y2 - pattern_full_y1
            stitch_final = cv2.resize(stitch_crop, (final_w, final_h))
            
            # CHECK: Only add to completed mask if stitch overlaps with actual pattern line
            if self.pattern_mask is not None:
                # Debug first stitch
                if processed_count == 1 and self.frame_counter % 15 == 0:
                    print(f"      Checking overlap with pattern mask...")
                    
                # Get the pattern region this stitch covers (use FULL mask, not thin skeleton)
                pattern_region = self.pattern_mask[pattern_full_y1:pattern_full_y2, 
                                                   pattern_full_x1:pattern_full_x2]
                
                # Check if ANY part of this stitch overlaps with the pattern
                stitch_binary = (stitch_final > 0).astype(np.uint8)
                overlap = np.logical_and(stitch_binary > 0, pattern_region > 0)
                overlap_pixels = np.sum(overlap)
                
                # Calculate overlap percentage
                stitch_pixels = np.sum(stitch_binary > 0)
                overlap_ratio = (overlap_pixels / stitch_pixels * 100) if stitch_pixels > 0 else 0
                
                # Debug first stitch overlap
                if processed_count == 1 and self.frame_counter % 15 == 0:
                    print(f"      Pattern mask shape: {pattern_region.shape}, stitch shape: {stitch_binary.shape}")
                    print(f"      Overlap: {overlap_pixels}/{stitch_pixels} pixels = {overlap_ratio:.1f}%")
                
                # Only turn cyan if there's sufficient overlap (at least 20% of stitch on pattern)
                if overlap_ratio >= 20:
                    accepted_count += 1
                    # Add this stitch to the accumulated mask
                    self.completed_stitch_mask[pattern_full_y1:pattern_full_y2, 
                                               pattern_full_x1:pattern_full_x2] = np.maximum(
                        self.completed_stitch_mask[pattern_full_y1:pattern_full_y2, 
                                                   pattern_full_x1:pattern_full_x2],
                        stitch_binary * 255
                    )
                    
                    # ONLY track successful stitches to prevent re-detection
                    if 'box_cam' in stitch_data:
                        box_x, box_y, box_w, box_h = stitch_data['box_cam']
                        self.detected_stitch_positions.append((box_x, box_y, box_w, box_h))
                    
                    # Debug: Show individual stitch mapping more frequently
                    if self.frame_counter % 15 == 0:
                        print(f"  ✅ Stitch #{len(self.detected_stitch_positions)}: ({pattern_full_x1},{pattern_full_y1})-({pattern_full_x2},{pattern_full_y2}) size={final_w}x{final_h}, overlap={overlap_ratio:.0f}%")
                else:
                    rejected_overlap += 1
                    # Stitch doesn't match pattern line, skip it
                    if self.frame_counter % 15 == 0:
                        print(f"  ❌ Stitch REJECTED ({overlap_ratio:.0f}% overlap, need 20%)")
            else:
                # No pattern mask loaded, add all stitches (fallback)
                if processed_count == 1 and self.frame_counter % 15 == 0:
                    print(f"      ⚠️ NO PATTERN MASK - accepting all stitches")
                    
                accepted_count += 1
                self.completed_stitch_mask[pattern_full_y1:pattern_full_y2, 
                                           pattern_full_x1:pattern_full_x2] = np.maximum(
                    self.completed_stitch_mask[pattern_full_y1:pattern_full_y2, 
                                               pattern_full_x1:pattern_full_x2],
                    (stitch_final > 0).astype(np.uint8) * 255
                )
                
                # Track position for fallback case too
                if 'box_cam' in stitch_data:
                    box_x, box_y, box_w, box_h = stitch_data['box_cam']
                    self.detected_stitch_positions.append((box_x, box_y, box_w, box_h))
                
                if self.frame_counter % 15 == 0:
                    print(f"  🧵 Stitch #{len(self.detected_stitch_positions)}: ({pattern_full_x1},{pattern_full_y1})-({pattern_full_x2},{pattern_full_y2}) size={final_w}x{final_h} [No mask check]")
        
        # Debug summary
        if self.frame_counter % 15 == 0:
            print(f"📊 MAPPING SUMMARY: {len(detected_stitch_masks)} total → {processed_count} processed → {accepted_count} ACCEPTED ✅")
            print(f"   🎯 Now tracking {len(self.detected_stitch_positions)} successful stitches")
            if skipped_no_keys > 0:
                print(f"   ⚠️ Skipped {skipped_no_keys} (missing keys)")
            if skipped_no_intersection > 0:
                print(f"   ⚠️ Skipped {skipped_no_intersection} (no pattern intersection)")
            if skipped_invalid_bounds > 0:
                print(f"   ⚠️ Skipped {skipped_invalid_bounds} (invalid bounds)")
            if rejected_overlap > 0:
                print(f"   ❌ Rejected {rejected_overlap} (insufficient overlap <20%) - Can retry!")
        
        # NO LONGER storing all detected stitches - only successful ones are tracked above
        # This allows failed stitches to be re-detected when camera/fabric moves
                if self.frame_counter % 15 == 0:
                    print(f"    📌 Position stored")
        
        # Calculate accuracy for the current frame detections (for display purposes)
        if len(detected_stitch_masks) > 0:
            # Build a temporary combined mask for accuracy calculation only
            roi_combined_mask = None
            roi_bounds = None
            
            for stitch_data in detected_stitch_masks:
                if stitch_data.get('roi_bounds') is not None:
                    roi_bounds = stitch_data['roi_bounds']
                    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
                    
                    if roi_combined_mask is None:
                        roi_combined_mask = np.zeros((roi_y2 - roi_y1, roi_x2 - roi_x1), dtype=np.uint8)
                    
                    roi_combined_mask = np.maximum(roi_combined_mask, stitch_data['mask'])
            
            if roi_combined_mask is not None:
                # Get pattern mask in ROI coordinates
                pattern_crop = (pattern_alpha[0:actual_h, 0:actual_w] * 255).astype(np.uint8)
                pattern_mask_roi = (pattern_crop > 128).astype(np.uint8)
                
                # Ensure masks are same size
                if roi_combined_mask.shape != pattern_mask_roi.shape:
                    roi_combined_mask = cv2.resize(roi_combined_mask, (actual_w, actual_h))
                
                # Calculate accuracy (intersection over union) for current frame
                intersection = np.logical_and(roi_combined_mask > 0, pattern_mask_roi > 0)
                union = np.logical_or(roi_combined_mask > 0, pattern_mask_roi > 0)
                
                intersection_pixels = np.sum(intersection)
                union_pixels = np.sum(union)
                
                if union_pixels > 0:
                    accuracy = (intersection_pixels / union_pixels) * 100.0
                    self.current_accuracy = accuracy
                    self.total_score = int(self.current_accuracy)
        
        # Calculate progress from ACCUMULATED cyan coverage (completed_stitch_mask)
        # This matches what the user sees visually - using raw mask for accurate pattern following
        if self.completed_stitch_mask is not None and pattern_alpha is not None:
            # Get full pattern size
            pattern_h, pattern_w = pattern_alpha.shape[:2]
            
            # Resize completed mask to match pattern size
            completed_resized = cv2.resize(self.completed_stitch_mask, (pattern_w, pattern_h))
            completed_binary = (completed_resized > 128).astype(np.uint8)
            
            # Calculate what percentage of the pattern is now cyan (using raw mask)
            pattern_pixels = np.sum(pattern_alpha > 0.1)
            if pattern_pixels > 0:
                cyan_pixels = np.sum(np.logical_and(pattern_alpha > 0.1, completed_binary > 0))
                raw_progress = min(100.0, (cyan_pixels / pattern_pixels) * 100.0)
                
                # Debug: Show detection accuracy every 60 frames
                if self.frame_counter % 60 == 0 and cyan_pixels > 0:
                    print(f"📊 PROGRESS: {cyan_pixels}/{pattern_pixels} pixels cyan = {raw_progress:.1f}%")
                
                # Store raw progress for evaluation
                self.raw_progress = raw_progress
                
                # Direct progress calculation: 100% cyan = 100% progress
                # Only allow progress forward, never backwards (prevent flickering)
                if raw_progress > self.pattern_progress:
                    self.pattern_progress = raw_progress
    
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
        
        # Progress bar (smooth, reflects actual cyan coverage)
        bar_y = progress_y + 15
        bar_width = w - 40
        bar_height = 20
        
        # Draw bar background
        cv2.rectangle(frame, (content_x, bar_y), (content_x + bar_width, bar_y + bar_height), 
                     self.COLORS['dark_blue'], -1)
        
        # Draw filled progress bar (cyan)
        fill_width = int((self.pattern_progress / 100.0) * bar_width)
        if fill_width > 0:
            overlay_bar = frame.copy()
            cv2.rectangle(overlay_bar, (content_x, bar_y), 
                        (content_x + fill_width, bar_y + bar_height), 
                        self.segment_colors['completed'], -1)
            cv2.addWeighted(overlay_bar, 0.7, frame, 0.3, 0, frame)
        
        # Draw outer border
        cv2.rectangle(frame, (content_x, bar_y), (content_x + bar_width, bar_y + bar_height), 
                     self.COLORS['medium_blue'], 2)
        
        # Progress percentage text (centered on bar)
        progress_text = f"{self.pattern_progress:.1f}%"
        (prog_w, prog_h), _ = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_TRIPLEX, 0.6, 2)
        cv2.putText(frame, progress_text, (content_x + (bar_width - prog_w) // 2, bar_y + 16), 
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, self.COLORS['text_primary'], 2)
    
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
            self.reset_progress()  # Clear all progress when leaving level
            return 'back'
        
        # Check evaluate button (only if not evaluated yet)
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
                self.reset_progress()  # Use centralized reset function
                return None
        
        # Check next level button (only if evaluated and passed)
        if self.is_evaluated and self.level_completed:
            nb = self.next_level_button
            if nb['x'] <= x <= nb['x'] + nb['w'] and nb['y'] <= y <= nb['y'] + nb['h']:
                self.play_button_click_sound()
                self.reset_progress()  # Clear current level progress before moving to next
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
