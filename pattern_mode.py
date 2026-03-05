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
        self.stitches_detected = 0
        
        # Load models from models directory
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Detection settings
        self.confidence_threshold = 0.3  # Lowered for INT8 model (produces lower confidence scores)
        self.iou_threshold = 0.6  # Intersection over Union threshold
        
        # Performance optimization: Frame skipping for detection
        # Process detection every N frames to reduce CPU load
        self.frame_skip = 3  # Run YOLO every 3rd frame (~3x FPS boost)
        self.frame_counter = 0
        self.last_detected_masks = []  # Cache last detection results
        self.last_detected_boxes = []  # Cache bounding boxes for tracking
        self.detection_mode = "DETECTING"  # "DETECTING" or "TRACKING"
        
        # Fixed ROI configuration (adjustable)
        self.roi_width = 440
        self.roi_height = 380
        
        # Process only top N detections for mask computation
        self.max_detections_to_process = 12
        
        # Load stitch detection model (ONNX model)
        try:
            # Use ONNX model (best.onnx)
            stitch_model_path = os.path.join(models_dir, 'best_int8.onnx')
            
            if os.path.exists(stitch_model_path):
                print(f"Loading stitch detection model: {stitch_model_path}")
                self.stitch_model = YOLO(stitch_model_path, task='segment')
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
                test_img = np.zeros((320, 320, 3), dtype=np.uint8)
                test_result = self.stitch_model(test_img, conf=self.confidence_threshold, verbose=False)
                print("  ✓ Model test successful!")
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
            
            # ========== STEP 1: ROI CROPPING (Pattern-based dynamic ROI) ==========
            # Use pattern dimensions to set ROI bounds
            pattern_alpha_roi = None  # Will be used to filter detections
            if pattern_overlay is not None and pattern_alpha is not None:
                overlay_h, overlay_w = pattern_overlay.shape[:2]
                # Center pattern on camera frame
                pattern_x_offset = (self.camera_width - overlay_w) // 2
                pattern_y_offset = (self.camera_height - overlay_h) // 2
                
                # ROI matches pattern boundaries exactly
                roi_x1 = max(0, pattern_x_offset)
                roi_y1 = max(0, pattern_y_offset)
                roi_x2 = min(self.camera_width, pattern_x_offset + overlay_w)
                roi_y2 = min(self.camera_height, pattern_y_offset + overlay_h)
                
                # Extract ROI from camera frame
                roi_frame = cam_frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                
                # Extract pattern alpha for this ROI (for filtering detections later)
                actual_h = roi_y2 - roi_y1
                actual_w = roi_x2 - roi_x1
                pattern_alpha_roi = pattern_alpha[0:actual_h, 0:actual_w]
            else:
                # Fallback to center ROI if no pattern
                roi_x1 = max(0, (self.camera_width - self.roi_width) // 2)
                roi_y1 = max(0, (self.camera_height - self.roi_height) // 2)
                roi_x2 = min(self.camera_width, roi_x1 + self.roi_width)
                roi_y2 = min(self.camera_height, roi_y1 + self.roi_height)
                roi_frame = cam_frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            
            roi_bounds = (roi_x1, roi_y1, roi_x2, roi_y2)
            
            # ========== STEP 2: SMART DETECTION (Every 3 frames) ==========
            detected_stitch_masks = []
            detection_overlay = None
            
            self.frame_counter += 1
            process_detection = (self.frame_counter % self.frame_skip == 0)
            
            if self.stitch_model is not None:
                try:
                    if not process_detection:
                        # TRACKING MODE: Reuse previous boxes
                        self.detection_mode = "TRACKING"
                        detected_stitch_masks = self.last_detected_masks
                    else:
                        # DETECTING MODE: Run YOLO inference
                        self.detection_mode = "DETECTING"
                        
                        # Resize ROI to 320x320 for YOLO
                        roi_resized = cv2.resize(roi_frame, (320, 320))
                        
                        # Run YOLO detection on ROI
                        results = self.stitch_model(
                            roi_resized,
                            conf=self.confidence_threshold,
                            iou=self.iou_threshold,
                            imgsz=320,
                            verbose=False
                        )
                        
                        # ========== STEP 3: MASK GENERATION (YOLO Segmentation) ==========
                        detected_boxes = []
                        
                        # Extract masks from YOLO segmentation results
                        for result in results:
                            if hasattr(result, 'masks') and result.masks is not None:
                                masks = result.masks.data.cpu().numpy()
                                boxes = result.boxes
                                roi_h = roi_y2 - roi_y1
                                roi_w = roi_x2 - roi_x1
                                
                                # Sort by confidence and take top N
                                confidences = boxes.conf.cpu().numpy()
                                top_indices = np.argsort(confidences)[::-1][:self.max_detections_to_process]
                                
                                for idx in top_indices:
                                    if idx >= len(masks):  # Safety check
                                        break
                                    
                                    mask = masks[idx]
                                    box = boxes[idx]
                                    conf = float(box.conf[0].cpu().numpy())
                                    
                                    # Resize mask to ROI size
                                    mask_resized = cv2.resize(mask, (roi_w, roi_h))
                                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                                    
                                    # Filter: only keep if mask overlaps with pattern
                                    if pattern_alpha_roi is not None:
                                        # Check if mask overlaps with pattern alpha
                                        pattern_mask_binary = (pattern_alpha_roi > 0.1).astype(np.uint8)
                                        overlap = np.logical_and(mask_binary > 0, pattern_mask_binary > 0)
                                        overlap_ratio = np.sum(overlap) / max(1, np.sum(mask_binary > 0))
                                        
                                        # Skip if less than 30% of the detection overlaps with pattern
                                        if overlap_ratio < 0.3:
                                            continue
                                    
                                    # Get box coordinates for tracking
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    scale_x = roi_w / 320.0
                                    scale_y = roi_h / 320.0
                                    x1 = int(x1 * scale_x)
                                    y1 = int(y1 * scale_y)
                                    x2 = int(x2 * scale_x)
                                    y2 = int(y2 * scale_y)
                                    
                                    detected_stitch_masks.append({
                                        'mask': mask_binary,
                                        'confidence': conf,
                                        'roi_bounds': (roi_x1, roi_y1, roi_x2, roi_y2),
                                        'box': (x1, y1, x2, y2)  # Box within ROI
                                    })
                                    
                                    detected_boxes.append((x1, y1, x2, y2, conf))
                        
                        # Cache results for tracking
                        self.last_detected_masks = detected_stitch_masks
                        self.last_detected_boxes = detected_boxes
                    
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
                        
                        # Calculate accuracy by comparing detected stitches with pattern
                        self.update_game_stats(detected_stitch_masks, pattern_alpha, 
                                             x_offset, y_offset, actual_w, actual_h)
            
            # If detection exists but pattern wasn't applied, use YOLO's visualization
            if not pattern_applied and detection_overlay is not None:
                cam_frame = detection_overlay
            
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
    
    def draw_detection_status(self, frame):
        """Draw detection status indicator (DETECTING or TRACKING)"""
        # Position at top-right corner of camera view
        status_x = self.camera_x + self.camera_width - 150
        status_y = self.camera_y + 25
        
        # Choose color based on mode
        if self.detection_mode == "DETECTING":
            status_color = (0, 255, 0)  # Green - actively detecting
            bg_color = (0, 100, 0)
        else:  # TRACKING
            status_color = (0, 255, 255)  # Yellow - using cached results
            bg_color = (0, 100, 100)
        
        # Draw background box
        (text_w, text_h), _ = cv2.getTextSize(self.detection_mode, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 2)
        padding = 8
        cv2.rectangle(frame, 
                     (status_x - padding, status_y - text_h - padding), 
                     (status_x + text_w + padding, status_y + padding), 
                     bg_color, -1)
        cv2.rectangle(frame, 
                     (status_x - padding, status_y - text_h - padding), 
                     (status_x + text_w + padding, status_y + padding), 
                     status_color, 2)
        
        # Draw status text
        cv2.putText(frame, self.detection_mode, (status_x, status_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, 0.5, status_color, 2)
    
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
        
        # Get pattern dimensions
        overlay_h, overlay_w = pattern_alpha.shape[:2]
        
        # Calculate proper coordinate alignment between ROI and pattern
        # Pattern bounds in camera coordinates
        pattern_x1, pattern_y1 = x_offset, y_offset
        pattern_x2 = x_offset + overlay_w
        pattern_y2 = y_offset + overlay_h
        
        # ROI bounds in camera coordinates (extract from roi_bounds if available)
        if roi_bounds is not None and len(roi_bounds) == 4:
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
        else:
            # Fallback: calculate ROI position
            roi_x1 = (self.camera_width - self.roi_width) // 2
            roi_y1 = (self.camera_height - self.roi_height) // 2
            roi_x2 = roi_x1 + self.roi_width
            roi_y2 = roi_y1 + self.roi_height
        
        # Calculate overlap region in camera coordinates
        overlap_x1 = max(roi_x1, pattern_x1)
        overlap_y1 = max(roi_y1, pattern_y1)
        overlap_x2 = min(roi_x2, pattern_x2)
        overlap_y2 = min(roi_y2, pattern_y2)
        
        # Check if there's actual overlap
        if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
            self.out_of_segment_warning = False
            return
        
        # Convert overlap to ROI-relative coordinates (for extracting from stitch mask)
        roi_overlap_x1 = overlap_x1 - roi_x1
        roi_overlap_y1 = overlap_y1 - roi_y1
        roi_overlap_x2 = overlap_x2 - roi_x1
        roi_overlap_y2 = overlap_y2 - roi_y1
        
        # Convert overlap to pattern-relative coordinates (for extracting from pattern)
        pat_overlap_x1 = overlap_x1 - pattern_x1
        pat_overlap_y1 = overlap_y1 - pattern_y1
        pat_overlap_x2 = overlap_x2 - pattern_x1
        pat_overlap_y2 = overlap_y2 - pattern_y1
        
        # Extract the overlapping regions
        roi_overlap_mask = roi_combined_mask[roi_overlap_y1:roi_overlap_y2, roi_overlap_x1:roi_overlap_x2]
        pattern_crop = (pattern_alpha[pat_overlap_y1:pat_overlap_y2, pat_overlap_x1:pat_overlap_x2] * 255).astype(np.uint8)
        pattern_mask_roi = (pattern_crop > 128).astype(np.uint8)
        
        # Ensure masks are same size
        if roi_overlap_mask.shape != pattern_mask_roi.shape:
            roi_overlap_mask = cv2.resize(roi_overlap_mask, (pattern_mask_roi.shape[1], pattern_mask_roi.shape[0]))
        
        # Use the aligned overlap masks for all subsequent calculations
        roi_combined_mask = roi_overlap_mask
        
        # Extract stitches that are within the pattern area (all in ROI coordinates)
        stitches_in_pattern = np.logical_and(roi_combined_mask > 0, pattern_mask_roi > 0)
        
        # Convert detected stitches in pattern to pattern coordinates and accumulate
        if np.sum(stitches_in_pattern) > 0:
            # Resize to pattern dimensions (from ROI to pattern coords)
            stitches_pattern_coords = cv2.resize(roi_combined_mask, (self.uniform_width, self.uniform_height))
            
            # Accumulate into completed mask (using maximum to not lose previous stitches)
            self.completed_stitch_mask = np.maximum(self.completed_stitch_mask, 
                                                     (stitches_pattern_coords > 0).astype(np.uint8) * 255)
        
        # Check if stitches are going outside the pattern boundary
        total_stitch_pixels = np.sum(roi_combined_mask > 0)
        stitches_outside_pattern = np.sum(np.logical_and(roi_combined_mask > 0, pattern_mask_roi == 0))
        
        if total_stitch_pixels > 0:
            outside_percentage = (stitches_outside_pattern / total_stitch_pixels) * 100.0
            
            # Trigger warning if more than 40% of stitches are outside pattern
            if outside_percentage > 40:
                self.out_of_segment_warning = True
                self.warning_message = "⚠ STITCHING OUTSIDE PATTERN!"
                self.warning_flash_phase += 0.2
            else:
                self.out_of_segment_warning = False
        else:
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
        
        # Draw outer border
        cv2.rectangle(frame, (content_x, bar_y), (content_x + bar_width, bar_y + bar_height), 
                     self.COLORS['medium_blue'], 2)
        
        # Fill progress bar based on current progress
        if self.pattern_progress > 0:
            fill_width = int((bar_width - 4) * (self.pattern_progress / 100.0))
            if fill_width > 0:
                # Draw progress fill with gradient effect
                bar_overlay = frame.copy()
                cv2.rectangle(bar_overlay, (content_x + 2, bar_y + 2), 
                            (content_x + 2 + fill_width, bar_y + bar_height - 2), 
                            self.COLORS['glow_cyan'], -1)
                cv2.addWeighted(bar_overlay, 0.7, frame, 0.3, 0, frame)
        
        # Progress percentage text
        progress_text = f"{self.pattern_progress:.0f}%"
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
            self.reset_progress()  # Reset all progress when leaving level
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
