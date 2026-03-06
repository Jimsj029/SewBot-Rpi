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
        
        # Segment tracking (divide pattern into 10 regions)
        self.num_segments = 10  # Total number of segments
        self.current_segment = 1  # Current active segment (1-10)
        self.completed_segments = set()  # Set of completed segment numbers {1, 2, 3, ...}
        self.segment_masks = {}  # Dictionary to store individual segment masks
        self.segment_colors = {
            'completed': (255, 255, 0),    # Cyan in BGR (completed sections)
            'current': (0, 255, 255),      # Yellow in BGR (current active segment)
            'upcoming': (100, 100, 100),   # Dim Gray (upcoming segments)
            'deviation': (0, 0, 255)       # Red in BGR (wrong segment detected)
        }
        
        # Out-of-segment detection and deviation tracking
        self.out_of_segment_warning = False
        self.warning_message = ""
        self.warning_flash_phase = 0
        self.deviation_detected = False
        self.deviation_segment = None  # Which wrong segment was detected
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
        self.last_detected_masks = []  # Cache last detection results
        
        # Process only top N detections for mask computation (huge performance boost)
        self.max_detections_to_process = 2  # Only compute masks for top 2 detections
        
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
        self.completed_segments = set()
        self.segment_masks = {}
        self.current_accuracy = 0.0
        self.total_score = 0
        self.out_of_segment_warning = False
        self.warning_message = ""
        self.deviation_detected = False
        self.deviation_segment = None
        self.current_combo = 0
        self.max_combo = 0
        self.stitches_detected = 0
        self.is_evaluated = False
        self.level_completed = False
        # Clear accumulated stitch mask for real-time coloring
        self.completed_stitch_mask = None
        print(f"🔄 Progress reset for Level {self.current_level}")
        print(f"📍 Starting at Segment {self.current_segment} (BOTTOM)")
        print(f"   Total segments: {self.num_segments}")
    
    def create_segment_masks(self, pattern_alpha):
        """Divide pattern into 10 vertical segments for tracking
        
        START FROM BOTTOM: Segment 1 is at the bottom, Segment 10 is at the top
        Segments are based on actual pattern content area, not full image height.
        
        Args:
            pattern_alpha: Pattern alpha mask (height x width)
            
        Returns:
            Dictionary mapping segment numbers (1-10) to their mask regions
        """
        if pattern_alpha is None:
            return {}
        
        height, width = pattern_alpha.shape[:2]
        
        # Find where actual pattern content exists
        pattern_rows_with_content = np.where(np.any(pattern_alpha > 0.1, axis=1))[0]
        
        if len(pattern_rows_with_content) == 0:
            print("⚠️ No pattern content found in alpha mask!")
            return {}
        
        # Get content bounds
        content_start_y = pattern_rows_with_content[0]
        content_end_y = pattern_rows_with_content[-1] + 1
        content_height = content_end_y - content_start_y
        
        segment_height = content_height // self.num_segments
        segment_masks = {}
        
        print(f"📐 Creating segments from content area: Y={content_start_y}-{content_end_y} (height={content_height})")
        
        # Create mask for each segment (REVERSED: bottom to top)
        # Segment 1 = bottom, Segment 10 = top
        for seg_num in range(1, self.num_segments + 1):
            # Reverse the Y coordinate calculation
            reversed_seg = self.num_segments - seg_num + 1
            
            # Calculate in content coordinates
            start_y_in_content = (reversed_seg - 1) * segment_height
            end_y_in_content = content_height if reversed_seg == self.num_segments else reversed_seg * segment_height
            
            # Convert to absolute pattern coordinates
            start_y = content_start_y + start_y_in_content
            end_y = content_start_y + end_y_in_content
            
            # Create segment mask
            seg_mask = np.zeros_like(pattern_alpha, dtype=np.uint8)
            seg_mask[start_y:end_y, :] = (pattern_alpha[start_y:end_y, :] * 255).astype(np.uint8)
            segment_masks[seg_num] = seg_mask
        
        return segment_masks
    
    def get_current_segment_roi(self, pattern_alpha):
        """Get ROI bounds for current active segment only
        
        REVERSED ORDER: Segment 1 is at bottom, Segment 10 is at top
        
        Args:
            pattern_alpha: Pattern alpha mask
            
        Returns:
            Tuple of (y_start, y_end) for current segment, or None
        """
        if pattern_alpha is None:
            return None
        
        height = pattern_alpha.shape[0]
        segment_height = height // self.num_segments
        
        # Reverse the segment mapping (1=bottom, 10=top)
        reversed_seg = self.num_segments - self.current_segment + 1
        start_y = (reversed_seg - 1) * segment_height
        end_y = height if reversed_seg == self.num_segments else reversed_seg * segment_height
        
        return (start_y, end_y)
    
    def create_realtime_pattern(self, overlay, alpha):
        """Create real-time pattern overlay showing ONLY current segment + accumulated stitches
        
        - Only shows current active segment pattern (yellow)
        - Overlays accumulated stitch mask in cyan (keeps completed work visible)
        - Hides completed and upcoming segments
        
        Args:
            overlay: Original pattern overlay (BGR)
            alpha: Pattern alpha mask
            
        Returns:
            Colored pattern overlay with current segment + stitch accumulation
        """
        if overlay is None or alpha is None:
            return None
        
        # Create segment masks if not already created
        if not self.segment_masks:
            self.segment_masks = self.create_segment_masks(alpha)
            # Debug: Show initial segment 1 position (bottom)
            seg1_roi = self.get_current_segment_roi(alpha)
            if seg1_roi:
                start_y, end_y = seg1_roi
                print(f"📐 Segment 1 position: Y={start_y}-{end_y} (pattern coords)")
        
        # Create colored overlay - start with transparent/black
        colored_overlay = np.zeros_like(overlay)
        height, width = overlay.shape[:2]
        pattern_pixels = alpha > 0.1
        
        # ONLY show current segment pattern (yellow)
        if self.current_segment in self.segment_masks:
            seg_mask = cv2.resize(self.segment_masks[self.current_segment], (width, height)) > 0
            seg_pattern_pixels = np.logical_and(pattern_pixels, seg_mask)
            colored_overlay[seg_pattern_pixels] = self.segment_colors['current']  # Yellow
        
        # Overlay accumulated stitch mask in cyan (shows completed work)
        if self.completed_stitch_mask is not None:
            # Resize completed stitch mask to match pattern overlay size
            completed_resized = cv2.resize(self.completed_stitch_mask, (width, height))
            stitch_pixels = completed_resized > 0
            
            # Show accumulated stitches in cyan (overrides pattern color)
            colored_overlay[stitch_pixels] = self.segment_colors['completed']  # Cyan
        
        # Highlight deviation if detected (flash red over wrong segment area)
        if self.deviation_detected and self.deviation_segment is not None:
            if self.deviation_segment in self.segment_masks:
                dev_seg_mask = cv2.resize(self.segment_masks[self.deviation_segment], (width, height)) > 0
                dev_pattern_pixels = np.logical_and(pattern_pixels, dev_seg_mask)
                # Flash red for wrong segment
                flash_intensity = 0.5 + 0.5 * abs(math.sin(self.warning_flash_phase))
                flash_color = tuple(int(c * flash_intensity) for c in self.segment_colors['deviation'])
                colored_overlay[dev_pattern_pixels] = flash_color
        
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
            pattern_overlay_original, pattern_alpha_original = self.load_blueprint(self.current_level)
            
            # STRETCH pattern to full camera height FIRST (before any calculations)
            if pattern_overlay_original is not None and pattern_alpha_original is not None:
                # Stretch to full camera height while keeping width
                overlay_h = self.camera_height  # 420
                overlay_w = self.uniform_width  # 200
                
                pattern_overlay = cv2.resize(pattern_overlay_original, (overlay_w, overlay_h))
                pattern_alpha = cv2.resize(pattern_alpha_original, (overlay_w, overlay_h))
                
                print(f"✅ Pattern stretched: {pattern_overlay_original.shape} → {pattern_overlay.shape}")
            else:
                pattern_overlay = pattern_overlay_original
                pattern_alpha = pattern_alpha_original
                print(f"❌ Pattern NOT loaded")
            
            
            # Run stitch detection using ONNX segmentation model with ROI optimization
            detected_stitch_masks = []
            detection_overlay = None
            roi_bounds = None  # Store ROI boundaries for visualization
            
            # Calculate ROI bounds for visualization (do this EVERY frame, not just detection frames)
            if pattern_overlay is not None and pattern_alpha is not None:
                # Pattern is already stretched to full camera height
                overlay_h, overlay_w = pattern_alpha.shape[:2]  # Should be 420x200
                
                # Position: edge-to-edge vertically (y=0), centered horizontally
                x_offset = (self.camera_width - overlay_w) // 2
                y_offset = 0  # Start from top edge
                
                # Find where actual pattern content exists (non-zero alpha pixels)
                pattern_rows_with_content = np.where(np.any(pattern_alpha > 0.1, axis=1))[0]
                
                if len(pattern_rows_with_content) > 0:
                    pattern_start_y = pattern_rows_with_content[0]
                    pattern_end_y = pattern_rows_with_content[-1]
                    pattern_content_height = pattern_end_y - pattern_start_y
                    
                    print(f"📋 Pattern content: Y={pattern_start_y}-{pattern_end_y} (height={pattern_content_height}px)")
                    
                    # Divide actual content into 10 segments
                    segment_height = pattern_content_height // self.num_segments
                    
                    # Calculate segment 1 position (bottom of actual content)
                    reversed_seg = self.num_segments - self.current_segment + 1
                    seg_start_in_content = (reversed_seg - 1) * segment_height
                    seg_end_in_content = pattern_content_height if reversed_seg == self.num_segments else reversed_seg * segment_height
                    
                    # Convert to absolute pattern coordinates
                    seg_start_y = pattern_start_y + seg_start_in_content
                    seg_end_y = pattern_start_y + seg_end_in_content
                    
                    padding_x = int(overlay_w * 0.2)
                    padding_y = int((seg_end_y - seg_start_y) * 0.15)
                    
                    roi_x1 = max(0, x_offset - padding_x)
                    roi_x2 = min(self.camera_width, x_offset + overlay_w + padding_x)
                    roi_y1 = max(0, y_offset + seg_start_y - padding_y)
                    roi_y2 = min(self.camera_height, y_offset + seg_end_y + padding_y)
                    
                    roi_bounds = (roi_x1, roi_y1, roi_x2, roi_y2)
                    print(f"✅ ROI: Y={roi_y1}-{roi_y2} | Seg{self.current_segment} content at Y={seg_start_y}-{seg_end_y}")
                else:
                    print(f"⚠️ No pattern content found!")
            
            
            # Frame skipping for performance optimization
            self.frame_counter += 1
            process_detection = (self.frame_counter % self.frame_skip == 0)
            
            if self.stitch_model is not None:
                try:
                    # Use cached results if skipping this frame
                    if not process_detection:
                        detected_stitch_masks = self.last_detected_masks
                    else:
                        # ========== SEGMENT-SPECIFIC ROI METHOD ==========
                        # Use segment-specific ROI if pattern is loaded and ROI bounds were calculated
                        if pattern_overlay is not None and pattern_alpha is not None and roi_bounds is not None:
                            # ROI bounds already calculated above
                            roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
                            
                            # Extract ROI from camera frame
                            roi_frame = cam_frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                            
                            # ONNX model requires 320x320 input (fixed size)
                            # Still faster than full frame since ROI is smaller area
                            roi_resized = cv2.resize(roi_frame, (320, 320))
                            
                            # Run inference ONLY on ROI (still faster due to smaller crop area)
                            results = self.stitch_model(
                                roi_resized, 
                                conf=self.confidence_threshold,
                                iou=self.iou_threshold,
                                imgsz=320,
                                verbose=False
                            )
                            
                            # Debug: Check if detections found
                            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                                num_detections = len(results[0].boxes)
                                print(f"🎯 Detections found: {num_detections}")
                                for i, box in enumerate(results[0].boxes):
                                    conf = float(box.conf[0].cpu().numpy())
                                    print(f"  Detection {i+1}: confidence={conf:.3f}")
                            
                            # Get detection visualization from YOLO
                            roi_detection_overlay = results[0].plot(boxes=False, labels=False)
                            
                            # Resize detection overlay back to ROI size
                            roi_detection_overlay = cv2.resize(roi_detection_overlay, (roi_x2 - roi_x1, roi_y2 - roi_y1))
                            
                            # Create full frame detection overlay and place ROI result
                            detection_overlay = cam_frame.copy()
                            detection_overlay[roi_y1:roi_y2, roi_x1:roi_x2] = roi_detection_overlay
                            
                            # Extract masks - keep at ROI resolution only (no full-frame expansion)
                            # OPTIMIZATION: Only process top 2 detections to reduce CPU load
                            for result in results:
                                if hasattr(result, 'masks') and result.masks is not None:
                                    masks = result.masks.data.cpu().numpy()
                                    boxes = result.boxes
                                    
                                    # ROI dimensions
                                    roi_h = roi_y2 - roi_y1
                                    roi_w = roi_x2 - roi_x1
                                    
                                    # Sort by confidence and take top N only
                                    confidences = boxes.conf.cpu().numpy()
                                    top_indices = np.argsort(confidences)[::-1][:self.max_detections_to_process]
                                    
                                    for idx in top_indices:
                                        if idx >= len(masks):  # Safety check
                                            break
                                        
                                        mask = masks[idx]
                                        box = boxes[idx]
                                        conf = float(box.conf[0].cpu().numpy())
                                        
                                        # Resize mask to ROI size only (not full screen)
                                        mask_resized = cv2.resize(mask, (roi_w, roi_h))
                                        mask_binary = (mask_resized > 0.5).astype(np.uint8)
                                        
                                        # Store ROI mask with its bounds (avoid full-frame expansion)
                                        detected_stitch_masks.append({
                                            'mask': mask_binary,  # ROI-sized mask only
                                            'confidence': conf,
                                            'roi_bounds': (roi_x1, roi_y1, roi_x2, roi_y2)  # ROI position
                                        })
                        else:
                            # FALLBACK: Use full frame detection if pattern not loaded
                            print("⚠ Pattern not loaded, using full frame detection")
                            results = self.stitch_model(
                                cam_frame, 
                                conf=self.confidence_threshold,
                                iou=self.iou_threshold,
                                imgsz=320,
                                verbose=False
                            )
                            
                            # Use YOLO's built-in plot method
                            detection_overlay = results[0].plot(boxes=False, labels=False)
                            
                            # Extract masks (full frame fallback - keep at camera resolution)
                            # OPTIMIZATION: Only process top 2 detections to reduce CPU load
                            for result in results:
                                if hasattr(result, 'masks') and result.masks is not None:
                                    masks = result.masks.data.cpu().numpy()
                                    boxes = result.boxes
                                    orig_shape = result.orig_shape
                                    
                                    # Sort by confidence and take top N only
                                    confidences = boxes.conf.cpu().numpy()
                                    top_indices = np.argsort(confidences)[::-1][:self.max_detections_to_process]
                                    
                                    for idx in top_indices:
                                        if idx >= len(masks):  # Safety check
                                            break
                                        
                                        mask = masks[idx]
                                        box = boxes[idx]
                                        conf = float(box.conf[0].cpu().numpy())
                                        mask_resized = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
                                        mask_binary = (mask_resized > 0.5).astype(np.uint8)
                                        
                                        detected_stitch_masks.append({
                                            'mask': mask_binary,
                                            'confidence': conf,
                                            'roi_bounds': None  # No ROI for full-frame detection
                                        })
                        
                        # Cache detection results for next frame
                        self.last_detected_masks = detected_stitch_masks
                    
                except Exception as e:
                    print(f"Stitch detection error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # First, overlay pattern mask (draw pattern first, detection on top)
            pattern_applied = False
            if pattern_overlay is not None and pattern_alpha is not None:
                # Pattern is already stretched to full camera height (done after loading)
                overlay_h, overlay_w = pattern_overlay.shape[:2]
                
                # Position: edge-to-edge vertically (y=0), centered horizontally
                x_offset = (self.camera_width - overlay_w) // 2
                y_offset = 0  # Start from top edge
                
                # Ensure we don't go out of bounds
                if x_offset >= 0:
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
                        
                        # DEBUG: Draw pattern overlay boundary (BRIGHT GREEN box edge-to-edge)
                        # Draw at edges of camera frame
                        cv2.rectangle(cam_frame, (x_offset, 0), 
                                     (x_offset + overlay_w - 1, self.camera_height - 1), 
                                     (0, 255, 0), 3)  # Bright green border (full height)
                        
                        # Add corner markers for pattern overlay (top and bottom)
                        corner_size = 20
                        # Top corners
                        cv2.line(cam_frame, (x_offset, 0), (x_offset + corner_size, 0), (0, 255, 0), 4)
                        cv2.line(cam_frame, (x_offset, 0), (x_offset, corner_size), (0, 255, 0), 4)
                        cv2.line(cam_frame, (x_offset + overlay_w, 0), (x_offset + overlay_w - corner_size, 0), (0, 255, 0), 4)
                        cv2.line(cam_frame, (x_offset + overlay_w, 0), (x_offset + overlay_w, corner_size), (0, 255, 0), 4)
                        # Bottom corners
                        cv2.line(cam_frame, (x_offset, self.camera_height-1), (x_offset + corner_size, self.camera_height-1), (0, 255, 0), 4)
                        cv2.line(cam_frame, (x_offset, self.camera_height-1), (x_offset, self.camera_height-1-corner_size), (0, 255, 0), 4)
                        cv2.line(cam_frame, (x_offset + overlay_w, self.camera_height-1), (x_offset + overlay_w - corner_size, self.camera_height-1), (0, 255, 0), 4)
                        cv2.line(cam_frame, (x_offset + overlay_w, self.camera_height-1), (x_offset + overlay_w, self.camera_height-1-corner_size), (0, 255, 0), 4)
                        
                        # Add label with background
                        label_text = "PATTERN (FULL HEIGHT)"
                        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(cam_frame, (x_offset + 3, 3), (x_offset + lw + 7, lh + 7), (0, 0, 0), -1)
                        cv2.putText(cam_frame, label_text, (x_offset + 5, lh + 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Calculate accuracy by comparing detected stitches with pattern
                        self.update_game_stats(detected_stitch_masks, pattern_alpha, 
                                             x_offset, y_offset, actual_w, actual_h)
            
            # If detection exists but pattern wasn't applied, use YOLO's visualization
            if not pattern_applied and detection_overlay is not None:
                cam_frame = detection_overlay
            
            # Draw ROI detection box overlay (shows where detection is happening)
            if roi_bounds is not None:
                roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
                # Draw BOLD ROI box in cyan
                cv2.rectangle(cam_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), 
                             (255, 255, 0), 3)  # Cyan, thicker line
                
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
                
                # Add label showing current segment
                label_text = f"SEGMENT {self.current_segment}/{self.num_segments}"
                font_scale = 0.5
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
                           self.COLORS['neon_blue'], thickness)
                
                # Add ROI position debug info (small text at bottom)
                debug_text = f"ROI: Y={roi_y1}-{roi_y2}"
                debug_scale = 0.3
                cv2.putText(cam_frame, debug_text, (roi_x1 + 5, roi_y2 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, debug_scale, 
                           self.COLORS['cyan'], 1)
            
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
                ("You need 75% or more progress", self.COLORS['text_secondary'], 0.7, 2),
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
        """Update game statistics based on detected stitches vs pattern (Segment-optimized)"""
        if len(detected_stitch_masks) == 0:
            self.out_of_segment_warning = False
            self.deviation_detected = False
            return
        
        # Initialize completed stitch mask if needed (use actual pattern dimensions)
        if self.completed_stitch_mask is None:
            pattern_h, pattern_w = pattern_alpha.shape[:2]
            self.completed_stitch_mask = np.zeros((pattern_h, pattern_w), dtype=np.uint8)
            print(f"📝 Initialized stitch mask: {pattern_w}x{pattern_h}")
        
        # Create segment masks if not already created
        if not self.segment_masks:
            self.segment_masks = self.create_segment_masks(pattern_alpha)
        
        # Create combined stitch mask in ROI coordinates only (avoid full-frame allocation)
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
                # Full-frame fallback
                if roi_combined_mask is None:
                    roi_combined_mask = np.zeros((actual_h, actual_w), dtype=np.uint8)
                    roi_bounds = (x_offset, y_offset, x_offset + actual_w, y_offset + actual_h)
                
                roi_mask = stitch_data['mask'][y_offset:y_offset+actual_h, x_offset:x_offset+actual_w]
                roi_combined_mask = np.maximum(roi_combined_mask, roi_mask)
        
        if roi_combined_mask is None:
            return
        
        # Dilate the detected stitch mask to be more forgiving
        kernel = np.ones((3, 3), np.uint8)
        roi_combined_mask = cv2.dilate(roi_combined_mask, kernel, iterations=1)
        
        # Get pattern mask in ROI coordinates
        pattern_crop = (pattern_alpha[0:actual_h, 0:actual_w] * 255).astype(np.uint8)
        pattern_mask_roi = (pattern_crop > 128).astype(np.uint8)
        
        # Ensure masks are same size
        if roi_combined_mask.shape != pattern_mask_roi.shape:
            roi_combined_mask = cv2.resize(roi_combined_mask, (actual_w, actual_h))
        
        # Convert detected stitches to pattern coordinates (PROPER MAPPING)
        # Use actual pattern dimensions (now stretched to camera height)
        pattern_h, pattern_w = pattern_alpha.shape[:2]
        stitches_pattern_coords = np.zeros((pattern_h, pattern_w), dtype=np.uint8)
        
        # Map ROI position back to pattern coordinates
        if roi_bounds is not None:
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
            
            # Calculate ROI position relative to pattern overlay
            # Pattern overlay is at (x_offset, y_offset) in camera frame
            pattern_x1 = max(0, roi_x1 - x_offset)
            pattern_y1 = max(0, roi_y1 - y_offset)
            pattern_x2 = min(pattern_w, roi_x2 - x_offset)
            pattern_y2 = min(pattern_h, roi_y2 - y_offset)
            
            # Resize ROI mask to match pattern region size
            pattern_region_w = pattern_x2 - pattern_x1
            pattern_region_h = pattern_y2 - pattern_y1
            
            if pattern_region_w > 0 and pattern_region_h > 0:
                roi_resized = cv2.resize(roi_combined_mask, (pattern_region_w, pattern_region_h))
                # Place at correct position in pattern coords
                stitches_pattern_coords[pattern_y1:pattern_y2, pattern_x1:pattern_x2] = roi_resized
                
                # Debug: Show where stitches are being mapped
                if np.sum(roi_resized) > 0:
                    print(f"🎯 Stitch mapping: ROI cam[{roi_y1}-{roi_y2}] → pattern[{pattern_y1}-{pattern_y2}]")
        else:
            # Fallback: use the ROI combined mask resized to pattern area
            stitches_pattern_coords = cv2.resize(roi_combined_mask, (pattern_w, pattern_h))
        
        # ========== SEGMENT-BASED DEVIATION DETECTION ==========
        # Check which segment(s) the current stitches intersect
        detected_segments = set()
        self.deviation_detected = False
        self.deviation_segment = None
        
        for seg_num in range(1, self.num_segments + 1):
            if seg_num not in self.segment_masks:
                continue
            
            seg_mask = self.segment_masks[seg_num]
            # Check if stitch overlaps with this segment
            overlap = np.logical_and(stitches_pattern_coords > 0, seg_mask > 128)
            if np.sum(overlap) > 0:
                detected_segments.add(seg_num)
        
        # Check for deviation (stitching in wrong segment)
        # STRICT MODE: Only allow stitching in current segment
        for seg in detected_segments:
            if seg == self.current_segment:
                # Stitching in current segment - perfect!
                continue
            else:
                # Stitching in ANY other segment - DEVIATION!
                self.deviation_detected = True
                self.deviation_segment = seg
                self.out_of_segment_warning = True
                
                if seg in self.completed_segments:
                    self.warning_message = f"⚠ Segment {seg} already done! Focus on Segment {self.current_segment}"
                elif seg < self.current_segment:
                    self.warning_message = f"⚠ Going backwards! Complete Segment {self.current_segment}"
                else:
                    self.warning_message = f"⚠ Skip detected! Complete Segment {self.current_segment} first"
                
                self.warning_flash_phase += 0.2
                print(f"⚠ DEVIATION: Stitching in segment {seg}, but current segment is {self.current_segment}")
        
        # ========== SEGMENT PROGRESS TRACKING ==========
        # Accumulate stitches into completed mask (only for current segment area)
        if np.sum(stitches_pattern_coords) > 0:
            self.completed_stitch_mask = np.maximum(self.completed_stitch_mask, 
                                                     (stitches_pattern_coords > 0).astype(np.uint8) * 255)
        
        # Check ONLY current segment for completion (strict one-at-a-time)
        segment_completion_threshold = 0.70  # 70% coverage to mark segment as complete
        
        seg_num = self.current_segment
        if seg_num in self.segment_masks and seg_num not in self.completed_segments:
            seg_mask = self.segment_masks[seg_num]
            seg_pattern_pixels = np.sum(seg_mask > 128)
            
            if seg_pattern_pixels > 0:
                # Count how many pixels of this segment are covered by stitches
                seg_covered_pixels = np.sum(np.logical_and(self.completed_stitch_mask > 0, seg_mask > 128))
                seg_coverage = seg_covered_pixels / seg_pattern_pixels
                
                print(f"📊 Segment {seg_num} coverage: {seg_coverage*100:.1f}%")
                
                # Mark segment as completed if threshold reached
                if seg_coverage >= segment_completion_threshold:
                    self.completed_segments.add(seg_num)
                    print(f"\n✅ ========== SEGMENT {seg_num} COMPLETED! ==========")
                    print(f"   Coverage: {seg_coverage*100:.1f}%")
                    
                    # Advance to next segment
                    if self.current_segment < self.num_segments:
                        old_segment = self.current_segment
                        self.current_segment += 1
                        print(f"➡ ROI MOVING: Segment {old_segment} → Segment {self.current_segment}")
                        
                        # Calculate new ROI position for debugging
                        new_seg_roi = self.get_current_segment_roi(pattern_alpha)
                        if new_seg_roi:
                            new_start_y, new_end_y = new_seg_roi
                            print(f"   New ROI pattern coords: Y={new_start_y}-{new_end_y}")
                        
                        self.out_of_segment_warning = False
                        self.deviation_detected = False
                    else:
                        print(f"\n🎉 ========== ALL SEGMENTS COMPLETED! ==========")
        
        # Calculate overall pattern progress (0-100%)
        total_pattern_pixels = np.sum(pattern_alpha > 0.1)
        if total_pattern_pixels > 0:
            total_covered_pixels = np.sum(np.logical_and(
                cv2.resize(self.completed_stitch_mask, (pattern_alpha.shape[1], pattern_alpha.shape[0])) > 0,
                pattern_alpha > 0.1
            ))
            self.raw_progress = min(100.0, (total_covered_pixels / total_pattern_pixels) * 100.0)
            self.pattern_progress = (len(self.completed_segments) / self.num_segments) * 100.0
        
        # Calculate accuracy (IoU)
        intersection = np.logical_and(roi_combined_mask > 0, pattern_mask_roi > 0)
        union = np.logical_or(roi_combined_mask > 0, pattern_mask_roi > 0)
        
        intersection_pixels = np.sum(intersection)
        union_pixels = np.sum(union)
        
        if union_pixels > 0:
            accuracy = (intersection_pixels / union_pixels) * 100.0
            self.current_accuracy = accuracy
            self.total_score = int(self.pattern_progress)  # Score based on segment completion
            
            # Update combo system
            if accuracy > 70 and not self.deviation_detected:
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
        
        # Progress bar background (divided into 10 segments)
        bar_y = progress_y + 15
        bar_width = w - 40
        bar_height = 25
        segment_width = (bar_width - 4) // self.num_segments
        
        # Draw 10 segment boxes (2 rows of 5)
        rows = 2
        cols = self.num_segments // rows
        segment_height = (bar_height - 4) // rows
        
        for seg in range(1, self.num_segments + 1):
            row = (seg - 1) // cols
            col = (seg - 1) % cols
            seg_x = content_x + 2 + col * segment_width
            seg_y = bar_y + 2 + row * segment_height
            
            # Determine segment color based on completion status
            if seg in self.completed_segments:
                # Completed segment - cyan
                seg_color = self.segment_colors['completed']
            elif seg == self.current_segment:
                # Current active segment - yellow with pulse
                pulse = 0.5 + 0.5 * abs(math.sin(self.glow_phase * 2))
                seg_color = tuple(int(c * pulse) for c in self.segment_colors['current'])
            elif seg < self.current_segment and seg not in self.completed_segments:
                # Skipped segment - dim red (shouldn't happen often)
                seg_color = (50, 50, 150)
            else:
                # Upcoming segment - gray
                seg_color = self.segment_colors['upcoming']
            
            # Draw segment
            overlay_bar = frame.copy()
            cv2.rectangle(overlay_bar, (seg_x, seg_y), 
                        (seg_x + segment_width - 2, seg_y + segment_height - 2), 
                        seg_color, -1)
            cv2.addWeighted(overlay_bar, 0.7, frame, 0.3, 0, frame)
            
            # Draw segment border
            cv2.rectangle(frame, (seg_x, seg_y), 
                        (seg_x + segment_width - 2, seg_y + segment_height - 2), 
                        self.COLORS['medium_blue'], 1)
            
            # Draw segment number (tiny text)
            seg_num_text = str(seg)
            seg_font_scale = 0.25
            seg_thickness = 1
            (seg_text_w, seg_text_h), _ = cv2.getTextSize(seg_num_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                           seg_font_scale, seg_thickness)
            seg_text_x = seg_x + (segment_width - seg_text_w) // 2 - 1
            seg_text_y = seg_y + (segment_height + seg_text_h) // 2
            cv2.putText(frame, seg_num_text, (seg_text_x, seg_text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, seg_font_scale, (200, 200, 200), seg_thickness)
        
        # Draw outer border
        cv2.rectangle(frame, (content_x, bar_y), (content_x + bar_width, bar_y + bar_height), 
                     self.COLORS['medium_blue'], 2)
        
        # Progress percentage text (below bar)
        progress_text = f"{self.pattern_progress:.0f}%"
        (prog_w, prog_h), _ = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_TRIPLEX, 0.6, 2)
        cv2.putText(frame, progress_text, (content_x + (bar_width - prog_w) // 2, bar_y + bar_height + 18), 
                   cv2.FONT_HERSHEY_TRIPLEX, 0.6, self.COLORS['text_primary'], 2)
        
        # Segment indicator text
        segment_text_y = bar_y + bar_height + 38
        segment_label = f"SEGMENT {self.current_segment}/{self.num_segments}"
        (seg_label_w, _), _ = cv2.getTextSize(segment_label, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 2)
        cv2.putText(frame, segment_label, (content_x + (bar_width - seg_label_w) // 2, segment_text_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.COLORS['text_secondary'], 2)
        
        # Completed segments count
        completed_text_y = segment_text_y + 20
        completed_label = f"✓ Completed: {len(self.completed_segments)}/{self.num_segments}"
        cv2.putText(frame, completed_label, (content_x, completed_text_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, 0.45, self.COLORS['text_secondary'], 1)
    
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
        req_text = "(Need 75% to pass)"
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
                # Clear accumulated stitch mask (remove cyan lines)
                self.completed_stitch_mask = None
                # Clear completed segments progress
                self.completed_segments.clear()
                self.deviation_segment = None
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
        
        # Check if level is completed (75%+ raw progress)
        if self.raw_progress >= 75:
            self.level_completed = True
            print(f"✅ Level {self.current_level} completed! Progress: {self.raw_progress:.1f}%")
        else:
            self.level_completed = False
            print(f"📊 Evaluation: {self.raw_progress:.1f}% progress. Need 75%+ to complete.")
