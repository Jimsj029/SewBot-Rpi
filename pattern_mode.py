import cv2
import numpy as np
import math
import os
import time
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
        self.uniform_width = 200
        self.uniform_height = 300
        self.alpha_blend = 0.9
        self.glow_phase = 0
        
        # FPS tracking
        self.fps_values = []
        self.fps_max_samples = 30  # Average over 30 frames
        self.last_frame_time = time.time()
        self.current_fps = 0.0
        
        # Performance optimization: frame skipping for detection
        self.frame_skip = 2  # Run detection every 2 frames (can be adjusted)
        self.frame_counter = 0
        self.last_detection_overlay = None  # Cache last detection result
        self.last_detected_masks = []  # Cache last detected masks
        
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
        
        # Game tracking variables
        self.current_accuracy = 0.0
        self.total_score = 0
        self.pattern_progress = 0.0  # 0-100%
        self.session_start_time = None
        
        # Load models from models directory
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Detection settings
        self.confidence_threshold = 0.35  # Optimized for stitch line detection
        self.iou_threshold = 0.6  # Intersection over Union threshold
        
        # Load stitch detection model (ONNX model)
        try:
            # Use ONNX model (best.onnx)
            stitch_model_path = os.path.join(models_dir, 'best.onnx')
            
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
                test_img = np.zeros((640, 640, 3), dtype=np.uint8)
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
    
    def update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        
        if delta_time > 0:
            fps = 1.0 / delta_time
            self.fps_values.append(fps)
            
            # Keep only recent samples
            if len(self.fps_values) > self.fps_max_samples:
                self.fps_values.pop(0)
            
            # Calculate moving average
            self.current_fps = sum(self.fps_values) / len(self.fps_values)
        
        self.last_frame_time = current_time
    
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
        # Update FPS
        self.update_fps()
        
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
        (level_w, level_h), _ = cv2.getTextSize(level_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        level_x = (self.width - level_w) // 2
        level_y = self.level_display_y
        
        # Draw with glow effect
        cv2.putText(frame, level_text, (level_x, level_y), cv2.FONT_HERSHEY_DUPLEX, 
                   font_scale, self.COLORS['glow_cyan'], thickness + 1)
        cv2.putText(frame, level_text, (level_x, level_y), cv2.FONT_HERSHEY_DUPLEX, 
                   font_scale, self.COLORS['bright_blue'], thickness)
        
        # Draw difficulty text below
        diff_font_scale = 0.6
        diff_thickness = 1
        (diff_w, diff_h), _ = cv2.getTextSize(difficulty_text, cv2.FONT_HERSHEY_SIMPLEX, diff_font_scale, diff_thickness)
        diff_x = (self.width - diff_w) // 2
        diff_y = level_y + 30
        cv2.putText(frame, difficulty_text, (diff_x, diff_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   diff_font_scale, self.COLORS['text_secondary'], diff_thickness)
        
        # Draw back button (top left)
        self.draw_back_button(frame)
        
        # Draw camera feed
        self.draw_camera_feed(frame, camera_frame)
        
        # Draw score/stats panel
        self.draw_score_panel(frame)
    
    def draw_back_button(self, frame):
        """Draw back button in top left"""
        bb = self.back_button
        pulse = 0.4 + 0.3 * abs(math.sin(self.glow_phase))
        self.draw_glow_rect(frame, bb['x'], bb['y'], bb['w'], bb['h'], self.COLORS['medium_blue'], pulse)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (bb['x'] + 2, bb['y'] + 2), (bb['x'] + bb['w'] - 2, bb['y'] + bb['h'] - 2), self.COLORS['button_normal'], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        text, font_scale, thickness = "< BACK", 0.6, 2
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = bb['x'] + (bb['w'] - text_w) // 2
        text_y = bb['y'] + (bb['h'] + text_h) // 2
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.COLORS['text_primary'], thickness)
    
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
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS['text_primary'], 2)
        else:
            cam_frame = cv2.resize(camera_frame, (self.camera_width, self.camera_height))
            
            # Load pattern mask for comparison
            pattern_overlay, pattern_alpha = self.load_blueprint(self.current_level)
            
            # Run stitch detection using ONNX segmentation model with ROI optimization
            # Use frame skipping to improve FPS
            detected_stitch_masks = []
            detection_overlay = None
            roi_bounds = None  # Store ROI boundaries for visualization
            
            # Frame skipping optimization: only run detection every N frames
            self.frame_counter += 1
            should_run_detection = (self.frame_counter % self.frame_skip == 0)
            
            if self.stitch_model is not None and should_run_detection:
                try:
                    # ========== ROI METHOD FOR FPS BOOST ==========
                    # Use ROI only if pattern is loaded (otherwise use full frame)
                    if pattern_overlay is not None and pattern_alpha is not None:
                        # Calculate ROI based on pattern position (only detect where pattern is)
                        overlay_h, overlay_w = pattern_overlay.shape[:2]
                        x_offset = (self.camera_width - overlay_w) // 2
                        y_offset = (self.camera_height - overlay_h) // 2
                        
                        # Reduced width padding for better FPS, keep height padding normal
                        padding_x = int(overlay_w * 0.05)  # Minimal horizontal padding for FPS boost
                        padding_y = int(overlay_h * 0.2)   # Normal vertical padding
                        
                        # Calculate ROI boundaries (ensure within frame bounds)
                        roi_x1 = max(0, x_offset - padding_x)
                        roi_y1 = max(0, y_offset - padding_y)
                        roi_x2 = min(self.camera_width, x_offset + overlay_w + padding_x)
                        roi_y2 = min(self.camera_height, y_offset + overlay_h + padding_y)
                        
                        # Store ROI bounds for visualization
                        roi_bounds = (roi_x1, roi_y1, roi_x2, roi_y2)
                        
                        # Extract ROI from camera frame
                        roi_frame = cam_frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                        
                        # ONNX model requires 640x640 input (fixed size)
                        # Still faster than full frame since ROI is smaller area
                        roi_resized = cv2.resize(roi_frame, (640, 640))
                        
                        # Run inference ONLY on ROI (still faster due to smaller crop area)
                        results = self.stitch_model(
                            roi_resized, 
                            conf=self.confidence_threshold,
                            iou=self.iou_threshold,
                            imgsz=640,
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
                        
                        # Extract masks for accuracy calculation and map back to full frame
                        for result in results:
                            if hasattr(result, 'masks') and result.masks is not None:
                                masks = result.masks.data.cpu().numpy()
                                boxes = result.boxes
                                
                                # ROI dimensions
                                roi_h = roi_y2 - roi_y1
                                roi_w = roi_x2 - roi_x1
                                
                                for i, (mask, box) in enumerate(zip(masks, boxes)):
                                    conf = float(box.conf[0].cpu().numpy())
                                    
                                    # Resize mask to ROI size
                                    mask_resized = cv2.resize(mask, (roi_w, roi_h))
                                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                                    
                                    # Create full frame mask and place ROI mask at correct position
                                    full_frame_mask = np.zeros((self.camera_height, self.camera_width), dtype=np.uint8)
                                    full_frame_mask[roi_y1:roi_y2, roi_x1:roi_x2] = mask_binary
                                    
                                    # Store the full frame mask for accuracy calculation
                                    detected_stitch_masks.append({
                                        'mask': full_frame_mask,
                                        'confidence': conf
                                    })
                        
                        # Cache detection results for frame skipping
                        self.last_detection_overlay = detection_overlay.copy()
                        self.last_detected_masks = detected_stitch_masks.copy()
                    else:
                        # FALLBACK: Use full frame detection if pattern not loaded
                        print("⚠ Pattern not loaded, using full frame detection")
                        results = self.stitch_model(
                            cam_frame, 
                            conf=self.confidence_threshold,
                            iou=self.iou_threshold,
                            imgsz=640,
                            verbose=False
                        )
                        
                        # Use YOLO's built-in plot method
                        detection_overlay = results[0].plot(boxes=False, labels=False)
                        
                        # Extract masks
                        for result in results:
                            if hasattr(result, 'masks') and result.masks is not None:
                                masks = result.masks.data.cpu().numpy()
                                boxes = result.boxes
                                orig_shape = result.orig_shape
                                
                                for i, (mask, box) in enumerate(zip(masks, boxes)):
                                    conf = float(box.conf[0].cpu().numpy())
                                    mask_resized = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
                                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                                    
                                    detected_stitch_masks.append({
                                        'mask': mask_binary,
                                        'confidence': conf
                                    })
                        
                        # Cache detection results for frame skipping
                        self.last_detection_overlay = detection_overlay.copy() if detection_overlay is not None else None
                        self.last_detected_masks = detected_stitch_masks.copy()
                    
                except Exception as e:
                    print(f"Stitch detection error: {e}")
                    import traceback
                    traceback.print_exc()
            elif self.stitch_model is not None and not should_run_detection:
                # Use cached detection results for skipped frames
                detection_overlay = self.last_detection_overlay.copy() if self.last_detection_overlay is not None else None
                detected_stitch_masks = self.last_detected_masks.copy()
            
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
                        
                        # Apply pattern overlay on top
                        roi = cam_frame[y_offset:overlay_end_y, x_offset:overlay_end_x]
                        pattern_crop = pattern_overlay[0:actual_h, 0:actual_w]
                        alpha_crop = pattern_alpha[0:actual_h, 0:actual_w]
                        
                        for c in range(3):
                            roi[:, :, c] = (alpha_crop * pattern_crop[:, :, c] * 0.5 + 
                                          (1 - alpha_crop * 0.5) * roi[:, :, c])
                        cam_frame[y_offset:overlay_end_y, x_offset:overlay_end_x] = roi
                        pattern_applied = True
                        
                        # Calculate accuracy by comparing detected stitches with pattern
                        self.update_game_stats(detected_stitch_masks, pattern_alpha, 
                                             x_offset, y_offset, actual_w, actual_h)
            
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
                
                # Add label "DETECTION ZONE"
                label_text = "DETECTION ZONE"
                font_scale = 0.4
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                       font_scale, thickness)
                label_x = roi_x1 + 5
                label_y = roi_y1 - 5 if roi_y1 > 20 else roi_y1 + 15
                
                # Draw text background
                cv2.rectangle(cam_frame, (label_x - 2, label_y - text_h - 2), 
                            (label_x + text_w + 2, label_y + 2), 
                            self.COLORS['dark_blue'], -1)
                # Draw text
                cv2.putText(cam_frame, label_text, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                           self.COLORS['neon_blue'], thickness)
            
            frame[self.camera_y:self.camera_y+self.camera_height, 
                  self.camera_x:self.camera_x+self.camera_width] = cam_frame
            
            # Draw FPS counter on camera feed (top-right corner)
            fps_text = f"FPS: {self.current_fps:.1f}"
            fps_font_scale = 0.6
            fps_thickness = 2
            (fps_w, fps_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                fps_font_scale, fps_thickness)
            
            # Position top-right of camera feed
            fps_x = self.camera_x + self.camera_width - fps_w - 10
            fps_y = self.camera_y + 25
            
            # Draw background for better readability
            cv2.rectangle(frame, (fps_x - 5, fps_y - fps_h - 5), 
                         (fps_x + fps_w + 5, fps_y + 5), 
                         self.COLORS['dark_blue'], -1)
            
            # Color code FPS (green if good, yellow if medium, red if low)
            if self.current_fps >= 25:
                fps_color = (0, 255, 0)  # Green
            elif self.current_fps >= 15:
                fps_color = (0, 255, 255)  # Yellow
            else:
                fps_color = (0, 100, 255)  # Orange/Red
            
            cv2.putText(frame, fps_text, (fps_x, fps_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       fps_font_scale, fps_color, fps_thickness)
        
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
    
    def update_game_stats(self, detected_stitch_masks, pattern_alpha, x_offset, y_offset, actual_w, actual_h):
        """Update game statistics based on detected stitches vs pattern"""
        if len(detected_stitch_masks) == 0:
            return
        
        # Create combined stitch mask
        combined_stitch_mask = np.zeros((self.camera_height, self.camera_width), dtype=np.uint8)
        for stitch_data in detected_stitch_masks:
            combined_stitch_mask = np.maximum(combined_stitch_mask, stitch_data['mask'])
        
        # Get pattern mask in the ROI area
        pattern_mask_roi = np.zeros((self.camera_height, self.camera_width), dtype=np.uint8)
        pattern_crop = (pattern_alpha[0:actual_h, 0:actual_w] * 255).astype(np.uint8)
        pattern_mask_roi[y_offset:y_offset+actual_h, x_offset:x_offset+actual_w] = (pattern_crop > 128).astype(np.uint8)
        
        # Calculate overlap (intersection) and union
        intersection = np.logical_and(combined_stitch_mask > 0, pattern_mask_roi > 0)
        union = np.logical_or(combined_stitch_mask > 0, pattern_mask_roi > 0)
        
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
                covered_pixels = np.sum(np.logical_and(combined_stitch_mask > 0, pattern_mask_roi > 0))
                self.pattern_progress = min(100.0, (covered_pixels / pattern_pixels) * 100.0)
            
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
        (title_w, title_h), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_DUPLEX, title_font_scale, title_thickness)
        title_x = x + (w - title_w) // 2
        title_y = y + 40
        cv2.putText(frame, title_text, (title_x, title_y), cv2.FONT_HERSHEY_DUPLEX, 
                   title_font_scale, self.COLORS['glow_cyan'], title_thickness + 1)
        cv2.putText(frame, title_text, (title_x, title_y), cv2.FONT_HERSHEY_DUPLEX, 
                   title_font_scale, self.COLORS['bright_blue'], title_thickness)
        
        # Draw horizontal divider
        cv2.line(frame, (x + 15, title_y + 15), (x + w - 15, title_y + 15), self.COLORS['medium_blue'], 2)
        
        # Stats content
        content_x = x + 20
        start_y = title_y + 45
        line_height = 35
        
        # Accuracy
        self.draw_stat_item(frame, "ACCURACY", f"{self.current_accuracy:.1f}%", 
                           content_x, start_y, w - 40)
        
        # Score
        self.draw_stat_item(frame, "SCORE", str(self.total_score), 
                           content_x, start_y + line_height, w - 40)
        
        # Progress bar
        progress_y = start_y + line_height * 2 + 30
        cv2.putText(frame, "PROGRESS", (content_x, progress_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, self.COLORS['text_secondary'], 1)
        
        # Progress bar background
        bar_y = progress_y + 15
        bar_width = w - 40
        bar_height = 20
        cv2.rectangle(frame, (content_x, bar_y), (content_x + bar_width, bar_y + bar_height), 
                     self.COLORS['medium_blue'], 2)
        
        # Progress bar fill
        if self.pattern_progress > 0:
            fill_width = int((bar_width - 4) * (self.pattern_progress / 100.0))
            if fill_width > 0:
                overlay_bar = frame.copy()
                cv2.rectangle(overlay_bar, (content_x + 2, bar_y + 2), 
                            (content_x + 2 + fill_width, bar_y + bar_height - 2), 
                            self.COLORS['neon_blue'], -1)
                cv2.addWeighted(overlay_bar, 0.7, frame, 0.3, 0, frame)
        
        # Progress percentage text
        progress_text = f"{self.pattern_progress:.0f}%"
        (prog_w, prog_h), _ = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, progress_text, (content_x + (bar_width - prog_w) // 2, bar_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text_primary'], 1)
        
    
    def draw_stat_item(self, frame, label, value, x, y, max_width, value_color=None):
        """Helper method to draw a stat item (label and value)"""
        if value_color is None:
            value_color = self.COLORS['text_primary']
        
        # Draw label
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, self.COLORS['text_secondary'], 1)
        
        # Draw value (right-aligned on same line)
        (val_w, val_h), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        value_x = x + max_width - val_w
        cv2.putText(frame, value, (value_x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, value_color, 2)

    def handle_click(self, x, y):
        """Handle mouse clicks in pattern mode"""
        # Check back button
        bb = self.back_button
        if bb['x'] <= x <= bb['x'] + bb['w'] and bb['y'] <= y <= bb['y'] + bb['h']:
            self.play_button_click_sound()
            return 'back'
        
        return None
