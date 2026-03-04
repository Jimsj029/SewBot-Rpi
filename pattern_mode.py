"""
Pattern Mode - Sewing Pattern Recognition
"""

import cv2
import numpy as np
import math
import os
from ultralytics import YOLO


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
        
        # Level info display at top center
        self.level_display_y = 30
        
        # Camera display area (centered, moved higher)
        self.camera_width = 560
        self.camera_height = 420
        self.camera_x = 216  # Centered horizontally (1000 - 560) // 2
        self.camera_y = 80  # Moved up since no level buttons
        
        # Back button (top left)
        self.back_button = {'x': 20, 'y': 20, 'w': 120, 'h': 50}
        
        # Load models from models directory
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Load stitch detection model
        try:
            stitch_model_path = os.path.join(models_dir, 'stitch.pt')
            if os.path.exists(stitch_model_path):
                self.stitch_model = YOLO(stitch_model_path)
                print(f"✓ Stitch detection model loaded: {stitch_model_path}")
            else:
                print(f"⚠ Stitch model not found: {stitch_model_path}")
                self.stitch_model = None
        except Exception as e:
            print(f"Warning: Could not load stitch model: {e}")
            self.stitch_model = None
        
        # Load cloth detection model
        try:
            cloth_model_path = os.path.join(models_dir, 'cloth.pt')
            if os.path.exists(cloth_model_path):
                self.cloth_model = YOLO(cloth_model_path)
                print(f"✓ Cloth detection model loaded: {cloth_model_path}")
            else:
                print(f"⚠ Cloth model not found: {cloth_model_path}")
                self.cloth_model = None
        except Exception as e:
            print(f"Warning: Could not load cloth model: {e}")
            self.cloth_model = None
        
        # Angle smoothing variables
        self.previous_angle = 0
        self.angle_smoothing = 0.3  # Lower = more smoothing (0-1)
    
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
            
            # Detect cloth using cloth.pt model (mask method for accurate centering)
            cloth_centroid = None
            cloth_angle = 0  # Store cloth rotation angle
            if self.cloth_model is not None:
                try:
                    cloth_results = self.cloth_model(cam_frame, verbose=False)
                    
                    for result in cloth_results:
                        # Use masks for more accurate cloth detection
                        if hasattr(result, 'masks') and result.masks is not None:
                            masks = result.masks.data.cpu().numpy()
                            boxes = result.boxes
                            
                            # Get the largest detected cloth mask (by area)
                            best_mask = None
                            best_area = 0
                            
                            for i, (mask, box) in enumerate(zip(masks, boxes)):
                                # Resize mask to frame size
                                mask_resized = cv2.resize(mask, (self.camera_width, self.camera_height))
                                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                                
                                # Calculate mask area
                                mask_area = np.sum(mask_binary)
                                
                                if mask_area > best_area:
                                    best_area = mask_area
                                    best_mask = mask_binary
                            
                            if best_mask is not None:
                                # Find contours to draw outline
                                contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    # Get the largest contour
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    
                                    # Draw cloth outline (green)
                                    cv2.drawContours(cam_frame, [largest_contour], -1, (0, 255, 0), 2)
                                    
                                    # Calculate centroid using moments
                                    M = cv2.moments(best_mask)
                                    if M["m00"] != 0:
                                        cloth_centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                                        # Draw centroid point
                                        cv2.circle(cam_frame, cloth_centroid, 5, (0, 255, 0), -1)
                                    
                                    # Use PCA to find the orientation of the cloth
                                    # Reshape contour for PCA
                                    contour_points = largest_contour.reshape(-1, 2).astype(np.float32)
                                    
                                    # Perform PCA
                                    mean, eigenvectors = cv2.PCACompute(contour_points, mean=None)
                                    
                                    # The first eigenvector (principal component) gives the orientation
                                    # Calculate angle from the first eigenvector
                                    raw_angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
                                    
                                    # Smooth the angle to prevent jittering
                                    # Handle angle wrapping (e.g., -179 to 179 transition)
                                    angle_diff = raw_angle - self.previous_angle
                                    if angle_diff > 180:
                                        angle_diff -= 360
                                    elif angle_diff < -180:
                                        angle_diff += 360
                                    
                                    # Exponential moving average for smooth transitions
                                    cloth_angle = self.previous_angle + self.angle_smoothing * angle_diff
                                    self.previous_angle = cloth_angle
                                    
                                    # Add 90 degrees to align pattern with vertical cloth orientation
                                    cloth_angle += 90
                                    
                                break
                
                except Exception as e:
                    print(f"Cloth detection error: {e}")
            
            # Run stitch detection using stitch.pt model (mask only)
            detected_stitch_masks = []
            
            if self.stitch_model is not None:
                try:
                    results = self.stitch_model(cam_frame, verbose=False)
                    
                    for result in results:
                        # Only use masks (segmentation) for stitch detection
                        if hasattr(result, 'masks') and result.masks is not None:
                            masks = result.masks.data.cpu().numpy()
                            boxes = result.boxes
                            
                            for i, (mask, box) in enumerate(zip(masks, boxes)):
                                conf = box.conf[0].cpu().numpy()
                                
                                # Resize mask to frame size
                                mask_resized = cv2.resize(mask, (self.camera_width, self.camera_height))
                                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                                
                                # Store the mask for visualization
                                detected_stitch_masks.append({
                                    'mask': mask_binary,
                                    'confidence': float(conf)
                                })
                    
                except Exception as e:
                    print(f"Stitch detection error: {e}")
            
            # Display detected stitch masks as semi-transparent overlay
            if len(detected_stitch_masks) > 0:
                # Create a combined mask overlay
                stitch_overlay = np.zeros_like(cam_frame)
                
                for stitch_data in detected_stitch_masks:
                    mask = stitch_data['mask']
                    # Color the detected stitches in purple for better visibility
                    stitch_overlay[mask > 0] = [180, 0, 255]  # Purple color in BGR
                
                # Blend the stitch overlay with the camera frame
                alpha = 0.6  # Transparency level
                cam_frame = cv2.addWeighted(cam_frame, 1, stitch_overlay, alpha, 0)
            
            # Overlay pattern mask and compare with detected stitches
            if pattern_overlay is not None and pattern_alpha is not None:
                overlay_h, overlay_w = pattern_overlay.shape[:2]
                
                # Center pattern on detected cloth centroid (no rotation)
                if cloth_centroid is not None:
                    # Center on cloth centroid
                    x_offset = cloth_centroid[0] - (overlay_w // 2)
                    y_offset = cloth_centroid[1] - (overlay_h // 2)
                    # Clamp to camera bounds to prevent overflow
                    x_offset = max(0, min(x_offset, self.camera_width - overlay_w))
                    y_offset = max(0, min(y_offset, self.camera_height - overlay_h))
                else:
                    # Fallback to camera center if no cloth detected
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
                        # Apply pattern overlay (semi-transparent white lines)
                        roi = cam_frame[y_offset:overlay_end_y, x_offset:overlay_end_x]
                        pattern_crop = pattern_overlay[0:actual_h, 0:actual_w]
                        alpha_crop = pattern_alpha[0:actual_h, 0:actual_w]
                        
                        for c in range(3):
                            roi[:, :, c] = (alpha_crop * pattern_crop[:, :, c] * 0.5 + 
                                          (1 - alpha_crop * 0.5) * roi[:, :, c])
                        cam_frame[y_offset:overlay_end_y, x_offset:overlay_end_x] = roi
            
            frame[self.camera_y:self.camera_y+self.camera_height, 
                  self.camera_x:self.camera_x+self.camera_width] = cam_frame
        
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
    

    def handle_click(self, x, y):
        """Handle mouse clicks in pattern mode"""
        # Check back button
        bb = self.back_button
        if bb['x'] <= x <= bb['x'] + bb['w'] and bb['y'] <= y <= bb['y'] + bb['h']:
            return 'back'
        
        return None
