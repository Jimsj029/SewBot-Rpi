"""
Pattern Mode - Sewing Pattern Recognition with Fabric Color Detection
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
        
        # Camera display area (centered)
        self.camera_width = 560
        self.camera_height = 420
        self.camera_x = 216  # Centered horizontally (1000 - 560) // 2
        self.camera_y = 80
        
        # Fabric color panel (right side of camera)
        self.color_panel_x = self.camera_x + self.camera_width + 37
        self.color_panel_y = 100
        self.color_panel_width = 180
        self.color_panel_height = 400
        
        # Back button (top left)
        self.back_button = {'x': 20, 'y': 20, 'w': 120, 'h': 50}
        
        # Level buttons at top (centered over camera)
        self.level_buttons = []
        button_width = 80
        button_height = 40
        # Center buttons over camera area
        buttons_total_width = (button_width * 5) + (20 * 4)
        camera_center = self.camera_x + (self.camera_width // 2)
        start_x = camera_center - (buttons_total_width // 2)
        start_y = 20
        
        for i in range(1, 6):
            self.level_buttons.append({
                'level': i,
                'x': start_x + (i - 1) * (button_width + 20),
                'y': start_y,
                'w': button_width,
                'h': button_height
            })
        
        # Fabric color detection settings
        self.fabric_colors = {
            'White': {'bgr': (255, 255, 255), 'hsv_range': ([0, 0, 200], [180, 30, 255])},
            'Black': {'bgr': (0, 0, 0), 'hsv_range': ([0, 0, 0], [180, 255, 50])},
            'Blue': {'bgr': (255, 0, 0), 'hsv_range': ([100, 50, 50], [130, 255, 255])},
            'Gray': {'bgr': (128, 128, 128), 'hsv_range': ([0, 0, 50], [180, 50, 200])},
            'Red': {'bgr': (0, 0, 255), 'hsv_range': ([0, 50, 50], [10, 255, 255])},
            'Yellow': {'bgr': (0, 255, 255), 'hsv_range': ([20, 50, 50], [35, 255, 255])},
            'Green': {'bgr': (0, 255, 0), 'hsv_range': ([40, 50, 50], [80, 255, 255])}
        }
        
        self.selected_fabric_color = None
        self.cloth_mask = None  # Mask of detected cloth
        
        # Load YOLO model
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
            self.yolo_model = YOLO(model_path)
            print(f"YOLOv8 model loaded: {model_path}")
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            self.yolo_model = None
        
        # Create color buttons (adjusted to fit in panel)
        self.color_buttons = []
        btn_h = 38
        btn_spacing = 7
        start_y = self.color_panel_y + 70
        
        for idx, color_name in enumerate(['White', 'Black', 'Blue', 'Gray', 'Red', 'Yellow', 'Green']):
            self.color_buttons.append({
                'name': color_name,
                'x': self.color_panel_x + 10,
                'y': start_y + idx * (btn_h + btn_spacing),
                'w': self.color_panel_width - 20,
                'h': btn_h
            })
    
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
    
    def detect_cloth(self, camera_frame):
        """Detect cloth of selected color in camera frame"""
        if camera_frame is None or camera_frame.size == 0 or self.selected_fabric_color is None:
            return None
        
        # Convert frame to HSV
        hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
        
        # Get color range for selected fabric color
        color_data = self.fabric_colors[self.selected_fabric_color]
        lower = np.array(color_data['hsv_range'][0])
        upper = np.array(color_data['hsv_range'][1])
        
        # Create mask for selected color
        if self.selected_fabric_color == 'Red':
            # Special handling for red (wraps around hue)
            mask1 = cv2.inRange(hsv_frame, lower, upper)
            mask2 = cv2.inRange(hsv_frame, np.array([170, 50, 50]), np.array([180, 255, 255]))
            cloth_mask = cv2.bitwise_or(mask1, mask2)
        else:
            cloth_mask = cv2.inRange(hsv_frame, lower, upper)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, kernel)
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_OPEN, kernel)
        
        return cloth_mask
    
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
        
        # Draw level buttons at top
        for btn in self.level_buttons:
            is_selected = (btn['level'] == self.current_level)
            pulse = 0.5 + 0.5 * abs(math.sin(self.glow_phase * 1.3))
            
            if is_selected:
                self.draw_glow_rect(frame, btn['x'], btn['y'], btn['w'], btn['h'], self.COLORS['neon_blue'], pulse)
                overlay = frame.copy()
                cv2.rectangle(overlay, (btn['x'] + 2, btn['y'] + 2), (btn['x'] + btn['w'] - 2, btn['y'] + btn['h'] - 2), self.COLORS['button_hover'], -1)
                cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                
                text = f"LVL {btn['level']}"
                font_scale, thickness = 0.7, 2
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
                text_x = btn['x'] + (btn['w'] - text_w) // 2
                text_y = btn['y'] + (btn['h'] + text_h) // 2
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, self.COLORS['glow_cyan'], thickness + 1)
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, self.COLORS['text_primary'], thickness)
            else:
                cv2.rectangle(frame, (btn['x'], btn['y']), (btn['x'] + btn['w'], btn['y'] + btn['h']), self.COLORS['medium_blue'], 2)
                overlay = frame.copy()
                cv2.rectangle(overlay, (btn['x'] + 2, btn['y'] + 2), (btn['x'] + btn['w'] - 2, btn['y'] + btn['h'] - 2), self.COLORS['dark_blue'], -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
                text = f"LVL {btn['level']}"
                font_scale, thickness = 0.7, 1
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
                text_x = btn['x'] + (btn['w'] - text_w) // 2
                text_y = btn['y'] + (btn['h'] + text_h) // 2
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, self.COLORS['text_secondary'], thickness)
        
        # Draw back button (top left)
        self.draw_back_button(frame)
        
        # Draw camera feed
        self.draw_camera_feed(frame, camera_frame)
        
        # Draw fabric color panel
        self.draw_fabric_color_panel(frame, camera_frame)
    
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
            
            # Detect fabric to center pattern mask
            fabric_centroid = None
            if self.selected_fabric_color:
                self.cloth_mask = self.detect_cloth(cam_frame)
                if self.cloth_mask is not None:
                    # Find largest contour (main fabric piece)
                    contours, _ = cv2.findContours(self.cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        # Get centroid of fabric
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            fabric_centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                            # Draw fabric outline
                            cv2.drawContours(cam_frame, [largest_contour], -1, (0, 255, 0), 2)
            
            # Load pattern mask for comparison
            pattern_overlay, pattern_alpha = self.load_blueprint(self.current_level)
            
            # Run YOLO stitch detection
            detected_stitches = []
            stitch_positions = []
            
            if self.yolo_model is not None:
                try:
                    results = self.yolo_model(cam_frame, verbose=False)
                    
                    for result in results:
                        # Check if model has masks (segmentation)
                        if hasattr(result, 'masks') and result.masks is not None:
                            masks = result.masks.data.cpu().numpy()
                            boxes = result.boxes
                            
                            for i, (mask, box) in enumerate(zip(masks, boxes)):
                                conf = box.conf[0].cpu().numpy()
                                cls = int(box.cls[0].cpu().numpy())
                                
                                # Resize mask to frame size
                                mask_resized = cv2.resize(mask, (self.camera_width, self.camera_height))
                                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                                
                                # Get center of stitch for comparison
                                M = cv2.moments(mask_binary)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    stitch_positions.append((cx, cy))
                                
                                detected_stitches.append({
                                    'mask': mask_binary,
                                    'confidence': float(conf),
                                    'class': cls
                                })
                        
                        # If no segmentation, use bounding boxes as stitch locations
                        elif hasattr(result, 'boxes') and result.boxes is not None:
                            boxes = result.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf[0].cpu().numpy()
                                cls = int(box.cls[0].cpu().numpy())
                                
                                # Draw stitch point at center of box
                                center_x = int((x1 + x2) / 2)
                                center_y = int((y1 + y2) / 2)
                                stitch_positions.append((center_x, center_y))
                                
                                detected_stitches.append({
                                    'position': (center_x, center_y),
                                    'confidence': float(conf),
                                    'class': cls
                                })
                    
                except Exception as e:
                    print(f"YOLO stitch detection error: {e}")
            
            # Overlay pattern mask and compare with detected stitches
            if pattern_overlay is not None and pattern_alpha is not None:
                overlay_h, overlay_w = pattern_overlay.shape[:2]
                
                # Center pattern on fabric centroid if detected, otherwise center on camera
                if fabric_centroid is not None:
                    x_offset = fabric_centroid[0] - (overlay_w // 2)
                    y_offset = fabric_centroid[1] - (overlay_h // 2)
                    # Clamp to camera bounds
                    x_offset = max(0, min(x_offset, self.camera_width - overlay_w))
                    y_offset = max(0, min(y_offset, self.camera_height - overlay_h))
                else:
                    # Fallback to center of camera
                    x_offset = (self.camera_width - overlay_w) // 2
                    y_offset = (self.camera_height - overlay_h) // 2
                
                if x_offset >= 0 and y_offset >= 0 and x_offset + overlay_w <= self.camera_width and y_offset + overlay_h <= self.camera_height:
                    # Create pattern mask for comparison
                    pattern_mask_full = np.zeros((self.camera_height, self.camera_width), dtype=np.uint8)
                    pattern_mask_region = (pattern_alpha * 255).astype(np.uint8)
                    pattern_mask_full[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w] = pattern_mask_region
                    
                    # Compare detected stitches with pattern (pixel-perfect accuracy)
                    correct_stitches = 0
                    total_stitches = len(stitch_positions)
                    
                    for sx, sy in stitch_positions:
                        # Pixel-perfect check (tolerance = 1 pixel)
                        tolerance = 1
                        y_min = max(0, sy - tolerance)
                        y_max = min(self.camera_height, sy + tolerance + 1)
                        x_min = max(0, sx - tolerance)
                        x_max = min(self.camera_width, sx + tolerance + 1)
                        
                        region = pattern_mask_full[y_min:y_max, x_min:x_max]
                        if np.any(region > 0):
                            correct_stitches += 1
                            # Mark correct stitch with green
                            cv2.circle(cam_frame, (sx, sy), 4, (0, 255, 0), -1)
                        else:
                            # Mark incorrect stitch with red
                            cv2.circle(cam_frame, (sx, sy), 4, (0, 0, 255), -1)
                    
                    # Apply pattern overlay (semi-transparent white lines)
                    roi = cam_frame[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w]
                    for c in range(3):
                        roi[:, :, c] = (pattern_alpha * pattern_overlay[:, :, c] * 0.5 + 
                                      (1 - pattern_alpha * 0.5) * roi[:, :, c])
                    cam_frame[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w] = roi
                    
                    # Display accuracy feedback
                    if total_stitches > 0:
                        accuracy = (correct_stitches / total_stitches) * 100
                        accuracy_text = f"Accuracy: {accuracy:.1f}% ({correct_stitches}/{total_stitches})"
                        accuracy_color = (0, 255, 0) if accuracy >= 80 else (0, 165, 255) if accuracy >= 60 else (0, 0, 255)
                        cv2.putText(cam_frame, accuracy_text, (10, self.camera_height - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, accuracy_color, 2)
            
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
    
    def draw_fabric_color_panel(self, frame, camera_frame):
        """Draw fabric color selection and detection panel"""
        # Panel background
        panel_color = self.COLORS['dark_blue']
        cv2.rectangle(frame, 
                     (self.color_panel_x, self.color_panel_y),
                     (self.color_panel_x + self.color_panel_width, self.color_panel_y + self.color_panel_height),
                     self.COLORS['bright_blue'], 2)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (self.color_panel_x + 2, self.color_panel_y + 2),
                     (self.color_panel_x + self.color_panel_width - 2, self.color_panel_y + self.color_panel_height - 2),
                     panel_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        title = "Fabric Color"
        font_scale = 0.7
        thickness = 2
        (title_w, title_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        title_x = self.color_panel_x + (self.color_panel_width - title_w) // 2
        title_y = self.color_panel_y + 35
        cv2.putText(frame, title, (title_x, title_y), cv2.FONT_HERSHEY_DUPLEX, 
                   font_scale, self.COLORS['bright_blue'], thickness)
        
        # Draw color buttons
        for btn in self.color_buttons:
            is_selected = (btn['name'] == self.selected_fabric_color)
            
            # Get color
            color_bgr = self.fabric_colors[btn['name']]['bgr']
            
            # Button border
            border_color = self.COLORS['neon_blue'] if is_selected else self.COLORS['medium_blue']
            
            cv2.rectangle(frame, (btn['x'], btn['y']), 
                         (btn['x'] + btn['w'], btn['y'] + btn['h']),
                         border_color, 2 if is_selected else 1)
            
            # Color swatch (left side)
            swatch_size = btn['h'] - 8
            cv2.rectangle(frame, (btn['x'] + 4, btn['y'] + 4),
                         (btn['x'] + 4 + swatch_size, btn['y'] + 4 + swatch_size),
                         color_bgr, -1)
            cv2.rectangle(frame, (btn['x'] + 4, btn['y'] + 4),
                         (btn['x'] + 4 + swatch_size, btn['y'] + 4 + swatch_size),
                         (255, 255, 255), 1)
            
            # Text
            text_x = btn['x'] + swatch_size + 12
            text_y = btn['y'] + (btn['h'] + 10) // 2
            text_color = self.COLORS['text_primary'] if is_selected else self.COLORS['text_secondary']
            cv2.putText(frame, btn['name'], (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    def handle_click(self, x, y):
        """Handle mouse clicks in pattern mode"""
        # Check back button
        bb = self.back_button
        if bb['x'] <= x <= bb['x'] + bb['w'] and bb['y'] <= y <= bb['y'] + bb['h']:
            return 'back'
        
        # Check level buttons
        for btn in self.level_buttons:
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.current_level = btn['level']
                return 'level_changed'
        
        # Check color buttons
        for btn in self.color_buttons:
            if btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= y <= btn['y'] + btn['h']:
                self.selected_fabric_color = btn['name']
                return 'color_selected'
        
        return None
