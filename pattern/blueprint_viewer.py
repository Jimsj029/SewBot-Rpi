import cv2
import numpy as np
import os

class BlueprintCameraOverlay:
    def __init__(self):
        self.current_level = 1
        self.blueprint_folder = 'blueprint'
        self.window_name = 'Sewing Guide Camera'
        
        # Uniform size for all blueprints
        self.uniform_width = 200
        self.uniform_height = 300
        
        # Initialize camera
        print("Opening camera...")
        self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            print("Error: Could not open camera")
            exit()
        else:
            print("Camera opened successfully!")
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.alpha_blend = 0.9
        
    def load_blueprint(self, level):
        img_path = os.path.join(self.blueprint_folder, f'level{level}.png')
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return None, None
        
        # Load PNG with alpha channel
        overlay_png = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if overlay_png is None:
            print(f"Failed to load image: {img_path}")
            return None, None
        
        print(f"Loaded level{level}.png - Original: {overlay_png.shape[1]}x{overlay_png.shape[0]}")
        
        # Resize to uniform size
        overlay_png = cv2.resize(overlay_png, (self.uniform_width, self.uniform_height))
        
        # Check if image has alpha channel
        if len(overlay_png.shape) == 3 and overlay_png.shape[2] == 4:
            # Has alpha channel
            overlay = overlay_png[:, :, :3]  # BGR
            alpha = overlay_png[:, :, 3] / 255.0  # Normalize alpha 0-1
        else:
            # No alpha channel - convert to grayscale and create alpha from dark pixels
            if len(overlay_png.shape) == 2:
                overlay_png = cv2.cvtColor(overlay_png, cv2.COLOR_GRAY2BGR)
            
            overlay = overlay_png
            # Create alpha channel: only show lines, not background
            gray = cv2.cvtColor(overlay_png, cv2.COLOR_BGR2GRAY)
            # Apply threshold to separate lines from background
            # Lines (dark pixels below threshold) = opaque, background = transparent
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            # Invert so dark lines become opaque (1.0) and white background becomes transparent (0.0)
            alpha = (255 - binary) / 255.0
        
        return overlay, alpha
    
    def draw_buttons(self, img):
        button_height = 40
        button_width = 80
        start_x = 10
        start_y = 10
        spacing = 8
        
        for i in range(1, 6):
            x = start_x + (button_width + spacing) * (i - 1)
            y = start_y
            
            if i == self.current_level:
                color = (0, 255, 0)
                thickness = -1
            else:
                color = (200, 200, 200)
                thickness = 2
            
            cv2.rectangle(img, (x, y), (x + button_width, y + button_height), color, thickness)
            
            text = f"Lvl {i}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            
            text_x = x + (button_width - text_size[0]) // 2
            text_y = y + (button_height + text_size[1]) // 2
            
            text_color = (0, 0, 0) if i == self.current_level else (255, 255, 255)
            cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
        
        # Instructions
        mode_text = f"Press 1-5 to switch | Q/ESC to QUIT"
        cv2.putText(img, mode_text, (start_x, img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            button_height = 40
            button_width = 80
            start_x = 10
            start_y = 10
            spacing = 8
            
            for i in range(1, 6):
                btn_x = start_x + (button_width + spacing) * (i - 1)
                btn_y = start_y
                
                if btn_x <= x <= btn_x + button_width and btn_y <= y <= btn_y + button_height:
                    self.current_level = i
                    self.level_changed = True
                    break
    
    def run(self):
        print("\nSewing Guide - Blueprint Camera Overlay")
        print("=" * 50)
        print("Controls:")
        print("- Click buttons or press 1-5 to switch levels")
        print("- Press Q or ESC to QUIT")
        print("=" * 50)
        
        current_overlay = None
        current_alpha = None
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Load current level if needed
            if current_overlay is None or hasattr(self, 'level_changed'):
                current_overlay, current_alpha = self.load_blueprint(self.current_level)
                if hasattr(self, 'level_changed'):
                    delattr(self, 'level_changed')
            
            if current_overlay is None or current_alpha is None:
                print("Failed to load blueprint. Exiting...")
                break
            
            # Get frame dimensions
            frame_h, frame_w = frame.shape[:2]
            overlay_h, overlay_w = current_overlay.shape[:2]
            
            # Calculate position to center the overlay
            x_offset = (frame_w - overlay_w) // 2
            y_offset = (frame_h - overlay_h) // 2
            
            # Ensure overlay fits in frame
            if x_offset >= 0 and y_offset >= 0 and (x_offset + overlay_w) <= frame_w and (y_offset + overlay_h) <= frame_h:
                # Blend overlay with frame using alpha channel
                roi = frame[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w]
                
                for c in range(3):  # BGR channels
                    roi[:, :, c] = (current_alpha * current_overlay[:, :, c] * self.alpha_blend +
                                    (1 - current_alpha * self.alpha_blend) * roi[:, :, c])
                
                frame[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w] = roi
            else:
                # If overlay is larger than frame, resize frame instead
                frame = cv2.resize(frame, (overlay_w + 100, overlay_h + 100))
                frame_h, frame_w = frame.shape[:2]
                x_offset = (frame_w - overlay_w) // 2
                y_offset = (frame_h - overlay_h) // 2
                
                roi = frame[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w]
                
                for c in range(3):
                    roi[:, :, c] = (current_alpha * current_overlay[:, :, c] * self.alpha_blend +
                                    (1 - current_alpha * self.alpha_blend) * roi[:, :, c])
                
                frame[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w] = roi
            
            # Draw buttons
            frame = self.draw_buttons(frame)
            
            cv2.imshow(self.window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q to quit
                print("Quitting...")
                break
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                self.current_level = int(chr(key))
                self.level_changed = True
                current_overlay = None
                current_alpha = None
        
        self.camera.release()
        cv2.destroyAllWindows()
        print("Program closed.")

if __name__ == '__main__':
    viewer = BlueprintCameraOverlay()
    viewer.run()
