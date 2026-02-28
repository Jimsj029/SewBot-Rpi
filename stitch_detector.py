"""
Stitch Line Detection Module using YOLOv8 Segmentation
Optimized for Raspberry Pi
"""

import cv2
import numpy as np
import os


class StitchDetector:
    """YOLOv8-based stitch line detector optimized for Raspberry Pi"""
    
    def __init__(self, model_path='yolov8-stitch-detection.pt', confidence=0.35, iou_threshold=0.6, img_size=416):
        """
        Initialize the stitch detector
        
        Args:
            model_path (str): Path to the trained YOLOv8 model
            confidence (float): Confidence threshold for detection (0.0-1.0)
            iou_threshold (float): IOU threshold for NMS
            img_size (int): Input image size for model (smaller = faster)
        """
        self.model = None
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.model_loaded = False
        self.error_message = None
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv8 model"""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                self.error_message = f"Model file not found: {self.model_path}"
                print(f"⚠️  {self.error_message}")
                return False
            
            # Import ultralytics (YOLOv8)
            try:
                from ultralytics import YOLO
            except ImportError:
                self.error_message = "ultralytics not installed. Run: pip install ultralytics"
                print(f"⚠️  {self.error_message}")
                return False
            
            # Load the trained model
            print(f"📦 Loading YOLOv8 stitch detection model...")
            self.model = YOLO(self.model_path)
            self.model_loaded = True
            print(f"✅ Model loaded successfully!")
            print(f"   Confidence: {self.confidence}")
            print(f"   IOU Threshold: {self.iou_threshold}")
            print(f"   Image Size: {self.img_size}x{self.img_size}")
            return True
            
        except Exception as e:
            self.error_message = f"Error loading model: {str(e)}"
            print(f"❌ {self.error_message}")
            return False
    
    def detect(self, frame, show_boxes=False, show_labels=False, mask_alpha=0.5):
        """
        Detect stitch lines in a frame
        
        Args:
            frame (np.ndarray): Input frame (BGR image)
            show_boxes (bool): Whether to show bounding boxes
            show_labels (bool): Whether to show labels
            mask_alpha (float): Transparency of masks (0.0-1.0)
        
        Returns:
            tuple: (annotated_frame, detection_count, results)
                - annotated_frame: Frame with visualized detections
                - detection_count: Number of stitch lines detected
                - results: Raw YOLO results object
        """
        if not self.model_loaded:
            # Return original frame with error message if model not loaded
            error_frame = frame.copy()
            cv2.putText(error_frame, "Stitch Detection: Model not loaded", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if self.error_message:
                cv2.putText(error_frame, self.error_message[:50], 
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            return error_frame, 0, None
        
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False
            )
            
            # Get the annotated frame with segmentation masks
            annotated_frame = results[0].plot(
                boxes=show_boxes,
                labels=show_labels,
                masks=True,
                conf=True
            )
            
            # Count detections
            detection_count = len(results[0].boxes) if results[0].boxes is not None else 0
            
            return annotated_frame, detection_count, results
            
        except Exception as e:
            print(f"Error during detection: {e}")
            error_frame = frame.copy()
            cv2.putText(error_frame, f"Detection Error: {str(e)[:40]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return error_frame, 0, None
    
    def set_confidence(self, confidence):
        """Update confidence threshold"""
        self.confidence = max(0.1, min(0.9, confidence))
        print(f"Confidence threshold set to: {self.confidence:.2f}")
    
    def increase_confidence(self, step=0.05):
        """Increase confidence threshold"""
        self.set_confidence(self.confidence + step)
    
    def decrease_confidence(self, step=0.05):
        """Decrease confidence threshold"""
        self.set_confidence(self.confidence - step)
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model_loaded
    
    def get_model_info(self):
        """Get model information"""
        return {
            'loaded': self.model_loaded,
            'path': self.model_path,
            'confidence': self.confidence,
            'iou_threshold': self.iou_threshold,
            'img_size': self.img_size,
            'error': self.error_message
        }


# Example usage and testing
if __name__ == '__main__':
    import time
    
    print("=" * 60)
    print("YOLOv8 Stitch Line Detection Test")
    print("=" * 60)
    
    # Initialize detector
    detector = StitchDetector()
    
    if not detector.is_loaded():
        print("\n❌ Model could not be loaded. Please check:")
        print("   1. Model file exists: yolov8-stitch-detection.pt")
        print("   2. ultralytics is installed: pip install ultralytics")
        exit(1)
    
    # Open camera
    print("\n📹 Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        exit(1)
    
    print("✅ Camera opened successfully")
    print("\nControls:")
    print("  'q' - Quit")
    print("  '+' - Increase confidence")
    print("  '-' - Decrease confidence")
    print("  's' - Save current frame")
    print("\n" + "=" * 60)
    
    frame_count = 0
    fps_start = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Detect stitch lines
        annotated_frame, count, _ = detector.detect(frame, show_boxes=False, show_labels=False)
        
        # Calculate FPS
        frame_count += 1
        if time.time() - fps_start >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_start = time.time()
        
        # Add info overlay
        cv2.putText(annotated_frame, f'FPS: {fps} | Detections: {count} | Conf: {detector.confidence:.2f}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('Stitch Line Detection Test', annotated_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            detector.increase_confidence()
        elif key == ord('-') or key == ord('_'):
            detector.decrease_confidence()
        elif key == ord('s'):
            filename = f'stitch_detection_test_{int(time.time())}.jpg'
            cv2.imwrite(filename, annotated_frame)
            print(f"Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest completed!")
