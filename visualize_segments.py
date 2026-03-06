"""
Visual demonstration of segment-based pattern tracking
Creates a simple visualization to show how the system works
"""

import numpy as np
import cv2

def create_segment_visualization():
    """Create a visual diagram showing segment tracking"""
    
    # Create canvas
    width = 800
    height = 600
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 15, 10)  # Dark background
    
    # Colors (BGR)
    COMPLETED_COLOR = (255, 255, 0)  # Cyan
    CURRENT_COLOR = (0, 255, 255)    # Yellow
    UPCOMING_COLOR = (100, 100, 100) # Gray
    DEVIATION_COLOR = (0, 0, 255)    # Red
    TEXT_COLOR = (255, 255, 255)     # White
    
    # Example state
    current_segment = 4
    completed_segments = {1, 2, 3}
    num_segments = 10
    
    # Draw title
    cv2.putText(canvas, "SEGMENT-BASED PATTERN TRACKING", (150, 50),
                cv2.FONT_HERSHEY_TRIPLEX, 1.2, TEXT_COLOR, 2)
    
    # Draw pattern representation (vertical segments)
    pattern_x = 100
    pattern_y = 100
    pattern_width = 200
    pattern_height = 400
    segment_height = pattern_height // num_segments
    
    # Draw segments
    for seg in range(1, num_segments + 1):
        seg_y = pattern_y + (seg - 1) * segment_height
        
        # Determine color
        if seg in completed_segments:
            color = COMPLETED_COLOR
            status = "COMPLETED"
        elif seg == current_segment:
            color = CURRENT_COLOR
            status = "CURRENT"
        else:
            color = UPCOMING_COLOR
            status = "UPCOMING"
        
        # Draw segment
        cv2.rectangle(canvas, (pattern_x, seg_y), 
                     (pattern_x + pattern_width, seg_y + segment_height - 2),
                     color, -1)
        cv2.rectangle(canvas, (pattern_x, seg_y), 
                     (pattern_x + pattern_width, seg_y + segment_height - 2),
                     (200, 200, 200), 1)
        
        # Draw segment number
        cv2.putText(canvas, str(seg), (pattern_x + 10, seg_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0) if seg in completed_segments or seg == current_segment else TEXT_COLOR, 2)
    
    # Draw legend
    legend_x = 400
    legend_y = 150
    
    cv2.putText(canvas, "SEGMENT STATES:", (legend_x, legend_y),
                cv2.FONT_HERSHEY_TRIPLEX, 0.8, TEXT_COLOR, 2)
    
    legend_items = [
        ("Completed (Segments 1-3)", COMPLETED_COLOR, "Stitching done, 70%+ coverage"),
        ("Current (Segment 4)", CURRENT_COLOR, "Active segment - sew here now"),
        ("Upcoming (Segments 5-10)", UPCOMING_COLOR, "Not started yet"),
        ("Deviation", DEVIATION_COLOR, "Wrong segment detected!")
    ]
    
    item_y = legend_y + 40
    for label, color, description in legend_items:
        # Color box
        cv2.rectangle(canvas, (legend_x, item_y - 15), (legend_x + 30, item_y),
                     color, -1)
        cv2.rectangle(canvas, (legend_x, item_y - 15), (legend_x + 30, item_y),
                     (200, 200, 200), 1)
        
        # Label
        cv2.putText(canvas, label, (legend_x + 40, item_y - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
        
        # Description
        cv2.putText(canvas, description, (legend_x + 40, item_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        item_y += 60
    
    # Draw example scenarios
    scenario_y = 420
    cv2.putText(canvas, "EXAMPLES:", (legend_x, scenario_y),
                cv2.FONT_HERSHEY_TRIPLEX, 0.8, TEXT_COLOR, 2)
    
    scenarios = [
        ("Stitch in Segment 3", (0, 255, 0), "OK - Already completed"),
        ("Stitch in Segment 4", (0, 255, 0), "OK - Current segment"),
        ("Stitch in Segment 6", (0, 0, 255), "DEVIATION - Skip ahead!")
    ]
    
    scenario_y += 40
    for label, color, result in scenarios:
        # Checkmark or X
        symbol = "\u2713" if color == (0, 255, 0) else "\u2717"
        cv2.putText(canvas, symbol, (legend_x, scenario_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Label
        cv2.putText(canvas, f"{label}:", (legend_x + 30, scenario_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
        
        # Result
        cv2.putText(canvas, result, (legend_x + 30, scenario_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        scenario_y += 45
    
    # Draw arrow pointing to current segment
    arrow_x = pattern_x + pattern_width + 20
    arrow_y = pattern_y + (current_segment - 1) * segment_height + segment_height // 2
    cv2.arrowedLine(canvas, (arrow_x, arrow_y), (arrow_x + 50, arrow_y),
                   CURRENT_COLOR, 3, tipLength=0.3)
    cv2.putText(canvas, "ACTIVE", (arrow_x + 60, arrow_y + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, CURRENT_COLOR, 2)
    
    return canvas

def main():
    """Create and save the visualization"""
    print("Creating segment tracking visualization...")
    
    canvas = create_segment_visualization()
    
    # Save image
    output_path = "segment_tracking_visualization.png"
    cv2.imwrite(output_path, canvas)
    print(f"✓ Visualization saved to: {output_path}")
    
    # Display the image
    cv2.imshow("Segment-Based Pattern Tracking", canvas)
    print("\nPress any key to close the visualization...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nVisualization closed.")
    print("This diagram shows how the 10-segment tracking system works.")
    print("Refer to SEGMENT_TRACKING.md for detailed documentation.")

if __name__ == "__main__":
    main()
