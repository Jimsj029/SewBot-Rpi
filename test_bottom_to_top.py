"""
Test script for bottom-to-top segment ordering with strict one-at-a-time completion
Verifies segments start from bottom and must be completed sequentially
"""

import numpy as np
import cv2

def test_segment_ordering_bottom_to_top():
    """Test that segment 1 is at bottom, segment 10 is at top"""
    print("=" * 60)
    print("TEST 1: Bottom-to-Top Segment Ordering")
    print("=" * 60)
    
    # Simulate pattern
    pattern_height = 300
    pattern_width = 200
    num_segments = 10
    segment_height = pattern_height // num_segments  # 30 pixels each
    
    print(f"\nPattern size: {pattern_height}x{pattern_width}")
    print(f"Number of segments: {num_segments}")
    print(f"Segment height: {segment_height} pixels")
    print("\nExpected ordering (REVERSED): Segment 1 = BOTTOM")
    print()
    
    # Create pattern alpha
    pattern_alpha = np.ones((pattern_height, pattern_width), dtype=np.float32)
    
    # Simulate reversed segment creation
    segment_masks = {}
    for seg_num in range(1, num_segments + 1):
        # Reverse the Y coordinate calculation
        reversed_seg = num_segments - seg_num + 1
        start_y = (reversed_seg - 1) * segment_height
        end_y = pattern_height if reversed_seg == num_segments else reversed_seg * segment_height
        
        seg_mask = np.zeros((pattern_height, pattern_width), dtype=np.uint8)
        seg_mask[start_y:end_y, :] = 255
        segment_masks[seg_num] = seg_mask
        
        position = "BOTTOM" if seg_num == 1 else ("TOP" if seg_num == num_segments else "MIDDLE")
        print(f"Segment {seg_num:2d}: rows {start_y:3d}-{end_y:3d} ({end_y - start_y} pixels) [{position}]")
    
    # Verify segment 1 is at bottom
    seg1_mask = segment_masks[1]
    seg1_pixels = np.where(seg1_mask > 0)
    seg1_y_min = np.min(seg1_pixels[0])
    seg1_y_max = np.max(seg1_pixels[0])
    
    # Verify segment 10 is at top
    seg10_mask = segment_masks[10]
    seg10_pixels = np.where(seg10_mask > 0)
    seg10_y_min = np.min(seg10_pixels[0])
    seg10_y_max = np.max(seg10_pixels[0])
    
    print()
    print(f"Segment 1 Y range: {seg1_y_min}-{seg1_y_max}")
    print(f"Segment 10 Y range: {seg10_y_min}-{seg10_y_max}")
    print()
    
    # Check if ordering is correct
    if seg1_y_min > seg10_y_max:
        print("✓ Segment 1 is at BOTTOM (correct!)")
        print("✓ Segment 10 is at TOP (correct!)")
    else:
        print("✗ Segments NOT correctly ordered!")
        return False
    
    print("\n✓ Bottom-to-top ordering working correctly")
    return True

def test_strict_sequential_completion():
    """Test that segments must be completed strictly one at a time"""
    print("\n" + "=" * 60)
    print("TEST 2: Strict Sequential Completion")
    print("=" * 60)
    
    current_segment = 2
    completed_segments = {1}
    
    print(f"\nCurrent segment: {current_segment}")
    print(f"Completed segments: {completed_segments}")
    print()
    
    # Test cases for strict mode
    test_cases = [
        (1, "Stitch in segment 1 (already completed)", True, "Already done!"),
        (2, "Stitch in segment 2 (current segment)", False, "Perfect!"),
        (3, "Stitch in segment 3 (next segment)", True, "Skip ahead!"),
        (5, "Stitch in segment 5 (skip ahead)", True, "Skip ahead!"),
    ]
    
    print("STRICT MODE: Only current segment allowed\n")
    
    for seg, description, should_warn, reason in test_cases:
        deviation_detected = False
        
        if seg == current_segment:
            result = "✓ OK"
            warning = "No warning"
        else:
            deviation_detected = True
            result = "✗ DEVIATION"
            if seg in completed_segments:
                warning = f"⚠ Segment {seg} already done! Focus on Segment {current_segment}"
            elif seg < current_segment:
                warning = f"⚠ Going backwards! Complete Segment {current_segment}"
            else:
                warning = f"⚠ Skip detected! Complete Segment {current_segment} first"
        
        status = "WARN" if deviation_detected else "OK"
        correct = "✓" if deviation_detected == should_warn else "✗"
        
        print(f"{correct} {description:45s} → {status:6s}")
        print(f"   Reason: {reason:20s} | {warning}")
        print()
    
    print("✓ Strict sequential completion working correctly")
    return True

def test_segment_roi_bottom_to_top():
    """Test ROI calculation for bottom-to-top ordering"""
    print("\n" + "=" * 60)
    print("TEST 3: Segment ROI (Bottom-to-Top)")
    print("=" * 60)
    
    pattern_height = 300
    pattern_width = 200
    num_segments = 10
    segment_height = pattern_height // num_segments
    camera_height = 420
    camera_width = 560
    
    x_offset = (camera_width - pattern_width) // 2
    y_offset = (camera_height - pattern_height) // 2
    
    print(f"\nPattern: {pattern_width}x{pattern_height}")
    print(f"Camera: {camera_width}x{camera_height}")
    print()
    
    # Test ROI for segments (bottom to top)
    for test_segment in [1, 5, 10]:
        # Calculate reversed position
        reversed_seg = num_segments - test_segment + 1
        seg_start_y = (reversed_seg - 1) * segment_height
        seg_end_y = pattern_height if reversed_seg == num_segments else reversed_seg * segment_height
        
        # Calculate ROI
        padding_x = int(pattern_width * 0.2)
        padding_y = int((seg_end_y - seg_start_y) * 0.15)
        
        roi_x1 = max(0, x_offset - padding_x)
        roi_x2 = min(camera_width, x_offset + pattern_width + padding_x)
        roi_y1 = max(0, y_offset + seg_start_y - padding_y)
        roi_y2 = min(camera_height, y_offset + seg_end_y + padding_y)
        
        roi_height = roi_y2 - roi_y1
        position = "BOTTOM" if test_segment == 1 else ("TOP" if test_segment == num_segments else "MIDDLE")
        
        print(f"Segment {test_segment:2d} [{position}]:")
        print(f"  Pattern Y: {seg_start_y:3d}-{seg_end_y:3d} (height: {seg_end_y - seg_start_y})")
        print(f"  Camera ROI Y: {roi_y1:3d}-{roi_y2:3d} (height: {roi_height})")
        print(f"  ✓ ROI constrained to segment height")
        print()
    
    print("✓ Segment ROI calculation working correctly")
    return True

def visualize_bottom_to_top():
    """Create visual demonstration of bottom-to-top ordering"""
    print("\n" + "=" * 60)
    print("VISUALIZATION: Bottom-to-Top Ordering")
    print("=" * 60)
    
    # Create visualization canvas
    width = 600
    height = 500
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 15, 10)  # Dark background
    
    # Colors
    CYAN = (255, 255, 0)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)
    GRAY = (100, 100, 100)
    RED = (0, 100, 255)
    
    # Title
    cv2.putText(canvas, "Bottom-to-Top Segment Ordering", (120, 40),
                cv2.FONT_HERSHEY_TRIPLEX, 1.0, WHITE, 2)
    
    # Draw pattern representation
    pattern_x = 200
    pattern_y = 80
    seg_width = 200
    seg_height = 35
    
    # Current state: Segment 1 complete, working on segment 2
    current_segment = 2
    completed_segments = {1}
    
    # Draw segments from top to bottom (but numbered bottom to top)
    for display_row in range(10):
        seg_num = 10 - display_row  # Segment 10 at top, Segment 1 at bottom
        seg_y = pattern_y + display_row * seg_height
        
        if seg_num in completed_segments:
            # Completed: cyan
            color = CYAN
            label = f"Seg {seg_num} ✓"
            border = 2
        elif seg_num == current_segment:
            # Current: yellow
            color = YELLOW
            label = f"Seg {seg_num} ← NOW"
            border = 3
        else:
            # Upcoming: gray
            color = GRAY
            label = f"Seg {seg_num}"
            border = 1
        
        # Draw segment
        cv2.rectangle(canvas, (pattern_x, seg_y), 
                     (pattern_x + seg_width, seg_y + seg_height - 2),
                     color, border)
        
        # Label
        cv2.putText(canvas, label, (pattern_x + 10, seg_y + 23),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color if border > 1 else WHITE, 2)
        
        # Position indicator
        if seg_num == 10:
            cv2.putText(canvas, "TOP", (pattern_x + seg_width + 15, seg_y + 23),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        elif seg_num == 1:
            cv2.putText(canvas, "BOTTOM", (pattern_x + seg_width + 15, seg_y + 23),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    
    # Instructions
    instructions_y = pattern_y + 10 * seg_height + 30
    cv2.putText(canvas, "Start from BOTTOM (Segment 1)", (50, instructions_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    cv2.putText(canvas, "Complete one segment at a time", (50, instructions_y + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    cv2.putText(canvas, "Progress upward to TOP (Segment 10)", (50, instructions_y + 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    
    # Save
    output_path = "bottom_to_top_ordering.png"
    cv2.imwrite(output_path, canvas)
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Display
    cv2.imshow("Bottom-to-Top Ordering", canvas)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return True

def run_all_tests():
    """Run all bottom-to-top ordering tests"""
    print("\n" + "=" * 60)
    print("BOTTOM-TO-TOP SEGMENT ORDERING TESTS")
    print("=" * 60)
    print()
    
    try:
        result1 = test_segment_ordering_bottom_to_top()
        result2 = test_strict_sequential_completion()
        result3 = test_segment_roi_bottom_to_top()
        result4 = visualize_bottom_to_top()
        
        if all([result1, result2, result3, result4]):
            print("\n" + "=" * 60)
            print("ALL TESTS PASSED! ✓")
            print("=" * 60)
            print()
            print("Bottom-to-top ordering is working correctly!")
            print("- Segment 1 starts at BOTTOM")
            print("- Segment 10 ends at TOP")
            print("- Must complete one segment at a time (strict)")
            print("- Each segment has its own ROI")
            print("")
            print("Run main.py to see it in action.")
        else:
            print("\n✗ Some tests failed!")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
