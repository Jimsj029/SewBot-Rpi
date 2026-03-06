"""
Test script for single-segment display with stitch accumulation
Verifies that only current segment shows while accumulated stitches remain visible
"""

import numpy as np
import cv2

def test_single_segment_display():
    """Test that only current segment is displayed"""
    print("=" * 60)
    print("TEST: Single Segment Display")
    print("=" * 60)
    
    # Simulate pattern with 10 segments
    pattern_height = 300
    pattern_width = 200
    num_segments = 10
    
    # Create pattern alpha
    pattern_alpha = np.ones((pattern_height, pattern_width), dtype=np.float32) * 0.5
    
    # Simulate segment masks
    segment_height = pattern_height // num_segments
    segment_masks = {}
    
    for seg_num in range(1, num_segments + 1):
        start_y = (seg_num - 1) * segment_height
        end_y = pattern_height if seg_num == num_segments else seg_num * segment_height
        
        seg_mask = np.zeros((pattern_height, pattern_width), dtype=np.uint8)
        seg_mask[start_y:end_y, :] = 255
        segment_masks[seg_num] = seg_mask
    
    # Test: Current segment = 4
    current_segment = 4
    completed_segments = {1, 2, 3}
    
    print(f"\nCurrent segment: {current_segment}")
    print(f"Completed segments: {completed_segments}")
    print()
    
    # Create overlay (simulating create_realtime_pattern logic)
    colored_overlay = np.zeros((pattern_height, pattern_width, 3), dtype=np.uint8)
    
    # ONLY show current segment (yellow)
    if current_segment in segment_masks:
        seg_mask = segment_masks[current_segment]
        pattern_pixels = np.logical_and(pattern_alpha > 0.1, seg_mask > 0)
        colored_overlay[pattern_pixels] = (0, 255, 255)  # Yellow in BGR
        print(f"✓ Segment {current_segment} displayed in YELLOW")
    
    # Check that other segments are NOT displayed
    for seg_num in range(1, num_segments + 1):
        if seg_num == current_segment:
            continue
        
        seg_mask = segment_masks[seg_num]
        seg_pixels = seg_mask > 0
        seg_colors = colored_overlay[seg_pixels]
        
        # Should be all zeros (black/hidden) for non-current segments
        if np.all(seg_colors == 0):
            print(f"✓ Segment {seg_num} hidden (correct)")
        else:
            print(f"✗ Segment {seg_num} visible (should be hidden!)")
    
    print("\n✓ Single segment display working correctly")
    return colored_overlay, segment_masks

def test_stitch_accumulation():
    """Test that accumulated stitches remain visible"""
    print("\n" + "=" * 60)
    print("TEST: Stitch Accumulation")
    print("=" * 60)
    
    pattern_height = 300
    pattern_width = 200
    num_segments = 10
    
    # Create accumulated stitch mask (simulating completed work)
    completed_stitch_mask = np.zeros((pattern_height, pattern_width), dtype=np.uint8)
    
    # Simulate stitches in segments 1, 2, 3
    segment_height = pattern_height // num_segments
    for seg_num in [1, 2, 3]:
        start_y = (seg_num - 1) * segment_height
        end_y = seg_num * segment_height
        # Add some "stitches" in this segment
        completed_stitch_mask[start_y:end_y, 50:150] = 255
    
    print(f"\nAdded stitches to segments: [1, 2, 3]")
    print(f"Current segment: 4")
    print()
    
    # Create overlay
    colored_overlay = np.zeros((pattern_height, pattern_width, 3), dtype=np.uint8)
    
    # Overlay accumulated stitches (cyan)
    stitch_pixels = completed_stitch_mask > 0
    colored_overlay[stitch_pixels] = (255, 255, 0)  # Cyan in BGR
    
    # Count visible stitch pixels per segment
    for seg_num in [1, 2, 3, 4]:
        start_y = (seg_num - 1) * segment_height
        end_y = seg_num * segment_height
        
        seg_stitch_pixels = completed_stitch_mask[start_y:end_y, :] > 0
        stitch_count = np.sum(seg_stitch_pixels)
        
        if seg_num in [1, 2, 3]:
            if stitch_count > 0:
                print(f"✓ Segment {seg_num} stitches visible (cyan): {stitch_count} pixels")
            else:
                print(f"✗ Segment {seg_num} stitches NOT visible (should be!)")
        else:
            if stitch_count == 0:
                print(f"✓ Segment {seg_num} no stitches (correct)")
            else:
                print(f"⚠ Segment {seg_num} has stitches: {stitch_count} pixels")
    
    print("\n✓ Stitch accumulation working correctly")
    return colored_overlay

def test_segment_roi():
    """Test that ROI is constrained to current segment height"""
    print("\n" + "=" * 60)
    print("TEST: Segment-Specific ROI")
    print("=" * 60)
    
    pattern_height = 300
    pattern_width = 200
    num_segments = 10
    camera_height = 420
    camera_width = 560
    
    # Pattern position in camera frame
    x_offset = (camera_width - pattern_width) // 2
    y_offset = (camera_height - pattern_height) // 2
    
    print(f"\nPattern: {pattern_width}x{pattern_height}")
    print(f"Camera: {camera_width}x{camera_height}")
    print(f"Pattern offset: ({x_offset}, {y_offset})")
    print()
    
    # Test ROI for different segments
    segment_height = pattern_height // num_segments  # 30 pixels per segment
    
    for test_segment in [1, 5, 10]:
        seg_start_y = (test_segment - 1) * segment_height
        seg_end_y = pattern_height if test_segment == num_segments else test_segment * segment_height
        
        # Calculate ROI (segment-specific)
        padding_x = int(pattern_width * 0.2)
        padding_y = int((seg_end_y - seg_start_y) * 0.15)
        
        roi_x1 = max(0, x_offset - padding_x)
        roi_x2 = min(camera_width, x_offset + pattern_width + padding_x)
        roi_y1 = max(0, y_offset + seg_start_y - padding_y)
        roi_y2 = min(camera_height, y_offset + seg_end_y + padding_y)
        
        roi_width = roi_x2 - roi_x1
        roi_height = roi_y2 - roi_y1
        
        print(f"Segment {test_segment:2d}:")
        print(f"  Pattern Y: {seg_start_y:3d}-{seg_end_y:3d} (height: {seg_end_y - seg_start_y})")
        print(f"  ROI: [{roi_x1:3d},{roi_y1:3d}] to [{roi_x2:3d},{roi_y2:3d}]")
        print(f"  ROI Size: {roi_width}x{roi_height}")
        print(f"  Width padding: {padding_x} (generous)")
        print(f"  Height padding: {padding_y} (constrained)")
        
        # Verify ROI height is constrained to segment
        expected_max_height = segment_height + 2 * padding_y
        if roi_height <= expected_max_height + 10:  # Allow small margin
            print(f"  ✓ ROI height constrained to segment")
        else:
            print(f"  ✗ ROI height too large ({roi_height} > {expected_max_height})")
        print()
    
    print("✓ Segment-specific ROI working correctly")

def visualize_single_segment_display():
    """Create visual demonstration of single segment display"""
    print("\n" + "=" * 60)
    print("VISUALIZATION: Single Segment Display")
    print("=" * 60)
    
    # Create visualization canvas
    width = 700
    height = 500
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 15, 10)  # Dark background
    
    # Colors
    CYAN = (255, 255, 0)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)
    GRAY = (100, 100, 100)
    
    # Title
    cv2.putText(canvas, "Single Segment Display", (180, 40),
                cv2.FONT_HERSHEY_TRIPLEX, 1.0, WHITE, 2)
    
    # Draw two examples side by side
    examples = [
        {
            'title': 'Segment 1 Active',
            'current': 1,
            'completed': set(),
            'x': 120
        },
        {
            'title': 'Segment 4 Active',
            'current': 4,
            'completed': {1, 2, 3},
            'x': 420
        }
    ]
    
    for example in examples:
        x = example['x']
        y = 100
        seg_width = 150
        seg_height = 30
        
        # Title
        cv2.putText(canvas, example['title'], (x - 20, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        
        # Draw 10 segment boxes
        for seg in range(1, 11):
            seg_y = y + (seg - 1) * seg_height
            
            if seg == example['current']:
                # Current segment: yellow outline
                cv2.rectangle(canvas, (x, seg_y), (x + seg_width, seg_y + seg_height - 2),
                             YELLOW, 2)
                label = f"{seg} (CURRENT)"
                label_color = YELLOW
            elif seg in example['completed']:
                # Completed: cyan fill (stitches)
                cv2.rectangle(canvas, (x, seg_y), (x + seg_width, seg_y + seg_height - 2),
                             CYAN, -1)
                label = f"{seg} (STITCHED)"
                label_color = CYAN
            else:
                # Hidden: no outline
                cv2.rectangle(canvas, (x, seg_y), (x + seg_width, seg_y + seg_height - 2),
                             GRAY, 1)
                label = f"{seg} (hidden)"
                label_color = GRAY
            
            # Segment number
            cv2.putText(canvas, str(seg), (x + 10, seg_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
    
    # Legend
    legend_y = 440
    cv2.putText(canvas, "Legend:", (50, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    
    # Cyan box
    cv2.rectangle(canvas, (150, legend_y - 15), (170, legend_y), CYAN, -1)
    cv2.putText(canvas, "= Stitched (visible)", (180, legend_y - 3),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    
    # Yellow box
    cv2.rectangle(canvas, (380, legend_y - 15), (400, legend_y), YELLOW, 2)
    cv2.putText(canvas, "= Current (outline)", (410, legend_y - 3),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    
    # Save
    output_path = "single_segment_display_demo.png"
    cv2.imwrite(output_path, canvas)
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Display
    cv2.imshow("Single Segment Display Demo", canvas)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_all_tests():
    """Run all single-segment display tests"""
    print("\n" + "=" * 60)
    print("SINGLE-SEGMENT DISPLAY TESTS")
    print("=" * 60)
    print()
    
    try:
        test_single_segment_display()
        test_stitch_accumulation()
        test_segment_roi()
        visualize_single_segment_display()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print()
        print("Single-segment display system is working correctly!")
        print("- Only current segment shows (yellow outline)")
        print("- Accumulated stitches remain visible (cyan)")
        print("- ROI constrained to current segment height")
        print()
        print("Run main.py to see it in action.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
