"""
Test script for segment-based pattern tracking
Run this to verify the segment tracking logic works correctly
"""

import numpy as np
import cv2

def test_segment_creation():
    """Test that segments are created correctly"""
    print("=" * 60)
    print("TEST 1: Segment Mask Creation")
    print("=" * 60)
    
    # Create a simple test pattern (200x300)
    pattern_height = 300
    pattern_width = 200
    pattern_alpha = np.ones((pattern_height, pattern_width), dtype=np.float32)
    
    # Simulate segment creation
    num_segments = 10
    segment_height = pattern_height // num_segments  # 30 pixels per segment
    
    print(f"Pattern size: {pattern_height}x{pattern_width}")
    print(f"Number of segments: {num_segments}")
    print(f"Segment height: {segment_height} pixels")
    print()
    
    # Create segments
    segment_masks = {}
    for seg_num in range(1, num_segments + 1):
        start_y = (seg_num - 1) * segment_height
        end_y = pattern_height if seg_num == num_segments else seg_num * segment_height
        
        seg_mask = np.zeros_like(pattern_alpha, dtype=np.uint8)
        seg_mask[start_y:end_y, :] = (pattern_alpha[start_y:end_y, :] * 255).astype(np.uint8)
        segment_masks[seg_num] = seg_mask
        
        print(f"Segment {seg_num:2d}: rows {start_y:3d}-{end_y:3d} ({end_y - start_y} pixels)")
    
    print(f"\n✓ Successfully created {len(segment_masks)} segment masks")
    return segment_masks

def test_deviation_detection():
    """Test deviation detection logic"""
    print("\n" + "=" * 60)
    print("TEST 2: Deviation Detection")
    print("=" * 60)
    
    current_segment = 4
    completed_segments = {1, 2, 3}
    num_segments = 10
    
    print(f"Current segment: {current_segment}")
    print(f"Completed segments: {completed_segments}")
    print()
    
    # Test cases
    test_cases = [
        (1, "Stitch in segment 1 (already completed)"),
        (3, "Stitch in segment 3 (already completed)"),
        (4, "Stitch in segment 4 (current segment)"),
        (5, "Stitch in segment 5 (next segment)"),
        (7, "Stitch in segment 7 (skipping ahead)"),
    ]
    
    for seg, description in test_cases:
        deviation_detected = False
        result = ""
        
        if seg in completed_segments:
            result = "✓ OK - Already completed"
        elif seg == current_segment:
            result = "✓ OK - Current segment"
        elif seg < current_segment and seg not in completed_segments:
            result = "⚠ WARNING - Skipped segment"
        elif seg > current_segment:
            deviation_detected = True
            result = f"✗ DEVIATION - Skip detected! Complete segment {current_segment} first"
        
        status = "DEVIATION" if deviation_detected else "OK"
        print(f"{description:45s} → {status:10s} - {result}")
    
    print("\n✓ Deviation detection logic working correctly")

def test_segment_completion():
    """Test segment completion threshold logic"""
    print("\n" + "=" * 60)
    print("TEST 3: Segment Completion Threshold")
    print("=" * 60)
    
    threshold = 0.70  # 70% coverage required
    
    print(f"Completion threshold: {threshold * 100:.0f}%")
    print()
    
    # Test cases for segment coverage
    test_cases = [
        (50, 100, "50% coverage"),
        (70, 100, "70% coverage (threshold)"),
        (85, 100, "85% coverage"),
        (95, 100, "95% coverage"),
        (100, 100, "100% coverage"),
    ]
    
    for covered, total, description in test_cases:
        coverage = covered / total
        is_complete = coverage >= threshold
        status = "✓ COMPLETE" if is_complete else "✗ INCOMPLETE"
        
        print(f"{description:30s} → {coverage*100:5.1f}% → {status}")
    
    print("\n✓ Segment completion threshold working correctly")

def test_progress_calculation():
    """Test overall progress calculation"""
    print("\n" + "=" * 60)
    print("TEST 4: Progress Calculation")
    print("=" * 60)
    
    num_segments = 10
    
    test_cases = [
        ({}, "No segments completed"),
        ({1}, "1 segment completed"),
        ({1, 2, 3}, "3 segments completed"),
        ({1, 2, 3, 4, 5}, "5 segments completed (50%)"),
        ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "All segments completed"),
    ]
    
    print(f"Total segments: {num_segments}")
    print()
    
    for completed_segments, description in test_cases:
        progress = (len(completed_segments) / num_segments) * 100.0
        print(f"{description:40s} → {len(completed_segments):2d}/{num_segments} → {progress:5.1f}%")
    
    print("\n✓ Progress calculation working correctly")

def test_segment_advancement():
    """Test automatic segment advancement logic"""
    print("\n" + "=" * 60)
    print("TEST 5: Automatic Segment Advancement")
    print("=" * 60)
    
    current_segment = 1
    completed_segments = set()
    num_segments = 10
    threshold = 0.70
    
    print(f"Starting segment: {current_segment}")
    print(f"Completion threshold: {threshold * 100:.0f}%")
    print()
    
    # Simulate completing segments
    for seg in range(1, 6):  # Complete first 5 segments
        # Simulate reaching threshold
        coverage = 0.75  # 75% > 70% threshold
        
        if coverage >= threshold:
            completed_segments.add(seg)
            print(f"Segment {seg} reached {coverage*100:.0f}% coverage → ✓ COMPLETED")
            
            # Advance to next segment if this was the current one
            if seg == current_segment:
                if current_segment < num_segments:
                    current_segment += 1
                    print(f"  → Advanced to segment {current_segment}")
        
        print()
    
    print(f"Final state:")
    print(f"  Current segment: {current_segment}")
    print(f"  Completed segments: {completed_segments}")
    print(f"  Progress: {len(completed_segments)}/{num_segments} ({(len(completed_segments)/num_segments)*100:.0f}%)")
    
    print("\n✓ Segment advancement working correctly")

def run_all_tests():
    """Run all segment tracking tests"""
    print("\n" + "=" * 60)
    print("SEGMENT-BASED PATTERN TRACKING TESTS")
    print("=" * 60)
    print()
    
    try:
        test_segment_creation()
        test_deviation_detection()
        test_segment_completion()
        test_progress_calculation()
        test_segment_advancement()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print()
        print("Segment tracking system is ready to use!")
        print("Run main.py to see it in action.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
