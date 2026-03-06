"""Debug get_current_segment_roi Logic"""

def get_current_segment_roi_test(current_segment, num_segments=10, pattern_height=300):
    """Test the segment ROI calculation"""
    segment_height = pattern_height // num_segments
    
    # Reverse the segment mapping (1=bottom, 10=top)
    reversed_seg = num_segments - current_segment + 1
    start_y = (reversed_seg - 1) * segment_height
    end_y = pattern_height if reversed_seg == num_segments else reversed_seg * segment_height
    
    return (start_y, end_y)

print("="*60)
print("TEST get_current_segment_roi LOGIC")
print("="*60)

pattern_height = 300
num_segments = 10

for seg in [1, 2, 5, 10]:
    result = get_current_segment_roi_test(seg, num_segments, pattern_height)
    if result:
        start_y, end_y = result
        height = end_y - start_y
        position = "BOTTOM" if seg == 1 else "TOP" if seg == 10 else "MIDDLE"
        print(f"Segment {seg:2d} ({position:6s}): Y={start_y:3d}-{end_y:3d} (height: {height})")
    else:
        print(f"Segment {seg:2d}: None")

print("\nExpected:")
print("  Segment 1 (bottom): Y=270-300")
print("  Segment 10 (top):   Y=0-30")

# Check reversed_seg values
print("\n" + "="*60)
print("REVERSED MAPPING")
print("="*60)

for seg in range(1, 11):
    reversed_seg = num_segments - seg + 1
    print(f"current_segment={seg:2d} → reversed_seg={reversed_seg:2d}")

print("\nExpected: Segment 1 → reversed_seg 10 (bottom rows)")
print("          Segment 10 → reversed_seg 1 (top rows)")
