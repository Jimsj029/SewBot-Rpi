"""Test Stretched Pattern and ROI Alignment"""

print("="*60)
print("STRETCHED PATTERN ALIGNMENT TEST")
print("="*60)

# Pattern dimensions AFTER stretching
stretched_width = 200
stretched_height = 420  # Full camera height

# Camera dimensions
camera_width = 560
camera_height = 420

# Pattern position
x_offset = (camera_width - stretched_width) // 2
y_offset = 0  # Edge-to-edge from top

print(f"\n📐 Stretched Pattern:")
print(f"   Size: {stretched_width}x{stretched_height}")
print(f"   Position: X={x_offset}, Y={y_offset}")
print(f"   Covers: X={x_offset}-{x_offset+stretched_width}, Y={y_offset}-{y_offset+stretched_height}")

# Segment 1 (bottom) in stretched coordinates
num_segments = 10
segment_height = stretched_height // num_segments  # 42 pixels per segment

# Segment 1 = bottom
reversed_seg = num_segments - 1 + 1  # 10
seg_start_y = (reversed_seg - 1) * segment_height  # 378
seg_end_y = stretched_height  # 420

print(f"\n📍 Segment 1 (BOTTOM) in stretched pattern:")
print(f"   Pattern Y: {seg_start_y}-{seg_end_y}")
print(f"   Height: {seg_end_y - seg_start_y} pixels")

# Calculate ROI
padding_y = int((seg_end_y - seg_start_y) * 0.15)
roi_y1 = max(0, y_offset + seg_start_y - padding_y)
roi_y2 = min(camera_height, y_offset + seg_end_y + padding_y)

print(f"\n🎯 ROI for Segment 1:")
print(f"   Camera Y: {roi_y1}-{roi_y2}")
print(f"   Expected: ~378-420 (at bottom)")

# Check alignment
pattern_bottom = y_offset + stretched_height
roi_bottom = roi_y2

print(f"\n🔍 Alignment Check:")
print(f"   Pattern bottom: {pattern_bottom}")
print(f"   ROI bottom: {roi_bottom}")
print(f"   Pattern top: {y_offset}")
print(f"   ROI for seg1 top: {roi_y1}")

if roi_y1 >= seg_start_y and roi_y2 <= camera_height:
    print(f"   ✅ ROI is positioned at bottom of pattern!")
else:
    print(f"   ❌ ROI positioning issue")

# Test all segments
print(f"\n" + "="*60)
print("ALL SEGMENTS IN STRETCHED PATTERN")
print("="*60)

for seg in [1, 2, 5, 10]:
    reversed_seg = num_segments - seg + 1
    start_y = (reversed_seg - 1) * segment_height
    end_y = stretched_height if reversed_seg == num_segments else reversed_seg * segment_height
    
    position = "BOTTOM" if seg == 1 else "TOP" if seg == 10 else "MIDDLE"
    print(f"Segment {seg:2d} ({position:6s}): Y={start_y:3d}-{end_y:3d} (height: {end_y-start_y})")

print(f"\n✅ All segments calculated from stretched 420px height")
print(f"   Each segment: {segment_height} pixels tall")
print(f"   Pattern edge-to-edge: 0-{stretched_height}")
