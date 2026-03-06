"""Quick Test - Print ROI and Overlay Positions"""

print("="*60)
print("EXPECTED COORDINATES TEST")
print("="*60)

# Constants
camera_width = 560
camera_height = 420
overlay_w = 200
overlay_h = 300
uniform_width = 200
uniform_height = 300

# Calculate overlay position
x_offset = (camera_width - overlay_w) // 2
y_offset = (camera_height - overlay_h) // 2

print(f"\n📐 Pattern Overlay:")
print(f"   Size: {overlay_w}x{overlay_h}")
print(f"   Position in camera: ({x_offset},{y_offset})")
print(f"   Bounds: X={x_offset}-{x_offset+overlay_w}, Y={y_offset}-{y_offset+overlay_h}")

# Segment 1 (bottom) - pattern coordinates
num_segments = 10
segment_height = uniform_height // num_segments
reversed_seg = num_segments - 1 + 1  # segment 1
seg_start_y = (reversed_seg - 1) * segment_height
seg_end_y = uniform_height if reversed_seg == num_segments else reversed_seg * segment_height

print(f"\n📍 Segment 1 (BOTTOM):")
print(f"   Pattern coords: Y={seg_start_y}-{seg_end_y}")
print(f"   Height: {seg_end_y - seg_start_y} pixels")

# Calculate ROI in camera coordinates
padding_x = int(overlay_w * 0.2)
padding_y = int((seg_end_y - seg_start_y) * 0.15)

roi_x1 = max(0, x_offset - padding_x)
roi_x2 = min(camera_width, x_offset + overlay_w + padding_x)
roi_y1 = max(0, y_offset + seg_start_y - padding_y)
roi_y2 = min(camera_height, y_offset + seg_end_y + padding_y)

print(f"\n🎯 ROI for Segment 1:")
print(f"   Padding: X=±{padding_x}, Y=±{padding_y}")
print(f"   Camera coords: X={roi_x1}-{roi_x2}, Y={roi_y1}-{roi_y2}")
print(f"   Width: {roi_x2-roi_x1}, Height: {roi_y2-roi_y1}")

# Check overlap
overlay_y_start = y_offset
overlay_y_end = y_offset + overlay_h
roi_overlaps_overlay = (roi_y1 < overlay_y_end and roi_y2 > overlay_y_start)

print(f"\n🔍 Overlap Check:")
print(f"   Overlay Y range: {overlay_y_start}-{overlay_y_end}")
print(f"   ROI Y range: {roi_y1}-{roi_y2}")
print(f"   ROI overlaps overlay: {'✅ YES' if roi_overlaps_overlay else '❌ NO'}")

if roi_overlaps_overlay:
    overlap_start = max(roi_y1, overlay_y_start)
    overlap_end = min(roi_y2, overlay_y_end)
    overlap_height = overlap_end - overlap_start
    print(f"   Overlap region: Y={overlap_start}-{overlap_end} ({overlap_height} pixels)")
    
    # Check if ROI is mostly within overlay
    roi_height = roi_y2 - roi_y1
    overlap_percent = (overlap_height / roi_height) * 100
    print(f"   Overlap: {overlap_percent:.1f}% of ROI height")
    
    if overlap_percent > 80:
        print(f"   ✅ ROI is well-positioned on overlay!")
    else:
        print(f"   ⚠️  ROI extends significantly beyond overlay")

print(f"\n" + "="*60)
print("CONCLUSION")
print("="*60)

if roi_overlaps_overlay and overlap_percent > 80:
    print("✅ ROI positioning is CORRECT")
    print("\nThe ROI should cover segment 1 at the bottom of the pattern.")
    print("If it's not visible in the app, the issue may be:")
    print("  1. Camera frame positioning")
    print("  2. Drawing order")
    print("  3. Visual alignment perception")
else:
    print("❌ ROI positioning has issues")
    print(f"\nROI Y: {roi_y1}-{roi_y2}")
    print(f"Pattern Y: {overlay_y_start}-{overlay_y_end}")
