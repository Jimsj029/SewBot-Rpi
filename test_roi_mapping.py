"""Test ROI to Pattern Coordinate Mapping"""

import numpy as np
import cv2

print("=" * 60)
print("ROI TO PATTERN COORDINATE MAPPING TEST")
print("=" * 60)

# Pattern dimensions (uniform size)
uniform_width = 200
uniform_height = 300

# Camera dimensions
camera_width = 560
camera_height = 420

# Pattern overlay offset (centered in camera)
x_offset = (camera_width - uniform_width) // 2
y_offset = (camera_height - uniform_height) // 2

print(f"\n📐 Setup:")
print(f"   Pattern size: {uniform_width}x{uniform_height}")
print(f"   Camera size: {camera_width}x{camera_height}")
print(f"   Overlay offset: ({x_offset}, {y_offset})")
print(f"   Pattern in camera: X={x_offset}-{x_offset+uniform_width}, Y={y_offset}-{y_offset+uniform_height}")

# Test Case: Segment 1 (BOTTOM) ROI
print(f"\n" + "=" * 60)
print("TEST: Segment 1 (BOTTOM) ROI Mapping")
print("=" * 60)

# Segment 1 is at bottom (Y=270-300 in pattern coords)
seg1_pattern_y1 = 270
seg1_pattern_y2 = 300

# Calculate ROI in camera coordinates
padding_y = 5
roi_y1 = y_offset + seg1_pattern_y1 - padding_y
roi_y2 = y_offset + seg1_pattern_y2 + padding_y

# Full width ROI
padding_x = 40
roi_x1 = max(0, x_offset - padding_x)
roi_x2 = min(camera_width, x_offset + uniform_width + padding_x)

print(f"\n📍 Segment 1 ROI (camera coords):")
print(f"   ROI: X={roi_x1}-{roi_x2}, Y={roi_y1}-{roi_y2}")
print(f"   Expected pattern mapping: Y={seg1_pattern_y1}-{seg1_pattern_y2}")

# Simulate detection in ROI
roi_h = roi_y2 - roi_y1
roi_w = roi_x2 - roi_x1
roi_combined_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

# Add some "stitches" in the middle of ROI
stitch_y = roi_h // 2
stitch_x = roi_w // 2
cv2.circle(roi_combined_mask, (stitch_x, stitch_y), 10, 255, -1)

print(f"\n✏️  Simulated detection:")
print(f"   ROI size: {roi_w}x{roi_h}")
print(f"   Stitch at ROI coords: ({stitch_x}, {stitch_y})")

# ========== OLD METHOD (WRONG) ==========
stitches_pattern_OLD = cv2.resize(roi_combined_mask, (uniform_width, uniform_height))
covered_pixels_OLD = np.sum(stitches_pattern_OLD > 0)
covered_y_coords_OLD = np.where(stitches_pattern_OLD > 0)[0]
if len(covered_y_coords_OLD) > 0:
    min_y_OLD = covered_y_coords_OLD.min()
    max_y_OLD = covered_y_coords_OLD.max()
else:
    min_y_OLD = max_y_OLD = 0

print(f"\n❌ OLD METHOD (resize to full pattern):")
print(f"   Pixels covered: {covered_pixels_OLD}")
print(f"   Y range: {min_y_OLD}-{max_y_OLD}")
print(f"   Expected: {seg1_pattern_y1}-{seg1_pattern_y2}")
print(f"   ⚠️  Spreads across ENTIRE pattern height!")

# ========== NEW METHOD (CORRECT) ==========
stitches_pattern_NEW = np.zeros((uniform_height, uniform_width), dtype=np.uint8)

# Map ROI position back to pattern coordinates
pattern_x1 = max(0, roi_x1 - x_offset)
pattern_y1 = max(0, roi_y1 - y_offset)
pattern_x2 = min(uniform_width, roi_x2 - x_offset)
pattern_y2 = min(uniform_height, roi_y2 - y_offset)

# Resize ROI mask to match pattern region size
pattern_region_w = pattern_x2 - pattern_x1
pattern_region_h = pattern_y2 - pattern_y1

if pattern_region_w > 0 and pattern_region_h > 0:
    roi_resized = cv2.resize(roi_combined_mask, (pattern_region_w, pattern_region_h))
    stitches_pattern_NEW[pattern_y1:pattern_y2, pattern_x1:pattern_x2] = roi_resized

covered_pixels_NEW = np.sum(stitches_pattern_NEW > 0)
covered_y_coords_NEW = np.where(stitches_pattern_NEW > 0)[0]
if len(covered_y_coords_NEW) > 0:
    min_y_NEW = covered_y_coords_NEW.min()
    max_y_NEW = covered_y_coords_NEW.max()
else:
    min_y_NEW = max_y_NEW = 0

print(f"\n✅ NEW METHOD (correct mapping):")
print(f"   Mapped to pattern: X={pattern_x1}-{pattern_x2}, Y={pattern_y1}-{pattern_y2}")
print(f"   Pixels covered: {covered_pixels_NEW}")
print(f"   Y range: {min_y_NEW}-{max_y_NEW}")
print(f"   Expected: ~{seg1_pattern_y1}-{seg1_pattern_y2}")

if min_y_NEW >= seg1_pattern_y1 - 10 and max_y_NEW <= seg1_pattern_y2 + 10:
    print(f"   ✅ Correctly mapped to BOTTOM segment!")
else:
    print(f"   ❌ Mapping error!")

# ========== VISUALIZATION ==========
print(f"\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

# Create side-by-side comparison
vis = np.zeros((uniform_height, uniform_width * 2 + 50, 3), dtype=np.uint8)

# Left: OLD method (wrong)
old_rgb = np.zeros((uniform_height, uniform_width, 3), dtype=np.uint8)
old_rgb[stitches_pattern_OLD > 0] = [0, 255, 255]  # Cyan
vis[:, :uniform_width] = old_rgb

# Middle: separator
cv2.line(vis, (uniform_width + 25, 0), (uniform_width + 25, uniform_height), (255, 255, 255), 2)

# Right: NEW method (correct)
new_rgb = np.zeros((uniform_height, uniform_width, 3), dtype=np.uint8)
new_rgb[stitches_pattern_NEW > 0] = [0, 255, 255]  # Cyan
vis[:, uniform_width + 50:] = new_rgb

# Draw segment 1 boundary on both
seg1_line = seg1_pattern_y1
cv2.line(vis, (0, seg1_line), (vis.shape[1], seg1_line), (0, 0, 255), 2)
cv2.putText(vis, "Seg 1 top", (5, seg1_line - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# Labels
cv2.putText(vis, "OLD (WRONG)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.putText(vis, "NEW (CORRECT)", (uniform_width + 60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

output_file = "roi_mapping_comparison.png"
cv2.imwrite(output_file, vis)
print(f"✅ Comparison saved: {output_file}")
print(f"   Left: OLD method (stitches spread everywhere)")
print(f"   Right: NEW method (stitches only at bottom)")

# ========== SUMMARY ==========
print(f"\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if min_y_NEW >= seg1_pattern_y1 - 10 and max_y_NEW <= seg1_pattern_y2 + 10:
    print("✅ FIX SUCCESSFUL!")
    print("\nThe new mapping correctly places stitches only in the")
    print("detected ROI region (segment 1 at bottom), not across")
    print("the entire pattern.")
    print("\nThis fixes the issue where the whole pattern turned cyan!")
else:
    print("❌ Fix needs adjustment")
    print(f"   Expected Y: {seg1_pattern_y1}-{seg1_pattern_y2}")
    print(f"   Actual Y: {min_y_NEW}-{max_y_NEW}")

print("\nTest with main.py to verify the fix!")
