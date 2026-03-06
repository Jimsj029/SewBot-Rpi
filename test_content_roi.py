"""Test Content-Based Segment and ROI Positioning"""

import numpy as np
import cv2

print("="*60)
print("CONTENT-BASED ROI POSITIONING TEST")
print("="*60)

# Simulate stretched pattern (200x420)
pattern_width = 200
pattern_height = 420

# Create a pattern with content only in certain area (simulating real pattern)
# Most patterns don't fill the entire image - they have margins
pattern_alpha = np.zeros((pattern_height, pattern_width), dtype=np.float32)

# Simulate a zigzag pattern that spans from Y=50 to Y=350 (300px of content)
# (similar to original 300px pattern with some margin)
content_start = 50
content_end = 350
for y in range(content_start, content_end):
    # Create zigzag pattern
    for x in range(pattern_width):
        if (y // 10) % 2 == 0:
            if x < pattern_width // 2:
                pattern_alpha[y, x] = 1.0
        else:
            if x >= pattern_width // 2:
                pattern_alpha[y, x] = 1.0

print(f"\n📐 Simulated Pattern:")
print(f"   Full size: {pattern_width}x{pattern_height}")

# Find actual content
pattern_rows_with_content = np.where(np.any(pattern_alpha > 0.1, axis=1))[0]
if len(pattern_rows_with_content) > 0:
    pattern_start_y = pattern_rows_with_content[0]
    pattern_end_y = pattern_rows_with_content[-1]
    pattern_content_height = pattern_end_y - pattern_start_y
    
    print(f"   Content area: Y={pattern_start_y}-{pattern_end_y}")
    print(f"   Content height: {pattern_content_height}px")
else:
    print(f"   ❌ No content found!")
    exit()

# Calculate segment 1 (bottom)
num_segments = 10
segment_height = pattern_content_height // num_segments

reversed_seg = 10  # Segment 1 = reversed 10
seg_start_in_content = (reversed_seg - 1) * segment_height
seg_end_in_content = pattern_content_height

seg_start_y = pattern_start_y + seg_start_in_content
seg_end_y = pattern_start_y + seg_end_in_content

print(f"\n📍 Segment 1 (BOTTOM):")
print(f"   Content-relative: {seg_start_in_content}-{seg_end_in_content}")
print(f"   Absolute position: Y={seg_start_y}-{seg_end_y}")
print(f"   Height: {seg_end_y - seg_start_y}px")

# Calculate ROI
camera_height = 420
camera_width = 560
x_offset = (camera_width - pattern_width) // 2
y_offset = 0

padding_y = int((seg_end_y - seg_start_y) * 0.15)
roi_y1 = max(0, y_offset + seg_start_y - padding_y)
roi_y2 = min(camera_height, y_offset + seg_end_y + padding_y)

print(f"\n🎯 ROI for Segment 1:")
print(f"   Camera Y: {roi_y1}-{roi_y2}")
print(f"   Height: {roi_y2 - roi_y1}px")

# Check if ROI captures content
roi_captures_content = (roi_y1 <= seg_start_y and roi_y2 >= seg_end_y)
content_in_roi_percent = 100 * np.sum(pattern_alpha[roi_y1:roi_y2, :] > 0) / np.sum(pattern_alpha > 0) if np.sum(pattern_alpha > 0) > 0 else 0

print(f"\n🔍 Verification:")
print(f"   ROI captures segment 1: {'✅ YES' if roi_captures_content else '❌ NO'}")
print(f"   Content in ROI: {content_in_roi_percent:.1f}% of total pattern")

# Visualize
vis = np.zeros((pattern_height, pattern_width, 3), dtype=np.uint8)

# Draw full pattern in gray
vis[pattern_alpha > 0] = [100, 100, 100]

# Highlight segment 1 area in yellow
seg1_mask = np.zeros_like(pattern_alpha)
seg1_mask[seg_start_y:seg_end_y, :] = pattern_alpha[seg_start_y:seg_end_y, :]
vis[seg1_mask > 0] = [0, 255, 255]  # Yellow

# Draw ROI bounds
cv2.rectangle(vis, (0, roi_y1), (pattern_width-1, roi_y2), (255, 0, 255), 2)  # Magenta ROI

# Draw content bounds
cv2.line(vis, (0, pattern_start_y), (pattern_width, pattern_start_y), (0, 255, 0), 2)
cv2.line(vis, (0, pattern_end_y), (pattern_width, pattern_end_y), (0, 255, 0), 2)

# Labels
cv2.putText(vis, "Content Area", (5, pattern_start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(vis, "ROI", (5, roi_y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
cv2.putText(vis, "Segment 1", (5, seg_start_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

output_file = "content_based_roi.png"
cv2.imwrite(output_file, vis)
print(f"\n✅ Visualization saved: {output_file}")
print(f"   Gray = Full pattern")
print(f"   Yellow = Segment 1 (bottom)")
print(f"   Magenta = ROI box")
print(f"   Green = Content bounds")

print(f"\n" + "="*60)
print("SUMMARY")
print("="*60)

if roi_captures_content and content_in_roi_percent > 5:
    print("✅ ROI CORRECTLY POSITIONED!")
    print("\nThe ROI captures actual pattern content at the bottom.")
    print("Stitches detected in this ROI will mark segment 1 as complete.")
else:
    print("❌ ROI positioning issue")
    print(f"   ROI Y: {roi_y1}-{roi_y2}")
    print(f"   Content Y: {pattern_start_y}-{pattern_end_y}")
    print(f"   Segment 1 Y: {seg_start_y}-{seg_end_y}")
