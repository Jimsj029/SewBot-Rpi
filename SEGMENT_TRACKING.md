# Segment-Based Pattern Tracking System

## Overview

The SewBot now uses an advanced segment-based tracking system to provide precise feedback during pattern stitching. Instead of tracking the entire pattern as one continuous area, the pattern is divided into **10 discrete segments**.

## How It Works

### Pattern Segmentation

```
Pattern divided into 10 vertical regions:

[1]  ← Start here
[2]
[3]
[4]
[5]
[6]
[7]
[8]
[9]
[10] ← End here
```

Each segment represents approximately 10% of the total pattern height.

### Segment States

At any given time, each segment can be in one of these states:

1. **Completed** (Cyan) - Segment has been successfully completed (≥70% coverage)
2. **Current Active** (Yellow, pulsing) - The segment you should be working on now
3. **Upcoming** (Gray) - Segments that haven't been started yet
4. **Deviation** (Red, flashing) - Wrong segment detected

### Progress Tracking

The system maintains:

```python
completed_segments = {1, 2, 3}  # Set of completed segment numbers
current_segment = 4              # Current active segment
```

- **Completed segments remain stored** - Even after moving to the next segment
- **Active ROI only analyzes current segment** - Focuses detection on where you should be sewing
- **Deviation detection** - Alerts if stitching occurs in wrong segments

## Features

### 1. Sequential Progress
- You must complete segments in order (1 → 2 → 3 → ... → 10)
- A segment is marked complete when ≥70% of its pattern is covered with stitches
- Progress automatically advances to the next segment when current is complete

### 2. Deviation Detection

The system checks if the needle deviates to wrong segments:

```python
# Example scenarios:
Current segment: 4
Completed segments: {1, 2, 3}

✓ Stitch in segment 4 → OK (current segment)
✓ Stitch in segment 2 → OK (already completed)
✗ Stitch in segment 6 → DEVIATION! (skipping segment 4-5)
```

When deviation is detected:
- Warning banner displays: "⚠ Skip detected! Complete Segment X first"
- Wrong segment flashes red on the pattern overlay
- Combo counter resets

### 3. Visual Feedback

#### Pattern Overlay Colors:
- **Cyan** - Completed segments
- **Yellow** - Current segment (pulsing to draw attention)
- **Gray** - Upcoming segments
- **Red** (flashing) - Deviation detected

#### Progress Bar:
- 10 segment boxes arranged in 2 rows (5 per row)
- Each box shows its segment number
- Visual indicator shows current segment and completed count
- Real-time percentage display

### 4. Score Calculation

```python
Score = (completed_segments / total_segments) × 100
```

Example: 7 segments completed = 70 points

## Implementation Details

### Key Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `num_segments` | int | Total number of segments (10) |
| `current_segment` | int | Current active segment (1-10) |
| `completed_segments` | set | Set of completed segment numbers |
| `segment_masks` | dict | Dictionary mapping segment number to mask |
| `deviation_detected` | bool | Whether wrong segment was detected |
| `deviation_segment` | int | Which wrong segment was detected |

### Key Methods

#### `create_segment_masks(pattern_alpha)`
Divides the pattern into 10 vertical regions and creates a mask for each segment.

#### `create_realtime_pattern(overlay, alpha)`
Generates the colored pattern overlay based on segment states:
- Colors completed segments cyan
- Colors current segment yellow with pulse effect
- Colors upcoming segments gray
- Flashes deviation segments red

#### `update_game_stats(...)`
Core tracking logic:
1. Detects which segments current stitches intersect
2. Checks for deviations (stitching in wrong segments)
3. Calculates coverage for each segment
4. Marks segments complete when threshold reached
5. Automatically advances to next segment

### Completion Threshold

```python
segment_completion_threshold = 0.70  # 70% coverage required
```

A segment is marked complete when 70% or more of its pattern pixels are covered by detected stitches. This threshold balances accuracy requirements with user-friendliness.

## Benefits

1. **Better Progress Tracking** - Clear visual feedback on exactly where you are
2. **Deviation Prevention** - Immediate feedback if you skip ahead or go backwards
3. **Structured Learning** - Encourages systematic, sequential stitching
4. **Precise Scoring** - Score reflects actual segment completion
5. **Prevents Backtracking** - Completed segments remain marked even if you return to them

## User Experience

1. **Start**: Begin stitching segment 1 (yellow, pulsing)
2. **Progress**: As you sew, the pattern turns cyan in completed areas
3. **Advance**: When segment 1 reaches 70% coverage, segment 2 becomes active
4. **Complete**: Continue until all 10 segments are completed (100%)

### Warning System

If you accidentally start sewing segment 6 while still on segment 4:
```
⚠ Skip detected! Complete Segment 4 first
```

The system guides you back to the correct segment.

## Technical Notes

- Segments are divided vertically (top to bottom)
- Each segment mask is cached for performance
- Deviation detection runs every frame when stitches are detected
- Progress calculation is based on actual pixel coverage
- Segment advancement is automatic but only moves forward

## Configuration

To adjust the number of segments, modify `pattern_mode.py`:

```python
self.num_segments = 10  # Change to desired number of segments
```

To adjust completion threshold:

```python
segment_completion_threshold = 0.70  # 0.0 to 1.0 (70% = 0.70)
```

## Future Enhancements

Potential improvements:
- Horizontal segments for left-right patterns
- Custom segment shapes for complex patterns
- Per-segment time tracking
- Segment-specific difficulty ratings
- Bonus points for completing segments without deviations
