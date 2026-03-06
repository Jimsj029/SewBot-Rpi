# Segment-Based Pattern Tracking - Quick Reference

## 🎯 At a Glance

### Pattern Division
```
┌──────────┐
│    1     │  0-10%
├──────────┤
│    2     │  10-20%
├──────────┤
│    3     │  20-30%
├──────────┤
│    4     │  30-40%
├──────────┤
│    5     │  40-50%
├──────────┤
│    6     │  50-60%
├──────────┤
│    7     │  60-70%
├──────────┤
│    8     │  70-80%
├──────────┤
│    9     │  80-90%
├──────────┤
│   10     │  90-100%
└──────────┘
```

## 🎨 Color Guide

| Color | Meaning |
|-------|---------|
| 🟦 **Cyan** | Completed (≥70% coverage) |
| 🟨 **Yellow** | Current segment (sew here!) |
| ⬜ **Gray** | Upcoming (not started) |
| 🟥 **Red** | Deviation (wrong segment!) |

## 📋 Rules

1. **Start at Segment 1** - Always begin at the top
2. **Complete in Order** - Can't skip segments
3. **70% = Complete** - Segment needs 70% coverage
4. **Auto-Advance** - Moves to next when current done
5. **No Backtracking** - Completed stays completed

## ⚠️ Warnings

### "Skip detected! Complete Segment X first"
- **Cause**: Stitching ahead of current segment
- **Action**: Return to current segment (yellow)
- **No Penalty**: Just guidance

## 📊 Progress Bar

```
┌──┬──┬──┬──┬──┐
│ 1│ 2│ 3│ 4│ 5│  ← Row 1: Segments 1-5
├──┼──┼──┼──┼──┤
│ 6│ 7│ 8│ 9│10│  ← Row 2: Segments 6-10
└──┴──┴──┴──┴──┘
```

- Each box = 1 segment = 10% progress
- Completed boxes = cyan
- Current box = yellow (pulsing)
- Upcoming boxes = gray

## 📈 Score Calculation

```
Score = (Completed Segments / 10) × 100

Examples:
  3 completed → 30 points
  7 completed → 70 points
  10 completed → 100 points (Perfect!)
```

## 🔍 What System Tracks

```python
completed_segments = {1, 2, 3}  # Which segments are done
current_segment = 4              # Where you should be now
```

## ✅ Valid Stitching Locations

| Stitch Location | Result |
|-----------------|--------|
| Current segment | ✅ Perfect! |
| Completed segment | ✅ OK |
| Upcoming segment | ❌ Deviation! |

## 🎮 Example Playthrough

```
START
├─ Segment 1 (yellow) → Sew here
├─ 70% reached → ✓ Complete! (cyan)
├─ Auto-advance to Segment 2
├─ Segment 2 (yellow) → Sew here
├─ 70% reached → ✓ Complete! (cyan)
├─ Auto-advance to Segment 3
│  ...continue...
└─ Segment 10 complete → 🎉 100%!
```

## 🛠️ Troubleshooting

### Red flashing pattern?
→ You're stitching the wrong segment. Look for yellow segment.

### Progress stuck?
→ Current segment needs more coverage (70% min).

### Can't advance?
→ Ensure current segment (yellow) is fully sewn.

## 💡 Tips

1. **Focus on Yellow** - Only sew the yellow segment
2. **Complete Thoroughly** - Get full 70%+ before moving on
3. **Follow Order** - Don't skip around
4. **Watch Progress Bar** - Shows exactly where you are
5. **Ignore Red** - Just return to yellow segment

## 📐 Technical Specs

- **Total Segments**: 10
- **Division Type**: Vertical (top to bottom)
- **Completion Threshold**: 70%
- **Progress Range**: 0% to 100%
- **Segment Height**: 10% of pattern each

## 🚀 Quick Start

1. Select a level
2. Start at top (Segment 1 is yellow)
3. Sew the yellow segment
4. When it turns cyan, next segment becomes yellow
5. Repeat until all 10 are cyan
6. 100% = Level Complete!

## 📚 More Info

- **Full Documentation**: `SEGMENT_TRACKING.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Run Tests**: `python test_segment_tracking.py`
- **Visualization**: `python visualize_segments.py`

---

**Remember: Yellow = Current | Cyan = Done | Gray = Next**
