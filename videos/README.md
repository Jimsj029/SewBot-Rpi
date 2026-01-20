# Tutorial Videos

This folder contains the tutorial videos for the SewBot application.

## Folder Structure

```
videos/
└── sewing-set-up/
    ├── step1.mov (or step1.mp4)
    ├── step2.mov (or step2.mp4)
    ├── step3.mov (or step3.mp4)
    ├── step4.mov (or step4.mp4)
    └── step5.mov (or step5.mp4)
```

## Tutorial System

The tutorial system shows 5 sequential videos demonstrating how to set up the sewing machine.

### Video Requirements

- **File names:** Must be named `step1` through `step5`
- **File formats:** Supports both `.mov` and `.mp4` extensions
- **Location:** Place all videos in the `videos/sewing-set-up/` folder

### User Interface

When the tutorial starts:
- **Step Indicator:** Shows current progress (e.g., "Step 1 of 5") at the top
- **SKIP Button:** (Left side) Skips all remaining tutorial videos and proceeds to mode selection
- **NEXT Button:** (Right side) Advances to the next tutorial video
  - On the final video (Step 5), this button changes to **DONE**
  - Clicking DONE completes the tutorial and proceeds to mode selection

### Adding Tutorial Videos

1. Record or prepare your tutorial videos
2. Name them sequentially: `step1.mov`, `step2.mov`, etc.
3. Place them in the `videos/sewing-set-up/` folder
4. The application will automatically detect and play them in order

### Notes

- If a video file is missing, the system will show a placeholder and continue
- Videos can be either `.mov` or `.mp4` format
- The tutorial can be replayed anytime from the mode selection screen
