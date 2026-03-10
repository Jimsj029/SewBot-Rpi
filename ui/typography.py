"""Shared typography helpers for the SewBot OpenCV UI.

These helpers keep text readable on smaller displays while preserving
consistent visual style across all screens.
"""

import cv2

BASE_WIDTH = 1024
BASE_HEIGHT = 600

# DUPLEX is cleaner than TRIPLEX on low-resolution displays.
FONT_MAIN = cv2.FONT_HERSHEY_DUPLEX
FONT_DISPLAY = cv2.FONT_HERSHEY_DUPLEX
FONT_COMPACT = cv2.FONT_HERSHEY_SIMPLEX


def screen_scale(width, height, min_scale=0.9, max_scale=1.2):
    """Compute a stable UI scale factor from the current render size."""
    if width <= 0 or height <= 0:
        return 1.0
    raw = min(float(width) / BASE_WIDTH, float(height) / BASE_HEIGHT)
    return max(min_scale, min(max_scale, raw))


def text_scale(base_scale, width, height, floor=None, ceiling=None):
    """Scale text size by resolution, with optional min/max clamps."""
    scale = float(base_scale) * screen_scale(width, height)
    if floor is not None:
        scale = max(float(floor), scale)
    if ceiling is not None:
        scale = min(float(ceiling), scale)
    return scale


def text_thickness(base_thickness, width, height, min_thickness=1, max_thickness=6):
    """Scale stroke thickness with resolution and clamp to sane limits."""
    scale = screen_scale(width, height, min_scale=0.92, max_scale=1.25)
    thick = int(round(float(base_thickness) * scale))
    return max(min_thickness, min(max_thickness, thick))


def get_text_size(text, font, scale, thickness):
    """Wrapper around cv2.getTextSize with consistent type handling."""
    return cv2.getTextSize(str(text), font, float(scale), int(thickness))


def fit_text_scale(text, font, target_width, scale, thickness, min_scale=0.45):
    """Shrink text scale until it fits the target width."""
    fitted = float(scale)
    limit = max(1, int(target_width))

    for _ in range(48):
        (text_w, _), _ = get_text_size(text, font, fitted, thickness)
        if text_w <= limit or fitted <= min_scale:
            break
        fitted -= 0.02

    return max(float(min_scale), fitted)


def center_text_x(text, font, scale, thickness, left, width):
    """Return x-position that horizontally centers text within a rectangle."""
    (text_w, _), _ = get_text_size(text, font, scale, thickness)
    return int(left + (width - text_w) // 2)


def draw_text(
    img,
    text,
    x,
    y,
    scale,
    color,
    thickness,
    font=FONT_MAIN,
    outline_color=(0, 0, 0),
    outline_extra=2,
    line_type=cv2.LINE_AA,
):
    """Draw outlined anti-aliased text for reliable readability."""
    x = int(round(x))
    y = int(round(y))
    thickness = max(1, int(round(thickness)))
    outline_extra = max(1, int(round(outline_extra)))
    outline_thickness = thickness + outline_extra

    cv2.putText(
        img,
        str(text),
        (x + 1, y + 1),
        font,
        float(scale),
        outline_color,
        outline_thickness,
        line_type,
    )
    cv2.putText(
        img,
        str(text),
        (x, y),
        font,
        float(scale),
        color,
        thickness,
        line_type,
    )
