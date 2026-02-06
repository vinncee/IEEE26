"""Detect whether hands are actively signing in a landmark window.

Works with real MediaPipe data where each hand is a list of 21 keypoints,
each keypoint being [x, y, z] normalised to 0-1.

Key indices used:
    0  = wrist
    9  = middle-finger MCP (centre of palm)
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.cv.types import LandmarkWindow

# ── MediaPipe hand-keypoint indices ────────────────────────────────────────
WRIST = 0
MIDDLE_MCP = 9

# ── Thresholds (all in normalised 0-1 coordinate space) ───────────────────
# Average per-frame wrist displacement that counts as "moving"
MOTION_THRESHOLD = 0.003          # ~0.3 % of frame dimension per frame
# Minimum fraction of frames that must have hand data
MIN_HAND_COVERAGE = 0.3           # need hands in ≥30 % of window


def _wrist_xy(hand: list) -> np.ndarray | None:
    """Return [x, y] of the wrist keypoint, or None if data is bad."""
    try:
        kp = hand[WRIST]
        return np.array([kp[0], kp[1]], dtype=np.float64)
    except (IndexError, TypeError):
        return None


def is_signing(window: "LandmarkWindow") -> bool:
    """Return True if there is enough hand motion to count as signing.

    Strategy
    --------
    1. Collect wrist positions from consecutive frames (first detected hand).
    2. Compute average per-frame displacement in normalised coords.
    3. Compare against MOTION_THRESHOLD.

    Falls back gracefully when hand data is missing or malformed.
    """
    if not window or not window.frames or len(window.frames) < 2:
        return False

    frames_with_hands = [
        f for f in window.frames
        if f.hands is not None and len(f.hands) >= 1
    ]

    # Not enough hand data in this window
    if len(frames_with_hands) < max(2, int(len(window.frames) * MIN_HAND_COVERAGE)):
        return False

    # Collect wrist positions across consecutive hand-frames
    total_motion = 0.0
    motion_pairs = 0

    for i in range(len(frames_with_hands) - 1):
        curr_hands = frames_with_hands[i].hands
        next_hands = frames_with_hands[i + 1].hands

        # Compare each hand that appears in both frames
        for h_idx in range(min(len(curr_hands), len(next_hands))):
            curr_wrist = _wrist_xy(curr_hands[h_idx])
            next_wrist = _wrist_xy(next_hands[h_idx])
            if curr_wrist is not None and next_wrist is not None:
                total_motion += float(np.linalg.norm(next_wrist - curr_wrist))
                motion_pairs += 1

    if motion_pairs == 0:
        return False

    avg_motion = total_motion / motion_pairs
    return avg_motion > MOTION_THRESHOLD
