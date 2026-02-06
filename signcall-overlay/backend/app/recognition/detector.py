"""Detect if hands are actively signing (moving) in a landmark window."""

import numpy as np

def is_signing(window) -> bool:
    """
    Detect hand movement across the window to determine if user is signing.
    
    Args:
        window: LandmarkWindow with 10 frames of landmarks
        
    Returns:
        bool: True if movement exceeds threshold, False if idle
    """
    if not window or not window.frames or len(window.frames) < 2:
        return False
    
    # Check if hands data is available
    frames_with_hands = [f for f in window.frames if f.hands is not None]
    if len(frames_with_hands) < 2:
        return False
    
    # Calculate total hand movement across frames
    total_motion = 0.0
    for i in range(len(frames_with_hands) - 1):
        curr_hands = frames_with_hands[i].hands
        next_hands = frames_with_hands[i + 1].hands
        
        if isinstance(curr_hands, list) and isinstance(next_hands, list):
            # Try to extract positions from hands data
            # Could be [[x1, y1, ...], [x2, y2, ...]] for two hands
            for curr_hand, next_hand in zip(curr_hands, next_hands):
                if curr_hand and next_hand:
                    try:
                        # Extract x, y coordinates
                        curr_pos = np.array(curr_hand[:2]) if isinstance(curr_hand, (list, tuple)) else curr_hand
                        next_pos = np.array(next_hand[:2]) if isinstance(next_hand, (list, tuple)) else next_hand
                        
                        if isinstance(curr_pos, np.ndarray) and isinstance(next_pos, np.ndarray):
                            motion = np.linalg.norm(curr_pos - next_pos)
                            total_motion += motion
                    except (TypeError, ValueError):
                        continue
    
    # Threshold: if average motion per frame > 5 pixels, consider it signing
    avg_motion = total_motion / max(1, len(frames_with_hands) - 1)
    return avg_motion > 5.0
