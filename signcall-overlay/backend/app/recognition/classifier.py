"""Sign language phrase classifier using feature extraction and prototype matching."""

import numpy as np
from app.recognition.phrase_set import PHRASES

TOKENS = list(PHRASES.keys())

# Hand-coded prototype feature vectors for each phrase
# These represent "typical" features for each gesture
# Format: 4 dimensions = [hand_distance, hand_height, motion_magnitude, hand_speed]
# Prototypes initialized with simple heuristics; can be refined with labeled data
PROTOTYPES = {
    "HELLO": np.array([0.2, 0.7, 15.0, 8.0]),      # Hands apart, raised, moderate motion
    "THANKS": np.array([0.15, 0.5, 12.0, 6.0]),    # Hands closer, lower, slight motion
    "REPEAT": np.array([0.25, 0.4, 18.0, 10.0]),   # Hands further apart, lower, faster
    "SLOW": np.array([0.1, 0.6, 8.0, 4.0]),        # Hands close, moderate height, slow
}

def extract_features(window) -> np.ndarray:
    """
    Extract numeric features from landmark window for classification.
    
    Args:
        window: LandmarkWindow with 10 frames of hand landmarks
        
    Returns:
        np.ndarray: Feature vector [hand_distance, hand_height, motion_magnitude, hand_speed]
    """
    if not window or not window.frames or len(window.frames) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    frames_with_hands = [f for f in window.frames if f.hands is not None]
    if len(frames_with_hands) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    # Feature 1: Hand distance (average distance between two hands across frames)
    hand_distances = []
    for frame in frames_with_hands:
        if isinstance(frame.hands, list) and len(frame.hands) >= 2:
            # Assume hands is a list with at least 2 elements (left, right hand positions)
            h1, h2 = frame.hands[0], frame.hands[1]
            if isinstance(h1, (list, tuple)) and isinstance(h2, (list, tuple)):
                dist = np.linalg.norm(np.array(h1[:2]) - np.array(h2[:2]))
                hand_distances.append(dist)
    
    avg_hand_distance = np.mean(hand_distances) / 100.0 if hand_distances else 0.0  # Normalize
    
    # Feature 2: Hand height (average Y position of hands, normalized 0-1)
    hand_heights = []
    for frame in frames_with_hands:
        if isinstance(frame.hands, list) and len(frame.hands) >= 1:
            for hand in frame.hands[:2]:
                if isinstance(hand, (list, tuple)) and len(hand) >= 2:
                    hand_heights.append(hand[1])  # Y coordinate
    
    avg_hand_height = (np.mean(hand_heights) / 480.0) if hand_heights else 0.5  # Normalize for 480p video
    avg_hand_height = np.clip(avg_hand_height, 0.0, 1.0)
    
    # Feature 3: Motion magnitude (total movement across window)
    motion_magnitude = 0.0
    for i in range(len(frames_with_hands) - 1):
        curr = frames_with_hands[i].hands
        next_frame = frames_with_hands[i + 1].hands
        if isinstance(curr, list) and isinstance(next_frame, list):
            if len(curr) >= 1 and len(next_frame) >= 1:
                motion = np.linalg.norm(np.array(curr[0][:2]) - np.array(next_frame[0][:2]))
                motion_magnitude += motion
    
    # Feature 4: Hand speed (motion magnitude / num frames)
    hand_speed = motion_magnitude / max(1, len(frames_with_hands) - 1)
    
    return np.array([
        np.clip(avg_hand_distance, 0.0, 1.0),
        avg_hand_height,
        np.clip(motion_magnitude, 0.0, 30.0) / 30.0,  # Normalize to 0-1
        np.clip(hand_speed, 0.0, 10.0) / 10.0         # Normalize to 0-1
    ])

def predict(window) -> dict:
    """
    Classify landmarks into one of 4 phrase tokens using nearest-prototype matching.
    
    Args:
        window: LandmarkWindow with 10 frames
        
    Returns:
        dict: {"token": str, "confidence": float, "top2": [str, str], "ts": int}
    """
    # Extract features from the window
    features = extract_features(window)
    
    # Compute distance from extracted features to each prototype
    distances = {}
    for token, prototype in PROTOTYPES.items():
        dist = np.linalg.norm(features - prototype)
        distances[token] = dist
    
    # Sort by distance (closest match first)
    sorted_predictions = sorted(distances.items(), key=lambda x: x[1])
    top_token = sorted_predictions[0][0]
    top_distance = sorted_predictions[0][1]
    second_token = sorted_predictions[1][0] if len(sorted_predictions) > 1 else TOKENS[0]
    
    # Confidence: higher for closer matches (inverse of normalized distance)
    # Normalize distance to 0-1 range (use 5.0 as max typical distance)
    normalized_distance = np.clip(top_distance / 5.0, 0.0, 1.0)
    confidence = 1.0 - normalized_distance
    confidence = max(0.1, confidence)  # Floor confidence at 0.1
    
    return {
        "token": top_token,
        "confidence": confidence,
        "top2": [top_token, second_token],
        "ts": window.ts_end
    }
