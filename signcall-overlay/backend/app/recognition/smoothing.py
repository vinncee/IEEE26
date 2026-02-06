"""Temporal smoothing of sign recognition predictions to reduce flicker."""

from collections import deque
from typing import Dict, Optional

# Global smoothing state (in production, should be per-session)
_prediction_history: Dict[str, deque] = {}
SMOOTH_WINDOW_SIZE = 4  # Majority vote over last 4 predictions

def smooth(pred: dict, session_id: str = "default") -> dict:
    """
    Apply majority-vote smoothing to prediction history.
    
    Args:
        pred: Current prediction dict {"token": str, "confidence": float, "top2": [...], "ts": int}
        session_id: Session identifier to maintain separate smoothing histories
        
    Returns:
        dict: Smoothed prediction with same format
    """
    if session_id not in _prediction_history:
        _prediction_history[session_id] = deque(maxlen=SMOOTH_WINDOW_SIZE)
    
    history = _prediction_history[session_id]
    current_token = pred["token"]
    
    # Add current prediction to history
    history.append(current_token)
    
    if len(history) < 2:
        # Not enough history yet, return as-is
        return pred
    
    # Majority voting: find most common token in history
    token_counts = {}
    for token in history:
        token_counts[token] = token_counts.get(token, 0) + 1
    
    majority_token = max(token_counts, key=token_counts.get)
    majority_count = token_counts[majority_token]
    
    # If current token matches majority, boost confidence
    # If current token differs from majority, use majority with reduced confidence (debounce)
    if current_token == majority_token:
        # Token matches majority: confidence boost based on agreement
        boost = (majority_count - 1) / len(history) * 0.2  # Up to +0.2 boost
        smoothed_confidence = min(0.99, pred["confidence"] + boost)
    else:
        # Token differs from majority: use majority token with reduced confidence
        # This debounces occasional false positives
        smoothed_confidence = pred["confidence"] * 0.6  # Reduce by 40%
        majority_token = majority_token
    
    return {
        "token": majority_token,
        "confidence": smoothed_confidence,
        "top2": pred["top2"],
        "ts": pred["ts"]
    }

def reset_history(session_id: str = "default") -> None:
    """Reset smoothing history for a session (e.g., when user switches)."""
    if session_id in _prediction_history:
        _prediction_history[session_id].clear()
