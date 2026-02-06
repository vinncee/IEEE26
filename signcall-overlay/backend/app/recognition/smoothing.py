"""Temporal smoothing of sign recognition predictions to reduce flicker.

Uses confidence-weighted voting so that high-confidence predictions dominate
over noisy low-confidence ones, and recent predictions count more via
exponential recency weighting.
"""

from collections import deque
from typing import Dict, Optional, Tuple

# Global smoothing state (in production, should be per-session)
_prediction_history: Dict[str, deque] = {}   # stores (token, confidence) pairs
SMOOTH_WINDOW_SIZE = 5  # Weighted vote over last 5 predictions

# Recency weights: most recent prediction gets highest weight
_RECENCY_WEIGHTS = [0.5, 0.6, 0.75, 0.9, 1.0]   # oldest → newest


def smooth(pred: dict, session_id: str = "default") -> dict:
    """
    Apply confidence-weighted majority-vote smoothing.

    Each past prediction votes with weight = confidence × recency_factor.
    The token with the highest total weighted vote wins.
    """
    if session_id not in _prediction_history:
        _prediction_history[session_id] = deque(maxlen=SMOOTH_WINDOW_SIZE)

    history = _prediction_history[session_id]
    current_token = pred["token"]
    current_conf  = pred["confidence"]

    # Store (token, confidence)
    history.append((current_token, current_conf))

    if len(history) < 2:
        return pred

    # Weighted voting: weight = confidence × recency
    n = len(history)
    recency = _RECENCY_WEIGHTS[-n:]   # take the last n weights
    token_scores: Dict[str, float] = {}
    for i, (tok, conf) in enumerate(history):
        w = conf * recency[i]
        token_scores[tok] = token_scores.get(tok, 0.0) + w

    majority_token = max(token_scores, key=token_scores.get)
    total_weight   = sum(recency[i] for i in range(n))  # max possible score
    majority_score = token_scores[majority_token]

    if current_token == majority_token:
        # Agreement: boost confidence proportional to vote share
        vote_share = majority_score / max(total_weight, 1e-6)
        boost = vote_share * 0.15            # up to +0.15
        smoothed_confidence = min(0.99, current_conf + boost)
    else:
        # Disagreement: suppress with stronger penalty for low confidence
        smoothed_confidence = current_conf * 0.50

    return {
        "token": majority_token,
        "confidence": round(smoothed_confidence, 4),
        "top2": pred["top2"],
        "ts": pred["ts"]
    }

def reset_history(session_id: str = "default") -> None:
    """Reset smoothing history for a session (e.g., when user switches)."""
    if session_id in _prediction_history:
        _prediction_history[session_id].clear()
