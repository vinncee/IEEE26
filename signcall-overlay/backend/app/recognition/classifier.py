"""Sign-language phrase classifier.

**Primary**: Teachable Machine (TM) pose model — uses PoseNet backbone to
extract a 14739-dim feature vector from each video frame, then classifies
with a trained Dense head.  Supports 10 sign-language phrases.

**Fallback**: Nearest-prototype on a 7-D hand-geometry feature vector
extracted from MediaPipe landmarks.  Used when the TM model files are not
available (e.g. in unit tests or CI).

Temporal smoothing is applied to the raw 7-D features via EMA when using
the fallback classifier.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import TYPE_CHECKING, List, Optional

from app.recognition.phrase_set import PHRASES

if TYPE_CHECKING:
    from app.cv.types import LandmarkFrame, LandmarkWindow

logger = logging.getLogger(__name__)

TOKENS = list(PHRASES.keys())

# ── MediaPipe keypoint indices ─────────────────────────────────────────────
WRIST        = 0
THUMB_TIP    = 4
INDEX_TIP    = 8
MIDDLE_TIP   = 12
RING_TIP     = 16
PINKY_TIP    = 20
FINGERTIPS   = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# MCP (knuckle) joints — base of each finger
INDEX_MCP    = 5
MIDDLE_MCP   = 9
RING_MCP     = 13
PINKY_MCP    = 17
MCP_JOINTS   = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

L_SHOULDER   = 11
R_SHOULDER   = 12

# ── Prototype feature vectors (7-D, every dim in 0-1) ─────────────────────
# Dimensions:
#   0  inter_hand_dist   – wrist-to-wrist distance (0 = overlapping, 1 = full frame width)
#   1  hand_height       – hands' Y relative to shoulders (0 = above head, 1 = at waist)
#   2  fingertip_spread  – avg spread of 5 fingertips relative to wrist (0 = fist, 1 = open)
#   3  motion_magnitude  – total wrist displacement over window (clamped to 0-1)
#   4  hand_speed        – per-frame wrist displacement (clamped to 0-1)
#   5  vertical_motion   – fraction of motion in Y direction (0 = horizontal, 1 = vertical)
#   6  finger_curl       – avg curl of 4 fingers: tip-to-MCP vs wrist-to-MCP (0 = straight, 1 = curled)
#
# Values below are initial estimates; run calibrate.py for accurate values.

PROTOTYPES = {
    # HELLO  — open palm wave near face: hands apart, raised, open hand, moderate motion
    "HELLO":  np.array([0.48, 0.34, 0.35, 0.14, 0.06, 0.24, 0.00]),
    # THANKS — flat hand from chin outward: hands close, mid-height, moderate spread, downward
    "THANKS": np.array([0.10, 0.42, 0.24, 0.17, 0.07, 1.00, 0.07]),
    # REPEAT — circular motion, hands apart, fast movement, semi-curled
    "REPEAT": np.array([0.40, 0.55, 0.12, 0.82, 0.37, 0.64, 0.32]),
    # SLOW   — hand slides down other hand: very close, slow, mostly vertical, curled
    "SLOW":   np.array([0.06, 0.46, 0.04, 0.10, 0.04, 1.00, 0.56]),
}

# Per-dimension importance weights (higher = more discriminative).
# Learned from feature-separation analysis: motion and curl are strongest.
FEATURE_WEIGHTS = np.array([1.0, 0.8, 1.2, 1.4, 1.0, 0.9, 1.3])

# Maximum normalised distance used to map distance→confidence
_MAX_PROTO_DIST = 1.2

# ── EMA (Exponential Moving Average) feature smoothing ────────────────────
# Smooths the raw 7-D feature vector across consecutive calls per session
# to dampen frame-to-frame noise from MediaPipe tracking jitter.
_EMA_ALPHA = 0.4      # blend factor: 0 = fully smooth, 1 = no smoothing
_ema_state: dict[str, np.ndarray] = {}     # session_id → smoothed features


def _ema_smooth(features: np.ndarray, session_id: str = "default") -> np.ndarray:
    """Apply EMA smoothing to a feature vector."""
    if session_id not in _ema_state or np.allclose(features, 0.0):
        _ema_state[session_id] = features.copy()
        return features
    prev = _ema_state[session_id]
    smoothed = _EMA_ALPHA * features + (1.0 - _EMA_ALPHA) * prev
    _ema_state[session_id] = smoothed
    return smoothed


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _kp_xy(hand: list, idx: int) -> Optional[np.ndarray]:
    """Return [x, y] for a keypoint index, or None."""
    try:
        kp = hand[idx]
        return np.array([kp[0], kp[1]], dtype=np.float64)
    except (IndexError, TypeError):
        return None


def _avg_fingertip_spread(hand: list) -> float:
    """Average distance from each fingertip to wrist, normalised ~0-1."""
    wrist = _kp_xy(hand, WRIST)
    if wrist is None:
        return 0.0
    dists: list[float] = []
    for tip_idx in FINGERTIPS:
        tip = _kp_xy(hand, tip_idx)
        if tip is not None:
            dists.append(float(np.linalg.norm(tip - wrist)))
    return float(np.mean(dists)) if dists else 0.0


def _finger_curl(hand: list) -> float:
    """Average finger curl (0 = straight / extended, 1 = fully curled).

    Measured as  1 - (tip_to_mcp / max_finger_len)  per finger.
    max_finger_len is the sum of bone lengths MCP→PIP→DIP→TIP when the finger
    is fully extended.  When curled the tip folds back towards the MCP, so
    tip_to_mcp shrinks.  This avoids the wrist-reference problem where the
    tip is always further from the wrist than the MCP.
    """
    curls: list[float] = []
    # Each finger is 4 consecutive keypoints: MCP, PIP, DIP, TIP
    # Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20
    for base in [5, 9, 13, 17]:
        pts = []
        for j in range(4):
            p = _kp_xy(hand, base + j)
            if p is None:
                break
            pts.append(p)
        if len(pts) < 4:
            continue
        # max_len = sum of bone lengths (fully extended straight line)
        max_len = sum(
            float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(3)
        )
        if max_len < 1e-6:
            continue
        # tip_to_mcp = straight-line distance MCP→TIP
        tip_to_mcp = float(np.linalg.norm(pts[3] - pts[0]))
        # When straight: tip_to_mcp ≈ max_len → curl ≈ 0
        # When curled:   tip_to_mcp << max_len → curl → 1
        curls.append(np.clip(1.0 - tip_to_mcp / max_len, 0.0, 1.0))
    return float(np.mean(curls)) if curls else 0.5


def _palm_centre(hand: list) -> Optional[np.ndarray]:
    """Return [x, y] of the palm centre (avg of wrist + 4 MCP joints).

    More stable than wrist alone because it averages out tracking jitter.
    """
    pts: list[np.ndarray] = []
    for idx in [WRIST] + MCP_JOINTS:
        p = _kp_xy(hand, idx)
        if p is not None:
            pts.append(p)
    if not pts:
        return None
    return np.mean(pts, axis=0)


def _shoulder_midpoint_y(frame: "LandmarkFrame") -> Optional[float]:
    """Return Y of the midpoint between left and right shoulder, or None."""
    if frame.pose is None or len(frame.pose) < 13:
        return None
    try:
        ly = frame.pose[L_SHOULDER][1]
        ry = frame.pose[R_SHOULDER][1]
        return (ly + ry) / 2.0
    except (IndexError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(window: "LandmarkWindow") -> np.ndarray:
    """Extract a 7-D normalised feature vector from a LandmarkWindow.

    Returns np.zeros(7) when there is no usable hand data.
    """
    N_FEAT = 7
    if not window or not window.frames:
        return np.zeros(N_FEAT)

    frames_with_hands = [
        f for f in window.frames
        if f.hands is not None and len(f.hands) >= 1
    ]
    if not frames_with_hands:
        return np.zeros(N_FEAT)

    # ── Feature 0: inter-hand distance (wrist-to-wrist, avg over window) ──
    inter_dists: list[float] = []
    for f in frames_with_hands:
        if len(f.hands) >= 2:
            w0 = _kp_xy(f.hands[0], WRIST)
            w1 = _kp_xy(f.hands[1], WRIST)
            if w0 is not None and w1 is not None:
                inter_dists.append(float(np.linalg.norm(w1 - w0)))
    feat_inter_hand = float(np.mean(inter_dists)) if inter_dists else 0.0
    feat_inter_hand = np.clip(feat_inter_hand, 0.0, 1.0)

    # ── Feature 1: hand height relative to shoulders ──────────────────────
    # lower value = hands above shoulders, higher = below
    rel_heights: list[float] = []
    for f in frames_with_hands:
        shoulder_y = _shoulder_midpoint_y(f)
        for hand in f.hands[:2]:
            wrist = _kp_xy(hand, WRIST)
            if wrist is not None:
                if shoulder_y is not None:
                    # positive = hands below shoulder
                    rel_heights.append(wrist[1] - shoulder_y + 0.5)
                else:
                    # fallback: raw Y (0 = top of frame, 1 = bottom)
                    rel_heights.append(wrist[1])
    feat_hand_height = float(np.mean(rel_heights)) if rel_heights else 0.5
    feat_hand_height = np.clip(feat_hand_height, 0.0, 1.0)

    # ── Feature 2: fingertip spread (open hand vs fist) ───────────────────
    spreads: list[float] = []
    for f in frames_with_hands:
        for hand in f.hands[:2]:
            spreads.append(_avg_fingertip_spread(hand))
    feat_spread = float(np.mean(spreads)) if spreads else 0.0
    # Spread is already in 0-~0.4 range for normalised coords; rescale
    feat_spread = np.clip(feat_spread / 0.35, 0.0, 1.0)

    # ── Feature 3 & 4: motion magnitude & speed ──────────────────────────
    # Use palm centre (avg wrist+MCPs) for more stable tracking than wrist alone
    displacements: list[float] = []
    dy_fracs: list[float] = []
    for i in range(len(frames_with_hands) - 1):
        curr_hands = frames_with_hands[i].hands
        next_hands = frames_with_hands[i + 1].hands
        for h_idx in range(min(len(curr_hands), len(next_hands))):
            cp = _palm_centre(curr_hands[h_idx])
            np_ = _palm_centre(next_hands[h_idx])
            if cp is not None and np_ is not None:
                diff = np_ - cp
                d = float(np.linalg.norm(diff))
                displacements.append(d)
                if d > 1e-6:
                    dy_fracs.append(abs(diff[1]) / d)

    total_motion = sum(displacements)
    feat_motion = np.clip(total_motion / 1.2, 0.0, 1.0)   # wider range for real data

    n_steps = max(1, len(displacements))
    feat_speed = np.clip((total_motion / n_steps) / 0.15, 0.0, 1.0)

    # ── Feature 5: vertical-motion fraction ───────────────────────────────
    feat_vert = float(np.mean(dy_fracs)) if dy_fracs else 0.5

    # ── Feature 6: finger curl (replaces noisy symmetry metric) ───────────
    curls: list[float] = []
    for f in frames_with_hands:
        for hand in f.hands[:2]:
            curls.append(_finger_curl(hand))
    feat_curl = float(np.mean(curls)) if curls else 0.5

    return np.array([
        feat_inter_hand,
        feat_hand_height,
        feat_spread,
        feat_motion,
        feat_speed,
        feat_vert,
        feat_curl,
    ])


def _wrist_disp(hand_a: list, hand_b: list) -> Optional[float]:
    """Wrist displacement between two hand snapshots."""
    wa = _kp_xy(hand_a, WRIST)
    wb = _kp_xy(hand_b, WRIST)
    if wa is None or wb is None:
        return None
    return float(np.linalg.norm(wb - wa))


# ═══════════════════════════════════════════════════════════════════════════
# Classification
# ═══════════════════════════════════════════════════════════════════════════

def predict(window: "LandmarkWindow", session_id: str = "default") -> dict:
    """Classify a LandmarkWindow into one of the known phrase tokens.

    Returns
    -------
    dict  with keys  token, confidence, top2, ts
        Exactly the interface Member 4 / the translator expects.
    """
    features = extract_features(window)

    # Apply EMA temporal smoothing on the feature vector
    features = _ema_smooth(features, session_id=session_id)

    # Weighted Euclidean distance — discriminative features count more
    distances = {
        tok: float(np.linalg.norm(FEATURE_WEIGHTS * (features - proto)))
        for tok, proto in PROTOTYPES.items()
    }

    sorted_preds = sorted(distances.items(), key=lambda kv: kv[1])
    top_token   = sorted_preds[0][0]
    top_dist    = sorted_preds[0][1]
    second_token = sorted_preds[1][0] if len(sorted_preds) > 1 else TOKENS[0]

    # Confidence: inverse of normalised distance, floored at 0.10
    norm_dist  = np.clip(top_dist / _MAX_PROTO_DIST, 0.0, 1.0)
    confidence = max(0.10, 1.0 - norm_dist)

    # Penalise when top two are very close (ambiguous)
    if len(sorted_preds) >= 2:
        gap = sorted_preds[1][1] - top_dist
        if gap < 0.05:  # nearly tied
            confidence *= 0.75

    return {
        "token": top_token,
        "confidence": round(confidence, 4),
        "top2": [top_token, second_token],
        "ts": window.ts_end,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TM-model-aware classification (primary path)
# ═══════════════════════════════════════════════════════════════════════════

_tm_available: Optional[bool] = None  # cached after first check


def _check_tm_available() -> bool:
    """Return True if the TM model files are present."""
    global _tm_available
    if _tm_available is None:
        try:
            from app.recognition.tm_model import is_available
            _tm_available = is_available()
        except Exception:
            _tm_available = False
        logger.info("TM model available: %s", _tm_available)
    return _tm_available


def predict_with_tm(
    window: "LandmarkWindow",
    frames_bgr: list,
    session_id: str = "default",
    ts: int = 0,
) -> dict:
    """Classify using the TM pose model (primary) with prototype fallback.

    Parameters
    ----------
    window : LandmarkWindow
        Landmark data from MediaPipe (used for prototype fallback).
    frames_bgr : list[np.ndarray]
        Raw BGR frames from the same window (used by TM model).
    session_id : str
        Per-session identifier for EMA state.
    ts : int
        Timestamp for the prediction.

    Returns
    -------
    dict  with keys  token, confidence, top2, ts
    """
    if _check_tm_available() and frames_bgr:
        try:
            from app.recognition.tm_model import predict_window
            result = predict_window(frames_bgr)
            result["ts"] = ts
            logger.debug(
                "TM predict: token=%s  conf=%.3f",
                result["token"], result["confidence"],
            )
            return result
        except Exception as exc:
            logger.warning("TM model failed, falling back to prototype: %s", exc)

    # Fallback to prototype-based classifier
    return predict(window, session_id=session_id)
