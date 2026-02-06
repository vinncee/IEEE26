#!/usr/bin/env python3
"""Unit tests for Member 3 recognition module.

All mock data uses the real MediaPipe hand shape:
    hands : [num_hands][21][3]   (x, y, z each normalised 0-1)
    pose  : [33][3]

Run:  python test_recognition.py   (from backend/)
"""

import os
import sys

# Ensure the backend package is importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from app.cv.types import LandmarkFrame, LandmarkWindow
from app.recognition.detector import is_signing
from app.recognition.classifier import predict, extract_features, _ema_state
from app.recognition.smoothing import smooth, reset_history


# ── helpers to generate realistic [21][3] hand data ──────────────────────

def _make_hand(wrist_xy, spread=0.10, curl=0.0, z=0.0):
    """Generate a 21-keypoint hand centred on *wrist_xy* with *spread*.

    Returns list[21][3] – mimics MediaPipe normalised coords.
    spread controls finger length; curl (0-1) bends fingers via angular folding
    at PIP/DIP/TIP joints, mimicking real finger flexion.
    """
    wx, wy = wrist_xy
    kps = []
    kps.append([wx, wy, z])  # 0: wrist

    def _finger(base_x, bone_len, c):
        """4 joints using angular curl: MCP, PIP, DIP, TIP.
        
        Each successive joint bends by an angle proportional to curl.
        direction starts pointing up (-Y) and rotates with each joint.
        """
        import math
        pts = []
        # Start from base position (MCP is directly above wrist area)
        x, y = base_x, wy
        angle = -math.pi / 2  # pointing up initially
        for j in range(4):
            # Move along current direction by bone_len
            x += bone_len * math.cos(angle)
            y += bone_len * math.sin(angle)
            pts.append([x, y, z])
            # Bend at this joint: more curl = sharper bend
            # PIP bends a lot, DIP bends a lot, creating fist-like curl
            if j < 3:  # don't bend after TIP
                angle += c * (math.pi / 2)  # up to 90° per joint at full curl
        return pts

    bone = spread * 0.22  # fixed bone length
    # 1-4: thumb
    kps += _finger(wx + spread * 0.25, bone, curl)
    # 5-8: index
    kps += _finger(wx - spread * 0.05, bone, curl)
    # 9-12: middle
    kps += _finger(wx + spread * 0.05, bone * 1.08, curl)
    # 13-16: ring
    kps += _finger(wx + spread * 0.15, bone * 0.96, curl)
    # 17-20: pinky
    kps += _finger(wx + spread * 0.25, bone * 0.80, curl)
    return kps


def _make_pose(shoulder_y=0.45):
    """Return a 33-keypoint pose with shoulders at *shoulder_y*."""
    pose = [[0.5, 0.5, 0.0]] * 33
    pose[11] = [0.35, shoulder_y, 0.0]   # left shoulder
    pose[12] = [0.65, shoulder_y, 0.0]   # right shoulder
    return pose


def create_mock_window(token: str, num_frames: int = 10) -> LandmarkWindow:
    """Build a LandmarkWindow with realistic [2][21][3] hand data.

    Each gesture token has distinct motion / position / spread profiles that
    are designed to be separable by the classifier's 7-D features.
    """
    frames = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)  # 0 → 1 over the window

        if token == "HELLO":
            # Open palm wave near face — hands spread, raised, lateral motion, fingers open
            lw = (0.30 - t * 0.08, 0.28 + t * 0.02)
            rw = (0.70 + t * 0.08, 0.28 + t * 0.02)
            spread, curl = 0.14, 0.05
        elif token == "THANKS":
            # Chin-out motion — hands close, moderate height, forward push, slightly curled
            lw = (0.45, 0.32 + t * 0.10)
            rw = (0.55, 0.32 + t * 0.10)
            spread, curl = 0.10, 0.30
        elif token == "REPEAT":
            # Circular motion — hands apart, lower, fast displacement, moderately curled
            angle = t * 2 * np.pi
            lw = (0.30 + 0.08 * np.cos(angle), 0.50 + 0.08 * np.sin(angle))
            rw = (0.70 - 0.08 * np.cos(angle), 0.50 - 0.08 * np.sin(angle))
            spread, curl = 0.08, 0.65
        elif token == "SLOW":
            # Slow downward slide — hands very close, slow, mostly vertical, fully curled
            lw = (0.47, 0.38 + t * 0.06)
            rw = (0.53, 0.38 + t * 0.06)
            spread, curl = 0.06, 0.90
        else:
            lw = (0.45, 0.45)
            rw = (0.55, 0.45)
            spread, curl = 0.08, 0.3

        left_hand  = _make_hand(lw, spread=spread, curl=curl)
        right_hand = _make_hand(rw, spread=spread, curl=curl)

        frames.append(LandmarkFrame(
            ts=i * 100,
            hands=[left_hand, right_hand],
            handedness=["Left", "Right"],
            pose=_make_pose(shoulder_y=0.45),
            face=None,
        ))

    return LandmarkWindow(
        frames=frames,
        ts_start=frames[0].ts,
        ts_end=frames[-1].ts,
    )


def _make_idle_window(num_frames: int = 10) -> LandmarkWindow:
    """Hands present but perfectly still — should NOT count as signing."""
    hand = _make_hand((0.50, 0.50), spread=0.10)
    frames = [
        LandmarkFrame(
            ts=i * 100,
            hands=[hand, hand],
            handedness=["Left", "Right"],
            pose=_make_pose(),
        )
        for i in range(num_frames)
    ]
    return LandmarkWindow(frames=frames, ts_start=0, ts_end=(num_frames - 1) * 100)


# ── tests ─────────────────────────────────────────────────────────────────

def test_detection():
    print("\n=== Test 1: Signing Detection ===")

    # Active signing should be detected
    for tok in ["HELLO", "REPEAT"]:
        w = create_mock_window(tok)
        assert is_signing(w), f"{tok} should be detected as signing"
        print(f"  {tok:8} → signing=True  ✓")

    # No hands at all → not signing
    no_hands = LandmarkWindow(
        frames=[LandmarkFrame(ts=i * 100, hands=None) for i in range(10)],
        ts_start=0, ts_end=900,
    )
    assert not is_signing(no_hands), "No-hands should not be signing"
    print("  NO_HANDS → signing=False ✓")

    # Idle hands (present but still) → not signing
    idle = _make_idle_window()
    assert not is_signing(idle), "Idle hands should not be signing"
    print("  IDLE     → signing=False ✓")

    print("✓ Detection tests passed")


def test_feature_extraction():
    print("\n=== Test 2: Feature Extraction ===")

    for tok in ["HELLO", "THANKS", "REPEAT", "SLOW"]:
        w = create_mock_window(tok)
        feats = extract_features(w)
        print(f"  {tok:8} features: [{', '.join(f'{v:.3f}' for v in feats)}]")
        assert feats.shape == (7,), f"Expected 7-D, got {feats.shape}"
        assert all(0.0 <= v <= 1.0 for v in feats), f"{tok} has out-of-range feature"

    # Empty / no-hands window → zeros
    empty = LandmarkWindow(frames=[], ts_start=0, ts_end=0)
    assert np.allclose(extract_features(empty), 0.0)
    print("  EMPTY    features: all zeros ✓")

    print("✓ Feature extraction tests passed")


def test_classification():
    print("\n=== Test 3: Phrase Classification ===")

    for tok in ["HELLO", "THANKS", "REPEAT", "SLOW"]:
        _ema_state.clear()  # reset EMA so each gesture is independent
        w = create_mock_window(tok)
        pred = predict(w)
        print(f"  {tok:8} → predicted={pred['token']:8}  "
              f"conf={pred['confidence']:.2f}  top2={pred['top2']}")

        # Contract checks
        assert isinstance(pred["token"], str) and pred["token"] in [
            "HELLO", "THANKS", "REPEAT", "SLOW",
            "HOW", "YOU", "CAN", "SEE_YOU_LATER", "FATHER", "MOTHER",
        ], "token must be a valid phrase key"
        assert 0.0 <= pred["confidence"] <= 1.0, "confidence out of range"
        assert len(pred["top2"]) == 2, "top2 must have 2 elements"
        assert pred["ts"] == w.ts_end, "ts must equal window end"

    print("✓ Classification tests passed")


def test_feature_separation():
    """Verify that different gestures produce distinguishable features."""
    print("\n=== Test 4: Feature Separation ===")

    feat_map = {}
    for tok in ["HELLO", "THANKS", "REPEAT", "SLOW"]:
        feat_map[tok] = extract_features(create_mock_window(tok))

    # Every pair of gestures should have non-trivial distance
    tokens = list(feat_map.keys())
    all_ok = True
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            d = float(np.linalg.norm(feat_map[tokens[i]] - feat_map[tokens[j]]))
            ok = d > 0.02
            status = "✓" if ok else "✗"
            print(f"  {tokens[i]:8} ↔ {tokens[j]:8}  dist={d:.4f}  {status}")
            if not ok:
                all_ok = False

    assert all_ok, "Some gesture pairs are not separable"
    print("✓ Feature separation tests passed")


def test_smoothing():
    print("\n=== Test 5: Temporal Smoothing ===")

    session = "test_session"
    reset_history(session)
    _ema_state.clear()

    # Feed 3× HELLO then 1× THANKS (outlier)
    results = []
    for tok in ["HELLO", "HELLO", "HELLO", "THANKS"]:
        w = create_mock_window(tok)
        pred = predict(w)
        smoothed = smooth(pred, session_id=session)
        results.append(smoothed)
        print(f"  input={tok:8} → smoothed={smoothed['token']:8}  "
              f"conf={smoothed['confidence']:.2f}")

    # The 4th prediction should be debounced back to the majority
    assert results[-1]["token"] != "THANKS" or results[-1]["confidence"] < 0.5, \
        "Outlier should be debounced or have reduced confidence"
    print("✓ Smoothing tests passed")


def test_member4_contract():
    """Verify the exact output dict shape that Member 4's translator expects."""
    print("\n=== Test 6: Member 4 Interface Contract ===")

    _ema_state.clear()
    w = create_mock_window("HELLO")
    pred = predict(w)

    required_keys = {"token", "confidence", "top2", "ts"}
    assert set(pred.keys()) == required_keys, \
        f"pred keys {set(pred.keys())} ≠ expected {required_keys}"
    assert isinstance(pred["token"], str)
    assert isinstance(pred["confidence"], float)
    assert isinstance(pred["top2"], list) and len(pred["top2"]) == 2
    assert isinstance(pred["ts"], int)
    print("  pred dict shape matches Member 4 contract ✓")

    # After smoothing the contract is preserved
    smoothed = smooth(pred, session_id="contract_test")
    assert set(smoothed.keys()) == required_keys
    print("  smoothed dict shape matches Member 4 contract ✓")

    print("✓ Member 4 contract tests passed")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("Member 3 Recognition Module — Unit Tests")
    print("=" * 55)

    try:
        test_detection()
        test_feature_extraction()
        test_classification()
        test_feature_separation()
        test_smoothing()
        test_member4_contract()

        print("\n" + "=" * 55)
        print("✓ All tests passed!")

    except AssertionError as e:
        print(f"\n✗ Test FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
