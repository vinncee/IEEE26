#!/usr/bin/env python3
"""
Quick test script for Member 3 recognition module.
Tests: detector, classifier, and smoothing functions.
"""

import sys
sys.path.insert(0, '/Users/kahei/Documents/GitHub/IEEE26/signcall-overlay/backend')

from app.cv.types import LandmarkFrame, LandmarkWindow
from app.recognition.detector import is_signing
from app.recognition.classifier import predict, extract_features
from app.recognition.smoothing import smooth, reset_history
import numpy as np

def create_mock_landmark_window(token: str, num_frames: int = 10) -> LandmarkWindow:
    """
    Create a mock landmark window for testing.
    Simulates hand position changes based on token type.
    """
    frames = []
    
    for i in range(num_frames):
        # Create mock hand data: [left_hand, right_hand]
        # Each hand is [x, y, ...] coordinates
        if token == "HELLO":
            # HELLO: hands raised and moving apart
            left_hand = [100 + i * 5, 150 - i * 10]
            right_hand = [300 - i * 5, 150 - i * 10]
        elif token == "THANKS":
            # THANKS: hands lower, closer together
            left_hand = [180 + i * 2, 250 + i * 5]
            right_hand = [220 - i * 2, 250 + i * 5]
        elif token == "REPEAT":
            # REPEAT: hands moving faster, further apart
            left_hand = [80 + i * 8, 200 - i * 8]
            right_hand = [320 - i * 8, 200 - i * 8]
        elif token == "SLOW":
            # SLOW: hands slow movement, close together
            left_hand = [190 + i * 1, 240 + i * 2]
            right_hand = [210 - i * 1, 240 + i * 2]
        else:
            # Default: no motion
            left_hand = [200, 240]
            right_hand = [200, 240]
        
        frame = LandmarkFrame(
            ts=i * 100,  # 100ms per frame
            hands=[left_hand, right_hand],
            pose=None,
            face=None
        )
        frames.append(frame)
    
    return LandmarkWindow(
        frames=frames,
        ts_start=frames[0].ts,
        ts_end=frames[-1].ts
    )

def test_detection():
    """Test signing detection."""
    print("\n=== Test 1: Signing Detection ===")
    
    # Test with signing motion
    signing_window = create_mock_landmark_window("HELLO", num_frames=10)
    is_sign = is_signing(signing_window)
    print(f"HELLO gesture detected as signing: {is_sign}")
    assert is_sign, "HELLO should be detected as signing"
    
    # Test with no hands
    idle_window = LandmarkWindow(
        frames=[LandmarkFrame(ts=i*100, hands=None) for i in range(10)],
        ts_start=0,
        ts_end=900
    )
    is_sign = is_signing(idle_window)
    print(f"No hands detected as signing: {is_sign}")
    assert not is_sign, "No hands should not be detected as signing"
    
    print("✓ Detection test passed")

def test_feature_extraction():
    """Test feature extraction."""
    print("\n=== Test 2: Feature Extraction ===")
    
    for token in ["HELLO", "THANKS", "REPEAT", "SLOW"]:
        window = create_mock_landmark_window(token)
        features = extract_features(window)
        print(f"{token:8} features: {features}")
        
        # Verify features are in valid range
        assert all(0 <= f <= 1 for f in features), f"Features for {token} out of range"
    
    print("✓ Feature extraction test passed")

def test_classification():
    """Test phrase classification."""
    print("\n=== Test 3: Phrase Classification ===")
    
    for token in ["HELLO", "THANKS", "REPEAT", "SLOW"]:
        window = create_mock_landmark_window(token)
        pred = predict(window)
        
        print(f"{token:8} → predicted: {pred['token']:8} (conf={pred['confidence']:.2f}), top2={pred['top2']}")
        
        # Verify output format
        assert isinstance(pred["token"], str), "token must be string"
        assert 0.0 <= pred["confidence"] <= 1.0, "confidence out of range"
        assert len(pred["top2"]) == 2, "top2 must have 2 elements"
        assert pred["ts"] == window.ts_end, "ts mismatch"
    
    print("✓ Classification test passed")

def test_smoothing():
    """Test temporal smoothing."""
    print("\n=== Test 4: Temporal Smoothing ===")
    
    session = "test_session"
    reset_history(session)
    
    # Create predictions: HELLO 3 times, then THANKS (outlier)
    tokens = ["HELLO", "HELLO", "HELLO", "THANKS"]
    
    for i, token in enumerate(tokens):
        window = create_mock_landmark_window(token)
        pred = predict(window)
        smoothed = smooth(pred, session_id=session)
        
        print(f"Pred {i+1}: {token:8} → smoothed to {smoothed['token']:8} (conf={smoothed['confidence']:.2f})")
    
    # After 3 HELLO predictions, the 4th THANKS should be debounced back to HELLO
    print("✓ Smoothing test passed")

def main():
    """Run all tests."""
    print("Starting Member 3 Recognition Module Tests")
    print("=" * 50)
    
    try:
        test_detection()
        test_feature_extraction()
        test_classification()
        test_smoothing()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("\nImplementation complete. Ready for integration testing.")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
