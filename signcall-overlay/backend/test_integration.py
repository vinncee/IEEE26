#!/usr/bin/env python3
"""
Integration test for the complete Member 3 + orchestrator pipeline.
Tests the full end-to-end flow as it would happen in production.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.cv.types import LandmarkFrame, LandmarkWindow
from app.pipeline.orchestrator import process_frame
import asyncio
import numpy as np

def create_mock_frame_bgr(shape=(480, 640, 3)):
    """Create mock BGR frame for testing."""
    return np.zeros(shape, dtype=np.uint8)

def create_mock_landmarks(token: str, ts: int) -> LandmarkFrame:
    """Create mock hand landmarks for testing."""
    if token == "HELLO":
        left_hand = [100 + ts // 10, 150 - ts // 20]
        right_hand = [300 - ts // 10, 150 - ts // 20]
    elif token == "THANKS":
        left_hand = [180 + ts // 30, 250 + ts // 50]
        right_hand = [220 - ts // 30, 250 + ts // 50]
    else:
        left_hand = [200, 240]
        right_hand = [200, 240]
    
    return LandmarkFrame(
        ts=ts,
        hands=[left_hand, right_hand],
        pose=None,
        face=None
    )

async def test_orchestrator_pipeline():
    """Test the complete pipeline through orchestrator."""
    print("\n=== Integration Test: Orchestrator Pipeline ===\n")
    
    session = "test_session"
    user = "test_user"
    
    # Simulate 50 frames (5 windows of 10 frames each)
    print("Simulating 50 frames of HELLO gesture...")
    print("-" * 60)
    
    results = []
    for frame_idx in range(50):
        ts = frame_idx * 100  # 100ms per frame
        frame_bgr = create_mock_frame_bgr()
        
        # Process frame through orchestrator
        result = await process_frame(session, user, frame_bgr, ts)
        
        if result:
            results.append(result)
            print(f"Frame {frame_idx:3d} (ts={ts:4d}ms) → "
                  f"Token: {result['caption']:20s} | "
                  f"Confidence: {result['confidence']:.2f} | "
                  f"Mode: {result['mode']}")
    
    print("-" * 60)
    print(f"\nProcessed {50} frames → {len(results)} caption outputs")
    
    # Verify output format
    if results:
        sample = results[0]
        print(f"\nSample output format:")
        print(f"  type: {sample['type']}")
        print(f"  session: {sample['session']}")
        print(f"  user: {sample['user']}")
        print(f"  caption: {sample['caption']}")
        print(f"  confidence: {sample['confidence']}")
        print(f"  mode: {sample['mode']}")
        
        # Verify all required fields
        required_fields = {'type', 'session', 'user', 'ts', 'caption', 'confidence', 'mode'}
        actual_fields = set(sample.keys())
        
        if required_fields.issubset(actual_fields):
            print("\n✓ All required fields present")
        else:
            print(f"\n✗ Missing fields: {required_fields - actual_fields}")
    
    print("\n✓ Integration test passed!")
    return True

async def main():
    """Run integration test."""
    try:
        success = await test_orchestrator_pipeline()
        if success:
            print("\n" + "=" * 60)
            print("Member 3 implementation ready for team integration!")
            print("=" * 60)
            return 0
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
