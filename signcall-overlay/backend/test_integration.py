#!/usr/bin/env python3
"""Integration test – exercises the full orchestrator pipeline with realistic
MediaPipe-shaped mock data.

Because the orchestrator calls the real MediaPipe extractor (which needs a
camera), we monkeypatch `extract_landmarks` so the pipeline runs purely from
synthetic hand / pose data.

Run:  python test_integration.py   (from backend/)
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from app.cv.types import LandmarkFrame, LandmarkWindow
from app.pipeline import orchestrator


# ── realistic mock hand data ──────────────────────────────────────────────

def _make_hand(wrist_xy, spread=0.10, z=0.0):
    wx, wy = wrist_xy
    kps = [[wx, wy, z]]
    for j in range(1, 5):
        kps.append([wx + spread * j * 0.3, wy - spread * j * 0.2, z])
    for j in range(1, 5):
        kps.append([wx - spread * 0.05, wy - spread * j * 0.25, z])
    for j in range(1, 5):
        kps.append([wx + spread * 0.05, wy - spread * j * 0.27, z])
    for j in range(1, 5):
        kps.append([wx + spread * 0.15, wy - spread * j * 0.24, z])
    for j in range(1, 5):
        kps.append([wx + spread * 0.25, wy - spread * j * 0.20, z])
    return kps


def _make_pose(shoulder_y=0.45):
    pose = [[0.5, 0.5, 0.0]] * 33
    pose[11] = [0.35, shoulder_y, 0.0]
    pose[12] = [0.65, shoulder_y, 0.0]
    return pose


# Track how many frames we've generated for motion simulation
_frame_counter = 0


def _fake_extract_landmarks(frame_bgr, ts: int) -> LandmarkFrame:
    """Replaces the real MediaPipe extractor with synthetic HELLO gesture."""
    global _frame_counter
    t = (_frame_counter % 10) / 9.0  # 0→1 over each 10-frame window
    _frame_counter += 1

    lw = (0.30 - t * 0.08, 0.28 + t * 0.02)
    rw = (0.70 + t * 0.08, 0.28 + t * 0.02)
    return LandmarkFrame(
        ts=ts,
        hands=[_make_hand(lw, spread=0.14), _make_hand(rw, spread=0.14)],
        handedness=["Left", "Right"],
        pose=_make_pose(),
        face=None,
    )


# ── integration test ──────────────────────────────────────────────────────

async def test_orchestrator_pipeline():
    print("\n=== Integration Test: Orchestrator Pipeline ===")
    print("    (MediaPipe extractor monkeypatched with synthetic HELLO)\n")

    # Monkeypatch so we don't need a real camera / mediapipe install
    original = orchestrator.extract_landmarks
    orchestrator.extract_landmarks = _fake_extract_landmarks

    global _frame_counter
    _frame_counter = 0
    # Clear any stale per-session buffers
    orchestrator._buffers.clear()

    session, user = "int_test", "user1"
    results = []

    try:
        for idx in range(30):
            ts = idx * 100
            frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            result = await orchestrator.process_frame(session, user, frame_bgr, ts)
            if result is not None:
                results.append(result)
                print(f"  frame {idx:2d}  ts={ts:4d}ms → "
                      f"caption={result['caption']!r:30s}  "
                      f"conf={result['confidence']:.2f}  "
                      f"mode={result['mode']}")
    finally:
        orchestrator.extract_landmarks = original

    print(f"\n  Processed 30 frames → {len(results)} caption outputs")

    # ── assertions ────────────────────────────────────────────────────────
    assert len(results) > 0, "Should produce at least one caption"

    required = {"type", "session", "user", "ts", "caption", "confidence", "mode"}
    for r in results:
        missing = required - set(r.keys())
        assert not missing, f"Missing keys: {missing}"
        assert r["type"] == "caption"
        assert r["mode"] in ("template", "llm", "uncertain")
        assert 0.0 <= r["confidence"] <= 1.0

    print("\n✓ Integration test passed!")
    return True


async def main():
    try:
        ok = await test_orchestrator_pipeline()
        if ok:
            print("\n" + "=" * 60)
            print("Member 3 pipeline ready for team integration!")
            print("=" * 60)
        return 0 if ok else 1
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
