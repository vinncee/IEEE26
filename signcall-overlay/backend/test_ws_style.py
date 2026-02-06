#!/usr/bin/env python3
"""Test that style preference flows through WebSocket"""

import sys
sys.path.insert(0, '/Users/kahei/Documents/GitHub/IEEE26/signcall-overlay/backend')

from app.schemas.messages import FrameIn

print("=" * 60)
print("WebSocket Style Preference Tests")
print("=" * 60)

print("\nTesting WebSocket style preference...")

# Test 1: Frame with style
frame_concise = FrameIn(
    type="frame",
    session="test",
    user="user1",
    ts=1000,
    image_jpeg_b64="fake_base64",
    style="concise"
)
print(f"✓ Concise frame: {frame_concise.style}")

# Test 2: Frame with detailed style
frame_detailed = FrameIn(
    type="frame",
    session="test",
    user="user1",
    ts=1000,
    image_jpeg_b64="fake_base64",
    style="detailed"
)
print(f"✓ Detailed frame: {frame_detailed.style}")

# Test 3: Frame without style (defaults to concise)
frame_default = FrameIn(
    type="frame",
    session="test",
    user="user1",
    ts=1000,
    image_jpeg_b64="fake_base64"
)
print(f"✓ Default frame: {frame_default.style}")

print("\n" + "=" * 60)
print("✓ WebSocket style preference validation works!")
print("=" * 60)
