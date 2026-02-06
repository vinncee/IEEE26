#!/usr/bin/env python3
"""Calibration tool — record real gestures and compute optimal prototypes.

Usage:
    cd backend
    .venv/bin/python3 calibrate.py

The script will:
  1. Open your webcam
  2. Ask you to perform each gesture (HELLO, THANKS, REPEAT, SLOW) several times
  3. Extract the 7-D feature vector for each recording
  4. Compute the mean prototype for each gesture
  5. Write the new prototypes into classifier.py automatically

Press SPACE to start/stop each recording. Press Q to quit.
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from app.cv.mediapipe_extractor import extract_landmarks
from app.cv.types import LandmarkFrame, LandmarkWindow
from app.recognition.classifier import extract_features
from app.recognition.detector import is_signing

TOKENS = ["HELLO", "THANKS", "REPEAT", "SLOW"]
WINDOW_SIZE = 10           # frames per recognition window
SAMPLES_PER_GESTURE = 5    # how many recordings per gesture
FPS_TARGET = 8             # match the backend capture rate

# ASL descriptions to guide the user
GESTURE_GUIDE = {
    "HELLO":  "Open hand, wave near face (palm outward)",
    "THANKS": "Flat hand at chin, move outward",
    "REPEAT": "Circular hand motion, hands apart",
    "SLOW":   "One hand slides slowly down the other",
}


def capture_window(cap, window_size: int = WINDOW_SIZE) -> LandmarkWindow:
    """Capture one window of landmark frames from the webcam."""
    frames: list[LandmarkFrame] = []
    interval = 1.0 / FPS_TARGET

    for i in range(window_size):
        t0 = time.time()
        ret, bgr = cap.read()
        if not ret:
            continue
        ts = int(time.time() * 1000)
        lf = extract_landmarks(bgr, ts)
        frames.append(lf)

        # Show the frame with a recording indicator
        display = bgr.copy()
        cv2.putText(display, f"RECORDING  frame {i+1}/{window_size}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("Calibration", display)
        cv2.waitKey(1)

        # Maintain target FPS
        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    if not frames:
        return LandmarkWindow(frames=[], ts_start=0, ts_end=0)

    return LandmarkWindow(
        frames=frames,
        ts_start=frames[0].ts,
        ts_end=frames[-1].ts,
    )


def run_calibration():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    all_features: dict[str, list[np.ndarray]] = {tok: [] for tok in TOKENS}

    print("\n" + "=" * 60)
    print("  GESTURE CALIBRATION TOOL")
    print("=" * 60)
    print("\nFor each gesture you will record it 5 times.")
    print("  SPACE  = start recording (hold the gesture for ~1.5 sec)")
    print("  Q      = quit early\n")

    for token in TOKENS:
        print(f"\n{'─' * 60}")
        print(f"  Gesture: {token}")
        print(f"  How-to:  {GESTURE_GUIDE[token]}")
        print(f"{'─' * 60}")

        sample_idx = 0
        while sample_idx < SAMPLES_PER_GESTURE:
            # Show preview
            while True:
                ret, bgr = cap.read()
                if not ret:
                    continue
                display = bgr.copy()
                cv2.putText(display, f"{token}  sample {sample_idx+1}/{SAMPLES_PER_GESTURE}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)
                cv2.putText(display, "Press SPACE to record, Q to quit",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                cv2.imshow("Calibration", display)
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):
                    break
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\nCalibration aborted.")
                    return None

            # Record one window
            window = capture_window(cap)

            if not window.frames or len(window.frames) < WINDOW_SIZE // 2:
                print(f"  ⚠ Too few frames captured, try again")
                continue

            feats = extract_features(window)
            signing = is_signing(window)
            hands_count = sum(1 for f in window.frames if f.hands)

            print(f"  Sample {sample_idx+1}: features=[{', '.join(f'{v:.3f}' for v in feats)}]"
                  f"  hands={hands_count}/{len(window.frames)}  signing={signing}")

            if hands_count < WINDOW_SIZE // 3:
                print(f"  ⚠ Too few hands detected ({hands_count}), try again")
                continue

            all_features[token].append(feats)
            sample_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # ── Compute prototypes ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CALIBRATION RESULTS")
    print("=" * 60)

    prototypes: dict[str, np.ndarray] = {}
    for token in TOKENS:
        feats_list = all_features[token]
        if not feats_list:
            print(f"  {token}: NO DATA — keeping old prototype")
            continue
        mean = np.mean(feats_list, axis=0)
        std = np.std(feats_list, axis=0)
        prototypes[token] = mean
        print(f"  {token:8s}: mean=[{', '.join(f'{v:.3f}' for v in mean)}]"
              f"  std=[{', '.join(f'{v:.3f}' for v in std)}]")

    # Check separation
    print(f"\n  Feature separation:")
    toks = list(prototypes.keys())
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            d = float(np.linalg.norm(prototypes[toks[i]] - prototypes[toks[j]]))
            status = "✓" if d > 0.05 else "⚠ LOW"
            print(f"    {toks[i]:8s} ↔ {toks[j]:8s}  dist={d:.4f}  {status}")

    return prototypes


def write_prototypes(prototypes: dict[str, np.ndarray]):
    """Write new prototypes into classifier.py."""
    classifier_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "app", "recognition", "classifier.py"
    )

    with open(classifier_path, "r") as f:
        content = f.read()

    # Build the new PROTOTYPES block
    lines = ["PROTOTYPES = {"]
    comments = {
        "HELLO":  "calibrated from live webcam recordings",
        "THANKS": "calibrated from live webcam recordings",
        "REPEAT": "calibrated from live webcam recordings",
        "SLOW":   "calibrated from live webcam recordings",
    }
    for token in TOKENS:
        if token in prototypes:
            arr_str = ", ".join(f"{v:.4f}" for v in prototypes[token])
            lines.append(f'    "{token}":  np.array([{arr_str}]),    # {comments[token]}')
    lines.append("}")
    new_block = "\n".join(lines)

    # Find and replace the existing PROTOTYPES block
    import re
    pattern = r"PROTOTYPES = \{[^}]+\}"
    if not re.search(pattern, content, re.DOTALL):
        print("ERROR: Could not find PROTOTYPES block in classifier.py")
        return False

    new_content = re.sub(pattern, new_block, content, count=1, flags=re.DOTALL)

    with open(classifier_path, "w") as f:
        f.write(new_content)

    print(f"\n✓ Prototypes written to {classifier_path}")
    return True


def save_raw_data(prototypes: dict[str, np.ndarray]):
    """Save calibration data as JSON for reproducibility."""
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "calibration_data.json"
    )
    data = {tok: proto.tolist() for tok, proto in prototypes.items()}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Raw calibration data saved to {out_path}")


if __name__ == "__main__":
    prototypes = run_calibration()

    if prototypes is None:
        sys.exit(1)

    print("\n" + "─" * 60)
    response = input("Write these prototypes to classifier.py? [Y/n] ").strip().lower()
    if response in ("", "y", "yes"):
        save_raw_data(prototypes)
        write_prototypes(prototypes)
        print("\n✅ Calibration complete! Restart the backend to use new prototypes.")
        print("   The backend will auto-reload if running with --reload.")
    else:
        # Still save the data so they can apply it later
        save_raw_data(prototypes)
        print("\nPrototypes NOT written to classifier.py.")
        print("You can manually copy them from calibration_data.json.")
