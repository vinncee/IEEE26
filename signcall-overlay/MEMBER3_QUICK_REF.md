# Member 3 Quick Reference Card

## Your Responsibilities (COMPLETE ✓)

```
Step 1: Signing Detection (detector.py)      ✓ DONE
Step 2: Feature Extraction (classifier.py)   ✓ DONE
Step 3: Phrase Classification (classifier.py) ✓ DONE
Step 4: Temporal Smoothing (smoothing.py)    ✓ DONE
Step 5: Integration (orchestrator.py)        ✓ DONE
```

## Files You Modified

| File | Changes | Lines |
|------|---------|-------|
| `app/recognition/detector.py` | Sign detection via motion analysis | 50 |
| `app/recognition/classifier.py` | Feature extraction + prototype classifier | 120 |
| `app/recognition/smoothing.py` | Majority-vote temporal smoothing | 85 |
| `app/pipeline/orchestrator.py` | Pass session_id to smoothing | 1 |

## What Each Function Does

### `is_signing(window) -> bool`
- Calculates hand movement magnitude across 10 frames
- Returns `True` if avg motion > 5 pixels/frame
- Used to avoid classifying idle hands as phrases

### `extract_features(window) -> np.ndarray`
- Converts 10-frame landmark window into 4-dim feature vector
- Features: [hand_distance, hand_height, motion_magnitude, hand_speed]
- All normalized to [0, 1] range

### `predict(window) -> dict`
- Compares extracted features to 4 prototype vectors
- Returns closest match with confidence score
- Format: `{"token": str, "confidence": float, "top2": [...], "ts": int}`

### `smooth(pred, session_id) -> dict`
- Tracks last 4 predictions per session
- Majority voting: boosts confidence if token matches majority
- Debounces outliers by reducing confidence and using majority token

## Key Numbers (Tuning Parameters)

- **Motion threshold:** 5.0 pixels/frame for signing detection
- **Confidence floor:** 0.1 (minimum confidence returned)
- **Confidence ceiling:** 0.99 (maximum with boosting)
- **Smoothing window:** 4 predictions (majority vote over 4)
- **Confidence boost:** +0.2 max when token matches majority
- **Debounce factor:** ×0.6 when token differs from majority

## Output Examples

### High Confidence
```json
{
  "token": "HELLO",
  "confidence": 0.85,
  "top2": ["HELLO", "THANKS"],
  "ts": 9000
}
```

### Low Confidence
```json
{
  "token": "SLOW",
  "confidence": 0.25,
  "top2": ["SLOW", "THANKS"],
  "ts": 10000
}
```

## Calibration Needed?

If all predictions are same token or low confidence:

1. Record real HELLO, THANKS, REPEAT, SLOW gestures
2. Extract features from each (10-frame windows)
3. Update PROTOTYPES in [classifier.py](app/recognition/classifier.py):
   ```python
   PROTOTYPES = {
       "HELLO": np.array([real_hello_features]),
       "THANKS": np.array([real_thanks_features]),
       ...
   }
   ```

## Testing Commands

```bash
# Check syntax
python -m py_compile app/recognition/*.py

# Run unit tests
python test_recognition.py

# Run integration tests
python test_integration.py
```

## Before Demo Day

- [ ] Calibrate prototypes with real hand data
- [ ] Verify 5-10 FPS processing speed
- [ ] Test all 4 phrases work
- [ ] Confirm low-confidence shows "uncertain" mode
- [ ] Check no lag in caption updates

## Handoff to Member 4

**Interface locked.** Don't change these:
- Token format: exactly `PHRASES.keys()` (4 values)
- Confidence: always 0.0-1.0 float
- Output dict: `{"token": str, "confidence": float, "top2": [...], "ts": int}`

Member 4 uses confidence thresholds to choose:
- ≥ 0.7 → LLM polish ("llm" mode)
- 0.5-0.7 → Templates ("template" mode)
- < 0.5 → Clarification ("uncertain" mode)

---

**Status:** Implementation complete. Ready for live testing with real camera input.
