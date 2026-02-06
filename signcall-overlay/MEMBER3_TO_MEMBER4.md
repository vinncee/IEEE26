# Member 3 → Member 4 Handoff Guide

## What You're Receiving from Member 3

The recognition module is **complete and working**. It recognizes sign language phrases and passes structured data to your translator.

### Input to Your Module

When a gesture window is recognized, Member 3 produces:

```python
{
    "token": str,           # Phrase key: "HELLO", "THANKS", "REPEAT", or "SLOW"
    "confidence": float,    # Score 0.0-1.0 (higher = more certain)
    "top2": [str, str],     # Top 2 predictions for clarification UI
    "ts": int               # Timestamp when window ended
}
```

**Frequency:** ~Every 2 seconds at 5 FPS camera input (10-frame windows)

### Available Phrase Tokens

Defined in [phrase_set.py](app/recognition/phrase_set.py):

```python
PHRASES = {
    "HELLO": {"gloss": "HELLO"},
    "REPEAT": {"gloss": "PLEASE REPEAT"},
    "SLOW": {"gloss": "PLEASE SLOW"},
    "THANKS": {"gloss": "THANK YOU"},
}
```

## Your Gating Logic (from README)

Use these confidence thresholds for translation strategy:

```python
def translate(pred: dict, profile) -> dict:
    token = pred["token"]
    confidence = pred["confidence"]
    
    if confidence >= 0.7:  # CONF_HIGH
        # Use LLM for polished captions
        caption = call_llm(token, ...)
        mode = "llm"
        
    elif confidence >= 0.5:  # CONF_MED
        # Use templates for medium confidence
        caption = TEMPLATES[token]
        mode = "template"
        
    else:  # Low confidence
        # Ask for clarification
        top2 = pred["top2"]
        caption = f"Unclear — did you mean {top2[0]} or {top2[1]}?"
        mode = "uncertain"
    
    return {
        "caption": caption,
        "confidence": confidence,
        "mode": mode
    }
```

## Output Format Required

Your `translate()` function must return:

```python
{
    "caption": str,       # Human-readable caption
    "confidence": float,  # Pass through from pred
    "mode": str           # One of: "llm", "template", "uncertain"
}
```

This gets wrapped into the WebSocket message:
```python
{
    "type": "caption",
    "session": str,
    "user": str,
    "ts": int,
    "caption": str,
    "confidence": float,
    "mode": str
}
```

## Expected Behavior

### High Confidence (conf ≥ 0.7)
- **Input:** `{"token": "HELLO", "confidence": 0.85, ...}`
- **Output:** `{"caption": "Hello there!", "mode": "llm"}`
- User sees polished LLM-generated caption

### Medium Confidence (0.5 ≤ conf < 0.7)
- **Input:** `{"token": "THANKS", "confidence": 0.62, ...}`
- **Output:** `{"caption": "Thank you", "mode": "template"}`
- User sees template response

### Low Confidence (conf < 0.5)
- **Input:** `{"token": "SLOW", "confidence": 0.25, "top2": ["SLOW", "REPEAT"]}`
- **Output:** `{"caption": "Unclear — did you mean SLOW or REPEAT?", "mode": "uncertain"}`
- Frontend shows clarification prompt with correction buttons

## Integration with Member 1 (Frontend)

Frontend expects captions with mode field to:
- **mode: "llm"** → Display polished caption
- **mode: "template"** → Display simpler caption
- **mode: "uncertain"** → Show "Did you mean?" prompt with buttons

When user clicks correction button:
- Signal to Member 3 to bias that token (future enhancement)
- Or just accept it and move on

## Testing Your Integration

**Quick smoke test:**
1. Start backend with test data
2. Verify captions appear in frontend with correct mode
3. Check that high-confidence predictions are "llm" mode
4. Check that low-confidence predictions are "uncertain" mode
5. Verify caption text matches expected gloss

## If Something Breaks

### Symptom: All predictions are "uncertain" mode
- **Cause:** Confidence consistently low
- **Solution:** Member 3 prototypes need calibration with real hand data
- **Action:** Collect gesture examples and update PROTOTYPES in [classifier.py](app/recognition/classifier.py)

### Symptom: Wrong phrases recognized
- **Cause:** Prototype feature vectors don't match real features
- **Solution:** Manual calibration or ML training
- **Action:** See [MEMBER3_SUMMARY.md](MEMBER3_SUMMARY.md#next-steps-prototype-tuning)

### Symptom: Captions lag or appear out of order
- **Cause:** Race condition in WebSocket or async handling
- **Solution:** Check buffer ordering in [orchestrator.py](app/pipeline/orchestrator.py)

## Questions?

The interface is locked (won't change per README rule #1):
- Token format: string from PHRASES.keys()
- Confidence: float [0-1]
- Output format: dict with "caption", "confidence", "mode"

Build confidently within this interface!
