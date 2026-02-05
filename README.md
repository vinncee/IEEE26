# IEEE26

Here are **shareable, step-by-step instructions** for how each member should build on top of the backbone you just generated. You can paste this directly into your team chat / Notion.

---

# Team Build Instructions (on top of the backbone)

## 0) Common rules for everyone (to avoid conflicts)

1. **Do not change the WS message formats** unless everyone agrees.

   * `type: "frame"` in, `type: "caption"` out, optional `type: "correction"`.
2. Work only in your assigned folders (below).
3. If you need a new field in JSON, add it in BOTH:

   * `backend/app/schemas/messages.py`
   * `frontend/src/types/messages.ts`
4. Always keep the pipeline “end-to-end runnable” (even if your module is stubbed).

---

# Member 1 — Frontend (React + Overlay UI)

**Owns:** `frontend/src/pages`, `frontend/src/components`, `frontend/src/services`

### Goal

Turn the current single-stream demo into a **video-call UI** with:

* signer tile + caption overlay
* active signer indicator (optional)
* correction UI (already stubbed)
* optional: 2-person WebRTC

### What to build next

1. **Make UI demo-clean**

   * Improve caption placement, readability
   * Display “Low confidence” clearly when mode is `"uncertain"`
2. **Add Debug Mode (optional but great)**

   * Add checkbox in `Controls.tsx`
   * When enabled, show extra backend debug messages (if backend adds them later)
3. **Upgrade from single webcam to 2-person WebRTC (if time)**

   * Keep sending frames only from the signer tile (to save bandwidth)
   * Make sure the tile you capture is the signer tile

### Integration guidance

* Keep frame sending at **5–10 FPS**
* Avoid sending frames if `videoWidth/videoHeight` are 0

### Definition of done

* You can start frontend alone and see caption updates live
* Correction button updates caption text (“Noted correction: …”)

---

# Member 2 — Backend CV + Frame Processing (FastAPI + MediaPipe)

**Owns:** `backend/app/api/ws.py`, `backend/app/cv/*`, WS stability

### Goal

Replace the stub landmark extractor so backend produces real landmark data.

### What to build next

1. Implement MediaPipe extraction in:

   * `backend/app/cv/mediapipe_extractor.py`
2. Landmarks should include at least **hands**:

   * number of hands detected
   * keypoints coordinates
3. Optional but helpful:

   * pose landmarks (upper body)
   * face landmarks (for expression)

### Integration guidance

* Don’t change orchestrator signature:

  * `extract_landmarks(frame_bgr, ts) -> LandmarkFrame`
* Use `LandmarkFrame.hands/pose/face` fields (can store lists/dicts)

### Quick sanity check

* Print/log:

  * `hands_detected = len(hands)` per frame
* Add *optional* debug message emission later (only if needed)

### Definition of done

* With webcam on, landmarks are non-null frequently
* Performance is stable at ~5–10 FPS backend-side

---

# Member 3 — Recognition + Temporal Model + Smoothing

**Owns:** `backend/app/recognition/*`

### Goal

Replace the rotating stub classifier with real **phrase recognition** on landmark windows.

### What to build next (in order)

1. **Signing detection** (`detector.py`)

   * Detect when hands are moving enough to count as “signing”
   * Can be simple: movement threshold across last N frames
2. **Feature extraction** inside `classifier.py`

   * Convert `LandmarkWindow` → feature vector
   * Examples:

     * relative hand position to shoulders
     * distance between fingertips
     * motion magnitude
3. **Simple classifier**

   * Start with a rule-based or nearest-prototype approach first
   * If you have data, use scikit-learn (SVM/RandomForest)
4. **Smoothing**

   * Use `smoothing.py` to reduce flicker
   * Majority vote over last k predictions OR debounce stable token for X ms

### Integration guidance

* Keep output format exactly:

```py
{
  "token": str,
  "confidence": float,
  "top2": [str, str],
  "ts": window.ts_end
}
```

* Start with 4–8 phrases, only expand once stable
* Update phrase set only in:

  * `backend/app/recognition/phrase_set.py`
  * and mirror in `shared/phrase_list.md`

### Definition of done

* When you perform a known gesture, token stabilizes after smoothing
* Confidence reflects quality (high for clear signing, low for messy)

---

# Member 4 — NLP Translator + Personalization + OpenAI Integration

**Owns:** `backend/app/nlp/*`, and personalization behavior

### Goal

Make captions readable, controllable, and safe (no hallucinations).

### What to build next

1. **Improve templates**

   * Add clean human captions to `templates.py`
2. **LLM prompting improvements** (`llm_client.py`)

   * Keep outputs short
   * No extra info / no hallucination
   * Add a system message if needed
3. **Personalization controls**

   * Store user preferences in `profile.py`
   * Add:

     * `style`: concise/detailed
     * `bias`: token weights
4. **Use corrections**

   * When correction comes in, bias the correct token
   * Optional: incorporate bias into recognition (hand to Member 3 later)
5. **Safety gating**

   * Only call LLM if `confidence >= CONF_HIGH`
   * Otherwise use templates
   * For low confidence: ask “Did you mean X or Y?”

### Integration guidance

* Translator must always return:

```py
{ "caption": str, "mode": "...", "confidence": float }
```

* Keep the backend always functional even if OpenAI key missing

### Definition of done

* With high confidence, captions are polished by LLM
* With medium confidence, templates are used
* With low confidence, clarification prompt appears

---

# Integration checkpoints (everyone should align here)

## Checkpoint A (by ~12:30)

* Frontend sends frames
* Backend responds with captions (even stubbed)
* UI shows captions live

## Checkpoint B (by ~15:00)

* MediaPipe landmarks working
* Recognizer outputs *something meaningful* (even if only 4 phrases)
* Translator gating works (template vs LLM)

## Checkpoint C (final polish)

* Smoothing reduces flicker
* Correction feedback visibly changes behavior
* Demo script ready + backup recording

---

# “Don’t break the demo” safety tips

* If your module is incomplete, keep a fallback:

  * CV fails → still send a caption like “No hands detected”
  * classifier fails → token = “UNKNOWN”, confidence low
  * LLM fails → template fallback

---

If you want, I can also write a **single “integration smoke test” checklist** (5 minutes) that the team runs before demo rehearsal.
