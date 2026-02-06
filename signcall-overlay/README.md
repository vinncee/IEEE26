# SignCall Overlay

Video calls became the default workplace—and they quietly excluded signers.
Deaf and non-verbal people shouldn’t need a separate accommodation to be heard. With signcall, signing becomes readable instantly for everyone in the call through real-time caption overlays, helping people participate naturally in interviews, classes, telehealth, and team meetings.


## Features
- Live webcam capture with caption overlay UI
- WebSocket protocol for frames and captions
- CV pipeline with MediaPipe landmarks
- Recognition + smoothing + translator (template/LLM)
- Optional LLM polish with safe fallback when no API key is set

## Tech Stack
- Frontend: Next.js + React + Tailwind
- Backend: FastAPI + MediaPipe + OpenCV + NumPy
- Optional LLM: OpenAI API

## Quickstart

### 1) Frontend
```bash
npm install
npm run dev
```

### 2) Backend
Requires Python 3.11 (MediaPipe does not support 3.12+).

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000 --log-level debug
```

The frontend connects to `ws://localhost:8000/ws` by default.

## Environment Variables
Create a `backend/.env` file if you want to customize behavior.

- `OPENAI_API_KEY`: Optional. Enables LLM polishing.
- `LLM_MODEL`: Default `gpt-4.1-mini`.
- `CAPTURE_FPS`: Frame sampling target (default `8`).
- `WINDOW_SECONDS`: Recognition window size (default `0.9`).
- `CONF_HIGH`: LLM threshold (default `0.80`).
- `CONF_MED`: Template threshold (default `0.55`).
- `DEBUG_TOKENS`: Set to `1` to bypass CV/classifier and cycle debug tokens.

## WebSocket Message Formats
**Incoming frame**:
```json
{
  "type": "frame",
  "session": "room-1",
  "user": "alice",
  "ts": 1730000000000,
  "image_jpeg_b64": "...",
  "style": "concise"
}
```

**Incoming correction**:
```json
{
  "type": "correction",
  "session": "room-1",
  "user": "alice",
  "ts": 1730000000000,
  "incorrect_token": "HELLO",
  "correct_token": "THANKS"
}
```

**Outgoing caption**:
```json
{
  "type": "caption",
  "session": "room-1",
  "user": "alice",
  "ts": 1730000000000,
  "caption": "Hello",
  "confidence": 0.83,
  "mode": "llm"
}
```

## Project Layout
- `frontend/`: Next.js UI and overlay components
- `backend/`: FastAPI server and CV/recognition pipeline
- `shared/`: Architecture notes and shared docs

## Notes For Contributors
- Keep the WebSocket message format stable.
- If you add fields to message schemas, update both `backend/app/schemas/messages.py` and `frontend/src/types/messages.ts`.
- Maintain end-to-end runnability even when a module is stubbed.

For the original team build guidance, see `TEAM_BUILD_INSTRUCTIONS.md`.

## Troubleshooting
- **MediaPipe install issues**: confirm Python 3.11 and a clean venv.
- **No captions**: start backend first, then frontend; check `DEBUG_TOKENS`.
- **WebSocket disconnects**: check for invalid JSON or bad base64 frames.

## Future Enhancements
- Multi‑participant call support with active‑signer detection.
- Personalization profiles per user (style, vocabulary bias, preferred phrasing).
- Better low‑confidence handling (clarification prompts, top‑k suggestions in UI).
- On‑device / edge inference modes for privacy‑sensitive settings.
- Expanded language support and community‑driven phrase packs.
