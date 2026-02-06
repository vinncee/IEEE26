import logging

from app.cv.types import LandmarkWindow
from app.cv.mediapipe_extractor import extract_landmarks
from app.recognition.classifier import predict
from app.recognition.smoothing import smooth
from app.nlp.translator import translate
from app.nlp.profile import get_profile

logger = logging.getLogger(__name__)

# Per-session sliding buffer of LandmarkFrames
_buffers: dict[str, list] = {}
WINDOW_SIZE = 10       # number of frames in one recognition window
MAX_BUF_LEN = 30       # cap to prevent unbounded memory growth


def _get_buf_key(session: str, user: str) -> str:
    return f"{session}:{user}"


async def process_frame(session: str, user: str, frame_bgr, ts: int):
    key = _get_buf_key(session, user)
    buf = _buffers.setdefault(key, [])

    lf = extract_landmarks(frame_bgr, ts)
    buf.append(lf)

    # Keep buffer bounded
    if len(buf) > MAX_BUF_LEN:
        _buffers[key] = buf[-MAX_BUF_LEN:]
        buf = _buffers[key]

    # Need at least WINDOW_SIZE frames before running recognition
    if len(buf) < WINDOW_SIZE:
        return None

    frames = buf[-WINDOW_SIZE:]
    window = LandmarkWindow(
        frames=frames,
        ts_start=frames[0].ts,
        ts_end=frames[-1].ts,
    )

    pred = predict(window)
    # Pass session:user as session_id for per-session smoothing history
    pred = smooth(pred, session_id=f"{session}:{user}")
    profile = get_profile(session, user)
    out = translate(pred, profile)

    # Count how many frames in the window had at least one hand
    hands_count = sum(1 for f in frames if f.hands)

    return {
        "type": "caption",
        "session": session,
        "user": user,
        "ts": ts,
        "caption": out["caption"],
        "confidence": out["confidence"],
        "mode": out["mode"],
        "hands_detected": hands_count,
    }
