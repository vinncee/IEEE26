import logging
import time

from app import settings
from app.cv.types import LandmarkWindow
from app.cv.mediapipe_extractor import extract_landmarks
from app.recognition.detector import is_signing
from app.recognition.classifier import predict, predict_with_tm
from app.recognition.smoothing import smooth
from app.nlp.translator import translate
from app.nlp.profile import get_profile

logger = logging.getLogger(__name__)

# Per-session sliding buffer of LandmarkFrames + raw BGR frames
_buffers: dict[str, list] = {}
_frame_buffers: dict[str, list] = {}   # raw BGR frames for TM model
WINDOW_SIZE = 10       # number of frames in one recognition window
MAX_BUF_LEN = 30       # cap to prevent unbounded memory growth

# ── Debug token rotation (enabled by DEBUG_TOKENS=1 in .env) ──
_DEBUG_TOKENS = ["HELLO", "THANKS", "REPEAT", "SLOW"]
_DEBUG_CONFS  = [0.90, 0.65, 0.30, 0.85]  # high, med, low, high
_debug_counter: dict[str, int] = {}


def _get_buf_key(session: str, user: str) -> str:
    return f"{session}:{user}"


async def process_frame(session: str, user: str, frame_bgr, ts: int, style: str = "concise"):
    t_start = time.perf_counter()

    # ── Debug mode: cycle through tokens every ~2s (16 frames at 8fps) ──
    if settings.DEBUG_TOKENS:
        key = _get_buf_key(session, user)
        _debug_counter[key] = _debug_counter.get(key, 0) + 1
        # Emit a caption every 16 frames (~2 seconds)
        if _debug_counter[key] % 16 != 0:
            return None
        idx = (_debug_counter[key] // 16) % len(_DEBUG_TOKENS)
        token = _DEBUG_TOKENS[idx]
        conf = _DEBUG_CONFS[idx]
        pred = {"token": token, "confidence": conf, "top2": [token, _DEBUG_TOKENS[(idx+1) % len(_DEBUG_TOKENS)]], "ts": ts}
        profile = get_profile(session, user)
        out = translate(pred, profile, style=style)
        logger.info(
            "[DEBUG] token=%s  conf=%.2f  mode=%s  caption=%s",
            token, conf, out["mode"], out["caption"],
        )
        return {
            "type": "caption", "session": session, "user": user, "ts": ts,
            "caption": out["caption"], "confidence": out["confidence"],
            "mode": out["mode"], "hands_detected": -1,
        }

    key = _get_buf_key(session, user)
    buf = _buffers.setdefault(key, [])
    fbuf = _frame_buffers.setdefault(key, [])

    lf = extract_landmarks(frame_bgr, ts)
    buf.append(lf)
    fbuf.append(frame_bgr)

    # Keep buffers bounded
    if len(buf) > MAX_BUF_LEN:
        _buffers[key] = buf[-MAX_BUF_LEN:]
        buf = _buffers[key]
    if len(fbuf) > MAX_BUF_LEN:
        _frame_buffers[key] = fbuf[-MAX_BUF_LEN:]
        fbuf = _frame_buffers[key]

    # Need at least WINDOW_SIZE frames before running recognition
    if len(buf) < WINDOW_SIZE:
        return None

    frames = buf[-WINDOW_SIZE:]
    raw_frames = fbuf[-WINDOW_SIZE:]
    window = LandmarkWindow(
        frames=frames,
        ts_start=frames[0].ts,
        ts_end=frames[-1].ts,
    )

    # Skip classification when hands are idle / absent
    if not is_signing(window):
        logger.debug("ts=%d  not signing – skipping recognition", ts)
        return None

    pred = predict_with_tm(window, raw_frames, session_id=key, ts=ts)
    # Gate: suppress very low-confidence predictions (noise)
    if pred["confidence"] < 0.35:
        logger.debug("ts=%d  low confidence %.2f – skipping", ts, pred["confidence"])
        return None
    # Pass session:user as session_id for per-session smoothing history
    pred = smooth(pred, session_id=key)
    profile = get_profile(session, user)
    out = translate(pred, profile, style=style)

    # Count how many frames in the window had at least one hand
    hands_count = sum(1 for f in frames if f.hands)

    pipeline_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "pipeline latency=%.0fms  token=%s  conf=%.2f  mode=%s  hands=%d",
        pipeline_ms, pred["token"], out["confidence"], out["mode"], hands_count,
    )

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
