import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas.messages import FrameIn, CorrectionIn
from app.cv.preprocess import b64jpeg_to_bgr
from app.pipeline.orchestrator import process_frame
from app.nlp.profile import get_profile, apply_correction

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connected")
    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "frame":
                try:
                    frame_in = FrameIn(**data)
                    img = b64jpeg_to_bgr(frame_in.image_jpeg_b64)
                    if img is None:
                        continue
                    caption = await process_frame(
                        frame_in.session, frame_in.user, img, frame_in.ts,
                        style=frame_in.style
                    )
                    if caption:
                        await ws.send_json(caption)
                except Exception:
                    # Never let a single bad frame kill the connection
                    logger.exception("Error processing frame – skipping")

            elif msg_type == "correction":
                try:
                    corr = CorrectionIn(**data)
                    profile = get_profile(corr.session, corr.user)
                    apply_correction(profile, corr.incorrect_token, corr.correct_token)
                    await ws.send_json({
                        "type": "caption",
                        "session": corr.session,
                        "user": corr.user,
                        "ts": corr.ts,
                        "caption": f"Noted correction: {corr.correct_token}",
                        "confidence": 1.0,
                        "mode": "template",
                    })
                except Exception:
                    logger.exception("Error processing correction – skipping")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
