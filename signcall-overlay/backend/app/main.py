import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ws import router as ws_router

# Configure root logger so our app messages are visible
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy third-party HTTP/model logs
for _noisy in ("httpx", "httpcore", "openai", "mediapipe", "absl"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

app = FastAPI(title="SignCall Overlay Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)


@app.on_event("startup")
async def _startup():
    from app.cv.mediapipe_extractor import HAS_MEDIAPIPE

    logger = logging.getLogger("app.startup")
    if HAS_MEDIAPIPE:
        logger.info("✅ MediaPipe solutions available – CV extraction is LIVE")
    else:
        logger.warning("⚠️  MediaPipe solutions NOT available – CV returns empty landmarks")
