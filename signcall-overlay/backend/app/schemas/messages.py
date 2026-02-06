from pydantic import BaseModel
from typing import Literal, Optional

class FrameIn(BaseModel):
    type: Literal["frame"]
    session: str
    user: str
    ts: int
    image_jpeg_b64: str
    style: Optional[Literal["concise", "detailed"]] = "concise"

class CaptionOut(BaseModel):
    type: Literal["caption"]
    session: str
    user: str
    ts: int
    caption: str
    confidence: float
    mode: Literal["template", "llm", "uncertain"]

class CorrectionIn(BaseModel):
    type: Literal["correction"]
    session: str
    user: str
    ts: int
    incorrect_token: str
    correct_token: str
