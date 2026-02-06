"""Translate recognized tokens to captions with LLM + template fallback."""

import logging
from app import settings
from app.recognition.phrase_set import PHRASES
from app.nlp.templates import TEMPLATES
from app.nlp.llm_client import gloss_to_english

logger = logging.getLogger(__name__)


def translate(pred: dict, profile: dict, style: str = "concise") -> dict:
    """Translate recognized token to caption.
    
    Applies profile bias, respects user style preference, and uses LLM
    for high-confidence predictions (with graceful fallback to templates).
    """
    token = pred["token"]
    conf = float(pred["confidence"])
    top2 = pred.get("top2", [token])
    
    # Override style from profile if needed
    if style is None:
        style = profile.get("style", "concise")

    # Apply profile bias: if user frequently corrects a token to another, use the correction
    bias = profile.get("bias", {})
    if bias and token in bias:
        # Find most-corrected-to token
        corrected_to = max(bias, key=lambda k: bias.get(k, 0))
        if bias[corrected_to] > bias.get(token, 0):
            logger.debug(f"Applying bias: {token} → {corrected_to}")
            token = corrected_to

    gloss = PHRASES.get(token, {}).get("gloss", token)

    # Low confidence -> ask for clarification
    if conf < settings.CONF_MED:
        alt = top2[1] if len(top2) > 1 else token
        return {
            "caption": f"Unclear — did you mean {token} or {alt}?",
            "mode": "uncertain",
            "confidence": conf
        }

    # Medium confidence -> use template
    if conf < settings.CONF_HIGH:
        template_text = TEMPLATES.get(token, {}).get(style, token)
        return {
            "caption": template_text,
            "mode": "template",
            "confidence": conf
        }

    # High confidence -> try LLM polish with style preference, fallback to template
    llm_caption = gloss_to_english(gloss, style=style)
    if llm_caption:
        return {"caption": llm_caption, "mode": "llm", "confidence": conf}

    # Fallback to template if LLM unavailable
    template_text = TEMPLATES.get(token, {}).get(style, token)
    return {
        "caption": template_text,
        "mode": "template",
        "confidence": conf
    }
