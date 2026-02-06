"""Translate recognized tokens to captions with LLM + template fallback."""

import logging
import time
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
    t0 = time.perf_counter()
    token = pred["token"]
    conf = float(pred["confidence"])
    top2 = pred.get("top2", [token])
    
    # Override style from profile if needed
    if style is None:
        style = profile.get("style", "concise")

    # Apply profile bias: if the user has corrected this token before,
    # replace it with the most-frequently-corrected-to token.
    # Bias layout: { incorrect_token: { correct_token: count } }
    bias = profile.get("bias", {})
    if token in bias:
        corrections = bias[token]  # {correct_token: count}
        corrected_to = max(corrections, key=corrections.get)
        if corrections[corrected_to] >= 2:  # require ≥2 corrections to activate
            logger.info("Bias override: %s → %s (count=%d)", token, corrected_to, corrections[corrected_to])
            token = corrected_to

    gloss = PHRASES.get(token, {}).get("gloss", token)

    # Low confidence -> ask for clarification
    if conf < settings.CONF_MED:
        alt = top2[1] if len(top2) > 1 else token
        result = {
            "caption": f"Unclear — did you mean {token} or {alt}?",
            "mode": "uncertain",
            "confidence": conf
        }
        logger.info("translate latency=%.1fms  mode=%s", (time.perf_counter() - t0) * 1000, result["mode"])
        return result

    # Medium confidence -> use template
    if conf < settings.CONF_HIGH:
        template_text = TEMPLATES.get(token, {}).get(style, token)
        result = {
            "caption": template_text,
            "mode": "template",
            "confidence": conf
        }
        logger.info("translate latency=%.1fms  mode=%s", (time.perf_counter() - t0) * 1000, result["mode"])
        return result

    # High confidence -> try LLM polish with style preference, fallback to template
    llm_caption = gloss_to_english(gloss, style=style)
    if llm_caption:
        result = {"caption": llm_caption, "mode": "llm", "confidence": conf}
        logger.info("translate latency=%.1fms  mode=%s", (time.perf_counter() - t0) * 1000, result["mode"])
        return result

    # Fallback to template if LLM unavailable
    template_text = TEMPLATES.get(token, {}).get(style, token)
    result = {
        "caption": template_text,
        "mode": "template",
        "confidence": conf
    }
    logger.info("translate latency=%.1fms  mode=%s", (time.perf_counter() - t0) * 1000, result["mode"])
    return result
