"""LLM client for caption generation with anti-hallucination guardrails."""

import logging
import time
from openai import OpenAI
from app import settings

logger = logging.getLogger(__name__)

_client = None

# Simple cooldown to avoid 429 rate limits: cache recent results
_llm_cache: dict[str, tuple[float, str]] = {}  # key -> (timestamp, result)
_LLM_COOLDOWN = 5.0  # seconds between LLM calls for the same gloss+style


def get_client():
    global _client
    if _client is None and settings.OPENAI_API_KEY:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def gloss_to_english(gloss: str, style: str = "concise") -> str | None:
    """Convert sign language gloss to English with LLM.
    
    Returns None if API unavailable or fails (graceful degradation).
    """
    # Check cache first to avoid rate limits
    cache_key = f"{gloss}:{style}"
    now = time.perf_counter()
    if cache_key in _llm_cache:
        cached_time, cached_result = _llm_cache[cache_key]
        if now - cached_time < _LLM_COOLDOWN:
            return cached_result

    client = get_client()
    if client is None:
        logger.warning("OpenAI client not configured – falling back to templates")
        return None

    system_message = (
        "You are a sign language caption generator. Your job is to convert sign language glosses "
        "into natural English captions. Rules:\n"
        "1. NEVER add information not in the gloss\n"
        "2. NEVER hallucinate context or meaning\n"
        "3. Keep concise style under 12 words\n"
        "4. Keep detailed style under 20 words\n"
        "5. Be natural and conversational"
    )

    prompt = (
        f"Convert this sign language gloss to {style} English:\n"
        f"Gloss: {gloss}\n"
        f"Output only the caption, no explanations."
    )

    try:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=50,
        )
        llm_ms = (time.perf_counter() - t0) * 1000
        caption = resp.choices[0].message.content.strip()
        logger.info("LLM latency=%.0fms  caption=%r", llm_ms, caption)
        _llm_cache[cache_key] = (time.perf_counter(), caption)
        return caption
    except Exception as e:
        logger.warning(f"LLM request failed ({e}) – falling back to templates")
        return None
