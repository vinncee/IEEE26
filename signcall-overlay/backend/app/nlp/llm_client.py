"""LLM client for caption generation with anti-hallucination guardrails."""

import logging
from openai import OpenAI
from app import settings

logger = logging.getLogger(__name__)

_client = None


def get_client():
    global _client
    if _client is None and settings.OPENAI_API_KEY:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def gloss_to_english(gloss: str, style: str = "concise") -> str | None:
    """Convert sign language gloss to English with LLM.
    
    Returns None if API unavailable or fails (graceful degradation).
    """
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
        resp = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=50,
        )
        caption = resp.choices[0].message.content.strip()
        logger.debug(f"LLM caption: {caption}")
        return caption
    except Exception as e:
        logger.warning(f"LLM request failed ({e}) – falling back to templates")
        return None
