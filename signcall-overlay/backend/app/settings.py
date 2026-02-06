import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
CAPTURE_FPS = float(os.getenv("CAPTURE_FPS", "8"))
WINDOW_SECONDS = float(os.getenv("WINDOW_SECONDS", "0.9"))
CONF_HIGH = float(os.getenv("CONF_HIGH", "0.80"))
CONF_MED = float(os.getenv("CONF_MED", "0.55"))

# Set DEBUG_TOKENS=1 in .env to bypass real CV/classifier and cycle
# through all 4 tokens at low/med/high confidence for translator testing.
DEBUG_TOKENS = os.getenv("DEBUG_TOKENS", "").strip().lower() in ("1", "true", "yes")
