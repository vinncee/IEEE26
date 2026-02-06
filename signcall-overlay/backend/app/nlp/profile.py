from typing import Dict

_profiles: Dict[str, Dict] = {}

def get_profile(session: str, user: str) -> Dict:
    key = f"{session}:{user}"
    if key not in _profiles:
        _profiles[key] = {"style": "concise", "bias": {}}
    return _profiles[key]

def apply_correction(profile: Dict, incorrect: str, correct: str):
    """Record that *incorrect* was wrong and *correct* was intended.

    Bias layout:  { incorrect_token: { correct_token: count } }
    This lets the translator look up "when we see token X, what should it be?"
    """
    bias = profile.setdefault("bias", {})
    if incorrect not in bias:
        bias[incorrect] = {}
    bias[incorrect][correct] = bias[incorrect].get(correct, 0) + 1
