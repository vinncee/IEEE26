"""Template variants for sign language captions.

Each phrase has 'concise' and 'detailed' templates:
- concise: 1-2 words, quick recognition
- detailed: full context, natural conversation
"""

TEMPLATES = {
    "HELLO": {
        "concise": "Hello",
        "detailed": "Hello! The user just greeted me.",
    },
    "REPEAT": {
        "concise": "Repeat?",
        "detailed": "Could you please repeat that? I didn't quite catch it.",
    },
    "SLOW": {
        "concise": "Slower",
        "detailed": "Please go slower. I'm having trouble keeping up.",
    },
    "THANKS": {
        "concise": "Thanks",
        "detailed": "Thank you for that. I really appreciate it.",
    },
}
