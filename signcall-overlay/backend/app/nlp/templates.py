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
    "HOW": {
        "concise": "How?",
        "detailed": "How? The user is asking about a method or way.",
    },
    "YOU": {
        "concise": "You",
        "detailed": "You — the user is referring to you.",
    },
    "CAN": {
        "concise": "Can",
        "detailed": "Can — the user is expressing ability or asking permission.",
    },
    "SEE_YOU_LATER": {
        "concise": "See you later",
        "detailed": "See you later! The user is saying goodbye.",
    },
    "FATHER": {
        "concise": "Father",
        "detailed": "Father — the user is referring to their father.",
    },
    "MOTHER": {
        "concise": "Mother",
        "detailed": "Mother — the user is referring to their mother.",
    },
}
