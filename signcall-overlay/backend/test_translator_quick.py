#!/usr/bin/env python3
"""Quick manual test for the NLP translator module (Member 4)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.nlp.translator import translate
from app.nlp.profile import get_profile, apply_correction

profile = get_profile("test", "userA")

print("=" * 60)
print("TRANSLATOR MANUAL TEST")
print("=" * 60)

# --- High confidence: should try LLM (falls back to template if no API key) ---
pred = {"token": "HELLO", "confidence": 0.90, "top2": ["HELLO", "THANKS"], "ts": 1000}
print(f"\n1) HIGH conf (0.90), concise:  {translate(pred, profile, style='concise')}")
print(f"   HIGH conf (0.90), detailed: {translate(pred, profile, style='detailed')}")

# --- Medium confidence: should use template ---
pred = {"token": "REPEAT", "confidence": 0.65, "top2": ["REPEAT", "SLOW"], "ts": 2000}
print(f"\n2) MED conf (0.65), concise:   {translate(pred, profile, style='concise')}")
print(f"   MED conf (0.65), detailed:  {translate(pred, profile, style='detailed')}")

# --- Low confidence: should ask clarification ---
pred = {"token": "SLOW", "confidence": 0.30, "top2": ["SLOW", "REPEAT"], "ts": 3000}
print(f"\n3) LOW conf (0.30):            {translate(pred, profile, style='concise')}")

# --- Test all 4 phrases at medium confidence ---
print("\n4) All phrases at medium confidence (template mode):")
for token in ["HELLO", "THANKS", "REPEAT", "SLOW"]:
    pred = {"token": token, "confidence": 0.65, "top2": [token, "HELLO"], "ts": 4000}
    result = translate(pred, profile, style="concise")
    print(f"   {token:8s} → {result['caption']:40s} mode={result['mode']}")

# --- Test bias correction ---
print("\n5) Bias correction test:")
print(f"   Profile before: {profile}")
apply_correction(profile, "SLOW", "REPEAT")
apply_correction(profile, "SLOW", "REPEAT")
print(f"   Profile after 2x (SLOW→REPEAT): {profile}")

pred = {"token": "SLOW", "confidence": 0.65, "top2": ["SLOW", "REPEAT"], "ts": 5000}
result = translate(pred, profile, style="concise")
print(f"   SLOW now translates as: {result['caption']}  (bias should override to REPEAT)")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
