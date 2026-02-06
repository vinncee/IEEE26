#!/usr/bin/env python3
"""Manual test for Member 4 (NLP/Translation layer)"""

import sys
sys.path.insert(0, '/Users/kahei/Documents/GitHub/IEEE26/signcall-overlay/backend')

from app.nlp.templates import TEMPLATES
from app.nlp.translator import translate
from app.nlp.llm_client import gloss_to_english

print("=" * 60)
print("Member 4 Translation Layer Tests")
print("=" * 60)

# Test 1: Template variants exist
print("\n1️⃣ Testing templates (concise + detailed)...")
for phrase in ["HELLO", "THANKS", "SLOW", "REPEAT"]:
    concise = TEMPLATES[phrase].get("concise")
    detailed = TEMPLATES[phrase].get("detailed")
    print(f"  {phrase}:")
    print(f"    Concise:  '{concise}'")
    print(f"    Detailed: '{detailed}'")
    assert concise and detailed, f"Missing templates for {phrase}"
print("  ✓ All templates present")

# Test 2: Low confidence (uncertain mode)
print("\n2️⃣ Testing low confidence (should show uncertainty)...")
pred_low = {"token": "HELLO", "confidence": 0.15, "top2": ["HELLO", "THANKS"]}
profile = {"bias": {}}
result = translate(pred_low, profile, style="concise")
print(f"  Prediction: {pred_low}")
print(f"  Output: {result['caption']}")
print(f"  Mode: {result['mode']}")
assert result["mode"] == "uncertain", "Low confidence should be uncertain"
print("  ✓ Low confidence handling works")

# Test 3: Medium confidence (template mode)
print("\n3️⃣ Testing medium confidence (should use template)...")
pred_med = {"token": "HELLO", "confidence": 0.70, "top2": ["HELLO", "THANKS"]}
result_concise = translate(pred_med, profile, style="concise")
result_detailed = translate(pred_med, profile, style="detailed")
print(f"  Prediction: {pred_med}")
print(f"  Concise: '{result_concise['caption']}'")
print(f"  Detailed: '{result_detailed['caption']}'")
print(f"  Mode: {result_concise['mode']}")
assert result_concise["mode"] == "template", "Medium confidence should use template"
assert result_concise["caption"] != result_detailed["caption"], "Styles should differ"
print("  ✓ Medium confidence & style toggling works")

# Test 4: Profile bias applied
print("\n4️⃣ Testing profile bias (correction learning)...")
pred = {"token": "SLOW", "confidence": 0.75, "top2": ["SLOW", "REPEAT"]}
profile_with_bias = {
    "bias": {"REPEAT": 5, "SLOW": 1}  # User corrected SLOW→REPEAT 5 times
}
result = translate(pred, profile_with_bias, style="concise")
print(f"  Input: SLOW (user corrected to REPEAT 5 times)")
print(f"  Output: '{result['caption']}'")
print(f"  Mode: {result['mode']}")
# Bias should redirect SLOW → REPEAT
assert "Repeat" in result['caption'] or result['caption'] == "Repeat?", "Bias not applied"
print("  ✓ Profile bias is applied")

# Test 5: LLM graceful degradation
print("\n5️⃣ Testing LLM fallback (no API key)...")
gloss = "HELLO"
caption = gloss_to_english(gloss, style="concise")
if caption is None:
    print(f"  LLM unavailable (expected with test key)")
    print(f"  ✓ Graceful fallback enabled")
else:
    print(f"  LLM response: '{caption}'")
    print(f"  ✓ LLM integration working")

# Test 6: High confidence (LLM mode if available)
print("\n6️⃣ Testing high confidence (LLM polish if available)...")
pred_high = {"token": "HELLO", "confidence": 0.85, "top2": ["HELLO", "THANKS"]}
result = translate(pred_high, profile, style="detailed")
print(f"  Prediction: {pred_high}")
print(f"  Output: '{result['caption']}'")
print(f"  Mode: {result['mode']}")
assert result["mode"] in ["llm", "template"], "High confidence should try LLM or template"
print(f"  ✓ High confidence handling works (mode: {result['mode']})")

print("\n" + "=" * 60)
print("✓ All Member 4 tests passed!")
print("=" * 60)
print("\nMember 4 is ready to start implementation!")
