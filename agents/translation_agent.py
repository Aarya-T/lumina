"""
Translation Agent (Step 0).

Takes raw Japanese manga/webtoon script text and produces a contextual
English translation that preserves emotion, tone, and dramatic weight.
This output feeds into Agent 1 (Cultural Adaptor).
"""

from dataclasses import dataclass
from typing import Any, Dict

import json
import os
import sys

from groq import Groq

# Ensure the project root is on sys.path so we can import local utilities.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.groq_client import load_environment  # type: ignore


@dataclass
class TranslationAgentConfig:
    """
    Placeholder configuration for the Translation Agent.
    """

    # Extendable later (e.g., model name, temperature).
    pass


# NOTE: Original spec requested `llama3-70b-8192`, but that model has been
# decommissioned by Groq. We use the current Groq 70B Llama 3 model instead.
MODEL_NAME = "llama-3.3-70b-versatile"


def detect_language(text: str) -> str:
    """
    Very simple Unicode-based language detection.

    Returns:
        "japanese", "korean", or "unknown"
    """
    has_japanese = False
    has_korean = False

    for ch in text:
        code = ord(ch)

        # Japanese ranges: Hiragana, Katakana, Kanji/CJK
        if (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF) or (
            0x4E00 <= code <= 0x9FFF
        ):
            has_japanese = True

        # Korean range: Hangul syllables
        if 0xAC00 <= code <= 0xD7A3:
            has_korean = True

    # Prioritize Japanese when mixed.
    if has_japanese:
        return "japanese"
    if has_korean:
        return "korean"
    return "unknown"


def run_translation_agent(
    raw_text: str,
    client: Groq,
    character_name: str = "",
) -> str:
    """
    Run the Translation Agent on a raw Japanese line.

    Args:
        raw_text: Original script text (Japanese).
        client: Initialized Groq client.

    Returns:
        Contextual English translation as plain text.
    """
    language = detect_language(raw_text)

    system_prompt = (
        "You are a professional manga translator specializing in Japanese to English translation.\n"
        "Translate the given Japanese manga dialogue to English.\n\n"
        "This is a CONTEXTUAL translation. Follow these rules strictly:\n"
        "- Preserve emotion, tone, and dramatic weight; do NOT do word-for-word translation.\n"
        "- Preserve Japanese speech levels:\n"
        "  * Keigo (formal/polite) → formal English.\n"
        "  * Casual/plain form → casual English.\n"
        "  * Rough/masculine speech (e.g., ore, ore-sama) → aggressive English.\n"
        "  * Feminine speech (e.g., watashi, kashira) → soft, feminine English.\n"
        "- Preserve the FEELING of Japanese honorifics (like -san, -kun, -sama) as respect levels,\n"
        "  but do NOT include the honorific words themselves in the output.\n"
        "- For ambiguous phrases, always consider emotional context:\n"
        "  * 終わった said by someone in distress =\n"
        "    'I'm done for' / 'I'm screwed' / 'It's over for me'\n"
        "    NOT 'it's finished' or 'it's wrapped up'\n"
        "  * 決まりね？ = 'So it's settled then?' / 'That's decided?'\n"
        "    — always a question when it ends with ね？\n"
        "  * ぼっち = 'loner' — always preserve this word or concept,\n"
        "    never replace with generic anxiety references\n"
        "  * なら私が出してあげる after ふふっ =\n"
        "    'Hehe, I'll treat you' / 'I'll cover it, teehee'\n"
        "    — preserve the playful feminine offer\n"
        "  * はずだった = narrative irony callback meaning\n"
        "    'or so I thought' / 'that was the plan, anyway' /\n"
        "    '...supposedly' — always preserve the ironic disappointed tone,\n"
        "    never translate as neutral past tense\n"
        "- Never confuse the speaker. The dialogue line belongs to the character named in the panel.\n"
        "  * なら私が〜してあげる = 'Then I'll [do X] for you' —\n"
        "    私 (I/me) is the SPEAKER offering something TO someone else.\n"
        "    Never translate as the speaker receiving something.\n"
        "  * あげる after a verb = the speaker is giving/doing for others,\n"
        "    not receiving.\n"
        "- Do NOT include any romaji, Japanese script, explanations, translation notes, or comments.\n"
        "- Output ONLY the translated English dialogue text, with no extra lines.\n"
    )

    # Provide detected language as context in the user message.
    user_content = (
        f"Detected language (heuristic): {language}\n"
        + (
            f"SPEAKER: {character_name} — this line is spoken BY {character_name}, not to them.\n"
            if character_name
            else ""
        )
        + "Source dialogue:\n"
        + f"{raw_text}\n"
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    translated = completion.choices[0].message.content or ""
    return translated.strip()


def grade_translation_output(
    original: str,
    translated: str,
    client: Groq,
) -> Dict[str, Any]:
    """
    Grade the translation quality.

    Metrics:
    - contextual_accuracy (1-10) — meaning preserved.
    - tone_preservation (1-10) — emotion/formality preserved.
    - naturalness (1-10) — reads like natural English dialogue.
    """
    system_prompt = (
        "You are an expert bilingual Japanese ↔ English manga translator.\n"
        "You will receive the ORIGINAL line (Japanese) and the TRANSLATED\n"
        "English line. Grade the English line on three axes from 1 to 10:\n"
        "- contextual_accuracy: How well it captures the intended meaning and nuance.\n"
        "- tone_preservation: How well it preserves emotion, formality level,\n"
        "  and dramatic weight.\n"
        "- naturalness: How natural and fluent it reads as English dialogue.\n\n"
        "Scoring rules:\n"
        "- 1 is terrible, 10 is excellent.\n"
        "- The final `pass` value must be true ONLY if all three scores are >= 7.\n\n"
        "Output rules (IMPORTANT):\n"
        "- Output ONLY a valid JSON object with the exact keys:\n"
        "  contextual_accuracy, tone_preservation, naturalness, pass\n"
        "- Example:\n"
        '  {\"contextual_accuracy\": 8, \"tone_preservation\": 9, '
        '\"naturalness\": 8, \"pass\": true}\n'
        "- Do NOT include any explanations, comments, or extra text.\n"
    )

    user_content = (
        "ORIGINAL (Japanese):\n"
        f"{original}\n\n"
        "TRANSLATED (English):\n"
        f"{translated}\n"
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    raw = (completion.choices[0].message.content or "").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "contextual_accuracy": 0,
            "tone_preservation": 0,
            "naturalness": 0,
            "pass": False,
            "raw_response": raw,
        }

    contextual_accuracy = int(parsed.get("contextual_accuracy", 0))
    tone_preservation = int(parsed.get("tone_preservation", 0))
    naturalness = int(parsed.get("naturalness", 0))

    passed = (
        contextual_accuracy >= 7
        and tone_preservation >= 7
        and naturalness >= 7
    )

    return {
        "contextual_accuracy": contextual_accuracy,
        "tone_preservation": tone_preservation,
        "naturalness": naturalness,
        "pass": passed,
    }


def test_translation_agent() -> None:
    """
    Simple manual test for the Translation Agent.

    Uses the sample Japanese text:
    \"たとえ天が崩れ落ちようとも、私は一歩も退かない。\"
    """
    try:
        client = load_environment()
    except Exception as exc:
        print(f"[TranslationAgent Test] Failed to initialize Groq client: {exc}")
        return

    raw_text = "たとえ天が崩れ落ちようとも、私は一歩も退かない。"
    language = detect_language(raw_text)

    print("[TranslationAgent Test] Detected language:", language)

    try:
        translated = run_translation_agent(raw_text, client)
    except Exception as exc:
        print(f"[TranslationAgent Test] Error during translation: {exc}")
        return

    print("\n[TranslationAgent Test] Translated text:")
    print(translated)

    try:
        scores = grade_translation_output(raw_text, translated, client)
    except Exception as exc:
        print(f"[TranslationAgent Test] Error during grading: {exc}")
        return

    print("\n[TranslationAgent Test] Scores:")
    print(scores)

    status = "PASS" if scores.get("pass") else "FAIL"
    print(f"\n[TranslationAgent Test] Result: {status}")

