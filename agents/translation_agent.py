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
from utils.groq_client import FAST_MODEL, QUALITY_MODEL, load_environment  # type: ignore


@dataclass
class TranslationAgentConfig:
    """
    Placeholder configuration for the Translation Agent.
    """

    # Extendable later (e.g., model name, temperature).
    pass


# NOTE: Original spec requested `llama3-70b-8192`, but that model has been
# decommissioned by Groq. We use the current Groq 70B Llama 3 model instead.
MODEL_NAME = QUALITY_MODEL


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


def translate_chapter_batch(panels: list, client: Groq) -> dict:
    """
    Translate all panel Japanese lines in one LLM call.

    Args:
        panels: List of panel dicts from a chapter JSON. Each panel should include:
                 - panel_id (or id/index)
                 - character (speaker name)
                 - text (Japanese dialogue)
        client: Initialized Groq client.

    Returns:
        Dict mapping panel_id (as string) -> translated English text.
    """
    items: list[tuple[str, str, str, str]] = []
    for idx, panel in enumerate(panels):
        if not isinstance(panel, dict):
            continue

        panel_id = panel.get("panel_id") or panel.get("id") or panel.get("index")
        panel_id_str = str(panel_id if panel_id is not None else idx)

        character = str(panel.get("character") or "").strip()
        if not character:
            character = "UNKNOWN"

        text = str(panel.get("text") or "").strip()
        if not text:
            continue

        # Try to read page info from the panel dict.
        page_val = None
        try:
            page_val = panel.get("page")
        except Exception:
            page_val = None
        if page_val is None:
            try:
                page_val = panel.get("page_number")
            except Exception:
                page_val = None
        if page_val is None:
            try:
                page_val = panel.get("page_id")
            except Exception:
                page_val = None

        if page_val is None:
            page_key = "1"
        else:
            page_key = str(page_val)

        items.append((panel_id_str, character, text, page_key))

    if not items:
        return {}

    system_prompt = (
        "You are a professional manga translator specializing in Japanese to English.\n"
        "You will be given multiple Japanese dialogue lines for the same chapter.\n\n"
        "Rules (IMPORTANT):\n"
        "1) Read ALL lines first to understand full scene context and conversation flow.\n"
        "2) Then translate each line individually.\n"
        "3) Preserve speaker identity. Each line is labeled with panel_id and the character.\n"
        "4) Output ONLY a single strict JSON object.\n"
        "5) JSON keys MUST exactly match the panel_id strings from the input.\n"
        "6) JSON values MUST be the translated English dialogue for that panel.\n"
    )

    panel_page_groups: dict[str, list[tuple[str, str, str]]] = {}
    page_order: list[str] = []

    for panel_id_str, character, text, page_key in items:
        if page_key not in panel_page_groups:
            panel_page_groups[page_key] = []
            page_order.append(page_key)
        panel_page_groups[page_key].append((panel_id_str, character, text))

    preamble = (
        "You are translating a manga chapter.\n"
        "Read ALL lines below first to understand the full scene and character relationships\n"
        "before translating any line.\n"
        "Each line shows: [panel_id] CHARACTER: japanese_text\n"
        "Translate preserving emotional state and who is speaking TO whom.\n"
    )

    user_content_lines: list[str] = [preamble.rstrip()]
    for page_key in page_order:
        user_content_lines.append(f"--- Page {page_key} ---")
        for panel_id_str, character, text in panel_page_groups[page_key]:
            user_content_lines.append(f"[{panel_id_str}] {character}: {text}")

    user_content = "\n".join(user_content_lines).strip()

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    raw = (completion.choices[0].message.content or "").strip()
    try:
        import json

        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    result: dict[str, str] = {}
    for k, v in parsed.items():
        if v is None:
            continue
        result[str(k)] = str(v).strip()
    return result


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
        "- Before translating, identify the speaker's emotional state: panicking / happy / sad /\n"
        "  angry / teasing / casual farewell / ironic.\n"
        "- Let emotional state drive word choice:\n"
        "  * Panic phrases → desperate English equivalents\n"
        "  * Ironic narrator lines → preserve the irony with phrases like\n"
        "    'or so I thought' / 'at least that was the plan'\n"
        "  * Casual farewells → 'later' / 'see ya' / 'catch you later'\n"
        "  * Offers and giving phrases → preserve who is giving and who is receiving,\n"
        "    never reverse them\n"
        "  * Definitive statements → translate as confident, not hesitant\n"
        "    (never add 'I guess' / 'I suppose' to a character who spoke with conviction)\n"
        "- Never soften strong emotions or add uncertainty to confident statements\n"
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

