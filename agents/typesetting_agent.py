"""
Typesetting Editor Agent.

Takes continuity-approved dialogue and ensures it physically fits inside
the speech bubble by trimming or rewording, without changing meaning or
character voice.
"""

from dataclasses import dataclass
from typing import Any, Dict

import json
import os
import sys

from groq import Groq

# Make the project root importable so we can access local packages and data files.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.groq_client import load_environment  # type: ignore


@dataclass
class TypesettingEditorConfig:
    """
    Configuration for the Typesetting Editor Agent.

    This can be extended later with:
    - Default bubble type.
    - Language/style preferences.
    """

    # Placeholder for future configuration fields.
    pass


class TypesettingEditorAgent:
    """
    Thin wrapper class kept for future CrewAI integration.
    """

    def __init__(self, config: TypesettingEditorConfig) -> None:
        self.config = config


# NOTE: Original spec requested `llama3-70b-8192`, but that model has been
# decommissioned. We use the current Groq 70B Llama 3 model instead.
MODEL_NAME = "llama-3.3-70b-versatile"


def load_bubble_config(config_path: str) -> Dict[str, Any]:
    """
    Load and return the bubble configuration JSON as a dictionary.

    Args:
        config_path: Path to `bubble_config.json`.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_max_chars_for_bubble(bubble_type: str, config: Dict[str, Any]) -> int:
    """
    Resolve the maximum character limit for a given bubble type.
    """
    default_max = int(config.get("default_max_chars", 80))
    bubbles = config.get("bubbles", {}) or {}
    bubble_max = bubbles.get(bubble_type)
    return int(bubble_max) if bubble_max is not None else default_max


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """
    Truncate text to fit within max_chars, cutting at the last word boundary.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip()


def run_typesetting_editor(
    text: str,
    bubble_type: str,
    client: Groq,
    *,
    bubble_char_limit: int | None = None,
) -> str:
    """
    Ensure the dialogue fits the specified bubble type's character limit.

    Args:
        text: Continuity-approved dialogue.
        bubble_type: Bubble category (e.g., 'small', 'medium', 'large', 'thought').
        client: Initialized Groq client.

    Returns:
        Final dialogue text that fits within the bubble constraints.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "data", "bubble_config.json")
    config = load_bubble_config(config_path)
    max_chars = (
        int(bubble_char_limit)
        if bubble_char_limit is not None
        else _get_max_chars_for_bubble(bubble_type, config)
    )

    if len(text) <= max_chars:
        return text

    system_prompt = (
        "You are a typesetting editor for localized manga/webtoon dialogue.\n"
        "Your job is to make sure the text physically fits inside a speech bubble\n"
        "with a strict character limit, without changing meaning or character voice.\n\n"
        f"Bubble type: {bubble_type}\n"
        f"Maximum characters allowed (including spaces and punctuation): {max_chars}\n\n"
        "Rules:\n"
        "- Preserve the character's voice, tone, and intent.\n"
        "- Preserve the meaning as much as possible while shortening.\n"
        "- Do NOT add new information or remove key emotional beats.\n"
        "- The final text MUST be at most the specified character limit.\n"
        "- Output ONLY the final dialogue text, with no explanations or notes.\n"
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )

    rewritten = (completion.choices[0].message.content or "").strip()

    # Enforce the hard limit programmatically, regardless of model behavior.
    if len(rewritten) > max_chars:
        rewritten = _truncate_at_word_boundary(rewritten, max_chars)

    return rewritten


def grade_typesetting_output(
    original: str,
    final: str,
    bubble_type: str,
    client: Groq,
    *,
    bubble_char_limit: int | None = None,
) -> Dict[str, Any]:
    """
    Grade the typesetting output quality.

    Metrics:
    - fits_constraint (1-10) — based on actual character count.
    - meaning_preserved (1-10).
    - voice_maintained (1-10).
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "data", "bubble_config.json")
    config = load_bubble_config(config_path)
    max_chars = (
        int(bubble_char_limit)
        if bubble_char_limit is not None
        else _get_max_chars_for_bubble(bubble_type, config)
    )

    final_len = len(final)
    fits_constraint = 10 if final_len <= max_chars else 1

    system_prompt = (
        "You are an expert localization editor focusing on typesetting.\n"
        "You will receive the ORIGINAL continuity-approved dialogue and the FINAL\n"
        "typeset-safe dialogue.\n\n"
        "Your job is to grade the FINAL text on two axes, from 1 to 10:\n"
        "- meaning_preserved: How well the meaning of the ORIGINAL is preserved.\n"
        "- voice_maintained: How well the character's voice, tone, and style are preserved.\n\n"
        "Scoring rules:\n"
        "- 1 is terrible, 10 is excellent.\n\n"
        "Output rules (IMPORTANT):\n"
        "- Output ONLY a valid JSON object with the exact keys:\n"
        "  meaning_preserved, voice_maintained\n"
        "- Example:\n"
        '  {\"meaning_preserved\": 9, \"voice_maintained\": 8}\n'
        "- Do NOT include any explanations, comments, or extra text.\n"
    )

    user_content = (
        "ORIGINAL (continuity-approved):\n"
        f"{original}\n\n"
        "FINAL (typeset-safe):\n"
        f"{final}\n"
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
            "fits_constraint": fits_constraint,
            "meaning_preserved": 0,
            "voice_maintained": 0,
            "pass": False,
            "raw_response": raw,
        }

    meaning_preserved = int(parsed.get("meaning_preserved", 0))
    voice_maintained = int(parsed.get("voice_maintained", 0))

    passed = (
        fits_constraint >= 7
        and meaning_preserved >= 7
        and voice_maintained >= 7
    )

    return {
        "fits_constraint": fits_constraint,
        "meaning_preserved": meaning_preserved,
        "voice_maintained": voice_maintained,
        "pass": passed,
    }


def test_typesetting_agent() -> None:
    """
    Simple manual test for the Typesetting Editor Agent.

    Steps:
    - Use bubble_type 'small' (max 40 chars).
    - Input text: \"I'm not backing down, no way!\".
    - Run through `run_typesetting_editor`.
    - Print original and final lengths and the result text.
    - Grade via `grade_typesetting_output` and print scores and PASS/FAIL.
    """
    try:
        client = load_environment()
    except Exception as exc:
        print(f"[TypesettingAgent Test] Failed to initialize Groq client: {exc}")
        return

    bubble_type = "small"
    original = "I'm not backing down, no way!"

    print("[TypesettingAgent Test] Bubble type:", bubble_type)
    print("[TypesettingAgent Test] Original text:", original)
    print("[TypesettingAgent Test] Original length:", len(original))

    try:
        final = run_typesetting_editor(original, bubble_type, client)
    except Exception as exc:
        print(f"[TypesettingAgent Test] Error during typesetting: {exc}")
        return

    print("\n[TypesettingAgent Test] Final text:", final)
    print("[TypesettingAgent Test] Final length:", len(final))

    try:
        scores = grade_typesetting_output(original, final, bubble_type, client)
    except Exception as exc:
        print(f"[TypesettingAgent Test] Error during grading: {exc}")
        return

    print("\n[TypesettingAgent Test] Scores:")
    print(scores)

    status = "PASS" if scores.get("pass") else "FAIL"
    print(f"\n[TypesettingAgent Test] Result: {status}")

