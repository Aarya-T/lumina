"""
Cultural Adaptor Agent.

Responsible for:
- Interpreting Japanese/Korean idioms, jokes, and culture-specific references.
- Rewriting them into natural, culturally appropriate English equivalents.
- Preserving tone, intent, and character voice guidelines.
"""

from dataclasses import dataclass
from typing import Any, Dict

import json
import os
import sys

from groq import Groq

# Ensure the project root is on sys.path so we can import local packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.groq_client import FAST_MODEL, QUALITY_MODEL, load_environment

# TODO: Import CrewAI Agent/Task abstractions when integrating with orchestration.
# from crewai import Agent


@dataclass
class CulturalAdaptorConfig:
    """
    Configuration for the Cultural Adaptor Agent.

    This may include:
    - Model name / temperature settings.
    - Style guides or localization preferences.
    - Any prompt templates or system messages.
    """

    # TODO: Add concrete configuration fields (e.g., model_name: str).
    pass


class CulturalAdaptorAgent:
    """
    Skeleton for the Cultural Adaptor Agent.

    This class will encapsulate:
    - Prompt construction tailored to cultural adaptation.
    - Calls to the Groq API via the selected LLM.
    - Interfaces required by CrewAI for orchestration.
    """

    def __init__(self, config: CulturalAdaptorConfig) -> None:
        # TODO: Store config and initialize any external clients/tools.
        self.config = config

    def adapt_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt a single script segment culturally.

        Args:
            segment: A structured representation of one script unit
                     (e.g., panel or bubble data).

        Returns:
            A modified segment with culturally localized English text.
        """
        # TODO: Implement call to LLM with prompts that focus on cultural adaptation.
        # TODO: Ensure outputs remain consistent with character voice guidelines.
        raise NotImplementedError("Cultural adaptation logic is not implemented yet.")

MODEL_NAME = QUALITY_MODEL

def run_cultural_adaptor(raw_text: str, client: Groq) -> str:
    """
    Run the Cultural Adaptor on a single text block.

    Args:
        raw_text: Input English text that may contain JP/KR idioms or
                  culturally specific phrasing.
        client:  Initialized Groq client.

    Returns:
        Culturally adapted English text.
    """
    system_prompt = (
        "You are a cultural localization editor for manga and webtoon dialogue.\n"
        "You receive English text that may be a literal translation from Japanese\n"
        "or Korean.\n\n"
        "Follow these rules exactly:\n"
        "1) Identify Japanese/Korean idioms, cultural references, honorific-driven tone,\n"
        "   or unnatural literal phrasing.\n"
        "2) Rewrite into natural, culturally appropriate English equivalents.\n"
        "3) Preserve original meaning, emotion, and tone.\n"
        "4) Keep output length similar to input length.\n"
        "5) Output only the rewritten dialogue.\n"
        "6) Do not add explanations, notes, labels, or extra formatting.\n"
        "8) If the input already reads as natural correct English with clear\n"
        "   meaning and correct speaker attribution — preserve it. Only rewrite\n"
        "   when there is a genuine cultural equivalence improvement to make.\n"
        "   Never change who is speaking or who is receiving an action.\n"
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ],
    )

    adapted = completion.choices[0].message.content or ""
    return adapted.strip()


def grade_cultural_output(original: str, adapted: str, client: Groq) -> Dict[str, Any]:
    """
    Ask the LLM to grade the cultural adaptation quality.

    The grading dimensions are:
    - cultural_accuracy (1-10)
    - tone_preservation (1-10)
    - naturalness (1-10)

    The result includes:
    - the three scores
    - a `pass` boolean, which is True only if all scores are >= 7.
    """
    system_prompt = (
        "You are a strict localization quality grader.\n"
        "You will grade an adapted English dialogue line against the original line.\n\n"
        "Score each metric from 1 to 10:\n"
        "- cultural_accuracy: faithful handling of idioms/cultural references.\n"
        "- tone_preservation: preserves emotion, attitude, and voice.\n"
        "- naturalness: reads as fluent, natural English dialogue.\n\n"
        "Set pass=true only when ALL three scores are >= 7.\n\n"
        "Output ONLY valid JSON with exactly these keys:\n"
        "cultural_accuracy, tone_preservation, naturalness, pass\n"
        "No prose. No markdown. No extra keys.\n"
    )

    user_content = (
        "ORIGINAL:\n"
        f"{original}\n\n"
        "ADAPTED:\n"
        f"{adapted}\n"
    )

    completion = client.chat.completions.create(
        model=FAST_MODEL,
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
            "cultural_accuracy": 0,
            "tone_preservation": 0,
            "naturalness": 0,
            "pass": False,
        }

    cultural_accuracy = int(parsed.get("cultural_accuracy", 0))
    tone_preservation = int(parsed.get("tone_preservation", 0))
    naturalness = int(parsed.get("naturalness", 0))

    passed = (
        cultural_accuracy >= 7
        and tone_preservation >= 7
        and naturalness >= 7
    )

    return {
        "cultural_accuracy": cultural_accuracy,
        "tone_preservation": tone_preservation,
        "naturalness": naturalness,
        "pass": passed,
    }


def test_cultural_agent() -> None:
    """
    Simple manual test for the Cultural Adaptor agent.

    Steps:
    - Use the sample raw text:
      "Even if the heavens fall, I will not retreat a single step."
    - Run it through `run_cultural_adaptor`.
    - Print the adapted result.
    - Run grading via `grade_cultural_output`.
    - Print scores and PASS/FAIL.
    """
    try:
        client = load_environment()
    except Exception as exc:
        print(f"[CulturalAgent Test] Failed to initialize Groq client: {exc}")
        return

    raw_text = "Even if the heavens fall, I will not retreat a single step."
    print("[CulturalAgent Test] Original:")
    print(raw_text)

    try:
        adapted = run_cultural_adaptor(raw_text, client)
    except Exception as exc:
        print(f"[CulturalAgent Test] Error during adaptation: {exc}")
        return

    print("\n[CulturalAgent Test] Adapted:")
    print(adapted)

    try:
        scores = grade_cultural_output(raw_text, adapted, client)
    except Exception as exc:
        print(f"[CulturalAgent Test] Error during grading: {exc}")
        return

    print("\n[CulturalAgent Test] Scores:")
    print(scores)

    status = "PASS" if scores.get("pass") else "FAIL"
    print(f"\n[CulturalAgent Test] Result: {status}")

