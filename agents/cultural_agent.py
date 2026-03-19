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
from utils.groq_client import load_environment

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

# NOTE: The original spec referenced `llama3-70b-8192`, which Groq has since
# been decommissioned by Groq. This constant uses the current 70B Llama 3 model
# recommended by Groq. If Groq updates models again, you can change this in one
# place.
MODEL_NAME = "llama-3.3-70b-versatile"


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
        "You are a professional manga and webtoon localization writer.\n"
        "Your task is to take the given English text, which may contain\n"
        "literal translations of Japanese or Korean idioms, honorifics,\n"
        "and culture-specific references, and rewrite it into natural,\n"
        "culturally appropriate English dialogue.\n\n"
        "Strict rules:\n"
        "1. Identify any Japanese/Korean idioms, honorifics, references, or stiff phrasing.\n"
        "2. Rewrite them into natural English equivalents that an English-native\n"
        "   manga/webtoon reader would immediately understand.\n"
        "3. Preserve the original meaning, emotion, intent, and tone (e.g., comedic,\n"
        "   serious, dramatic, sarcastic, shy).\n"
        "4. Keep the overall length roughly similar to the input (do not expand into\n"
        "   long explanations or shorten it excessively).\n"
        "5. Output ONLY the final rewritten dialogue text.\n"
        "6. Do NOT include any explanations, analysis, notes, or formatting beyond the\n"
        "   dialogue itself.\n"
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
        "You are an expert localization editor.\n"
        "You will be given an ORIGINAL line of translated English text and\n"
        "an ADAPTED line that attempts to make it sound natural to an\n"
        "English-native manga/webtoon reader.\n\n"
        "Your job is to grade the adaptation on three axes, each from 1 to 10:\n"
        "- cultural_accuracy: How well the adapted line preserves cultural meaning\n"
        "  and avoids mistranslating or erasing important nuance.\n"
        "- tone_preservation: How well the emotional tone, attitude, and character\n"
        "  voice of the original are preserved.\n"
        "- naturalness: How natural and fluent the line sounds in English dialogue\n"
        "  (for manga/webtoon readers).\n\n"
        "Scoring rules:\n"
        "- 1 is terrible, 10 is excellent.\n"
        "- The final `pass` value must be true ONLY if all three scores are >= 7.\n\n"
        "Output rules (IMPORTANT):\n"
        "- Output ONLY a valid JSON object with the exact keys:\n"
        "  cultural_accuracy, tone_preservation, naturalness, pass\n"
        "- Example:\n"
        '  {\"cultural_accuracy\": 8, \"tone_preservation\": 9, \"naturalness\": 8, \"pass\": true}\n'
        "- Do NOT include any explanations, comments, or extra text.\n"
    )

    user_content = (
        "ORIGINAL:\n"
        f"{original}\n\n"
        "ADAPTED:\n"
        f"{adapted}\n"
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
        # Fallback in case the model does not strictly follow JSON instructions.
        # In that case, mark as an automatic fail with minimal defaults.
        return {
            "cultural_accuracy": 0,
            "tone_preservation": 0,
            "naturalness": 0,
            "pass": False,
            "raw_response": raw,
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

