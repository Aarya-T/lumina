from typing import Any, Dict

from groq import Groq

from agents.cultural_agent import grade_cultural_output, run_cultural_adaptor
from agents.continuity_agent import (
    grade_continuity_output,
    run_continuity_director,
)
from agents.translation_agent import (
    detect_language,
    grade_translation_output,
    run_translation_agent,
)
from agents.typesetting_agent import (
    grade_typesetting_output,
    run_typesetting_editor,
)
from utils.groq_client import load_environment

# TODO: Import CrewAI orchestration primitives when implementing multi-agent Crew.
# from crewai import Crew


def test_groq_connection() -> None:
    """
    Perform a basic test call to the Groq API to confirm connectivity.

    This function sends a simple message to the llama3-70b-8192 model and
    prints the response. Errors (e.g., invalid API key) are handled with
    clear messages instead of crashing the application.
    """
    try:
        client = load_environment()
    except RuntimeError as exc:
        # Clear message if the API key is missing.
        print(f"[Groq Test] Configuration error: {exc}")
        return
    except Exception as exc:  # Defensive: unexpected errors during initialization.
        print(f"[Groq Test] Unexpected error while initializing Groq client: {exc}")
        return

    try:
        completion = client.chat.completions.create(
            # Updated to a currently supported Groq Llama 3 70B model.
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": "Hello, respond in one sentence.",
                }
            ],
        )
        message = completion.choices[0].message.content
        print("[Groq Test] Connection successful. Model response:")
        print(message)
    except Exception as exc:
        # Catch errors such as invalid API keys or network failures.
        print("[Groq Test] Error while communicating with Groq API:")
        print(str(exc))


def build_crew():
    """
    Placeholder for future CrewAI-based orchestration.
    """
    raise NotImplementedError("Crew construction is not implemented yet.")


def _run_with_retries(
    step_name: str,
    run_fn,
    grade_fn,
    *,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Run an agent step with grading and up to `max_retries` attempts.

    The caller is responsible for providing small lambdas for `run_fn` and
    `grade_fn` that capture the appropriate arguments.
    """
    attempt = 0
    last_output: Any = None
    last_scores: Dict[str, Any] = {}

    while attempt < max_retries:
        attempt += 1
        print(f"[Pipeline] Running {step_name} (attempt {attempt}/{max_retries})...")

        candidate = run_fn()
        scores = grade_fn(candidate)

        last_output = candidate
        last_scores = scores

        if scores.get("pass"):
            print(f"[Pipeline] {step_name} PASS.")
            break

        if attempt < max_retries:
            print(f"[Pipeline] {step_name} FAIL — retry {attempt}/{max_retries}")
        else:
            print(
                f"[Pipeline] {step_name} FAIL after {max_retries} attempts — "
                "continuing with last output."
            )

    return {"output": last_output, "scores": last_scores}


def run_pipeline(
    raw_text: str,
    character_name: str,
    bubble_type: str,
    client: Groq,
) -> Dict[str, Any]:
    """
    Run the full Lumina Pipeline on a single line of script.

    Pipeline stages:
    - Step 0: Translation (Japanese → rough English)
    - Agent 1: Cultural Adaptation
    - Agent 2: Continuity Director
    - Agent 3: Typesetting Editor
    """
    print("[Pipeline] Starting full pipeline run.")

    # STEP 0 — Translation (New Agent)
    detected_language = detect_language(raw_text)
    print(f"[Pipeline] Detected language: {detected_language}")

    if detected_language in ("korean", "japanese"):
        translation_result = _run_with_retries(
            "Step 0 (Translation Agent)",
            run_fn=lambda: run_translation_agent(raw_text, client),
            grade_fn=lambda translated: grade_translation_output(
                original=raw_text,
                translated=translated,
                client=client,
            ),
        )
        translated_output = translation_result["output"]
        translation_scores = translation_result["scores"]
    else:
        print(
            "[Pipeline] Input appears to be English — skipping translation."
        )
        translated_output = raw_text
        translation_scores = {
            "contextual_accuracy": None,
            "tone_preservation": None,
            "naturalness": None,
            "pass": True,
        }

    # STEP A — Cultural Adaptation (Agent 1)
    cultural_result = _run_with_retries(
        "Agent 1 (Cultural Adaptor)",
        run_fn=lambda: run_cultural_adaptor(raw_text, client),
        grade_fn=lambda adapted: grade_cultural_output(raw_text, adapted, client),
    )
    cultural_output = cultural_result["output"]
    cultural_scores = cultural_result["scores"]

    # STEP B — Continuity Check (Agent 2)
    continuity_result = _run_with_retries(
        "Agent 2 (Continuity Director)",
        run_fn=lambda: run_continuity_director(
            cultural_output,
            character_name,
            client,
        ),
        grade_fn=lambda continuity_text: grade_continuity_output(
            adapted_text=cultural_output,
            continuity_text=continuity_text,
            character_name=character_name,
            client=client,
        ),
    )
    continuity_output = continuity_result["output"]
    continuity_scores = continuity_result["scores"]

    # STEP C — Typesetting (Agent 3)
    typesetting_result = _run_with_retries(
        "Agent 3 (Typesetting Editor)",
        run_fn=lambda: run_typesetting_editor(
            continuity_output,
            bubble_type,
            client,
        ),
        grade_fn=lambda final_text: grade_typesetting_output(
            original=continuity_output,
            final=final_text,
            bubble_type=bubble_type,
            client=client,
        ),
    )
    final_output = typesetting_result["output"]
    typesetting_scores = typesetting_result["scores"]

    print("[Pipeline] Pipeline run complete.")

    return {
        "original": raw_text,
        "detected_language": detected_language,
        "translated_output": translated_output,
        "cultural_output": cultural_output,
        "continuity_output": continuity_output,
        "final_output": final_output,
        "translation_scores": translation_scores,
        "cultural_scores": cultural_scores,
        "continuity_scores": continuity_scores,
        "typesetting_scores": typesetting_scores,
        "status": "complete",
    }


def test_full_pipeline() -> None:
    """
    Manual end-to-end test of the Lumina Pipeline with all three agents.
    """
    try:
        client = load_environment()
    except Exception as exc:
        print(f"[Pipeline Test] Failed to initialize Groq client: {exc}")
        return

    raw_text = "たとえ天が崩れ落ちようとも、私は一歩も退かない。"
    character_name = "Kira"
    bubble_type = "medium"

    print("[Pipeline Test] Raw text:")
    print(raw_text)
    print("[Pipeline Test] Character:", character_name)
    print("[Pipeline Test] Bubble type:", bubble_type)

    result = run_pipeline(
        raw_text=raw_text,
        character_name=character_name,
        bubble_type=bubble_type,
        client=client,
    )

    print("\n[Pipeline Test] Cultural output:")
    print(result["cultural_output"])

    print("\n[Pipeline Test] Continuity output:")
    print(result["continuity_output"])

    print("\n[Pipeline Test] Final typeset output:")
    print(result["final_output"])

    print("\n[Pipeline Test] Cultural scores:")
    print(result["cultural_scores"])

    print("\n[Pipeline Test] Continuity scores:")
    print(result["continuity_scores"])

    print("\n[Pipeline Test] Typesetting scores:")
    print(result["typesetting_scores"])


def main() -> None:
    """
    CLI-style entrypoint for running the pipeline.

    For now, this runs a simple end-to-end test of the full pipeline.
    """
    test_full_pipeline()


if __name__ == "__main__":
    main()

