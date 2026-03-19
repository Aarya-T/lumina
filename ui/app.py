"""
Streamlit frontend for the Lumina Pipeline.

Provides an interactive UI to run the full localization pipeline:
Step 0 Translation → Agent 1 Cultural Adaptation → Agent 2 Continuity →
Agent 3 Typesetting.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# Ensure project root is importable when running `streamlit run ui/app.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import run_pipeline  # type: ignore
from memory.vector_store import load_characters_from_json  # type: ignore


@st.cache_resource
def get_client():
    from utils.groq_client import load_environment

    return load_environment()


def _load_bubble_config() -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "data" / "bubble_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _bubble_limit(bubble_type: str, config: Dict[str, Any]) -> int:
    default_max = int(config.get("default_max_chars", 80))
    bubbles = config.get("bubbles", {}) or {}
    val = bubbles.get(bubble_type)
    return int(val) if val is not None else default_max


def _metric_int(label: str, value: Any) -> None:
    """
    Render a metric that may be None/unknown.
    """
    if value is None:
        st.metric(label, "—")
    else:
        st.metric(label, int(value))


def _init_page() -> None:
    st.set_page_config(layout="wide")
    st.title("🎌 Lumina Pipeline")
    st.caption("Autonomous Japanese Manga Localization Engine")


def _sidebar() -> Dict[str, str]:
    with st.sidebar:
        st.header("Settings")

        character_name = st.selectbox(
            "Select Character",
            options=["Kira", "Unknown"],
            index=0,
        )

        bubble_type = st.selectbox(
            "Bubble Type",
            options=["small", "medium", "large", "thought"],
            index=1,
        )

        st.divider()
        st.subheader("Agents")
        st.write("🌐 Agent 0: Detects Japanese and translates contextually")
        st.write("🎭 Agent 1: Adapts cultural idioms to natural English")
        st.write("🧠 Agent 2: Matches character voice from memory")
        st.write("📐 Agent 3: Fits text into speech bubble limits")

    return {"character_name": character_name, "bubble_type": bubble_type}


def _load_characters_on_startup() -> None:
    """
    Load all character profiles into ChromaDB.
    """
    characters_folder = PROJECT_ROOT / "data" / "characters"
    load_characters_from_json(characters_folder)


def _results_tabs(raw_text: str, result: Dict[str, Any]) -> None:
    tabs = st.tabs(
        [
            "🌐 Translation",
            "🎭 Cultural Adaptation",
            "🧠 Continuity Check",
            "📐 Final Typeset Output",
        ]
    )

    translation_scores = result.get("translation_scores", {}) or {}
    cultural_scores = result.get("cultural_scores", {}) or {}
    continuity_scores = result.get("continuity_scores", {}) or {}
    typesetting_scores = result.get("typesetting_scores", {}) or {}

    detected_language = result.get("detected_language", "unknown")

    with tabs[0]:
        st.info(f"Detected language: {detected_language}")
        left, right = st.columns(2)
        with left:
            st.subheader("Original Input")
            st.text_area(
                "Original",
                value=raw_text,
                height=150,
                key="orig_view",
                disabled=True,
            )
        with right:
            st.subheader("Translated English")
            st.text_area(
                "Translated",
                value=result.get("translated_output", ""),
                height=150,
                key="translated_view",
                disabled=True,
            )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            _metric_int("Contextual accuracy", translation_scores.get("contextual_accuracy"))
        with m2:
            _metric_int("Tone preservation", translation_scores.get("tone_preservation"))
        with m3:
            _metric_int("Naturalness", translation_scores.get("naturalness"))
        with m4:
            st.metric("Pass", "✅" if translation_scores.get("pass") else "❌")

    with tabs[1]:
        left, right = st.columns(2)
        with left:
            st.subheader("Translation input")
            st.text_area(
                "Input to cultural adaptor",
                value=result.get("translated_output", ""),
                height=150,
                key="cultural_in_view",
                disabled=True,
            )
        with right:
            st.subheader("Cultural output")
            st.text_area(
                "Cultural output",
                value=result.get("cultural_output", ""),
                height=150,
                key="cultural_out_view",
                disabled=True,
            )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            _metric_int("Cultural accuracy", cultural_scores.get("cultural_accuracy"))
        with m2:
            _metric_int("Tone preservation", cultural_scores.get("tone_preservation"))
        with m3:
            _metric_int("Naturalness", cultural_scores.get("naturalness"))
        with m4:
            st.metric("Pass", "✅" if cultural_scores.get("pass") else "❌")

    with tabs[2]:
        left, right = st.columns(2)
        with left:
            st.subheader("Cultural input")
            st.text_area(
                "Input to continuity director",
                value=result.get("cultural_output", ""),
                height=150,
                key="cont_in_view",
                disabled=True,
            )
        with right:
            st.subheader("Continuity output")
            st.text_area(
                "Continuity output",
                value=result.get("continuity_output", ""),
                height=150,
                key="cont_out_view",
                disabled=True,
            )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            _metric_int("Voice consistency", continuity_scores.get("voice_consistency"))
        with m2:
            _metric_int(
                "Forbidden phrase compliance",
                continuity_scores.get("forbidden_phrase_compliance"),
            )
        with m3:
            _metric_int("Meaning preservation", continuity_scores.get("meaning_preservation"))
        with m4:
            st.metric("Pass", "✅" if continuity_scores.get("pass") else "❌")

    with tabs[3]:
        final_output = result.get("final_output", "")
        st.success(final_output)

        bubble_cfg = _load_bubble_config()
        max_chars = _bubble_limit(st.session_state.get("bubble_type", "medium"), bubble_cfg)
        st.write(f"**Character count**: {len(final_output)} / {max_chars}")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            _metric_int("Fits constraint", typesetting_scores.get("fits_constraint"))
        with m2:
            _metric_int("Meaning preserved", typesetting_scores.get("meaning_preserved"))
        with m3:
            _metric_int("Voice maintained", typesetting_scores.get("voice_maintained"))
        with m4:
            st.metric("Pass", "✅" if typesetting_scores.get("pass") else "❌")

        st.divider()

        download_payload = {
            "original_japanese": raw_text,
            "detected_language": result.get("detected_language"),
            "translated_output": result.get("translated_output"),
            "cultural_output": result.get("cultural_output"),
            "continuity_output": result.get("continuity_output"),
            "final_output": result.get("final_output"),
            "all_scores": {
                "translation": result.get("translation_scores"),
                "cultural": result.get("cultural_scores"),
                "continuity": result.get("continuity_scores"),
                "typesetting": result.get("typesetting_scores"),
            },
        }

        st.download_button(
            label="Download JSON output",
            data=json.dumps(download_payload, ensure_ascii=False, indent=2),
            file_name="lumina_pipeline_output.json",
            mime="application/json",
        )


def main() -> None:
    _init_page()

    try:
        _load_characters_on_startup()
    except Exception as exc:
        st.error(f"Failed to load character profiles: {exc}")

    settings = _sidebar()
    st.session_state["bubble_type"] = settings["bubble_type"]

    st.divider()

    raw_text = st.text_area(
        "Enter Raw Japanese Manga Script",
        placeholder=(
            "Paste raw Japanese manga dialogue here...\n"
            "e.g. たとえ天が崩れ落ちようとも、私は一歩も退かない。"
        ),
        height=150,
    )

    run_clicked = st.button("🚀 Run Localization Pipeline", type="primary")

    if run_clicked:
        if not raw_text.strip():
            st.error("Please paste some script text before running the pipeline.")
            return

        try:
            client = get_client()
        except Exception as exc:
            st.error(f"Failed to initialize Groq client: {exc}")
            return

        with st.spinner("Pipeline running... this may take 30 seconds"):
            try:
                result = run_pipeline(
                    raw_text=raw_text.strip(),
                    character_name=settings["character_name"],
                    bubble_type=settings["bubble_type"],
                    client=client,
                )
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                return

        st.session_state["last_result"] = result
        st.session_state["last_raw_text"] = raw_text.strip()

    # Render results (if present)
    if "last_result" in st.session_state and "last_raw_text" in st.session_state:
        st.divider()
        st.subheader("Results")
        _results_tabs(st.session_state["last_raw_text"], st.session_state["last_result"])


if __name__ == "__main__":
    main()

