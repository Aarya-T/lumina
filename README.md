## Lumina Pipeline

Autonomous multi-agent localization engine for serial graphic novels (manga/webtoon).

### Overview

This project processes a raw Japanese/Korean script through three sequential agents to produce a polished, localized English script.

- Cultural Adaptor Agent
- Continuity Director Agent
- Typesetting Editor Agent

### Tech Stack

- Python 3.11+
- CrewAI (agent orchestration)
- Groq API (LLM)
- ChromaDB (vector memory)
- Streamlit (frontend)
- python-dotenv (environment management)
- LangChain (optional utilities / integrations)

### Getting Started

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file (or update the existing one) with your keys:

```bash
GROQ_API_KEY=your-groq-api-key
```

4. Run the Streamlit app:

```bash
streamlit run ui/app.py
```

### Development Notes

- All core logic is organized under the `agents`, `memory`, `data`, and `utils` packages.
- The entrypoint for programmatic use is `main.py`.
- The Streamlit UI lives in `ui/app.py`.

# lumina
