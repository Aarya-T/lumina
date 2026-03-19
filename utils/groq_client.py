from pathlib import Path
from typing import Optional

import os

from dotenv import load_dotenv
from groq import Groq


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_environment() -> Groq:
    """
    Load environment variables from the .env file and initialize the Groq client.

    This ensures that secrets like GROQ_API_KEY are available
    to the rest of the application and returns a configured Groq client.

    Returns:
        Groq: An initialized Groq client instance.

    Raises:
        RuntimeError: If the GROQ_API_KEY is missing.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Please set it in your .env file."
        )

    # Initialize and return the Groq client using the official Python package.
    return Groq(api_key=api_key)

