"""
ChromaDB setup and query utilities.

This module provides:
- A configured ChromaDB client instance.
- Collection initialization for character profiles and dialogue snippets.
- High-level helper functions for agents to query and update memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import json

import chromadb
from chromadb.api.models.Collection import Collection


@dataclass
class VectorStoreConfig:
    """
    Configuration for the ChromaDB vector store.

    This includes:
    - Persistent directory path.
    - Collection name for character profiles.
    """

    persist_directory: Path
    character_collection_name: str = "character_profiles"


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default configuration: persist vector data under `data/chroma_db/`.
DEFAULT_CONFIG = VectorStoreConfig(
    persist_directory=PROJECT_ROOT / "data" / "chroma_db",
)

_CLIENT: Optional[chromadb.PersistentClient] = None
_CHARACTER_COLLECTION: Optional[Collection] = None


def get_chroma_client(config: VectorStoreConfig = DEFAULT_CONFIG) -> chromadb.PersistentClient:
    """
    Initialize and return a ChromaDB client instance.

    Args:
        config: Vector store configuration. Defaults to `DEFAULT_CONFIG`.

    Returns:
        A ChromaDB client object.
    """
    global _CLIENT

    if _CLIENT is None:
        config.persist_directory.mkdir(parents=True, exist_ok=True)
        _CLIENT = chromadb.PersistentClient(path=str(config.persist_directory))

    return _CLIENT


def get_character_collection(
    config: VectorStoreConfig = DEFAULT_CONFIG,
) -> Collection:
    """
    Get or create the ChromaDB collection used for character profiles.

    Args:
        config: Vector store configuration. Defaults to `DEFAULT_CONFIG`.

    Returns:
        A ChromaDB Collection for character data.
    """
    global _CHARACTER_COLLECTION

    if _CHARACTER_COLLECTION is None:
        client = get_chroma_client(config)
        _CHARACTER_COLLECTION = client.get_or_create_collection(
            name=config.character_collection_name,
        )

    return _CHARACTER_COLLECTION


def _character_id_from_name(character_name: str) -> str:
    """
    Derive a stable identifier for a character based on their name.
    """
    return f"character:{character_name.strip().lower()}"


def add_character_profile(
    character_name: str,
    profile_data: Dict[str, Any],
    *,
    config: VectorStoreConfig = DEFAULT_CONFIG,
) -> None:
    """
    Add or update a single character profile in the vector store.

    Args:
        character_name: Display name of the character.
        profile_data: Dictionary containing profile details.
    """
    collection = get_character_collection(config)
    character_id = _character_id_from_name(character_name)

    document = json.dumps(profile_data, ensure_ascii=False)

    # Use upsert so re-running loaders will refresh profiles.
    collection.upsert(
        ids=[character_id],
        documents=[document],
        metadatas=[{"name": character_name}],
    )


def query_character_profile(
    character_name: str,
    *,
    config: VectorStoreConfig = DEFAULT_CONFIG,
) -> Optional[str]:
    """
    Retrieve a character profile document from ChromaDB.

    Args:
        character_name: Display name of the character.

    Returns:
        The stored profile as a JSON string, or None if not found.
    """
    collection = get_character_collection(config)
    character_id = _character_id_from_name(character_name)

    result = collection.get(ids=[character_id])

    documents = result.get("documents") or []
    if not documents or not documents[0]:
        return None

    # `documents` is a list of lists for some Chroma versions; handle both shapes.
    doc = documents[0]
    if isinstance(doc, list):
        return doc[0] if doc else None
    return doc


def load_characters_from_json(folder_path: str | Path) -> None:
    """
    Load all JSON character profiles from the specified folder into ChromaDB.

    Args:
        folder_path: Path to the folder containing `.json` character files.
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"[VectorStore] Character folder not found: {folder}")
        return

    for json_file in folder.glob("*.json"):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[VectorStore] Failed to read {json_file.name}: {exc}")
            continue

        character_name = data.get("name") or json_file.stem
        add_character_profile(character_name, data)
        print(f"[VectorStore] Loaded character profile: {character_name}")


def test_vector_store() -> None:
    """
    Simple test helper to validate that the vector store is working.

    This will:
    - Load character profiles from `data/characters/`.
    - Query back the sample character "Kira".
    - Print the result for manual inspection.
    """
    characters_folder = PROJECT_ROOT / "data" / "characters"
    load_characters_from_json(characters_folder)

    profile_str = query_character_profile("Kira")
    if profile_str is None:
        print("[VectorStore Test] Failed to retrieve profile for 'Kira'.")
        return

    print("[VectorStore Test] Retrieved profile for 'Kira':")
    print(profile_str)

