## Character Profiles

This directory is intended to hold JSON files that describe character profiles and voice guidelines.

### Suggested Schema

Each character file might include fields such as:
- `id`: Stable identifier for the character.
- `name`: Display name.
- `role`: Narrative role (e.g., "protagonist", "rival", "mentor").
- `voice`: Free-form description of speech patterns, quirks, and tone.
- `notes`: Additional localization notes or constraints.

These files will ultimately be:
- Loaded at startup.
- Indexed into ChromaDB via `memory/vector_store.py`.
- Queried by the Continuity Director Agent to enforce voice consistency.

