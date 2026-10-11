"""Emotion taxonomy.

The data folders hold eight classes, but the original app hard-coded seven (no ``contempt``)
and spelled one of them ``suprise``, so its label lookup was off by one class after
``angry``. Here the class list is defined once and travels inside the exported model.
"""

from __future__ import annotations

EMOTIONS: tuple[str, ...] = (
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

# Image sources, told apart by file format and size (see ``data.manifest.source_of``).
SOURCES: tuple[str, ...] = ("fer48", "hr224")

_ALIASES = {"anger": "angry", "suprise": "surprise", "happiness": "happy", "sadness": "sad"}


def parse_folder(name: str) -> str:
    """Map a class folder name to a canonical emotion, tolerating the old misspellings."""
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in EMOTIONS:
        raise ValueError(f"unrecognised emotion folder: {name!r}")
    return key
