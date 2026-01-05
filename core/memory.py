"""
EcoMoveAI - Lightweight memory notes.

Purpose
-------
Provide a local, append-only memory store for policy notes and context.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json


@dataclass(frozen=True)
class MemoryNote:
    """
    Simple memory note for policy context.
    """

    timestamp_utc: str
    title: str
    content: str
    tags: list[str]


def append_memory_note(
    path: Path,
    title: str,
    content: str,
    tags: Optional[list[str]] = None,
) -> None:
    """
    Append a memory note to a JSONL file.
    """

    if not title or not title.strip():
        raise ValueError("title must be a non-empty string.")
    if not content or not content.strip():
        raise ValueError("content must be a non-empty string.")

    note = MemoryNote(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        title=title.strip(),
        content=content.strip(),
        tags=[tag.strip() for tag in (tags or []) if tag.strip()],
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(note.__dict__) + "\n")


def load_memory_notes(path: Path, limit: int = 5) -> list[MemoryNote]:
    """
    Load up to the most recent memory notes.
    """

    if not path.exists():
        return []

    notes: list[MemoryNote] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            notes.append(
                MemoryNote(
                    timestamp_utc=payload.get("timestamp_utc", ""),
                    title=payload.get("title", ""),
                    content=payload.get("content", ""),
                    tags=list(payload.get("tags", []) or []),
                )
            )

    if limit <= 0:
        return []
    return notes[-limit:]


def format_memory_notes(notes: list[MemoryNote]) -> list[dict[str, object]]:
    """
    Convert memory notes to dictionaries for serialization.
    """

    return [note.__dict__ for note in notes]
