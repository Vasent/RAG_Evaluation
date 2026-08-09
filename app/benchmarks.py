"""Loading and matching helpers for the small labeled evaluation dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_benchmarks(path: str | Path) -> List[Dict[str, Any]]:
    """Load benchmark records and fail early when the file shape is invalid."""
    benchmark_path = Path(path)
    with benchmark_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError("Evaluation dataset must be a JSON list")

    required = {"id", "question", "reference_answer", "answerable", "relevant_documents"}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Evaluation record {index} must be an object")
        missing = required.difference(record)
        if missing:
            raise ValueError(
                f"Evaluation record {record.get('id', index)!r} is missing: "
                f"{', '.join(sorted(missing))}"
            )
    return records


def normalize_question(question: str) -> str:
    """Normalize harmless punctuation/spacing differences for UI matching."""
    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


def find_benchmark(
    question: str,
    records: Iterable[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return a record only for an exact normalized benchmark question."""
    normalized = normalize_question(question)
    return next(
        (record for record in records if normalize_question(record["question"]) == normalized),
        None,
    )


def document_id(source: Any, page: Any) -> str:
    """Create the source/page identifier used by the gold labels."""
    return f"{Path(str(source)).name}#page={page}"


def metadata_document_id(metadata: Dict[str, Any]) -> str:
    """Get a stable page-level ID from chunk metadata."""
    return str(
        metadata.get("document_id")
        or document_id(metadata.get("source", "Unknown"), metadata.get("page", "N/A"))
    )


def relevant_document_ids(relevant_documents: Iterable[Dict[str, Any]]) -> set[str]:
    """Convert benchmark source/page pairs into comparable IDs."""
    return {
        document_id(item.get("source", "Unknown"), item.get("page", "N/A"))
        for item in relevant_documents
    }
