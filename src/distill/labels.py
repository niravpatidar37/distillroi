"""The closed banking77 label set and the prompts shared by teacher, student training and serving."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

LABELS_RESOURCE = "banking77_labels.json"

# Student prompt: identical at training time (scripts/train.py) and serving time (student.py).
# The student has learned the label set, so the list is NOT in its prompt (short prompt = cheap serving).
STUDENT_SYSTEM = (
    "Classify the bank customer message into one banking77 intent. Reply with the intent label only."
)

MAX_QUERY_CHARS = 500


@lru_cache(maxsize=1)
def labels() -> tuple[str, ...]:
    raw = resources.files("distill").joinpath(LABELS_RESOURCE).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list) or len(data) != 77 or len(set(data)) != 77:
        raise ValueError("banking77 label file must hold 77 unique labels")
    return tuple(str(x) for x in data)


@lru_cache(maxsize=1)
def label_set() -> frozenset[str]:
    return frozenset(labels())


@lru_cache(maxsize=1)
def _lower_map() -> dict[str, str]:
    # banking77 has one capitalised label ("Refund_not_showing_up"); match case-insensitively.
    return {lab.lower(): lab for lab in labels()}


def canonical_label(raw: str | None) -> str | None:
    """Map a model's raw label string onto the closed enum, or None if it is not a valid label.

    Only trivial formatting noise is forgiven (case, surrounding quotes/whitespace/period). Anything else is
    invalid: the router treats invalid output as a schema failure and escalates.
    """
    if raw is None:
        return None
    s = raw.strip().strip("`\"' .").lower().replace(" ", "_").replace("-", "_")
    return _lower_map().get(s)


def clean_query(text: str) -> str:
    """Single-line, length-bounded query text (inputs are untrusted)."""
    return " ".join(text.split())[:MAX_QUERY_CHARS]
