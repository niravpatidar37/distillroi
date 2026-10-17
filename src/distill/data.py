"""Dataset utilities: near-duplicate normalisation, deterministic splits, JSONL I/O.

Pure Python (no numpy/pandas) so CI can test it without the GPU extras.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Canonical form used for exact-ish duplicate detection (case/punctuation/space insensitive)."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def token_signature(text: str) -> str:
    """Order-insensitive bag-of-words signature: catches re-ordered near-duplicates."""
    return " ".join(sorted(set(normalize(text).split())))


def dedupe_keys(text: str) -> tuple[str, str]:
    return normalize(text), token_signature(text)


def dedupe(
    rows: Iterable[dict[str, Any]], *, exclude: Iterable[str] = (), key: str = "text"
) -> list[dict[str, Any]]:
    """Drop rows whose text is a near-duplicate of an earlier row or of any text in ``exclude``.

    ``exclude`` is used to remove training inputs that near-duplicate test inputs (leakage control).
    """
    seen: set[str] = set()
    for t in exclude:
        seen.update(dedupe_keys(t))
    out: list[dict[str, Any]] = []
    for row in rows:
        k1, k2 = dedupe_keys(str(row[key]))
        if not k1 or k1 in seen or k2 in seen:
            continue
        seen.add(k1)
        seen.add(k2)
        out.append(row)
    return out


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def shuffled(rows: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    out = list(rows)
    random.Random(seed).shuffle(out)  # noqa: S311  # nosec B311 - reproducible data shuffling, not crypto
    return out


def split_train_val(
    rows: Sequence[dict[str, Any]], val_frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic random split. Called once, before any tuning, and the result is frozen to disk."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in (0, 1)")
    rows = shuffled(rows, seed)
    n_val = max(1, round(len(rows) * val_frac))
    return rows[n_val:], rows[:n_val]


def stratified_sample(
    rows: Sequence[dict[str, Any]], per_class: int, seed: int, label_key: str = "gold"
) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_label[str(r[label_key])].append(r)
    out: list[dict[str, Any]] = []
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        out.extend(group[:per_class])
    return out


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
