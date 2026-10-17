"""Freeze train/val BEFORE any tuning, from the teacher-labelled pool.

Keeps only rows whose teacher label passed enum validation (failures are dropped, not taught).
Writes data/processed/{train,val}.jsonl and results/splits_summary.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distill.data import dedupe, read_jsonl, split_train_val, write_jsonl  # noqa: E402
from distill.labels import label_set  # noqa: E402
from distill.metrics import accuracy  # noqa: E402

SEED = 2024
VAL_FRAC = 0.1


def main() -> None:
    pool = read_jsonl(ROOT / "data/processed/teacher_labeled_pool.jsonl")
    valid = [r for r in pool if r.get("teacher") in label_set()]
    valid = dedupe(valid)  # idempotent safety net (pool is already deduped)
    train, val = split_train_val(valid, VAL_FRAC, SEED)
    write_jsonl(ROOT / "data/processed/train.jsonl", train)
    write_jsonl(ROOT / "data/processed/val.jsonl", val)
    teacher_vs_gold = accuracy([r["gold"] for r in valid], [r["teacher"] for r in valid])
    summary = {
        "teacher_labelled_attempted": len(pool),
        "dropped_invalid_teacher_output": len(pool) - len(valid),
        "kept": len(valid),
        "train": len(train),
        "val": len(val),
        "labels_covered_in_train": len({r["teacher"] for r in train}),
        "min_examples_per_label_train": min(Counter(r["teacher"] for r in train).values()),
        "teacher_label_agreement_with_gold_on_kept": teacher_vs_gold,
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "test_set": "data/processed/test_770.jsonl (banking77 test split, gold; untouched until Phase 4)",
    }
    (ROOT / "results/splits_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
