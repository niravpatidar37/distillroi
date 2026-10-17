"""Download banking77 (pinned commit), dedupe, build the frozen test subset and the teacher-labelling pool.

Outputs (small, committed except raw/):
  data/raw/{train,test}.csv            raw CSVs from the PolyAI repo (gitignored)
  src/distill/banking77_labels.json    the 77-label enum
  data/processed/test_full.jsonl       all 3,080 test rows, gold labels (student-only eval, free)
  data/processed/test_770.jsonl        stratified 10-per-class test subset: teacher + student eval (frozen)
  data/processed/pool.jsonl            deduped training-split inputs, shuffled, for teacher labelling
  results/data_manifest.json           checksums and counts
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distill.data import dedupe, read_csv_rows, shuffled, stable_id, stratified_sample, write_jsonl  # noqa: E402

COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"  # PolyAI-LDN/task-specific-datasets, CC-BY-4.0
BASE = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{COMMIT}/banking_data"
SEED = 1337


def fetch(name: str) -> Path:
    dest = ROOT / "data" / "raw" / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(f"{BASE}/{name}", timeout=60, follow_redirects=True)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


def main() -> None:
    train_csv, test_csv = fetch("train.csv"), fetch("test.csv")
    train = [{"text": r["text"].strip(), "gold": r["category"].strip()} for r in read_csv_rows(train_csv)]
    test = [{"text": r["text"].strip(), "gold": r["category"].strip()} for r in read_csv_rows(test_csv)]
    labels = sorted({r["gold"] for r in train})
    assert len(labels) == 77, len(labels)
    assert {r["gold"] for r in test} <= set(labels)
    (ROOT / "src" / "distill" / "banking77_labels.json").write_text(json.dumps(labels, indent=0) + "\n")

    for r in test:
        r["id"] = "test-" + stable_id(r["text"])
    # shuffled so teacher batches are not label-homogeneous (that would make the ceiling too easy)
    test_770 = shuffled(stratified_sample(test, per_class=10, seed=SEED), SEED)
    write_jsonl(ROOT / "data/processed/test_full.jsonl", test)
    write_jsonl(ROOT / "data/processed/test_770.jsonl", test_770)

    # Training pool: dedupe near-identical inputs AND drop anything near-duplicating a test input (leakage).
    pool = dedupe(train, exclude=[r["text"] for r in test])
    for r in pool:
        r["id"] = "train-" + stable_id(r["text"])
    pool = shuffled(pool, SEED)
    write_jsonl(ROOT / "data/processed/pool.jsonl", pool)

    manifest = {
        "source": f"https://github.com/PolyAI-LDN/task-specific-datasets/tree/{COMMIT}/banking_data",
        "license": "CC-BY-4.0",
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_csv, test_csv)},
        "counts": {
            "train_raw": len(train),
            "test_raw": len(test),
            "pool_after_dedupe_and_test_leakage_filter": len(pool),
            "dropped_by_dedupe": len(train) - len(pool),
            "test_770_stratified_10_per_class": len(test_770),
        },
        "seed": SEED,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
