"""Evaluate trained students with HF transformers (no server needed) and merge the best adapter for vLLM.

  python scripts/eval_student_hf.py q05_r8 --split val --split test_770 --split test_full
  python scripts/eval_student_hf.py q05_r8 --merge      # writes artifacts/merged/q05_r8 (bf16, gitignored)

Uses the best epoch recorded in results/train/<name>/summary.json (chosen on VAL only).
Predictions go to results/preds/<name>_<split>.jsonl, metrics to results/preds/<name>_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from student_hf import generate_labels, load_tokenizer  # noqa: E402

from distill.data import read_jsonl, write_jsonl  # noqa: E402
from distill.metrics import accuracy, macro_f1  # noqa: E402

SPLITS = {
    "val": ROOT / "data/processed/val.jsonl",
    "test_770": ROOT / "data/processed/test_770.jsonl",
    "test_full": ROOT / "data/processed/test_full.jsonl",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--split", action="append", default=[])
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    summary = json.loads((ROOT / f"results/train/{a.name}/summary.json").read_text())
    cfg = summary["config"]
    adapter = ROOT / f"artifacts/adapters/{a.name}/epoch-{summary['best_epoch']}"

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(cfg["base_model"], cfg.get("revision"))
    # Evaluate/merge on a bf16 base: this is exactly what vLLM serves after merging.
    base = AutoModelForCausalLM.from_pretrained(  # nosec B615 - revision pinned in config
        cfg["base_model"], revision=cfg.get("revision"), dtype=torch.bfloat16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()

    if a.merge:
        out = ROOT / f"artifacts/merged/{a.name}"
        model.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)
        print("merged ->", out)

    metrics_path = ROOT / f"results/preds/{a.name}_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    for split in a.split:
        rows = read_jsonl(SPLITS[split])
        t0 = time.time()
        preds, confs = generate_labels(model, tok, [str(r["text"]) for r in rows], batch_size=32)
        el = time.time() - t0
        write_jsonl(
            ROOT / f"results/preds/{a.name}_{split}.jsonl",
            (
                {"id": r["id"], "gold": r["gold"], "teacher": r.get("teacher"), "student": p, "conf": c}
                for r, p, c in zip(rows, preds, confs, strict=True)
            ),
        )
        gold = [str(r["gold"]) for r in rows]
        m = {
            "n": len(rows),
            "acc_vs_gold": accuracy(gold, preds),
            "macro_f1_vs_gold": macro_f1(gold, preds),
            "invalid_rate": sum(p is None for p in preds) / len(preds),
            "hf_batch32_seconds": round(el, 2),
            "best_epoch": summary["best_epoch"],
        }
        if rows and rows[0].get("teacher") is not None:
            m["acc_vs_teacher"] = accuracy([str(r["teacher"]) for r in rows], preds)
        metrics[split] = m
        print(split, json.dumps(m))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
