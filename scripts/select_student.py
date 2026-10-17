"""Pick the served student using VALIDATION metrics only (never test).

Rule: highest val accuracy vs teacher labels; ties broken by val macro-F1 vs teacher, then lower val loss.
Writes results/served_config.json and a per-config comparison table results/configs_comparison.json that also
includes peak VRAM / spill / train time and (report-only) test metrics from the HF evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    for summ in sorted((ROOT / "results/train").glob("*/summary.json")):
        s = json.loads(summ.read_text())
        b = s["best"]
        m_path = ROOT / f"results/preds/{s['name']}_metrics.json"
        m = json.loads(m_path.read_text()) if m_path.exists() else {}
        rows.append(
            {
                "name": s["name"],
                "base_model": s["config"]["base_model"],
                "lora_r": s["config"]["lora"]["r"],
                "quantization": s["config"]["quantization"],
                "trainable_params": s["trainable_params"],
                "best_epoch": s["best_epoch"],
                "val_acc_vs_teacher": b["val_acc_vs_teacher"],
                "val_f1_vs_teacher": b["val_f1_vs_teacher"],
                "val_acc_vs_gold": b["val_acc_vs_gold"],
                "val_loss": b["val_loss"],
                "train_time_s": s["train_time_s"],
                "peak_vram_gb": s["peak_vram_gb"],
                "peak_reserved_gb": s.get("peak_reserved_gb"),
                "spilled_to_shared_memory": s.get("spilled_to_shared_memory"),
                "test_770_acc_hf": m.get("test_770", {}).get("acc_vs_gold"),
                "test_770_f1_hf": m.get("test_770", {}).get("macro_f1_vs_gold"),
                "test_full_acc_hf": m.get("test_full", {}).get("acc_vs_gold"),
                "test_full_f1_hf": m.get("test_full", {}).get("macro_f1_vs_gold"),
            }
        )
    best = max(rows, key=lambda r: (r["val_acc_vs_teacher"], r["val_f1_vs_teacher"], -r["val_loss"]))
    rule = "max val acc vs teacher; tie-break val macro-F1 vs teacher, then lower val loss (test never used)"
    (ROOT / "results/configs_comparison.json").write_text(
        json.dumps({"rule": rule, "configs": rows}, indent=2) + "\n"
    )
    (ROOT / "results/served_config.json").write_text(
        json.dumps({**best, "selection_rule": rule}, indent=2) + "\n"
    )
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
