"""Tune the escalation threshold on VALIDATION, then apply it once to the untouched test subset.

Inputs (all produced by earlier steps):
  results/preds/served_val.jsonl       vLLM student predictions + confidence on val (val rows carry teacher labels)
  results/preds/served_test_770.jsonl  vLLM student predictions + confidence on the frozen test subset
  results/teacher_test_770.jsonl       teacher predictions on the same test subset (Phase 1 ceiling run)

Output: results/router_eval.json (+ threshold curves for plotting)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distill.data import read_jsonl  # noqa: E402
from distill.metrics import accuracy, macro_f1  # noqa: E402
from distill.router import simulate_routing, tune_threshold  # noqa: E402

QUALITY_TARGET = 0.99  # routed accuracy must reach 99% of the teacher's (on val), at the lowest escalation


def curve(
    gold: list[str], s_lab: list[str | None], s_conf: list[float], t_lab: list[str | None]
) -> list[dict]:
    out = []
    for t in [i / 20 for i in range(0, 20)] + [0.97, 0.99]:
        routed, esc = simulate_routing(s_lab, s_conf, t_lab, t)
        out.append({"threshold": t, "accuracy": accuracy(gold, routed), "escalation_rate": esc})
    return out


def main() -> None:
    val = read_jsonl(ROOT / "results/preds/served_val.jsonl")
    test = read_jsonl(ROOT / "results/preds/served_test_770.jsonl")
    teacher_test = {r["id"]: r.get("teacher") for r in read_jsonl(ROOT / "results/teacher_test_770.jsonl")}

    v_gold = [str(r["gold"]) for r in val]
    v_s = [r["student"] for r in val]
    v_c = [float(r["conf"]) for r in val]
    v_t = [r["teacher"] for r in val]
    best = tune_threshold(v_gold, v_s, v_c, v_t, quality_target=QUALITY_TARGET)
    thr = best["threshold"]

    t_gold = [str(r["gold"]) for r in test]
    t_s = [r["student"] for r in test]
    t_c = [float(r["conf"]) for r in test]
    t_t = [teacher_test.get(r["id"]) for r in test]
    routed, esc = simulate_routing(t_s, t_c, t_t, thr)
    invalid = sum(s is None for s in t_s)
    low_conf = sum(s is not None and c < thr for s, c in zip(t_s, t_c, strict=True))
    out = {
        "selection": f"threshold tuned on VAL only: lowest escalation with routed acc >= {QUALITY_TARGET} x teacher",
        "threshold": thr,
        "val": best,
        "test_770": {
            "n": len(test),
            "escalation_rate": esc,
            "escalated_invalid_output": invalid,
            "escalated_low_confidence": low_conf,
            "routed_accuracy": accuracy(t_gold, routed),
            "routed_macro_f1": macro_f1(t_gold, routed),
            "student_accuracy": accuracy(t_gold, t_s),
            "student_macro_f1": macro_f1(t_gold, t_s),
            "teacher_accuracy": accuracy(t_gold, t_t),
            "teacher_macro_f1": macro_f1(t_gold, t_t),
            "note": "escalated items use the teacher's Phase-1 prediction for that query (batched prompt)",
        },
        "curve_val": curve(v_gold, v_s, v_c, v_t),
        "curve_test_report_only": curve(t_gold, t_s, t_c, t_t),
    }
    (ROOT / "results/router_eval.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("threshold", "val", "test_770")}, indent=2))


if __name__ == "__main__":
    main()
