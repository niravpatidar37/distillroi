"""Task metrics without sklearn (CI installs no ML stack). Matches sklearn's macro-F1 definition:
labels = union of gold and predicted labels; a None prediction counts as wrong."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def accuracy(gold: Sequence[str], pred: Sequence[str | None]) -> float:
    if len(gold) != len(pred):
        raise ValueError("length mismatch")
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold)


def macro_f1(gold: Sequence[str], pred: Sequence[str | None]) -> float:
    if len(gold) != len(pred):
        raise ValueError("length mismatch")
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for g, p in zip(gold, pred, strict=True):
        if g == p:
            tp[g] += 1
        else:
            fn[g] += 1
            if p is not None:
                fp[p] += 1
    labels = set(gold) | {p for p in pred if p is not None}
    if not labels:
        return 0.0
    f1s = []
    for lab in labels:
        denom = 2 * tp[lab] + fp[lab] + fn[lab]
        f1s.append(0.0 if denom == 0 else 2 * tp[lab] / denom)
    return sum(f1s) / len(f1s)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default method)."""
    if not values:
        raise ValueError("empty")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
