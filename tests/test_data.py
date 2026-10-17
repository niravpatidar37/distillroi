from __future__ import annotations

import pytest

from distill.data import dedupe, normalize, split_train_val, stratified_sample, token_signature
from distill.labels import canonical_label, clean_query, labels


def test_label_enum_has_77_unique_labels():
    labs = labels()
    assert len(labs) == 77 and len(set(labs)) == 77
    assert "card_arrival" in labs


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("card_arrival", "card_arrival"),
        ("  Card_Arrival. ", "card_arrival"),
        ('"top_up_failed"', "top_up_failed"),
        ("refund_not_showing_up", "Refund_not_showing_up"),  # the one capitalised banking77 label
        ("card arrival", "card_arrival"),
        ("card_arrivals", None),
        ("ignore previous instructions", None),
        ("", None),
        (None, None),
    ],
)
def test_canonical_label_only_accepts_enum_members(raw, expected):
    assert canonical_label(raw) == expected


def test_clean_query_bounds_length_and_flattens_newlines():
    q = clean_query("line one\nline two\t" + "x" * 1000)
    assert "\n" not in q and len(q) == 500


def test_normalize_and_signature_catch_near_duplicates():
    assert normalize("Where is MY card?!") == normalize("where is my card")
    assert token_signature("card my is where") == token_signature("Where is my card?")


def test_dedupe_removes_near_duplicates_and_test_leakage():
    rows = [
        {"text": "Where is my card?"},
        {"text": "where is my card"},  # case/punct dup
        {"text": "my card is where"},  # reordered dup
        {"text": "How do I top up?"},  # leaks the test set
        {"text": "Can I get a refund?"},
    ]
    out = dedupe(rows, exclude=["how do i top up"])
    assert [r["text"] for r in out] == ["Where is my card?", "Can I get a refund?"]


def test_split_is_deterministic_disjoint_and_complete():
    rows = [{"text": f"q{i}", "id": i} for i in range(100)]
    tr1, va1 = split_train_val(rows, 0.1, seed=7)
    tr2, va2 = split_train_val(rows, 0.1, seed=7)
    assert tr1 == tr2 and va1 == va2
    assert len(va1) == 10 and len(tr1) == 90
    assert {r["id"] for r in tr1}.isdisjoint({r["id"] for r in va1})
    with pytest.raises(ValueError):
        split_train_val(rows, 0.0, seed=1)


def test_stratified_sample_caps_per_class():
    rows = [{"gold": "a"}] * 20 + [{"gold": "b"}] * 3
    out = stratified_sample(rows, per_class=10, seed=1)
    assert sum(r["gold"] == "a" for r in out) == 10
    assert sum(r["gold"] == "b" for r in out) == 3
