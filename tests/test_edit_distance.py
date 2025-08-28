from openeval.metrics.edit_distance import CharEditDistance


def test_edit_distance_basic():
    m = CharEditDistance()
    out = m.compute(["kitten", "abc"], ["sitting", "abc"])
    assert out["avg_distance"] >= 0
    assert 0.0 <= out["avg_similarity"] <= 1.0


def test_edit_distance_empty():
    m = CharEditDistance()
    out = m.compute([], [])
    assert out["avg_distance"] == 0.0
    assert out["avg_similarity"] == 0.0
