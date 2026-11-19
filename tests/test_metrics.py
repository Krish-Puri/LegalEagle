from src.evals.metrics import classification_report, hit_at_k, span_overlap


def test_span_overlap():
    assert span_overlap((0, 10), (5, 15)) == 0.3333333333333333


def test_hit_at_k():
    assert hit_at_k({"a"}, ["b", "a"], 2) == 1.0
    assert hit_at_k({"a"}, ["b", "c"], 2) == 0.0


def test_classification_report():
    report = classification_report(["a", "b", "b"], ["a", "a", "b"])
    assert report["a"]["precision"] == 0.5
    assert report["b"]["recall"] == 0.5
