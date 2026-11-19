from __future__ import annotations

from typing import Any


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": tp + fn,
    }


def classification_report(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    report: dict[str, Any] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    f1_list = []

    # Confusion matrix tracker
    matrix: dict[str, dict[str, int]] = {true_lbl: {pred_lbl: 0 for pred_lbl in labels} for true_lbl in labels}

    for true, pred in zip(y_true, y_pred):
        if true in matrix and pred in matrix[true]:
            matrix[true][pred] += 1

    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        stats = precision_recall_f1(tp, fp, fn)
        report[label] = stats
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if stats["support"] > 0:
            f1_list.append(stats["f1"])

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    overall = precision_recall_f1(total_tp, total_fp, total_fn)

    report["_macro_avg"] = {
        "precision": round(sum(report[lbl]["precision"] for lbl in labels if report[lbl]["support"] > 0) / max(1, len(f1_list)), 4),
        "recall": round(sum(report[lbl]["recall"] for lbl in labels if report[lbl]["support"] > 0) / max(1, len(f1_list)), 4),
        "f1": round(macro_f1, 4),
    }
    report["_micro_avg"] = overall
    report["_confusion_matrix"] = matrix
    return report


def span_overlap(expected: tuple[int, int], predicted: tuple[int, int]) -> float:
    expected_start, expected_end = expected
    predicted_start, predicted_end = predicted
    intersection = max(0, min(expected_end, predicted_end) - max(expected_start, predicted_start))
    union = max(expected_end, predicted_end) - min(expected_start, predicted_start)
    return intersection / union if union else 0.0


def hit_at_k(expected_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not expected_ids or not ranked_ids:
        return 0.0
    top_k = ranked_ids[:k]
    return 1.0 if any(eid in top_k for eid in expected_ids) else 0.0


def mean_reciprocal_rank(queries: list[tuple[set[str], list[str]]]) -> float:
    """Compute Mean Reciprocal Rank (MRR) across queries."""
    if not queries:
        return 0.0
    rr_sum = 0.0
    for expected_ids, ranked_ids in queries:
        rr = 0.0
        for rank, rid in enumerate(ranked_ids, start=1):
            if rid in expected_ids:
                rr = 1.0 / rank
                break
        rr_sum += rr
    return round(rr_sum / len(queries), 4)


def validate_schema_rate(outputs: list[Any]) -> float:
    if not outputs:
        return 0.0
    valid = sum(1 for item in outputs if getattr(item, "overall_risk_score", None) is not None and getattr(item, "findings", None) is not None)
    return round(valid / len(outputs), 4)
