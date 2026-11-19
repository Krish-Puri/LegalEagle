from src.risk.scoring import (
    CATEGORY_BY_CLAUSE,
    category_for_clause,
    category_scores,
    evaluate_clause_risk,
    overall_score,
)
from src.risk.triggers import detailed_risk_triggers, triggering_language


def clause_risk_score(risk_level: str, confidence: float, trigger_count: int = 0) -> int:
    """Backwards-compatible helper."""
    base = {"low": 20, "medium": 45, "high": 72, "critical": 90}.get(risk_level, 20)
    conf_adj = int((confidence - 0.5) * 12)
    trig_adj = min(15, trigger_count * 4)
    return max(5, min(100, base + conf_adj + trig_adj))


__all__ = [
    "CATEGORY_BY_CLAUSE",
    "category_for_clause",
    "category_scores",
    "evaluate_clause_risk",
    "clause_risk_score",
    "overall_score",
    "detailed_risk_triggers",
    "triggering_language",
]
