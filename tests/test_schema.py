from pydantic import ValidationError

from src.schema import RiskFinding, RiskScorecard


def test_scorecard_validates_score_bounds():
    finding = RiskFinding(
        clause_type="termination",
        quote="Vendor may terminate without notice.",
        page=1,
        start_offset=0,
        end_offset=36,
        risk_level="high",
        rationale="Immediate termination risk.",
        business_impact="Operational disruption.",
        suggested_negotiation_note="Add notice and cure period.",
    )
    scorecard = RiskScorecard(
        contract_summary="Review complete.",
        overall_risk_score=60,
        risk_level="high",
        findings=[finding],
    )

    assert scorecard.findings[0].clause_type == "termination"


def test_scorecard_rejects_invalid_score():
    try:
        RiskScorecard(contract_summary="x", overall_risk_score=101, risk_level="high")
    except ValidationError:
        return
    raise AssertionError("Expected validation error")
