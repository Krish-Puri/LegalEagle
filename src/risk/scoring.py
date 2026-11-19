from __future__ import annotations

import re
from collections import defaultdict
from typing import Literal

from src.risk.triggers import detailed_risk_triggers
from src.schema import ReviewingParty, RiskFinding, RiskLevel, risk_weight


CATEGORY_BY_CLAUSE: dict[str, str] = {
    "indemnification": "liability_exposure",
    "limitation of liability": "liability_exposure",
    "termination": "termination_risk",
    "assignment": "operational_control",
    "ip assignment": "ip_ownership",
    "confidentiality": "information_protection",
    "exclusivity": "commercial_restrictions",
    "non-compete / non-solicit": "commercial_restrictions",
    "governing law": "dispute_resolution",
    "change of control": "operational_control",
    "most favored nation": "commercial_restrictions",
    "audit rights": "compliance_audit",
    "revenue / payment obligations": "payment_terms",
}

# Base score anchors per severity tier
BASE_SCORES: dict[RiskLevel, int] = {
    "low": 20,
    "medium": 45,
    "high": 75,
    "critical": 92,
}


def category_for_clause(clause_type: str) -> str:
    return CATEGORY_BY_CLAUSE.get(clause_type, "general")


def evaluate_clause_risk(
    clause_type: str,
    clause_text: str,
    base_level: RiskLevel,
    confidence: float,
    reviewing_party: ReviewingParty = "customer",
    reviewer_preferences: str = "",
) -> tuple[int, RiskLevel, list[str], str]:
    """Calculate an explainable, perspective-aware risk score (0-100) and rationale.

    Components:
    1. Base tier score (low: 20, medium: 45, high: 75, critical: 92)
    2. Detailed trigger detection (asymmetry, dollar caps, duration, broad indemnities)
    3. Reviewing party alignment (Customer vs Vendor vs Neutral)
    4. Reviewer preference sensitivity
    5. Confidence calibration
    """
    triggers = detailed_risk_triggers(clause_text)
    risk_factors: list[str] = [t["id"] for t in triggers]
    trigger_displays: list[str] = [t["display"] for t in triggers]

    # Perspective evaluation
    is_customer = reviewing_party == "customer"
    is_vendor = reviewing_party == "vendor"

    # Start with base score
    score = float(BASE_SCORES.get(base_level, 30))

    # Evaluate specific high-impact risk conditions
    has_unlimited_customer_liability = "unlimited_customer_liability" in risk_factors
    has_nominal_vendor_cap = "nominal_vendor_cap" in risk_factors
    has_immediate_termination = "unilateral_immediate_termination" in risk_factors
    has_broad_indemnity = "broad_indemnity_all_losses" in risk_factors
    has_irrevocable_ip = "irrevocable_ip_assignment" in risk_factors or "vendor_retains_all_deliverables" in risk_factors
    has_extended_noncompete = any("noncompete" in f for f in risk_factors)
    has_unilateral_assignment = "unilateral_assignment" in risk_factors

    if is_customer:
        # From Customer perspective:
        # A. Asymmetric unlimited liability + nominal $50 vendor cap is catastrophic
        if has_unlimited_customer_liability and has_nominal_vendor_cap:
            score = 98.0
            base_level = "critical"
        elif has_unlimited_customer_liability or has_nominal_vendor_cap:
            score = max(score, 94.0)
            base_level = "critical"

        # B. Unilateral immediate termination
        if has_immediate_termination:
            score = max(score, 92.0)
            base_level = "critical"

        # C. Broad indemnity for all losses
        if has_broad_indemnity:
            score = max(score, 85.0)
            if base_level != "critical":
                base_level = "high"

        # D. Complete work product & custom code IP forfeiture
        if has_irrevocable_ip:
            score = max(score, 82.0)
            if base_level != "critical":
                base_level = "high"

        # E. Extended non-compete (e.g. 5 years)
        if has_extended_noncompete:
            score = max(score, 84.0)
            if base_level != "critical":
                base_level = "high"

        # F. Unilateral assignment
        if has_unilateral_assignment:
            score = max(score, 78.0)
            if base_level != "critical":
                base_level = "high"

    elif is_vendor:
        # From Vendor perspective:
        # If the clause severely protects Vendor (e.g. vendor cap of $50, customer uncapped,
        # vendor immediate termination), the operational exposure for Vendor is minimal!
        # It is favorable language, though might cause negotiation friction.
        if has_nominal_vendor_cap or has_unlimited_customer_liability:
            # Low operational risk to Vendor
            score = 30.0
            base_level = "low"
        elif has_immediate_termination:
            score = 28.0
            base_level = "low"
        elif has_irrevocable_ip:
            score = 25.0
            base_level = "low"
        elif has_broad_indemnity:
            score = 25.0
            base_level = "low"
        elif has_unilateral_assignment:
            score = 25.0
            base_level = "low"
        else:
            # Normal vendor review
            score = score * 0.7

    else:
        # Neutral / balanced perspective
        # Penalize severe commercial asymmetry
        if (has_unlimited_customer_liability and has_nominal_vendor_cap) or has_immediate_termination:
            score = max(score, 88.0)
            base_level = "high"

    # Reviewer preference adjustments
    lower_pref = reviewer_preferences.lower()
    if lower_pref:
        if "liability" in lower_pref and clause_type == "limitation of liability":
            score = min(100.0, score + 8.0)
        if "termination" in lower_pref and clause_type == "termination":
            score = min(100.0, score + 8.0)
        if "ip" in lower_pref and clause_type == "ip assignment":
            score = min(100.0, score + 8.0)
        if "indemn" in lower_pref and clause_type == "indemnification":
            score = min(100.0, score + 8.0)

    # Confidence fine-tuning: small adjustment based on confidence
    conf_adj = (confidence - 0.5) * 6.0
    final_score = int(max(5, min(100, round(score + conf_adj))))

    # Level reconciliation with score
    if final_score >= 88:
        effective_level: RiskLevel = "critical"
    elif final_score >= 70:
        effective_level = "high"
    elif final_score >= 40:
        effective_level = "medium"
    else:
        effective_level = "low"

    explanation = _build_explanation(clause_type, effective_level, reviewing_party, trigger_displays)
    return final_score, effective_level, trigger_displays, explanation


def _build_explanation(clause_type: str, level: RiskLevel, party: ReviewingParty, triggers: list[str]) -> str:
    trigger_str = f" Triggered by: {'; '.join(triggers)}." if triggers else ""
    if party == "customer":
        if level == "critical":
            return f"Critical risk for Customer: Extreme commercial asymmetry or uncapped financial exposure detected.{trigger_str}"
        elif level == "high":
            return f"High risk for Customer: Disproportionate obligation or operational restriction requiring active negotiation.{trigger_str}"
        elif level == "medium":
            return f"Moderate risk for Customer: Standard terms with clauses that should be reviewed for mutuality and clear scope.{trigger_str}"
        return f"Low risk for Customer: Standard or benign clause structure.{trigger_str}"
    elif party == "vendor":
        if level in {"low", "medium"}:
            return f"Favorable or standard provision for Vendor: Language grants protective rights or limits Vendor exposure.{trigger_str}"
        return f"Review-worthy provision for Vendor: Verify commercial enforceability and customer acceptance.{trigger_str}"
    else:
        return f"Evaluated from Neutral perspective ({level.upper()} severity).{trigger_str}"


def category_scores(findings: list[RiskFinding]) -> dict[str, int]:
    """Compute average risk score per category."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for finding in findings:
        buckets[finding.category].append(finding.risk_score)
    return {category: min(100, round(sum(scores) / len(scores))) for category, scores in buckets.items()}


def score_to_level(score: int | float) -> RiskLevel:
    """Canonical conversion from numerical risk score (0-100) to RiskLevel.

    Thresholds:
    - Critical: 88 - 100 (Uncapped exposure, unilateral immediate termination)
    - High:     70 - 87  (Disproportionate commercial burden, broad indemnities)
    - Medium:   40 - 69  (Standard terms requiring negotiation/clarification)
    - Low:       0 - 39  (Benign or standard commercial terms)
    """
    s = round(float(score))
    if s >= 88:
        return "critical"
    if s >= 70:
        return "high"
    if s >= 40:
        return "medium"
    return "low"


def overall_score(findings: list[RiskFinding], missing: list[str]) -> int:
    """Documented aggregation strategy:

    Formula:
    1. Critical-Risk Aggregation Floor Override:
       If ANY finding is classified as Critical (risk_score >= 88 or risk_level == 'critical'),
       the contract's aggregate risk score is guaranteed to meet or exceed the Critical Floor (88).
       This ensures that a single fatal, catastrophic liability clause is never diluted away
       by 20 benign standard boilerplate clauses.
    2. High-Risk Floor:
       If ANY finding is High (risk_score >= 70 or risk_level == 'high'), the floor is guaranteed at least 65.
    3. Rank-decayed weighted score of findings:
       Scores are sorted descending; top finding has weight 1.0, 2nd has 0.7, 3rd has 0.5, etc.
    4. Missing clause penalty:
       Each missing critical clause adds 3 points (capped at 12).
    """
    if not findings and not missing:
        return 10
    if not findings:
        return min(40, len(missing) * 6)

    scores = sorted((f.risk_score for f in findings), reverse=True)
    max_score = scores[0]
    has_critical = any(f.risk_level == "critical" or f.risk_score >= 88 for f in findings)
    has_high = any(f.risk_level == "high" or f.risk_score >= 70 for f in findings)

    # Rank-decay weights: 1.0, 0.7, 0.5, 0.35, 0.25, 0.15...
    decay_weights = [1.0 / (1.0 + 0.45 * i) for i in range(len(scores))]
    weighted_sum = sum(s * w for s, w in zip(scores, decay_weights))
    sum_weights = sum(decay_weights)
    decayed_avg = weighted_sum / sum_weights

    # Floor based on maximum severity finding
    if has_critical:
        # Critical finding anchors overall severity to Critical (>= 88)
        floor = 88
    elif has_high:
        floor = 65
    elif max_score >= 50:
        floor = 45
    else:
        floor = 20

    missing_penalty = min(12, len(missing) * 3)
    aggregated = max(floor, round(decayed_avg + missing_penalty))
    return min(100, max(0, aggregated))
