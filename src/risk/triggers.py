from __future__ import annotations

import re


# Specific trigger patterns with severity and perspective annotations
TRIGGER_PATTERNS: list[dict[str, str | list[str]]] = [
    {
        "id": "unlimited_customer_liability",
        "phrase": "customer's liability.*unlimited",
        "display": "Customer liability is unlimited",
        "severity": "critical",
        "disfavored_party": "customer",
    },
    {
        "id": "nominal_vendor_cap",
        "phrase": r"exceed\s+(?:fifty\s+dollars|\$50|\$100|one\s+hundred\s+dollars)",
        "display": "Nominal Vendor liability cap ($50/$100)",
        "severity": "critical",
        "disfavored_party": "customer",
    },
    {
        "id": "unilateral_immediate_termination",
        "phrase": r"terminate\s+(?:this\s+agreement\s+)?immediately\s+without\s+notice",
        "display": "Immediate termination without notice",
        "severity": "critical",
        "disfavored_party": "customer",
    },
    {
        "id": "sole_discretion_termination",
        "phrase": r"in\s+its\s+sole\s+discretion.*(?:not\s+commercially\s+desirable|terminate)",
        "display": "Termination in sole discretion without cause",
        "severity": "high",
        "disfavored_party": "customer",
    },
    {
        "id": "asymmetric_termination_notice",
        "phrase": r"(?:180|90|120)\s+days\s+prior\s+written\s+notice",
        "display": "Disproportionate notice period for termination",
        "severity": "high",
        "disfavored_party": "customer",
    },
    {
        "id": "broad_indemnity_all_losses",
        "phrase": r"(?:any\s+and\s+all|all)\s+losses[,\s]+damages[,\s]+liabilities[,\s]+costs",
        "display": "Unbounded indemnity for any and all losses",
        "severity": "high",
        "disfavored_party": "customer",
    },
    {
        "id": "indemnity_includes_attorney_fees",
        "phrase": r"(?:full\s+)?attorney(?:s'|'s|\s+)?\s+fees",
        "display": "Indemnity includes attorney fees",
        "severity": "medium",
        "disfavored_party": "customer",
    },
    {
        "id": "irrevocable_ip_assignment",
        "phrase": r"irrevocably\s+assigns?.*(?:all\s+ownership|patent|copyright|work\s+product)",
        "display": "Irrevocable full IP & work product assignment",
        "severity": "high",
        "disfavored_party": "customer",
    },
    {
        "id": "vendor_retains_all_deliverables",
        "phrase": r"vendor\s+retains\s+all\s+right[,\s]+title[,\s]+and\s+interest",
        "display": "Vendor retains all deliverables and custom code",
        "severity": "high",
        "disfavored_party": "customer",
    },
    {
        "id": "multi_year_noncompete",
        "phrase": r"(?:[3-9]|\d{2})\s+years.*(?:non[- ]compete|competing|solicit)",
        "display": "Extended post-termination restriction (3+ years)",
        "severity": "high",
        "disfavored_party": "customer",
    },
    {
        "id": "unilateral_assignment",
        "phrase": r"freely\s+assign.*without\s+consent\s+or\s+notice",
        "display": "Unilateral transfer rights without consent or notice",
        "severity": "high",
        "disfavored_party": "customer",
    },
    {
        "id": "customer_assignment_prohibited",
        "phrase": r"customer\s+may\s+not\s+assign.*without.*written\s+consent",
        "display": "Asymmetric assignment restriction on Customer",
        "severity": "medium",
        "disfavored_party": "customer",
    },
    {
        "id": "waiver_consequential_damages",
        "phrase": r"(?:no|not\s+be)\s+liable\s+for\s+any\s+indirect[,\s]+(?:special[,\s]+)?(?:punitive[,\s]+)?or\s+consequential",
        "display": "Exclusion of indirect and consequential damages",
        "severity": "medium",
        "disfavored_party": "neutral",
    },
    {
        "id": "exclusive_jurisdiction",
        "phrase": r"consent\s+to\s+exclusive\s+jurisdiction",
        "display": "Exclusive foreign/specified jurisdiction venue",
        "severity": "medium",
        "disfavored_party": "neutral",
    },
    {
        "id": "indefinite_confidentiality_survival",
        "phrase": r"survive\s+(?:indefinitely|in\s+perpetuity)",
        "display": "Indefinite survival of confidentiality obligations",
        "severity": "medium",
        "disfavored_party": "neutral",
    },
    {
        "id": "late_fee_suspension",
        "phrase": r"(?:late\s+fee|suspend\s+services?\s+immediately)",
        "display": "Late fees and immediate service suspension rights",
        "severity": "medium",
        "disfavored_party": "customer",
    },
]


def triggering_language(text: str) -> list[str]:
    """Return human-readable descriptions of all triggered risk phrases in text."""
    lower = text.lower()
    matches = []
    for item in TRIGGER_PATTERNS:
        pattern = str(item["phrase"])
        if re.search(pattern, lower):
            matches.append(str(item["display"]))
    return matches


def detailed_risk_triggers(text: str) -> list[dict[str, str]]:
    """Return detailed metadata for all triggered risk factors."""
    lower = text.lower()
    matches = []
    for item in TRIGGER_PATTERNS:
        pattern = str(item["phrase"])
        if re.search(pattern, lower):
            matches.append({
                "id": str(item["id"]),
                "display": str(item["display"]),
                "severity": str(item["severity"]),
                "disfavored_party": str(item["disfavored_party"]),
            })
    return matches
