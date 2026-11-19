from __future__ import annotations

FOCUSED_LABELS = [
    "termination",
    "assignment",
    "ip assignment",
    "indemnification",
    "limitation of liability",
    "confidentiality",
    "exclusivity",
    "non-compete / non-solicit",
    "governing law",
    "change of control",
    "most favored nation",
    "audit rights",
    "revenue / payment obligations",
]

# Canonical display names for user-facing UI
LABEL_DISPLAY_NAMES = {
    "termination": "Termination Rights",
    "assignment": "Assignment & Agreement Transfer",
    "ip assignment": "Intellectual Property Ownership",
    "indemnification": "Indemnification & Defense",
    "limitation of liability": "Limitation of Liability",
    "confidentiality": "Confidentiality & Information Protection",
    "exclusivity": "Exclusivity Restrictions",
    "non-compete / non-solicit": "Non-Compete & Non-Solicitation",
    "governing law": "Governing Law & Jurisdiction",
    "change of control": "Change of Control",
    "most favored nation": "Most Favored Nation (MFN)",
    "audit rights": "Audit & Inspection Rights",
    "revenue / payment obligations": "Payment & Revenue Obligations",
}

MISSING_CLAUSE_IMPORTANCE = {
    "termination": "Contracts should define clear, mutual termination rights and reasonable notice/cure periods.",
    "limitation of liability": "Uncapped liability or missing liability caps can create fatal financial exposure.",
    "confidentiality": "Proprietary information and trade secrets need explicit protection and handling rules.",
    "governing law": "Missing governing law and venue creates forum uncertainty and heightened litigation expense.",
    "indemnification": "Indemnity scope should be balanced, fault-based, and limited to third-party claims.",
}
