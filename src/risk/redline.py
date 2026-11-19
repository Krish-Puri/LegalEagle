from __future__ import annotations

from src.schema import RedlineSuggestion


# Category-specific targeted redlines and objectives
BALANCED_TEMPLATES: dict[str, dict[str, str]] = {
    "termination": {
        "objective": "Establish bilateral termination rights with mandatory written notice and a reasonable 30-day cure period.",
        "proposed": "Either party may terminate this Agreement upon thirty (30) days prior written notice if the other party materially breaches this Agreement and fails to cure such breach within the thirty (30) day notice period.",
        "reason": "Eliminates unilateral immediate termination without notice, ensuring operational predictability and reasonable cure opportunities.",
    },
    "limitation of liability": {
        "objective": "Establish a balanced, mutual aggregate liability cap equal to fees paid in 12 months and mutual waiver of consequential damages.",
        "proposed": "Except for indemnity obligations or breach of confidentiality, each party's aggregate cumulative liability under this Agreement shall not exceed the total fees paid or payable by Customer in the twelve (12) months preceding the claim. Neither party shall be liable for indirect, incidental, or consequential damages.",
        "reason": "Replaces the one-sided, nominal $50 liability cap and uncapped customer liability with a commercially standard 12-month mutual fee cap.",
    },
    "indemnification": {
        "objective": "Narrow indemnity obligations strictly to third-party claims arising from gross negligence or material breach, conditioned on prompt notice.",
        "proposed": "Each party shall defend, indemnify, and hold harmless the other party against third-party claims, damages, and reasonable legal expenses to the extent arising directly from the indemnifying party's material breach of this Agreement, gross negligence, or willful misconduct, conditioned upon receiving prompt written notice and control of defense.",
        "reason": "Eliminates broad, open-ended liability for 'any and all losses' and ensures indemnification applies solely to proven third-party claims.",
    },
    "ip assignment": {
        "objective": "Preserve Customer ownership of custom deliverables and paid work product, while protecting pre-existing IP.",
        "proposed": "Customer shall own all right, title, and interest in and to all custom deliverables, code, and work product developed specifically for Customer under this Agreement upon payment. Vendor retains all rights in its pre-existing background tools and grants Customer a perpetual, non-exclusive license to use such background tools with the deliverables.",
        "reason": "Ensures the Customer owns what it pays for, rather than forfeiting custom code and modifications to the Vendor.",
    },
    "assignment": {
        "objective": "Ensure bilateral assignment consent rules with standard exceptions for M&A, affiliates, and corporate reorganizations.",
        "proposed": "Neither party may assign or transfer this Agreement without the prior written consent of the other party (such consent not to be unreasonably withheld or delayed), except that either party may assign this Agreement without consent to an affiliate or in connection with a merger, acquisition, or sale of substantially all assets.",
        "reason": "Removes asymmetric assignment restrictions and allows customary business transfers during corporate restructuring.",
    },
    "confidentiality": {
        "objective": "Ensure mutual confidentiality obligations, standard exceptions, and reasonable survival duration.",
        "proposed": "Each party agrees to maintain in confidence all proprietary information received from the other party using at least reasonable care, disclosing it only to representatives with a need to know. Confidentiality obligations shall survive for a period of three (3) years following agreement termination.",
        "reason": "Standardizes confidentiality as mutual, adds reasonable care standards, and bounds survival to a standard 3-year term.",
    },
    "non-compete / non-solicit": {
        "objective": "Eliminate overbroad commercial non-competes and confine restrictions to reasonable, 12-month non-solicitation of active personnel.",
        "proposed": "During the term and for a period of twelve (12) months following termination, neither party shall directly solicit for employment any employee of the other party who was directly involved in the provision or receipt of services hereunder. This restriction shall not apply to general employment advertisements.",
        "reason": "Strikes down anti-competitive multi-year software restrictions and limits scope to standard, enforceable 12-month non-solicitation.",
    },
    "governing law": {
        "objective": "Select a predictable, commercially established jurisdiction with mutual consent to venue.",
        "proposed": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to conflicts of law principles. The parties consent to jurisdiction in the state and federal courts located in Delaware.",
        "reason": "Confirms established, predictable commercial jurisdiction and venue.",
    },
    "revenue / payment obligations": {
        "objective": "Provide clear invoicing terms (Net 30) with dispute escalation mechanisms before any suspension of services.",
        "proposed": "Undisputed invoices are payable within thirty (30) days of receipt. In the event of a good-faith payment dispute, Customer shall provide prompt notice, and the parties shall negotiate in good faith without Vendor suspending services.",
        "reason": "Prevents immediate service shutdowns over disputed charges and guarantees reasonable Net 30 payment periods.",
    },
    "exclusivity": {
        "objective": "Narrow exclusivity to specified products or territories, conditioned on minimum volume quotas.",
        "proposed": "Any appointment under this Agreement shall be non-exclusive, and each party remains free to develop, license, and procure similar products and services from third parties.",
        "reason": "Preserves competitive freedom and eliminates restrictive commercial lock-in.",
    },
    "audit rights": {
        "objective": "Limit audits to once annually during regular hours with advance notice.",
        "proposed": "Upon at least fifteen (15) business days prior written notice, Customer may inspect relevant records once per calendar year during regular business hours to verify compliance, subject to customary confidentiality protections.",
        "reason": "Protects against disruptive or arbitrary inspections.",
    },
    "change of control": {
        "objective": "Clarify that normal M&A or financing does not trigger automatic termination.",
        "proposed": "A change of control or corporate reorganization shall not constitute an assignment or breach of this Agreement, provided the surviving entity assumes all obligations hereunder.",
        "reason": "Prevents disruptive contract terminations during financing or acquisition events.",
    },
}

DEFAULT_TEMPLATE = {
    "objective": "Replace aggressive or one-sided terms with balanced, mutual language limited in duration, scope, and liability.",
    "proposed": "The parties agree to execute balanced, mutual terms consistent with standard commercial practices, providing reasonable notice, fair liability limits, and proportionate remedies.",
    "reason": "Balances commercial risk exposure between both contracting parties.",
}


def generate_redline(clause_type: str, original_language: str) -> RedlineSuggestion:
    """Generate context-appropriate negotiation redline and objective."""
    template = BALANCED_TEMPLATES.get(clause_type, DEFAULT_TEMPLATE)
    original = _shorten(original_language)
    proposed = template["proposed"]
    objective = template["objective"]
    reason = template["reason"]

    # Construct clean side-by-side markdown comparison
    redline_markdown = (
        f"**Original Language (Problematic):**\n"
        f"~~{original}~~\n\n"
        f"**Proposed Balanced Language (Negotiation Target):**\n"
        f"*{proposed}*"
    )

    diff_preview = f"- {original}\n+ {proposed}"

    return RedlineSuggestion(
        original_language=original,
        proposed_language=proposed,
        redline_markdown=redline_markdown,
        negotiation_reason=reason,
        negotiation_objective=objective,
        reviewer_warning="Draft negotiation language for attorney review only. Not legal advice.",
        diff_preview=diff_preview,
    )


def _shorten(text: str, max_chars: int = 500) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= max_chars else cleaned[: max_chars - 3] + "..."
