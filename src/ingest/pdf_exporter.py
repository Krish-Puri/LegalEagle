from __future__ import annotations

import io
from datetime import datetime

from src.schema import RiskScorecard


def generate_pdf_report(scorecard: RiskScorecard, contract_name: str) -> bytes:
    """Generate a clean, professional executive PDF risk report using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        return b""

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    margin = 40
    y = 50

    # Header
    page.insert_text((margin, y), "LegalEagle Contract Risk Scorecard", fontsize=18, fontname="helv", color=(0.85, 0.29, 0.38))
    y += 20
    page.insert_text(
        (margin, y),
        f"Contract: {contract_name} | Reviewing Perspective: {scorecard.reviewing_party.upper()} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=9,
        fontname="helv",
        color=(0.4, 0.4, 0.4),
    )
    y += 25

    # Executive Summary Box
    rect = fitz.Rect(margin, y, 595 - margin, y + 60)
    page.draw_rect(rect, color=(0.2, 0.25, 0.3), fill=(0.95, 0.96, 0.98))
    page.insert_text((margin + 15, y + 22), f"Overall Risk Score: {scorecard.overall_risk_score}/100", fontsize=13, fontname="helv", color=(0.1, 0.1, 0.1))
    page.insert_text((margin + 200, y + 22), f"Risk Level: {scorecard.risk_level.upper()}", fontsize=13, fontname="helv", color=(0.85, 0.29, 0.38) if scorecard.risk_level in {"high", "critical"} else (0.2, 0.6, 0.4))
    page.insert_text((margin + 360, y + 22), f"High/Critical Findings: {scorecard.high_critical_count}/{scorecard.total_findings_count}", fontsize=11, fontname="helv", color=(0.2, 0.2, 0.2))

    summary_text = scorecard.contract_summary[:160] + "..." if len(scorecard.contract_summary) > 160 else scorecard.contract_summary
    page.insert_text((margin + 15, y + 46), summary_text, fontsize=8.5, fontname="helv", color=(0.3, 0.3, 0.3))
    y += 80

    # Findings Section
    page.insert_text((margin, y), "High-Priority Contract Findings & Recommendations", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.1))
    y += 18

    for idx, finding in enumerate(scorecard.findings[:5], start=1):
        if y > 730:
            page = doc.new_page(width=595, height=842)
            y = 50

        # Finding title bar
        page.insert_text((margin, y), f"{idx}. {finding.clause_type.upper()} ({finding.risk_level.upper()} - {finding.risk_score}/100)", fontsize=10, fontname="helv", color=(0.85, 0.29, 0.38) if finding.risk_level in {"critical", "high"} else (0.2, 0.2, 0.2))
        y += 14

        # Quote
        quote_text = f"Quote: \"{finding.quote[:180]}...\"" if len(finding.quote) > 180 else f"Quote: \"{finding.quote}\""
        page.insert_text((margin + 10, y), quote_text, fontsize=8, fontname="helv", color=(0.25, 0.25, 0.25))
        y += 13

        # Impact & Note
        page.insert_text((margin + 10, y), f"Impact: {finding.business_impact[:110]}", fontsize=8, fontname="helv", color=(0.35, 0.35, 0.35))
        y += 12
        page.insert_text((margin + 10, y), f"Negotiation: {finding.suggested_negotiation_note[:110]}", fontsize=8, fontname="helv", color=(0.15, 0.45, 0.3))
        y += 12

        # Redline recommendation
        if finding.redline:
            page.insert_text((margin + 10, y), f"Proposed: {finding.redline.proposed_language[:110]}...", fontsize=8, fontname="helv", color=(0.1, 0.3, 0.6))
            y += 16
        else:
            y += 8

    # Disclaimer Footer
    y = 800
    page.insert_text((margin, y), "Educational Research Prototype • Antigravity AI Engineering • Does not constitute formal legal advice.", fontsize=7.5, fontname="helv", color=(0.6, 0.6, 0.6))

    buffer = io.BytesIO()
    doc.save(buffer)
    doc.close()
    return buffer.getvalue()

