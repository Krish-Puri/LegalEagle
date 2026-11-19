from __future__ import annotations

import re
from pathlib import Path

from src.models.labels import FOCUSED_LABELS
from src.schema import ClauseClassification, ContractChunk, RiskLevel


# Distinct keyword and pattern dictionaries for each clause type
CLAUSE_PATTERNS: dict[str, list[str]] = {
    "ip assignment": [
        r"\bintellectual\s+property\b",
        r"\bip\s+(?:assignment|rights|ownership)\b",
        r"\bwork(?:\s+for\s+hire|\s+product)\b",
        r"\bdeliverables?\b",
        r"\binventions?\b",
        r"\bpatents?\b",
        r"\bcopyrights?\b",
        r"\btrade\s+secrets?\b",
        r"\bproprietary\s+rights?\b",
        r"\bmoral\s+rights?\b",
        r"\birrevocably\s+assigns?\b",
        r"\bhereby\s+assigns?\b",
        r"\bcustom\s+code\b",
        r"\bderivative\s+works?\b",
        r"\ball\s+right[,\s]+title[,\s]+and\s+interest\b",
        r"\bownership\s+of\b",
    ],
    "assignment": [
        r"\bassign(?:ment|s|ed)?\s+of\s+this\s+agreement\b",
        r"\bassign\s+(?:or\s+transfer\s+)?this\s+agreement\b",
        r"\btransfer\s+(?:or\s+delegate\s+)?this\s+agreement\b",
        r"\bneither\s+party\s+may\s+assign\b",
        r"\bmay\s+not\s+assign\b",
        r"\bsuccessors?\s+(?:and|or)\s+(?:permitted\s+)?assigns\b",
        r"\bdelegat(?:e|ion)\s+of\s+duties\b",
        r"\btransfer\s+of\s+rights\s+under\s+this\s+agreement\b",
        r"\bassign\s+to\s+any\s+(?:successor|affiliate)\b",
        r"\bprior\s+written\s+consent\b",
    ],
    "limitation of liability": [
        r"\blimitation\s+of\s+liability\b",
        r"\baggregate\s+(?:total\s+)?liability\b",
        r"\bcumulative\s+liability\b",
        r"\bliability\s+cap\b",
        r"\bunlimited\s+liability\b",
        r"\bconsequential\s+damages\b",
        r"\bindirect[,\s]+special[,\s]+(?:punitive\s+)?or\s+consequential\b",
        r"\bloss\s+of\s+profits\b",
        r"\bexceed\s+(?:the\s+amount\s+of\s+)?(?:\$[\d,]+|\w+\s+dollars)\b",
        r"\bsole\s+and\s+exclusive\s+remedy\b",
        r"\bin\s+no\s+event\s+shall.*liab(?:le|ility)\b",
    ],
    "indemnification": [
        r"\bindemnif(?:y|ies|ied|ication)\b",
        r"\bdefend[,\s]+(?:and\s+)?hold\s+harmless\b",
        r"\bhold\s+harmless\b",
        r"\bindemnity\s+obligations?\b",
        r"\bdefend\s+against\s+(?:any\s+and\s+all\s+)?claims\b",
        r"\bthird[- ]party\s+claims?\b",
    ],
    "termination": [
        r"\bterminate\s+this\s+agreement\b",
        r"\btermination\s+for\s+convenience\b",
        r"\btermination\s+for\s+(?:material\s+)?breach\b",
        r"\bterminat\w*\s+(?:immediately\s+)?without\s+(?:prior\s+)?notice\b",
        r"\bright\s+to\s+terminate\b",
        r"\bmay\s+terminate\b",
        r"\btermination\s+rights?\b",
        r"\bcure\s+period\b",
        r"\bnotice\s+of\s+termination\b",
        r"\bupon\s+\d+\s+days\s+prior\s+written\s+notice\b",
    ],
    "confidentiality": [
        r"\bconfidential\s+information\b",
        r"\bnon[- ]disclosure\b",
        r"\bmaintain\s+in\s+strict\s+confidence\b",
        r"\bproprietary\s+information\b",
        r"\bneed\s+to\s+know\b",
        r"\bconfidentiality\s+obligations?\b",
        r"\bnon[- ]public\s+information\b",
        r"\breceiving\s+party\s+shall\s+protect\b",
    ],
    "non-compete / non-solicit": [
        r"\bnon[- ]compete\b",
        r"\bnon[- ]competition\b",
        r"\bnon[- ]solicit(?:ation)?\b",
        r"\bcompeting\s+(?:software|product|business|services?|solutions?)\b",
        r"\bsolicit\s+(?:or\s+hire|for\s+employment)\b",
        r"\bengage\s+in\s+(?:any\s+)?competing\b",
        r"\brestrict\w*\s+from\s+engaging\b",
        r"\bshall\s+not\s+(?:directly\s+or\s+indirectly\s+)?develop\b",
    ],
    "governing law": [
        r"\bgoverning\s+law\b",
        r"\blaws\s+of\s+the\s+state\s+of\b",
        r"\bexclusive\s+jurisdiction\b",
        r"\bvenue\s+of\s+(?:the\s+)?courts?\b",
        r"\bstate\s+and\s+federal\s+courts\b",
        r"\bdispute\s+resolution\b",
        r"\barbitration\b",
    ],
    "change of control": [
        r"\bchange\s+of\s+control\b",
        r"\bmerger[,\s]+acquisition[,\s]+or\s+sale\b",
        r"\bsale\s+of\s+(?:all\s+or\s+substantially\s+all\s+)?assets\b",
        r"\bchange\s+in\s+ownership\b",
        r"\bupon\s+any\s+change\s+of\s+control\b",
    ],
    "most favored nation": [
        r"\bmost\s+favored\s+(?:nation|customer)\b",
        r"\bmfn\b",
        r"\bbest\s+price\b",
        r"\bmore\s+favorable\s+terms\b",
    ],
    "audit rights": [
        r"\baudit\s+access\b",
        r"\bunrestricted\s+audit\b",
        r"\binspect\s+(?:books\s+and\s+)?records\b",
        r"\bexamination\s+of\s+records\b",
        r"\baccess\s+to\s+(?:facilities|records)\b",
        r"\bverify\s+compliance\b",
        r"\baudit\s+rights?\b",
    ],
    "revenue / payment obligations": [
        r"\bpayment\s+obligations?\b",
        r"\binvoices?\b",
        r"\bfees\s+payable\b",
        r"\bnet\s+\d+\s+days\b",
        r"\blate\s+fees?\b",
        r"\bundisputed\s+invoices?\b",
        r"\bcurrency\b",
        r"\btaxes?\b",
    ],
    "exclusivity": [
        r"\bexclusive\s+(?:provider|partner|supplier|distributor|license)\b",
        r"\bexclusivity\b",
        r"\bsole\s+provider\b",
        r"\bshall\s+not\s+(?:purchase|license|procure)\s+from\s+any\s+other\b",
    ],
}

# Strong signals in clause titles / headings
HEADING_HINTS: dict[str, list[str]] = {
    "ip assignment": ["intellectual property", "ip", "work product", "inventions", "ownership of deliverables"],
    "assignment": ["assignment", "transfer", "successors and assigns"],
    "limitation of liability": ["limitation of liability", "liability", "damages", "limitation of damages"],
    "indemnification": ["indemnification", "indemnity", "hold harmless", "defense"],
    "termination": ["termination", "term and termination", "cancellation"],
    "confidentiality": ["confidentiality", "confidential information", "non-disclosure", "ndas"],
    "non-compete / non-solicit": ["non-compete", "non-competition", "non-solicitation", "restrictive covenants"],
    "governing law": ["governing law", "jurisdiction", "venue", "dispute resolution"],
    "change of control": ["change of control"],
    "most favored nation": ["most favored nation", "mfn"],
    "audit rights": ["audit", "inspection", "books and records"],
    "revenue / payment obligations": ["payment", "fees", "invoicing", "taxes", "compensation"],
    "exclusivity": ["exclusivity", "exclusive provider"],
}


class KeywordRiskClassifier:
    """Explainable, pattern-driven and context-aware clause classifier.

    Features:
    - Contextual disambiguation (IP ownership vs Agreement assignment, Liability vs Indemnity)
    - Heading and title aware scoring
    - Multi-factor confidence calibration
    - Optional fine-tuned Transformer pipeline when available
    """

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path
        self._pipeline = self._load_fine_tuned_pipeline(model_path)

    def classify(self, chunks: list[ContractChunk]) -> list[ClauseClassification]:
        if self._pipeline:
            return self._classify_with_pipeline(chunks)

        classifications: list[ClauseClassification] = []
        for chunk in chunks:
            classification = self._classify_chunk(chunk)
            if classification:
                classifications.append(classification)
        return classifications

    def _classify_chunk(self, chunk: ContractChunk) -> ClauseClassification | None:
        text = chunk.text
        title = chunk.clause_title or ""
        lower_text = text.lower()
        lower_title = title.lower()

        scores: dict[str, float] = {label: 0.0 for label in FOCUSED_LABELS}

        # 1. Match regex patterns
        for label, patterns in CLAUSE_PATTERNS.items():
            for pat in patterns:
                matches = len(re.findall(pat, lower_text))
                if matches > 0:
                    scores[label] += matches * 2.0

        # 2. Match headings/titles with significant weight
        for label, hints in HEADING_HINTS.items():
            for hint in hints:
                if hint in lower_title:
                    scores[label] += 6.0

        # 3. Contextual disambiguation rules

        # DISAMBIGUATION A: IP Assignment vs Contractual Assignment
        has_ip_terms = any(
            re.search(pat, lower_text)
            for pat in [
                r"\bintellectual\s+property\b",
                r"\bwork\s+product\b",
                r"\bdeliverables?\b",
                r"\binventions?\b",
                r"\bpatents?\b",
                r"\bcopyrights?\b",
                r"\bproprietary\s+rights?\b",
                r"\bcustom\s+code\b",
                r"\bderivative\s+works?\b",
            ]
        ) or "intellectual property" in lower_title or "ip" in lower_title.split()

        has_contract_transfer_terms = any(
            re.search(pat, lower_text)
            for pat in [
                r"\bassign(?:ment)?\s+(?:or\s+transfer\s+)?(?:of\s+)?this\s+agreement\b",
                r"\btransfer\s+this\s+agreement\b",
                r"\bneither\s+party\s+may\s+assign\b",
                r"\bassign\s+this\s+agreement\s+without\b",
                r"\bassign\s+to\s+any\s+successor\b",
                r"\bpermitted\s+assigns\b",
            ]
        ) or (lower_title.strip() in {"assignment", "7. assignment", "assignment and transfer"})

        if has_ip_terms:
            if not has_contract_transfer_terms:
                scores["ip assignment"] += 8.0
                scores["assignment"] = 0.0
            else:
                # If clause has both, evaluate title and primary subject
                if "intellectual" in lower_title or "ip" in lower_title:
                    scores["ip assignment"] += 10.0
                    scores["assignment"] = max(0.0, scores["assignment"] - 5.0)
                elif "assign" in lower_title and "intellectual" not in lower_title:
                    scores["assignment"] += 8.0

        # DISAMBIGUATION B: Limitation of liability vs Indemnification
        # A liability limitation clause that mentions indemnification as an exception
        # is still a limitation of liability clause
        if "limitation of liability" in lower_title or "liability" in lower_title:
            scores["limitation of liability"] += 6.0
        elif "indemnif" in lower_title or "hold harmless" in lower_title:
            scores["indemnification"] += 6.0

        if re.search(r"\baggregate\s+liability\b|\bliability\s+cap\b|\bexceed\s+\$", lower_text):
            scores["limitation of liability"] += 5.0

        # DISAMBIGUATION C: Termination vs Renewal
        if "termination" in lower_title or "terminate" in lower_title:
            scores["termination"] += 6.0

        # DISAMBIGUATION D: Confidentiality
        if "confidential" in lower_title or "non-disclosure" in lower_title:
            scores["confidentiality"] += 8.0

        # DISAMBIGUATION E: Non-compete / Non-solicit vs Termination
        # Clauses with "following termination" or "post-termination" describing non-compete
        # are Non-Compete clauses, not Termination clauses.
        has_compete_terms = bool(
            re.search(r"\bnon[- ]compete\b|\bnon[- ]solicit\b|\bcompeting\s+(?:software|product|business)\b|\bsolicit\s+(?:or\s+hire|for\s+employment)\b", lower_text)
        )
        if has_compete_terms:
            scores["non-compete / non-solicit"] += 10.0
            if "following termination" in lower_text or "upon termination" in lower_text:
                scores["termination"] = max(0.0, scores["termination"] - 6.0)

        # DISAMBIGUATION F: Audit Rights
        if re.search(r"\baudit\b|\bexamination\s+of\s+records\b|\binspect\s+(?:books\s+and\s+)?records\b", lower_text):
            scores["audit rights"] += 8.0

        # DISAMBIGUATION G: Change of Control
        if re.search(r"\bchange\s+of\s+control\b", lower_text):
            scores["change of control"] += 10.0

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_label, top_score = sorted_scores[0]
        second_label, second_score = sorted_scores[1]

        if top_score < 2.0:
            return None

        # Confidence calculation
        margin = (top_score - second_score) / (top_score + 1e-6)
        base_conf = min(0.98, 0.50 + (top_score / 20.0) * 0.35 + margin * 0.15)
        confidence = round(max(0.40, min(0.99, base_conf)), 2)

        requires_human_review = margin < 0.25 or confidence < 0.70

        # Formulate rationale
        matched_indicators = self._get_matching_indicators(top_label, lower_text, lower_title)
        rationale = f"Classified as '{top_label}' based on: {', '.join(matched_indicators[:4])}."
        if requires_human_review:
            rationale += f" [Note: Close secondary match '{second_label}' (score {second_score:.1f}) requires reviewer verification]."

        risk_level, initial_factors = self._heuristic_risk(top_label, lower_text)

        return ClauseClassification(
            chunk_id=chunk.id,
            clause_type=top_label,
            confidence=confidence,
            risk_level=risk_level,
            category=self._category_for_label(top_label),
            rationale=rationale,
            risk_factors=initial_factors,
            review_priority=risk_level,
            requires_human_review=requires_human_review,
        )

    def _get_matching_indicators(self, label: str, lower_text: str, lower_title: str) -> list[str]:
        indicators = []
        if label in HEADING_HINTS:
            for hint in HEADING_HINTS[label]:
                if hint in lower_title:
                    indicators.append(f"heading '{hint}'")
                    break
        if label in CLAUSE_PATTERNS:
            for pat in CLAUSE_PATTERNS[label]:
                found = re.findall(pat, lower_text)
                if found:
                    indicators.append(f"phrase '{found[0]}'")
                    if len(indicators) >= 4:
                        break
        return indicators or ["contextual semantic signals"]

    def _heuristic_risk(self, label: str, lower: str) -> tuple[RiskLevel, list[str]]:
        factors: list[str] = []

        # Critical severity indicators
        if "unlimited liability" in lower:
            factors.append("unlimited_liability")
        if re.search(r"exceed\s+(?:fifty\s+dollars|\$50|\$100|one\s+hundred\s+dollars)", lower):
            factors.append("nominal_liability_cap")
        if "immediately without notice" in lower or "terminate immediately" in lower or "sole discretion" in lower:
            factors.append("unilateral_immediate_termination")
        if "any and all losses" in lower or "all losses" in lower:
            factors.append("broad_indemnity_all_losses")
        if re.search(r"(\d+)\s+years", lower):
            match = re.search(r"(\d+)\s+years", lower)
            if match and int(match.group(1)) >= 3:
                factors.append(f"extended_restriction_{match.group(1)}_years")
        if "irrevocably assigns" in lower or "all right, title, and interest" in lower:
            factors.append("full_work_product_assignment")
        if "freely assign" in lower or "without consent or notice" in lower:
            factors.append("unilateral_assignment_rights")

        if any(f in factors for f in ["unlimited_liability", "nominal_liability_cap", "unilateral_immediate_termination"]):
            return "critical", factors
        if factors or any(term in lower for term in ["sole discretion", "hold harmless", "consequential damages", "non-compete"]):
            return "high", factors
        if any(term in lower for term in ["30 days", "material breach", "reasonable efforts", "shall", "must"]):
            return "medium", factors
        return "low", factors

    def _category_for_label(self, label: str) -> str:
        mapping = {
            "ip assignment": "ip_ownership",
            "assignment": "operational_control",
            "limitation of liability": "liability_exposure",
            "indemnification": "liability_exposure",
            "termination": "termination_risk",
            "confidentiality": "information_protection",
            "non-compete / non-solicit": "commercial_restrictions",
            "governing law": "dispute_resolution",
            "change of control": "operational_control",
            "most favored nation": "commercial_restrictions",
            "audit rights": "compliance_audit",
            "revenue / payment obligations": "payment_terms",
            "exclusivity": "commercial_restrictions",
        }
        return mapping.get(label, "general")

    def _load_fine_tuned_pipeline(self, model_path: Path | None):
        if not model_path or not model_path.exists() or not any(model_path.iterdir()):
            return None
        try:
            from transformers import pipeline

            return pipeline("text-classification", model=str(model_path), tokenizer=str(model_path), top_k=None)
        except Exception:
            return None

    def _classify_with_pipeline(self, chunks: list[ContractChunk]) -> list[ClauseClassification]:
        classifications: list[ClauseClassification] = []
        for chunk in chunks:
            fallback = self._classify_chunk(chunk)
            try:
                result = self._pipeline(_truncate_for_transformer(chunk.text))[0]
                best = max(result, key=lambda item: item["score"]) if isinstance(result, list) else result
                label = str(best["label"]).lower().replace("_", " ")
                transformer_conf = float(best["score"])

                if label in FOCUSED_LABELS and transformer_conf > 0.60:
                    if fallback:
                        # Combine models: transformer label if high confidence, enriched by rule-based factors
                        fallback.clause_type = label
                        fallback.confidence = round(transformer_conf, 2)
                        fallback.rationale = f"Transformer classification '{label}' (confidence: {transformer_conf:.2f})."
                    classifications.append(fallback)
                elif fallback:
                    classifications.append(fallback)
            except Exception:
                if fallback:
                    classifications.append(fallback)
        return classifications


def _truncate_for_transformer(text: str, max_chars: int = 2200) -> str:
    return text if len(text) <= max_chars else text[:max_chars]
