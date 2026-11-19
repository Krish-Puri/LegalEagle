from __future__ import annotations

from collections import Counter

from pydantic import ValidationError

from src.agent.prompts import SYSTEM_PROMPT, build_review_prompt
from src.config import AppConfig
from src.memory.store import MemoryStore
from src.models.labels import FOCUSED_LABELS, MISSING_CLAUSE_IMPORTANCE
from src.models.risk_classifier import KeywordRiskClassifier
from src.observability.tracing import TraceRecorder, traceable_step
from src.retrieval.vector_store import LocalRetriever
from src.risk.redline import generate_redline
from src.risk.scoring import (
    category_for_clause,
    category_scores,
    evaluate_clause_risk,
    overall_score,
    score_to_level,
)
from src.risk.triggers import detailed_risk_triggers, triggering_language
from src.schema import (
    ClauseClassification,
    MarketStandard,
    ParsedContract,
    RetrievalEvidence,
    ReviewingParty,
    RiskFinding,
    RiskScorecard,
    risk_weight,
)


# Category-specific negotiation notes tailored to party perspective and risk tier
CATEGORY_NEGOTIATION_NOTES: dict[str, dict[str, str]] = {
    "termination": {
        "critical": "Priority 1 Redline: Strike unilateral immediate termination without notice; demand a mutual 30-day written notice with a mandatory cure period for remediable breach.",
        "high": "High-Priority Redline: Equalize termination notice periods and remove sole-discretion cancellation rights.",
        "medium": "Negotiation Note: Ensure termination rights are bilateral and specify clear post-termination transition obligations.",
        "low": "Acceptable as drafted: Mutual termination rights align with commercial standards.",
    },
    "limitation of liability": {
        "critical": "Priority 1 Redline: Eliminate uncapped customer exposure and strike the nominal vendor cap ($50); replace with a mutual aggregate liability cap equal to 12 months of fees.",
        "high": "High-Priority Redline: Establish reciprocal liability ceilings and ensure carveouts are strictly limited to confidentiality, gross negligence, and IP infringement.",
        "medium": "Negotiation Note: Verify that exclusions for indirect and consequential damages apply reciprocally to both parties.",
        "low": "Acceptable as drafted: Liability caps and damages exclusions appear commercially balanced.",
    },
    "indemnification": {
        "critical": "Priority 1 Redline: Reject one-sided indemnity for 'any and all losses'; restrict indemnity strictly to direct third-party claims arising from proven material breach or gross misconduct.",
        "high": "High-Priority Redline: Condition indemnification on prompt written notice, right to control defense, and no unapproved settlements.",
        "medium": "Negotiation Note: Ensure indemnity obligations are reciprocal or fault-based with reasonable defense cooperation.",
        "low": "Acceptable as drafted: Standard third-party IP indemnity with customary defense procedure.",
    },
    "ip assignment": {
        "critical": "Priority 1 Redline: Strike blanket assignment of all code and inventions; require that Customer retains sole ownership of custom deliverables and bespoke work product paid for under the agreement.",
        "high": "High-Priority Redline: Carve out Customer's pre-existing intellectual property and limit Vendor's rights to pre-existing background tools.",
        "medium": "Negotiation Note: Clarify license scope to ensure Customer receives a perpetual, worldwide license to utilize delivered materials.",
        "low": "Acceptable as drafted: Work product ownership and pre-existing IP reservations are properly allocated.",
    },
    "assignment": {
        "critical": "Priority 1 Redline: Strike asymmetric assignment terms; ensure Customer has reciprocal transfer rights in connection with an M&A, merger, or corporate restructuring.",
        "high": "High-Priority Redline: Require mutual written consent for general assignment, with customary exceptions for affiliates and bona fide successors.",
        "medium": "Negotiation Note: Add requirement that consent to assignment shall not be unreasonably withheld, conditioned, or delayed.",
        "low": "Acceptable as drafted: Assignment restrictions provide standard affiliate and successor carveouts.",
    },
    "confidentiality": {
        "critical": "Priority 1 Redline: Bound confidentiality survival to a standard 3-to-5 year term; strike perpetual non-disclosure obligations for standard commercial data.",
        "high": "High-Priority Redline: Ensure confidentiality obligations are mutual and add standard exclusions (public knowledge, independent development, lawful third-party receipt).",
        "medium": "Negotiation Note: Define clear standard of care (at least reasonable care) and specify permitted disclosures to legal and financial advisors.",
        "low": "Acceptable as drafted: Confidentiality definitions and exclusions reflect standard commercial practice.",
    },
    "non-compete / non-solicit": {
        "critical": "Priority 1 Redline: Strike broad multi-year commercial non-competes; replace with a targeted 12-month non-solicitation of active project personnel.",
        "high": "High-Priority Redline: Exclude general employment solicitations (job board postings) and limit duration to 12 months post-termination.",
        "medium": "Negotiation Note: Confirm that non-solicitation restrictions are mutual and restricted strictly to directly engaged key personnel.",
        "low": "Acceptable as drafted: Non-solicitation clause is narrowly tailored and geographically reasonable.",
    },
    "governing law": {
        "critical": "Priority 1 Redline: Reject hostile or distant foreign jurisdiction; designate a neutral, predictable commercial forum (e.g. Delaware or New York).",
        "high": "High-Priority Redline: Align governing law and court venue with the primary operational jurisdiction of the contracting parties.",
        "medium": "Negotiation Note: Ensure dispute resolution provisions specify mutual venue and waive jury trial or provide for arbitration if appropriate.",
        "low": "Acceptable as drafted: Governing law (e.g., Delaware) and venue are commercially standard and predictable.",
    },
    "revenue / payment obligations": {
        "critical": "Priority 1 Redline: Remove vendor rights to immediately suspend services upon billing dispute; establish good-faith dispute notice and escalation periods.",
        "high": "High-Priority Redline: Standardize payment terms to Net 30 or Net 45 days, and cap late interest charges to statutory maximums.",
        "medium": "Negotiation Note: Clarify that taxes and reimbursable expenses must be pre-approved in writing.",
        "low": "Acceptable as drafted: Standard Net 30 invoicing and payment procedure.",
    },
    "exclusivity": {
        "critical": "Priority 1 Redline: Reject exclusivity commitments that restrict commercial freedom; make appointment non-exclusive.",
        "high": "High-Priority Redline: Condition exclusivity on counterparty meeting minimum revenue and sales quotas with right to terminate if unmet.",
        "medium": "Negotiation Note: Limit exclusivity scope to specific defined products and geographic territories.",
        "low": "Acceptable as drafted: Non-exclusive appointment preserves market flexibility.",
    },
    "audit rights": {
        "critical": "Priority 1 Redline: Prohibit unrestricted on-site inspections; limit audits to once annually during business hours upon 15 business days notice.",
        "high": "High-Priority Redline: Require audits to be performed by an independent certified auditor bound by confidentiality.",
        "medium": "Negotiation Note: Specify that auditing party bears audit expenses unless a material underpayment (>5%) is proven.",
        "low": "Acceptable as drafted: Annual audit rights with advance notice are commercially standard.",
    },
    "change of control": {
        "critical": "Priority 1 Redline: Prevent automatic termination upon change of control; allow termination only if the acquiring entity is a direct commercial competitor.",
        "high": "High-Priority Redline: Require a 30-day consultation period before any termination right can be exercised following a merger or financing.",
        "medium": "Negotiation Note: Clarify that internal reorganizations and equity financings do not trigger change-of-control provisions.",
        "low": "Acceptable as drafted: Change of control terms protect legitimate commercial interests without impeding financing.",
    },
}

DEFAULT_NEGOTIATION_NOTES: dict[str, str] = {
    "critical": "Priority 1 Redline: Demand reciprocal obligations, narrow restriction scope, and introduce commercial liability caps.",
    "high": "High-Priority Redline: Strike one-sided restrictions and conform terms to market benchmarks.",
    "medium": "Negotiation Note: Clarify definitions and ensure obligations are balanced.",
    "low": "Acceptable as drafted: Standard commercial provision.",
}


class ContractRiskAgent:
    def __init__(self, config: AppConfig, retriever: LocalRetriever, memory: MemoryStore):
        self.config = config
        self.retriever = retriever
        self.memory = memory
        self.classifier = KeywordRiskClassifier(config.classifier_path)

    @traceable_step("analyze_contract")
    def analyze(
        self,
        parsed: ParsedContract,
        reviewer_preferences: str = "",
        reviewing_party: ReviewingParty = "customer",
    ) -> RiskScorecard:
        trace = TraceRecorder()
        with trace.step("index_contract_clauses", contract_id=parsed.id, chunks=len(parsed.chunks)):
            self.retriever.add_contract_chunks(parsed.chunks)
        with trace.step("classify_clauses", chunks=len(parsed.chunks)) as event:
            classifications = self.classifier.classify(parsed.chunks)
            event.token_count = sum(len(chunk.text.split()) for chunk in parsed.chunks)
        with trace.step("retrieve_precedents", classifications=len(classifications)):
            evidence_by_chunk = self._retrieve_evidence(parsed, classifications)

        with trace.step("llm_structured_review", model=self.config.openai_model, enabled=bool(self.config.openai_api_key)):
            llm_scorecard = self._try_openai_review(parsed, classifications, evidence_by_chunk, reviewer_preferences, reviewing_party)
        if llm_scorecard:
            llm_scorecard.agent_trace = trace.scorecard_trace()
            return llm_scorecard

        scorecard = self._offline_review(parsed, classifications, evidence_by_chunk, reviewer_preferences, reviewing_party)
        scorecard.agent_trace = trace.scorecard_trace()
        return scorecard

    def _retrieve_evidence(
        self,
        parsed: ParsedContract,
        classifications: list[ClauseClassification],
    ) -> dict[str, list[RetrievalEvidence]]:
        """Retrieve external playbook and precedent guidance strictly matching each clause's category."""
        chunks_by_id = {chunk.id: chunk for chunk in parsed.chunks}
        evidence: dict[str, list[RetrievalEvidence]] = {}
        for classification in classifications:
            chunk = chunks_by_id[classification.chunk_id]
            query = f"{classification.clause_type} {chunk.text[:250]}"
            # Only search playbook and market_precedent sources with strict clause_type filtering
            try:
                evidence[classification.chunk_id] = self.retriever.search(
                    query,
                    k=3,
                    source_types=["playbook", "market_precedent"],
                    clause_type=classification.clause_type,
                    min_score=0.08,
                )
            except TypeError:
                try:
                    evidence[classification.chunk_id] = self.retriever.search(
                        query,
                        k=3,
                        source_types=["playbook", "market_precedent"],
                        clause_type=classification.clause_type,
                    )
                except TypeError:
                    evidence[classification.chunk_id] = self.retriever.search(query, k=3)
        return evidence

    def retrieve_clauses(self, query: str, k: int = 5, contract_id: str | None = None) -> list[RetrievalEvidence]:
        """Retrieve clauses from uploaded contracts with strict contract isolation."""
        filters = {"contract_id": contract_id} if contract_id else None
        return self.retriever.search(query, k=k, filters=filters, source_types=["uploaded_contract"])

    def classify_clause(self, text: str) -> list[ClauseClassification]:
        from src.schema import ContractChunk

        chunk = ContractChunk(contract_id="ad_hoc", source_name="question", page=1, start_offset=0, end_offset=len(text), text=text)
        return self.classifier.classify([chunk])

    def get_risk_score(
        self,
        classification: ClauseClassification,
        clause_text: str,
        reviewing_party: ReviewingParty = "customer",
    ) -> int:
        score, _, _, _ = evaluate_clause_risk(
            clause_type=classification.clause_type,
            clause_text=clause_text,
            base_level=classification.risk_level,
            confidence=classification.confidence,
            reviewing_party=reviewing_party,
        )
        return score

    def search_precedents(self, query: str, k: int = 5, clause_type: str | None = None) -> list[RetrievalEvidence]:
        try:
            return self.retriever.search(
                query,
                k=k,
                source_types=["playbook", "market_precedent"],
                clause_type=clause_type,
                min_score=0.08,
            )
        except TypeError:
            try:
                return self.retriever.search(
                    query,
                    k=k,
                    source_types=["playbook", "market_precedent"],
                    clause_type=clause_type,
                )
            except TypeError:
                return self.retriever.search(query, k=k)

    def generate_redline(self, clause_type: str, original_language: str):
        return generate_redline(clause_type, original_language)

    def _try_openai_review(
        self,
        parsed: ParsedContract,
        classifications: list[ClauseClassification],
        evidence_by_chunk: dict[str, list[RetrievalEvidence]],
        reviewer_preferences: str,
        reviewing_party: ReviewingParty,
    ) -> RiskScorecard | None:
        if not self.config.openai_api_key:
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None

        excerpt = _compact_contract(parsed.full_text)
        classifier_notes = _classification_notes(classifications)
        retrieved_context = _evidence_context(evidence_by_chunk)
        reviewer_memory = "\n".join(
            part for part in [reviewer_preferences, self.memory.feedback_context()] if part.strip()
        )
        prompt = build_review_prompt(excerpt, classifier_notes, retrieved_context, reviewer_memory)
        party_instruction = f"\nReviewing Party Perspective: {reviewing_party.upper()}. Evaluate risks from this party's viewpoint."
        prompt += party_instruction

        try:
            client = OpenAI(api_key=self.config.openai_api_key)
            if hasattr(client.responses, "parse"):
                response = client.responses.parse(
                    model=self.config.openai_model,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    text_format=RiskScorecard,
                )
                scorecard = response.output_parsed
                scorecard.reviewing_party = reviewing_party
                return scorecard

            response = client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            scorecard = RiskScorecard.model_validate_json(content)
            scorecard.reviewing_party = reviewing_party
            return scorecard
        except (Exception, ValidationError):
            return None

    def _offline_review(
        self,
        parsed: ParsedContract,
        classifications: list[ClauseClassification],
        evidence_by_chunk: dict[str, list[RetrievalEvidence]],
        reviewer_preferences: str,
        reviewing_party: ReviewingParty,
    ) -> RiskScorecard:
        chunks_by_id = {chunk.id: chunk for chunk in parsed.chunks}
        findings: list[RiskFinding] = []

        for classification in classifications:
            chunk = chunks_by_id[classification.chunk_id]
            evidence = evidence_by_chunk.get(chunk.id, [])
            category = category_for_clause(classification.clause_type)

            risk_score, effective_level, triggers, explanation = evaluate_clause_risk(
                clause_type=classification.clause_type,
                clause_text=chunk.text,
                base_level=classification.risk_level,
                confidence=classification.confidence,
                reviewing_party=reviewing_party,
                reviewer_preferences=reviewer_preferences,
            )

            # Skip benign low-risk clauses unless confidence is high
            if effective_level == "low" and classification.confidence < 0.60:
                continue

            redline = generate_redline(classification.clause_type, chunk.text)
            market_standard = _market_standard(classification.clause_type, evidence)

            findings.append(
                RiskFinding(
                    clause_type=classification.clause_type,
                    quote=_trim_quote(chunk.text),
                    page=chunk.page,
                    section_path=chunk.section_path,
                    clause_number=chunk.clause_number,
                    clause_title=chunk.clause_title,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    risk_level=effective_level,
                    category=category,
                    risk_score=risk_score,
                    review_priority=effective_level,
                    rationale=classification.rationale,
                    explanation=explanation,
                    risk_factors=triggers,
                    triggering_language=triggers,
                    business_impact=_business_impact(classification.clause_type, effective_level, reviewing_party),
                    suggested_negotiation_note=_negotiation_note(classification.clause_type, effective_level, reviewing_party),
                    reviewing_party=reviewing_party,
                    confidence=classification.confidence,
                    requires_human_review=classification.requires_human_review,
                    redline=redline,
                    market_standard=market_standard,
                    retrieved_evidence_ids=market_standard.precedent_ids if market_standard else [],
                )
            )

        missing = self._missing_clauses(classifications)
        score = overall_score(findings, missing)
        level = score_to_level(score)
        summary = _summary(parsed, findings, missing, reviewing_party)
        notes = reviewer_preferences or self.memory.feedback_context() or "No reviewer-specific preferences applied."

        high_critical = sum(1 for f in findings if f.risk_level in {"high", "critical"})

        return RiskScorecard(
            contract_summary=summary,
            overall_risk_score=score,
            risk_level=level,
            reviewing_party=reviewing_party,
            category_scores=category_scores(findings),
            findings=findings,
            missing_clauses=missing,
            reviewer_notes=notes,
            total_findings_count=len(findings),
            high_critical_count=high_critical,
        )

    def _missing_clauses(self, classifications: list[ClauseClassification]) -> list[str]:
        seen = {classification.clause_type for classification in classifications}
        return [label for label in MISSING_CLAUSE_IMPORTANCE if label not in seen]


def _compact_contract(text: str, max_chars: int = 18000) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n\n[Truncated for review prompt.]"


def _classification_notes(classifications: list[ClauseClassification]) -> str:
    if not classifications:
        return "No focused CUAD labels were detected by the local classifier."
    return "\n".join(
        f"- {item.clause_type} ({item.risk_level}, {item.confidence:.2f}): {item.rationale}"
        for item in classifications
    )


def _evidence_context(evidence_by_chunk: dict[str, list[RetrievalEvidence]]) -> str:
    items = []
    for evidence_items in evidence_by_chunk.values():
        for evidence in evidence_items:
            items.append(f"[{evidence.id}] {evidence.text}")
    return "\n".join(items[:20]) or "No retrieved context."


def _trim_quote(text: str, max_chars: int = 420) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= max_chars else cleaned[: max_chars - 3] + "..."


def _business_impact(clause_type: str, risk_level: str, party: ReviewingParty) -> str:
    if party == "customer":
        if risk_level == "critical":
            return f"Fatal financial/operational risk: The {clause_type} clause exposes Customer to unbounded downside, severe asymmetry, or immediate operational shutdown."
        if risk_level == "high":
            return f"Significant commercial exposure: The {clause_type} clause creates heavy obligations, broad restrictions, or loss of deliverable ownership."
        if risk_level == "medium":
            return f"Moderate friction: The {clause_type} provision should be checked for mutuality and alignment with standard operational risk tolerances."
        return f"Standard commercial term: {clause_type} provisions present low operational risk."
    elif party == "vendor":
        if risk_level in {"high", "critical"}:
            return f"Review note for Vendor: Unusually aggressive terms may trigger pushback from enterprise customer procurement or face enforceability challenges."
        return f"Favorable terms for Vendor: Language reinforces Vendor control and limits operational liability."
    else:
        return f"Commercial review impact: The {clause_type} provision exhibits {risk_level.upper()} severity imbalance."


def _negotiation_note(clause_type: str, risk_level: str, party: ReviewingParty) -> str:
    """Category-specific and perspective-aware negotiation note generator."""
    clause_key = clause_type.lower().strip()
    notes_for_clause = CATEGORY_NEGOTIATION_NOTES.get(clause_key, DEFAULT_NEGOTIATION_NOTES)
    note = notes_for_clause.get(risk_level, DEFAULT_NEGOTIATION_NOTES.get(risk_level, "Verify deal consistency."))

    if party == "vendor":
        if risk_level in {"low", "medium"}:
            return f"Vendor Position: Maintain protective language as drafted. If Customer procurement objects, offer standard bilateral fallback."
        return f"Vendor Position: Review commercial reasonableness. Be prepared to offer balanced language if required by enterprise deal terms."
    elif party == "neutral":
        return f"Neutral Recommendation: Standardize {clause_type} on market precedent to avoid dispute friction."
    return note


def _summary(parsed: ParsedContract, findings: list[RiskFinding], missing: list[str], party: ReviewingParty) -> str:
    if not findings:
        return f"{parsed.source_name} was reviewed from the {party.title()} perspective with no high-confidence risky clauses detected."

    critical_count = sum(1 for f in findings if f.risk_level == "critical")
    high_count = sum(1 for f in findings if f.risk_level == "high")

    label_counts = Counter(finding.clause_type for finding in findings)
    top_clauses = ", ".join(label for label, _ in label_counts.most_common(3))
    missing_text = f" Missing key clauses: {', '.join(missing)}." if missing else ""

    override_note = ""
    if critical_count > 0:
        override_note = f" (Critical Severity Floor Applied: {critical_count} critical finding(s) anchor the aggregate score)."

    return (
        f"Reviewed {parsed.source_name} from the {party.title()} perspective. Identified {len(findings)} finding(s) "
        f"({critical_count} critical, {high_count} high), concentrated primarily in {top_clauses}.{missing_text}{override_note}"
    )


def _market_standard(clause_type: str, evidence: list[RetrievalEvidence]) -> MarketStandard | None:
    # Filter strictly to external reference material matching this clause category
    target_type = clause_type.lower().strip()
    precedents = [
        item for item in evidence
        if item.source_type in {"playbook", "market_precedent"}
        and (
            target_type == (item.metadata.get("clause_type") or "").lower()
            or target_type in (item.clause_category or "").lower()
            or target_type in item.document_title.lower()
        )
    ]
    if not precedents:
        return MarketStandard(
            summary=f"No external market reference material was found specifically matching '{clause_type}'.",
            deviation="Independent benchmark precedent is not indexed for this specific clause category; review using standard commercial guidelines.",
            source_title="Playbook Reference",
            source_type="market_precedent",
            precedent_ids=[],
        )

    best = precedents[0]
    return MarketStandard(
        summary=f"{best.document_title}: {best.text}",
        deviation="Compare contract wording against this benchmark. Deviations usually involve mutuality, monetary caps, notice, or cure rights.",
        source_title=best.document_title or "Market Reference Standard",
        source_type=best.source_type,
        precedent_ids=[p.id for p in precedents],
    )
