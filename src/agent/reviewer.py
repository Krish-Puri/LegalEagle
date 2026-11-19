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
)
from src.risk.triggers import triggering_language
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
        """Retrieve external playbook and precedent guidance (excluding uploaded contract chunks)."""
        chunks_by_id = {chunk.id: chunk for chunk in parsed.chunks}
        evidence: dict[str, list[RetrievalEvidence]] = {}
        for classification in classifications:
            chunk = chunks_by_id[classification.chunk_id]
            query = f"{classification.clause_type} {chunk.text[:250]}"
            # Only search playbook and market_precedent sources to prevent circular citations
            evidence[classification.chunk_id] = self.retriever.search(
                query,
                k=3,
                source_types=["playbook", "market_precedent"],
                clause_type=classification.clause_type,
            )
        return evidence

    def retrieve_clauses(self, query: str, k: int = 5, contract_id: str | None = None) -> list[RetrievalEvidence]:
        """Retrieve clauses from uploaded contracts with optional contract isolation."""
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

    def search_precedents(self, query: str, k: int = 5) -> list[RetrievalEvidence]:
        return self.retriever.search(query, k=k, source_types=["playbook", "market_precedent"])

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

            # Skip benign low-risk clauses unless confidence is high or requested
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
                    retrieved_evidence_ids=[item.id for item in evidence],
                )
            )

        missing = self._missing_clauses(classifications)
        score = overall_score(findings, missing)
        level = _score_to_level(score)
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
    if party == "customer":
        if risk_level == "critical":
            return f"Priority 1 Redline: Strike unilateral terms; demand a mutual 12-month liability cap and mandatory 30-day notice with cure period."
        if risk_level == "high":
            return f"High-Priority Redline: Narrow {clause_type} scope, demand reciprocal obligations, and retain ownership of custom work product."
        if risk_level == "medium":
            return f"Negotiation Note: Propose standard bilateral language with 3-year confidentiality survival and reasonable Net 30 payment terms."
        return f"Acceptable as drafted: Verify consistency with transaction terms."
    elif party == "vendor":
        return f"Vendor Position: Maintain protective language where feasible; offer mutual 12-month cap if customer procurement pushes back."
    else:
        return f"Balanced Recommendation: Standardize {clause_type} on market precedents with mutual rights and clear cure periods."


def _score_to_level(score: int) -> RiskLevel:
    if score >= 75:
        return "critical"
    if score >= 50:
        return "high"
    if score >= 25:
        return "medium"
    return "low"


def _summary(parsed: ParsedContract, findings: list[RiskFinding], missing: list[str], party: ReviewingParty) -> str:
    if not findings:
        return f"{parsed.source_name} was reviewed from the {party.title()} perspective with no high-confidence risky clauses detected."

    critical_count = sum(1 for f in findings if f.risk_level == "critical")
    high_count = sum(1 for f in findings if f.risk_level == "high")

    label_counts = Counter(finding.clause_type for finding in findings)
    top_clauses = ", ".join(label for label, _ in label_counts.most_common(3))
    missing_text = f" Missing key clauses: {', '.join(missing)}." if missing else ""

    return (
        f"Reviewed {parsed.source_name} from the {party.title()} perspective. Identified {len(findings)} finding(s) "
        f"({critical_count} critical, {high_count} high), concentrated primarily in {top_clauses}.{missing_text}"
    )


def _market_standard(clause_type: str, evidence: list[RetrievalEvidence]) -> MarketStandard | None:
    # Filter strictly to external reference material
    precedents = [item for item in evidence if item.source_type in {"playbook", "market_precedent"}]
    if not precedents:
        return MarketStandard(
            summary=f"No external market reference material was found for '{clause_type}'.",
            deviation="Independent benchmark precedent is not indexed for this specific clause category; review using standard commercial guidelines.",
            source_title="Local Playbook",
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
