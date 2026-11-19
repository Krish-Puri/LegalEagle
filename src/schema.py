from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


RiskLevel = Literal["low", "medium", "high", "critical"]
ReviewingParty = Literal["customer", "vendor", "neutral"]
SourceType = Literal["playbook", "market_precedent", "uploaded_contract", "reviewer_feedback"]


class ContractPage(BaseModel):
    page_number: int = Field(ge=1)
    text: str
    sections: list[str] = Field(default_factory=list)


class ContractChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    contract_id: str
    source_name: str
    page: int = Field(ge=1)
    section_path: list[str] = Field(default_factory=list)
    clause_number: str | None = None
    clause_title: str | None = None
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    text: str

    @field_validator("end_offset")
    @classmethod
    def end_after_start(cls, end_offset: int, info):
        start = info.data.get("start_offset", 0)
        if end_offset < start:
            raise ValueError("end_offset must be greater than or equal to start_offset")
        return end_offset


class ParsedContract(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    pages: list[ContractPage]
    chunks: list[ContractChunk] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


class RetrievalEvidence(BaseModel):
    id: str
    source: str
    source_type: SourceType = "playbook"
    document_title: str = ""
    clause_category: str = ""
    source_location: str = ""
    text: str
    score: float = Field(ge=0)
    similarity_score: float | None = None
    reason_retrieved: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class ClauseClassification(BaseModel):
    chunk_id: str
    clause_type: str
    confidence: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    category: str = "general"
    rationale: str
    risk_factors: list[str] = Field(default_factory=list)
    review_priority: RiskLevel = "medium"
    requires_human_review: bool = True


class RedlineSuggestion(BaseModel):
    original_language: str
    proposed_language: str
    redline_markdown: str
    negotiation_reason: str
    negotiation_objective: str = ""
    reviewer_warning: str = "Draft recommendation for review only. Not legal advice."
    diff_preview: str = ""


class MarketStandard(BaseModel):
    summary: str
    deviation: str
    source_title: str = "Risk Playbook Precedent"
    source_type: str = "market_precedent"
    precedent_ids: list[str] = Field(default_factory=list)


class RiskFinding(BaseModel):
    clause_type: str
    quote: str
    page: int = Field(ge=1)
    section_path: list[str] = Field(default_factory=list)
    clause_number: str | None = None
    clause_title: str | None = None
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    risk_level: RiskLevel
    category: str = "general"
    risk_score: int = Field(default=0, ge=0, le=100)
    review_priority: RiskLevel = "medium"
    rationale: str
    explanation: str = ""
    risk_factors: list[str] = Field(default_factory=list)
    triggering_language: list[str] = Field(default_factory=list)
    business_impact: str
    suggested_negotiation_note: str
    reviewing_party: ReviewingParty = "customer"
    confidence: float | None = None
    requires_human_review: bool = True
    redline: RedlineSuggestion | None = None
    market_standard: MarketStandard | None = None
    retrieved_evidence_ids: list[str] = Field(default_factory=list)


class RiskScorecard(BaseModel):
    contract_summary: str
    overall_risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    reviewing_party: ReviewingParty = "customer"
    category_scores: dict[str, int] = Field(default_factory=dict)
    findings: list[RiskFinding] = Field(default_factory=list)
    missing_clauses: list[str] = Field(default_factory=list)
    reviewer_notes: str = ""
    agent_trace: list[dict[str, str]] = Field(default_factory=list)
    total_findings_count: int = 0
    high_critical_count: int = 0

    @field_validator("findings")
    @classmethod
    def sort_findings(cls, findings: list[RiskFinding]) -> list[RiskFinding]:
        return sorted(findings, key=lambda item: (-risk_weight(item.risk_level), -item.risk_score, item.page, item.start_offset))


class AgentToolCall(BaseModel):
    tool_name: str
    args: dict[str, str] = Field(default_factory=dict)
    result_summary: str
    latency_ms: float = 0


class ContractQuestionAnswer(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    memory_summary: str = ""
    contract_id: str | None = None
    contract_name: str | None = None
    grounded: bool = True


def risk_weight(level: str) -> int:
    return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(level, 0)
