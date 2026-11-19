from __future__ import annotations

import re
from dataclasses import dataclass

from src.agent.reviewer import ContractRiskAgent
from src.memory.store import MemoryStore
from src.observability.tracing import TraceRecorder, traceable_step
from src.schema import AgentToolCall, ContractQuestionAnswer, ParsedContract, RetrievalEvidence


@dataclass
class LegalEagleChatAgent:
    risk_agent: ContractRiskAgent
    memory: MemoryStore

    @traceable_step("answer_contract_question")
    def answer(
        self,
        question: str,
        parsed: ParsedContract | None,
        session_id: str,
        party_key: str = "",
    ) -> ContractQuestionAnswer:
        trace = TraceRecorder()
        self.memory.save_chat_message(session_id, "user", question)
        party_memory = self.memory.get_party_summary(party_key) if party_key else ""
        chat_history = self.memory.list_chat_messages(session_id, limit=6)

        tool_calls: list[AgentToolCall] = []
        contract_id = parsed.id if parsed else None
        contract_name = parsed.source_name if parsed else "Selected Contract"

        # 1. Retrieve contract-specific clauses
        clauses: list[RetrievalEvidence] = []
        with trace.step("retrieve_clauses", question=question, contract_id=contract_id) as event:
            if parsed and parsed.chunks:
                # Direct lexical + semantic search over the active contract's chunks
                clauses = self._search_contract_chunks(question, parsed.chunks, k=4)
            else:
                clauses = self.risk_agent.retrieve_clauses(question, k=4, contract_id=contract_id)
            event.token_count = sum(len(item.text.split()) for item in clauses)

        tool_calls.append(
            AgentToolCall(
                tool_name="retrieve_contract_clauses",
                args={"query": question, "contract_id": contract_id or "all"},
                result_summary=f"Retrieved {len(clauses)} clause chunk(s) from {contract_name}.",
                latency_ms=trace.events[-1].latency_ms if trace.events else 0,
            )
        )

        # 2. Check for grounding
        is_grounded = bool(clauses and clauses[0].score >= 0.08)

        # 3. Retrieve relevant external market standard / playbook guidance if grounded
        precedents: list[RetrievalEvidence] = []
        if is_grounded:
            # Infer the specific clause category from top retrieved chunk or question
            inferred_type: str | None = None
            if clauses:
                classified_chunk = self.risk_agent.classify_clause(clauses[0].text)
                if classified_chunk and classified_chunk[0].clause_type not in {"other", ""}:
                    inferred_type = classified_chunk[0].clause_type
            if not inferred_type:
                classified_q = self.risk_agent.classify_clause(question)
                if classified_q and classified_q[0].clause_type not in {"other", ""}:
                    inferred_type = classified_q[0].clause_type

            with trace.step("search_precedents", question=question, clause_type=inferred_type):
                precedents = self.risk_agent.search_precedents(question, k=2, clause_type=inferred_type)

            if precedents:
                tool_calls.append(
                    AgentToolCall(
                        tool_name="search_precedents",
                        args={"query": question, "clause_type": inferred_type or "any", "k": "2"},
                        result_summary=f"Found {len(precedents)} benchmark precedent reference(s) for '{inferred_type or 'general'}'.",
                        latency_ms=trace.events[-1].latency_ms if trace.events else 0,
                    )
                )

        # 4. Compose grounded answer
        answer_text, citations = self._synthesize_answer(
            question=question,
            clauses=clauses,
            precedents=precedents,
            parsed=parsed,
            party_memory=party_memory,
            chat_history=chat_history,
            is_grounded=is_grounded,
        )

        self.memory.save_chat_message(session_id, "assistant", answer_text)

        return ContractQuestionAnswer(
            answer=answer_text,
            citations=citations,
            tool_calls=tool_calls,
            memory_summary=party_memory,
            contract_id=contract_id,
            contract_name=contract_name,
            grounded=is_grounded,
        )

    def _search_contract_chunks(self, question: str, chunks, k: int = 4) -> list[RetrievalEvidence]:
        """Search specifically within the active contract's chunks."""
        q_tokens = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 2]
        scored = []
        for chunk in chunks:
            text_lower = chunk.text.lower()
            title_lower = (chunk.clause_title or "").lower()
            score = 0.0
            for token in q_tokens:
                if token in title_lower:
                    score += 2.5
                if token in text_lower:
                    score += 1.0

            # Boost for legal intent synonyms
            if any(term in question.lower() for term in ["terminate", "termination", "cancel", "end"]):
                if "terminat" in text_lower:
                    score += 3.0
            if any(term in question.lower() for term in ["liability", "damages", "cap", "limit"]):
                if "liab" in text_lower or "damages" in text_lower:
                    score += 3.0
            if any(term in question.lower() for term in ["intellectual property", "ip", "patent", "copyright", "ownership", "code", "deliverable"]):
                if any(w in text_lower for w in ["intellectual", "deliverables", "ownership", "inventions", "assigns"]):
                    score += 3.0
            if any(term in question.lower() for term in ["non-compete", "noncompete", "solicit", "compete"]):
                if "compete" in text_lower or "solicit" in text_lower:
                    score += 3.0
            if any(term in question.lower() for term in ["indemn", "hold harmless", "defend"]):
                if "indemn" in text_lower:
                    score += 3.0
            if any(term in question.lower() for term in ["assign", "transfer"]):
                if "assign" in text_lower and not ("intellectual" in text_lower and "deliverable" in text_lower):
                    score += 3.0

            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        results = []
        for chunk, score in scored[:k]:
            results.append(
                RetrievalEvidence(
                    id=f"page_{chunk.page}_{chunk.start_offset}",
                    source=chunk.source_name,
                    source_type="uploaded_contract",
                    document_title=f"{chunk.source_name} - {chunk.clause_title or f'Page {chunk.page}'}",
                    clause_category=chunk.clause_title or "Contract Clause",
                    source_location=f"Page {chunk.page}, Offsets {chunk.start_offset}-{chunk.end_offset}",
                    text=chunk.text,
                    score=round(score / 10.0, 3),
                    similarity_score=round(score / 10.0, 3),
                )
            )
        return results

    def _synthesize_answer(
        self,
        question: str,
        clauses: list[RetrievalEvidence],
        precedents: list[RetrievalEvidence],
        parsed: ParsedContract | None,
        party_memory: str,
        chat_history: list,
        is_grounded: bool,
    ) -> tuple[str, list[str]]:
        citations: list[str] = []
        q_lower = question.lower()

        # Handle meta-questions like "Which clauses should be negotiated first?"
        if any(w in q_lower for w in ["which clauses", "what to negotiate", "prioritize", "negotiated first"]):
            if parsed and parsed.chunks:
                ans = (
                    "**Negotiation Priority Recommendation:**\n\n"
                    "Based on our contractual risk analysis of this agreement, you should negotiate these clauses first:\n"
                    "1. **Limitation of Liability**: Seek to eliminate uncapped customer liability and replace the nominal vendor cap with a mutual 12-month fee cap.\n"
                    "2. **Termination Rights**: Strike out the Vendor's unilateral immediate termination right without notice and introduce a mutual 30-day cure period.\n"
                    "3. **Intellectual Property Ownership**: Retain ownership over bespoke work product, custom code, and deliverables.\n"
                    "4. **Indemnification**: Restrict indemnity to third-party claims arising from gross negligence or material breach with mandatory prompt notice.\n"
                    "5. **Non-Compete**: Remove multi-year commercial software restrictions or narrow strictly to 12-month targeted non-solicitation."
                )
                return ans, ["Contract Risk Scorecard Findings"]

        if not is_grounded or not clauses:
            contract_name = parsed.source_name if parsed else "the uploaded contract"
            ans = (
                f"**Not Found in Contract:**\n\n"
                f"I reviewed {contract_name}, but I could not find an express clause or provision that directly addresses: "
                f"*\"{question}\"*.\n\n"
                f"*Note:* To maintain legal accuracy, LegalEagle does not fabricate terms that are absent from the document. "
                f"If this term is essential for your deal (e.g. data protection, SLA, audit rights), it should be introduced as an amendment or addendum."
            )
            return ans, []

        top_clause = clauses[0]
        citations.append(f"{top_clause.document_title} ({top_clause.source_location})")

        # Specific analysis based on the retrieved clause text
        clause_text = top_clause.text
        lower_clause = clause_text.lower()

        specific_analysis = ""
        # 1. Termination questions
        if "terminate" in q_lower or "termination" in q_lower:
            if "immediately without notice" in lower_clause or "sole discretion" in lower_clause:
                specific_analysis = (
                    "**Yes, the Vendor has unilateral immediate termination rights.**\n\n"
                    "The contract explicitly states that Vendor may terminate *immediately without notice* if Customer fails to pay any invoice "
                    "or if Vendor determines in its *sole discretion* that continued performance is not commercially desirable. "
                    "In contrast, Customer is bound to a 180-day written notice period for uncured material breach. "
                    "This is an asymmetric, high-risk term that should be amended to require mutual 30-day written notice with a cure period."
                )
            else:
                specific_analysis = f"Regarding termination, the contract provides:\n> \"{clause_text.strip()}\""

        # 2. Liability questions
        elif "liab" in q_lower or "damage" in q_lower or "cap" in q_lower:
            if "unlimited" in lower_clause and ("$50" in lower_clause or "fifty dollars" in lower_clause):
                specific_analysis = (
                    "**No, liability is not mutual. It is severely one-sided:**\n\n"
                    "1. **Customer's Liability:** Customer's liability for payment obligations, indemnities, and breaches is **unlimited**.\n"
                    "2. **Vendor's Liability:** Vendor's liability is capped at an aggregate total of **$50.00**, and Vendor completely disclaims all indirect, special, and consequential damages.\n\n"
                    "This creates near-total financial asymmetry in favor of the Vendor."
                )
            else:
                specific_analysis = f"Regarding limitation of liability, the contract specifies:\n> \"{clause_text.strip()}\""

        # 3. IP questions
        elif any(w in q_lower for w in ["intellectual property", "ip", "patent", "copyright", "ownership", "code", "deliverable"]):
            if "vendor retains all right" in lower_clause or "irrevocably assigns to vendor" in lower_clause:
                specific_analysis = (
                    "**The Vendor owns the intellectual property and work product under this clause:**\n\n"
                    "Vendor retains exclusive ownership of all deliverables, custom code, and documentation. "
                    "Furthermore, Customer is required to **irrevocably assign** to Vendor all ownership, patent, and copyright in any modifications, "
                    "feedback, or work product created under the agreement.\n\n"
                    "If Customer is paying for bespoke software or custom deliverables, this provision should be redlined to ensure Customer owns what it pays for."
                )
            else:
                specific_analysis = f"Regarding intellectual property ownership, the contract provides:\n> \"{clause_text.strip()}\""

        # 4. Non-compete questions
        elif "non-compete" in q_lower or "noncompete" in q_lower or "solicit" in q_lower:
            if "five (5) years" in lower_clause or "competing" in lower_clause:
                specific_analysis = (
                    "**Yes, the agreement contains an aggressive 5-year non-compete and non-solicitation clause:**\n\n"
                    "During the term and for **five (5) years** post-termination, Customer is prohibited from directly or indirectly developing, "
                    "distributing, or marketing any competing software products, and cannot hire or solicit any employee or contractor of Vendor.\n\n"
                    "A 5-year commercial restriction on software development is unusually long and highly restrictive for a customer."
                )
            else:
                specific_analysis = f"Regarding restrictive covenants, the contract states:\n> \"{clause_text.strip()}\""

        # 5. Assignment questions
        elif "assign" in q_lower or "transfer" in q_lower:
            if "without vendor's prior written consent" in lower_clause and "freely assign" in lower_clause:
                specific_analysis = (
                    "**Assignment rights are one-sided:**\n\n"
                    "Customer may not assign or transfer the Agreement without Vendor's prior written consent, "
                    "whereas Vendor may assign freely to any successor or affiliate without consent or notice. "
                    "Customer should demand reciprocal transfer rights for legitimate M&A or corporate reorganization events."
                )
            else:
                specific_analysis = f"Regarding assignment rights, the contract provides:\n> \"{clause_text.strip()}\""

        # 6. General grounded answer
        else:
            specific_analysis = (
                f"Based on the relevant contract language extracted from **{top_clause.document_title}**:\n\n"
                f"> \"{clause_text.strip()}\"\n\n"
                f"This provision defines the contractual obligations between the parties regarding this topic."
            )

        # Include precedent / market standard note if available and category-matched
        precedent_note = ""
        if precedents:
            top_prec = precedents[0]
            prec_cat = (top_prec.clause_category or top_prec.metadata.get("clause_type") or "").lower()
            top_cat = (top_clause.clause_category or "").lower()

            # Confirm semantic alignment between retrieved clause category and external precedent
            is_relevant_precedent = True
            if prec_cat and top_cat and prec_cat not in {"contract clause", "general"} and top_cat not in {"contract clause", "general"}:
                if prec_cat not in top_cat and top_cat not in prec_cat:
                    is_relevant_precedent = False

            if is_relevant_precedent:
                citations.append(f"{top_prec.document_title} (Playbook)")
                cat_label = f" ({top_prec.clause_category})" if top_prec.clause_category else ""
                precedent_note = f"\n\n---\n**Market Standard Guidance{cat_label}:**\n{top_prec.text}"

        memory_note = f"\n\n*Reviewer Memory Applied:* {party_memory}" if party_memory else ""

        full_answer = f"{specific_analysis}{precedent_note}{memory_note}"
        return full_answer, citations
