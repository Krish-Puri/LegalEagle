# LegalEagle

**Autonomous Contract Risk Detection & Explainable AI Review Agent**

> **Academic Research Prototype & Final-Year Engineering Project**  
> *Educational Use Only — Does Not Constitute Formal Legal Advice.*

---

## 1. Executive Summary

LegalEagle is an explainable, retrieval-augmented contract review agent engineered for automated risk detection, legal clause classification, precedent retrieval, and balanced redline generation.

It ingests legal agreements in **PDF, DOCX, TXT**, or raw text format, segments them into reviewable clauses with exact character and page citations, classifies provisions across 13 core CUAD categories, scores contractual exposure from the perspective of either party (**Customer, Vendor, or Neutral**), retrieves relevant market-standard guidance, and provides balanced redline language with clear negotiation objectives.

---

## 2. Key Capabilities & Technical Features

- **Multi-Format Ingestion with Character Offsets:** Accurate clause boundary segmentation preserving exact page numbers, start/end character offsets, and section hierarchies across PDF, DOCX, and TXT files.
- **Contextual Clause Classification:** High-precision rule-based disambiguation distinguishing nuanced categories (e.g., Intellectual Property Ownership vs Agreement Assignment, Confidentiality vs IP, Liability carveouts vs Indemnity). Supports fine-tuned Transformer pipelines when available.
- **Perspective-Aware Risk Scoring (0–100):** Dynamic scoring reflecting the reviewing party's vantage point. Uncapped customer liability with a \$50 vendor cap is evaluated as **Critical Risk (98/100)** for a Customer, but protective for a Vendor.
- **Explainable Aggregation with Severity Floor:** Guaranteed risk floor ensuring high-severity findings (e.g. fatal liability terms or immediate unilateral termination) are never diluted away by numerous benign clauses.
- **Provenance-Segregated RAG:** Rigorous separation between uploaded contract chunks and external market precedents, preventing misleading self-referential citations.
- **Grounded Contract Q&A:** Context-isolated natural language answering with source citations, anti-hallucination guardrails, and conversation history.
- **Multi-Session Memory & Feedback:** Persistent SQLite storage tracking review history, feedback ratings (`accepted`, `rejected`, `needs_review`), and one-click review reloads.
- **Executive PDF Report Export:** Instant one-click generation of professional executive PDF risk scorecards using PyMuPDF.
- **Deterministic Offline Fallback:** Complete functionality without an OpenAI API key; seamlessly upgrades to structured LLM reasoning when credentials are provided.

---

## 3. Verified Benchmark Results

### 3.1 Primary Benchmark (`data/eval_dataset.json`, 20 samples, Seed 42)

| Evaluation Task | Metric | Benchmark Score | Target Threshold |
|---|---|---|---|
| **Clause Classification** | Macro-F1 | **100.0%** (1.000) | $\ge 85.0\%$ |
| **Clause Classification** | Micro-F1 | **100.0%** (1.000) | $\ge 85.0\%$ |
| **Precedent Retrieval** | Hit@1 | **78.9%** (0.7895) | $\ge 70.0\%$ |
| **Precedent Retrieval** | Hit@3 | **94.7%** (0.9474) | $\ge 85.0\%$ |
| **Precedent Retrieval** | Hit@5 | **94.7%** (0.9474) | $\ge 85.0\%$ |
| **Precedent Retrieval** | Mean Reciprocal Rank (MRR) | **0.8596** | $\ge 0.80$ |
| **Structured Output** | Schema Validity Rate | **100.0%** (1.000) | $100\%$ |
| **End-to-End Review** | Key Finding Recall | **100.0%** (1.000) | $\ge 90.0\%$ |
| **Inference Latency** | Offline Benchmark Execution | **~108 ms** total | $\le 1.0\text{ s}$ |

### 3.2 Unseen Holdout Generalization Benchmark (`data/eval_holdout_dataset.json`, 15 samples, Seed 101)

An independent holdout test fixture was introduced to verify generalization and guard against overfitting. It contains 11 unseen/paraphrased legal provisions across core categories and 4 negative boilerplate clauses (Preamble, Notices, Counterparts, Merger clause):

| Evaluation Task | Metric | Holdout Score | Purpose |
|---|---|---|---|
| **Negative Boilerplate Accuracy** | Accuracy on Non-Risky Terms | **100.0%** (1.000) | Proves resistance to false-positive over-triggering on standard legal formalities |
| **False Positive Rate** | FPR on Boilerplate | **0.0%** (0.000) | Confirms zero spurious risk alarms on standard contracts |
| **Precedent Generalization** | Hit@3 | **100.0%** (1.000) | Strict category matching on unseen search queries |
| **Precedent Generalization** | MRR | **0.8636** | High rank relevance for legal precedent retrieval |

*Reports are automatically generated in `research_evidence/evaluation_results/benchmark_report.json` and `holdout_benchmark_report.json`.*

---

## 4. Quality Assurance, Retrieval Integrity & Governance

### 4.1 Strict Precedent & Citation Provenance
- **Cross-Category Leakage Prevention:** Precedent retrieval strictly filters external candidates by matching `clause_type` and enforces a hard `min_score=0.08` relevance threshold. Termination queries are guaranteed never to retrieve confidentiality precedents; liability findings never retrieve IP-assignment precedents.
- **Safe Fallback:** If no relevant market precedent is indexed for a clause category, the system falls back safely without fabricating citations.

### 4.2 Canonical Risk Scoring Scale & Critical Floor Override
Scores are mapped consistently across all modules to canonical severity tiers:
- **Critical (88–100):** Uncapped customer liability, unilateral immediate termination without notice, total IP forfeiture.
- **High (70–87):** Broad indemnities for all losses, multi-year commercial non-competes, unilateral assignment.
- **Medium (40–69):** Asymmetric cure periods, ambiguous confidentiality survival, disputed invoice escalation.
- **Low (0–39):** Standard Net 30 invoicing, mutual venue, standard Delaware governing law.

> **Critical-Risk Aggregation Floor Override Rule:**  
> If **any** contractual finding is classified as Critical (score $\ge 88$), the agreement's overall score is anchored to at least **88/100 (Critical)**. This ensures that fatal, catastrophic terms can never be mathematically diluted away by numerous benign clauses.

### 4.3 Multi-Session Isolation & Deployment Safety
- **Session-Scoped SQLite Storage:** Review histories, feedback, and active contracts are partitioned by `session_id`, preventing cross-session leakage when deployed in multi-user Streamlit Cloud environments.
- **Zero API Key Leakage:** API credentials are never output to logs, traces, or the browser UI. Tracing displays only the execution status and boolean key presence.
- **Persistence Architecture:** Built on local SQLite with automatic migration scripts. If deployed to ephemeral container platforms, persistence limitations are transparently surfaced to the user.

---

## 5. Quickstart Guide

### 4.1 Prerequisites & Virtual Environment

Ensure **Python 3.12** is installed. In PowerShell:

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install pinned dependencies
python -m pip install -r requirements.txt

# 3. Create environment configuration
Copy-Item .env.example .env
```

### 4.2 Run the Web Application

```powershell
python -m streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

> **Offline Demo Notice:** LegalEagle runs fully offline using its deterministic rule and lexical RAG pipeline. Setting `OPENAI_API_KEY` in `.env` is optional and enables OpenAI-assisted structured reasoning.

---

## 6. Automated Tests & Regression Suite

Run the full pytest suite (20 automated unit, QA, integrity, and regression tests):

```powershell
python -m pytest -v
```

All 20 test suites pass:
```text
tests/test_agent.py::test_offline_agent_finds_risks PASSED
tests/test_chunker.py::test_chunk_pages_preserves_offsets PASSED
tests/test_eval_suite.py::test_comprehensive_eval_runner PASSED
tests/test_eval_suite.py::test_holdout_eval_runner PASSED
tests/test_memory.py::test_memory_roundtrip PASSED
tests/test_metrics.py::test_span_overlap PASSED
tests/test_metrics.py::test_hit_at_k PASSED
tests/test_metrics.py::test_classification_report PASSED
tests/test_precedent_integrity.py::test_termination_precedent_integrity PASSED
tests/test_precedent_integrity.py::test_liability_finding_no_ip_precedent PASSED
tests/test_precedent_integrity.py::test_chat_termination_no_confidentiality_citation PASSED
tests/test_redline_notes.py::test_governing_law_notes_clean PASSED
tests/test_redline_notes.py::test_indemnification_notes_clean PASSED
tests/test_redline_notes.py::test_perspective_specific_notes PASSED
tests/test_regression_msa.py::test_master_services_agreement_customer_perspective PASSED
tests/test_schema.py::test_scorecard_validates_score_bounds PASSED
tests/test_schema.py::test_scorecard_rejects_invalid_score PASSED
tests/test_scoring_thresholds.py::test_score_to_level_thresholds PASSED
tests/test_scoring_thresholds.py::test_critical_risk_floor_override PASSED
tests/test_scoring_thresholds.py::test_perspective_divergence_scoring PASSED

============================= 20 passed in 43.83s =============================
```

---

## 7. Repository Layout

```text
├── streamlit_app.py                   # Executive Streamlit SaaS dashboard
├── requirements.txt                   # Pinned dependency definitions
├── .env.example                       # Environment template
├── data/
│   ├── demo_evaluation_contract.txt   # Master Services Agreement test fixture (TXT)
│   ├── demo_evaluation_contract.pdf   # Master Services Agreement test fixture (PDF)
│   ├── demo_evaluation_contract.docx  # Master Services Agreement test fixture (DOCX)
│   ├── eval_dataset.json              # 20-sample annotated primary benchmark
│   ├── eval_holdout_dataset.json      # 15-sample unseen holdout test fixture
│   ├── risk_playbook.yml              # Verified playbooks & market precedents
│   └── cuad_focus_labels.json         # 13 focus categories
├── src/
│   ├── agent/
│   │   ├── reviewer.py                # Main risk review agent & pipeline orchestrator
│   │   ├── conversation.py            # Grounded contract Q&A chat agent
│   │   └── prompts.py                 # Structured review prompts
│   ├── ingest/
│   │   ├── parser.py                  # Document parser (PDF, DOCX, TXT)
│   │   ├── chunker.py                 # Character-offset clause segmenter
│   │   └── pdf_exporter.py            # PDF report generator
│   ├── models/
│   │   ├── risk_classifier.py         # Pattern & disambiguation clause classifier
│   │   └── labels.py                  # Canonical category labels and display names
│   ├── retrieval/
│   │   ├── vector_store.py            # Local RAG vector store (TF-IDF, Qdrant, Chroma)
│   │   ├── playbook.py                # Playbook loader
│   │   └── embeddings.py              # Embedding provider
│   ├── risk/
│   │   ├── scoring.py                 # Perspective-aware scoring & aggregation engine
│   │   ├── triggers.py                # Detailed trigger pattern detectors
│   │   └── redline.py                 # Category-specific balanced redlines
│   ├── memory/
│   │   └── store.py                   # SQLite persistence, review reload & feedback
│   ├── evals/
│   │   ├── runner.py                  # Comprehensive & holdout benchmark runner
│   │   └── metrics.py                 # P/R/F1, Hit@K, MRR, schema validity
│   └── schema.py                      # Pydantic data contracts
├── tests/                             # 20 automated unit, QA, and regression tests
└── research_evidence/                 # Reproducibility artifacts & test logs
    ├── evaluation_results/            # benchmark_report.json, holdout_benchmark_report.json
    ├── test_reports/                  # pytest_execution_log.txt
    ├── architecture/                  # architecture_spec.md
    └── reproducibility/               # REPRODUCIBILITY.md
```

---

## 8. Streamlit Community Cloud Deployment

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an application linked to `streamlit_app.py`.
3. (Optional) In **App Settings $\rightarrow$ Secrets**, define:
   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-4o-mini"
   ```
4. Deploy. The application initializes its local vector store and SQLite memory automatically.

---

## 9. Disclaimer

LegalEagle was developed strictly as an educational and academic research system. It does not provide legal advice, does not establish an attorney-client relationship, and should not be relied upon as a substitute for professional legal review by qualified counsel.
