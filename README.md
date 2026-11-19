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

Evaluated on the **LegalEagle Contract Risk Benchmark** (`data/eval_dataset.json`, Seed: 42, 20 annotated multi-class clauses and query pairs):

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

*Reproducible benchmark reports are automatically exported to `research_evidence/evaluation_results/benchmark_report.json`.*

---

## 4. Quickstart Guide

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

## 5. Automated Tests & Regression Suite

Run the full pytest suite (10 automated unit and regression tests):

```powershell
python -m pytest -v
```

All 10 test suites pass:
```text
tests/test_agent.py::test_offline_agent_finds_risks PASSED
tests/test_chunker.py::test_chunk_pages_preserves_offsets PASSED
tests/test_eval_suite.py::test_comprehensive_eval_runner PASSED
tests/test_memory.py::test_memory_roundtrip PASSED
tests/test_metrics.py::test_span_overlap PASSED
tests/test_metrics.py::test_hit_at_k PASSED
tests/test_metrics.py::test_classification_report PASSED
tests/test_regression_msa.py::test_master_services_agreement_customer_perspective PASSED
tests/test_schema.py::test_scorecard_validates_score_bounds PASSED
tests/test_schema.py::test_scorecard_rejects_invalid_score PASSED
```

---

## 6. Repository Layout

```text
├── streamlit_app.py                   # Executive Streamlit SaaS dashboard
├── requirements.txt                   # Dependency definitions
├── .env.example                       # Environment template
├── data/
│   ├── demo_evaluation_contract.txt   # Master Services Agreement test fixture (TXT)
│   ├── demo_evaluation_contract.pdf   # Master Services Agreement test fixture (PDF)
│   ├── demo_evaluation_contract.docx  # Master Services Agreement test fixture (DOCX)
│   ├── eval_dataset.json              # 20-sample annotated benchmark dataset
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
│   │   ├── runner.py                  # Comprehensive benchmark runner
│   │   └── metrics.py                 # P/R/F1, Hit@K, MRR, schema validity
│   └── schema.py                      # Pydantic data contracts
├── tests/                             # 10 automated unit and regression tests
└── research_evidence/                 # Reproducibility artifacts & test logs
    ├── evaluation_results/            # benchmark_report.json
    ├── test_reports/                  # pytest_execution_log.txt
    ├── architecture/                  # architecture_spec.md
    └── reproducibility/               # REPRODUCIBILITY.md
```

---

## 7. Streamlit Community Cloud Deployment

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an application linked to `streamlit_app.py`.
3. (Optional) In **App Settings $\rightarrow$ Secrets**, define:
   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-4o-mini"
   ```
4. Deploy. The application initializes its local vector store and SQLite memory automatically.

---

## 8. Disclaimer

LegalEagle was developed strictly as an educational and academic research system. It does not provide legal advice, does not establish an attorney-client relationship, and should not be relied upon as a substitute for professional legal review by qualified counsel.
