# LegalEagle Project Handoff

This file captures the current state of the LegalEagle project so future work can continue with a different LLM, engineer, or coding session without needing the original chat history.

## Project Goal

LegalEagle is an educational portfolio project: a contract risk detection agent that ingests contracts, identifies risky clauses, retrieves market-standard context, produces a plain-English risk scorecard, suggests redlines, and supports contract Q&A.

It is not legal advice. It is designed to demonstrate applied AI engineering skills:

- agent workflows from first principles
- RAG
- structured outputs
- memory across sessions
- CUAD-style clause classification
- eval discipline
- deployable Streamlit demo

## Current Status

The app is a Python + Streamlit project. It has moved beyond the initial v1 scaffold and now includes early v2 core-feature work.

Verified most recently:

```powershell
python -m pytest
# 8 passed

python -m compileall streamlit_app.py src tests
# passed
```

A backend smoke test on `data/sample_contract.txt` returned:

```text
risk_level: critical
overall_risk_score: 75
findings: 5
category_scores present: yes
redlines present: yes
agent_trace present: yes
```

## How To Run

From the repo root:

```powershell
cd C:\Users\Krish\Documents\LegalEagle
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python -m streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

Use `data/sample_contract.txt` as the first test contract.

The app works without `OPENAI_API_KEY` using deterministic local fallbacks. Add `OPENAI_API_KEY` to `.env` to enable OpenAI-powered structured review and embeddings.

## Important Environment Variables

Defined in `.env.example`:

```text
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
LEGALEAGLE_DB_PATH=data/memory/legaleagle.sqlite3
LEGALEAGLE_VECTOR_PATH=data/vector_store
LEGALEAGLE_VECTOR_BACKEND=local
LEGALEAGLE_QDRANT_COLLECTION=legaleagle_clauses
LEGALEAGLE_LANGSMITH_ENABLED=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=legaleagle
```

Default vector backend is `local` so development starts quickly. Qdrant support exists but should be enabled deliberately with:

```text
LEGALEAGLE_VECTOR_BACKEND=qdrant
```

## Current User Flows

### Analyze Contract

In `streamlit_app.py`, the **Analyze** tab supports:

- uploading PDF, DOCX, or TXT
- pasting raw contract text
- optional reviewer preferences
- optional party/counterparty memory key
- risk scorecard generation
- JSON export
- findings CSV export
- feedback storage

### Ask Contract

The **Ask** tab supports a conversational agent flow:

- asks questions like `What happens if I terminate early?`
- retrieves relevant clauses
- classifies relevant clause text
- searches market-standard precedent context
- computes risk score
- generates balanced redline language
- stores chat messages in SQLite

### Memory

The **Memory** tab shows:

- recent reviewed contracts
- feedback history

### Evals

The **Evals** tab runs a small demo eval suite and lists intended benchmark categories:

- clause classification precision / recall / F1
- retrieval hit@k
- structured output validity
- end-to-end risk finding recall

## Architecture

```text
streamlit_app.py
  Streamlit UI and user workflows.

src/schema.py
  Pydantic domain models:
  ParsedContract, ContractChunk, RiskFinding, RiskScorecard,
  RedlineSuggestion, MarketStandard, ContractQuestionAnswer.

src/config.py
  Environment-driven app configuration.

src/ingest/
  parser.py:
    PDF parsing with PyMuPDF, DOCX parsing with python-docx, TXT/pasted text parsing.
  chunker.py:
    Clause-aware chunking using regex first, optional spaCy sentence grouping, then window fallback.

src/models/
  labels.py:
    Focused CUAD labels and missing-clause importance.
  risk_classifier.py:
    Keyword fallback classifier plus optional fine-tuned transformer pipeline loading.

src/retrieval/
  embeddings.py:
    OpenAI embeddings, sentence-transformers fallback, deterministic hash fallback.
  vector_store.py:
    Local TF-IDF retrieval by default, optional Chroma/Qdrant support.
  playbook.py:
    Loads legal risk playbook YAML.

src/risk/
  scoring.py:
    Per-clause score, category mapping, category scores, overall score.
  redline.py:
    Balanced alternate language generation.
  triggers.py:
    Trigger phrase detection.

src/agent/
  reviewer.py:
    Main contract analysis pipeline.
  conversation.py:
    First-principles tool-loop Q&A agent.
  prompts.py:
    OpenAI review prompt helpers.

src/memory/
  store.py:
    SQLite storage for contracts, feedback, preferences, party memory, and chat messages.

src/observability/
  tracing.py:
    Local trace recorder plus optional LangSmith-compatible decorator.

src/evals/
  metrics.py:
    Classification, retrieval, and span metrics.
  runner.py:
    Demo eval runner.

notebooks/
  01_cuad_preprocessing.ipynb
  02_legalbert_finetuning.ipynb
  03_benchmark_eval_suite.ipynb
```

## Current Risk Scorecard Shape

`RiskScorecard` includes:

- `contract_summary`
- `overall_risk_score`
- `risk_level`
- `category_scores`
- `findings`
- `missing_clauses`
- `reviewer_notes`
- `agent_trace`

Each `RiskFinding` includes:

- clause type
- quote
- page and offsets
- section path / clause metadata
- risk level
- risk category
- numeric risk score
- triggering language
- rationale
- business impact
- negotiation note
- redline suggestion
- market-standard context
- retrieved evidence IDs

## Data Files

```text
data/sample_contract.txt
  Sample test contract with risky termination, liability, confidentiality,
  governing law, and assignment clauses.

data/risk_playbook.yml
  Local risk guidance and market-standard precedent snippets.

data/cuad_focus_labels.json
  Focused CUAD-style label list.
```

Current focused labels include:

- termination
- assignment
- ip assignment
- indemnification
- limitation of liability
- confidentiality
- exclusivity
- non-compete / non-solicit
- governing law
- change of control
- most favored nation
- audit rights
- revenue / payment obligations

## Fine-Tuning Path

Fine-tuning is intended to happen in Google Colab, not inside the Streamlit app.

Use:

```text
notebooks/01_cuad_preprocessing.ipynb
notebooks/02_legalbert_finetuning.ipynb
notebooks/03_benchmark_eval_suite.ipynb
requirements-colab.txt
```

The app looks for an exported fine-tuned model under:

```text
models/fine_tuned/
```

If no model is present, it uses `KeywordRiskClassifier`.

## Testing Commands

Run all tests:

```powershell
python -m pytest
```

Compile/import syntax check:

```powershell
python -m compileall streamlit_app.py src tests
```

Backend smoke test:

```powershell
python -c "from pathlib import Path; from src.config import AppConfig; from src.ingest.parser import parse_contract; from src.memory.store import MemoryStore; from src.retrieval.vector_store import LocalRetriever; from src.agent.reviewer import ContractRiskAgent; cfg=AppConfig.from_env(); parsed=parse_contract(Path('data/sample_contract.txt')); agent=ContractRiskAgent(cfg, LocalRetriever(cfg.vector_path), MemoryStore(cfg.db_path)); score=agent.analyze(parsed); print(score.risk_level, score.overall_risk_score, len(score.findings)); print(score.category_scores)"
```

## Known Design Decisions

- The project prioritizes a working local demo over production complexity.
- OpenAI usage is optional; offline fallbacks keep the app demoable without API keys.
- `LEGALEAGLE_VECTOR_BACKEND=local` is the default to avoid slow startup and dependency friction.
- Qdrant is supported as an upgrade path, but should be tested carefully before making it the default.
- LangSmith tracing is represented through local trace records and an optional decorator path; full LangSmith project integration still needs validation with real credentials.
- The current redline generator is deterministic template-based. This is good for reliability, but future versions can add LLM-based redline generation.

## Recently Fixed Issue

The app previously crashed with:

```text
PydanticUserError: Decorators defined with incorrect fields:
src.schema.ContractQuestionAnswer.sort_findings
```

Cause:

`sort_findings` was accidentally indented under `ContractQuestionAnswer` instead of `RiskScorecard`.

Fix:

The validator now belongs to `RiskScorecard`, and tests pass.

## Suggested Next Work

### Core Features

- Add real Qdrant integration tests using a temporary local Qdrant path.
- Add actual transformer classifier adapter tests with a mocked HuggingFace pipeline.
- Improve clause boundary detection with richer legal regexes and spaCy noun/section cues.
- Add true LLM redline generation with schema validation and deterministic fallback.
- Add separate precedent corpus ingestion for legal commentary and standard clauses.

### UI/UX

- Redesign Streamlit layout for a professional contract-review workflow.
- Add side-by-side contract viewer and findings panel.
- Highlight exact triggering language in context.
- Add category score visual cards.
- Add exportable PDF/HTML report.

### Architecture

- Split Streamlit UI from backend service layer.
- Add dependency-injected interfaces for retrieval, embeddings, classifier, and LLM.
- Add persistent migration handling for SQLite schema changes.
- Add structured logging.
- Add config profiles: local, demo, cloud, eval.

### Evals And Differentiators

- Build real CUAD eval loader.
- Add retrieval hit@k using labeled clause IDs.
- Add redline quality rubric.
- Add hallucination/citation faithfulness evals.
- Add baseline comparison:
  - zero-shot OpenAI
  - RAG-only
  - classifier + RAG
  - classifier + RAG + memory

## Guidance For Future LLMs

Before editing:

1. Read this file.
2. Run `git status --short`.
3. Run `python -m pytest`.
4. Inspect the relevant modules before patching.

When implementing:

- Keep OpenAI, Qdrant, sentence-transformers, spaCy, and LangSmith optional.
- Preserve offline local fallback behavior.
- Do not break `data/sample_contract.txt` as a smoke-test path.
- Add tests for every new backend behavior.
- Avoid turning the Streamlit app into a large monolith; move logic into `src/`.

When done:

1. Run `python -m pytest`.
2. Run `python -m compileall streamlit_app.py src tests`.
3. Run the backend smoke test above.
4. Update this handoff file if architecture or commands changed.
