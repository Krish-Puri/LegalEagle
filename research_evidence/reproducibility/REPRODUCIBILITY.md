# LegalEagle Reproducibility Protocol

This document provides exact instructions to reproduce all experimental metrics, regression benchmarks, and unit tests reported for LegalEagle.

---

## 1. Environment Specifications

- **Operating System:** Windows 10/11 or Linux (Ubuntu 22.04+)
- **Python Version:** 3.12.x
- **Key Dependencies:**
  - `numpy`: 1.26.4
  - `pandas`: 2.3.3 (`pandas < 3.0`)
  - `scikit-learn`: 1.5.0
  - `pyarrow`: 17.0.0 (`pyarrow < 18.0`)
  - `streamlit`: >= 1.36.0
  - `PyMuPDF`: >= 1.24.7
  - `python-docx`: >= 1.1.2
  - `pydantic`: >= 2.7.0

---

## 2. Benchmark Dataset (`data/eval_dataset.json`)

The academic evaluation benchmark consists of 20 annotated contract clauses representing real-world and synthetic edge cases across all 13 focused CUAD categories:

- **Random Seed:** 42
- **Split:** Complete Test Set (no training overlap; zero-shot pattern & lexical evaluation)
- **Categories Tested:**
  1. `termination`
  2. `assignment`
  3. `ip assignment`
  4. `indemnification`
  5. `limitation of liability`
  6. `confidentiality`
  7. `exclusivity`
  8. `non-compete / non-solicit`
  9. `governing law`
  10. `change of control`
  11. `most favored nation`
  12. `audit rights`
  13. `revenue / payment obligations`

---

## 3. Commands to Reproduce Benchmark Results

To execute the automated benchmark runner and regenerate `research_evidence/evaluation_results/benchmark_report.json`:

```powershell
# In PowerShell with virtual environment activated:
python -c "
from src.agent.reviewer import ContractRiskAgent
from src.config import AppConfig
from src.evals.runner import run_comprehensive_evaluation
from src.memory.store import MemoryStore
from src.retrieval.vector_store import LocalRetriever

config = AppConfig.from_env()
retriever = LocalRetriever()
agent = ContractRiskAgent(config, retriever, MemoryStore(config.db_path))
report = run_comprehensive_evaluation(agent)
print('Evaluation Complete! Macro-F1:', report['classification_metrics']['macro_f1'])
print('Retrieval Hit@3:', report['retrieval_metrics']['hit@3'])
print('Finding Recall:', report['end_to_end_review']['finding_recall'])
"
```

Expected output:
```text
Evaluation Complete! Macro-F1: 1.0
Retrieval Hit@3: 0.9474
Finding Recall: 1.0
```

---

## 4. Commands to Run Unit & Regression Tests

Execute all 10 automated test suites:

```powershell
python -m pytest -v
```

Verification suite breakdown:
- `tests/test_agent.py`: Offline reviewer functionality
- `tests/test_chunker.py`: Character offset preservation and boundary chunking
- `tests/test_eval_suite.py`: Automated assertion of benchmark metrics
- `tests/test_memory.py`: SQLite persistence and feedback context
- `tests/test_metrics.py`: Classification, Hit@K, and MRR calculations
- `tests/test_regression_msa.py`: Comprehensive regression tests on Master Services Agreement fixture
- `tests/test_schema.py`: Pydantic schema validation and score bounds

---

## 5. Running the Application Locally

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run Streamlit dashboard
python -m streamlit run streamlit_app.py
```

The application will be accessible at `http://localhost:8501`. Test contract files are located in `data/demo_evaluation_contract.txt`, `data/demo_evaluation_contract.pdf`, and `data/demo_evaluation_contract.docx`.

