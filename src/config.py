from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None
    openai_model: str
    db_path: Path
    vector_path: Path
    playbook_path: Path
    labels_path: Path
    classifier_path: Path
    openai_embedding_model: str = "text-embedding-3-small"
    vector_backend: str = "local"
    qdrant_collection: str = "legaleagle_clauses"
    langsmith_enabled: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            db_path=Path(os.getenv("LEGALEAGLE_DB_PATH", "data/memory/legaleagle.sqlite3")),
            vector_path=Path(os.getenv("LEGALEAGLE_VECTOR_PATH", "data/vector_store")),
            vector_backend=os.getenv("LEGALEAGLE_VECTOR_BACKEND", "local").lower(),
            qdrant_collection=os.getenv("LEGALEAGLE_QDRANT_COLLECTION", "legaleagle_clauses"),
            langsmith_enabled=os.getenv("LEGALEAGLE_LANGSMITH_ENABLED", "false").lower() == "true",
            playbook_path=Path(os.getenv("LEGALEAGLE_PLAYBOOK_PATH", "data/risk_playbook.yml")),
            labels_path=Path(os.getenv("LEGALEAGLE_LABELS_PATH", "data/cuad_focus_labels.json")),
            classifier_path=Path(os.getenv("LEGALEAGLE_CLASSIFIER_PATH", "models/fine_tuned")),
        )
