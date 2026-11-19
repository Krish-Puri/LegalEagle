from __future__ import annotations

import math
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import AppConfig
from src.retrieval.embeddings import EmbeddingProvider
from src.retrieval.playbook import load_playbook
from src.schema import ContractChunk, RetrievalEvidence, SourceType


class LocalRetriever:
    def __init__(self, vector_path: Path | None = None, config: AppConfig | None = None):
        self.config = config or AppConfig.from_env()
        self.vector_path = vector_path or Path("data/vector_store")
        self.documents: list[RetrievalEvidence] = []
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = None
        self._embedding_provider = EmbeddingProvider(self.config)
        self._qdrant_client = self._init_qdrant()
        self._chroma_collection = self._init_chroma()
        self.seed_from_disk()

    def seed_from_disk(self) -> None:
        entries = load_playbook(self.config.playbook_path)
        evidence = [
            RetrievalEvidence(
                id=entry["id"],
                source=entry.get("source", "risk_playbook"),
                source_type=entry.get("source_type", "playbook"),
                document_title=entry.get("document_title", entry.get("id", "Playbook Precedent")),
                clause_category=entry.get("clause_category", entry.get("clause_type", "general")),
                source_location=entry.get("source_location", "data/risk_playbook.yml"),
                text=entry["text"],
                score=1.0,
                similarity_score=1.0,
                reason_retrieved="Pre-indexed playbook guidance / market standard precedent",
                metadata={k: str(v) for k, v in entry.items() if k not in {"id", "text"}},
            )
            for entry in entries
        ]
        self.upsert(evidence)

    def upsert(self, evidence: list[RetrievalEvidence]) -> None:
        known = {item.id for item in self.documents}
        new_items = [item for item in evidence if item.id not in known]
        self.documents.extend(new_items)
        self._upsert_qdrant(new_items)
        self._upsert_chroma(new_items)
        self._reindex()

    def add_contract_chunks(self, chunks: list[ContractChunk]) -> None:
        """Index contract chunks as uploaded_contract source type."""
        evidence = [
            RetrievalEvidence(
                id=f"contract:{chunk.id}",
                source=chunk.source_name,
                source_type="uploaded_contract",
                document_title=f"Contract: {chunk.source_name}",
                clause_category=chunk.clause_title or "Contract Clause",
                source_location=f"Page {chunk.page}, Offsets {chunk.start_offset}-{chunk.end_offset}",
                text=chunk.text,
                score=1.0,
                similarity_score=1.0,
                reason_retrieved=f"Indexed chunk from uploaded contract {chunk.source_name}",
                metadata={
                    "page": str(chunk.page),
                    "contract_id": chunk.contract_id,
                    "clause_title": chunk.clause_title or "",
                    "clause_number": chunk.clause_number or "",
                    "source_type": "uploaded_contract",
                },
            )
            for chunk in chunks
        ]
        self.upsert(evidence)

    def search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, str] | None = None,
        source_types: list[SourceType] | None = None,
        clause_type: str | None = None,
    ) -> list[RetrievalEvidence]:
        """Clause-aware, source-type filtered semantic and lexical search."""
        if not self.documents:
            return []

        # 1. Candidate pre-filtering
        candidate_indexes: list[int] = []
        for idx, doc in enumerate(self.documents):
            # Check source type constraint (e.g., exclude uploaded contract chunks)
            if source_types and doc.source_type not in source_types:
                continue
            # Check arbitrary metadata filters
            if filters and not all(doc.metadata.get(k) == v for k, v in filters.items()):
                continue
            candidate_indexes.append(idx)

        if not candidate_indexes:
            return []

        # Check Qdrant or Chroma if configured and active
        qdrant_results = self._search_qdrant(query, k, filters)
        if qdrant_results:
            return qdrant_results

        # Fallback to TF-IDF cosine similarity over candidates
        if self._matrix is None:
            return [self.documents[idx] for idx in candidate_indexes[:k]]

        query_vector = self._vectorizer.transform([query])
        candidate_matrix = self._matrix[candidate_indexes]
        similarities = cosine_similarity(query_vector, candidate_matrix).flatten()

        # Score boosting if clause_type matches candidate category
        boosted_scores: list[tuple[int, float]] = []
        for local_idx, sim in enumerate(similarities):
            doc_idx = candidate_indexes[local_idx]
            doc = self.documents[doc_idx]
            boost = 1.0
            if clause_type:
                target = clause_type.lower()
                doc_cat = (doc.clause_category or "").lower()
                doc_text = doc.text.lower()
                if target in doc_cat or target in doc.metadata.get("clause_type", "").lower():
                    boost = 1.35
                elif target in doc_text:
                    boost = 1.15
            boosted_scores.append((doc_idx, float(sim * boost)))

        ranked = sorted(boosted_scores, key=lambda item: item[1], reverse=True)[:k]

        results: list[RetrievalEvidence] = []
        for doc_idx, sim_score in ranked:
            if sim_score <= 0.001 and len(results) >= 1:
                continue
            doc = self.documents[doc_idx]
            clean_score = _safe_score(sim_score)
            results.append(
                doc.model_copy(
                    update={
                        "score": clean_score,
                        "similarity_score": clean_score,
                        "reason_retrieved": f"Matched query terms with similarity score {clean_score:.2f}."
                    }
                )
            )
        return results

    def _reindex(self) -> None:
        if not self.documents:
            self._matrix = None
            return
        self._matrix = self._vectorizer.fit_transform([document.text for document in self.documents])

    def _init_chroma(self):
        if self.config.vector_backend != "chroma":
            return None
        try:
            import chromadb

            self.vector_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.vector_path))
            return client.get_or_create_collection("legaleagle_memory")
        except Exception:
            return None

    def _init_qdrant(self):
        if self.config.vector_backend != "qdrant":
            return None
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self.vector_path.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=str(self.vector_path / "qdrant"))
            existing = {collection.name for collection in client.get_collections().collections}
            if self.config.qdrant_collection not in existing:
                client.create_collection(
                    collection_name=self.config.qdrant_collection,
                    vectors_config=VectorParams(size=self._embedding_provider.dimensions, distance=Distance.COSINE),
                )
            return client
        except Exception:
            return None

    def _upsert_chroma(self, evidence: list[RetrievalEvidence]) -> None:
        if not self._chroma_collection or not evidence:
            return
        try:
            self._chroma_collection.upsert(
                ids=[item.id for item in evidence],
                documents=[item.text for item in evidence],
                metadatas=[{"source": item.source, "source_type": item.source_type, **item.metadata} for item in evidence],
            )
        except Exception:
            return

    def _upsert_qdrant(self, evidence: list[RetrievalEvidence]) -> None:
        if not self._qdrant_client or not evidence:
            return
        try:
            from qdrant_client.models import PointStruct

            vectors = self._embedding_provider.embed_texts([item.text for item in evidence])
            if not vectors:
                return
            points = [
                PointStruct(
                    id=_stable_qdrant_id(item.id),
                    vector=vector,
                    payload={
                        "logical_id": item.id,
                        "source": item.source,
                        "source_type": item.source_type,
                        "document_title": item.document_title,
                        "clause_category": item.clause_category,
                        "text": item.text,
                        **item.metadata,
                    },
                )
                for item, vector in zip(evidence, vectors)
            ]
            self._qdrant_client.upsert(collection_name=self.config.qdrant_collection, points=points)
        except Exception:
            return

    def _search_chroma(self, query: str, k: int, filters: dict[str, str] | None) -> list[RetrievalEvidence]:
        if not self._chroma_collection:
            return []
        try:
            result = self._chroma_collection.query(
                query_texts=[query],
                n_results=k,
                where=filters or None,
            )
        except Exception:
            return []

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0] if result.get("distances") else [0.0] * len(ids)
        evidence: list[RetrievalEvidence] = []
        for item_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            source = str(metadata.pop("source", "chroma"))
            source_type = str(metadata.pop("source_type", "playbook"))
            score = 1 / (1 + float(distance or 0))
            evidence.append(
                RetrievalEvidence(
                    id=item_id,
                    source=source,
                    source_type=source_type,  # type: ignore[arg-type]
                    document_title=metadata.get("document_title", item_id),
                    clause_category=metadata.get("clause_category", "general"),
                    text=document or "",
                    score=score,
                    similarity_score=score,
                    metadata={str(key): str(value) for key, value in metadata.items()},
                )
            )
        return evidence

    def _search_qdrant(self, query: str, k: int, filters: dict[str, str] | None) -> list[RetrievalEvidence]:
        if not self._qdrant_client:
            return []
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_vector = self._embedding_provider.embed_query(query)
            query_filter = None
            if filters:
                query_filter = Filter(
                    must=[FieldCondition(key=key, match=MatchValue(value=value)) for key, value in filters.items()]
                )
            points = self._qdrant_client.search(
                collection_name=self.config.qdrant_collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=k,
            )
        except Exception:
            return []

        evidence: list[RetrievalEvidence] = []
        for point in points:
            payload = point.payload or {}
            logical_id = str(payload.get("logical_id", point.id))
            source = str(payload.get("source", "qdrant"))
            source_type = str(payload.get("source_type", "playbook"))
            text = str(payload.get("text", ""))
            metadata = {
                str(key): str(value)
                for key, value in payload.items()
                if key not in {"logical_id", "source", "source_type", "text"}
            }
            score = float(point.score or 0)
            evidence.append(
                RetrievalEvidence(
                    id=logical_id,
                    source=source,
                    source_type=source_type,  # type: ignore[arg-type]
                    document_title=str(payload.get("document_title", logical_id)),
                    clause_category=str(payload.get("clause_category", "general")),
                    text=text,
                    score=score,
                    similarity_score=score,
                    metadata=metadata,
                )
            )
        return evidence


def _safe_score(value: float) -> float:
    if math.isnan(value) or value < 0:
        return 0.0
    return round(float(value), 3)


def _stable_qdrant_id(value: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))
