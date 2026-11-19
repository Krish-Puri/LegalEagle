from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from src.config import AppConfig


@dataclass
class EmbeddingProvider:
    config: AppConfig
    dimensions: int = 384

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        openai_vectors = self._openai_embeddings(texts)
        if openai_vectors:
            self.dimensions = len(openai_vectors[0])
            return openai_vectors
        sentence_vectors = self._sentence_transformer_embeddings(texts)
        if sentence_vectors:
            self.dimensions = len(sentence_vectors[0])
            return sentence_vectors
        return [_hash_embedding(text, self.dimensions) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _openai_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not self.config.openai_api_key:
            return []
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.config.openai_api_key)
            response = client.embeddings.create(model=self.config.openai_embedding_model, input=texts)
            return [item.embedding for item in response.data]
        except Exception:
            return []

    def _sentence_transformer_embeddings(self, texts: list[str]) -> list[list[float]]:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            vectors = model.encode(texts, normalize_embeddings=True)
            return [vector.astype(float).tolist() for vector in vectors]
        except Exception:
            return []


def _hash_embedding(text: str, dimensions: int) -> list[float]:
    vector = np.zeros(dimensions, dtype=np.float32)
    tokens = [token.lower() for token in text.split() if token.strip()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1 if digest[4] % 2 == 0 else -1
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm:
        vector = vector / norm
    return vector.astype(float).tolist()
