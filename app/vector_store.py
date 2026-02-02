# app/vector_store.py

from typing import List, Tuple

import faiss
import numpy as np

from .models import PolicyChunk


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[PolicyChunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[PolicyChunk]) -> None:
        if embeddings.size == 0 or not chunks:
            return
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings count does not match chunks")
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[PolicyChunk, float]]:
        if self.index.ntotal == 0:
            return []
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        results: List[Tuple[PolicyChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append((self.chunks[idx], float(score)))
        return results
