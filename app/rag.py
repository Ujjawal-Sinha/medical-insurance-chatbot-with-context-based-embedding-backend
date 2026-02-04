# app/rag.py

from typing import List, Tuple

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

from .models import PolicyChunk
from .vector_store import VectorStore
from .debug_dump import dump_chunks_with_embeddings



class RAGPipeline:
    def __init__(self, embed_model_name: str, llm_model: str) -> None:
        self.embedder = SentenceTransformer(embed_model_name)
        self.llm_model = llm_model
        self.vector_store: VectorStore = VectorStore(self.embedder.get_sentence_embedding_dimension())
        self.ready = False

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def build_index(self, chunks: List[PolicyChunk]) -> None:
        texts = [c.text for c in chunks if c.text.strip()]
        valid_chunks = [c for c in chunks if c.text.strip()]
        if not texts:
            self.ready = False
            return
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        embeddings = self._normalize(np.array(embeddings))

        dump_chunks_with_embeddings(valid_chunks, embeddings)
        
        self.vector_store = VectorStore(self.embedder.get_sentence_embedding_dimension())
        self.vector_store.add(embeddings, valid_chunks)
        self.ready = True

    def retrieve(self, question: str, top_k: int = 5) -> List[Tuple[PolicyChunk, float]]:
        if not self.ready:
            return []
        query_vec = self.embedder.encode([question], show_progress_bar=False)
        query_vec = self._normalize(np.array(query_vec))
        return self.vector_store.search(query_vec, top_k=top_k)

    def answer(self, question: str, retrieved: List[Tuple[PolicyChunk, float]]) -> str:
        if not retrieved:
            return "I couldn't find the answer in the provided policy context."

        context_lines: List[str] = []
        for chunk, score in retrieved:
            meta = f"[Section: {chunk.section} | Clause: {chunk.clause or 'N/A'} | Page: {chunk.page} | Type: {chunk.content_type}]"
            context_lines.append(meta)
            context_lines.append(chunk.text)
            context_lines.append("")
        context = "\n".join(context_lines).strip()

        system_prompt = (
            "You are a policy-aware assistant for medical insurance documents. "
            "Answer only using the provided context. "
            "If the answer is not in the context, say you cannot find it. "
            "Cite section and page numbers in your answer."
        )
        user_prompt = (
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "Answer using only the context. Include citations like (Section: ..., Page: ...)."
        )

        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.1},
        )
        return response["message"]["content"].strip()
