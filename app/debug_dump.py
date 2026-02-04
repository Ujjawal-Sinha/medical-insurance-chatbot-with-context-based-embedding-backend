import json
from typing import List

import numpy as np

from .models import PolicyChunk


def dump_chunks_with_embeddings(
    chunks: List[PolicyChunk],
    embeddings: np.ndarray,
    output_path: str = "debug_chunks_embeddings.json",
) -> None:
    data = []

    for chunk, emb in zip(chunks, embeddings):
        data.append({
            "text": chunk.text,
            "section": chunk.section,
            "clause": chunk.clause,
            "page": chunk.page,
            "content_type": chunk.content_type,
            "embedding_dim": len(emb),
            "embedding": emb.tolist(),  # important: JSON serializable
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
