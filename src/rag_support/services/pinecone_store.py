from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from pinecone import Pinecone, ServerlessSpec
from rag_support.config import settings
from rag_support.logging import logger


class PineconeStore:
    def __init__(self) -> None:
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index
        self._dim = settings.pinecone_dim
        if self._index_name not in [i.name for i in self._pc.list_indexes()]:
            logger.info("creating_pinecone_index", extra={"index": self._index_name})
            self._pc.create_index(
                name=self._index_name,
                dimension=self._dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=settings.pinecone_env),
            )
        self._index = self._pc.Index(self._index_name)

    def upsert(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        self._index.upsert(vectors=vectors)

    def query(
        self, vector: np.ndarray, top_k: int, metadata_filter: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        res = self._index.query(
            vector=vector.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter or {},
        )
        out: List[Dict[str, Any]] = []
        for match in res.get("matches", []):
            out.append(
                {
                    "id": match["id"],
                    "score": match["score"],
                    "metadata": match.get("metadata", {}),
                }
            )
        return out
