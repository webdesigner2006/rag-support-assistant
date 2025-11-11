from __future__ import annotations

from typing import List

import numpy as np
from rag_support.api.v1.models import IngestItem
from rag_support.config import settings
from rag_support.logging import logger
from rag_support.utils import stable_id
from .vertex import VertexClient
from .pinecone_store import PineconeStore
from .bm25_index import BM25Index


class IngestionService:
    """
    Simple ingestion: users provide full text payloads (or URLs they've pre-extracted).
    We chunk into ~1-2K tokens (here we approximate by chars) with overlap, create embeddings, and upsert.
    """

    def __init__(self, cfg=settings, bm25_index: BM25Index | None = None):
        self.cfg = cfg
        self.vertex = VertexClient()
        self.pinecone = PineconeStore()
        # For demo simplicity we use a process-wide singleton-like instance.
        # In production, use a persistent keyword index (e.g., Whoosh or ES).
        self.bm25 = bm25_index or _GLOBAL_BM25_INDEX

    async def ingest_items(self, items: List[IngestItem]) -> None:
        chunks: list[tuple[str, str, dict]] = []
        for it in items:
            for i, chunk in enumerate(self._chunk_text(it.text)):
                chunk_id = f"{stable_id(it.source)}-{i:04d}"
                meta = {
                    "source_id": stable_id(it.source),
                    "chunk_id": chunk_id,
                    "title": it.title or "",
                    "url": it.url or "",
                    "tags": it.tags or [],
                }
                chunks.append((chunk_id, chunk, meta))

        texts = [c[1] for c in chunks]
        vecs = self.vertex.embed(texts)
        vectors = []
        for (chunk_id, _, meta), vec in zip(chunks, vecs):
            vectors.append((chunk_id, vec.tolist(), meta))
        self.pinecone.upsert(vectors)

        # BM25 keyword index
        doc_ids = [c[0] for c in chunks]
        self.bm25.add_docs(doc_ids, texts)
        logger.info("ingest_complete", extra={"chunks": len(chunks)})

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 5500, overlap: int = 500):
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + max_chars)
            yield text[start:end]
            if end == n:
                break
            start = end - overlap


# Global in-memory BM25 index instance
_GLOBAL_BM25_INDEX = BM25Index()
