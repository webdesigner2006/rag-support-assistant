from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from rag_support.config import settings
from rag_support.logging import logger
from rag_support.utils import rrf_merge, normalize_list, cosine_similarity
from .pinecone_store import PineconeStore
from .vertex import VertexClient
from .ingestion import _GLOBAL_BM25_INDEX


@dataclass
class RetrievedDoc:
    id: str
    text: str
    title: str
    url: str
    source_id: str
    chunk_id: str
    semantic_score: float
    keyword_score: float


class RetrievalService:
    def __init__(self, cfg=settings):
        self.cfg = cfg
        self.vertex = VertexClient()
        self.store = PineconeStore()
        self.bm25 = _GLOBAL_BM25_INDEX

    async def hybrid_search(
        self, query: str, alpha: float, top_k: int, metadata_filters: Dict[str, Any]
    ) -> List[RetrievedDoc]:
        qv = self.vertex.embed([query])[0]

        # semantic (pinecone)
        sem = self.store.query(qv, top_k=self.cfg.semantic_top_k, metadata_filter=metadata_filters)
        sem_ids = [m["id"] for m in sem]

        # keyword (bm25)
        kw_pairs = self.bm25.search(query, top_k=self.cfg.bm25_top_k)
        kw_ids = [doc_id for doc_id, _ in kw_pairs]
        kw_dict = {doc_id: score for doc_id, score in kw_pairs}

        fused = rrf_merge(sem_ids, kw_ids)
        ranked_ids = [doc_id for doc_id, _ in fused][: top_k * 2]  # widen before later pruning

        # Build results with text from Pinecone metadata; if not available, reconstruct basic.
        # Here we rely on Pinecone metadata to include text; if not, consider adding it on upsert.
        results: List[RetrievedDoc] = []
        for doc_id in ranked_ids:
            sem1 = self.store.query(qv, top_k=1, metadata_filter={"chunk_id": {"$eq": doc_id}})
            if sem1:
                md = sem1[0]["metadata"]
                text = md.get("text", "")
                results.append(
                    RetrievedDoc(
                        id=doc_id,
                        text=text,
                        title=md.get("title", ""),
                        url=md.get("url", ""),
                        source_id=md.get("source_id", ""),
                        chunk_id=md.get("chunk_id", doc_id),
                        semantic_score=next((m["score"] for m in sem if m["id"] == doc_id), 0.0),
                        keyword_score=kw_dict.get(doc_id, 0.0),
                    )
                )

        # score fusion (alpha between semantic and keyword)
        if not results:
            return []
        sem_scores = [r.semantic_score for r in results]
        kw_scores = [r.keyword_score for r in results]
        sem_norm = normalize_list(sem_scores)
        kw_norm = normalize_list(kw_scores)

        fused_scores = [(alpha * s + (1 - alpha) * k) for s, k in zip(sem_norm, kw_norm)]
        pairs = list(zip(results, fused_scores))
        pairs.sort(key=lambda x: x[1], reverse=True)

        top = [p[0] for p in pairs][:top_k]
        logger.info("hybrid_search", extra={"top_k": len(top)})
        return top

    def validator(
        self, query_vec: np.ndarray, docs: List[RetrievedDoc], w1=0.6, w2=0.25, w3=0.15
    ) -> List[Dict[str, Any]]:
        """
        Compute confidence per chunk: normalized cosine + normalized reciprocal rank + source prior.
        """
        if not docs:
            return []
        cosines = []
        for d in docs:
            dv = self._embed_text(d.text) if d.text else np.zeros_like(query_vec)
            cos = float(cosine_similarity(query_vec, dv)) if dv.any() else 0.0
            cosines.append(cos)

        ranks = list(range(1, len(docs) + 1))
        rank_norm = [1.0 / r for r in ranks]  # larger for top
        rank_norm = [x / max(rank_norm) for x in rank_norm]  # normalize to [0,1]
        cos_norm = normalize_list(cosines)
        priors = [0.8 if d.url.startswith("http") else 0.5 for d in docs]

        conf = [w1 * c + w2 * r + w3 * p for c, r, p in zip(cos_norm, rank_norm, priors)]

        out = []
        for d, c, cn, rn, sp in zip(docs, conf, cos_norm, rank_norm, priors):
            out.append(
                {
                    "source_id": d.source_id,
                    "chunk_id": d.chunk_id,
                    "text": d.text,
                    "confidence": c,
                    "breakdown": {"cosine": cn, "rank_norm": rn, "source_prior": sp, "confidence": c},
                    "title": d.title,
                    "url": d.url,
                }
            )
        return out

    def _embed_text(self, text: str) -> np.ndarray:
        return self.vertex.embed([text])[0]
