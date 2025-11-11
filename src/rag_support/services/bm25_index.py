from __future__ import annotations

from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi


class BM25Index:
    """
    In-memory BM25 index (OK for demos and unit tests).
    Keep doc_id alignment with Pinecone doc ids.
    """

    def __init__(self) -> None:
        self._tokenized: List[List[str]] = []
        self._doc_ids: List[str] = []
        self._bm25: BM25Okapi | None = None

    def add_docs(self, doc_ids: List[str], docs: List[str]) -> None:
        toks = [self._tokenize(d) for d in docs]
        self._tokenized.extend(toks)
        self._doc_ids.extend(doc_ids)
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self._bm25 or not self._doc_ids:
            return []
        toks = self._tokenize(query)
        scores = self._bm25.get_scores(toks)
        pairs = list(zip(self._doc_ids, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in text.lower().split() if t.isascii()]
