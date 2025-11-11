from __future__ import annotations

import os
import pytest

os.environ.setdefault("GOOGLE_PROJECT_ID", "dummy")
os.environ.setdefault("GOOGLE_LOCATION", "us-central1")
os.environ.setdefault("VERTEX_MODEL_ID", "gemini-1.5-pro")
os.environ.setdefault("VERTEX_EMBED_MODEL_ID", "text-embedding-004")
os.environ.setdefault("PINECONE_API_KEY", "fake")
os.environ.setdefault("PINECONE_ENV", "us-east-1")
os.environ.setdefault("PINECONE_INDEX", "test-index")
os.environ.setdefault("PINECONE_DIM", "8")

@pytest.fixture(autouse=True)
def no_vertex_calls(monkeypatch):
    import numpy as np

    # Mock embeddings: map each text to a fixed small vector based on length
    class MockVertex:
        def __init__(self): ...
        def embed(self, texts):
            return [np.ones(8) * (len(t) % 7 + 1) for t in texts]
        def generate(self, prompt, temperature, max_tokens):
            return type("R", (), {"text": "Mock answer [#source-1].", "usage": {"prompt_tokens": 5, "candidates_tokens": 10}})
        def judge(self, system_prompt, message_json, temperature=0.0):
            return {"verdict": "pass", "scores": {"groundedness": 5, "relevance": 5, "completeness": 4, "clarity": 5},
                    "flags": {"leakage_risk": False, "toxicity": False, "policy_violation": False}, "reasons": [], "suggested_fixes": []}

    from rag_support.services import vertex as vertex_mod
    monkeypatch.setattr(vertex_mod, "VertexClient", MockVertex)

    # Mock Pinecone store
    class MockPinecone:
        def __init__(self):
            self.vectors = {}  # id -> (vec, meta)
        def upsert(self, vectors):
            for vid, vec, meta in vectors:
                meta = dict(meta)
                meta["text"] = meta.get("text", f"chunk-{vid} text")
                self.vectors[vid] = (vec, meta)
        def query(self, vector, top_k, metadata_filter=None):
            out = []
            for vid, (vec, meta) in self.vectors.items():
                score = sum(a*b for a, b in zip(vector, vec[:len(vector)]))
                out.append({"id": vid, "score": score, "metadata": meta})
            out.sort(key=lambda x: x["score"], reverse=True)
            return out[:top_k]

    from rag_support.services import pinecone_store as pc_mod
    monkeypatch.setattr(pc_mod, "PineconeStore", MockPinecone)

    from rag_support.services import ingestion as ing_mod
    ing_mod._GLOBAL_BM25_INDEX._tokenized.clear()
    ing_mod._GLOBAL_BM25_INDEX._doc_ids.clear()
    ing_mod._GLOBAL_BM25_INDEX._bm25 = None

    yield
