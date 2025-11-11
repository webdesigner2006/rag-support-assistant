import numpy as np
from rag_support.services.retrieval import RetrievalService, RetrievedDoc

def test_confidence_scoring():
    svc = RetrievalService()
    qv = np.ones(8)
    docs = [
        RetrievedDoc(id="1", text="alpha", title="", url="", source_id="s1", chunk_id="c1", semantic_score=0.9, keyword_score=0.1),
        RetrievedDoc(id="2", text="", title="", url="", source_id="s2", chunk_id="c2", semantic_score=0.2, keyword_score=0.8),
    ]
    vals = svc.validator(qv, docs)
    assert len(vals) == 2
    confs = [v["confidence"] for v in vals]
    assert 0 <= min(confs) <= max(confs) <= 1.0
