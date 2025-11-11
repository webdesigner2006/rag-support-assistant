from rag_support.utils import rrf_merge

def test_rrf_merge_basic():
    s = ["a","b","c"]
    k = ["b","c","d"]
    fused = rrf_merge(s,k)
    ids = [x[0] for x in fused]
    assert ids[0] in ("b","c")
    assert set(ids[:4]) == {"a","b","c","d"}
