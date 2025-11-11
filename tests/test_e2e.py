import pytest
from httpx import AsyncClient
from rag_support.main import app

@pytest.mark.asyncio
async def test_e2e():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/v1/healthz")
        assert r.status_code == 200

        payload = {
            "items": [
                {"source": "doc1.md", "text": "FastAPI runs with ASGI. LangGraph builds stateful graphs.", "title": "Doc1"},
                {"source": "doc2.md", "text": "Vertex AI provides generative models and embeddings.", "title": "Doc2"},
            ]
        }
        r = await ac.post("/v1/rag/ingest", json=payload)
        assert r.status_code == 200

        q = {"query":"How do I build a graph with LangGraph and serve via FastAPI?", "top_k": 4, "alpha": 0.7}
        r = await ac.post("/v1/rag/query", json=q)
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data and data["answer"]
        assert data["debug"]["judge_report"]["verdict"] == "pass"
        assert isinstance(data["citations"], list)
