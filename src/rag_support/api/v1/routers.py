from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from .models import IngestRequest, QueryRequest, QueryResponse
from .deps import get_settings
from ...logging import logger
from ...services.ingestion import IngestionService
from ...services.retrieval import RetrievalService
from ....rag_graph import RagGraph
from ...config import Settings

router = APIRouter(prefix="/v1", tags=["v1"])


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}


@router.post("/rag/ingest")
async def ingest(req: IngestRequest, cfg: Settings = Depends(get_settings)):
    ing = IngestionService(cfg)
    try:
        await ing.ingest_items(req.items)
    except Exception as e:
        logger.exception("ingest_failed")
        raise HTTPException(status_code=500, detail=f"ingest_failed: {e}")
    return {"status": "ingested", "count": len(req.items)}


@router.post("/rag/query", response_model=QueryResponse)
async def rag_query(req: QueryRequest, cfg: Settings = Depends(get_settings)):
    retrieval = RetrievalService(cfg)
    graph = RagGraph(cfg, retrieval=retrieval)

    try:
        answer, citations, debug, usage = await graph.run_query(
            query=req.query,
            top_k=req.top_k,
            alpha=req.alpha,
            metadata_filters=req.metadata_filters or {},
        )
    except Exception as e:
        logger.exception("query_failed")
        raise HTTPException(status_code=500, detail=f"query_failed: {e}")

    return QueryResponse(answer=answer, citations=citations, debug=debug, usage=usage)
