from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class IngestItem(BaseModel):
    source: str = Field(description="Source identifier (file path or URL)")
    text: str = Field(description="Full text content")
    title: Optional[str] = None
    url: Optional[str] = None
    tags: Optional[List[str]] = None


class IngestRequest(BaseModel):
    items: List[IngestItem]


class Citation(BaseModel):
    source_id: str
    title: str | None = None
    url: str | None = None
    chunk_id: str | None = None


class ScoreBreakdown(BaseModel):
    cosine: float
    rank_norm: float
    source_prior: float
    confidence: float


class ValidatedChunk(BaseModel):
    source_id: str
    chunk_id: str
    text: str
    confidence: float
    breakdown: ScoreBreakdown
    title: str | None = None
    url: str | None = None


class JudgeFlags(BaseModel):
    leakage_risk: bool = False
    toxicity: bool = False
    policy_violation: bool = False


class JudgeReport(BaseModel):
    verdict: str
    scores: Dict[str, float]
    flags: JudgeFlags
    reasons: List[str]
    suggested_fixes: List[str]


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=8)
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata_filters: Optional[Dict[str, str]] = None


class QueryDebug(BaseModel):
    retrieved_ids: List[str]
    validated: List[ValidatedChunk]
    scores: Dict[str, float]
    judge_report: JudgeReport


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    debug: QueryDebug
    usage: Dict[str, int]
