from __future__ import annotations

import os
from pydantic import BaseModel, Field


class Settings(BaseModel):
    google_project_id: str = Field(default_factory=lambda: os.getenv("GOOGLE_PROJECT_ID", ""))
    google_location: str = Field(default_factory=lambda: os.getenv("GOOGLE_LOCATION", "us-central1"))
    vertex_model_id: str = Field(default_factory=lambda: os.getenv("VERTEX_MODEL_ID", "gemini-1.5-pro"))
    vertex_embed_model_id: str = Field(
        default_factory=lambda: os.getenv("VERTEX_EMBED_MODEL_ID", "text-embedding-004")
    )

    pinecone_api_key: str = Field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    pinecone_env: str = Field(default_factory=lambda: os.getenv("PINECONE_ENV", "us-east-1"))
    pinecone_index: str = Field(default_factory=lambda: os.getenv("PINECONE_INDEX", "rag-support-assistant"))
    pinecone_dim: int = Field(default_factory=lambda: int(os.getenv("PINECONE_DIM", "3072")))

    rag_top_k: int = Field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "8")))
    hybrid_alpha: float = Field(default_factory=lambda: float(os.getenv("HYBRID_ALPHA", "0.7")))
    bm25_top_k: int = Field(default_factory=lambda: int(os.getenv("BM25_TOP_K", "12")))
    semantic_top_k: int = Field(default_factory=lambda: int(os.getenv("SEMANTIC_TOP_K", "12")))
    similarity_threshold: float = Field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.25")))

    gen_temperature: float = Field(default_factory=lambda: float(os.getenv("GEN_TEMPERATURE", "0.2")))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "1024")))

    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8080")))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    google_application_credentials: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    )


settings = Settings()
