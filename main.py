from __future__ import annotations

import os
from fastapi import FastAPI
from .api.v1.routers import router as v1_router

app = FastAPI(title="RAG Support Assistant", version="0.1.0")
app.include_router(v1_router)

@app.get("/")
async def root():
    return {"message": "RAG Support Assistant is running"}
