from __future__ import annotations

from fastapi import Depends
from .models import QueryRequest 

from ...config import settings


def get_settings():
    return settings


def get_query_defaults(req: QueryRequest = Depends()):
    # Could enrich defaults or validate alpha/top_k further
    return req

