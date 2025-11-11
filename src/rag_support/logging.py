from __future__ import annotations

import logging
import os
from typing import Any, Dict

LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
_handler.setFormatter(_formatter)

logger = logging.getLogger("rag_support")
logger.setLevel(LEVEL)
if not logger.handlers:
    logger.addHandler(_handler)


def redact(d: Dict[str, Any], keys: set[str]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        out[k] = "***REDACTED***" if k in keys else v
    return out
