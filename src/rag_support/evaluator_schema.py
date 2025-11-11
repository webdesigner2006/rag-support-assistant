from __future__ import annotations

from pydantic import BaseModel
from typing import Dict, List


class JudgeFlags(BaseModel):
    leakage_risk: bool = False
    toxicity: bool = False
    policy_violation: bool = False


class JudgeReport(BaseModel):
    verdict: str  # "pass" | "fail"
    scores: Dict[str, float]  # groundedness, relevance, completeness, clarity
    flags: JudgeFlags
    reasons: List[str]
    suggested_fixes: List[str]
