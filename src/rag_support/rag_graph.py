from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from langgraph.graph import StateGraph, END

from rag_support.config import settings
from rag_support.logging import logger
from rag_support.api.v1.models import Citation, QueryDebug
from rag_support.services.retrieval import RetrievalService
from rag_support.services.vertex import VertexClient


GENERATOR_SYSTEM_PROMPT = """You are a Support Assistant. Follow STRICT grounding rules:
- Only answer using the provided VALIDATED CONTEXTS.
- If insufficient evidence, reply: "I don't have enough information to answer."
- Always include citations as [#source-id] where source-id is the chunk's source_id or chunk_id.
Answer clearly and concisely.
"""

EVALUATOR_SYSTEM_PROMPT = """You are a strict JSON-only judge. Output ONLY a JSON object with fields:
{
  "verdict": "pass" | "fail",
  "scores": {"groundedness": 0-5, "relevance": 0-5, "completeness": 0-5, "clarity": 0-5},
  "flags": {"leakage_risk": bool, "toxicity": bool, "policy_violation": bool},
  "reasons": [str],
  "suggested_fixes": [str]
}
Evaluate whether the answer is grounded in the provided validated chunks and free of hallucinations, secrets, PII/PHI.
"""


class RagGraph:
    """
    LangGraph RAG pipeline: retriever -> validator -> generator -> evaluator -> (optional repair)
    """

    def __init__(self, cfg=settings, retrieval: RetrievalService | None = None):
        self.cfg = cfg
        self.vertex = VertexClient()
        self.retrieval = retrieval or RetrievalService(cfg)

    async def run_query(
        self, query: str, top_k: int, alpha: float, metadata_filters: Dict[str, Any]
    ) -> Tuple[str, List[Citation], QueryDebug, Dict[str, int]]:
        # Build graph on-demand (small graph, OK to compile per request)
        def node_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
            docs = state["retrieved"]  # already retrieved in outer control to use async
            return {"retrieved": docs}

        def node_validator(state: Dict[str, Any]) -> Dict[str, Any]:
            qv = state["query_vec"]
            docs = state["retrieved"]
            validated = self.retrieval.validator(qv, docs)
            validated = [v for v in validated if v["confidence"] >= self.cfg.similarity_threshold]
            return {"validated": validated}

        def node_generator(state: Dict[str, Any]) -> Dict[str, Any]:
            validated = state.get("validated", [])
            if not validated:
                answer = "I don't have enough information to answer."
                usage = {"prompt_tokens": 0, "candidates_tokens": 0}
                return {"answer": answer, "citations": [], "usage": usage}

            ctx_lines = [f"[{i}] {v['text']}" for i, v in enumerate(validated, start=1)]
            cite_map = {str(i): v for i, v in enumerate(validated, start=1)}
            user_prompt = state["query"]
            prompt = (
                GENERATOR_SYSTEM_PROMPT
                + "

CONTEXT:
"
                + "
".join(ctx_lines)
                + "

USER:
"
                + user_prompt
                + "

Remember to cite using [#source-id]."
            )
            res = self.vertex.generate(prompt, temperature=self.cfg.gen_temperature, max_tokens=self.cfg.max_tokens)
            citations: List[Citation] = []
            used_ids: set[str] = set()
            for i, v in enumerate(validated, start=1):
                sid = v.get("source_id") or v.get("chunk_id")
                if sid and sid not in used_ids:
                    citations.append(Citation(source_id=sid, title=v.get("title"), url=v.get("url"), chunk_id=v["chunk_id"]))
                    used_ids.add(sid)
            return {"answer": res.text, "citations": citations, "usage": res.usage}

        def node_evaluator(state: Dict[str, Any]) -> Dict[str, Any]:
            payload = {
                "query": state["query"],
                "answer": state.get("answer", ""),
                "validated": state.get("validated", []),
            }
            report = self.vertex.judge(EVALUATOR_SYSTEM_PROMPT, payload, temperature=0.0)
            return {"judge_report": report}

        graph = StateGraph(dict)
        graph.add_node("retriever", node_retriever)
        graph.add_node("validator", node_validator)
        graph.add_node("generator", node_generator)
        graph.add_node("evaluator", node_evaluator)

        graph.set_entry_point("retriever")
        graph.add_edge("retriever", "validator")
        graph.add_edge("validator", "generator")
        graph.add_edge("generator", "evaluator")
        graph.add_edge("evaluator", END)
        app = graph.compile()

        qv = self.vertex.embed([query])[0]
        retrieved = await self.retrieval.hybrid_search(
            query=query, alpha=alpha, top_k=top_k, metadata_filters=metadata_filters
        )

        result = app.invoke({"query": query, "query_vec": qv, "retrieved": retrieved})
        answer: str = result.get("answer", "")
        citations: List[Citation] = result.get("citations", [])
        report = result.get("judge_report", {"verdict": "pass"})

        if report.get("verdict") == "fail":
            tightened = await self.retrieval.hybrid_search(
                query=query, alpha=alpha, top_k=top_k, metadata_filters=metadata_filters
            )
            result2 = app.invoke({"query": query, "query_vec": qv, "retrieved": tightened})
            answer = result2.get("answer", answer)
            citations = result2.get("citations", citations)
            report = result2.get("judge_report", report)

        debug = QueryDebug(
            retrieved_ids=[d.id for d in retrieved],
            validated=result.get("validated", []),
            scores={},
            judge_report=report,
        )
        usage = result.get("usage", {"prompt_tokens": 0, "candidates_tokens": 0})

        return answer, citations, debug, usage
