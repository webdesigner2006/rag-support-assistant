from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from google.cloud import aiplatform
from google.cloud.aiplatform import initializer
from rag_support.config import settings


@dataclass
class VertexTextResult:
    text: str
    usage: Dict[str, int]


class VertexClient:
    """
    Simple wrapper around Vertex AI gen and embed APIs.
    """

    def __init__(self, project: str | None = None, location: str | None = None) -> None:
        initializer.global_config.project = project or settings.google_project_id
        initializer.global_config.location = location or settings.google_location
        aiplatform.init(project=initializer.global_config.project, location=initializer.global_config.location)

        self._model_id = settings.vertex_model_id
        self._embed_model_id = settings.vertex_embed_model_id

    def embed(self, texts: List[str]) -> np.ndarray:
        from vertexai.language_models import TextEmbeddingModel

        model = TextEmbeddingModel.from_pretrained(self._embed_model_id)
        res = model.get_embeddings(texts)
        vecs = [np.array(r.values, dtype=float) for r in res]
        return np.vstack(vecs)

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> VertexTextResult:
        from vertexai.generative_models import GenerativeModel

        model = GenerativeModel(self._model_id)
        # A simple text-only prompt
        resp = model.generate_content(
            [prompt],
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        text = resp.text or ""
        # usage extraction may vary by SDK version; safeguard with defaults
        usage = {
            "prompt_tokens": getattr(resp, "prompt_token_count", 0) or 0,
            "candidates_tokens": getattr(resp, "candidates_token_count", 0) or 0,
        }
        return VertexTextResult(text=text, usage=usage)

    def judge(self, system_prompt: str, message_json: Dict[str, Any], temperature: float = 0.0) -> Dict[str, Any]:
        """
        LLM-as-judge: returns structured JSON per schema.
        """
        from vertexai.generative_models import GenerativeModel

        model = GenerativeModel(self._model_id)
        prompt = system_prompt + ""

JSON:
" + json.dumps(message_json, ensure_ascii=False)
        resp = model.generate_content([prompt], generation_config={"temperature": temperature, "max_output_tokens": 512})
        text = (resp.text or "").strip()
        # ensure valid json
        try:
            obj = json.loads(text)
        except Exception:
            obj = {"verdict": "fail", "reasons": ["invalid_json"], "flags": {"policy_violation": True}}
        return obj
