# RAG Support Assistant (LangGraph + Vertex AI + Pinecone)

Production-grade, **hybrid RAG** assistant with **Retriever → Validator → Generator → Evaluator** pipeline.  
- **Generation & Judge:** Vertex AI text model  
- **Embeddings:** Vertex AI Text Embeddings  
- **Vector Store:** Pinecone (cosine)  
- **Keyword Search:** BM25 (rank-bm25)  
- **Frameworks:** FastAPI, LangGraph, Pydantic v2, pytest

---

## Architecture

```
          ┌──────────────────────────────────────────────────────────┐
          │                          FastAPI                         │
          │   POST /v1/rag/ingest           POST /v1/rag/query       │
          └───────────────┬──────────────────────────┬───────────────┘
                          │                          │
                          │                          │
                    ┌─────▼─────┐             ┌─────▼─────┐
                    │ Ingestion │             │  RagGraph │
                    └─────┬─────┘             └─────┬─────┘
                          │                          │
                ┌─────────▼─────────┐         ┌──────▼─────────────────────────────────────────┐
                │ Vertex Embeddings │         │ 1) retriever_node (hybrid)                     │
                └─────────┬─────────┘         │    - Pinecone ANN (semantic)                   │
                          │                   │    - BM25 (keyword)                            │
                ┌─────────▼─────────┐         │    - RRF + alpha fusion                        │
                │   Pinecone Index  │         └──────┬─────────────────────────────────────────┘
                └─────────┬─────────┘                │
                          │                   ┌──────▼──────────────────────┐
               ┌──────────▼──────────┐       │ 2) validator_node            │
               │  BM25 (in-memory)   │       │    - cosine reuse + priors   │
               └──────────────────────┘       │    - thresholds + rationale  │
                                              └──────┬──────────────────────┘
                                                     │
                                              ┌──────▼──────────────────────────────────────────┐
                                              │ 3) generator_node                                │
                                              │    - strict grounding prompt                     │
                                              │    - Vertex gen model (temp, max tokens)         │
                                              │    - citations [#source-id]                     │
                                              └──────┬──────────────────────────────────────────┘
                                                     │
                                              ┌──────▼──────────────────────────────────────────┐
                                              │ 4) evaluator_node (LLM-as-judge)                │
                                              │    - scores: groundedness/relevance/...         │
                                              │    - guardrails: PII/PHI, secrets, toxicity     │
                                              │    - pass/fail + fixes                          │
                                              └─────────────────────────────────────────────────┘
```

---

## Design Decisions

- **LangGraph** orchestrates a clear, auditable path through nodes and allows future branching/repair strategies.  
- **Hybrid retrieval** merges semantic ANN (Pinecone) with BM25 using **RRF** and a configurable `alpha`.  
- **Validator** computes per-chunk **confidence** = `w1*cosine_norm + w2*rank_norm + w3*source_prior`.  
- **Generator** uses a strict grounding prompt and returns **[#source-id]** style citations.  
- **Evaluator** (LLM-as-judge) enforces guardrails and yields a **structured JSON**.  
- **Observability**: structured logs (no secrets) and simple usage counters from Vertex responses.

---

## Local Development

### Prereqs
- Python 3.11+
- `gcloud auth application-default login` (for ADC) **or** set `GOOGLE_APPLICATION_CREDENTIALS` to a JSON key.
- Pinecone account + index (dimension must match Vertex embedding).

### Setup
```bash
git clone YOUR_FORK_URL rag-support-assistant
cd rag-support-assistant
cp .env.example .env
# edit .env with your project, location, model ids, pinecone keys, index, etc.

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]

make fmt lint typecheck
make test

make run
# API -> http://localhost:8080
# Docs -> http://localhost:8080/docs
```

### Docker
```bash
docker compose up --build
```

---

## GCP Setup

1. Enable APIs:
   - Vertex AI API
   - Artifact Registry
   - Cloud Build
   - (If using GKE) Kubernetes Engine

2. Auth:
   - Local: `gcloud auth application-default login`
   - CI/Prod: create a **Service Account**, grant `Vertex AI User`, `Artifact Registry Writer`, `Storage Object Viewer`; download a JSON key for local dev or use Workload Identity on GKE/Cloud Run.

3. Set env vars:
   - `GOOGLE_PROJECT_ID`, `GOOGLE_LOCATION`, `VERTEX_MODEL_ID`, `VERTEX_EMBED_MODEL_ID`.

---

## Pinecone Setup

- Create index (once). Dimension must equal embedding size (e.g., **text-embedding-004 → 3072**).
- `.env`:
  - `PINECONE_API_KEY`, `PINECONE_ENV`, `PINECONE_INDEX`, `PINECONE_DIM=3072`

Example Python snippet (outside this service):
```python
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="...")

if "rag-support-assistant" not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name="rag-support-assistant",
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
```

---

## Ingestion Workflow

- `POST /v1/rag/ingest` accepts `items[] = {source, text, title?, url?, tags?}`
- Text is chunked (~5.5k char with 500 overlap), embedded with Vertex, stored in Pinecone.
- Keyword index (BM25) is kept in memory for demo; switch to Whoosh/ES for persistence.

**Example**
```bash
curl -X POST http://localhost:8080/v1/rag/ingest   -H "Content-Type: application/json"   -d '{
    "items":[
      {"source":"kb/setup.md", "title":"Setup", "text":"Install, configure, deploy steps...", "url":"https://kb/setup"}
    ]
  }'
```

---

## Querying

- `POST /v1/rag/query` with `{ query, top_k?, alpha?, metadata_filters? }`
- Returns: `{ answer, citations, debug{retrieved, validated, judge_report}, usage }`

**Example**
```bash
curl -X POST http://localhost:8080/v1/rag/query   -H "Content-Type: application/json"   -d '{"query":"How to deploy on GKE?","top_k":8,"alpha":0.7}'
```

---

## Troubleshooting

- **403 / Auth**: ensure ADC or service account key is configured; check `GOOGLE_PROJECT_ID`.
- **Pinecone dimension mismatch**: set `PINECONE_DIM` to your embedding size.
- **Empty answers**: increase ingestion corpus; confirm `/v1/rag/ingest` ran successfully.
- **Evaluator failing often**: lower similarity threshold or adjust alpha; improve chunk quality.

---

## Security

- Never log secrets; logs include only counts/ids.  
- Redact sensitive fields if you add request/response logging.  
- For Cloud Run/GKE, pass secrets via env vars / Secret Manager and restrict egress.

---

## Tests

```bash
make test
```
Covers:
- RRF merge
- Confidence math
- Prompt assembly
- Evaluator schema
- End-to-end happy path (mocks)

---

## CI

- GitHub Actions: lint, mypy, pytest, Docker build
- Cloud Build: container build & push to Artifact Registry (edit `PROJECT_ID`, `REGION`)

---

## Deploy

### Cloud Build → Artifact Registry
```bash
gcloud builds submit --config cloudbuild.yaml   --substitutions=_TAG=prod   --project $GOOGLE_PROJECT_ID
```

### Cloud Run (simple)
```bash
gcloud run deploy rag-support-assistant   --image=REGION-docker.pkg.dev/$GOOGLE_PROJECT_ID/rag/rag-support-assistant:prod   --region=us-central1 --allow-unauthenticated   --set-env-vars GOOGLE_PROJECT_ID=$GOOGLE_PROJECT_ID,GOOGLE_LOCATION=us-central1,VERTEX_MODEL_ID=gemini-1.5-pro,VERTEX_EMBED_MODEL_ID=text-embedding-004,PINECONE_INDEX=rag-support-assistant
```

### GKE (advanced)
```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret-example.yaml    # replace with real secret source
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

---

## Notes: Vertex Endpoints Alternative

If you later serve **only via Cloud Run**, `cloudrun.yaml` included. For **Vertex AI Endpoints**, you can deploy a custom container; this app already speaks HTTP, so Cloud Run suffices for most cases.

---

## License

MIT
