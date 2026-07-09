# 🧠 WMS AI Services: Intelligent Warehouse Reasoning & RAG Engine

WMS AI Services is an intelligent backend microservice designed for heavy Retrieval-Augmented Generation (RAG) queries, LangGraph-driven agent workflows, search index update pipelines, and query template fine-tuning. 

This service runs as an opt-in component in the ecosystem, managed via the `ai` Docker Compose profile.

---

## 🛠️ Technology Stack

- **Frameworks**: LangGraph, LangChain, FastAPI (for HTTP metrics/health)
- **Vector Search & Embedding**: ChromaDB, BM25 retriever, HuggingFace (`all-MiniLM-L6-v2`)
- **LLM Integrations**: Groq API, OpenAI API
- **Fine-Tuning**: PyTorch, HuggingFace Transformers, LoRA / PEFT
- **Communication Protocol**: gRPC (for microservice orchestration), HTTP (for observability)

---

## 🧭 Ingestion & Processing Pipeline

The service codebase is organized into modular pipeline boundaries:

- `ai_service.pipeline.ingestion` — Translates incoming domain events or projection snapshots into indexing jobs.
- `ai_service.pipeline.indexing` — Queue adapter that indexes processed data blocks into vector databases.
- `ai_service.pipeline.retrieval` — Retrieval context boundary, performing Hybrid Search (Vector similarity + Keyword search).
- `ai_service.pipeline.routing` — Classifies prompts into knowledge queries (RAG-based) or operational data queries.
- `ai_service.pipeline.templates` — Extracts parameters from unstructured data queries into standardized JSON templates. Defaults to `GroqQueryTemplateExtractor` or uses a local fine-tuned model.
- `ai_service.pipeline.backend_query` — Routes extracted JSON templates to the parent `api-gateway` to query operational databases securely.
- `ai_service.pipeline.generation` — Coordinates the end-to-end query workflow (Router -> RAG/Template Extraction -> Context Synthesis).
- `ai_service.pipeline.providers` — Adapters and clients for external LLM engines.

> [!IMPORTANT]
> To preserve domain boundaries, the AI Service never queries operational databases directly. It only ingests streaming events and read-model snapshots to rebuild vector search representations asynchronously.

---

## 🔄 Query Execution Flow

1. **Client Call**: A client initiates a request via the gRPC `Query` endpoint defined on `wms.ai.v1.AIService`.
2. **Query Routing**: The router classifies the prompt to determine whether it is a semantic knowledge question or an operational database query.
3. **Knowledge Path (RAG)**: The prompt passes through a RAG workflow, fetching context using hybrid retrieval, generating responses via LLM, and verifying accuracy.
4. **Data Path (Template Extraction)**: The prompt is passed to the extractor to generate a JSON structure (`{intent, target, filters, metrics}`). If `AI_BACKEND_QUERY_URL` is configured, the adapter posts the template to the gateway for execution; otherwise, it returns the structured query to the client to ensure boundaries remain clean.
5. **Observability**: Health checks and metrics are exposed over HTTP via `GET /health` and `GET /metrics`.

---

## 🏋️ Fine-Tuning Workflow

To enable local query template extraction without relying on external LLM APIs, you can train a local Small Language Model (SLM) using the LoRA script:

```bash
uv run python training/fine_tuning/train_wms.py
```

### Dataset & Artifacts
- `training/fine_tuning/data/wms_data_enriched.jsonl` — Multi-domain WMS dataset containing English and Vietnamese paraphrased queries.
- `training/fine_tuning/build_enriched_dataset.py` — Generator utility script to compile or expand the training datasets.
- `training/fine_tuning/wms_final_adapter` — LoRA / PEFT adapter weights.
- `training/fine_tuning/wms_final_model` — Merged final model ready for inference.

### Inference Configuration
Once the model is fine-tuned, configure runtime variables in your service environment file:

```env
FINE_TUNED_MODEL_PATH=training/fine_tuning/wms_final_model
FINE_TUNED_MODEL_DEVICE=cpu
```

> [!TIP]
> The runtime also supports PEFT folders containing `adapter_config.json`. Simply point `FINE_TUNED_MODEL_PATH` directly to `training/fine_tuning/wms_final_adapter`.
