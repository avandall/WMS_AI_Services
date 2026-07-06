# AI Service

AI / Knowledge Service chạy opt-in qua compose profile `ai`.

Phạm vi:
- AI engine, RAG, embeddings, agent logic
- gRPC query endpoint + health/metrics HTTP endpoint
- Event-driven reindex queue từ domain events hoặc projection snapshots

Runtime mặc định của dự án không build/start service này. Khi cần AI:

```bash
docker compose --profile ai up -d ai-service
```

Pipeline boundary:

- `ai_service.pipeline.ingestion`: đổi event envelope/snapshot thành reindex job.
- `ai_service.pipeline.indexing`: queue adapter AI-owned cho reindex job.
- `ai_service.pipeline.retrieval`: boundary cho retrieval context.
- `ai_service.pipeline.routing`: phân loại prompt thành knowledge/RAG hoặc data query.
- `ai_service.pipeline.templates`: đổi data query thành object template key-value. Mặc định dùng
  Groq qua `GroqQueryTemplateExtractor`; có thể thay bằng local fine-tuned model khi set
  `FINE_TUNED_MODEL_PATH`, hoặc tự cài extractor khác qua `QueryTemplateExtractor`.
- `ai_service.pipeline.backend_query`: boundary gửi template sang backend sở hữu truy vấn dữ liệu.
- `ai_service.pipeline.generation`: query pipeline điều phối router, RAG, template extraction, và
  backend query.
- `ai_service.pipeline.providers`: adapter vào RAG/LLM engine nặng.

AI không đọc database vận hành của service khác; dữ liệu vào phải đến từ event hoặc read-model
snapshot có thể replay.

Query flow:

1. Client gọi gRPC `Query` của `wms.ai.v1.AIService` với `mode=auto` mặc định.
2. Router kiểm tra prompt có phải data/SQL/DB-style query không.
3. Knowledge prompt đi qua RAG workflow, dùng LLM API và quality evaluator/critic.
4. Data prompt đi qua template extractor để tạo object key-value, rồi chuyển object đó cho
   backend query adapter. Nếu cấu hình `AI_BACKEND_QUERY_URL`, adapter sẽ POST template sang
   backend; nếu chưa cấu hình, response trả về template đã chuẩn bị để giữ boundary sạch.
5. HTTP chỉ dùng cho `GET /health` và `GET /metrics`.

Fine-tuned local model, nếu bật, chỉ thay extractor ở bước 4. Nó không chạm vào RAG/agent path
và không thay đổi flow knowledge prompt.

Fine-tune workflow:

```bash
uv run python training/fine_tuning/train_wms.py
```

Script train dùng cùng prompt JSON template với runtime extractor, tự chuyển dataset SQL hiện tại
thành template có `intent`, `target`, `filters`, `metrics`, `limit`, và `sql`. Artifact mặc định:

- `training/fine_tuning/data/wms_data_enriched.jsonl`: dataset mặc định, đa domain WMS và có
  paraphrase tiếng Việt/Anh.
- `training/fine_tuning/build_enriched_dataset.py`: generator để tái tạo hoặc mở rộng dataset.

- `training/fine_tuning/wms_final_adapter`: LoRA/PEFT adapter.
- `training/fine_tuning/wms_final_model`: merged model, dùng thuận tiện nhất cho runtime.

Sau khi train xong, set:

```bash
FINE_TUNED_MODEL_PATH=training/fine_tuning/wms_final_model
FINE_TUNED_MODEL_DEVICE=cpu
```

Nếu chỉ muốn dùng adapter, trỏ `FINE_TUNED_MODEL_PATH` vào `wms_final_adapter`; runtime cũng hỗ trợ
PEFT adapter folder có `adapter_config.json`.
