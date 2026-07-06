# Kế Hoạch Di Trú Hệ Thống AI từ WMS_Core sang WMS_AI_Services

Tài liệu này mô tả chi tiết kế hoạch di trú toàn bộ mã nguồn, cấu hình, và workflow liên quan đến AI từ repository `WMS_Core` sang `WMS_AI_Services` (nhánh `move_AI_service` ở `WMS_Core`) mà không làm gián đoạn hệ thống hiện tại.

---

## 1. Phân Tích Hiện Trạng & Thành Phần Cần Di Trú

Hiện tại, các thành phần AI nằm rải rác trong `WMS_Core` bao gồm:

1. **AI Service & Engine**:
   - Thư mục: `WMS_Core/Services/ai-service/`
     - `src/ai_engine/`: Chứa core RAG engine, agent logic, LLM wrappers, hybrid retriever (ChromaDB + BM25).
     - `src/ai_service/`: FastAPI HTTP server (`/health`, `/metrics`), gRPC server (port `50059`), và Redis Event Consumer để phục vụ reindexing.
2. **AI Fine-tuning & Datasets**:
   - Thư mục: `WMS_Core/training/`
     - `fine_tuning/`: Script train LoRA/PEFT (`train_wms.py`), dataset generator (`build_enriched_dataset.py`), và dữ liệu huấn luyện (`data/wms_data_enriched.jsonl`).
     - `kaggle_wms_finetune/`: Gói bootstrap và script chạy train trên môi trường Kaggle GPU.
3. **Unit Tests**:
   - Thư mục: `WMS_Core/tests/unit/test_fine_tuning_templates.py` (Kiểm thử cấu trúc template và dữ liệu training).
4. **Protobuf & gRPC Definitions**:
   - Định nghĩa gốc: `WMS_Core/proto/wms/ai/v1/ai.proto`
   - File Python được sinh ra (generated code): `WMS_Core/Services/ai-service/src/ai_service/gen/`

---

## 2. Các Edge Cases & Chi Tiết Nhỏ Cần Lưu Ý (Critical Details)

### 2.1. Phụ thuộc vào Thư viện dùng chung (`shared-utils`)
- **Vấn đề**: `ai-service` sử dụng `shared_utils.events` (Durable Redis Stream Consumer), `shared_utils.observability` (Prometheus metrics), và `shared_utils.security`.
- **Giải pháp**: 
  - Trong môi trường local / Docker Compose, ta sẽ mount thư mục `WMS_Core/Libraries/shared-utils` vào container của `WMS_AI_Services` thông qua volume hoặc thêm vào `PYTHONPATH`.
  - Trong file `requirements.txt` của `WMS_AI_Services`, ta sẽ khai báo `shared-utils` dưới dạng local editable package: `-e ../WMS_Core/Libraries/shared-utils`.

### 2.2. Giữ nguyên Tên Package / Imports tuyệt đối
- **Vấn đề**: Mã nguồn sử dụng các import tuyệt đối như `from ai_engine.core.engine import WMSEngine` hoặc `from ai_service.pipeline import ...`.
- **Giải pháp**: Không đổi tên thư mục gốc hoặc đóng gói lại thành `app` ngay lập tức để tránh sửa đổi quá nhiều code. Ta sẽ copy trực tiếp `ai_engine` và `ai_service` vào thư mục `src/` của `WMS_AI_Services` và cấu hình `PYTHONPATH` trỏ tới `/app/src`.

### 2.3. Đường dẫn lưu trữ Adapter & Model Huấn luyện (`FINE_TUNED_MODEL_PATH`)
- **Vấn đề**: Adapter mô hình sau khi train mặc định được ghi vào `training/fine_tuning/wms_final_adapter`. Container chạy gRPC server cần đường dẫn này để nạp model.
- **Giải pháp**: Phải mount thư mục `training/` của `WMS_AI_Services` vào container hoặc copy nó vào Docker image của `WMS_AI_Services`.

### 2.4. Đóng gói Kaggle Bundle (`kaggle_wms_finetune`)
- **Vấn đề**: `kaggle_wms_finetune/README.md` chỉ ra gói này cần copy file `Services/ai-service/src/ai_service/pipeline/templates.py` để chạy huấn luyện độc lập.
- **Giải pháp**: Cập nhật script build hoặc tài liệu hướng dẫn trong Kaggle Bundle để lấy `templates.py` từ đường dẫn mới ở `WMS_AI_Services/src/ai_service/pipeline/templates.py`.

### 2.5. Sinh mã Protobuf (gRPC) từ `WMS_Core/scripts/gen_protos.py`
- **Vấn đề**: Khi chạy generator script `gen_protos.py`, nó cần sinh ra code gRPC và đặt trực tiếp vào thư mục gen của AI service mới.
- **Giải pháp**: Sửa đổi biến `TARGETS` trong `WMS_Core/scripts/gen_protos.py` để trỏ output đường dẫn của `ai-service` sang thư mục đích tương ứng ở `WMS_AI_Services`.

---

## 3. Các Bước Thực Hiện Chi Tiết

### Bước 1: Sao chép tài nguyên sang `WMS_AI_Services`
1. Sao chép toàn bộ thư mục `WMS_Core/Services/ai-service/src/` sang `WMS_AI_Services/src/` (chứa `ai_engine` và `ai_service`).
2. Sao chép thư mục `WMS_Core/training/` sang `WMS_AI_Services/training/`.
3. Sao chép file cấu hình `.env.example` và `README.md` từ `ai-service` cũ sang `WMS_AI_Services/`.
4. Sao chép file test `WMS_Core/tests/unit/test_fine_tuning_templates.py` sang `WMS_AI_Services/tests/unit/test_fine_tuning_templates.py`.

### Bước 2: Cấu hình Môi trường ảo & Dependency ở `WMS_AI_Services`
1. Cập nhật `WMS_AI_Services/requirements.txt` với toàn bộ dependencies lấy từ `pyproject.toml` của `ai-service` cũ, kèm theo tham chiếu local `-e ../WMS_Core/Libraries/shared-utils`.
2. Tạo file `WMS_AI_Services/Dockerfile` sử dụng `uv` hoặc `pip` tương tự như của Core nhưng điều chỉnh context để copy toàn bộ `src/` và `training/` vào container.

### Bước 3: Sửa đổi cấu hình trong `WMS_Core` (Nhánh `move_AI_service`)
1. **`WMS_Core/pyproject.toml`**:
   - Xóa thành viên `"Services/ai-service"` khỏi danh sách `tool.uv.workspace.members`.
   - Xóa `packages = ["training/fine_tuning"]` khỏi `hatch.build.targets`.
2. **`WMS_Core/scripts/gen_protos.py`**:
   - Cập nhật phần tử `"Services/ai-service"` thành:
     `"../WMS_AI_Services": ("src/ai_service/gen", ["wms/ai"])`
3. **`WMS_Core/docker-compose.yml`**:
   - Sửa cấu hình build của service `ai-service`:
     ```yaml
     ai-service:
       build:
         context: ../WMS_AI_Services
         dockerfile: Dockerfile
       volumes:
         - ../WMS_AI_Services/src:/app/src
         - ../WMS_AI_Services/training:/app/training
         - ./Libraries/shared-utils/src/shared_utils:/app/shared_utils
         - ai-hf-cache:/root/.cache/huggingface
       environment:
         PYTHONPATH: /app/src:/app/shared_utils
     ```
4. **`WMS_Core/run_all.py`**:
   - Cập nhật đường dẫn chạy local:
     - `service_dir = ROOT.parent / "WMS_AI_Services"`
     - `src_dir = ROOT.parent / "WMS_AI_Services" / "src"`
     - `PYTHONPATH` tương ứng.
5. **Xóa file cũ ở `WMS_Core`**:
   - Xóa thư mục `WMS_Core/Services/ai-service/`.
   - Xóa thư mục `WMS_Core/training/`.
   - Xóa file `WMS_Core/tests/unit/test_fine_tuning_templates.py`.

---

## 4. Kế Hoạch Xác Minh (Verification Plan)

### 4.1. Kiểm thử Tự động (Automated Tests)
1. **Compile Protobuf**:
   Chạy script để đảm bảo code gRPC được sinh thành công sang thư mục mới:
   ```bash
   python scripts/gen_protos.py
   ```
2. **Chạy Unit Tests**:
   Đảm bảo các test cho dataset template vẫn pass khi chạy từ môi trường mới:
   ```bash
   pytest ../WMS_AI_Services/tests/unit/test_fine_tuning_templates.py
   ```

### 4.2. Kiểm thử Tích hợp Thủ công (Manual Integration Tests)
1. **Khởi động Docker Compose với Profile `ai`**:
   ```bash
   docker compose --profile ai up --build -d
   ```
   Kiểm tra log của `ai-service` để đảm bảo RAG engine khởi tạo thành công và kết nối Redis thành công:
   ```bash
   docker compose logs -f ai-service
   ```
2. **Kiểm tra HTTP Health & Metrics**:
   ```bash
   curl http://localhost:8009/health
   curl http://localhost:8009/metrics
   ```
3. **Test AI Query thông qua API Gateway (gRPC proxy)**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/ai/query \
     -H "Authorization: Bearer <ACCESS_TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"question": "How many warehouses do we have?", "mode": "auto"}'
   ```
   Phản hồi phải trả về thành công dạng JSON chứa câu trả lời RAG hoặc template query.
