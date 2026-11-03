# Copilot Instructions — Vietnamese Fake News Detection System

Bạn là Agent Coder trong dự án **phát hiện tin giả tiếng Việt** (RAG + fine-tuned PhoBERT/XLM-R).
Các quy tắc dưới đây là **bắt buộc**. Nếu yêu cầu của người dùng mâu thuẫn với quy tắc, hãy nêu rõ mâu thuẫn trước khi code.

## 0. Quy trình làm việc (đọc trước khi viết bất kỳ dòng code nào)

1. Đọc `docs/ARCHITECTURE.md` (kiến trúc, data contract) và `docs/ROADMAP_TODO.md` (task hiện tại).
2. Chỉ làm **các task thuộc Phase đang mở**. Không tự ý làm trước task của Phase sau.
3. Trước khi code, tóm tắt kế hoạch trong 3–5 gạch đầu dòng: file sẽ tạo/sửa, hàm public, test sẽ viết.
4. Sau khi xong: liệt kê file đã thay đổi, cách chạy test, và **tick `- [x]`** task tương ứng trong `docs/ROADMAP_TODO.md`.
5. Nếu cần đổi schema/contract/kiến trúc → sửa `docs/ARCHITECTURE.md` trong cùng thay đổi và ghi vào Decision Log. Không đổi âm thầm.
6. Không chắc chắn về dữ liệu hoặc yêu cầu → **hỏi**, không đoán.

## 1. Ngôn ngữ & phong cách

- Python **3.11**. Type hints đầy đủ cho mọi hàm public; `mypy --strict` trên `src/`.
- Định dạng/lint: `ruff format` + `ruff check`, độ dài dòng **100**.
- Identifier, comment, docstring: **tiếng Anh**, docstring kiểu Google. Tài liệu `.md`: tiếng Việt. Thông báo lỗi trả về cho end-user qua API: tiếng Việt.
- Dùng `pathlib.Path`, không nối chuỗi đường dẫn. Dùng f-string. Không `import *`.
- Dùng `logging` qua `vnfake.utils.logging.get_logger(__name__)`. **Cấm `print`** trong `src/`.
- **Cấm** `except:` trần và nuốt lỗi âm thầm. Bắt exception cụ thể, log, rồi raise hoặc trả kết quả lỗi có cấu trúc.
- Hàm ≤ ~50 dòng, một trách nhiệm. Ưu tiên hàm thuần (pure) cho xử lý dữ liệu để dễ test.

## 2. Cấu trúc & phân tầng

```
src/vnfake/{schemas,data,retrieval,models,training,pipeline,serving,evaluation,utils}
```
- Chiều phụ thuộc cho phép: `utils ← schemas ← data ← retrieval/models ← training/pipeline ← serving/evaluation`.
  **Tầng dưới không được import tầng trên.**
- Code chạy được nằm trong `src/`. `scripts/` chỉ là CLI mỏng gọi `src/`. `notebooks/` chỉ để khám phá, **không** import ngược từ notebook vào `src/`.
- Mọi client Qdrant chỉ được tạo trong `retrieval/vector_store.py`. Mọi thao tác tách từ chỉ qua `data/segmentation.py`.

## 3. Cấu hình

- Mọi đường dẫn, tên model, siêu tham số, ngưỡng, tên collection → `configs/*.yaml`, nạp qua `vnfake.config` (Pydantic).
- **Cấm magic number/string** trong logic (ví dụ `256`, `0.5`, `"BAAI/bge-m3"`). Đưa vào config hoặc hằng số có tên.
- Bí mật (API key, URL nội bộ) chỉ đọc từ biến môi trường / `.env`; **không** commit; cập nhật `.env.example`.
- Ánh xạ nhãn lấy từ `label_map` trong config. **Không hard-code `0 = real`** hay `1 = fake`.

## 4. Quy tắc dữ liệu & NLP tiếng Việt

- Mọi văn bản vào pipeline phải qua `normalize_text()` (Unicode **NFC**, gộp whitespace, bỏ ký tự điều khiển) trước khi tách từ/embedding/tokenize.
- **PhoBERT:** đầu vào **bắt buộc đã tách từ** (từ ghép nối bằng `_`), **không lowercase**, `max_length` theo config (mặc định 256). Không bao giờ đưa văn bản chưa tách từ vào PhoBERT tokenizer.
- **XLM-R:** dùng văn bản thô đã NFC, không tách từ.
- Đọc JSONL bằng **streaming** (generator), không `json.load` cả file lớn. Validate từng dòng bằng Pydantic; dòng lỗi ghi vào file `rejected`, không làm sập pipeline.
- Schema dữ liệu nằm ở `src/vnfake/schemas/`. KB `date` được đổi tên thành `crawl_date`; **không** dùng nó để suy luận thời gian đăng bài.
- ViFactCheck: model chỉ nhận `(Statement, Evidence)`. **Không đưa `Context` vào input của model.**
- **Chống leakage:** không đọc split `test` trong code huấn luyện/tuning. Chỉ `evaluation/` mới được gọi `load_split("test")`, và phải có cờ `--final` rõ ràng.
- ID chunk phải **deterministic** (`uuid5(doc_id, chunk_idx)`); upsert phải **idempotent**.

## 5. Quy tắc ML / huấn luyện

- Dùng Hugging Face `Trainer`/`accelerate`; bật mixed precision khi có GPU. Gọi `set_seed(cfg.seed)` ở đầu mọi entrypoint.
- Chỉ số chọn model: **macro-F1 trên dev**. Luôn log accuracy, macro-F1, per-class P/R/F1, ECE.
- Mỗi run: log config đầy đủ + git commit hash + metrics vào MLflow; lưu `config.yaml` cạnh checkpoint.
- Không tải model/dữ liệu ở thời điểm import module. Tải lazy trong hàm/class, cache theo đường dẫn trong config.
- Calibration (temperature scaling) chỉ fit trên **dev**.

## 6. Quy tắc Retrieval / Inference

- Embedding theo **batch**; không gọi model từng câu trong vòng lặp lớn.
- Hybrid search = dense + sparse, hợp nhất RRF; rerank chỉ trên top-`retrieve_k`.
- Evidence đưa vào classifier **phải vừa `max_length`**: cắt ở ranh giới câu, ưu tiên câu có điểm cao; không cắt giữa từ.
- Verdict chỉ được là `REAL | FAKE | UNVERIFIABLE`. Trả `UNVERIFIABLE` khi dưới ngưỡng retrieval hoặc confidence. **Không bịa bằng chứng**; mọi evidence trả ra phải truy được về `chunk_id`/`url` trong KB.
- Module sinh giải thích (nếu có) chỉ được diễn đạt lại từ evidence đã chọn, không thêm sự kiện mới.

## 7. API (FastAPI)

- Request/response dùng Pydantic model trong `schemas/api.py`; không trả `dict` tự do.
- Model/index nạp một lần ở `lifespan`, truyền qua dependency injection; không nạp lại mỗi request.
- Giới hạn độ dài `claim` (≤ 512 ký tự), trả mã lỗi HTTP đúng ngữ nghĩa (422/503…).
- Mọi response có `model_version`, `index_version`, `latency_ms`.

## 8. Kiểm thử

- Mỗi module mới đi kèm test trong `tests/unit/` (pytest). Mục tiêu coverage `src/` ≥ 80%.
- Unit test **không** gọi mạng, **không** cần GPU, **không** tải model thật: dùng fixture nhỏ (`tests/fixtures/`) và fake/mocks (FAISS in-memory hoặc `QdrantClient(":memory:")`).
- Test nặng (tải model, Qdrant thật) gắn marker `@pytest.mark.slow` / `@pytest.mark.integration`.
- Bug fix phải có test tái hiện lỗi.

## 9. Git & PR

- Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- PR nhỏ, một mục đích. Mô tả PR gồm: task trong ROADMAP, thay đổi chính, cách kiểm thử.
- Không commit dữ liệu thô, checkpoint, index (`data/`, `artifacts/` nằm trong `.gitignore`; dùng DVC).

## 10. Những điều KHÔNG được làm

- Không thêm dependency mới khi chưa cập nhật `pyproject.toml` và nêu lý do.
- Không sửa nội dung `docs/` ngoài phạm vi task (trừ tick checkbox ROADMAP và Decision Log).
- Không đổi tên/cấu trúc thư mục đã chuẩn hóa trong ARCHITECTURE khi chưa được yêu cầu.
- Không sinh dữ liệu giả rồi báo cáo như kết quả thật; không hard-code kết quả đánh giá.
- Không dùng code từ notebook chưa được refactor vào `src/`.

## 11. Mẫu tham chiếu

```python
# src/vnfake/data/example.py
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from vnfake.schemas.kb import KBRecord
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)


def iter_kb_records(path: Path, rejected_path: Path) -> Iterator[KBRecord]:
    """Stream and validate KB records from a JSONL file.

    Args:
        path: Input JSONL file.
        rejected_path: Where invalid lines are written.

    Yields:
        Validated ``KBRecord`` objects.
    """
    ...
```

## 12. Definition of Done (một task chỉ "xong" khi)

- [ ] Đúng contract trong `docs/ARCHITECTURE.md`.
- [ ] `ruff check`, `ruff format --check`, `mypy` sạch.
- [ ] Test mới + test cũ đều pass.
- [ ] Không còn magic number / đường dẫn hard-code.
- [ ] Docstring cho API public; cập nhật docs nếu đổi hành vi.
- [ ] Checkbox tương ứng trong `docs/ROADMAP_TODO.md` đã được tick.
