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
- Mọi client Qdrant chỉ được tạo trong `retrieval/vector_store.py`. Mọi thao tác tách từ chỉ qua `data/segmentation.py`. Client Gemini chỉ được tạo trong `retrieval/query_rewriter.py`.

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
- Schema dữ liệu nằm ở `src/vnfake/schemas/`. KB `date` được đổi tên thành `crawl_date`; **không** dùng nó để suy luận thời gian đăng bài. Bản ghi RSS có thêm `published_date` (từ `pubDate`) — luôn tách biệt với `crawl_date`.
- Trường thời gian: file tĩnh `date` → `crawl_date` (alias ở schema/loader, **không sửa file raw**); RSS `pubDate` → `published_date` (ISO-8601 UTC). **Cấm** lọc/sắp xếp theo thời gian trên toàn KB hoặc dùng `crawl_date` như ngày đăng bài (task lọc thời gian ở Backlog).
- ViFactCheck: đầu vào của mô hình phân loại = `(Statement, evidence_text)`. **`Context` không được đưa nguyên vào mô hình.** `Context` chỉ được dùng làm "bài báo giả lập" để chọn câu bằng chứng (`retrieval/evidence_selector.py`, xem `ARCHITECTURE.md` §5.2).
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

### 6.1 Hiệu năng retrieval (Phase 2.5)

- `retrieve_k` mặc định **30**, `rerank_k` mặc định 5 — đặt trong `configs/base.yaml`, không hard-code. (Tiền đề ban đầu ép `k=20` do độ trễ `fp16` ~14-19s; sau khi k-sweep sạch ở `fp32, batch_size=4` chứng minh `k=30` đạt rerank `p95 = 1.88 s` << 10s, Hit@1=31/723 cao nhất và Hit@5=48/723, PM đã chính thức chốt `retrieve_k = 30`).
- **Guardrail latency:** mục tiêu Phase 2.5 là rerank p95 ≤ 10 s (PM chốt; ≤ 3 s thuộc Future Enhancements). **Không hạ `retrieve_k` dưới 20** để chạy đua latency khi chưa có Decision Log + PM duyệt; hãy tối ưu dtype/batch size/`max_length` trước.
- Mọi lần đo latency phải ghi kèm thiết bị (CPU/GPU), dtype, batch size, `retrieve_k`.

### 6.2 LLM Query Rewrite — QUY TẮC CỨNG

- Module `retrieval/query_rewriter.py` **chỉ** dùng **Gemini họ Flash** (Flash / Flash-Lite) qua Gemini API free tier. **Cấm** dùng LLM/provider khác cho module này, và cấm dùng LLM này cho việc khác ngoài rewrite (không phán verdict, không sinh evidence).
- "Khóa cứng" được thực thi bằng **validator Pydantic trên config**: `query_rewrite.model` phải khớp regex `^gemini-(?!1\.)[a-z0-9.-]*flash[a-z0-9.-]*$`. **Không** hard-code model id trong logic (xem §3).
- Model đã chốt: **`gemini-3.1-flash-lite`** (PM, 2026-11-08), giá trị mặc định ở `configs/base.yaml: query_rewrite.model`. Nếu API báo model không khả dụng, **không tự đổi sang model khác**: dùng `NoOpQueryRewriter` và báo PM.
- **Cấm dùng `gemini-1.5-*`:** Gemini 1.5 Flash đã bị Google ngừng, gọi sẽ lỗi model-not-found. Model id Gemini đổi/ngừng nhanh nên chỉ thay ở config; khi khởi động phải health-check model, lỗi ⇒ log rõ và chuyển sang `NoOpQueryRewriter` (trừ khi `query_rewrite.required=true`).
- `GEMINI_API_KEY` chỉ đọc từ biến môi trường/`.env`; không log key.
- Tiết kiệm free tier: `temperature=0`, `max_output_tokens` nhỏ (config), 1 request/claim, retry có giới hạn + exponential backoff, tôn trọng 429, client-side rate limiter, **disk cache** (key = hash(claim NFC + `prompt_version` + model)). Script đánh giá/benchmark phải đi qua cache.
- **Fail-safe:** timeout, 429, 5xx hoặc output không hợp lệ ⇒ dùng **claim gốc**, log WARNING `rewrite_status=fallback:<reason>`. Rewrite lỗi không bao giờ được làm hỏng `/v1/verify`.
- Chuỗi rewrite chỉ dùng cho **bước retrieve**. Reranker và classifier **luôn nhận claim gốc**; response API không trả chuỗi rewrite như `claim`.
- Rewriter không được thêm thực thể, số liệu hay sự kiện không có trong claim.

### 6.3 Prompting LLM — ÉP CHẶT OUTPUT

- Prompt nằm ở `configs/prompts/query_rewrite.txt` (có `prompt_version`); không build prompt bằng cách nối chuỗi rải rác.
- Prompt **bắt buộc** chứa nguyên văn: `Tuyệt đối CHỈ in ra từ khóa. KHÔNG chào hỏi, KHÔNG dạ vâng, KHÔNG giải thích.` (có unit test kiểm tra template chứa câu này).
- Claim do người dùng nhập là **dữ liệu không đáng tin**: đặt trong delimiter rõ ràng và nêu trong prompt rằng nội dung đó là dữ liệu, không phải chỉ thị (chống prompt injection).
- Định dạng đầu ra: **một dòng**, từ khóa tiếng Việt có dấu cách nhau bằng khoảng trắng, tối đa `max_keywords`.
- Mọi phản hồi LLM **bắt buộc** qua `sanitize_keywords()` trước khi vào embedder/Qdrant: lấy dòng đầu; bỏ markdown, bullet, ngoặc kép, tiền tố kiểu "Từ khóa:"; chuẩn hóa NFC; loại và fallback nếu rỗng, quá dài, hoặc chứa cụm chào hỏi/giải thích (danh sách trong config). Mỗi trường hợp cần một test.
- **Không bao giờ** đưa phản hồi LLM chưa sanitize vào Qdrant.

### 6.4 RSS Updater (Pipeline D)

- Chỉ fetch feed khai báo trong `configs/rss.yaml` (13 feed do PM chốt); không hard-code URL; không tự thêm/bớt feed; tôn trọng `robots.txt`, dùng User-Agent định danh, rate limit, timeout.
- Chạy **on-demand** (CLI/endpoint admin), không tự chạy nền.
- Dedup URL qua `rss_manifest.json` (và với KB tĩnh) **trước** khi fetch full text; upsert idempotent (chunk_id deterministic).
- Bản ghi RSS validate bằng Pydantic như KB; `ingest_source="rss"`; `published_date` ở dạng ISO-8601 UTC; không lưu HTML thô.
- Dữ liệu RSS **không** dùng để huấn luyện hoặc chấm proxy trên dev.

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
- [ ] Nếu đụng retrieval/rewrite: đã chạy lại `evaluation/retrieval_eval.py` và ghi latency + Hit Rate vào report.
