# ROADMAP_TODO.md — Vietnamese Fake News Detection System

> **Cách dùng:** Agent/Coder chỉ làm task thuộc **Phase đang mở**. Hoàn thành task → đổi `- [ ]` thành `- [x]` và ghi PR/commit.
> Mỗi Phase kết thúc bằng một **Exit criteria** — chưa đạt thì chưa qua Phase sau.

## Tổng quan Milestones

| Milestone | Phase | Kết quả bàn giao | Trạng thái |
|---|---|---|---|
| **M0** Nền tảng & hiểu dữ liệu | 0 | Repo chạy được, báo cáo data audit, câu hỏi mở được chốt | ⬜ |
| **M1** Data layer | 1 | Loader/cleaner/segmenter/chunker có test | ⬜ |
| **M2** KB searchable | 2 | Qdrant index + hybrid search + rerank, có số đo retrieval | ⬜ |
| **M3** Classifier baseline | 3 | PhoBERT & XLM-R fine-tune xong, báo cáo trên dev | ⬜ |
| **M4** RAG end-to-end | 4 | Pipeline C chạy, có đường cong coverage–F1, chốt ngưỡng | ⬜ |
| **M5** Serving & demo | 5 | API + Docker + UI demo chạy bằng 1 lệnh | ⬜ |
| **M6** Hardening & báo cáo | 6 | CI, ablation, error analysis, báo cáo cuối | ⬜ |

---

## Phase 0 — Khởi tạo & kiểm toán dữ liệu (M0)

**Setup**
- [ ] Tạo repo theo cấu trúc thư mục chuẩn (xem `ARCHITECTURE.md`), `pyproject.toml` (uv), `Makefile`, `.gitignore`, `.env.example`
- [ ] Cấu hình `ruff`, `mypy`, `pytest`, `pre-commit`
- [ ] `docker-compose.yml` với Qdrant (+ MLflow)
- [ ] Viết `utils/logging.py`, `utils/seed.py`, `config.py` (Pydantic) + `configs/base.yaml`
- [ ] Khởi tạo DVC, track `data/raw/`

**Data audit** (`scripts/01_audit_data.py` + `notebooks/00_data_audit.ipynb`)
- [ ] Thống kê KB: số bài, độ dài `text`, phân bố nguồn, tỉ lệ rỗng/trùng
- [ ] Thống kê ViFactCheck 3 split: kích thước, **phân bố nhãn**, độ dài `Statement`/`Evidence` (theo token PhoBERT)
- [ ] **Xác định ngữ nghĩa nhãn 0/1** bằng cách đọc mẫu thực tế → ghi vào `label_map`
- [ ] Đo **mức trùng lặp** giữa gold `Evidence` và KB (exact/n-gram) → quyết định cách đánh giá retrieval
- [ ] Kiểm tra trùng `Statement` chéo train/dev/test (leakage)
- [ ] Xác minh model HF dự kiến (tên, license, kích thước) và phần cứng sẵn có
- [ ] Trả lời các **Câu hỏi mở** ở mục 10 của `ARCHITECTURE.md`

**Exit criteria:** `make test` xanh với test rỗng; `docs/DATA_AUDIT.md` được viết; `label_map` và câu hỏi mở đã chốt.

---

## Phase 1 — Data layer (M1)

- [ ] `schemas/kb.py`, `schemas/vifactcheck.py` (Pydantic) + test
- [ ] `data/loaders.py`: `iter_kb_records()` streaming + file `rejected`; `load_split(name)` cho ViFactCheck
- [ ] `data/cleaning.py`: `normalize_text()` (NFC…), bỏ boilerplate, lọc độ dài tối thiểu + test với ví dụ tiếng Việt có dấu
- [ ] `data/dedup.py`: dedup theo `url` + MinHash near-duplicate + test
- [ ] `data/segmentation.py`: wrapper VnCoreNLP/underthesea, cache, batch + test
- [ ] `data/chunking.py`: tách câu + cửa sổ trượt (config: `window`, `overlap`, `target_tokens`) + test (không cắt giữa câu, overlap đúng)
- [ ] `data/dataset.py`: `PairDataset` cho PhoBERT (đã tách từ) và XLM-R (thô)
- [ ] `scripts/02_prepare_data.py` chạy end-to-end ra `data/processed/`

**Exit criteria:** chạy được trên toàn bộ KB không crash; test coverage `data/` ≥ 80%; có báo cáo tỉ lệ dòng bị loại.

---

## Phase 2 — Retrieval / Knowledge Base (M2)

- [ ] `retrieval/embedder.py`: nạp model lazy, encode batch, hỗ trợ dense + sparse
- [ ] **Benchmark embedding** (bge-m3 vs ≥1 model tiếng Việt khác) trên tập truy vấn dev → ghi kết quả vào `docs/` và Decision Log
- [ ] `retrieval/vector_store.py`: tạo collection (dense + sparse), upsert idempotent, search, `index_manifest.json`
- [ ] `scripts/03_build_index.py`: build index từ chunks (resume được nếu gián đoạn)
- [ ] `retrieval/hybrid.py`: hợp nhất RRF + test với dữ liệu giả
- [ ] `retrieval/reranker.py`: cross-encoder rerank top-50 → top-5
- [ ] **Benchmark reranker** (có/không rerank)
- [ ] `evaluation/retrieval_eval.py`: Recall@k, MRR, nDCG@k theo cách đã chốt ở Phase 0
- [ ] Unit test dùng `QdrantClient(":memory:")`; integration test đánh dấu `slow`

**Exit criteria:** index build xong, truy vấn mẫu trả kết quả hợp lý; báo cáo retrieval có số liệu; thời gian truy vấn p95 < 300 ms (không tính rerank).

---

## Phase 3 — Classifier baseline (M3)

- [ ] `models/classifier.py`: wrapper `AutoModelForSequenceClassification` cho PhoBERT & XLM-R theo config
- [ ] `models/tokenization.py`: mã hóa cặp `(Statement, Evidence)`, truncation ưu tiên cắt Evidence
- [ ] `training/metrics.py`: accuracy, macro-F1, per-class, ECE
- [ ] `training/train.py` + `configs/train_phobert.yaml`, `configs/train_xlmr.yaml`
- [ ] Tích hợp MLflow (config, commit hash, metrics, artifact)
- [ ] Chạy baseline **gold evidence**: PhoBERT-base, XLM-R-base
- [ ] Hyper-parameter sweep nhỏ (lr, batch, epochs, max_len) — chỉ đánh giá trên **dev**
- [ ] `models/calibration.py`: temperature scaling + test
- [ ] Báo cáo baseline `docs/REPORT_baseline.md` (confusion matrix, lỗi tiêu biểu)

**Exit criteria:** ≥ 1 checkpoint có macro-F1 dev ổn định (3 seed, độ lệch chuẩn được báo cáo); run tái lập được từ config.

---

## Phase 4 — Tích hợp RAG end-to-end (M4)

- [ ] `retrieval/evidence_selector.py`: chọn câu liên quan nhất, vừa `max_length`, giữ thứ tự và nguồn
- [ ] `data/augment.py`: dựng cặp **retrieved evidence** cho train (chế độ `gold` / `retrieved` / `mixed`) + hard-negative mining từ Qdrant
- [ ] Huấn luyện lại với `mixed`; so sánh với baseline trên dev
- [ ] (Tùy chọn) NEI augmentation: ghép Statement với evidence không liên quan → nhãn `UNVERIFIABLE` (3-class)
- [ ] `pipeline/verifier.py`: ghép Pipeline C (normalize → retrieve → rerank → select → classify)
- [ ] `pipeline/decision.py`: luật `UNVERIFIABLE` theo `τ_retrieval`, `τ_conf`; tune ngưỡng trên **dev**
- [ ] `evaluation/e2e_eval.py`: macro-F1, coverage, đường cong coverage–F1
- [ ] Phân tích lỗi: retrieval miss / evidence nhiễu / model sai (`notebooks/`)
- [ ] Chốt cấu hình cuối; chạy **test set đúng 1 lần** (`--final`)

**Exit criteria:** E2E chạy qua CLI (`scripts/04_run_verifier.py "<claim>"`); số liệu E2E trên dev/test được báo cáo; ngưỡng được lưu cạnh checkpoint.

---

## Phase 5 — Serving & demo (M5)

- [ ] `schemas/api.py`: `VerifyRequest`, `VerifyResponse`, `Evidence`
- [ ] `serving/app.py`: FastAPI, nạp model/index ở `lifespan`, DI
- [ ] Endpoint `POST /v1/verify`, `GET /healthz`, `GET /v1/version`
- [ ] Validate đầu vào, xử lý lỗi (422/503), timeout, giới hạn độ dài claim
- [ ] `docker/Dockerfile.api`, `docker-compose` chạy api + qdrant
- [ ] UI demo (Streamlit/Gradio): nhập claim → verdict + bằng chứng có link nguồn
- [ ] Test API bằng `TestClient` (mock verifier)
- [ ] Đo latency: p50/p95 end-to-end trên máy mục tiêu

**Exit criteria:** `docker compose up` rồi gọi API/UI thành công; latency p95 được ghi vào `docs/`.

---

## Phase 6 — Hardening & báo cáo (M6)

- [ ] GitHub Actions: lint + mypy + unit test trên PR
- [ ] Ablation đầy đủ: gold/retrieved/mixed, PhoBERT/XLM-R, có/không rerank, base/large
- [ ] Đánh giá độ bền: claim ngắn/dài, lỗi chính tả, không dấu, claim đa mệnh đề
- [ ] Logging có cấu trúc + (tùy chọn) theo dõi drift điểm retrieval
- [ ] (Tùy chọn) Module giải thích bằng LLM, ràng buộc chỉ dùng evidence đã chọn
- [ ] Quy trình cập nhật KB định kỳ (re-crawl → re-index → đổi `index_version`)
- [ ] Hoàn thiện `README.md`, hướng dẫn tái lập, báo cáo cuối `docs/REPORT_final.md`

**Exit criteria:** CI xanh; báo cáo cuối có ablation và giới hạn đã biết; repo tái lập được từ máy sạch.

---

## Backlog (chưa xếp lịch)

- [ ] Crawl lại KB có `published_date` để lọc theo thời gian
- [ ] Mở rộng KB sang thêm nguồn báo chính thống & trang fact-check
- [ ] Fine-tune embedding/reranker trên cặp `(Statement, Evidence)` của ViFactCheck
- [ ] Claim decomposition cho câu nhiều mệnh đề
- [ ] Cache kết quả truy vấn phổ biến
