# ROADMAP_TODO.md — Vietnamese Fake News Detection System

> **Cách dùng:** Agent/Coder chỉ làm task thuộc **Phase đang mở**. Hoàn thành task → đổi `- [ ]` thành `- [x]` và ghi PR/commit.
> Mỗi Phase kết thúc bằng một **Exit criteria** — chưa đạt thì chưa qua Phase sau.

## Tổng quan Milestones

| Milestone                      | Phase | Kết quả bàn giao                                                                                 | Trạng thái                               |
| ------------------------------ | ----- | ------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| **M0** Nền tảng & hiểu dữ liệu | 0     | Repo chạy được, báo cáo data audit, câu hỏi mở được chốt                                         | ✅                                       |
| **M1** Data layer              | 1     | Loader/cleaner/segmenter/chunker có test                                                         | ✅                                       |
| **M2** KB searchable           | 2     | Qdrant index + hybrid search + rerank, có số đo retrieval                                        | ✅                                       |
| **M2.5** System Optimization   | 2.5   | Query Rewrite (Gemini Flash) + `retrieve_k=30` + RSS updater, rerank p95 ≤ 10 s, đo lại Hit Rate | ⏸ đóng băng do chuyển hướng (2026-11-09) |
| **M3** Verifier + Web Search   | 3     | PhoBERT fine-tune (XLM-R hoãn) + calibration (gold/selected) + Tavily spike 30 claims                 | ✅ Hoàn thành (đóng theo xác nhận của Kỹ sư trưởng 2026-11-10; mục 11.3 chuyển sang Phase 4) |
| **M4** RAG end-to-end          | 4     | Pipeline C chạy (Web Search chính + KB phụ), chế độ dự phòng cache, demo Gradio                  | ⬜                                       |
| **M5** Serving & demo          | 5     | Chạy demo trên laptop RTX 3050 4 GB, đóng băng mã, chạy `test` 1 lần (`--final`)                 | ⬜                                       |
| **M6** Hardening & báo cáo     | 6     | Slide + báo cáo Word, tập dượt demo trực tiếp ngày 24/11                                         | ⬜                                       |

### Sprint cuối 9/11–24/11 (Brief #4 §6)

| Ngày     | Mốc                                                                                                                      |
| -------- | ------------------------------------------------------------------------------------------------------------------------ |
| 9–11/11  | Task 1: đo token, tokenization, wrapper, huấn luyện PhoBERT V-A và baseline. Task 2 xong, **cổng quyết định ngày 11/11** |
| 12–15/11 | V-B/V-C, XLM-R, calibration, 3 seed. **Verifier chốt ngày 15/11**                                                        |
| 15–18/11 | Module bằng chứng trực tuyến + chạy ~100 claim test, chế độ dự phòng (cache), demo Gradio trên PC                        |
| 18–19/11 | Chạy demo trên laptop RTX 3050 4 GB, đóng băng mã, chạy `test` một lần                                                   |
| 19–23/11 | Slide + báo cáo Word, tập dượt demo                                                                                      |
| 24/11    | Demo                                                                                                                     |

> **Kế hoạch B (nếu Task 2 < 15%):** (1) kết quả học sâu chính = verifier trên gold/selected evidence; (2) demo trên claim thời sự (nơi tìm kiếm hoạt động tốt); (3) phân tích trung thực giới hạn của kiểm chứng đầu-cuối.

---

## Phase 0 — Khởi tạo & kiểm toán dữ liệu (M0)

**Setup**

- [x] Tạo repo theo cấu trúc thư mục chuẩn (xem `ARCHITECTURE.md`), `pyproject.toml` (uv), `Makefile`, `.gitignore`, `.env.example`
- [x] Cấu hình `ruff`, `mypy`, `pytest`, `pre-commit`
- [x] `docker-compose.yml` với Qdrant (+ MLflow)
- [x] Viết `utils/logging.py`, `utils/seed.py`, `config.py` (Pydantic) + `configs/base.yaml`
- [x] Khởi tạo DVC, track `data/raw/`

**Data audit** (`scripts/01_audit_data.py` + `notebooks/00_data_audit.ipynb`)

- [x] Thống kê KB: số bài, độ dài `text`, phân bố nguồn, tỉ lệ rỗng/trùng
- [x] Thống kê ViFactCheck 3 split: kích thước, phân bố nhãn, độ dài `Statement`/`Evidence` (theo **ký tự**)
- [x] Đo độ dài `Statement`/`Evidence` theo **token PhoBERT** (sau word-segment) — chốt `max_length = 256` (`p50=69, p95=122, p99=160` cho gold; `p50=132, p95=179, p99=205` cho selected)
- [x] **Xác định ngữ nghĩa nhãn 0/1/2** bằng cách đọc mẫu thực tế → ghi vào `label_map` (`0=real, 1=fake, 2=unverifiable`)
- [x] Đo **mức trùng lặp** giữa gold `Evidence` và KB (exact/n-gram) → kết quả ~0.1% ở ngưỡng 0.7 (`docs/DATA_AUDIT.md §6`); Phase 2 dùng **proxy metric** thay vì Recall@k
- [x] Kiểm tra trùng `Statement` chéo train/dev/test (leakage) — `scripts/01c_check_leakage.py`; phát hiện 4 trùng (đã xử lý ở Phase 1 bằng `dedup_vifactcheck_splits`)
- [x] Xác minh model HF dự kiến (tên, license, kích thước) và phần cứng sẵn có
- [x] Trả lời các **Câu hỏi mở** ở mục 10 của `ARCHITECTURE.md`

**Exit criteria:** `make test` xanh với test rỗng; `docs/DATA_AUDIT.md` được viết; `label_map` và câu hỏi mở đã chốt.

> **Trạng thái Phase 0 (2026-11-04):** _Exit criteria_ **đã đạt**. Còn **1 item** kỹ thuật chưa xong:
>
> 1. **PhoBERT token length** — `max_len=256` hiện là giả định từ ký tự; cần đo token thật để xác nhận `R3` trước khi bước vào Phase 3 (huấn luyện classifier).
>
> Các item khác (DVC, overlap, leakage) **đã đóng**. Có thể mở Phase 1/2 song song.

---

## Phase 1 — Data layer (M1)

- [x] `schemas/kb.py`, `schemas/vifactcheck.py` (Pydantic) + test
- [x] `data/loaders.py`: `iter_kb_records()` streaming + file `rejected`; `load_split(name)` cho ViFactCheck
- [x] `data/cleaning.py`: `normalize_text()` (NFC…), bỏ boilerplate, lọc độ dài tối thiểu + test với ví dụ tiếng Việt có dấu
- [x] `data/dedup.py`: dedup theo `url` + MinHash near-duplicate + test; **bổ sung `dedup_vifactcheck_splits()`** khắc phục leakage (giữ dev/test, xoá train trùng)
- [x] `data/segmentation.py`: wrapper VnCoreNLP/underthesea, cache, batch + test
- [x] `data/chunking.py`: tách câu + cửa sổ trượt (config: `window`, `overlap`) + test; chunk_id deterministic qua `uuid5(doc_id, chunk_idx)`
- [x] `data/dataset.py`: `PairDataset` cho PhoBERT (đã tách từ) và XLM-R (thô); `truncation="only_second"` cắt Evidence trước
- [x] `data/prepare.py` + `scripts/02_prepare_data.py` chạy end-to-end ra `data/processed/` (kb_chunks + {train,dev,test}\_clean + rejected)

**Exit criteria:** chạy được trên toàn bộ KB không crash; test coverage `data/` ≥ 80%; có báo cáo tỉ lệ dòng bị loại.

**Kết quả Phase 1:** KB tạo ra **41.466 chunks**; ViFactCheck `train_clean` giữ **5.058** samples (sau khi xoá 4 dòng leakage nội bộ + chéo split). Đạt exit criteria.

---

## Phase 2 — Retrieval / Knowledge Base (M2)

- [x] `retrieval/embedder.py`: lazy load, batch encode, **dense + sparse**
- [ ] **Benchmark embedding** (bge-m3 vs ≥1 model tiếng Việt khác) trên tập truy vấn dev → ghi kết quả vào `docs/` và Decision Log
- [x] `retrieval/vector_store.py`: collection **named vectors (dense+sparse)**, upsert idempotent, `search_dense` / `search_sparse` / **`search_hybrid` (server-side RRF)**; **`index_manifest.json` + resume**
- [x] `scripts/03_index_kb.py` (+ `retrieval/index_builder.py`): build index từ chunks, **manifest + resume**
- [x] `retrieval/hybrid.py`: RRF **client-side** (fallback / benchmark) + test
- [x] `retrieval/reranker.py`: cross-encoder `bge-reranker-v2-m3` lazy, rerank `retrieve_k → rerank_k`
- [x] **Benchmark reranker** (có/không rerank) — số liệu ở `docs/REPORT_retrieval.md` §2 (2026-11-07)
- [x] `evaluation/retrieval_eval.py`: **proxy metric** theo `docs/DATA_AUDIT.md §6`
- [x] Unit test dùng `QdrantClient(":memory:")`; integration test đánh dấu `slow`

**Exit criteria:** index build xong, truy vấn mẫu trả kết quả hợp lý; báo cáo retrieval có số liệu; thời gian truy vấn p95 < 300 ms (không tính rerank).

> **Trạng thái Phase 2 (2026-11-07, theo `docs/REPORT_retrieval.md`):** exit criteria cho hybrid search **đã đạt** (p95 = 43.07 ms < 300 ms). Còn hai nút thắt: **rerank p50 14,652 ms / p95 19,533 ms** (`retrieve_k=50`) và **Hit Rate@5 proxy = 6.78%** (trần Hit Rate@50 trước rerank = 10.51%). → Mở **Phase 2.5**. Task còn mở của Phase 2: _Benchmark embedding_.

---

## Phase 2.5 — System Optimization (M2.5)

> **Lý do mở phase (2026-11-08):** `REPORT_retrieval.md` (2026-11-07) cho thấy hai nút thắt: độ trễ rerank và Hit Rate thấp do lệch miền ngôn ngữ giữa claim và văn phong báo chí.

> **Quyết định PM đã chốt (2026-11-08):**
>
> 1. Model query rewrite: **`gemini-3.1-flash-lite`** (validator họ Flash chấp nhận; chặn `gemini-1.5-*`).
> 2. **13 RSS feed** (8 VnExpress + 5 Dân Trí) trong `configs/rss.yaml`.
> 3. Mục tiêu latency rerank Phase 2.5: **p95 ≤ 10 s**. Ngưỡng ≤ 3 s chuyển sang _Future Enhancements_ (ONNX). **Không hạ `retrieve_k` dưới 20** để đạt ngưỡng nếu chưa có Decision Log + PM duyệt.
> 4. `index_version` = snapshot nền (tĩnh); lớp RSS theo dõi bằng `rss_last_updated` + `dynamic_chunks` trong `index_manifest.json`.
> 5. Trường thời gian: `date` tĩnh → `crawl_date`; RSS `pubDate` → `published_date` (ISO-8601 UTC). Chấp nhận **không lọc được theo thời gian** trên toàn KB; task lọc thời gian ở Backlog.
>
> **Giao trước hai task:** _Task 1_ (profiling rerank) và _Task 2_ (xác minh model Gemini).

| Chỉ số (dev, n=723)                                     |   Baseline Phase 2 | Nguồn     |
| ------------------------------------------------------- | -----------------: | --------- |
| Hybrid Search p95 (top-50)                              |           43.07 ms | REPORT §1 |
| Rerank p50 / p95 (50 → 5)                               | 14,652 / 19,533 ms | REPORT §1 |
| Hit Rate@5 proxy sau rerank                             |             0.0678 | REPORT §2 |
| Hit Rate@50 proxy trước rerank (trần recall của rerank) |             0.1051 | REPORT §2 |
| Hit Rate@5 strict gold (theo dõi drift, 3-gram ≥ 0.50)  |             0.0041 | REPORT §4 |

**Giả thuyết cần kiểm chứng (chưa phải kết luận):**

- **H1:** giảm `retrieve_k` 50 → 20 làm latency rerank giảm gần tuyến tính; đổi lại trần recall có thể giảm.
- **H2:** LLM Query Rewrite chuyển claim lóng/ngắn thành từ khóa báo chí ⇒ tăng khớp từ vựng với KB. Không tạo ra bằng chứng không có trong KB.
- **H3:** RSS tăng độ phủ **tin mới**. **Không** kỳ vọng tăng proxy trên dev ViFactCheck (evidence ViFactCheck gần như vắng trong nguồn báo này — `DATA_AUDIT.md` §6).

**A. Latency & `retrieve_k`**

- [x] **[Task 1 — giao trước]** Profiling rerank trước khi tối ưu: ghi rõ thiết bị (CPU/GPU), dtype, batch size, `max_length` cặp (claim, chunk) của lần đo 14.6 s
- [x] Đặt `retrieve_k = 20` trong `configs/base.yaml` (không hard-code); chạy lại `evaluation/retrieval_eval.py` với k ∈ {20, 50} cùng cấu hình
- [x] Ghi Hit Rate@{1,3,5,20} + latency rerank p50/p95 vào mục "Phase 2.5" của `docs/REPORT_retrieval.md`; nếu Hit Rate@5 giảm so với k=50 ⇒ ghi Decision Log, cân nhắc k trung gian (vd. 30)
- [x] Nếu p95 rerank đo thực tế ở k=20 vẫn > 10 s: tối ưu tham số suy luận (dtype, batch size, `max_length`) trước. **Không hạ `retrieve_k` dưới 20** để đạt ngưỡng nếu chưa có Decision Log + PM duyệt (mục tiêu ≤ 3 s thuộc Future Enhancements)
- [x] **[Task 5a — k-sweep sạch]** Chạy k-sweep trên full dev (723 claims) với `k ∈ {20, 30, 50}` ở `fp32, batch_size=4`; chứng minh `k=30` đạt p95 = 1.88 s << 10 s, Hit@1=31/723 cao nhất, trần 72/723. **PM chính thức chốt `retrieve_k = 30`** (2026-11-08).

**B. LLM Query Rewrite (Gemini họ Flash)**

- [x] **[Task 2 — giao trước]** Xác minh `gemini-3.1-flash-lite` (PM chốt 2026-11-08) với API key thực tế: `models.get`/`models.list` + 1 request thử; ghi quota/RPM free tier thực tế. Nếu không khả dụng ⇒ **không tự đổi model**, báo PM. **Không dùng `gemini-1.5-*` (đã bị Google ngừng)**
- [x] Thêm dependency (`google-genai`…) vào `pyproject.toml` kèm lý do; thêm `GEMINI_API_KEY` vào `.env.example`
- [x] `configs/base.yaml: query_rewrite` (`enabled`, `model`, `timeout_s`, `max_retries`, `max_output_tokens`, `max_keywords`, `fusion`, `cache_dir`, `prompt_version`) + validator Pydantic (allowlist họ Flash, chặn 1.5); `model` mặc định `gemini-3.1-flash-lite`, có test xác nhận validator chấp nhận id này
- [x] `configs/prompts/query_rewrite.txt`: few-shot lóng → từ khóa báo chí; **bắt buộc có câu** `Tuyệt đối CHỈ in ra từ khóa. KHÔNG chào hỏi, KHÔNG dạ vâng, KHÔNG giải thích.`; claim đặt trong delimiter và coi là dữ liệu
- [x] `retrieval/query_rewriter.py`: `QueryRewriter` Protocol, `GeminiQueryRewriter` (lazy client, timeout, retry/backoff, rate limiter, disk cache), `NoOpQueryRewriter`
- [x] `sanitize_keywords()`: một dòng, bỏ markdown/ngoặc/tiền tố, chặn cụm chào hỏi/giải thích (danh sách trong config); lỗi ⇒ fallback claim gốc + log `rewrite_status`
- [x] Tích hợp vào luồng truy vấn (`pipeline/verifier.py` / script eval): rewrite → hybrid search; reranker và classifier **luôn dùng claim gốc**
- [x] Unit test (mock client, không mạng): sanitizer, fallback khi timeout/429/output rác, cache hit, validator từ chối `gemini-1.5-*`, template chứa câu quy tắc
- [x] A/B trên dev: `original` vs `rewritten` vs `original+rewritten` (RRF) — tạo gói gán nhãn mù 60 claims (C1) tại `artifacts/ab_testing/human_label_pack_60.csv`, tính proxy metric, latency p50/p95, lưu manifest `manifest_c1.json` và xuất báo cáo `docs/REPORT_ab_c1.md`.
- [x] Ghi chú quyền riêng tư: claim được gửi tới Google; kiểm tra điều khoản free tier (dữ liệu có thể được dùng để cải thiện sản phẩm) trước khi dùng với dữ liệu ngoài tập công khai

**C. On-demand RSS Updater (Pipeline D)**

- [x] Chốt danh sách feed ⇒ `configs/rss.yaml` (13 feed; PM chốt 2026-11-08)
- [x] Kiểm tra từng feed: truy cập (HTTP 200, đúng định dạng RSS) + `robots.txt`/điều khoản — URL do PM cung cấp, **chưa xác minh**
- [x] Audit RSS bằng `scripts/01d_audit_rss.py` (xem `docs/DATA_AUDIT.md` §2b)
- [x] `schemas/kb.py`: thêm `published_date` (optional) và `ingest_source` (`static|rss`) — tương thích ngược; cập nhật payload chunk. `crawl_date` nhận alias `date` của file tĩnh (file raw không sửa); `published_date` chuẩn ISO-8601 UTC từ `pubDate`
- [x] `data/rss_cleaner.py`: làm sạch đặc thù RSS (bỏ title trùng trên dòng 1, sửa lỗi dính chữ địa danh `location-stamp` của VnExpress thành `Địa danh - Tiêu đề`, giữ `(Dân trí) - ` đồng bộ KB tĩnh)
- [x] `data/rss_updater.py` & `scripts/05_update_kb_rss.py`: fetch feed, 3 tầng dedup URL (trong session, với KB tĩnh, với `rss_manifest.json`), rate limit, trích full text qua `trafilatura`, chunking, embed dense + sparse, `VectorStore.upsert` (idempotent; tuân thủ nghiêm ngặt **Rule D8**: chặn ghi đè vào collection chính `kb_chunks_8b525e25ad54`)
- [x] Ghi `dynamic_chunks`, `rss_last_updated` vào `artifacts/rss_manifest.json`
- [x] Test: `test_rss_cleaner.py` và `test_rss_updater.py` (mock fetcher, 2 lần chạy chứng minh idempotent thêm 0 bản sao, Rule D8 bảo vệ collection chính)

**D. Đánh giá & tài liệu**

- [x] Gói gán nhãn thủ công 60 claim (theo `DATA_AUDIT.md` §6 và Brief #3 §C1) được tạo tại `artifacts/ab_testing/human_label_pack_60.csv` (20 Real, 20 Fake, 20 Unverifiable; seed=42; ẩn danh & xáo trộn 3 nhánh); PM đã hoàn thành gán nhãn 451 candidate chunks (`artifacts/ab_testing/human_label_pack_60_labeled.csv`). Phát hiện thực nghiệm: 447/451 chunk (99.1%) bị bác bỏ do lệch pha thời gian (claims ViFactCheck 2020-2023 vs KB 2026). **PM chính thức chốt Hướng 1: Contemporary News Fact-Checker (~2026+)** (2026-11-09).
- [x] Cập nhật `docs/REPORT_retrieval.md` (trước/sau Phase 2.5), `docs/REPORT_gemini_verify.md` (v1/v2/v3), `docs/REPORT_ab_c1.md` và Decision Log trong `docs/ARCHITECTURE.md`.

**Exit criteria Phase 2.5:**

- Hybrid p95 vẫn < 300 ms (`43.07 ms`).
- Latency rerank p95 giảm so với baseline (19,533 ms) -> đạt `1,878.53 ms` ở `k=30, fp32, batch_size=4` (thỏa mãn sâu mục tiêu `p95 ≤ 10 s`).
- Có số liệu A/B query rewrite; `rewritten_only` phát hiện thêm bằng chứng hợp lệ (`item_0171`) so với baseline; disk cache & rate limit bảo vệ quota.
- Gemini lỗi/timeout **không** làm hỏng pipeline (test phủ 100% fallback scenarios).
- RSS chạy end-to-end ≥ 1 lần: thêm được bài mới vào Qdrant; chạy lại không tạo bản sao (idempotent, 0 chunks added on rerun).
- **Trạng thái Phase 2.5:** **⏸ đóng băng do chuyển hướng (2026-11-09)** (không phải `✅`).
- **Danh sách các hạng mục đóng băng (Brief #4 §1 — không làm nữa, không xóa code):**
  - C2 (A/B 723 claim trên toàn dev);
  - Nạp RSS vào collection chính `kb_chunks_8b525e25ad54` (quy tắc D8 vẫn hiệu lực);
  - Phần bổ sung 3 claim fallback của C1;
  - Các mục A20–A22 của Brief #3;
  - ONNX export, dynamic chunk size, tinh chỉnh trọng số alpha;
  - FastAPI / Docker / CI (chuyển sang demo trực tiếp bằng Gradio + cache dự phòng);
  - Hard-negative mining từ Qdrant;
  - Sinh dataset 2026 bằng LLM.

---

## Phase 3 — Verifier & Thử nghiệm Tìm kiếm Web (M3 — ✅ Hoàn thành)

**Task 1 — Verifier (Phase 3, time-box vòng đầu: 3 ngày)**

- [x] Đo độ dài token thật (task Phase 0 còn mở): báo p50/p95/p99 của cặp `(Statement, evidence)` sau tách từ cho cả `gold` (`p50=69.0, p95=122.0, p99=160.2, max=248`) và `selected` (`p50=132.0, p95=179.0, p99=205.0, max=344`); chốt `max_length = 256` cho PhoBERT (`truncation="only_second"`, bao phủ 100% cặp gold và 99.93% cặp selected không bị cắt)
- [x] `retrieval/evidence_selector.py` & `scripts/09_select_evidence.py`: tách câu từ `Context`, chọn top-3 câu giữ thứ tự gốc (tổng ≤ ~150 token; dùng unigram+bigram BM25 do reranker ước tính tốn ~93 phút cho 124,000 cặp câu trên GTX 1660); cache vào `data/interim/selected_{train,dev}.jsonl`; tỉ lệ dòng dev có 3-gram containment của gold `Evidence` ≥ 0.5 đạt **93.78% (678/723)** (`≥ 0.3` đạt **96.40%**)
- [x] `models/tokenization.py`: mã hóa cặp `(Statement, evidence_text)` (PhoBERT tách từ, XLM-R văn bản thô, `truncation="only_second"`) và hỗ trợ chế độ **Baseline chỉ `Statement`** (không bằng chứng)
- [x] `models/classifier.py`: wrapper `AutoModelForSequenceClassification` cho `vinai/phobert-base-v2` & `xlm-roberta-base` theo config (`num_labels = 3`)
- [ ] Huấn luyện XLM-R (hoãn — ưu tiên PhoBERT + Tavily theo Brief #4 và Brief #5)
- [ ] Quét siêu tham số Verifier (hoãn — cố định cấu hình chống OOM fp32 trên GTX 1660 theo Brief #4 §8.2)
- [x] `training/metrics.py`: accuracy, macro-F1, P/R/F1 từng lớp (`real/fake/unverifiable`), confusion matrix, ECE (Expected Calibration Error)
- [x] `models/calibration.py`: temperature scaling **chỉ fit trên dev**; báo ECE trước/sau; đầu ra là softmax đã hiệu chỉnh (`real/fake/unverifiable` + xác suất %)
- [x] `training/train.py` & `scripts/10_train_verifier.py` (+ `configs/train_phobert.yaml`, `configs/train_xlmr.yaml`): hỗ trợ 4 biến thể (`V-A` gold, `V-B` selected, `V-C` trộn 50/50, `Baseline` chỉ claim); chạy trên PC GTX 1660 ở `fp32` (`fp16=false`, `lr=2e-5`, effective batch 32, 4 epochs, warmup 10%, weight decay 0.01); lưu checkpoint + config + temperature vào `artifacts/models/<run_id>/`
- [x] `scripts/pack_for_colab.py`: đóng gói mã nguồn + `data/processed/*` + `data/interim/selected_*` kèm notebook ngắn dự phòng
- [x] Chạy huấn luyện vòng đầu (`Baseline` chỉ claim: Macro-F1 = `0.5644`, Accuracy = `56.98%`; `V-A` PhoBERT: dev gold Macro-F1 = `0.8233`, Accuracy = `82.30%`, dev selected Macro-F1 = `0.6842`, Peak VRAM = `3,349.96 MB` (~3.35 GB / 6 GB), ECE `0.0846 -> 0.0821`) và báo cáo đầy đủ tại `docs/REPORT_baseline.md` (confusion matrix, phân tích 10 lỗi tiêu biểu)
- [x] Huấn luyện biến thể `V-B` (Selected Evidence từ `Context`) trên GTX 1660 ở `fp32` (seed 42): dev gold Macro-F1 = `0.7419`, dev selected Macro-F1 = `0.7520`, Peak VRAM = `3,371.34 MB`
- [x] Huấn luyện biến thể `V-C` (Trộn 50/50 Gold + Selected) trên GTX 1660 ở `fp32` (seed 42): dev gold Macro-F1 = `0.8062`, dev selected Macro-F1 = `0.7501`, F1 trung bình = `0.7782`, Peak VRAM = `3,373.84 MB`
- [x] So sánh đối đầu `V-A`, `V-B`, `V-C` trên cả dev gold và dev selected; chọn ra biến thể tối ưu nhất: **`V-C` được chọn** (duy trì F1 cao trên Gold `0.8062` đồng thời nâng vọt F1 trên Selected từ `0.6842` lên `0.7501`)
- [x] Chạy 3 seed (42, 43, 44) cho cấu hình tốt nhất `V-C`: Dev Gold Macro-F1 = **`0.8075 ± 0.0122`** (Accuracy = **`0.8073 ± 0.0125`**), Dev Selected Macro-F1 = **`0.7493 ± 0.0009`** (Accuracy = **`0.7492 ± 0.0008`**), ECE = **`0.0843 ± 0.0049`**
- [x] Cập nhật kết quả cuối cùng vào báo cáo `docs/REPORT_baseline.md`

**Task 2 — Thử nghiệm tìm kiếm web Tavily (time-box: 1 ngày)**

- [x] `configs/search.yaml` (+ `SearchConfig` trong `config.py`): `provider: tavily`, `search_depth: basic`, `max_results: 5`, `topic: general`, không giới hạn thời gian, `max_credits_spike: 150`, `timeout_s: 15.0`, `include_domains` (16 tên miền báo chính thống)
- [x] `retrieval/web_search.py` & `scripts/11_run_tavily_spike.py`: gọi REST Tavily bằng `httpx`, đọc `TAVILY_API_KEY` từ `.env`, cache mọi phản hồi thô vào `artifacts/cache/web_search/`, kiểm tra trần credit (`max_credits_spike: 150`), khẳng định mọi URL thuộc `include_domains`, trích xuất văn bản (trafilatura / nội dung Tavily)
- [x] Chạy thử trên **30 claim dev** (10 mỗi lớp, seed 42) với truy vấn **Q0** (claim ≤ 400 ký tự) và **Q1** (từ khóa theo luật); báo cáo tỉ lệ hit 3-gram containment ≥ 0.5 (**Q0: 6/30 = 20.00%**, **Q1: 9/30 = 30.00%**) và ≥ 0.3 (**Q0: 8/30 = 26.67%**, **Q1: 12/30 = 40.00%**) theo lớp nhãn, số credit đã dùng (`60/150`), độ trễ `p50 = 4,705.05 ms / p95 = 9,092.76 ms`, và phân tích **5 claim trúng + 5 claim trượt** tại `docs/REPORT_tavily_spike.md`

**Task 3 — Đồng bộ tài liệu & dọn nhỏ**

- [x] Đồng bộ `docs/ARCHITECTURE.md` (§1, §3, §5, §7, §8, §9), `docs/ROADMAP_TODO.md` (Sprint cuối, đóng băng Phase 2.5, mở Phase 3), `.github/copilot-instructions.md` (§4 quy tắc dùng `Context`)
- [x] **A18:** cập nhật bảng đối chiếu output thô v1/v2/v3 trong `docs/REPORT_gemini_verify.md`
- [x] **A19:** bổ sung `total_deadline_s` và tách `max_retries` giữa batch/online (online mặc định không retry khi timeout) trong `config.py`, `query_rewriter.py` + unit test

**Exit criteria Phase 3 (✅ ĐÃ HOÀN TẤT & ĐÓNG CHÍNH THỨC):**

- Số liệu dev cho từng biến thể và baseline: Baseline (F1 `0.5644`), V-A (Gold F1 `0.8233`, Selected F1 `0.6842`), V-B (Gold F1 `0.7419`, Selected F1 `0.7520`), V-C (Gold F1 `0.8062`, Selected F1 `0.7501`).
- Cấu hình tốt nhất (`V-C`) chạy đủ 3 seed (`42, 43, 44`): Dev Gold Macro-F1 = **`0.8075 ± 0.0122`** (Accuracy = `0.8073 ± 0.0125`), Dev Selected Macro-F1 = **`0.7493 ± 0.0009`** (Accuracy = `0.7492 ± 0.0008`), ECE = `0.0843 ± 0.0049`, kèm ma trận nhầm lẫn và 10 ca lỗi có độ tin cậy cao nhất trích xuất trực tiếp từ `eval_report.json` (xem `docs/REPORT_baseline.md`).
- Báo cáo thử nghiệm Tavily 30 claim hoàn tất; Cổng quyết định ĐẠT (**`30.00%`** $\ge 0.5$ và **`40.00%`** $\ge 0.3$).
- Cổng G1–G3 và Task C0 đã được giải quyết triệt để; Kỹ sư trưởng và PM chính thức duyệt đóng M3 ngày 2026-11-10 (Bản vá 1).
- **Trạng thái Phase 3:** **✅ ĐÃ ĐÓNG HOÀN TOÀN** (theo xác nhận của Kỹ sư trưởng và PM ngày 2026-11-10; mục 11.3 chuyển sang Phase 4).

---

## Phase 4 — Pipeline đầu-cuối (live), độ bền của verifier, demo (M4 — 🔄 Đang mở: Phase 4 GO theo Brief #5)

- [x] **[Task 4.0]** Chia tập tách biệt theo bài báo trên ViFactCheck (`726` Train / `103` V / `207` H bài, băm `Context`) và huấn luyện V-C (seed 42): Dev/V & Holdout `H` giữ nguyên hiệu năng (`0.7994` gold, `0.7490` selected)
- [x] **[Task 4.1 & 4.1b]** Tạo bộ bằng chứng lệch (`random`, `hard_bm25` trước/sau lọc overlap 3-gram `< 0.3`, `same_article`) + đánh giá 3 seed V-C + huấn luyện đối chứng V-D (giữ nguyên **V-C** làm mô hình chốt theo quy tắc chọn)
- [x] **[Task 4.2 & `gate-v1`]** `retrieval/reranker.py` (`bge-reranker-v2-m3`) + `pipeline/decision.py` (`DecisionGate`, `EnsembleVerifier` 3 seed V-C có verify `sha256`, `show_probs=False` khi bị cổng chặn); chốt ngưỡng `τ_rel = 0.7559` (Youden trên dev, ROC-AUC `0.9265` tổng hợp / `0.9753` hard BM25 lọc / `0.8057` same article) và `τ_conf = 0.55`, ECE in-sample (`0.1022` gold / `0.0643` selected); gắn tag `gate-v1`
- [x] **[Task 4.3]** `retrieval/evidence_selector.py` (gộp BM25 top-3 ≤ 150 từ giữ `{text, url, domain}`) + `build_keyword_query_v2` (Q1v2 giữ từ ghép, từ phủ định & lọc hư từ ghép 2 âm tiết) + `pipeline/live_verifier.py` (Tavily song song tối đa 3 trang → cache → KB cục bộ lazy `bge-m3` → `unverifiable`, deadline ≤ 20 s, `DEMO_MODE`, không nạp `bge-m3` trong đường live chính)
- [x] **[Task 4.4 — Code UI & Script Smoke Test]** `app/demo.py` (Gradio tiếng Việt: claim, 6 mẫu có cache, `DEMO_MODE`, tiến trình, ẩn xác suất khi `show_probs=False`, link nguồn, độ trễ từng bước, ghi chú §6.2) + `scripts/25_smoke_test_laptop.py` + `tests/unit/test_app_demo.py` (đã verify trên PC GTX 1660; chờ chạy 1 lệnh trên laptop RTX 3050 4 GB vật lý)
- [ ] **[Benchmark 2026 / E2]** Xây dựng và thẩm định bộ **Contemporary Claims Benchmark (2026)** (tách `tune` và `test`, kiểm chỉnh ngưỡng trên `tune`), đánh giá E2 đầu-cuối kèm tỉ lệ tìm đúng bài nguồn & 10 ca phân tích lỗi
- [ ] **[E1 & E3]** Chạy `test_clean.jsonl` **đúng 1 lần** (`--final`) với ensemble V-C chốt + thử nghiệm tham khảo E3 (~100 claim `test` qua live pipeline khi PM duyệt ngân sách)

**Exit criteria:** Pipeline live + demo `app/demo.py` chạy ổn định trên PC & laptop 4 GB VRAM; E1–E3 và ablation được báo cáo đầy đủ trước khi đóng băng (`demo-freeze`).

---

## Phase 5 — Serving & demo (M5)

- [ ] `schemas/api.py`: `VerifyRequest`, `VerifyResponse`, `Evidence`
- [ ] `serving/app.py`: FastAPI, nạp model/index ở `lifespan`, DI
- [ ] Endpoint `POST /v1/verify`, `GET /healthz`, `GET /v1/version`
- [ ] Endpoint admin `POST /v1/admin/refresh-kb` (API key, nội bộ) kích hoạt RSS updater (Pipeline D); `GET /v1/version` trả thêm `rss_last_updated`
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

## Future Enhancements (ý tưởng dự phòng — phục vụ báo cáo bảo vệ)

> Chưa xếp lịch. Mỗi mục nêu giả thuyết, cách làm và tiêu chí để kết quả dùng được trong báo cáo.

- [ ] **(1) Reranker sang ONNX để tăng tốc suy luận trên CPU / thiết bị khác (Đã hạ ưu tiên)**
  - **Trạng thái (2026-11-08):** mục tiêu `p95 ≤ 3 s` **đã đạt trên thiết bị đo PC R&D (GTX 1660)** với PyTorch `fp32 + batch_size=4, retrieve_k=20` (`p50 = 1,005.78 ms`, `p95 = 1,369.19 ms` — xem `docs/REPORT_retrieval.md` §5.3). Hạ ưu tiên; giữ lại cho môi trường CPU-only hoặc thiết bị khác.
  - Cách làm: export `bge-reranker-v2-m3` sang ONNX (`optimum`), thử quantization int8 động, chạy bằng ONNX Runtime.
  - Đo: latency p50/p95 so với PyTorch (cùng thiết bị, batch, `retrieve_k`); độ lệch thứ hạng (overlap top-5, Spearman) và Hit Rate proxy.
  - **Mục tiêu (PM chốt 2026-11-08): rerank p95 ≤ 3 s**; chấp nhận khi thứ hạng top-5 gần như không đổi và đạt mục tiêu latency.
- [ ] **(2) Dynamic Chunk Size — so sánh mức bảo toàn ngữ cảnh**
  - Cách làm: so chunk cố định (3–5 câu, overlap 1) với chunk co giãn theo ranh giới đoạn/ngữ nghĩa; cần re-index sang collection mới theo `index_version`.
  - Đo: Hit Rate proxy; tỉ lệ câu liên quan nằm trọn trong cùng chunk (spot-check thủ công ~50 mẫu); số chunk và dung lượng index.
- [ ] **(3) Tinh chỉnh trọng số Alpha (vector vs sparse) cho Hybrid Search**
  - Lưu ý: RRF server-side mặc định không có trọng số alpha ⇒ dùng fusion client-side ở `retrieval/hybrid.py` (hoặc kiểm bản Qdrant có hỗ trợ fusion có trọng số).
  - Cách làm: quét α ∈ {0.0, 0.1, …, 1.0} trên **dev**, chọn theo Hit Rate proxy sau rerank; ghi rõ nguy cơ overfit vào proxy metric.

---

## Backlog (chưa xếp lịch)

- [ ] **Lọc theo thời gian trên toàn KB**: hiện KHÔNG thể do dữ liệu tĩnh chỉ có `crawl_date` (PM chấp nhận 2026-11-08). Cần crawl lại KB có `published_date` + payload index; hiện chỉ phần RSS có `published_date`
- [ ] Mở rộng KB sang thêm nguồn báo chính thống & trang fact-check
- [ ] Fine-tune embedding/reranker trên cặp `(Statement, Evidence)` của ViFactCheck
- [ ] Claim decomposition cho câu nhiều mệnh đề
- [ ] Cache kết quả truy vấn phổ biến
