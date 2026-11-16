# DATA_AUDIT.md — Kiểm toán dữ liệu

> **Trạng thái (cập nhật 2026-11-08):** audit Phase 0 hoàn tất (thống kê §2–§4, leakage §5, overlap §6). **Mới:** §2b nguồn động **On-demand RSS** — **ĐÃ AUDIT** (`scripts/01d_audit_rss.py`, 13 feed, 980 item, 260 bài full-text). §6.1 đối chiếu với kết quả Phase 2 (`REPORT_retrieval.md`).

## 1. Phần cứng

| Máy | VRAM | Ghi chú |
|---|---|---|
| Laptop (chạy code / dev) | 4 GB | Suy luận, test, giao diện |
| PC (huấn luyện) | 6 GB | Fine-tune với `fp16` + gradient accumulation |

⇒ **Chỉ dùng model `base`** (PhoBERT-base-v2 / XLM-R-base). Không dùng `large`.

## 2. KB tĩnh (Dân Trí + VnExpress)

| Nguồn | Bài | Text rỗng | URL trùng | Text mean (chars) | Text p95 (chars) |
|---|---:|---:|---:|---:|---:|
| Dân Trí | 1 000 | 0 | 0 | 3 509.3 | 7 442.6 |
| VnExpress | 2 053 | 0 | 0 | 3 507.0 | 7 142.6 |
| **Tổng** | **3 053** | 0 | 0 | ~3 508 | ~7 300 |

Tổng ~**10.7 M chars**; sau cleaning + dedup sẽ giảm nhẹ.
Ước lượng chunk ban đầu (`~120–160` token, cửa sổ trượt 3–5 câu): **~6–10 chunk/bài** → **~20–30 K chunk**. **Thực tế sau Phase 1: 41 466 chunks** (cao hơn ước lượng).
Kích thước này **rất nhỏ** với Qdrant (vài chục MB sau embedding 1024-d) ⇒ **Qdrant local/Docker** là đủ, không cần cluster.

### Trường `date` của KB tĩnh (sự cố dữ liệu lịch sử — PM chốt 2026-11-08)

Trong các file `.jsonl` tĩnh, `date` thực chất là **ngày cào**, **không phải** ngày bài viết được tạo. Quyết định: đổi tên thành `crawl_date` (alias ở schema/loader; file raw do DVC quản lý không sửa tại chỗ). Với nguồn động (RSS), `pubDate` được map vào `published_date` (ISO-8601 UTC). Hệ quả chấp nhận: **không thể lọc theo thời gian trên toàn KB**; task lọc thời gian ở Backlog (`ROADMAP_TODO.md`).

## 2b. Nguồn dữ liệu động — On-demand RSS Feeds (Dân Trí, VnExpress)

**Phân loại:** luồng **Dynamic (động)**, **bổ trợ** cho tập `.jsonl` tĩnh ở §2 — không thay thế.

| Thuộc tính | KB tĩnh (`.jsonl`) | RSS on-demand (động) |
|---|---|---|
| Nguồn | Dân Trí, VnExpress (đã crawl) | Cùng hai báo, qua RSS |
| Cách nạp | Pipeline A (offline, build một lần) | Pipeline D (khi được gọi: CLI / endpoint admin) |
| Quy mô | 3 053 bài → 41 466 chunk | **980 item / lần quét 13 feed** → **946 URL duy nhất** (VnExpress: 60 item/feed × 8; Dân Trí: 100 item/feed × 5) |
| Trường thời gian | Chỉ `crawl_date` (không phải ngày đăng) | `published_date` (từ `pubDate`, 100% hợp lệ ISO-8601 UTC) + `crawl_date` |
| Văn bản | Có sẵn `text` | RSS chỉ có tiêu đề + mô tả ngắn; trích full text từ trang bài viết bằng `trafilatura` đạt **100% (260/260 bài)** |
| `label` | `real` | `real` (cùng nguồn chính thống) |
| Vai trò | KB truy xuất | KB truy xuất; **không** dùng huấn luyện, **không** dùng chấm proxy trên dev |
| Trạng thái audit | Đã audit | **Đã audit (`2026-11-08T11:54:23Z`)** |

**Ánh xạ trường RSS → KB record** (`ARCHITECTURE.md` §4.1b):

| Trường RSS / nguồn | Trường KB |
|---|---|
| `link` | `url` |
| `guid` (hoặc hash của URL) | `id` |
| `title` | `title` |
| Full text trích từ trang bài viết | `text` (không dùng `description` làm `text` nếu đã lấy được full text) |
| `pubDate` | `published_date` (ISO-8601 UTC) |
| Thời điểm fetch | `crawl_date` |
| (cố định) | `label="real"`, `ingest_source="rss"` |

**Kết quả Audit RSS thực tế** (ĐÃ ĐO — `scripts/01d_audit_rss.py`, lưu tại `artifacts/audit_rss/rss_audit_summary.json` và `artifacts/audit_rss/manual_spotcheck_20_articles.json`):

- **Thời điểm đo:** `2026-11-08T11:54:23.246447+00:00`
- **Tổng số feed:** `13` (`8` VnExpress + `5` Dân Trí), `user_agent = "vnfake-research-bot/0.1 (contact: sonbeo2015@gmail.com)"`, `min_delay_between_requests_s = 1.0 s`.
- **Kiểm tra `robots.txt`:** cả `https://vnexpress.net/robots.txt` và `https://dantri.com.vn/robots.txt` đều **CHO PHÉP (`allowed`)** đối với toàn bộ 13 feed URL và 260 URL bài viết kiểm tra (`0` bài bị chặn).
- **Quy tắc chuẩn hóa URL khi đếm trùng (`canonicalize_url` tại `src/vnfake/data/rss_audit.py` — A6):**
  1. `scheme` chỉ chấp nhận `http`/`https` và đồng nhất về chữ thường `https`.
  2. `host` (`netloc`) chuyển về chữ thường, loại bỏ cổng mặc định (`:80`, `:443`) và tiền tố `www.` hoặc `amp.`.
  3. Loại bỏ hoàn toàn phần `query` (`?utm_source=...`) và `fragment` (`#...`).
  4. Phần `path` gộp các dấu `/` liên tiếp, cắt bỏ dấu `/` cuối, cắt hậu tố `/amp` hoặc `.amp` (không phân biệt hoa thường), rồi cắt `/` cuối một lần nữa.
- **Tổng số item thô / URL duy nhất:** `980` item thô → `946` URL duy nhất toàn cục.
- **Trùng URL nội bộ trong từng feed:** `0 / 980` (`0.0%`).
- **Chồng lấn giữa `dantri_tin_moi_nhat` và 4 feed chuyên mục Dân Trí khác:** **`33 / 100` bài (`33.0%`)** của `dantri_tin_moi_nhat` xuất hiện đồng thời ở `suc-khoe` (`7` bài), `the-thao` (`9` bài), `giai-tri` (`7` bài), và `phap-luat` (`10` bài). Giữa `vnexpress_thoi_su` và `vnexpress_du_lich` cũng có `1 / 60` bài (`1.67%`) trùng chéo. ⇒ Khẳng định bắt buộc phải dedup URL qua `rss_manifest.json` và bộ nhớ tạm trong phiên.
- **Trùng URL với KB tĩnh (`3,053` bài):** `12 / 13` feed có tỉ lệ trùng `0.0%`; riêng **`vnexpress_du_lich` trùng `16 / 60` bài (`26.67%`)** với KB tĩnh. ⇒ Khẳng định bắt buộc phải đối chiếu URL với KB tĩnh **trước** khi tải trang bài viết để tiết kiệm băng thông.
- **Tỉ lệ Near-duplicate (MinHash) với KB tĩnh trên các bài mới:** chưa đo toàn bộ `946` bài (chỉ fetch tối đa `20` bài/feed theo giới hạn lịch sự của audit).

| Feed ID | Nguồn | Chuyên mục | `robots.txt` | HTTP | Content-Type | Số item | `pubDate` hợp lệ | `link` hợp lệ | Trùng trong feed | Trùng chéo feed | Trùng KB tĩnh | Full-text thành công (trên 20 bài) | Sống sót sau `cleaning` (≥100 chars) | Rỗng / nhiễu | Độ dài `mean` (chars) | Độ dài `p95` (chars) | Dính mẫu `"Xem thêm"` / bình luận |
|---|---|---|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vnexpress_thoi_su` | vnexpress | `thoi-su` | `allowed` | 200 | `application/xml; charset=utf-8` | 60 | 100% (60) | 100% (60) | 0.0% (0) | 1.67% (1) | 0.0% (0) | 100% (20/20) | 100% (20/20) | 0.0% (0) | 2,935.3 | 5,583.5 | 0 |
| `vnexpress_khoa_hoc_cong_nghe` | vnexpress | `khoa-hoc-cong-nghe` | `allowed` | 200 | `application/xml; charset=utf-8` | 60 | 100% (60) | 100% (60) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 100% (20/20) | 100% (20/20) | 0.0% (0) | 4,423.9 | 8,133.2 | 0 |
| `vnexpress_the_thao` | vnexpress | `the-thao` | `allowed` | 200 | `application/xml; charset=utf-8` | 60 | 100% (60) | 100% (60) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 100% (20/20) | 100% (20/20) | 0.0% (0) | 4,773.6 | 8,722.8 | 0 |
| `vnexpress_oto_xe_may` | vnexpress | `oto-xe-may` | `allowed` | 200 | `application/xml; charset=utf-8` | 60 | 100% (60) | 100% (60) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 100% (20/20) | 100% (20/20) | 0.0% (0) | 2,147.7 | 3,920.0 | 0 |
| `vnexpress_giai_tri` | vnexpress | `giai-tri` | `allowed` | 200 | `application/xml; charset=utf-8` | 60 | 100% (60) | 100% (60) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 100% (20/20) | 100% (20/20) | 0.0% (0) | 3,158.5 | 5,373.9 | 0 |
| `vnexpress_phap_luat` | vnexpress | `phap-luat` | `allowed` | 200 | `application/xml; charset=utf-8` | 60 | 100% (60) | 100% (60) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 100% (20/20) | 100% (20/20) | 0.0% (0) | 3,155.5 | 5,473.4 | 0 |
| `vnexpress_du_lich` | vnexpress | `du-lich` | `allowed` | 200 | `application/xml; charset=utf-8` | 60 | 100% (60) | 100% (60) | 0.0% (0) | 1.67% (1) | **26.67% (16)** | 100% (20/20) | 100% (20/20) | 0.0% (0) | 2,938.2 | 4,359.5 | 0 |
| `vnexpress_doi_song` | vnexpress | `doi-song` | `allowed` | 200 | `application/xml` | 60 | 100% (60) | 100% (60) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 100% (20/20) | 100% (20/20) | 0.0% (0) | 3,450.6 | 5,679.6 | 0 |
| `dantri_suc_khoe` | dantri | `suc-khoe` | `allowed` | 200 | `application/rss+xml` | 100 | 100% (100) | 100% (100) | 0.0% (0) | 7.0% (7) | 0.0% (0) | 100% (20/20) | 100% (20/20) | 0.0% (0) | 4,160.1 | 10,938.6 | 0 |
| `dantri_the_thao` | dantri | `the-thao` | `allowed` | 200 | `application/rss+xml` | 100 | 100% (100) | 100% (100) | 0.0% (0) | 9.0% (9) | 0.0% (0) | 100% (20/20) | 100% (20/20) | 0.0% (0) | 2,749.0 | 4,052.8 | 0 |
| `dantri_tin_moi_nhat` | dantri | `tin-moi-nhat` | `allowed` | 200 | `application/rss+xml` | 100 | 100% (100) | 100% (100) | 0.0% (0) | **33.0% (33)** | 0.0% (0) | 100% (20/20) | 100% (20/20) | 0.0% (0) | 3,700.5 | 5,666.7 | 0 |
| `dantri_giai_tri` | dantri | `giai-tri` | `allowed` | 200 | `application/rss+xml` | 100 | 100% (100) | 100% (100) | 0.0% (0) | 7.0% (7) | 0.0% (0) | 100% (20/20) | 100% (20/20) | 0.0% (0) | 4,155.2 | 6,216.6 | 0 |
| `dantri_phap_luat` | dantri | `phap-luat` | `allowed` | 200 | `application/rss+xml` | 100 | 100% (100) | 100% (100) | 0.0% (0) | 10.0% (10) | 0.0% (0) | 100% (20/20) | 100% (20/20) | 0.0% (0) | 2,734.8 | 4,925.8 | 0 |

**Nhận xét soi tay 20 bài (10 VnExpress + 10 Dân Trí trong `artifacts/audit_rss/manual_spotcheck_20_articles.json`):**

1. **Độ sạch nội dung (`trafilatura==2.3.1`):** cả 20/20 bài hoàn toàn **không lẫn** menu điều hướng, hộp `"Xem thêm"` / `"Tin liên quan"`, chân trang hay khối bình luận độc giả (`residual_boilerplate_count = 0` trên toàn bộ 260 bài). Độ dài văn bản trung bình (`~3,421.7` chars) tương đương với KB tĩnh (`~3,508` chars).
2. **Chi tiết định dạng nhỏ trên VnExpress (10/10 bài):**
   - Dòng đầu tiên của văn bản do `trafilatura` trích ra lặp lại tiêu đề bài viết (`title`).
   - Với các bài có nhãn địa danh đầu sapo (`<span class="location-stamp">`), thẻ địa danh dính liền từ đầu tiên của câu mở đầu không có dấu cách (ví dụ: `"Hải PhòngHỏa hoạn bùng phát..."`, `"Quảng NinhKhu di tích Vũng Đục..."`, `"Nghệ AnDự kiến năm 2027..."`, `"Hà NộiPhường Kiến Hưng..."`).
   - Dòng cuối cùng chứa bút danh tác giả (ví dụ: `"Lê Tuyết"`, `"Gia Chính"`).
3. **Chi tiết định dạng nhỏ trên Dân Trí (10/10 bài):**
   - Dòng đầu tiên lặp lại tiêu đề bài viết (`title`).
   - Đoạn sapo bắt đầu bằng tiền tố `"(Dân trí) - "` (hoàn toàn giống định dạng của `data/raw/kb/dantri.jsonl`).
4. **Khuyến nghị kỹ thuật cho Task 6 (RSS Ingest, chờ PM duyệt):** dùng `trafilatura` làm bộ trích xuất chính; bổ sung bước hậu xử lý nhỏ bỏ dòng đầu nếu trùng khít `title` (vì `chunk_record` đã ghép `title` khi nhúng) và tách khoảng trắng sau địa danh đầu dòng VnExpress hoặc bỏ tiền tố `"(Dân trí) - "`.

**Hệ quả:**

- RSS cải thiện **độ phủ tin thời sự thực tế**, nhưng **không** kỳ vọng tăng Hit Rate trên dev ViFactCheck: bài mới không chứa evidence cho các claim ViFactCheck cũ (xem §6).
- Không dùng RSS để huấn luyện (toàn nhãn `real`, dễ gây thiên lệch) và không dùng để chấm proxy trên dev.
- `published_date` là dữ liệu thời gian đáng tin đầu tiên của hệ thống, mở đường cho lọc theo thời gian (Backlog) — nhưng chỉ cho phần động; KB tĩnh vẫn chỉ có `crawl_date`.
- Rủi ro: ToS/`robots.txt`, đổi cấu trúc HTML, bài trùng KB tĩnh, nhiễu nội dung ⇒ dedup URL + MinHash, rate limit, log `rejected`.

## 3. ViFactCheck

| Split | Dòng | Nhãn 0 (real) | Nhãn 1 (fake) | Nhãn 2 (unverifiable) | Stmt mean (chars) | Evid mean (chars) |
|---|---:|---:|---:|---:|---:|---:|
| train | 5 062 | 1 751 | 1 658 | 1 653 | 161.4 | 196.8 |
| dev | 723 | 256 | 244 | 223 | 160.2 | 196.6 |
| test | 1 447 | 508 | 468 | 471 | 161.4 | 196.6 |

Phân bố **cân bằng ~33% mỗi lớp** trên cả 3 split.
Câu ngắn (≈160 chars), evidence ngắn (≈197 chars) ⇒ PhoBERT `max_len=256` **đủ** cho hầu hết mẫu, cắt Evidence rất ít (*giả định theo ký tự — chưa đo token PhoBERT, task còn mở ở Phase 0*).

## 4. Ngữ nghĩa nhãn (đã chốt)

```yaml
label_map:
  0: real
  1: fake
  2: unverifiable
```

Xác nhận bằng cách đọc mẫu thực tế + đối chiếu tài liệu ViFactCheck. **Không còn là 2-class** ⇒ classifier train **3 lớp**, không dùng luật ngưỡng để tạo UNVERIFIABLE (xem §7).

## 5. Kiểm toán leakage

**Đã chạy** (`scripts/01c_check_leakage.py`, Phase 0). Phát hiện **4** dòng `Statement` trùng (nội bộ và chéo split). Xử lý ở Phase 1 bằng `dedup_vifactcheck_splits()` (giữ dev/test, xóa dòng trùng khỏi train): `train` 5 062 → `train_clean` **5 058**.

## 6. Overlap gold Evidence ↔ KB (ĐÃ ĐO — `scripts/01b_verify_overlap.py`)

Cấu hình đo: word **3-gram**, containment (không dùng Jaccard), split=`dev` (n=723), top-200 candidate/doc.

| Ngưỡng containment | % Evidence dev "có trong KB" |
|---|---:|
| ≥ 0.5 | **1.1%** |
| ≥ 0.6 | **0.6%** |
| ≥ 0.7 | **0.1%** |
| ≥ 0.8 | **0.1%** |
| ≥ 0.9 | **0.0%** |

**Kết luận:** gold `Evidence` của ViFactCheck **gần như KHÔNG nằm trong KB** Dân Trí/VnExpress (overlap ≥0.7 chỉ ~0.1%). Hai tập dữ liệu đến từ nguồn biên tập khác nhau — điều này kỳ vọng với fact-check corpus vs. báo điện tử.

**Hệ quả bắt buộc cho Phase 2:**

- **KHÔNG dùng Recall@k / MRR / nDCG@k trực tiếp trên gold evidence** — sẽ ra ~0 và vô nghĩa.
- Đánh giá retrieval **phải dùng proxy**:
  1. **Khớp entity/n-gram** giữa claim và chunk được retrieve (precision proxy: bao nhiêu chunk chứa entity/n-gram của claim).
  2. **Manual annotation** một tập nhỏ (~50–100 claim) để đo relevance nhị phân.
  3. **Human spot-check** top-5 cho mỗi model embedding/reranker trong benchmark Phase 2.
- Ghi rõ trong `evaluation/retrieval_eval.py` (Phase 2) rằng metric chính là **proxy**, không phải Recall@k chuẩn.
- Vẫn log Recall@k để **theo dõi drift** khi KB thay đổi, nhưng **không dùng làm tiêu chí chọn model**.

### 6.1 Đối chiếu với kết quả Phase 2 (`REPORT_retrieval.md`, 2026-11-07)

| Chỉ số (dev, n=723) | Giá trị | Nguồn |
|---|---:|---|
| Overlap containment 3-gram ≥ 0.5 (top-200 candidate) | 1.1% | §6 |
| Strict gold 3-gram ≥ 0.50 — Hybrid Hit Rate@50 | 0.69% | REPORT §4 |
| Strict gold 3-gram ≥ 0.50 — Hybrid + Rerank Hit Rate@5 | 0.41% | REPORT §4 |
| Proxy Hit Rate@50, trước rerank | 10.51% | REPORT §2 |
| Proxy Hit Rate@5, sau rerank | 6.78% | REPORT §2 |

**Diễn giải (thận trọng):**

- Hai phép đo strict (1.1% và 0.69%) cùng bậc độ lớn nhưng dùng cách sinh candidate khác nhau nên không so trực tiếp; cả hai **củng cố** kết luận gold evidence gần như vắng trong KB.
- Proxy Hit Rate@50 (10.51%) là **trần recall** của rerank ở `retrieve_k=50`. Hạ `retrieve_k` xuống 20 chỉ có thể giữ hoặc giảm trần này ⇒ Phase 2.5 phải đo Hit Rate@20.
- Query rewrite có thể cải thiện khớp từ vựng nhưng **không tạo ra bằng chứng không có trong KB**; nút thắt chủ yếu là coverage/miền.
- **Bất thường chưa giải thích được** (REPORT §3): nhãn `fake` (7.79%) và `unverifiable` (7.62%) có Hit Rate@5 cao hơn `real` (5.08%). Chưa đủ dữ liệu để kết luận nguyên nhân; nghi ngờ proxy (n-gram ≥ 0.30) còn nhiễu ⇒ cần **manual annotation 50–100 claim** (§6 mục 2) trước khi đánh giá hiệu quả query rewrite.

## 7. Trả lời các "Câu hỏi mở" (ARCHITECTURE §10)

| # | Câu hỏi | Trả lời | Căn cứ |
|---|---|---|---|
| 1 | Gold Evidence có trong KB không? | **KHÔNG** — overlap ≥0.7 chỉ ~0.1% (dev). Đánh giá retrieval phải dùng proxy (entity/n-gram + manual annotation). | §6 |
| 2 | Ngữ nghĩa 0/1 + phân bố nhãn | **0=real, 1=fake, 2=unverifiable**, cân bằng ~33 % mỗi lớp | §3, §4 |
| 3 | Kích thước KB ⇒ Qdrant local hay server? | **Qdrant Docker local** (`docker-compose.yml`), không cần server | §2 |
| 4 | GPU có không ⇒ base hay large? | **Chỉ base** (`vinai/phobert-base-v2`), `fp16=true`, `gradient_accumulation_steps=4` | §1 + `configs/base.yaml` |
| 5 | UNVERIFIABLE là nhãn huấn luyện hay luật ngưỡng? | **Nhãn huấn luyện (3-class)** — ViFactCheck đã có sẵn nhãn 2 | §3 |
| 6 | Có cần nguồn dữ liệu động? | **Có** — On-demand RSS (Dân Trí/VnExpress) bổ trợ KB tĩnh; **đã audit 13 feed** (`HTTP 200`, `robots.txt allowed`, `trafilatura` đạt 100%) | §2b |
| 7 | Có lọc được theo thời gian không? | **Không** trên toàn KB (KB tĩnh chỉ có `crawl_date`); chỉ RSS có `published_date`. Task lọc thời gian ở Backlog | §2, §2b |

## 8. Hệ quả cho các Phase sau

**Phase 2.5:** đánh giá A/B query rewrite bằng **proxy metric** (không dùng Recall@k chuẩn); đo Hit Rate@20 so với Hit Rate@50; manual annotation để hiệu chỉnh proxy.
**Dữ liệu động:** RSS chỉ vào KB truy xuất, **không** vào train/dev/test; chạy audit §2b trước khi dùng Pipeline D thường xuyên.
**Phase 3:** classifier 3 lớp (`num_labels = len(label_map) = 3`), không có giá trị hard-code.
**Phase 4:** luật `UNVERIFIABLE` ở `decision.py` **vẫn giữ** như tầng bảo hiểm (retrieval score thấp) — không thay thế nhãn lớp 2.
**Phase 4 (test set):** chạy đúng **1 lần** với `--final` như quy định.
