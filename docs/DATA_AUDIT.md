> **Trạng thái:** hoàn tất phần thống kê. Mục 6 (overlap) chờ chạy `scripts/01b_verify_overlap.py`.

## 1. Phần cứng

| Máy | VRAM | Ghi chú |

|---|---|---|

| Laptop (chạy code / dev) | 4 GB | Suy luận, test, giao diện |

| PC (huấn luyện) | 6 GB | Fine-tune với `fp16` + gradient accumulation |

⇒ **Chỉ dùng model `base`** (PhoBERT-base-v2 / XLM-R-base). Không dùng `large`.

## 2. KB (Dân Trí + VnExpress)

| Nguồn | Bài | Text rỗng | URL trùng | Text mean (chars) | Text p95 (chars) |

|---|---:|---:|---:|---:|---:|

| Dân Trí | 1 000 | 0 | 0 | 3 509.3 | 7 442.6 |

| VnExpress | 2 053 | 0 | 0 | 3 507.0 | 7 142.6 |

| **Tổng** | **3 053** | 0 | 0 | ~3 508 | ~7 300 |

Tổng ~**10.7 M chars**; sau cleaning + dedup sẽ giảm nhẹ.
Ước lượng chunk (`~120–160` token, cửa sổ trượt 3–5 câu): **~6–10 chunk/bài** → **~20–30 K chunk**.
Kích thước này **rất nhỏ** với Qdrant (vài chục MB sau embedding 1024-d) ⇒ **Qdrant local/Docker** là đủ, không cần cluster.

## 3. ViFactCheck

| Split | Dòng | Nhãn 0 (real) | Nhãn 1 (fake) | Nhãn 2 (unverifiable) | Stmt mean (chars) | Evid mean (chars) |

|---|---:|---:|---:|---:|---:|---:|

| train | 5 062 | 1 751 | 1 658 | 1 653 | 161.4 | 196.8 |

| dev | 723 | 256 | 244 | 223 | 160.2 | 196.6 |

| test | 1 447 | 508 | 468 | 471 | 161.4 | 196.6 |

Phân bố **cân bằng ~33% mỗi lớp** trên cả 3 split.
Câu ngắn (≈160 chars), evidence ngắn (≈197 chars) ⇒ PhoBERT `max_len=256` **đủ** cho hầu hết mẫu, cắt Evidence rất ít.

## 4. Ngữ nghĩa nhãn (đã chốt)

```yaml
label_map:
  0: real

  1: fake

  2: unverifiable
```

Xác nhận bằng cách đọc mẫu thực tế + đối chiếu tài liệu ViFactCheck. **Không còn là 2-class** ⇒ classifier train **3 lớp**, không dùng luật ngưỡng để tạo UNVERIFIABLE (xem §7).

## 5. Kiểm toán leakage

**Chưa chạy** `check_dup_statements()` (sẽ làm ở cuối Phase 0 sau khi có schema). Task vẫn mở.

## 6. Overlap gold Evidence ↔ KB (ĐÃ ĐO — `scripts/01b_verify_overlap.py`)

Cấu hình đo: word **3-gram**, containment (không dùng Jaccard), split=`dev` (n=723), top-200 candidate/doc.

| Ngưỡng containment | % Evidence dev "có trong KB" |
| ------------------ | ---------------------------: |
| ≥ 0.5              |                     **1.1%** |
| ≥ 0.6              |                     **0.6%** |
| ≥ 0.7              |                     **0.1%** |
| ≥ 0.8              |                     **0.1%** |
| ≥ 0.9              |                     **0.0%** |

**Kết luận:** gold `Evidence` của ViFactCheck **gần như KHÔNG nằm trong KB** Dân Trí/VnExpress (overlap ≥0.7 chỉ ~0.1%). Hai tập dữ liệu đến từ nguồn biên tập khác nhau — điều này kỳ vọng với fact-check corpus vs. báo điện tử.

**Hệ quả bắt buộc cho Phase 2:**

- **KHÔNG dùng Recall@k / MRR / nDCG@k trực tiếp trên gold evidence** — sẽ ra ~0 và vô nghĩa.
- Đánh giá retrieval **phải dùng proxy**:
  1. **Khớp entity/n-gram** giữa claim và chunk được retrieve (precision proxy: bao nhiêu chunk chứa entity/n-gram của claim).
  2. **Manual annotation** một tập nhỏ (~50–100 claim) để đo relevance nhị phân.
  3. **Human spot-check** top-5 cho mỗi model embedding/reranker trong benchmark Phase 2.
- Ghi rõ trong `evaluation/retrieval_eval.py` (Phase 2) rằng metric chính là **proxy**, không phải Recall@k chuẩn.
- Vẫn log Recall@k để **theo dõi drift** khi KB thay đổi, nhưng **không dùng làm tiêu chí chọn model**.

## 7. Trả lời các "Câu hỏi mở" (ARCHITECTURE §10)

| # | Câu hỏi | Trả lời | Căn cứ |

|---|---|---|---|

| 1 | Gold Evidence có trong KB không? | **KHÔNG** — overlap ≥0.7 chỉ ~0.1% (dev). Đánh giá retrieval phải dùng proxy (entity/n-gram + manual annotation). | §6 |

| 2 | Ngữ nghĩa 0/1 + phân bố nhãn | **0=real, 1=fake, 2=unverifiable**, cân bằng ~33 % mỗi lớp | §3, §4 |

| 3 | Kích thước KB ⇒ Qdrant local hay server? | **Qdrant Docker local** (`docker-compose.yml`), không cần server | §2 |

| 4 | GPU có không ⇒ base hay large? | **Chỉ base** (`vinai/phobert-base-v2`), `fp16=true`, `gradient_accumulation_steps=4` | §1 + `configs/base.yaml` |

| 5 | UNVERIFIABLE là nhãn huấn luyện hay luật ngưỡng? | **Nhãn huấn luyện (3-class)** — ViFactCheck đã có sẵn nhãn 2 | §3 |

## 8. Hệ quả cho các Phase sau

**Phase 3:** classifier 3 lớp (`num_labels = len(label_map) = 3`), không có giá trị hard-code.
**Phase 4:** luật `UNVERIFIABLE` ở `decision.py` **vẫn giữ** như tầng bảo hiểm (retrieval score thấp) — không thay thế nhãn lớp 2.
**Phase 4 (test set):** chạy đúng **1 lần** với `--final` như quy định.
