# Báo cáo Đánh giá Retrieval & Reranking (Phase 2 — M2)

- **Thời gian tạo:** `2026-11-07T07:18:30.024371+00:00`
- **Collection Qdrant:** `kb_chunks_8b525e25ad54`
- **Số lượng mẫu đánh giá (dev):** `723`
- **Cấu hình:** `retrieve_k = 50`, `rerank_k = 5`

## 1. Hiệu năng Truy vấn (Latency)

| Bước xử lý | Chi tiết | Thời gian (ms) |
|---|---|---:|
| **Embedding (Batch)** | Trung bình / câu (`bge-m3` dense + sparse) | `180.30` |
| **Hybrid Search (Qdrant RRF)** | p50 (`top-50`) | `28.25` |
| **Hybrid Search (Qdrant RRF)** | **p95** (Exit criteria < 300 ms) | **`43.07`** |
| **Reranking (`bge-reranker-v2-m3`)** | p50 (`50 -> 5`) | `14652.13` |
| **Reranking (`bge-reranker-v2-m3`)** | p95 (`50 -> 5`) | `19532.86` |

## 2. Kết quả Proxy Metric (Chính — theo `docs/DATA_AUDIT.md §6`)

Do tập KB (Dân Trí / VnExpress) và ViFactCheck khác nguồn biên tập, chỉ số chính dùng **Proxy Relevance** (khớp `chunk_id`/`source` hoặc độ phủ word n-gram giữa Claim/Evidence và Chunk $\ge 0.30$).

| Giai đoạn | Hit Rate@1 | MRR@1 | nDCG@1 | Hit Rate@3 | MRR@3 | nDCG@3 | Hit Rate@5 | MRR@5 | nDCG@5 | Hit Rate@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Hybrid RRF (Pre-Rerank)** | 0.0360 | 0.0360 | 0.0360 | 0.0429 | 0.0387 | 0.0391 | 0.0512 | 0.0405 | 0.0417 | 0.1051 |
| **Hybrid + Reranker** | 0.0415 | 0.0415 | 0.0415 | 0.0636 | 0.0521 | 0.0547 | 0.0678 | 0.0531 | 0.0563 | — |

## 3. Phân tích theo Nhãn gốc (Post-Rerank Proxy Metric)

| Nhãn (`label_map`) | Số mẫu | Hit Rate@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| `real` | 256 | 0.0508 | 0.0400 | 0.0427 |
| `fake` | 244 | 0.0779 | 0.0618 | 0.0646 |
| `unverifiable` | 223 | 0.0762 | 0.0587 | 0.0628 |

## 4. Chỉ số Theo dõi Drift (Strict Gold 3-gram Containment $\ge 0.50$)

- **Hybrid RRF Hit Rate@50:** `0.0069`
- **Hybrid RRF Hit Rate@5:** `0.0014`
- **Hybrid + Reranker Hit Rate@5:** `0.0041`
- **Hybrid + Reranker MRR@5:** `0.0024`
- **Mean Top-1 Score (Hybrid / Rerank):** `0.6424` / `0.3100`

---

## 5. Phase 2.5 — Rerank profiling

### 5.1 Môi trường đo & Nguyên nhân gốc của độ trễ Baseline (14.65 s)

- **Thời điểm đo:** `2026-11-08T06:12:13+00:00`
- **Hệ điều hành:** `Windows-10-10.0.26300-SP0` (Môi trường PC R&D — PM xác nhận toàn bộ Phase 2 và 2.5 chạy trên PC này; laptop chỉ dùng từ Phase 4/demo)
- **CPU:** `Intel64 Family 6 Model 165 Stepping 3, GenuineIntel` (`12` luồng logic, `torch_num_threads = 6`)
- **GPU (`nvidia-smi`):** `NVIDIA GeForce GTX 1660` (`6144 MiB` / `6.00 GiB` / `6.44 GB` VRAM, NVIDIA Driver `610.88`, CUDA `12.6`, kiến trúc Turing TU116 — **không có Tensor Cores FP16**)
- **Phiên bản phần mềm:** Python `3.11.18`, `torch==2.14.1+cu126`, `transformers==5.22.1`, `FlagEmbedding==1.4.2`, `sentence-transformers==6.2.0`
- **Cấu hình model:** `BAAI/bge-reranker-v2-m3`, `max_length = 512`, `rerank_k = 5`
- **Tập con cố định để quét cấu hình:** `n = 100` claim dev (`seed = 42`, bỏ 3 claim warm-up đầu tiên → thống kê trên `97` claim). Danh sách `claim_id` lưu tại `artifacts/profiling/rerank_subset_n100_seed42.json`; dữ liệu thô lưu tại `artifacts/profiling/rerank_phase2_5.json`.
- **Phát hiện nguyên nhân nút thắt 14.65 s ở Phase 2:**
  1. `BgeReranker` mặc định bật `use_fp16=True` khi có CUDA. Trên GPU **GTX 1660 (TU116)** không có Tensor Cores chuyên dụng cho FP16, phép toán `fp16` bị rơi vào đường thực thi chậm (~`255 ms/cặp`), chậm hơn **`fp32` (~`45.2 ms/cặp`) tới 5.6 lần**.
  2. Trong `retrieval_eval.py` của Phase 2, mô hình `BgeM3Embedder` vẫn giữ nguyên trên VRAM (6.44 GB) trong suốt quá trình chạy `BgeReranker`, gây áp lực bộ nhớ GPU. Sau khi giải phóng `BgeM3Embedder` khỏi VRAM ngay sau bước nhúng batch và chuyển `BgeReranker` sang `fp32`, thời gian rerank giảm hơn **6 lần** ngay cả khi giữ nguyên `k = 50`, và giảm hơn **14 lần** khi kết hợp `k = 20, batch_size = 4, dtype = fp32`.

### 5.2 Bảng quét 30 cấu hình trên tập con cố định (`n = 100`, đo `97` claim sau `3` warm-up, `max_length = 512`)

- **Baseline so sánh overlap Top-5:** cấu hình mặc định `fp32, batch_size = 16, max_length = 512` của cùng `k`.
- Không có cấu hình nào bị OOM trên `batch_size ∈ {1, 4, 8, 16, 32}`.

| k | batch | dtype | p50 (ms) | p95 (ms) | tokenize (ms/claim) | forward (ms/claim) | postprocess (ms/claim) | ms/cặp | overlap top-5 vs baseline | Ghi chú / OOM |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | 1 | fp32 | 2503.33 | 3229.66 | 41.93 | 2440.29 | 8.80 | 49.82 | 1.0000 | OK |
| **50** | **4** | **fp32** | **2275.31** | **2825.34** | **20.74** | **2236.21** | **2.37** | **45.19** | **1.0000** | **Tốt nhất ở k=50** |
| 50 | 8 | fp32 | 2436.01 | 3226.88 | 15.34 | 2434.99 | 1.31 | 49.03 | 1.0000 | OK |
| 50 | 16 | fp32 | 2645.36 | 3399.51 | 12.86 | 2650.45 | 0.76 | 53.28 | 1.0000 | Baseline k=50 |
| 50 | 32 | fp32 | 2770.08 | 3666.68 | 11.34 | 2820.47 | 0.39 | 56.64 | 1.0000 | OK |
| 50 | 1 | fp16 | 12633.99 | 15614.51 | 41.07 | 12517.14 | 8.96 | 251.34 | 0.9979 | GTX 1660 thiếu FP16 Tensor Core |
| 50 | 4 | fp16 | 12868.80 | 15472.91 | 21.83 | 12671.23 | 2.48 | 253.91 | 0.9979 | Chậm 5.6× so với fp32 |
| 50 | 8 | fp16 | 12204.44 | 15668.23 | 15.49 | 12186.70 | 1.35 | 244.07 | 1.0000 | Chậm |
| 50 | 16 | fp16 | 12671.47 | 16245.35 | 12.91 | 12756.03 | 0.78 | 255.39 | 0.9979 | Cấu hình gây chậm ở Phase 2 |
| 50 | 32 | fp16 | 13139.78 | 16642.29 | 11.23 | 13297.75 | 0.41 | 266.19 | 0.9979 | Chậm |
| 50 | 1 | bf16 | 2986.98 | 3487.45 | 42.22 | 2929.32 | 8.44 | 59.60 | 0.9938 | OK |
| 50 | 4 | bf16 | 2693.13 | 3650.22 | 20.10 | 2712.41 | 2.38 | 54.70 | 0.9938 | OK |
| 50 | 8 | bf16 | 2820.61 | 3678.43 | 15.27 | 2821.42 | 1.29 | 56.76 | 0.9938 | OK |
| 50 | 16 | bf16 | 2980.65 | 3878.45 | 12.92 | 2997.93 | 0.75 | 60.23 | 0.9938 | OK |
| 50 | 32 | bf16 | 3105.54 | 4120.76 | 11.15 | 3172.74 | 0.39 | 63.69 | 0.9938 | OK |
| 20 | 1 | fp32 | 1001.55 | 1278.14 | 16.26 | 983.35 | 3.38 | 50.15 | 1.0000 | OK |
| **20** | **4** | **fp32** | **903.43** | **1118.82** | **7.79** | **897.48** | **0.91** | **45.31** | **1.0000** | **Tốt nhất ở k=20 (p95 = 1.12 s)** |
| 20 | 8 | fp32 | 958.69 | 1288.13 | 6.26 | 958.03 | 0.56 | 48.24 | 1.0000 | OK |
| 20 | 16 | fp32 | 1015.42 | 1439.81 | 5.49 | 1043.36 | 0.39 | 52.46 | 1.0000 | Baseline k=20 |
| 20 | 32 | fp32 | 1064.68 | 1529.26 | 4.48 | 1109.68 | 0.20 | 55.72 | 1.0000 | OK |
| 20 | 1 | fp16 | 5089.27 | 6172.79 | 16.24 | 5028.71 | 3.59 | 252.43 | 1.0000 | Chậm |
| 20 | 4 | fp16 | 5083.31 | 6239.30 | 7.95 | 5072.24 | 0.95 | 254.06 | 1.0000 | Chậm |
| 20 | 8 | fp16 | 4838.64 | 6449.06 | 6.59 | 4916.19 | 0.59 | 246.17 | 1.0000 | Chậm |
| 20 | 16 | fp16 | 4948.77 | 6811.20 | 5.47 | 5123.34 | 0.39 | 256.46 | 1.0000 | Chậm |
| 20 | 32 | fp16 | 5032.28 | 7091.92 | 4.51 | 5241.42 | 0.21 | 262.31 | 1.0000 | Chậm |
| 20 | 1 | bf16 | 1203.34 | 1388.46 | 16.01 | 1178.70 | 3.36 | 59.90 | 0.9897 | OK |
| 20 | 4 | bf16 | 1078.70 | 1454.26 | 7.83 | 1091.41 | 0.91 | 55.01 | 0.9897 | OK |
| 20 | 8 | bf16 | 1098.27 | 1471.88 | 6.23 | 1123.52 | 0.55 | 56.51 | 0.9897 | OK |
| 20 | 16 | bf16 | 1143.06 | 1589.83 | 5.42 | 1188.54 | 0.38 | 59.72 | 0.9897 | OK |
| 20 | 32 | bf16 | 1176.09 | 1753.65 | 4.50 | 1241.38 | 0.20 | 62.30 | 0.9897 | OK |

### 5.3 Kết quả Full Dev (`n = 723`) ở cấu hình `k = 20` (`dtype = fp32, batch_size = 4, max_length = 512`) và Giải trình A1 (Lệch số p50)

Dữ liệu chi tiết lưu tại `artifacts/profiling/eval_full_dev_k20_fp32_b4.json` (cho full dev `n = 723`) và `artifacts/profiling/rerank_phase2_5.json` (cho tập con `n = 100`).

#### Giải trình A1: Nguồn gốc hai cặp số `p50 / p95` của cấu hình `k = 20, fp32, batch_size = 4`

- **Phương pháp tính percentile (chung cho mọi bảng trong repo):** nội suy tuyến tính chuẩn (`linear interpolation` — hàm `_percentile()` trong `src/vnfake/evaluation/rerank_profiler.py` và `src/vnfake/evaluation/retrieval_eval.py`):
  $$\text{pos} = \frac{p}{100} \times (N - 1), \quad \text{lower} = \lfloor \text{pos} \rfloor, \quad \text{upper} = \lceil \text{pos} \rceil, \quad w = \text{pos} - \text{lower}$$
  $$\text{percentile}(p) = x_{[\text{lower}]} \cdot (1 - w) + x_{[\text{upper}]} \cdot w$$
- **Bảng đối chiếu nguyên văn hai tập đo:**

| Tập đo | File artifact nguồn | Số mẫu đưa vào hàm `_percentile` | `p50` nguyên văn (ms) | `p95` nguyên văn (ms) | `mean` (ms) | Ghi chú |
|---|---|---:|---:|---:|---:|---|
| **Tập con cố định `n = 100`** (`seed = 42`) | `artifacts/profiling/rerank_phase2_5.json` (dòng 146–147) | `N = 97` (bỏ `3` claim warm-up đầu tiên) | **`903.4317`** (`903.43`) | **`1118.8166`** (`1118.82`) | `906.1829` (`906.18`) | Đo trong quét 30 cấu hình (§5.2) |
| **Full Dev `n = 723`** (lần chạy `retrieval_eval.py`) | `artifacts/profiling/eval_full_dev_k20_fp32_b4.json` (dòng 10–11) | `N = 723` (không bỏ warm-up) | **`1005.7830`** (`1005.78`) | **`1369.1930`** (`1369.19`) | — | Đo trên toàn bộ 723 claim dev (§5.3) |
| **Full Dev `n = 723`** (lần chạy `k_sweep.py` có warm-up trước vòng lặp) | `artifacts/profiling/k_sweep_fp32_b4.json` (dòng 38–40) | `N = 723` (đã warm-up trước) | **`1005.6455`** (`1005.65`) | **`1354.6215`** (`1354.62`) | `1022.1184` (`1022.12`) | Đo trong k-sweep sạch `k ∈ {20, 30, 50}` (§5.5) |

> **Lý do bản tóm tắt chat Task 1 trước đó ghi `p50 = 903.43 ms / p95 = 1369.19 ms`:** Khi tóm tắt nhanh ra tin nhắn chat ở cuối Task 1, Coder đã sơ suất ghép nhầm `p50 = 903.43 ms` (lấy từ bảng tập con `n = 100` ở §5.2) với `p95 = 1369.19 ms` (lấy từ lần chạy full dev `n = 723` ở §5.3). Trong file JSON gốc và trong bảng §5.2 / §5.3 của `REPORT_retrieval.md`, hai bộ số vốn hoàn toàn tách biệt:
> - Trên **tập con `n = 100` (`N = 97`)**: `p50 = 903.43 ms`, `p95 = 1118.82 ms`.
> - Trên **full dev `n = 723`**: `p50 = 1005.78 ms`, `p95 = 1369.19 ms` (và ở lần chạy k-sweep §5.5 là `p50 = 1005.65 ms`, `p95 = 1354.62 ms`).

#### Bảng đối chiếu đầy đủ Hit Rate@1, @3, @5, @20 (Gate 0 — Xác minh con số `0.0636`)

> **Giải trình con số `0.0636`:**
> - Con số **`0.0636`** (`46 / 723 = 0.06362379`) ở cấu hình `k = 20` chính là **`Hit Rate@5` sau rerank** (`46/723` claim có ít nhất 1 chunk khớp proxy trong top-5 sau rerank).
> - Sở dĩ `0.0636` trùng khớp với `Hit Rate@3` của baseline `k = 50` là do trùng tử số nguyên trên mẫu số `723`:
>   - Ở **Baseline `k = 50` (sau rerank)**: `Hit@1 = 30/723 (0.0415)`, **`Hit@3 = 46/723 (0.0636)`**, `Hit@5 = 49/723 (0.0678)`, `Hit@50 (trần) = 76/723 (0.1051)`.
>   - Ở **Cấu hình chốt `k = 20` (sau rerank)**: `Hit@1 = 28/723 (0.0387)`, `Hit@3 = 42/723 (0.0581)`, **`Hit@5 = 46/723 (0.0636)`**, `Hit@20 (trần) = 62/723 (0.0858)`.
>   - Khi giảm `retrieve_k` từ `50 -> 20`, số claim trúng trong top-5 sau rerank giảm đúng `3` claim (`49/723 -> 46/723`), khiến `Hit Rate@5 (k=20)` bằng đúng `Hit Rate@3 (k=50)` (`46/723 = 0.0636`).

| Giai đoạn & Cấu hình (`n = 723`) | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | Hit Rate@20 | Hit Rate@50 | MRR@5 | nDCG@5 | Rerank p50 (ms) | Rerank p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Pre-Rerank Hybrid RRF (`k = 50`)** | `0.0360` (26/723) | `0.0429` (31/723) | `0.0512` (37/723) | — | `0.1051` (76/723) | `0.0405` | `0.0417` | — | — |
| **Post-Rerank (`k = 50`, `fp16, batch=16`)** | `0.0415` (30/723) | `0.0636` (46/723) | `0.0678` (49/723) | — | `0.1051` (trần) | `0.0531` | `0.0563` | `14652.13` | `19532.86` |
| **Pre-Rerank Hybrid RRF (`k = 20`)** | `0.0346` (25/723) | `0.0443` (32/723) | `0.0526` (38/723) | `0.0858` (62/723) | — | `0.0402` | `0.0418` | — | — |
| **Post-Rerank (`k = 20`, `fp32, batch=4`)** | **`0.0387`** (28/723) | **`0.0581`** (42/723) | **`0.0636`** (46/723) | **`0.0858`** (62/723) | — | **`0.0492`** | **`0.0524`** | **`1005.78`** | **`1369.19`** |

- **Overlap Top-5 của cấu hình chốt (`fp32, batch=4`) so với baseline cùng `k` (`fp32, batch=16`):** **`1.0000` (100%)** ở cả `k = 20` và `k = 50`.

### 5.4 Hồ sơ VRAM đỉnh (Peak VRAM) khi nạp đồng thời `bge-m3` và `bge-reranker-v2-m3` trên PC R&D (`GTX 1660, 6144 MiB`)

Đo trực tiếp bằng `torch.cuda.max_memory_allocated()`, `torch.cuda.max_memory_reserved()` và `nvidia-smi` khi giữ **đồng thời** cả `BgeM3Embedder` và `BgeReranker` trong VRAM và chạy suy luận `k = 20, batch_size = 4`:

| Chế độ nạp đồng thời trên GPU (`GTX 1660, 6144 MiB`) | Sau khi nạp `bge-m3` (MiB) | Trọng số cả 2 model trên VRAM (MiB) | Peak Allocated `max_memory_allocated` | Peak Reserved `max_memory_reserved` | `nvidia-smi` Total Used (gồm cả OS/Display 836 MiB) | VRAM còn trống (`nvidia-smi`) | Trạng thái |
|---|---:|---:|---:|---:|---:|---:|---|
| **Cấu hình `configs/base.yaml` (`bge-m3` fp16 + `reranker` fp32, `batch=4`)** | `1093.16 MiB` | `3259.10 MiB` | **`3292.53 MiB` (`3.215 GiB`)** | `3320.00 MiB` (`3.242 GiB`) | `4188 / 6144 MiB` | `1779 MiB` | **An toàn (< 4 GB & < 6 GB)** |
| **Cả hai model ở `fp32` (`bge-m3` fp32 + `reranker` fp32, `batch=4`)** | `2178.07 MiB` | `4344.01 MiB` | **`4378.07 MiB` (`4.275 GiB`)** | `4404.00 MiB` (`4.301 GiB`) | `5271 / 6144 MiB` | `696 MiB` | **Đạt trên PC 6 GB (4.28 GiB < 6 GB, cần lưu ý cho laptop 4 GB ở Phase 4)** |

### 5.5 Task 5a — k-sweep sạch trên Full Dev (`n = 723`, `dtype = fp32, batch_size = 4, max_length = 512`, không gọi Gemini)

- **Thời điểm đo:** `2026-11-08T14:27:36+00:00` (`scripts/08_sweep_k_clean.py`)
- **Artifact lưu trữ:** `artifacts/profiling/k_sweep_fp32_b4.json`
- **Collection Qdrant:** `kb_chunks_8b525e25ad54` (kiểm chứng `41,466` điểm — không đổi)
- **Mục đích:** Tách bạch hoàn toàn tác động của `retrieve_k ∈ {20, 30, 50}` khỏi yếu tố `dtype` (`fp16` vs `fp32`), giữ cố định `dtype = fp32, batch_size = 4, max_length = 512, seed = 42`.

#### Bảng Hit Rate & Độ trễ Rerank theo `k` (`n = 723`, `fp32, batch_size = 4`)

| `retrieve_k` | Post-Rerank Hit Rate@1 (số claim) | Post-Rerank Hit Rate@3 (số claim) | Post-Rerank Hit Rate@5 (số claim) | Trần Pre-Rerank Hit@k (số claim) | Post-Rerank MRR@5 | Post-Rerank nDCG@5 | Hybrid Search p50 / p95 (ms) | Rerank p50 (ms) | Rerank p95 (ms) | Rerank mean (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`20`** | `0.0387` (**28 / 723**) | `0.0581` (**42 / 723**) | `0.0636` (**46 / 723**) | `0.0858` (**62 / 723** @20) | `0.0492` | `0.0524` | `25.05` / `35.74` | **`1005.65`** | **`1354.62`** | `1022.12` |
| **`30`** | **`0.0429`** (**31 / 723**) | `0.0622` (**45 / 723**) | `0.0664` (**48 / 723**) | `0.0996` (**72 / 723** @30) | **`0.0534`** | `0.0562` | `23.87` / `33.54` | **`1400.45`** | **`1878.53`** | `1424.82` |
| **`50`** | `0.0415` (**30 / 723**) | **`0.0636`** (**46 / 723**) | **`0.0678`** (**49 / 723**) | **`0.1051`** (**76 / 723** @50) | `0.0531` | **`0.0563`** | `25.32` / `34.41` | **`2153.55`** | **`2865.82`** | `2196.72` |

#### Bảng độ trùng lặp (Overlap Top-5 và Đồng thuận Top-1) giữa các mốc `k` (`n = 723`)

| Cặp so sánh | Top-5 Set Overlap trung bình (`\|Top5(k_a) ∩ Top5(k_b)\| / 5`) | Tỉ lệ trùng khít vị trí #1 (`Top-1 Exact Agreement`) |
|---|---:|---:|
| **`k = 20` vs `k = 30`** | **`0.8041`** (`80.41%`) | `0.8575` (`85.75%` — `620 / 723` claim) |
| **`k = 30` vs `k = 50`** | **`0.8017`** (`80.17%`) | `0.8534` (`85.34%` — `617 / 723` claim) |
| **`k = 20` vs `k = 50`** | **`0.6559`** (`65.59%`) | `0.7372` (`73.72%` — `533 / 723` claim) |

#### Nhận xét kỹ thuật cho PM chốt `retrieve_k`

1. **Loại bỏ nhiễu `fp16` vs `fp32`:** Ở `k = 50`, kết quả `fp32` (`Hit@1 = 30/723`, `Hit@3 = 46/723`, `Hit@5 = 49/723`) **trùng khít 100%** với lần chạy `fp16` ở Phase 2 (`30/723`, `46/723`, `49/723`). Như vậy chênh lệch `3` claim trúng ở `Hit@5` giữa `k = 50` (`49/723`) và `k = 20` (`46/723`) hoàn toàn đến từ việc thu hẹp tập ứng viên (`50 -> 20`), không phải do `dtype`.
2. **Độ trễ ở cả 3 mốc `k ∈ {20, 30, 50}` đều nằm sâu dưới ngưỡng mục tiêu `p95 ≤ 10,000 ms`:**
   - `k = 20`: `p50 = 1.01 s`, `p95 = 1.35 s`, `Hit@5 = 46/723` (`6.36%`), trần `Hit@20 = 62/723` (`8.58%`).
   - `k = 30`: `p50 = 1.40 s`, `p95 = 1.88 s`, `Hit@1 = 31/723` (`4.29%`, cao nhất), `Hit@5 = 48/723` (`6.64%`), trần `Hit@30 = 72/723` (`9.96%` — giữ lại thêm `10` claim có bằng chứng trong tập ứng viên so với `k = 20` và thu hồi `2/3` số claim của `k = 50` chỉ với `+0.52 s` ở p95).
   - `k = 50`: `p50 = 2.15 s`, `p95 = 2.87 s`, `Hit@5 = 49/723` (`6.78%`), trần `Hit@50 = 76/723` (`10.51%`).
3. Theo đúng mục B.3 của Brief #3, `configs/base.yaml` vẫn giữ nguyên `retrieve_k: 20` chờ PM quyết định.


