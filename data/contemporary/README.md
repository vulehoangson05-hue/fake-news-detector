# Hướng Dẫn Soạn Thảo Bộ Benchmark Claim Thời Sự (Contemporary Claims Benchmark)

> **Dành cho PM / Người thẩm định dữ liệu (14–16/11)**
> Căn cứ theo: **Brief #4 §8.4**.

---

## 1. Cấu trúc tập tin `claims_template.csv`

Mỗi dòng đại diện cho 1 claim cần kiểm chứng thời sự với các cột bắt buộc:
1. `claim_id`: Định danh duy nhất (ví dụ: `claim_tune_001` .. `claim_tune_010`, `claim_test_001` .. `claim_test_030`).
2. `claim`: Nội dung phát biểu bằng tiếng Việt chuẩn (NFC), không viết tắt tùy tiện.
3. `label`: Một trong 3 nhãn hợp lệ:
   - `real`: Phát biểu diễn đạt lại một sự thật có trong bài báo gốc.
   - `fake`: Phát biểu sửa đổi một chi tiết (số liệu, tên người/đơn vị, thời gian, địa điểm, kết quả) để mâu thuẫn trực tiếp với bài báo gốc.
   - `unverifiable`: Phát biểu thêm vào một chi tiết hợp lý nhưng bài báo gốc hoàn toàn không nhắc tới (ngoài phạm vi kiểm chứng của bài).
4. `source_url`: Đường link bài báo gốc thuộc **16 domain báo chính thống đã duyệt** (xem danh sách bên dưới).
5. `source_date`: Ngày xuất bản của bài báo (`YYYY-MM-DD`), nên chọn bài trong vòng 1–2 tuần gần đây.
6. `note`: Ghi chú ngắn về chi tiết đã sửa hoặc thêm vào (để phục vụ thẩm định).
7. `split`: Phân chia nghiêm ngặt giữa **`tune`** (10 dòng đầu để chỉnh ngưỡng) và **`test`** (các dòng còn lại để chấm điểm cuối, tuyệt đối không chỉnh ngưỡng trên `test`).

---

## 2. Quy tắc soạn thảo từng nhãn (Kèm ví dụ)

Giả sử bài báo gốc trên *Báo Chính phủ* (`https://baochinhphu.vn/...`) ngày `2026-11-02` có đoạn:
> *"Tại cuộc họp thường kỳ, Bộ trưởng thông báo xuất khẩu gạo 10 tháng năm 2026 đạt 7,2 triệu tấn, kim ngạch đạt 4,5 tỷ USD."*

- **Soạn nhãn `real`:** Diễn đạt lại nội dung nhưng giữ nguyên toàn bộ sự kiện và số liệu.
  - *Claim:* `Theo Bộ Nông nghiệp và Phát triển nông thôn, trong 10 tháng đầu năm 2026, Việt Nam đã xuất khẩu được 7,2 triệu tấn gạo với giá trị 4,5 tỷ USD.`
- **Soạn nhãn `fake`:** Sửa 1 chi tiết then chốt (số liệu, tên, thời gian, kết quả).
  - *Claim:* `Xuất khẩu gạo 10 tháng đầu năm 2026 của Việt Nam giảm mạnh, chỉ đạt 4,2 triệu tấn với kim ngạch 2,5 tỷ USD.`
- **Soạn nhãn `unverifiable`:** Thêm thông tin giải thích hoặc chi tiết mới mà bài báo không hề nói đến.
  - *Claim:* `Xuất khẩu gạo 10 tháng năm 2026 đạt 7,2 triệu tấn chủ yếu nhờ vào đơn hàng đột biến từ thị trường Nam Mỹ.`

---

## 3. Danh sách 16 Domain Báo Chính Thống Được Phép (`include_domains`)

Mọi URL trong cột `source_url` bắt buộc phải thuộc một trong các domain sau:
- `vnexpress.net`
- `dantri.com.vn`
- `tuoitre.vn`
- `thanhnien.vn`
- `vietnamnet.vn`
- `vtv.vn`
- `vov.vn`
- `nhandan.vn`
- `baochinhphu.vn`
- `chinhphu.vn`
- `plo.vn`
- `laodong.vn`
- `tienphong.vn`
- `nld.com.vn`
- `vietnamplus.vn`
- `znews.vn`

---

## 4. Script Kiểm Tra Tính Hợp Lệ

Sau khi soạn xong file `claims_template.csv` (hoặc `claims_contemporary.csv`), chạy lệnh sau để kiểm tra tự động:
```bash
python scripts/12_validate_contemporary_claims.py --csv data/contemporary/claims_template.csv
```
Script sẽ kiểm tra:
1. Đúng định dạng header và không trùng lặp `claim_id` hoặc nội dung `claim`.
2. Đúng 3 nhãn quy định: `real`, `fake`, `unverifiable`.
3. URL hợp lệ và nằm trọn trong 16 tên miền whitelist.
4. Đủ tối thiểu 10 dòng `tune` và không có sự rò rỉ dữ liệu giữa `tune` và `test`.
