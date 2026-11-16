# BÁO CÁO TOÀN DIỆN HUẤN LUYỆN VERIFIER — PHASE 3 & CỔNG G1-G3 (Brief #4 & Brief #5)

- **Mô hình cốt lõi:** `vinai/phobert-base-v2` (tách từ qua `underthesea`, `max_length = 256`, `truncation='only_second'` preserving Statement)
- **Phần cứng thực thi:** NVIDIA GeForce GTX 1660 6 GB VRAM trên PC R&D
- **Cấu hình chống OOM (Brief #4 §8.2):** `fp32` (`fp16: false`), batch 8 x accum 4 (effective 32), dynamic padding, `lr = 2e-5`, 4 epochs
- **Tập dữ liệu:** ViFactCheck tiền xử lý (`train_clean.jsonl`: 5,058; `dev_clean.jsonl`: 723; **tuyệt đối không đụng split test**)

> [!IMPORTANT]
> **ĐÍNH CHÍNH QUAN TRỌNG THEO CỔNG G3 (Brief #5 §1):**
> 1. **"Dev Selected" là điều kiện bài báo đúng (oracle-article condition):** Các câu bằng chứng được chọn bằng BM25 từ *chính bài báo gốc* mà claim được tạo ra (BM25 chọn trúng câu gold ở 93.78% dòng dev). **Đây KHÔNG phải là "bằng chứng thực tế ngoài đời"** (ở điều kiện live web search, tỉ lệ trúng bài báo chỉ đạt ~30% theo Tavily spike).
> 2. **Baseline chỉ-claim mang artifact:** Baseline chỉ đọc `Statement` đạt Macro-F1 **0.5644** (vượt xa mức đoán ngẫu nhiên 3 lớp $\approx 0.3333$). Điều này chứng minh bản thân câu chữ, cú pháp của claim mang tín hiệu artifact từ quá trình gán nhãn/tạo dữ liệu của ViFactCheck. Cần luôn báo cáo **cả số baseline và mức tăng thực tế ($\Delta$)** nhờ bằng chứng.
> 3. **ECE là số đo in-sample:** Độ đo ECE "sau hiệu chỉnh" bằng Temperature Scaling được fit và đo trên **chính tập dev** (in-sample). ECE sau hiệu chỉnh vẫn ở mức $\sim 0.08$ (Gold) và $\sim 0.12$ (Selected).
> 4. **Phân tích lỗi là giả thuyết:** Toàn bộ giải thích nguyên nhân cho 10 ca lỗi dev và các ca trượt Tavily là **giả thuyết phân tích định tính**, chưa phải chân lý tuyệt đối.
> 5. **Chỉ số dev là lạc quan (optimistic):** Tập dev được dùng liên tục để chọn epoch tốt nhất, chọn biến thể V-C và fit nhiệt độ; kết quả khách quan duy nhất của hệ thống sẽ là lần chạy một lần trên split `test` (`--final`, E1).

---

## 1. Bảng Đối Đầu Toàn Diện Vòng Khám Phá (Seed 42)

*So sánh hiệu năng giữa Baseline (chỉ claim) và các biến thể nạp bằng chứng:*

| Biến thể | Dữ liệu Huấn luyện | Dev Gold Acc | Dev Gold F1 | $\Delta$ F1 so với Baseline | Dev Selected (Oracle-Article) Acc | Dev Selected F1 | $\Delta$ F1 so với Baseline | F1 Trung bình | Peak VRAM | Best Epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline** | Chỉ `Statement` | 0.5698 | **0.5644** | — | — | — | — | — | `2809.8 MB` | 4 |
| **V-A** | Gold `Evidence` | 0.8230 | **0.8233** | **+0.2589** | 0.6957 | 0.6842 | **+0.1198** | 0.7538 | `3350.0 MB` | 4 |
| **V-B** | Selected `Context` | 0.7441 | **0.7419** | **+0.1775** | 0.7524 | 0.7520 | **+0.1876** | 0.7470 | `3371.3 MB` | 4 |
| **V-C** | Trộn 50/50 Gold+Selected | 0.8064 | **0.8062** | **+0.2418** | 0.7497 | 0.7501 | **+0.1857** | **0.7782** | `3373.8 MB` | 4 |

> **Nhận xét & Quyết định chọn:**
> - Bản thân claim không có bằng chứng đã đạt F1 `0.5644` (artifact cú pháp).
> - Nạp bằng chứng giúp tăng mạnh hiệu năng: V-C tăng **+0.2418** trên Gold và **+0.1857** trên Selected (oracle-article).
> - Biến thể **`V-C`** được chọn để chạy 3 seed vì giữ được F1 cao trên Gold (`0.8062`) và F1 trên Selected (`0.7501`).

---

## 2. Kết Quả Huấn Luyện 3 Seed Của Cấu Hình V-C

Chạy trên 3 hạt ngẫu nhiên (`seeds: [42, 43, 44]`) để xác thực tính ổn định:

| Chỉ số đánh giá | Seed 42 | Seed 43 | Seed 44 | **Trung bình ± Độ lệch chuẩn** | Ghi chú |
|---|---|---|---|---|---|
| **Dev Gold Accuracy** | 0.8064 | 0.7953 | 0.8202 | **0.8073 ± 0.0125** | Chênh lệch 18 mẫu giữa các seed |
| **Dev Gold Macro-F1** | 0.8062 | 0.7960 | 0.8203 | **0.8075 ± 0.0122** | Mức tăng $\Delta = +0.2431$ so với Baseline |
| **Dev Gold ECE (in-sample)** | 0.0870 | 0.0873 | 0.0787 | **0.0843 ± 0.0049** | Fit trên chính dev split |
| F1 lớp `real` | 0.8121 | 0.8117 | 0.8159 | 0.8132 ± 0.0023 | |
| F1 lớp `fake` | 0.7856 | 0.7625 | 0.8033 | 0.7838 ± 0.0205 | |
| F1 lớp `unverifiable` | 0.8208 | 0.8137 | 0.8417 | 0.8254 ± 0.0146 | |
| **Dev Selected Macro-F1** | 0.7501 | 0.7484 | 0.7494 | **0.7493 ± 0.0009** | Điều kiện oracle-article |
| **Dev Selected Accuracy** | 0.7497 | 0.7483 | 0.7497 | **0.7492 ± 0.0008** | Khớp đường chéo ngẫu nhiên (542/723) |
| **Dev Selected ECE (in-sample)**| 0.1235 | 0.1218 | 0.1221 | **0.1225 ± 0.0009** | Vẫn ở mức $\sim 0.12$ |

---

## 3. Kết Quả Xác Minh Cổng G1 (Bất Thường Độ Lệch Chuẩn Seed)

Kiểm tra đối soát mã băm trọng số và tỉ lệ đồng thuận dự đoán giữa các seed:
- **Mã băm SHA-256 trọng số `model.safetensors`:**
  - Seed 42: `7445112ce1fd75e120f95d20372bf57e9403436fa3eb408b1e4679b2622108c6` (540,026,460 bytes)
  - Seed 43: `d7803837daeccadbe57c0d1b609c27accee863a71d54ad5f16dae40ffd41ed09` (540,026,460 bytes)
  - Seed 44: `d7b3f9aef84b0111de1f24fd3769a0b0070bcfac69dde922be266dd3aab0050a` (540,026,460 bytes)
  $\to$ **Khẳng định 100%:** 3 checkpoint hoàn toàn độc lập, không có lỗi nạp trùng file trọng số.
- **Tỉ lệ đồng thuận dự đoán theo cặp seed (Pairwise Agreement):**
  - Dev Gold: 42 vs 43 = **90.87%** (657/723); 42 vs 44 = **90.73%** (656/723); 43 vs 44 = **91.98%** (665/723).
  - Dev Selected: 42 vs 43 = **91.15%** (659/723); 42 vs 44 = **89.07%** (644/723); 43 vs 44 = **91.56%** (662/723).
- **Giải trình hiện tượng trùng Accuracy Selected 0.7497 giữa Seed 42 và Seed 44:**
  Hai seed có cùng tổng số mẫu đoán đúng là 542/723, nhưng ma trận nhầm lẫn và mẫu dự đoán khác nhau:
  - Seed 42 Selected CM: `[[198, 40, 18], [36, 176, 32], [14, 41, 168]]` (176 fake, 168 unv)
  - Seed 44 Selected CM: `[[198, 34, 24], [36, 174, 34], [15, 38, 170]]` (174 fake, 170 unv)
  - Giữa Seed 42 và Seed 44 trên tập Selected có **79 mẫu đưa ra phán quyết khác nhau** (bất đồng 10.93%).
  - **Kết luận:** Hoàn toàn KHÔNG CÓ lỗi vòng lặp đánh giá. Sự trùng lặp tổng số mẫu đúng là ngẫu nhiên thống kê.

---

## 4. Kết Quả Xác Minh Cổng G2 (Trùng Bài Báo / Context Leakage)

- **Số lượng Context duy nhất:** `train` = 1,035; `dev` = 496; `test` = 758.
- **Tỉ lệ trùng Context giữa Dev và Train:** **721 / 723 dòng dev (99.72%)** có Context nằm trong tập Train.
- **Tỉ lệ trùng Evidence cụ thể giữa Dev và Train:** Chỉ **10 / 723 dòng dev (1.38%)**.
- **Tỉ lệ trùng Context giữa Test và Train (đo thuần túy qua mã băm SHA-256):** **1,433 / 1,447 dòng test (99.03%)**.
- **Đánh giá trên tập Dev con không trùng bài (`dev_unseen_indices`):**
  Do tỉ lệ trùng lên tới 99.72%, chỉ có duy nhất **2 mẫu** trong dev có Context chưa từng xuất hiện ở train. Với $N=2$, việc đo Macro-F1 trên tập con này bị suy biến ($N=2$, Acc=1.0000, Macro-F1=0.3333).
- **Ý nghĩa cấu trúc:** ViFactCheck được tạo bằng cách gom ~1,000 bài báo rồi sinh nhiều claim cho mỗi bài và chia ngẫu nhiên claim vào train/dev/test, chứ không chia theo bài báo (Context-level split). Vì vậy mô hình đã thấy bài báo gốc trong train, biến toàn bộ bài toán ViFactCheck thành "oracle-article condition". Đây là lý do cốt lõi buộc chúng ta phải kiểm thử Task 4.1 (bằng chứng lệch ngoài bài) và E2 (claim thời sự 2026).

---

## 5. Phân Tích 10 Ca Lỗi Tiêu Biểu Của V-C (Seed 42) Trích Xuất Trực Tiếp Từ eval_report.json

> [!NOTE]
> **Quy định Cổng G3 & Brief #5 Bản vá 1 (§11.2):** Toàn bộ 10 ca lỗi có độ tin cậy cao nhất dưới đây được trích xuất trực tiếp, nguyên văn từ trường `top_10_errors` trong file `artifacts/models/phobert_v-c_s42_1794249027/eval_report.json`. Mọi phân tích nguyên nhân đưa ra đều là **giả thuyết nghiên cứu định tính** phục vụ mổ xẻ hành vi mô hình, không khẳng định là chân lý tuyệt đối.

1. **`sample_idx: 284`** (Nhãn thật: `unverifiable` | Dự đoán: `real` | Độ tin cậy: **0.8509**)
   - **Statement:** Chiều tối ngày 28-3 trong buổi giao lưu với khán giả của đoàn phim "Biệt đội rất ổn" cô chia sẽ rằng bản thân tuy cũng có không ít vai diễn thành công trước đây trên màn ảnh rộng nhưng tất cả đều là quá khứ, không thể bắt khán giả nhớ mãi những vai diễn đó rồi áp đặt vào thực tại.
   - **Evidence:** (NLĐO) – Nữ diễn viên Lê Khánh tái xuất màn ảnh rộng với vai diễn nữ đại gia miền Tây Tư Xoàn trong phim điện ảnh “Biệt đội rất ổn”. Trong buổi giao lưu với khán giả chiều tối ngày 28-3 của đoàn phim "Biệt đội rất ổn", Lê Khánh gây ấn tượng với vai diễn nữ đại gia Tư Xoàn. Cô nói thêm rằng bản thân tuy cũng có không ít vai diễn thành công trước đây trên màn ảnh rộng nhưng tất cả đều là quá khứ, không thể bắt khán giả nhớ mãi những vai diễn đó rồi áp đặt vào thực tại.
   - *Giả thuyết nguyên nhân:* Độ trùng lặp từ vựng gần như tuyệt đối giữa claim và evidence ("cô chia sẽ rằng bản thân tuy cũng có không ít vai diễn thành công..."). Dataset ViFactCheck gán `unverifiable` có thể do claim chỉ dùng đại từ "cô" thay vì danh từ riêng "Lê Khánh", nhưng mô hình cross-encoder nhìn thấy chuỗi văn bản trùng khớp hoàn toàn nên tự tin gán `real`.

2. **`sample_idx: 37`** (Nhãn thật: `fake` | Dự đoán: `real` | Độ tin cậy: **0.8503**)
   - **Statement:** Với các mái vòm cao tới 32m, Nhà thờ Đức Bà Paris được xây dựng vào giữa thế kỷ 12 và là tòa nhà cao nhất từng được xây dựng vào thời điểm đó.
   - **Evidence:** Nhà thờ Đức Bà Paris được xây dựng vào giữa thế kỷ 12 và là tòa nhà cao nhất từng được xây dựng vào thời điểm đó với các mái vòm cao tới 32m. Từ đó, nhóm phát hiện ra rằng các thợ xây sử dụng đinh kẹp sắt trong việc xây dựng từ những năm 1160 và nhà thờ Đức Bà Paris là nhà thờ đầu tiên sử dụng đinh kẹp sắt trong toàn bộ cấu trúc của nó. Các phân tích không chỉ giúp khám phá cách thức các thợ xây bậc thầy ở Paris xây dựng nhà thờ, mà còn cung cấp thông tin mới về thị trường sắt ở Paris giai đoạn thế kỷ 12 và 13.
   - *Giả thuyết nguyên nhân:* Toàn bộ mệnh đề của claim xuất hiện nguyên văn ở câu đầu của evidence ("Nhà thờ Đức Bà Paris được xây dựng vào giữa thế kỷ 12 và là tòa nhà cao nhất từng được xây dựng vào thời điểm đó với các mái vòm cao tới 32m"). Nhãn gán trong ViFactCheck là `fake` (có thể do lỗi nhãn gốc của dataset hoặc bối cảnh ngoài bài báo), song bằng chứng cục bộ xác thực hoàn toàn nội dung claim.

3. **`sample_idx: 700`** (Nhãn thật: `fake` | Dự đoán: `real` | Độ tin cậy: **0.8501**)
   - **Statement:** Ngày 29/3, đơn vị giữ quyền cử thí sinh Việt Nam dự thi Miss Universe đã đạt được thỏa thuận với Unicorp về việc chuyển quyền quản lý Á hậu Thảo Nhi nên cô được tham gia Miss Universe 2023.
   - **Evidence:** (Dân trí) - Miss Universe Vietnam và Unicorp không đạt được thỏa thuận về việc chuyển quyền quản lý Á hậu Thảo Nhi, đây là nguyên nhân khiến người đẹp không thể dự thi quốc tế. Hôm nay 29/3, đơn vị giữ quyền cử thí sinh Việt Nam dự thi Miss Universe chính thức xác nhận Á hậu Thảo Lê sẽ không tham gia Miss Universe 2023. Tuy nhiên, Tổ chức Miss Universe Vietnam và công ty quản lý của Á hậu Thảo Nhi Lê (Unicorp) lại không đạt được thỏa thuận về việc chuyển quyền quản lý của cô cho Miss Universe Vietnam.
   - *Giả thuyết nguyên nhân:* Claim có mật độ thực thể và từ khóa trùng khớp rất cao với evidence ("đơn vị giữ quyền cử thí sinh Việt Nam... chuyển quyền quản lý Á hậu Thảo Nhi... Miss Universe 2023"), nhưng đảo ngược từ phủ định ("đã đạt được thỏa thuận" vs "không đạt được thỏa thuận", "được tham gia" vs "không thể dự thi"). Cơ chế self-attention bị áp đảo bởi sự trùng lặp thực thể (bag-of-entities) mà bỏ sót từ phủ định "không".

4. **`sample_idx: 31`** (Nhãn thật: `fake` | Dự đoán: `real` | Độ tin cậy: **0.8471**)
   - **Statement:** Để đảm bảo tiến độ triển khai thành công, hiệu quả Chương trình phục hồi và phát triển kinh tế - xã hội cần khẩn trương, chủ động triển khai Nghị quyết số 44/2022/QH15 ngày 11/1/2022.
   - **Evidence:** * Để bảo đảm tiến độ triển khai thành công, hiệu quả Chương trình phục hồi và phát triển kinh tế - xã hội, tại Thông báo số 193/TB-VPCP, Phó Thủ tướng Lê Văn Thành cũng đã yêu cầu các bộ, ngành, địa phương khẩn trương, chủ động triển khai các cơ chế, giải pháp đã được quy định rõ trong các Nghị quyết của Quốc hội (số 43/2022/QH15 ngày 11/1/2022 và số 44/2022/QH15 ngày 11/1/2022) và các Nghị quyết của Chính phủ (số 11/NQ-CP ngày 30/1/2022 và số 18/NQ-CP ngày 11/2/2022) bảo đảm tiến độ, chất lượng các dự án thuộc Chương trình phục hồi và phát triển kinh tế - xã hội. Bộ Kế hoạch và Đầu tư có trách nhiệm kịp thời hướng dẫn các bộ, ngành, địa phương theo đề nghị của các Bộ, ngành, địa phương trong quá trình thực hiện. Vũ Phương Nhi
   - *Giả thuyết nguyên nhân:* Câu claim trích đúng chuỗi từ hành chính dài. Tuy nhiên, bằng chứng yêu cầu đồng thời 4 nghị quyết (43, 44, 11, 18), còn claim chỉ nêu Nghị quyết 44. Nhãn dataset coi việc thiếu sót điều kiện là `fake`, nhưng mô hình thấy Nghị quyết 44 có mặt nên phán quyết `real`.

5. **`sample_idx: 540`** (Nhãn thật: `fake` | Dự đoán: `real` | Độ tin cậy: **0.8468**)
   - **Statement:** Đoàn Trung Quốc đến Việt Nam lần này với mong muốn có thể nhận được sự hỗ trợ của Việt Nam cũng như thúc đẩy xuất khẩu nông sản của Việt Nam sang Trung Quốc.
   - **Evidence:** Ông Leo Liu, Giám đốc Kinh doanh của Tập đoàn Sunwah khu vực Vùng Vịnh Quảng Đông-Hồng Kông (Trung Quốc)-Ma Cao cho biết, đoàn đến Việt Nam lần này với mong muốn có thể hỗ trợ cũng như thúc đẩy xuất khẩu nông sản của Việt Nam sang Trung Quốc, với 4 nội dung chính: Làm thế nào để đơn giản hóa các thủ tục cũng như thúc đẩy nông sản Việt nam sang Trung Quốc dễ dàng hơn; tăng cường hợp tác về các lĩnh vực chè, cà phê (tổ chức các lớp bồi dưỡng đào tạo cho các doanh nghiệp về hai lĩnh vực này); tổ chức diễn đàn xúc tiến thương mại nông sản tại Côn Minh, Vân Nam nhân chuyến thăm của lãnh đạo Bộ NN&PTNT và các doanh nghiệp Việt Nam; nâng cao hình ảnh của hàng nông sản Việt Nam tại thị trường Trung
   - *Giả thuyết nguyên nhân:* Biến đổi quan hệ ngữ pháp chủ - khách: bằng chứng ghi "mong muốn có thể hỗ trợ [cho Việt Nam]", còn claim đổi thành "mong muốn có thể nhận được sự hỗ trợ của Việt Nam". Đây là bẫy ngữ nghĩa cực kỳ tinh vi khiến mô hình phân loại nhầm.

6. **`sample_idx: 2`** (Nhãn thật: `fake` | Dự đoán: `real` | Độ tin cậy: **0.8442**)
   - **Statement:** Công ty TNHH Mua bán nợ DSP, địa chỉ 91 Pasteur, phường Bến Nghé, quận 1, TP HCM mua các khoản nợ mà khách hàng đã vay.
   - **Evidence:** Công an TP HCM khám xét Công ty Mirae Asset Cảnh sát xác định Công ty TNHH Mua bán nợ DSP, địa chỉ: Tầng 4, Chung cư Lữ Gia, số 70 Lữ Gia, phường 15, quận 11, TP HCM, mua các khoản nợ mà khách hàng đã vay của Công ty Tài chính TNHH MTV Mirae Asset Việt Nam (Mã số doanh nghiệp: 0311132506; địa chỉ 91 Pasteur, phường Bến Nghé, quận 1, TP HCM) và một số tổ chức tín dụng khác nhưng không có khả năng trả. liên quan đến Công ty Tài chính TNHH MTV Mirae Asset Việt Nam, Công ty TNHH Mua bán nợ DSP đến Cơ quan Cảnh sát điều tra Công an TP Hà Nội để tố giác, cung cấp các thông tin, tài liệu liên quan đến vụ án.
   - *Giả thuyết nguyên nhân:* Lỗi gán sai thuộc tính thực thể (entity-attribute mismatch): địa chỉ "91 Pasteur..." trong bằng chứng thuộc về công ty Mirae Asset, còn công ty DSP nằm ở "70 Lữ Gia". Khi cả hai thực thể và địa chỉ cùng xuất hiện trong một đoạn văn phức tạp, mô hình bị nhầm lẫn liên kết giữa tên công ty và địa chỉ.

7. **`sample_idx: 88`** (Nhãn thật: `fake` | Dự đoán: `real` | Độ tin cậy: **0.8440**)
   - **Statement:** Cô Lê Hạnh 1 năm trước vẫn còn rất lo lắng khi mắc bệnh tiểu đường kèm mỡ máu xấu, nhưng cô đã tìm lại được cuộc sống yên bình nhờ có sản phẩm được tin dùng số 1 Việt Nam.
   - **Evidence:** 1 năm trước, cô Lê Thị Hạnh, 61 tuổi ở Đông Anh, Hà Nội vẫn còn rất lo lắng khi mắc tiểu đường kèm mỡ máu xấu. Nhưng với “Sản phẩm được tin dùng số 1 Việt Nam trong dòng thảo dược chiết xuất từ dây thìa canh sạch chuẩn quốc tế GACP - WHO cho người tiểu đường” cô đã tìm lại cuộc sống yên bình…. 11 năm kể từ ngày phát hiện mình mắc tiểu đường kèm mỡ máu, cô Hạnh vẫn đang chung sống hòa bình với bệnh và yên tâm thực hiện tốt chức trách của một người làm vợ, làm mẹ, làm bà, chăm nom thật tốt cho cuộc sống của những người thân yêu!
   - *Giả thuyết nguyên nhân:* Claim sao chép gần như nguyên văn nội dung câu chuyện trong bài báo quảng cáo. Dataset gán `fake` có thể do tên rút gọn ("Lê Hạnh" vs "Lê Thị Hạnh") hoặc bài PR thuốc mang yếu tố lừa dối, song bằng chứng cục bộ lại trùng khít câu chữ với claim.

8. **`sample_idx: 98`** (Nhãn thật: `fake` | Dự đoán: `real` | Độ tin cậy: **0.8422**)
   - **Statement:** Ủy ban Tài chính, Ngân sách Quốc hội tổ chức hội thảo lấy ý kiến về chính sách tài chính đất đai và giá đất trong dự thảo Luật Đất đai sửa đổi vào ngày hôm qua 23/3, tại hội thảo đã liệt kê các khoản thu tài chính từ đất đai.
   - **Evidence:** Hôm nay (23-3), Ủy ban Tài chính, Ngân sách Quốc hội tổ chức hội thảo lấy ý kiến về chính sách tài chính đất đai và giá đất trong dự thảo Luật Đất đai sửa đổi. Tại hội thảo, đại diện Cục Quản lý Công sản - Bộ Tài chính cho rằng về các khoản thu tài chính từ đất đai, Điều 147 dự thảo quy định liệt kê các khoản thu tài chính từ đất đai. Cụ thể, dự thảo chưa có quy định về khoản thu “thuế sử dụng đất” trong khi pháp luật về đất đai, thuế hiện hành và điểm c khoản 3, khoản 4 Điều 154 dự thảo có quy định khoản thu “thuế sử dụng đất” là một trong các khoản thu tài chính từ đất đai (gồm thuế sử dụng đất nông nghiệp, thuế sử dụng đất phi nông nghiệp).
   - *Giả thuyết nguyên nhân:* Lệch mốc thời gian: bằng chứng ghi "Hôm nay (23-3)", claim ghi "ngày hôm qua 23/3". Các cụm thực thể còn lại khớp tuyệt đối nên mô hình bỏ qua chi tiết trạng từ chỉ thời gian "hôm qua".

9. **`sample_idx: 311`** (Nhãn thật: `unverifiable` | Dự đoán: `real` | Độ tin cậy: **0.8422**)
   - **Statement:** Đề án sẽ nâng cao trình độ sản xuất lúa cũng như gia tăng giá trị lúa gạo.
   - **Evidence:** (Chinhphu.vn) - Phát triển bền vững 1 triệu ha lúa chuyên canh chất lượng cao gắn với tăng trưởng xanh vùng Đồng bằng sông Cửu Long sẽ nâng cao trình độ sản xuất lúa cũng như gia tăng giá trị lúa gạo. Thứ trưởng Bộ NN&PTNT Trần Thanh Nam khẳng định, Đề án sẽ chuyển đổi căn bản hệ thống sản xuất lúa ở vùng này, cùng với đó gắn với việc chuyển đổi tư duy sản xuất của nông dân về nông nghiệp nói chung và lúa gạo nói riêng. "Đề án không chỉ phát triển 1 triệu ha lúa chất lượng cao mà còn nâng cao vai trò của hợp tác xã để trở thành các doanh nghiệp hướng đến sản xuất quy mô lớn.
   - *Giả thuyết nguyên nhân:* Claim trích xuất mệnh đề kết quả ("sẽ nâng cao trình độ sản xuất lúa cũng như gia tăng giá trị lúa gạo") nhưng thay tên đề án đầy đủ bằng đại từ "Đề án". Dataset gán `unverifiable` do thiếu thông tin định danh, nhưng mô hình thấy mệnh đề này được xác nhận ngay sau đó bởi câu của Thứ trưởng nên dự đoán `real`.

10. **`sample_idx: 463`** (Nhãn thật: `fake` | Dự đoán: `real` | Độ tin cậy: **0.8392**)
    - **Statement:** Công an TP Hồ Chí Minh cho biết tối 28/3, đã làm việc với Lê Văn Bậm và Nguyễn Thảo Nguyên trong việc bé trai bị nghi cho hút ma túy, theo kết quả, cả hai người này đều dương tính với ma túy.
    - **Evidence:** Tối 27/3, Công an TP Hồ Chí Minh thông tin cho biết đã làm việc với Lê Văn Bậm và Nguyễn Thảo Nguyên (mẹ ruột của bé trai). Theo đó, cả hai người này đều dương tính với ma túy, riêng cháu T.N.A.T (bé trai trong clip nghi bị cho hút ma túy) âm tính với ma túy. Tiến hành xét nghiệm nhanh các chất ma túy, phát hiện Lê Văn Bậm và Nguyễn Thảo Nguyên dương tính với ma túy.
    - *Giả thuyết nguyên nhân:* Lệch duy nhất 1 ngày ("tối 28/3" so với "tối 27/3"). Toàn bộ tên đối tượng, cơ quan thụ lý, hành vi và kết quả xét nghiệm ma túy đều trùng khít, khiến mô hình không nhận diện được sai khác con số ngày tháng.

---

## 6. Trạng Thái Exit Criteria & Quyết Định

- Milestone M3: **✅ Hoàn thành (đóng theo xác nhận của Kỹ sư trưởng 2026-11-10; mục 11.3 chuyển sang Phase 4)**.
- Phase 4: **GO** theo phê duyệt của Kỹ sư trưởng và PM ngày 2026-11-10.

---

## 7. Đánh Giá Chia Dữ Liệu Tách Biệt Theo Bài Báo (Task 4.0 — Article-Disjoint Evaluation)

Theo yêu cầu Cổng G2 (mục 11.3 của Brief #5), toàn bộ 5,781 mẫu (train + dev) được gộp lại, nhóm theo mã băm SHA-256 của `Context` gốc (1,036 bài báo duy nhất) và chia ngẫu nhiên theo bài viết (`seed = 42`):
- **Tập Train:** 726 bài báo (**4,082** claims).
- **Tập Validation $V$:** 103 bài báo (**524** claims) — dùng để chọn epoch và fit nhiệt độ.
- **Tập Holdout $H$:** 207 bài báo (**1,175** claims) — toàn bộ các bài báo trong $H$ hoàn toàn không xuất hiện trong Train/Val.

### Kết quả thực nghiệm trên tập Holdout $H$ ($N = 1,175$, `seed = 42`)

| Mô hình | Cấu hình bằng chứng | Accuracy | Macro-F1 | ECE | F1 (`real` / `fake` / `unverifiable`) |
|---|---|---|---|---|---|
| **Baseline (Claim-only)** | Không có | 0.5413 | **`0.5346`** | 0.0168 | `0.5969` / `0.4859` / `0.5211` |
| **V-C (Disjoint)** | **Gold Evidence** | 0.7991 | **`0.7994`** | 0.0851 | `0.8105` / `0.7494` / `0.8384` |
| **V-C (Disjoint)** | **Selected (BM25)** | 0.7489 | **`0.7490`** | 0.0560 | `0.7846` / `0.6970` / `0.7656` |

### Diễn giải khoa học & Lưu ý giới hạn (Brief #5 §12.2)

1. **Giới hạn số lần chạy:** Thực nghiệm này mới chạy **1 seed (`seed = 42`)**, chưa lấy trung bình qua nhiều hạt ngẫu nhiên.
2. **Giới hạn phân phối của tập Holdout $H$:** Tập $H$ vẫn là các bài báo năm 2023 thuộc cùng phân phối dữ liệu ViFactCheck, **không phải** bài báo web thời sự năm 2026; do đó kết quả trên $H$ không thể suy diễn trực tiếp sang hiệu năng thực tế của bộ kiểm thử E2.
3. **Về mức chênh lệch của Baseline:** Mức giảm từ `0.5644` (chia theo claim, $n=723$) xuống `0.5346` (chia theo bài viết, $n=1,175$, chênh $\approx 0.03$) nằm trong biên độ sai số lấy mẫu của cỡ mẫu hiện có, chưa đủ cơ sở để kết luận cứng rằng mô hình trước đó đã ghi nhớ bài viết (học vẹt). Phần điểm $\sim 0.53$ tiếp tục phản ánh các đặc trưng văn phong/cú pháp nội tại của câu claim trong bộ dữ liệu.
4. **Hạn chế của phương pháp nhóm theo hash `Context`:** Việc gom nhóm dựa trên mã băm chuỗi văn bản `Context` nguyên văn vẫn tồn tại giới hạn kỹ thuật là cùng một sự kiện thời sự được tường thuật trên nhiều đầu báo khác nhau (khác câu chữ) vẫn có thể xuất hiện ở cả tập huấn luyện và tập Holdout $H$.
5. **Điểm đáng ghi nhận:** Trên nhóm bài viết chưa xuất hiện nguyên văn trong tập huấn luyện, biến thể V-C vẫn duy trì được mức hiệu năng tương đương với điều kiện chia theo claim trước đó (**`0.7994`** so với `0.8062` trên Gold; **`0.7490`** so với `0.7501` trên Selected; tăng +0.2144 Macro-F1 so với Baseline trên cùng tập $H$).
