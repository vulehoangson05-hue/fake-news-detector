# Báo cáo Thử nghiệm A/B Query Rewrite (C1: 60 Claims Dev) — Phase 2.5

- **Thời gian thực hiện:** `2026-11-09T01:27:07.624767+00:00`
- **Model LLM:** `gemini-3.1-flash-lite`
- **Phiên bản Prompt:** `v3` (`sha256`: `a4b1a732d63f3fc6cd742ba2ca1406620fec22c7420ccb06ac7e25a248863baf`)
- **Số lượng claim:** `60` (20 Real, 20 Fake, 20 Unverifiable; seed=42)
- **Tham số Retrieval:** `retrieve_k = 30`, `rerank_k = 5`

## 1. Bảng Tổng Hợp Proxy Metric (Hit Rate trên 60 claims dev)

| Nhánh Thử Nghiệm | Hit@1 (Proxy) | Hit@3 (Proxy) | Hit@5 (Proxy) | Số claim trúng @5 |
|---|---:|---:|---:|---:|
| **1. `original` (Baseline)** | `3.33%` | `5.00%` | `5.00%` | `3/60` |
| **2. `rewritten_only`** | `3.33%` | `5.00%` | `5.00%` | `3/60` |
| **3. `original+rewritten` (RRF)** | `3.33%` | `6.67%` | `6.67%` | `4/60` |

## 2. Kết Quả Gán Nhãn Thủ Công (Human Labeling Ground Truth — 451 chunks)

> [!IMPORTANT]
> PM/Người dùng đã hoàn thành việc gán nhãn thủ công toàn bộ **451 candidate chunks** trong file mù `artifacts/ab_testing/human_label_pack_60_labeled.csv`.
> Khóa giải mã `unblind_key_do_not_open_before_labeling.json` đã được mở để đối chiếu hiệu năng thật giữa các nhánh.

### 2.1. Phân bố nhãn thực tế
- **Tổng số cặp (claim, chunk) được đánh giá:** `451`
- **Số chunk bị bác bỏ (Nhãn = 0):** `447 / 451` (**99.11%**)
- **Số chunk được duyệt làm bằng chứng (Nhãn = 1):** `4 / 451` (**0.89%**)
- **Số claim có ít nhất 1 bằng chứng hợp lệ:** `2 / 60` claim (**3.33%**)

### 2.2. So sánh hiệu năng giữa các nhánh trên Ground Truth con người
| Nhánh Thử Nghiệm | Số chunk trúng nhãn 1 | Số claim trúng (trên 60) | Ghi chú phát hiện |
|---|:---:|:---:|---|
| **1. `original` (Baseline)** | `3` | `2 / 60` | Khớp `c1_dev_250` (2 chunks) và `c1_dev_294` (1 chunk) |
| **2. `original+rewritten` (RRF)** | `3` | `2 / 60` | Khớp `c1_dev_250` (2 chunks) và `c1_dev_294` (1 chunk) |
| **3. `rewritten_only`** | **`4`** | `2 / 60` | **Tìm thêm được `item_0171`** (bài cảnh báo triệu chứng ung thư sụt cân nhanh) mà `original` bỏ sót |

### 2.3. Phân tích nguyên nhân gốc rễ (Root Cause: Temporal Shift)
1. **Lệch pha thời gian (Temporal Misalignment):**
   - ViFactCheck được tổng hợp từ các sự kiện lịch sử giai đoạn **2020 – giữa 2023** (ví dụ: đạn urani nghèo Ukraine tháng 3/2023, sập ngân hàng SVB tháng 3/2023, vụ ngộ độc cá chép botulinum đầu 2023, kỷ niệm 47 năm TNXP 2023, quy định COVID-19).
   - Knowledge Base (41.466 chunks) được thu thập từ báo Dân Trí và VnExpress vào năm **2026**.
   - Do đó, dù công cụ truy vấn (Dense BGE-M3 + Sparse BM25 + Reranker) tìm đúng chủ thể thực thể và trường ngữ nghĩa ("Putin", "Ukraine", "ngân hàng", "ung thư", "doanh nghiệp"), sự kiện thực tế năm 2026 đã hoàn toàn khác năm 2023.
2. **Các claim duy nhất trúng bằng chứng:**
   - `c1_dev_250`: Triệu chứng ung thư phổi (kiến thức y khoa trường tồn, không bị trôi dạt theo thời gian).
   - `c1_dev_294`: Chính sách hỗ trợ doanh nghiệp nhỏ (chủ đề pháp lý có tính tiếp nối lâu dài).
3. **Quyết định chiến lược của PM (2026-11-09):**
   - Chốt **Hướng 1 (Contemporary Fact-Checker ~2026+)**: Hệ thống tập trung kiểm chứng tin giả theo thời gian thực đối chiếu với tri thức hiện tại, không hồi tố quá khứ.
   - Tập ViFactCheck được phân định lại: **chỉ dùng để huấn luyện tầng suy luận ngôn ngữ (NLI Classifier ở Phase 3)** vì logic ngữ nghĩa (Ủng hộ/Bác bỏ) không phụ thuộc mốc thời gian; không dùng để ép buộc đánh giá độ phủ của Knowledge Base 2026.
   - Xây dựng **Contemporary Claims Benchmark (2026)** để đánh giá Retrieval và End-to-End đúng thực tế.

## 3. Độ Khác Biệt Giữa Các Nhánh (Branch Divergence Top-5)

- **`original` vs `rewritten_only`:** khác nhau ở `52/60` claim (`86.7%`).
- **`original` vs `original+rewritten`:** khác nhau ở `43/60` claim (`71.7%`).
- **`rewritten_only` vs `original+rewritten`:** khác nhau ở `44/60` claim (`73.3%`).

## 4. Thống Kê Rewrite & Quota Gemini

- **Phân bố trạng thái Rewrite:** `{'cache_hit': 57, 'fallback:quota_exhausted': 3}`
- **Tỉ lệ Rewrite Thành công (`ok` / `cache_hit`):** `95.0%`
- **Độ trễ Rewrite p50:** `0.77 ms`
- **Độ trễ Rewrite p95:** `0.96 ms`

## 5. Gói Gán Nhãn Mù (Human Labeling Pack)

- File dữ liệu gán nhãn JSON: `artifacts/ab_testing/human_label_pack_60.json`
- File bảng tính gán nhãn CSV: `artifacts/ab_testing/human_label_pack_60.csv`
- File hướng dẫn & bảng Markdown: `artifacts/ab_testing/human_label_pack_60.md`
- Khóa giải mã nhánh (anonymization key): `artifacts/ab_testing/unblind_key_do_not_open_before_labeling.json`

## 6. Chi tiết 60 Claims Dev và Trạng thái Rewrite

| ID | Nhãn Gốc | Claim Gốc | Rewritten Query (v3) | Trạng thái | Rewrite ≠ Claim |
|---|---|---|---|---|:---:|
| `c1_dev_001` | `FAKE` | Nhiều chi bộ chỉ mua báo đảng mà không quan tâm đến việc đọc, nghiên cứu và áp dụng các nội dung của báo vào thực tiển nhưng việc công tác mua váo và đọc báo Đảng vẫn đạt hiệu quả cao. | `chi bộ mua báo Đảng nghiên cứu nội dung thực tiễn hiệu quả công tác` | `cache_hit` | Có |
| `c1_dev_024` | `REAL` | Ngày 21/3 Tổng thống Nga Vladimir Putin lên án và tuyên bố Mátxcơva sẽ đáp trả tương xứng về hành động gửi đạn xe tăng chứa urani nghèo cho Ukraine của Anh. | `Tổng thống Nga Vladimir Putin đạn xe tăng urani nghèo Ukraine Anh đáp trả` | `cache_hit` | Có |
| `c1_dev_025` | `REAL` | Một trong những thỏa thuận giữa hai bên là chuyển đổi cơ chế hoạt động của Trường Đại học Hà Tĩnh thành trường đại học thành viên của ĐH Quốc gia Hà Nội. | `chuyển đổi cơ chế hoạt động Trường Đại học Hà Tĩnh đại học thành viên` | `cache_hit` | Có |
| `c1_dev_032` | `REAL` | Sở GD-ĐT Hà Nội đã ra văn bản yêu cầu các nhà trường thực hiện ngay một số biện pháp phòng ngừa sau khi phụ huynh một số trường học ở Hà Nội nhận được điện thoại với chiêu lừa "con đang cấp cứu, chuyển tiền gấp". | `Sở Giáo dục và Đào tạo Hà Nội văn bản phòng ngừa lừa đảo phụ` | `cache_hit` | Có |
| `c1_dev_033` | `REAL` | Thời gian nhận hồ sơ đăng ký dự thi các môn năng khiếu đợt 1 để xét tuyển các ngành: kiến trúc, thiết kế đồ hoạ, thiết kế thời trang, thiết kế nội thất, quy hoạch vùng và đô thị, tổ hợp có môn năng khiếu của ngành quản lý thể dục thể thao đã được công bố. | `hồ sơ đăng ký dự thi môn năng khiếu xét tuyển đại học ngành kiến` | `cache_hit` | Có |
| `c1_dev_063` | `FAKE` | Trong đợt công bố kết quả trên bảng xếp hạng ĐH theo lĩnh vực 2023, ngành kỹ thuật dầu khí của ĐH Quốc gia TP.HCM là ngành duy nhất không đạt top 100 thế giới. | `bảng xếp hạng đại học lĩnh vực kỹ thuật dầu khí đại học quốc gia` | `cache_hit` | Có |
| `c1_dev_064` | `FAKE` | Điện ảnh Việt Nam mới chỉ sản xuất phim đề tài chiến tranh, cách mạng. | `điện ảnh Việt Nam sản xuất phim đề tài chiến tranh cách mạng` | `cache_hit` | Có |
| `c1_dev_066` | `FAKE` | Ngân hàng lớn nhất tại Mỹ là SVB hoạt động từ thập niên 1970  chi nhánh tại  Santa Monica, bang California | `ngân hàng SVB hoạt động chi nhánh Santa Monica California` | `cache_hit` | Có |
| `c1_dev_077` | `UNVERIFIABLE` | Một phụ huynh chia sẻ trên mạng xã hội rằng con gái mình bị hai đối tượng dụ dỗ và có ý đồ bắt cóc nhưng cháu hét lên và có bảo vệ can thiệp nên các đối tượng đã bỏ chạy. | `phụ huynh mạng xã hội đối tượng dụ dỗ bắt cóc trẻ em bảo vệ` | `cache_hit` | Có |
| `c1_dev_097` | `UNVERIFIABLE` | Du lịch Việt Nam, du lịch Bình Thuận và Năm Du lịch quốc gia 2023 được quảng bá tại chương trình giới thiệu du lịch Việt Nam và các tuần văn hóa Việt Nam tại thị trường trọng điểm quốc tế (ASEAN, Đông Bắc Á, châu Âu, Ấn Độ, Australia, Mỹ...) trên sóng truyền hình VTV.  | `quảng bá du lịch Việt Nam Bình Thuận Năm Du lịch quốc gia tuần văn` | `cache_hit` | Có |
| `c1_dev_111` | `FAKE` | Đề án đặt mục tiêu đến năm 2020 hoàn thành xử lý toàn bộ hồ sơ tồn đọng trên địa bàn tỉnh Thanh Hóa. | `đề án xử lý hồ sơ tồn đọng tỉnh Thanh Hóa` | `cache_hit` | Có |
| `c1_dev_113` | `REAL` | Bộ Tài chính đại diện Cục Quản lý Công sản cho rằng cần rà soát để quy định đầy đủ các khoản thu này và chỉnh sửa tên của mỗi khoản thu cho phù hợp. | `Bộ Tài chính Cục Quản lý Công sản rà soát quy định khoản thu chỉnh` | `cache_hit` | Có |
| `c1_dev_115` | `FAKE` | TP sẽ đầu tư nhiều sản phẩm du lịch mới ngoài tổ chức nhiều sự kiện lễ hội lớn, trong đó sẽ dành khoảng 800 tỉ đồng để thực hiện "Dòng sông ánh sáng" trên sông Hàn trong năm 2023. | `đầu tư sản phẩm du lịch sự kiện lễ hội dòng sông ánh sáng sông` | `cache_hit` | Có |
| `c1_dev_119` | `REAL` | Dù đã có thuốc đặc trị nhưng việc điều trị cho các bệnh  nhân đã ăn món cá chép muối chua chứa chất kịch độc Botunium vẫn gặp nhiều khó khăn. | `bệnh nhân ngộ độc thực phẩm cá chép muối chua độc tố botulinum điều trị` | `cache_hit` | Có |
| `c1_dev_122` | `UNVERIFIABLE` | TS. Phạm Quang Tú sinh năm 1880, Phó Giám đốc Tổ chức Oxfam Việt Nam nhận định có thể sẽ phải mất 10 năm hỗ trợ liên tục thì bà con mới ổn định được cuộc sống ở khu tái định cư thủy điện. | `Phạm Quang Tú Oxfam Việt Nam hỗ trợ tái định cư thủy điện ổn định` | `cache_hit` | Có |
| `c1_dev_138` | `REAL` | Với chủ đề "Đồng đội một thời và mãi mãi"  Lực lượng Thanh niên xung phong (TNXP) TP.HCM đã họp mặt kỷ niệm 47 năm Ngày thành lập Lực lượng TNXP (28.3.1976 - 28.3.2023) vào tối 28.3. | `họp mặt kỷ niệm thành lập lực lượng thanh niên xung phong thành phố hồ` | `cache_hit` | Có |
| `c1_dev_150` | `REAL` | Thi thể thiếu nữ được lực lượng cứu hộ tìm thấy sau hơn 24 giờ tìm kiếm dưới sông Hiếu. | `thi thể thiếu nữ lực lượng cứu hộ tìm kiếm sông Hiếu` | `cache_hit` | Có |
| `c1_dev_161` | `FAKE` | Việc phân bổ 11.400 máy tính bảng đến tay học sinh có hoàn cảnh khó khăn, hộ nghèo, học sinh có cha mẹ và người thân mất vì COVID-19 có tổng giá trị lên đến 70 tỷ đồng. | `phân bổ máy tính bảng học sinh hoàn cảnh khó khăn hộ nghèo trẻ mồ` | `cache_hit` | Có |
| `c1_dev_180` | `UNVERIFIABLE` | Công tác chuẩn bị cho hoạt động Tư vấn mùa thi ở An Giang diễn ra sáng 19/3 đã sẵn sàng để tư vấn cho các bạn học sinh về các ngành nghề tương lai. | `tư vấn mùa thi An Giang hướng nghiệp học sinh ngành nghề tương lai` | `cache_hit` | Có |
| `c1_dev_188` | `REAL` | Sở GTVT TP.HCM sẽ nâng cấp hệ thống đèn tín hiệu giao thông kết nối điều khiển tập trung về Trung tâm Quản lý điều hành giao thông đô thị bằng cách tối ưu hoạt động của hệ thống đèn tín hiệu giao thông khu vực trung tâm TP với Trung tâm Quản lý điều hành giao thông đô thị. | `Sở GTVT TP.HCM nâng cấp hệ thống đèn tín hiệu giao thông kết nối điều` | `cache_hit` | Có |
| `c1_dev_222` | `UNVERIFIABLE` | Thổng thống Florida Ron DeSantis, cùng là thành viên Đảng Cộng hòa với ông Trump, chỉ trích rằng quyết định truy tố là "Vũ khí hóa hệ thống pháp luật để thúc đẩy một chương trình nghị sự chính trị". | `Ron DeSantis Đảng Cộng hòa truy tố vũ khí hóa hệ thống pháp luật chương` | `cache_hit` | Có |
| `c1_dev_225` | `FAKE` | Chi phí cho việc số hóa tài liệu lưu trữ cao hơn chi phí xử lý tài liệu tồn đọng. | `chi phí số hóa tài liệu lưu trữ xử lý tài liệu tồn đọng` | `cache_hit` | Có |
| `c1_dev_250` | `UNVERIFIABLE` | Dấu hiệu đặc biệt của ung thư phổi là giảm cân rất nhanh vì người bệnh bị biếng ăn, không thể dung nạp dưỡng chất, ngoài ra ho cũng là dấu hiệu gặp trong ung thư phổi nhưng cũng có thể là dấu hiệu của nhiều bệnh lý khác. | `triệu chứng ung thư phổi sụt cân biếng ăn ho bệnh lý` | `cache_hit` | Có |
| `c1_dev_261` | `REAL` | Dùng quét mã QR để nhận tiền mừng cưới đã diễn ra tại đám cưới của chủ rể Vi Đình Quyết  và cô dâu Hà Thị Vân. | `quét mã QR nhận tiền mừng cưới đám cưới` | `cache_hit` | Có |
| `c1_dev_282` | `REAL` | Giải đấu cờ vua tranh cúp Siêu tốc Bá vương đã tạo được sân chơi trí tuệ chất lượng cho học sinh Hà Nội, các tuyển thủ có cơ hội gặp gỡ các bạn bè ở khắp địa bàn giúp tăng thêm bản lĩnh và kỹ năng chơi cờ. | `giải đấu cờ vua học sinh kỹ năng thi đấu trí tuệ` | `cache_hit` | Có |
| `c1_dev_283` | `REAL` | Học sinh 12 tại Hà Nội tham gia kì thi khảo sát chất lượng vào 7-8/4, học sinh sẽ thi ba môn và một bài tổ hợp. | `học sinh lớp 12 khảo sát chất lượng môn thi bài thi tổ hợp` | `cache_hit` | Có |
| `c1_dev_285` | `FAKE` | Tổng diện tích nhà ở xã hội hoàn thành tại Hà Nội vào năm 2023 dự kiến là 140.000 m2. | `diện tích nhà ở xã hội hoàn thành` | `cache_hit` | Có |
| `c1_dev_288` | `FAKE` | Bạn sinh viên chia sẻ trên trang HUFLIT nói rằng diễn giả nói nhiều câu rất khó nghe nhưng bạn vẫn phải cố nhịn cho đến lúc kết thúc chương trình, hơn nữa cũng có những bạn khác cảm thấy bức xúc giống như bạn. | `diễn giả phát ngôn xúc phạm sinh viên chương trình đào tạo phản ứng dư` | `cache_hit` | Có |
| `c1_dev_291` | `REAL` | Vợ chồng cựu cầu thủ David Beckham đã chứng tỏ được tiềm lực tài chính và chiến lược kinh doanh khi đã có lãi sau 13 năm gồng lỗ. | `David Beckham kinh doanh lợi nhuận tài chính` | `cache_hit` | Có |
| `c1_dev_294` | `FAKE` | Các chính sách hỗ trợ chỉ các doanh nghiệp nhỏ đã được triển khai trong  Luật Hỗ trợ doanh nghiệp. | `chính sách hỗ trợ doanh nghiệp nhỏ Luật Hỗ trợ doanh nghiệp triển khai` | `cache_hit` | Có |
| `c1_dev_301` | `REAL` | Xử phạt từ 50 đến 70 triệu đồng đối với ba quán nhậu trên đường Hoàng Thị Loan Đà Nẵng, trong đó hai quán phải dừng hoạt động vì mở nhạc ồn ào. | `xử phạt quán nhậu vi phạm tiếng ồn đình chỉ hoạt động` | `cache_hit` | Có |
| `c1_dev_304` | `UNVERIFIABLE` | Tại hội nghị thượng đỉnh của các lãnh đạo châu Âu tại Brussels diễn ra vào 18 giờ ngày 16/6, trong khi Tổng thống Pháp Emmanuel Macron trấn an là hệ thống ngân hàng rất vững chắc thì Thủ tướng Đức Olaf Scholz khẳng định Deutsche Bank là doanh nghiệp có lãi và không có lý do gì phải lo lắng. | `hội nghị thượng đỉnh lãnh đạo châu Âu hệ thống ngân hàng Deutsche Bank tình` | `cache_hit` | Có |
| `c1_dev_310` | `FAKE` | Đây là vụ hoả hoạn nghiêm trọng thứ ba trong lịch sử Cuba khi tàn phá 4 trong số 8 bể chứa tại kho nhiên liệu lớn nhất cả nước. | `hỏa hoạn kho nhiên liệu bể chứa Cuba` | `cache_hit` | Có |
| `c1_dev_314` | `UNVERIFIABLE` | Hội Trí thức Việt Nam tại Nhật có hơn 2.000 thành viên đều đang hoạt động và định cư tại Nhật 3 năm trở lên. | `Hội Trí thức Việt Nam tại Nhật thành viên hoạt động định cư` | `cache_hit` | Có |
| `c1_dev_319` | `REAL` | Người đầu tiên lên đến đỉnh núi Hòn Bà ở độ cao 1.500m là Krempf, một cộng sự của Alexandre Yersin. | `người đầu tiên chinh phục đỉnh núi Hòn Bà cộng sự Alexandre Yersin` | `cache_hit` | Có |
| `c1_dev_321` | `UNVERIFIABLE` | VCPMC đã phát hiện hàng trăm chương trình vi phạm quy định ở lĩnh vực biểu diễn, ngoài ra trên môi trường kỹ thuật số, số tiền phạt là 15-30 triệu đối với hành vi truyền đạt tác phẩm đến công chúng mà không được tác giả cho phép, nặng hơn sẽ bị tạm giam 5 đến 7 ngày. | `VCPMC vi phạm quy định biểu diễn môi trường kỹ thuật số xử phạt hành` | `cache_hit` | Có |
| `c1_dev_353` | `FAKE` | Do được thuê nên 3 đối tượng đã đến và hành hung anh L.T.T và bạn anh T. | `thuê hành hung gây thương tích đối tượng` | `cache_hit` | Có |
| `c1_dev_369` | `UNVERIFIABLE` | Ông Jean-Paul Guihaumé: Nhận định đầu tiên đó là việc chúng ta đã có một Hội nghị hợp tác phi tập trung giữa các địa phương của Pháp và Việt Nam lần thứ 12 được tổ chức vào thời điểm rất quan trọng, sau khi đại dịch Covid 19 vừa đi qua. | `hội nghị hợp tác phi tập trung địa phương Pháp Việt Nam đại dịch Covid` | `cache_hit` | Có |
| `c1_dev_373` | `REAL` | Ngành giáo dục nói chung và đào tạo Anh ngữ nói riêng sau đại dịch Covid-19  phải đối mặt với rất nhiều khó khăn để trở lại trạng thái "bình thường mới". | `ngành giáo dục đào tạo tiếng Anh đại dịch Covid-19 khó khăn phục hồi trạng` | `cache_hit` | Có |
| `c1_dev_376` | `REAL` | Người có bằng tốt nghiệp ngành quản trị kinh doanh có cơ hội ứng tuyển nhiều nhất, kế đến là marketing, tài chính ngân hàng, bất động sản trong hầu hết các vị trí công việc trên trang tuyển dụng. | `tốt nghiệp quản trị kinh doanh marketing tài chính ngân hàng bất động sản cơ` | `cache_hit` | Có |
| `c1_dev_447` | `UNVERIFIABLE` | Bộ GD&ĐT vừa có phúc đáp kiến nghị của cử tri Thanh Hóa sau khi phát hiện sự không công bằng trong học bạ. | `Bộ Giáo dục và Đào tạo kiến nghị cử tri Thanh Hóa học bạ không` | `cache_hit` | Có |
| `c1_dev_466` | `UNVERIFIABLE` | Tập trung ở 2 huyện Buôn Đôn và Lắk trên địa bàn Đắk Lắk có 37 con voi nhà, trong đó 20 con voi cái và 17 con voi đực, ngoài ra huyện Xà Bôn cũng có vài con voi nhỏ. | `voi nhà Buôn Đôn Lắk Đắk Lắk số lượng giới tính Xà Bôn` | `cache_hit` | Có |
| `c1_dev_483` | `UNVERIFIABLE` | Anh TTL (ngụ huyện Cư Kuin, Đắk Lắk) 35 tuổi đưa bố 60 tuổi đi khám bệnh nhưng bệnh viện không có loại thuốc đặc trị nên yêu cầu gia đình ra mua ở quầy thuốc đối diện. | `bệnh viện thiếu thuốc đặc trị yêu cầu bệnh nhân mua ngoài` | `cache_hit` | Có |
| `c1_dev_508` | `UNVERIFIABLE` | Bộ NN&PTNT cũng muốn cùng với Sunwah nối lại việc tổ chức diễn đàn xúc tiến thương mại nông sản thường niên và sự kiện xúc tiến thương mại tiêu thụ nông sản trên các lĩnh vực tại cả hai phía Trung Quốc và Việt Nam. | `Bộ Nông nghiệp và Phát triển nông thôn Sunwah diễn đàn xúc tiến thương mại` | `cache_hit` | Có |
| `c1_dev_514` | `UNVERIFIABLE` | P. gọi thêm N.K.H cùng 13 thanh, thiếu niên khác (từ 14 đến 16 tuổi) đang học tại THPT Nguyễn Du mang theo dao, rựa đến nơi đã hẹn, P. cùng nhóm bạn truy đuổi, dùng hung khí chém loạn xạ vào nhóm của K. | `xô xát nhóm thanh thiếu niên sử dụng hung khí gây rối trật tự công` | `cache_hit` | Có |
| `c1_dev_518` | `UNVERIFIABLE` | Chỉ số HNX-Index trên sàn Hà Nội giảm nhẹ gần 1 điểm còn 205 điểm so với năm 2022 là 210 điểm. | `chỉ số HNX-Index sàn giao dịch chứng khoán Hà Nội biến động điểm số` | `cache_hit` | Có |
| `c1_dev_542` | `FAKE` | Ở hạng mục ảnh bộ Giải thưởng Nhiếp ảnh Heritage, bộ ảnh dự thi phải gồm trên 20 ảnh cùng một chủ đề. | `giải thưởng nhiếp ảnh heritage hạng mục ảnh bộ quy định số lượng ảnh dự` | `cache_hit` | Có |
| `c1_dev_593` | `FAKE` | Với việc quán triệt chuyên đề “Những vấn đề về người cao tuổi Việt Nam”, Thường trực Ban Bí thư Trương Thị Mai không đề ra được các chủ trương và chính sách đối với người cao tuổi do sự phức tạp và đa dạng của tình hình người cao tuổi trong xã hội hiện nay. Điều này đòi hỏi sự nghiên cứu kỹ lưỡng và sự phối hợp chặt chẽ giữa các cấp ủy, các ngành và các tổ chức chuyên môn để đưa ra những chính sách phù hợp và hiệu quả nhằm đáp ứng đầy đủ nhu cầu và quyền lợi của người cao tuổi. | `chuyên đề người cao tuổi chủ trương chính sách an sinh xã hội` | `cache_hit` | Có |
| `c1_dev_594` | `FAKE` | Xu hướng chung của các trường tiên tiến trên thế giới là cung cấp môi trường thoải mái, vừa chơi vừa học, không gây áp lực cho học sinh hỗ trợ học sinh tìm được điều mình không thích và đam mê cái gì. | `phương pháp giáo dục trường học quốc tế môi trường học tập áp lực học` | `cache_hit` | Có |
| `c1_dev_598` | `REAL` | Nhiệm vụ chiến lược, thường xuyên của Đảng, Nhà nước chính là tiếp tục đổi mới, phát huy hiệu quả chính sách xã hội, nhưng cũng có những hạn chế sẽ gặp phải khi chính sách xã hội còn thiếu tính bao trùm, không có sự kết nối giữa các chính sách trong tổ chức, tỷ lệ bao phủ không cao. | `chính sách xã hội đổi mới hiệu quả hạn chế tính bao trùm kết nối` | `cache_hit` | Có |
| `c1_dev_599` | `UNVERIFIABLE` | Trước dịch Covid-19, tình trạng vẽ bậy chưa từng xảy ra do ý thức của người dân trước dịch cao hơn. | `tình trạng vẽ bậy ý thức cộng đồng` | `cache_hit` | Có |
| `c1_dev_602` | `REAL` | Ngày 31/3, thông tin vụ việc phát hiện một số thi thể người nước ngoài trong khu vực biển Đài Loan (Trung Quốc) gần đây đã Phó Phát ngôn Bộ Ngoại giao đưa ra câu trả lời. | `phát hiện thi thể người nước ngoài vùng biển Đài Loan Bộ Ngoại giao phản` | `cache_hit` | Có |
| `c1_dev_621` | `FAKE` | Khi thời tiết lạnh, không khí sẽ trở nên lạnh làm cho nước đóng băng, tạo nên các khôi băng khổng lồ. | `thời tiết lạnh không khí nước đóng băng khối băng` | `cache_hit` | Có |
| `c1_dev_626` | `FAKE` | Colombia tuy bị bất lợi khi chơi trên sân khách nhưng vẫn hòa 2-2 với Venezuela. | `Colombia Venezuela kết quả bóng đá hòa` | `cache_hit` | Có |
| `c1_dev_643` | `REAL` | Khách sạn Saigon Morin ở Huế, từng đón tiếp các nhân vật nổi tiếng, đã kỷ niệm 122 năm thành lập vào tối ngày 26.3. | `khách sạn Saigon Morin kỷ niệm thành lập` | `cache_hit` | Có |
| `c1_dev_659` | `UNVERIFIABLE` | Màn tranh đấu kì này chủ yếu là tranh tài giữa đương kim vô địch Sahako và á quân - cựu quán quân mùa 2019 Thái Sơn Nam. | `giải đấu futsal Sahako Thái Sơn Nam tranh chức vô địch` | `cache_hit` | Có |
| `c1_dev_663` | `FAKE` | Với tổng vốn đầu tư hơn 170 triệu USD vào năm 2028, trường đặt mục tiêu hoàn thành xây dựng toàn bộ ba giai đoạn của khuôn viên học tập. | `vốn đầu tư dự án xây dựng khuôn viên học tập trường học` | `cache_hit` | Có |
| `c1_dev_672` | `UNVERIFIABLE` | Thủ tướng Chính phủ phối hợp, đôn đốc triển khai đồng bộ các chủ trương chính sách về hội nhập quốc tế và đề nghị các bộ, ngành trung ương, địa phương và các cơ quan báo chí làm tốt hơn nữa công tác thông tin, tuyên truyền để các cấp, các ngành, doanh nghiệp và các tầng lớp nhân dân nắm vững chủ trương, đường lối của Đảng. | `Thủ tướng Chính phủ phối hợp, đôn đốc triển khai đồng bộ các chủ trương chính sách về hội nhập quốc tế và đề nghị các bộ, ngành trung ương, địa phương và các cơ quan báo chí làm tốt hơn nữa công tác thông tin, tuyên truyền để các cấp, các ngành, doanh nghiệp và các tầng lớp nhân dân nắm vững chủ trương, đường lối của Đảng.` | `fallback:quota_exhausted` | Không |
| `c1_dev_679` | `FAKE` | Tòa án tối cao Hoa Kỳ tuyên bố Bill Clinton phạm tội quấy rối tình dục vào năm 1998. | `Tòa án tối cao Hoa Kỳ tuyên bố Bill Clinton phạm tội quấy rối tình dục vào năm 1998.` | `fallback:quota_exhausted` | Không |
| `c1_dev_702` | `UNVERIFIABLE` | Ông Nguyễn Chí Thiện 60 tuổi, Chủ tịch UBND phường Tân Chánh Hiệp phối hợp với Đoàn y bác sĩ Bệnh viện Quân y 175, doanh nghiệp, nhà thuốc khám bệnh, phát thuốc và tặng quà cho bà con, bước đầu sàng lọc bệnh tật, tiếp đó sẽ có phương án hỗ trợ bà con bảo vệ sức khỏe tốt hơn. | `Ông Nguyễn Chí Thiện 60 tuổi, Chủ tịch UBND phường Tân Chánh Hiệp phối hợp với Đoàn y bác sĩ Bệnh viện Quân y 175, doanh nghiệp, nhà thuốc khám bệnh, phát thuốc và tặng quà cho bà con, bước đầu sàng lọc bệnh tật, tiếp đó sẽ có phương án hỗ trợ bà con bảo vệ sức khỏe tốt hơn.` | `fallback:quota_exhausted` | Không |
