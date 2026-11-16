# BÁO CÁO THỬ NGHIỆM TÌM KIẾM WEB TAVILY (Brief #4 — Task 2)

- **Thời gian chạy (UTC):** `2026-11-09T16:05:49.483004+00:00`
- **Số claim kiểm thử:** `30` (stratified 10 `real`, 10 `fake`, 10 `unverifiable` từ `dev_clean.jsonl`, `seed=42`)
- **Credits đã dùng:** `60 / 150`
- **Độ trễ tìm kiếm (API latency):** `p50 = 4705.05 ms` | `p95 = 9092.76 ms`
- **Tổng số URL trả về đã kiểm tra whitelist:** `300` (`all_urls_in_whitelist = True`)
- **Nguồn trích xuất văn bản:** `{"tavily_raw_content": 140, "trafilatura_fetch": 133, "tavily_snippet": 27}`
  - *Quy tắc chọn văn bản:* Ưu tiên dùng `raw_content` do Tavily trả về trực tiếp trong cùng 1 credit (`search_depth=basic`, `include_raw_content=True`) nếu độ dài `>= 400` ký tự; nếu ngắn hơn hoặc rỗng thì dùng `PoliteArticleFetcher` (`robots.txt` + `rate_limit >= 1.0s` + `trafilatura`).
- **Danh sách 16 domain báo chính thống (`include_domains`):** `vnexpress.net, dantri.com.vn, tuoitre.vn, thanhnien.vn, vietnamnet.vn, vtv.vn, vov.vn, nhandan.vn, baochinhphu.vn, chinhphu.vn, plo.vn, laodong.vn, tienphong.vn, nld.com.vn, vietnamplus.vn, znews.vn`

## 1. Bảng kết quả tổng hợp (Top-5 3-gram Containment so với Gold `Evidence`)

| Truy vấn | Hit >= 0.5 | Hit >= 0.3 | real (>=0.5) | fake (>=0.5) | unv (>=0.5) | real (>=0.3) | fake (>=0.3) | unv (>=0.3) |
|---|---|---|---|---|---|---|---|---|
| **Q0** | **6/30 (20.00%)** | **8/30 (26.67%)** | 2/10 (20.0%) | 1/10 (10.0%) | 3/10 (30.0%) | 3/10 (30.0%) | 1/10 (10.0%) | 4/10 (40.0%) |
| **Q1** | **9/30 (30.00%)** | **12/30 (40.00%)** | 3/10 (30.0%) | 3/10 (30.0%) | 3/10 (30.0%) | 4/10 (40.0%) | 4/10 (40.0%) | 4/10 (40.0%) |

## 2. Phân tích chi tiết — Biến thể `Q0`

> [!NOTE]
> **Lưu ý theo Cổng G3 (Brief #5 §1.4):** Các "lý do trượt" và giải thích cho 5 claim dưới đây (ví dụ về FA Cup mùa 2025, ChatGPT 2024–2026, đối tượng bắt gặp ở quán cafe...) là **giả thuyết phân tích định tính** dựa trên quan sát kết quả tìm kiếm thực tế, không phải chân lý tuyệt đối đã được xác minh toàn diện.

### 2.1. 5 Claim Trúng tiêu biểu (`Q0`)

#### Trúng #1: `tavily_dev_014` (Nhãn: `real` | 3-gram containment = `0.6667`)
- **Claim (`Statement`):** Tại buổi làm việc với lãnh đạo tỉnh Hòa Bình, văn phòng Chính phủ vừa ban hành Thông báo số 95 ngày 25/3/2023 kết luận của Thủ tướng Chính phủ Phạm Minh Chính.
- **Truy vấn (`Q0`):** `Tại buổi làm việc với lãnh đạo tỉnh Hòa Bình, văn phòng Chính phủ vừa ban hành Thông báo số 95 ngày 25/3/2023 kết luận của Thủ tướng Chính phủ Phạm Minh Chính.`
- **Gold `Evidence`:** Văn phòng Chính phủ vừa ban hành Thông báo số 95 ngày 25/3/2023 kết luận của Thủ tướng Chính phủ Phạm Minh Chính tại buổi làm việc với lãnh đạo tỉnh Hòa Bình.
- **Top URL khớp nhất:** `https://vanban.chinhphu.vn?pageid=27160&docid=207626` (nguồn text: `trafilatura_fetch`)
- **Tiêu đề bài báo:** Thông báo số 95/TB-VPCP của Văn phòng Chính phủ
- **Lý do trúng:** Câu truy vấn chứa cụm thực thể/sự kiện đặc trưng giúp Tavily định vị đúng bài báo gốc trên báo chính thống; văn bản trích xuất chứa nguyên văn câu bằng chứng.

#### Trúng #2: `tavily_dev_062` (Nhãn: `unverifiable` | 3-gram containment = `0.7273`)
- **Claim (`Statement`):** Trong 56 cháu có biểu hiện đau bụng, buồn nôn có 41 cháu đang nhập viện khám sàng lọc, số còn lại  đã được phụ huynh đón về tiếp tục theo dõi sức khỏe tại nhà với sự theo dõi của bộ y tế vì bệnh viện thiếu giường.
- **Truy vấn (`Q0`):** `Trong 56 cháu có biểu hiện đau bụng, buồn nôn có 41 cháu đang nhập viện khám sàng lọc, số còn lại đã được phụ huynh đón về tiếp tục theo dõi sức khỏe tại nhà với sự theo dõi của bộ y tế vì bệnh viện thiếu giường.`
- **Gold `Evidence`:** 56 cháu có biểu hiện đau bụng, buồn nôn. 41 cháu đang nhập viện khám sàng lọc Các cháu còn lại đã được phụ huynh đón về tiếp tục theo dõi sức khỏe tại nhà
- **Top URL khớp nhất:** `https://nhandan.vn/suc-khoe-cac-chau-nghi-ngo-doc-thuc-pham-o-truong-tieu-hoc-kim-giang-ra-sao-post745209.html` (nguồn text: `trafilatura_fetch`)
- **Tiêu đề bài báo:** Sức khỏe các cháu nghi ngộ độc thực phẩm ở Trường Tiểu học Kim ...
- **Lý do trúng:** Câu truy vấn chứa cụm thực thể/sự kiện đặc trưng giúp Tavily định vị đúng bài báo gốc trên báo chính thống; văn bản trích xuất chứa nguyên văn câu bằng chứng.

#### Trúng #3: `tavily_dev_089` (Nhãn: `unverifiable` | 3-gram containment = `1.0000`)
- **Claim (`Statement`):** Ở tuổi 58, Tấn Beo vẫn đi diễn cùng con trai ở các tỉnh miền Tây, miền Trung 
- **Truy vấn (`Q0`):** `Ở tuổi 58, Tấn Beo vẫn đi diễn cùng con trai ở các tỉnh miền Tây, miền Trung`
- **Gold `Evidence`:** Ở tuổi 58, Tấn Beo vẫn nhận lời mời lưu diễn ở các tỉnh miền Tây, miền Trung.
- **Top URL khớp nhất:** `https://tienphong.vn/nghe-si-tan-beo-bat-tinh-sau-khi-roi-tu-do-xuong-cau-thang-post1518609.tpo` (nguồn text: `tavily_snippet`)
- **Tiêu đề bài báo:** Nghệ sĩ Tấn Beo bất tỉnh sau khi rơi tự do xuống cầu thang
- **Lý do trúng:** Câu truy vấn chứa cụm thực thể/sự kiện đặc trưng giúp Tavily định vị đúng bài báo gốc trên báo chính thống; văn bản trích xuất chứa nguyên văn câu bằng chứng.

#### Trúng #4: `tavily_dev_237` (Nhãn: `fake` | 3-gram containment = `1.0000`)
- **Claim (`Statement`):** Theo Bộ trưởng Bộ Thể thao Nga Oleg Matytsin, các điều kiện do Ủy ban Olympic quốc tế (IOC) đưa ra đối với các vận động viên Nga và Belarus là phù hợp và cho rằng đó là việc phân biệt đối xử đối với các vận động viên.
- **Truy vấn (`Q0`):** `Theo Bộ trưởng Bộ Thể thao Nga Oleg Matytsin, các điều kiện do Ủy ban Olympic quốc tế (IOC) đưa ra đối với các vận động viên Nga và Belarus là phù hợp và cho rằng đó là việc phân biệt đối xử đối với các vận động viên.`
- **Gold `Evidence`:** Theo Bộ trưởng Matytsin, các điều kiện do IOC đưa ra đối với các vận động viên Nga và Belarus là không phù hợp. Đồng thời đó là việc phân biệt đối xử đối với các vận động viên.
- **Top URL khớp nhất:** `https://tuoitre.vn/bo-the-thao-nga-chi-trich-khuyen-nghi-cua-ioc-20230329113212004.htm` (nguồn text: `trafilatura_fetch`)
- **Tiêu đề bài báo:** Bộ Thể thao Nga chỉ trích khuyến nghị của IOC - Tuổi Trẻ Online
- **Lý do trúng:** Câu truy vấn chứa cụm thực thể/sự kiện đặc trưng giúp Tavily định vị đúng bài báo gốc trên báo chính thống; văn bản trích xuất chứa nguyên văn câu bằng chứng.

#### Trúng #5: `tavily_dev_453` (Nhãn: `unverifiable` | 3-gram containment = `0.6842`)
- **Claim (`Statement`):** Năm 2022, nhờ vào các kế hoạch triển khai phát triển lâm nghiệp của nhà nước đề ra đã đạt được sự thành công, người dân đã được tạo nguồn thu nhập và sinh kế thông qua các hoạt động sản xuất lâm nghiệp, chính sách khoán bảo vệ rừng và dịch vụ môi trường rừng. Nhờ vào đó góp phần xóa đói giảm nghèo, tăng thêm thu nhập cho dân.
- **Truy vấn (`Q0`):** `Năm 2022, nhờ vào các kế hoạch triển khai phát triển lâm nghiệp của nhà nước đề ra đã đạt được sự thành công, người dân đã được tạo nguồn thu nhập và sinh kế thông qua các hoạt động sản xuất lâm nghiệp, chính sách khoán bảo vệ rừng và dịch vụ môi trường rừng. Nhờ vào đó góp phần xóa đói giảm nghèo, tăng thêm thu nhập cho dân.`
- **Gold `Evidence`:** Các hoạt động sản xuất lâm nghiệp, chính sách khoán bảo vệ rừng, dịch vụ môi trường rừng cho hộ gia đình, cá nhân và cộng đồng dân cư đã tạo nguồn thu nhập và sinh kế cho người dân
- **Top URL khớp nhất:** `https://vietnamnet.vn/gia-tri-xuat-khau-go-va-lam-san-dat-binh-quan-15-8-ty-usd-nam-2262938.html` (nguồn text: `trafilatura_fetch`)
- **Tiêu đề bài báo:** Giá trị xuất khẩu gỗ và lâm sản đạt bình quân 15,8 tỷ USD/năm
- **Lý do trúng:** Câu truy vấn chứa cụm thực thể/sự kiện đặc trưng giúp Tavily định vị đúng bài báo gốc trên báo chính thống; văn bản trích xuất chứa nguyên văn câu bằng chứng.

### 2.2. 5 Claim Trượt tiêu biểu (`Q0`)

#### Trượt #1: `tavily_dev_019` (Nhãn: `real` | 3-gram containment = `0.0000`)
- **Claim (`Statement`):** Man Utd chơi không tốt trong trận tứ kết FA Cup với Fulham.
- **Truy vấn (`Q0`):** `Man Utd chơi không tốt trong trận tứ kết FA Cup với Fulham.`
- **Gold `Evidence`:** Cựu danh thủ Roy Keane nhấn mạnh Man Utd thi đấu nghèo nàn tới mức không thể tin được trước Fulham
- **Top URL trả về:** `https://thethao.vtv.vn/bong-da-quoc-te/boc-tham-tu-ket-fa-cup-man-utd-gap-fulham-man-city-gap-burnley-20230302083955717.htm`
- **Danh sách Top-5 URL:** `https://thethao.vtv.vn/bong-da-quoc-te/boc-tham-tu-ket-fa-cup-man-utd-gap-fulham-man-city-gap-burnley-20230302083955717.htm, https://dantri.com.vn/the-thao/nhan-lien-3-the-do-mot-qua-phat-den-fulham-thua-nguoc-man-utd-20230320060219092.htm, https://laodong.vn/bong-da-quoc-te/thong-ke-thanh-tich-doi-dau-man-united-vs-fulham-tai-fa-cup-1470806.ldo, https://dantri.com.vn/the-thao/duong-kim-vo-dich-man-utd-bi-loai-khoi-fa-cup-20250303055640830.htm, https://vnexpress.net/man-utd-bi-loai-khoi-cup-fa-4855914.html`
- **Lý do trượt:** Bài báo gốc trong ViFactCheck thuộc domain ngoài danh sách 16 báo hoặc đã đổi URL/tiêu đề trên chỉ mục tìm kiếm, hoặc bài tìm được cùng chủ đề nhưng khác bản tin cụ thể mà ViFactCheck trích.

#### Trượt #2: `tavily_dev_312` (Nhãn: `unverifiable` | 3-gram containment = `0.0000`)
- **Claim (`Statement`):** ChatGPT giúp người dùng có thể trao đổi trực tiếp và gián tiếp thông qua hình thức hỏi đáp là chính với những có sở dữ liệu có sẵn được thu nhập kể từ năm 2021 trở về trước.
- **Truy vấn (`Q0`):** `ChatGPT giúp người dùng có thể trao đổi trực tiếp và gián tiếp thông qua hình thức hỏi đáp là chính với những có sở dữ liệu có sẵn được thu nhập kể từ năm 2021 trở về trước.`
- **Gold `Evidence`:** ChatGPT giúp người dùng trao đổi trực tiếp thông qua hình thức hỏi đáp là chính với những cơ sở dữ liệu thu thập được trên internet, mạng xã hội, các trang web kể từ năm 2021 trở về trước
- **Top URL trả về:** `https://news.tuoitre.vn/chatgpt-users-can-now-browse-internet-openai-says-10375849.htm`
- **Danh sách Top-5 URL:** `https://news.tuoitre.vn/chatgpt-users-can-now-browse-internet-openai-says-10375849.htm, https://e.vnexpress.net/news/world/chatgpt-users-can-now-browse-internet-openai-says-4658348.html, https://news.laodong.vn/cong-nghe/chatgpt-co-mat-tren-powerpoint-ho-tro-tao-va-chinh-sua-slide-bang-ai-1706817.ldo, https://news.laodong.vn/cong-nghe/chatgpt-co-tinh-nang-suc-khoe-moi-ket-noi-apple-health-1740018.ldo, https://vietnamnet.vn/en/vietnam-s-big-tech-firms-remain-indifferent-to-chatgpt-2150525.html`
- **Lý do trượt:** Bài báo gốc trong ViFactCheck thuộc domain ngoài danh sách 16 báo hoặc đã đổi URL/tiêu đề trên chỉ mục tìm kiếm, hoặc bài tìm được cùng chủ đề nhưng khác bản tin cụ thể mà ViFactCheck trích.

#### Trượt #3: `tavily_dev_395` (Nhãn: `unverifiable` | 3-gram containment = `0.0000`)
- **Claim (`Statement`):** Một trong những đối tượng hành hung bắt gặp anh Thiện và anh Tây cùng với gia đình tại một quán cafe.
- **Truy vấn (`Q0`):** `Một trong những đối tượng hành hung bắt gặp anh Thiện và anh Tây cùng với gia đình tại một quán cafe.`
- **Gold `Evidence`:** Quốc Chiến đang ngồi tại một quán ở trước Công viên văn hóa An Hòa (TP Rạch Giá) thì thấy anh Thiện Chiến đang ngồi bàn bên cạnh cùng với anh Tây và một số người khác Thấy vậy, Quốc Chiến đi tìm và cho Thành, Điền, Hải, Kiệt hay, rồi cả nhóm cùng nhau đi lấy hung khí.
- **Top URL trả về:** `https://nhandan.vn/khoi-to-doi-tuong-sat-hai-3-nguoi-o-dong-nai-de-dieu-tra-ve-toi-giet-nguoi-va-su-dung-trai-phep-vu-khi-quan-dung-post986606.html`
- **Danh sách Top-5 URL:** `https://nhandan.vn/khoi-to-doi-tuong-sat-hai-3-nguoi-o-dong-nai-de-dieu-tra-ve-toi-giet-nguoi-va-su-dung-trai-phep-vu-khi-quan-dung-post986606.html, https://vietnamnet.vn/tinh-tiet-dau-long-cua-vu-an-mang-nan-nhan-va-hung-thu-hoan-toan-xa-la-2551804.html, https://nhandan.vn/siet-chat-cac-quy-dinh-ve-thuoc-la-khong-de-khoang-trong-bao-ve-gioi-tre-post1001256.html, https://nhandan.vn/thong-cao-bao-chi-so-18-ky-hop-thu-hai-quoc-hoi-khoa-xvi-post1000525.html, https://chinhphu.vn`
- **Lý do trượt:** Bài báo gốc trong ViFactCheck thuộc domain ngoài danh sách 16 báo hoặc đã đổi URL/tiêu đề trên chỉ mục tìm kiếm, hoặc bài tìm được cùng chủ đề nhưng khác bản tin cụ thể mà ViFactCheck trích.

#### Trượt #4: `tavily_dev_571` (Nhãn: `fake` | 3-gram containment = `0.0238`)
- **Claim (`Statement`):** Trịnh Tiến Dũng điều hành đường dây buôn bán trái phép từ nhà riêng ở Mỹ, không có đồng phạm tại Việt Nam.
- **Truy vấn (`Q0`):** `Trịnh Tiến Dũng điều hành đường dây buôn bán trái phép từ nhà riêng ở Mỹ, không có đồng phạm tại Việt Nam.`
- **Gold `Evidence`:** Dũng chỉ đạo 18 công ty ở Việt Nam ký hợp đồng mua CD Rom chứa phần mềm Adobe giả từ 7 công ty của Dũng ở nước ngoài. Tại Việt Nam, Dũng thuê người dùng chứng minh nhân dân để lập công ty.
- **Top URL trả về:** `https://media.chinhphu.vn/video/dieu-hanh-phuong-tien-khong-co-giay-phep-kinh-doanh-van-tai-bi-phat-16-trieu-dong-22684.htm`
- **Danh sách Top-5 URL:** `https://vtv.vn/duong-day-buon-ban-ma-tuy.html, https://baochinhphu.vn/triet-pha-duong-day-san-xuat-mua-ban-trai-phep-chat-doc-thach-tin-dung-trong-nha-khoa-102260128141351156.htm, https://media.chinhphu.vn/video/dieu-hanh-phuong-tien-khong-co-giay-phep-kinh-doanh-van-tai-bi-phat-16-trieu-dong-22684.htm, https://laodong.vn/bat-dong-san/muc-xu-phat-hanh-vi-xay-nha-khong-co-giay-phep-xay-dung-nam-2025-1520632.ldo, https://tienphong.vn/ha-noi-lan-chiem-dat-rung-xay-dung-khu-nghi-duong-trai-phep-post1738537.tpo`
- **Lý do trượt:** Bài báo gốc trong ViFactCheck thuộc domain ngoài danh sách 16 báo hoặc đã đổi URL/tiêu đề trên chỉ mục tìm kiếm, hoặc bài tìm được cùng chủ đề nhưng khác bản tin cụ thể mà ViFactCheck trích.

#### Trượt #5: `tavily_dev_522` (Nhãn: `fake` | 3-gram containment = `0.0333`)
- **Claim (`Statement`):** Phản hồi việc đi trễ, ông cho biết mình đã đến trường từ lúc 9h, nhưng phải đi tìm bãi đậu xe ở xa và đi bộ đến bộ, dẫn đến việc lên sân khấu trễ.
- **Truy vấn (`Q0`):** `Phản hồi việc đi trễ, ông cho biết mình đã đến trường từ lúc 9h, nhưng phải đi tìm bãi đậu xe ở xa và đi bộ đến bộ, dẫn đến việc lên sân khấu trễ.`
- **Gold `Evidence`:** Về vấn đề đi trễ, ông cho biết đã đến trường lúc 8h30 phải đi tìm chỗ gửi xe khác rồi đi bộ đến trường, dẫn tới việc lên sân khấu trễ.
- **Top URL trả về:** `https://tuoitre.vn/bai-giu-o-to-o-benh-vien-da-khoa-kien-giang-giam-doc-noi-cho-tinh-duyet-roi-lam-20231109174845749.htm`
- **Danh sách Top-5 URL:** `https://tuoitre.vn/bai-giu-o-to-o-benh-vien-da-khoa-kien-giang-giam-doc-noi-cho-tinh-duyet-roi-lam-20231109174845749.htm, https://tuoitre.vn/canh-sat-giao-thong-tp-hcm-xu-ly-tai-xe-lai-o-to-lap-den-nhu-san-khau-di-dong-20240528114248073.htm, https://laodong.vn/ban-doc/tranh-luan-ve-viec-cho-tre-di-hoc-truc-tiep-981786.ldo, https://tienphong.vn/gia-xang-dau-tang-anh-huong-den-nhip-song-gen-z-dan-thich-nghi-nhu-the-nao-post1827008.tpo, https://tuoitre.vn/dung-xe-hoi-cho-di-xa-rac-bua-bai-20231025103221421.htm`
- **Lý do trượt:** Bài báo gốc trong ViFactCheck thuộc domain ngoài danh sách 16 báo hoặc đã đổi URL/tiêu đề trên chỉ mục tìm kiếm, hoặc bài tìm được cùng chủ đề nhưng khác bản tin cụ thể mà ViFactCheck trích.


## 2. Phân tích chi tiết — Biến thể `Q1`

### 2.1. 5 Claim Trúng tiêu biểu (`Q1`)

#### Trúng #1: `tavily_dev_014` (Nhãn: `real` | 3-gram containment = `0.6667`)
- **Claim (`Statement`):** Tại buổi làm việc với lãnh đạo tỉnh Hòa Bình, văn phòng Chính phủ vừa ban hành Thông báo số 95 ngày 25/3/2023 kết luận của Thủ tướng Chính phủ Phạm Minh Chính.
- **Truy vấn (`Q1`):** `Tại buổi làm lãnh đạo tỉnh Hòa Bình văn phòng Chính phủ ban hành Thông báo số 95 ngày 25/3/2023 kết luận Thủ tướng Chính Phạm Minh Chính`
- **Gold `Evidence`:** Văn phòng Chính phủ vừa ban hành Thông báo số 95 ngày 25/3/2023 kết luận của Thủ tướng Chính phủ Phạm Minh Chính tại buổi làm việc với lãnh đạo tỉnh Hòa Bình.
- **Top URL khớp nhất:** `https://vanban.chinhphu.vn?pageid=27160&docid=207626` (nguồn text: `tavily_raw_content`)
- **Tiêu đề bài báo:** Thông báo số 95/TB-VPCP của Văn phòng Chính phủ: Kết luận của Thủ tướng Chính phủ Phạm Minh Chính tại buổi làm việc với lãnh đạo tỉnh Hòa Bình
- **Lý do trúng:** Chứa số hiệu văn bản (`Thông báo số 95`), ngày tháng (`25/3/2023`) và thực thể (`Phạm Minh Chính`, `Hòa Bình`), giúp Tavily định vị chính xác trang văn bản trên subdomain `vanban.chinhphu.vn`.

#### Trúng #2: `tavily_dev_089` (Nhãn: `unverifiable` | 3-gram containment = `1.0000`)
- **Claim (`Statement`):** Ở tuổi 58, Tấn Beo vẫn đi diễn cùng con trai ở các tỉnh miền Tây, miền Trung 
- **Truy vấn (`Q1`):** `Ở tuổi 58 Tấn Beo vẫn đi diễn cùng con trai tỉnh miền Tây Trung`
- **Gold `Evidence`:** Ở tuổi 58, Tấn Beo vẫn nhận lời mời lưu diễn ở các tỉnh miền Tây, miền Trung.
- **Top URL khớp nhất:** `https://tienphong.vn/nghe-si-tan-beo-bat-tinh-sau-khi-roi-tu-do-xuong-cau-thang-post1518609.tpo` (nguồn text: `tavily_raw_content`)
- **Tiêu đề bài báo:** Nghệ sĩ Tấn Beo bất tỉnh sau khi rơi tự do xuống cầu thang
- **Lý do trúng:** Dù claim thuộc nhãn `unverifiable` (bịa thêm chi tiết `"cùng con trai"`), phần khung thực thể (`"Ở tuổi 58"`, `"Tấn Beo"`, `"miền Tây, miền Trung"`) vẫn khớp hoàn toàn với bài báo gốc trên `tienphong.vn` (`containment = 1.0000`).

#### Trúng #3: `tavily_dev_237` (Nhãn: `fake` | 3-gram containment = `0.9091`)
- **Claim (`Statement`):** Theo Bộ trưởng Bộ Thể thao Nga Oleg Matytsin, các điều kiện do Ủy ban Olympic quốc tế (IOC) đưa ra đối với các vận động viên Nga và Belarus là phù hợp và cho rằng đó là việc phân biệt đối xử đối với các vận động viên.
- **Truy vấn (`Q1`):** `Theo Bộ trưởng Bộ Thể thao Nga Oleg Matytsin điều kiện Ủy ban Olympic quốc tế IOC đưa đối vận động viên Nga Belarus phù hợp phân biệt xử`
- **Gold `Evidence`:** Theo Bộ trưởng Matytsin, các điều kiện do IOC đưa ra đối với các vận động viên Nga và Belarus là không phù hợp. Đồng thời đó là việc phân biệt đối xử đối với các vận động viên.
- **Top URL khớp nhất:** `https://tuoitre.vn/bo-the-thao-nga-chi-trich-khuyen-nghi-cua-ioc-20230329113212004.htm` (nguồn text: `tavily_raw_content`)
- **Tiêu đề bài báo:** Bộ Thể thao Nga chỉ trích khuyến nghị của IOC - Tuổi Trẻ Online
- **Lý do trúng:** Thực thể tên riêng (`Oleg Matytsin`, `IOC`, `Nga`, `Belarus`) rất đặc trưng nên dù claim đảo ngược ngữ nghĩa (`"không phù hợp"` -> `"phù hợp"`), Tavily vẫn lấy đúng bài báo gốc trên `tuoitre.vn`.

#### Trúng #4: `tavily_dev_299` (Nhãn: `fake` | 3-gram containment = `0.7234`)
- **Claim (`Statement`):** Phương án thi tốt nghiệp THPT mới của Bộ GD-ĐT yêu cầu tất cả thí sinh phải sử dụng máy tính để làm bài thi.
- **Truy vấn (`Q1`):** `Phương án thi tốt nghiệp THPT Bộ GD ĐT yêu cầu tất thí sinh phải sử dụng máy tính làm bài`
- **Gold `Evidence`:** Giai đoạn sau năm 2030, sẽ phấn đấu để đến khi tất cả các địa phương trên toàn quốc có đủ điều kiện để tổ chức thi trên máy tính sẽ chuyển sang tổ chức kỳ thi tốt nghiệp THPT trên máy tính đối với các môn thi trắc nghiệm.
- **Top URL khớp nhất:** `https://xaydungchinhsach.chinhphu.vn/cong-bo-phuong-an-thi-tot-nghiep-thpt-tu-nam-2025-119231129112222933.htm` (nguồn text: `tavily_raw_content`)
- **Tiêu đề bài báo:** PHƯƠNG ÁN THI TỐT NGHIỆP THPT TỪ NĂM 2025
- **Lý do trúng:** Biến thể `Q1` lược bỏ các hư từ (`mới của`, `tất cả`, `để`) giúp tăng mật độ từ khóa (`Phương án thi tốt nghiệp THPT`, `Bộ GD ĐT`, `sử dụng máy tính`), nhờ đó `Q1` tìm trúng bài trên `xaydungchinhsach.chinhphu.vn` trong khi `Q0` bị trượt.

#### Trúng #5: `tavily_dev_408` (Nhãn: `unverifiable` | 3-gram containment = `0.9756`)
- **Claim (`Statement`):** Sự trở lại của show diễn nội y đã trở thành chủ đề bàn tán trên mạng xã hội, nhiều người bày tỏ sự ủng hộ, nhưng số khác lại tỏ ra hờ hững, thậm chí còn chỉ trích nặng nề và bình luận rằng show diễn này không nên tiếp tục.
- **Truy vấn (`Q1`):** `Sự trở show diễn nội y trở thành chủ đề bàn tán mạng xã hội bày tỏ ủng hộ số khác hờ hững thậm chí chỉ trích nặng nề bình luận tiếp tục`
- **Gold `Evidence`:** Tuyên bố tái xuất của show diễn nội y đang trở thành chủ đề bàn tán trên mạng xã hội Một số khán giả bày tỏ sự ủng hộ. Tuy nhiên, một số khác lại tỏ ra hờ hững với show diễn này.
- **Top URL khớp nhất:** `https://dantri.com.vn/giai-tri/khan-gia-tranh-cai-truoc-su-tro-lai-cua-show-dien-noi-y-nong-nhat-hanh-tinh-20230306202555433.htm` (nguồn text: `trafilatura_fetch`)
- **Tiêu đề bài báo:** Khán giả tranh cãi trước sự trở lại của show diễn nội y nóng ... - Dân trí
- **Lý do trúng:** Cụm từ khóa `"show diễn nội y"`, `"chủ đề bàn tán mạng xã hội"`, `"hờ hững"` khớp trực tiếp với sapo bài báo trên `dantri.com.vn` (được `PoliteArticleFetcher` tải đầy đủ qua `trafilatura`).

### 2.2. 5 Claim Trượt tiêu biểu (`Q1`)

#### Trượt #1: `tavily_dev_395` (Nhãn: `unverifiable` | 3-gram containment = `0.0357`)
- **Claim (`Statement`):** Một trong những đối tượng hành hung bắt gặp anh Thiện và anh Tây cùng với gia đình tại một quán cafe.
- **Truy vấn (`Q1`):** `Một đối tượng hành hung bắt gặp anh Thiện Tây cùng gia đình quán cafe`
- **Gold `Evidence`:** Quốc Chiến đang ngồi tại một quán ở trước Công viên văn hóa An Hòa (TP Rạch Giá) thì thấy anh Thiện Chiến đang ngồi bàn bên cạnh cùng với anh Tây và một số người khác Thấy vậy, Quốc Chiến đi tìm và cho Thành, Điền, Hải, Kiệt hay, rồi cả nhóm cùng nhau đi lấy hung khí.
- **Top URL trả về:** `https://vnexpress.net/tong-tai-gio-tay-ra-lenh-danh-nguoi-o-quan-ca-phe-bi-bat-4991757.html`
- **Danh sách Top-5 URL:** `https://vnexpress.net/tong-tai-gio-tay-ra-lenh-danh-nguoi-o-quan-ca-phe-bi-bat-4991757.html, https://tuoitre.vn/nld/tong-tai-trong-vu-hanh-hung-nhan-vien-quan-ca-phe-bi-bat-196251209163328435.htm, https://vnexpress.net/thanh-nien-danh-nguoi-sau-cai-chi-tay-cua-tong-tai-bi-bat-4941381.html, https://laodong.vn/xa-hoi/tong-tai-ra-lenh-danh-nhan-vien-quan-ca-phe-den-xin-loi-nhung-bi-tu-choi-moi-ra-ngoai-1577082.ldo, https://www.vietnamplus.vn/loi-khai-cua-ke-dot-quan-caphe-khien-11-nguoi-tu-vong-post1002930.vnp`
- **Lý do trượt (Claim thiếu định danh địa điểm/sự kiện):** Claim lược bỏ hoàn toàn địa danh `"Công viên văn hóa An Hòa (TP Rạch Giá)"` và họ tên đầy đủ `"Quốc Chiến"`, chỉ còn tên gọi cụt `"anh Thiện"`, `"anh Tây"` và `"quán cafe"` nên công cụ tìm kiếm trả về các vụ hành hung ở quán cà phê nổi tiếng gần đây.

#### Trượt #2: `tavily_dev_137` (Nhãn: `unverifiable` | 3-gram containment = `0.0488`)
- **Claim (`Statement`):** Phúc Long tiếp cận khách hàng thông qua nhiều kênh bán hàng trực tuyến và chiếm 35% tổng doanh thu, trong đó cao nhất là thông qua Grabfood.
- **Truy vấn (`Q1`):** `Phúc Long tiếp cận khách hàng thông kênh bán trực tuyến chiếm 35 tổng doanh thu cao nhất Grabfood`
- **Gold `Evidence`:** Phúc Long cũng cho biết đã có những bước tiến trong việc tiếp cận khách hàng thông qua nhiều kênh, bao gồm dịch vụ giao hàng và tiêu dùng tại cửa hàng. Doanh số bán hàng trực tuyến chiếm 35% tổng doanh thu.
- **Top URL trả về:** `https://vnexpress.net/grabfood-trien-khai-chuong-trinh-giup-doi-tac-nha-hang-tang-doanh-thu-5075942.html`
- **Danh sách Top-5 URL:** `https://vnexpress.net/grabfood-tang-truong-25-lan-sau-8-thang-van-hanh-3884521.html, https://vnexpress.net/grabfood-trien-khai-chuong-trinh-giup-doi-tac-nha-hang-tang-doanh-thu-5075942.html, https://vnexpress.net/60-quan-an-tai-tp-hcm-duoc-grabfood-trao-giai-tru-danh-5059743.html, https://vnexpress.net/grab-xanh-sm-chiem-gan-90-thi-truong-goi-xe-4889391.html, https://vnexpress.net/nguoi-tre-tan-dung-chuong-trinh-tich-diem-khi-dat-thuc-an-4941088.html`
- **Lý do trượt (Chi tiết bịa đặt trong claim dẫn lệch hướng tìm kiếm):** Phần bịa đặt của nhãn `unverifiable` (`"cao nhất là thông qua Grabfood"`) không hề tồn tại trong bài báo gốc nói về Phúc Long, khiến Tavily ưu tiên các bài viết về doanh thu của `GrabFood` thay vì báo cáo tài chính của `Phúc Long`.

#### Trượt #3: `tavily_dev_482` (Nhãn: `fake` | 3-gram containment = `0.0513`)
- **Claim (`Statement`):** Thí sinh chưa có chứng chỉ tiếng Anh theo quy định của trường sẽ được hỗ trợ thi đánh giá năng lực tiếng Anh với đề được ra dựa trên khung tham chiếu ngôn ngữ chung châu Phi.
- **Truy vấn (`Q1`):** `Thí sinh chưa chứng chỉ tiếng Anh quy định trường hỗ trợ thi đánh giá năng lực Anh đề dựa khung tham chiếu ngôn ngữ chung châu Phi`
- **Gold `Evidence`:** thí sinh chưa có chứng chỉ tiếng Anh quốc tế theo quy định của Trường trường hỗ trợ thi đánh giá năng lực tiếng Anh, với đề thi được xây dựng theo khung tham chiếu ngôn ngữ chung châu Âu (CEFR),
- **Top URL trả về:** `https://www.vietnamplus.vn/ha-noi-doi-moi-quan-tri-to-chuc-bo-may-va-su-dung-doi-ngu-post1146737.vnp`
- **Danh sách Top-5 URL:** `https://xaydungchinhsach.chinhphu.vn/10-diem-moi-cua-chuong-trinh-giao-duc-pho-thong-2018-119230206174054873.htm, https://www.vietnamplus.vn/ha-noi-doi-moi-quan-tri-to-chuc-bo-may-va-su-dung-doi-ngu-post1146737.vnp, https://xaydungchinhsach.chinhphu.vn/sang-5-9-truc-tiep-le-khai-giang-nam-hoc-nam-moi-va-ky-niem-80-nam-thanh-lap-bo-quoc-gia-giao-duc-119250903163552636.htm, https://www.vietnamplus.vn/bao-dam-nguoi-benh-khong-bi-ket-giua-co-che-mua-va-co-che-thanh-toan-post1146597.vnp, https://www.vietnamplus.vn/nhin-lai-nhung-con-bao-manh-do-bo-vao-viet-nam-nhung-nam-gan-day-post975874.vnp`
- **Lý do trượt (Từ khóa bị sửa sai của nhãn Fake + thiếu tên trường):** Claim vừa không nêu tên trường đại học cụ thể (chỉ ghi `"của trường"`), vừa sửa `"châu Âu"` thành `"châu Phi"` làm nhiễu truy vấn tìm kiếm.

#### Trượt #4: `tavily_dev_019` (Nhãn: `real` | 3-gram containment = `0.0526`)
- **Claim (`Statement`):** Man Utd chơi không tốt trong trận tứ kết FA Cup với Fulham.
- **Truy vấn (`Q1`):** `Man Utd chơi tốt trận tứ kết FA Cup Fulham`
- **Gold `Evidence`:** Cựu danh thủ Roy Keane nhấn mạnh Man Utd thi đấu nghèo nàn tới mức không thể tin được trước Fulham
- **Top URL trả về:** `https://vnexpress.net/henry-man-utd-da-choi-5-tran-chung-ket-arsenal-thi-khong-4891638.html`
- **Danh sách Top-5 URL:** `https://vnexpress.net/chu-de/cup-fa-943-p4, https://dantri.com.vn/the-thao/danh-bai-man-utd-man-city-vo-dich-fa-cup-20230603205114323.htm, https://vnexpress.net/man-utd-gap-hoa-truoc-chung-ket-fa-cup-4742479.html, https://vnexpress.net/man-utd-gap-doi-thu-ngoai-hang-anh-o-cup-fa-4847968.html, https://vnexpress.net/henry-man-utd-da-choi-5-tran-chung-ket-arsenal-thi-khong-4891638.html`
- **Lý do trượt (Diễn đạt khái quát + trùng sự kiện nhiều mùa giải):** Claim diễn giải lại câu nhận xét của `"Roy Keane"` thành câu chung chung `"Man Utd chơi không tốt"` và không có năm `2023`; hơn nữa Man Utd gặp lại Fulham ở FA Cup mùa 2025 nên các bài báo mới hơn (2024–2025) chiếm chỗ trong top-5.

#### Trượt #5: `tavily_dev_312` (Nhãn: `unverifiable` | 3-gram containment = `0.0526`)
- **Claim (`Statement`):** ChatGPT giúp người dùng có thể trao đổi trực tiếp và gián tiếp thông qua hình thức hỏi đáp là chính với những có sở dữ liệu có sẵn được thu nhập kể từ năm 2021 trở về trước.
- **Truy vấn (`Q1`):** `ChatGPT giúp dùng thể trao đổi trực tiếp gián thông hình thức hỏi đáp chính sở dữ liệu sẵn thu nhập kể năm 2021 trở`
- **Gold `Evidence`:** ChatGPT giúp người dùng trao đổi trực tiếp thông qua hình thức hỏi đáp là chính với những cơ sở dữ liệu thu thập được trên internet, mạng xã hội, các trang web kể từ năm 2021 trở về trước
- **Top URL trả về:** `https://vietnamnet.vn/6-ly-do-chatgpt-tra-loi-sai-khong-chinh-xac-2131255.html`
- **Danh sách Top-5 URL:** `https://vietnamnet.vn/6-ly-do-chatgpt-tra-loi-sai-khong-chinh-xac-2131255.html, https://vietnamnet.vn/chon-nham-ky-ke-khai-thue-qua-han-74-ngay-bi-xu-phat-the-nao-2519557.html, https://vietnamnet.vn/nguoi-dung-chatgpt-mien-phi-viet-nam-bat-dau-thay-quang-cao-tu-hom-nay-2558343.html, https://vietnamnet.vn/nguoi-dung-toan-cau-chi-85-ty-usd-cho-ung-dung-chatgpt-va-gemini-dan-dau-2486586.html, https://vietnamnet.vn/dai-hoc-nga-dua-mang-luong-tu-dai-hon-17-km-vao-nghien-cuu-va-dao-tao-2569958.html`
- **Lý do trượt (Mệnh đề định nghĩa chung chung về công nghệ):** Câu định nghĩa ChatGPT thời điểm đầu năm 2023 (`"dữ liệu từ năm 2021 trở về trước"`) mang tính mô tả phổ biến, hiện đã bị các bài tin mới (2024–2026) về ChatGPT trên báo chính thống đẩy xuống sâu.

