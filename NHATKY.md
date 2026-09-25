# NHẬT KÝ THỰC TẬP

**Dự án:** Nhận diện bệnh lá cà phê bằng model YOLO cải tiến, phục vụ nông nghiệp thông minh
**Thực tập sinh:** Ngô Đào Trung Tín
**Người hướng dẫn:** Trương Quang Đại

---

## TUẦN 1: DỰNG NỀN TẢNG (09/09/2026 - 11/09/2026)
**Mục tiêu:** Tìm hiểu bài toán, đặc điểm các loại bệnh và thiết lập môi trường huấn luyện thành công.

**Ngày 09/09/2026 (T4)**
*   **Công việc thực hiện:**
    *   Nghiên cứu bài toán phát hiện đối tượng áp dụng trong nông nghiệp thông minh.
    *   Tìm hiểu nguyên nhân, triệu chứng và dấu hiệu nhận biết bằng mắt thường của 6 lớp mục tiêu: gỉ sắt (leaf rust), đốm mắt cua (cercospora), sâu đục lá (leaf miner), nhện đỏ (red spider mite) và lá khoẻ mạnh (healthy).
*   **Kết quả đầu ra:** Lập bảng tổng hợp đặc điểm nhận dạng chi tiết cho từng loại bệnh phục vụ cho công tác gán nhãn.

**Ngày 10/09/2026 (T5)**
*   **Công việc thực hiện:**
    *   Thiết lập môi trường phát triển trên máy cá nhân: Cài đặt Python, Git, VS Code.
    *   Cài đặt các thư viện Deep Learning và Computer Vision: ultralytics, opencv-python, matplotlib.
*   **Vướng mắc & Xử lý:** Khắc phục lỗi đường dẫn pip trên Windows bằng cách gọi module trực tiếp (`python -m pip install`), thiết lập thành công môi trường.

**Ngày 11/09/2026 (T6)**
*   **Công việc thực hiện:**
    *   Viết script `test_env.py` kiểm tra độ ổn định của môi trường.
    *   Khởi tạo mô hình YOLO11n và chạy thử nghiệm huấn luyện 1 epoch trên bộ dữ liệu `coco8.yaml`.
*   **Kết quả đầu ra (Nghiệm thu Tuần 1):** Log Terminal ghi nhận CPU hoạt động ổn định, xuất file weights `best.pt` thành công.

---

## TUẦN 2: THỰC NGHIỆM VÀ ĐO ĐẠC MỐC M1 (15/09/2026 - 18/09/2026)
**Mục tiêu:** Hoàn thiện báo cáo lý thuyết, quản lý mã nguồn Git, thiết lập đám mây và đo đạc thông số thực tế.

**Ngày 15/09/2026 (T3)**
*   **Công việc thực hiện:**
    *   Quản lý mã nguồn: Khởi tạo Git Repository cục bộ tại `C:\THUCTAP`, cấu hình file `.gitignore` chuyên dụng cho Deep Learning (loại bỏ tệp trọng số `.pt` và thư mục log `runs/`), thực hiện commit và đẩy (push) thành công source code lên kho lưu trữ GitHub cá nhân.
    *   Hạ tầng đám mây: Thiết lập tài khoản Google Colab, kiểm tra và xác thực khả năng cấp phát tài nguyên GPU T4 thành công cho các tác vụ huấn luyện nặng ở các tuần sau.
    *   Nghiên cứu lý thuyết: Hoàn thiện Báo cáo nghiên cứu phân biệt các phiên bản model YOLO (N/S/M/L/X) phiên bản 2 theo đúng các góp ý kỹ thuật.
    *   Chuẩn bị thực nghiệm: Xây dựng script `tuan2_test.py` tích hợp hàm gọi `model.info()` cho 5 weights pretrained và kịch bản đo Latency/FPS có cơ chế warm-up 10 lần.
*   **Kết quả đầu ra:** Hoàn tất cấu trúc mã nguồn trên GitHub và sẵn sàng tiến hành trích xuất số liệu thực nghiệm cho ngày tiếp theo.

**Ngày 16/09/2026 (T4)**
*   **Công việc thực hiện:**
    *   Thực thi script `tuan2_test.py` trên thiết bị cá nhân để gọi hàm `model.info()` cho toàn bộ 5 phiên bản trọng số pretrained của YOLO11 (`yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`).
    *   Đối chiếu trực tiếp số liệu thực tế về số lượng tham số (parameters) và khối lượng tính toán (FLOPs) thu được từ Terminal với bảng thông số lý thuyết chuẩn của Ultralytics.
*   **Kết quả đầu ra:** Xác thực thành công tính chính xác của kiến trúc mô hình trên phần cứng cục bộ, đảm bảo số liệu khớp hoàn toàn với tài liệu nghiên cứu.

**Ngày 17/09/2026 (T5)**
*   **Công việc thực hiện:**
    *   Triển khai kịch bản đo đạc tốc độ suy luận (Inference Speed) thực tế trên CPU (Intel Core i5-12450HX) cho 3 biến thể chủ lực: Nano, Small và Medium.
    *   Áp dụng nghiêm ngặt quy trình kỹ thuật: thực hiện khởi động (warm-up) 10 lần trước khi tiến hành đo chính thức 50 vòng lặp liên tục để lấy giá trị trung bình.
*   **Kết quả đầu ra:** Thu thập đầy đủ các chỉ số Latency (ms/ảnh) và tốc độ khung hình (FPS) cho từng phiên bản mô hình.

**Ngày 18/09/2026 (T6) (Nghiệm thu Mốc M1)**
*   **Công việc thực hiện:**
    *   Tổng hợp toàn bộ phần lý thuyết nghiên cứu phiên bản YOLO và bảng số liệu đo đạc Latency/FPS thực tế vào báo cáo Word hoàn chỉnh.
    *   Cập nhật đầy đủ các tiến độ công việc và trạng thái task hoàn thành lên hệ thống quản lý dự án (Google Sheet của nhóm).
*   **Kết quả đầu ra (Nghiệm thu M1):** Hoàn thành và bàn giao hồ sơ báo cáo Mốc 1 (M1) cho người hướng dẫn đúng hạn, đạt yêu cầu kỹ thuật đề ra.

---

## TUẦN 3: THU THẬP VÀ GỘP DỮ LIỆU (22/09/2026 - 25/09/2026)
**Mục tiêu:** Xây dựng bộ dữ liệu thô quy mô 3.000 – 5.000 ảnh từ các nguồn công khai, đối chiếu hệ nhãn, đảm bảo bao quát đủ 5 lớp chuẩn và giải quyết bài toán thiếu hụt ảnh lá khoẻ mạnh.

**Ngày 22/09/2026 (T3)**
*   **Công việc thực hiện:**
    *   Nghiên cứu cấu trúc và tìm kiếm các bộ dữ liệu bệnh lá cà phê công khai trên nền tảng Roboflow Universe và Kaggle.
    *   Tiến hành tải 5 bộ dữ liệu dạng YOLO11 (.txt labels), rà soát và đối chiếu hệ nhãn gốc của các bộ dữ liệu tải về để lên phương án đồng bộ hoá ID nhãn.
*   **Kết quả đầu ra:** Thu thập thành công kho ảnh gốc đa dạng về điều kiện môi trường chụp, chuẩn bị cho bước làm sạch hệ nhãn.

**Ngày 23/09/2026 (T4)**
*   **Công việc thực hiện:**
    *   Bổ sung và phân loại ảnh mục tiêu. Quyết định thu hẹp phạm vi nhận diện xuống chuẩn 5 lớp cốt lõi dựa trên thực tế dữ liệu khả dụng: Gỉ sắt (0), Đốm mắt cua (1), Sâu đục lá (2), Nhện đỏ (3), Lá khoẻ mạnh (4).
    *   Phát hiện sự thiếu hụt trầm trọng của lớp "Lá khoẻ mạnh" trong các dataset công khai. Triển khai script `can_bang_du_lieu.py` ứng dụng thư viện Albumentations để tăng cường dữ liệu (lật ngang, dọc) bù đắp số lượng.
*   **Kết quả đầu ra:** Đảm bảo thu thập đủ ảnh lá khoẻ mạnh, kích số lượng Bounding Box của lớp này lên mức an toàn (~990 box) theo đúng lưu ý của Mentor.

**Ngày 24/09/2026 (T5)**
*   **Công việc thực hiện:**
    *   Xây dựng Dataset thô; lập biên bản nhật ký ghi nguồn và hệ nhãn gốc.
    *   Viết kịch bản tự động hóa bằng Python (`gop_du_lieu.py`) thực hiện ánh xạ (Label Mapping) để quét toàn bộ dữ liệu thô, loại bỏ các nhãn dư thừa (Pest, Phoma, Malnutrition) và tự động quy đổi ID nhãn về chuẩn 5 lớp của đề tài.
*   **Kết quả đầu ra:** Hợp nhất toàn bộ ảnh và nhãn vào thư mục `Master_Dataset` thành công. Biên soạn file `dataset_log.md` ghi nhận rõ nguồn gốc và quy tắc xử lý nhãn gốc của 5 bộ.

**Ngày 25/09/2026 (T6) (Nghiệm thu Tuần 3)**
*   **Công việc thực hiện:**
    *   Tổng duyệt số lượng ảnh thô, rà soát lại cơ cấu 5 lớp chuẩn bị cho tiền xử lý.
    *   Kiểm kê khối lượng, cấu trúc thư mục và xác nhận tính toàn vẹn của dữ liệu trước khi chuyển giao sang các kỹ thuật làm sạch sâu của tuần kế tiếp.
*   **Kết quả đầu ra (Nghiệm thu Tuần 3):** Bàn giao thành công bộ Dataset thô quy mô 7.517 ảnh (vượt chỉ tiêu ban đầu), hệ nhãn đã được đồng bộ 100% về 5 lớp, sẵn sàng bước vào khâu khử trùng lặp (pHash) và lọc ảnh mờ (Laplacian) ở Tuần 4.
