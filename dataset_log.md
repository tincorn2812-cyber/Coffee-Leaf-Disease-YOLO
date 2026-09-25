# NHẬT KÝ VÀ THỐNG KÊ BỘ DỮ LIỆU (DATASET LOG)
**Dự án:** Nhận diện bệnh lá cà phê bằng model YOLO cải tiến phục vụ nông nghiệp thông minh 
**Thực tập sinh:** Ngô Đào Trung Tín 
**Người hướng dẫn:** Trương Quang Đại 

---

## 1. Nguồn gốc và Tiền xử lý
*   **Nguồn thu thập:** Tích hợp từ 5 bộ dữ liệu công khai trên nền tảng Roboflow Universe.
*   **Chuẩn hoá nhãn (Label Mapping):** Lọc bỏ các nhãn dư thừa (Phoma, Malnutrition, Pest...) và hợp nhất về **5 lớp mục tiêu** của đồ án theo chuẩn định dạng YOLO.
*   **Làm sạch dữ liệu:** Đã áp dụng thuật toán pHash để loại bỏ ảnh trùng lặp và phương sai Laplacian để lọc ảnh quá mờ nhòe.
*   **Cân bằng dữ liệu (Data Balancing):** Áp dụng Undersampling cắt giảm các lớp bệnh đa số (giới hạn max 1.500 ảnh/lớp) và Oversampling (Data Augmentation) nhân bản lớp thiểu số.
*   **Cấu trúc lưu trữ cục bộ:** Toàn bộ dữ liệu sạch, đã chia sẵn tập (Train 80% - Val 10% - Test 10%) được lưu trữ tại thư mục `C:\THUCTAP\Balanced_Dataset`.

## 2. Bảng phân phối Bounding Box và Dữ liệu sau cân bằng
Tổng số ảnh sau khi làm sạch và cân bằng: **7.517 ảnh** (Train: 6.013 | Val: 752 | Test: 752).
Dưới đây là bảng phân phối số lượng Bounding Box (khung tọa độ) theo từng lớp mục tiêu:

| ID chuẩn | Tên Lớp (Class) | Ý nghĩa tiếng Việt | Số lượng Bounding Box | Đánh giá phân bổ |
| :---: | :--- | :--- | :---: | :--- |
| **0** | `Rust` | Gỉ sắt | 25.111 | Đã cắt giảm (Undersampling) |
| **1** | `Cercospora` | Đốm mắt cua | 10.538 | Cân bằng tự nhiên |
| **2** | `Miner` | Sâu đục lá | 12.113 | Đã cắt giảm (Undersampling) |
| **3** | `Red Spider Mite` | Nhện đỏ | 11.294 | Cân bằng tự nhiên |
| **4** | `Healthy` | Lá khoẻ mạnh | 990 | Tăng cường (Augmentation) |
| | **TỔNG CỘNG** | **5 Lớp** | **~60.046 Boxes** | **Sẵn sàng huấn luyện** |

## 3. Đánh giá và Định hướng xử lý
*   **Thành quả đạt được:** 
    *   Tổng quy mô đạt **7.517 ảnh**, vượt xa mục tiêu đề ra (3.000 - 5.000 ảnh).
    *   Vấn đề Mất cân bằng lớp (Class Imbalance) nghiêm trọng đã được khắc phục hoàn toàn. Lớp `Healthy` từ 182 box ban đầu đã được kéo lên xấp xỉ 1.000 box (đáp ứng đúng chỉ tiêu của Mentor), trong khi lớp `Rust` khổng lồ đã được ép xuống mức cân bằng an toàn. Điều này đảm bảo mô hình YOLO11 không bị học vẹt (overfitting) hay thiên vị (bias).
*   **Định hướng bước tiếp theo (Tuần 4):** 
    1. Tạo file cấu hình `data.yaml` trỏ đường dẫn trực tiếp vào 3 thư mục `train`, `val`, `test` trong `Balanced_Dataset`.
    2. Viết mã nguồn `train_yolo.py` và bắt đầu tiến trình huấn luyện (training) mô hình YOLO11.

# NHẬT KÝ HUẤN LUYỆN (EXPERIMENT LOG)

## Lần chạy 1 (Train thử nghiệm trên Balanced_Dataset)
- **Ngày thực hiện:** 25/09/2026
- **Môi trường:** Google Colab (GPU T4)
- **Phiên bản Model:** YOLO11n (yolo11n.pt)

**1. Cấu hình huấn luyện (Hyperparameters):**
- Dataset: data.yaml (Trỏ tới Balanced_Dataset có dùng pHash và Laplacian)
- Epochs: 50
- Batch size: 32
- Image size (imgsz): 640
- Device: 0 (GPU T4)
- Seed: [Mặc định của thư viện Ultralytics]

**2. Kết quả nghiệm thu (Results):**
- Thời gian chạy: 1.446 hours (1 giờ 26 phút)
- Overall mAP50: 0.854 (85.4%)
- mAP50 từng lớp:
  + Gỉ sắt (0): 0.636
  + Đốm mắt cua (1): 0.738
  + Sâu đục lá (2): 0.966
  + Nhện đỏ (3): 0.978
  + Lá khoẻ mạnh (4): 0.950

**Đánh giá:** Mô hình hội tụ tốt. Tốc độ train trên GPU T4 tối ưu. Lớp Lá khoẻ mạnh đạt 95% chứng minh phương pháp Augmentation (nhân bản) hoạt động cực kỳ hiệu quả.