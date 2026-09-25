import cv2
import os
import shutil

# Cấu hình thư mục
src_img_dir = "Cleaned_Dataset/images"
src_lbl_dir = "Cleaned_Dataset/labels"

final_img_dir = "Final_Dataset/images"
final_lbl_dir = "Final_Dataset/labels"
blurry_dir = "Blurry_Images" # Lưu riêng ảnh mờ để làm minh chứng báo cáo

os.makedirs(final_img_dir, exist_ok=True)
os.makedirs(final_lbl_dir, exist_ok=True)
os.makedirs(blurry_dir, exist_ok=True)

# Ngưỡng độ nét (thường để 100, dưới 100 coi là mờ)
threshold = 100.0 
blurry_count = 0
sharp_count = 0

print(f"Đang quét Laplacian variance với ngưỡng {threshold}...")

for img_name in os.listdir(src_img_dir):
    img_path = os.path.join(src_img_dir, img_name)
    lbl_name = os.path.splitext(img_name)[0] + ".txt"
    lbl_path = os.path.join(src_lbl_dir, lbl_name)

    # Đọc ảnh bằng OpenCV
    image = cv2.imread(img_path)
    if image is None:
        continue

    # Chuyển sang ảnh xám và tính Laplacian variance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Phân loại
    if fm < threshold:
        blurry_count += 1
        shutil.copy(img_path, os.path.join(blurry_dir, img_name))
    else:
        sharp_count += 1
        shutil.copy(img_path, os.path.join(final_img_dir, img_name))
        # Chỉ copy nhãn nếu ảnh đạt chuẩn nét
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(final_lbl_dir, lbl_name))

print("-" * 40)
print("BÁO CÁO LỌC ẢNH MỜ BẰNG LAPLACIAN VARIANCE:")
print(f"- Tổng số ảnh đưa vào: {blurry_count + sharp_count}")
print(f"- Số ảnh mờ đã loại bỏ: {blurry_count}")
print(f"- Số ảnh sắc nét giữ lại: {sharp_count}")
print("Dữ liệu SẠCH HOÀN TOÀN nằm trong folder 'Final_Dataset'.")