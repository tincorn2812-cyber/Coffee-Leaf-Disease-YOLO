import os
import shutil
from PIL import Image
import imagehash

# Cấu hình thư mục
src_img_dir = "Master_Dataset/images"
src_lbl_dir = "Master_Dataset/labels"

clean_img_dir = "Cleaned_Dataset/images"
clean_lbl_dir = "Cleaned_Dataset/labels"

os.makedirs(clean_img_dir, exist_ok=True)
os.makedirs(clean_lbl_dir, exist_ok=True)

seen_hashes = set()
duplicate_count = 0
valid_count = 0

print("Đang quét pHash để khử trùng lặp, vui lòng đợi...")

# Quét toàn bộ ảnh trong Master_Dataset
for img_name in os.listdir(src_img_dir):
    img_path = os.path.join(src_img_dir, img_name)
    lbl_name = os.path.splitext(img_name)[0] + ".txt"
    lbl_path = os.path.join(src_lbl_dir, lbl_name)
    
    try:
        # Mở ảnh và tính pHash
        with Image.open(img_path) as img:
            img_hash = imagehash.phash(img)
            
        if img_hash in seen_hashes:
            # Nếu hash đã tồn tại -> Đây là ảnh trùng lặp
            duplicate_count += 1
        else:
            # Nếu hash mới -> Lưu vào set và copy sang folder Cleaned
            seen_hashes.add(img_hash)
            
            # Copy ảnh
            shutil.copy(img_path, os.path.join(clean_img_dir, img_name))
            # Copy nhãn nếu có
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, os.path.join(clean_lbl_dir, lbl_name))
            
            valid_count += 1
            
    except Exception as e:
        print(f"Lỗi khi đọc file {img_name}: {e}")

print("-" * 40)
print("BÁO CÁO KHỬ TRÙNG LẶP BẰNG pHASH:")
print(f"- Tổng số ảnh ban đầu: {duplicate_count + valid_count}")
print(f"- Số ảnh bị trùng lặp đã loại bỏ: {duplicate_count}")
print(f"- Số ảnh sạch giữ lại: {valid_count}")
print("Toàn bộ dữ liệu sạch đã được lưu vào 'Cleaned_Dataset'.")