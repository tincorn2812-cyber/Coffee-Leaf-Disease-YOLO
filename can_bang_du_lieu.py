import os
import glob
import random
import shutil
import cv2

# Nếu máy chưa có albumentations thì chạy cài đặt bằng lệnh trong terminal:
# pip install albumentations opencv-python
try:
    import albumentations as A
except ImportError:
    print("Vui lòng cài đặt thư viện albumentations trước khi chạy.")
    print("Lệnh: pip install albumentations opencv-python")
    exit()

# 1. CẤU HÌNH ĐƯỜNG DẪN & MỤC TIÊU
source_dir = "YOLO_Dataset"
balanced_dir = "Balanced_Dataset"
master_labels = "Master_Dataset/labels"
master_images = "Master_Dataset/images"

# Xóa và tạo mới thư mục Balanced_Dataset
if os.path.exists(balanced_dir):
    shutil.rmtree(balanced_dir)
os.makedirs(os.path.join(balanced_dir, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(balanced_dir, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(balanced_dir, "images", "test"), exist_ok=True)
os.makedirs(os.path.join(balanced_dir, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(balanced_dir, "labels", "val"), exist_ok=True)
os.makedirs(os.path.join(balanced_dir, "labels", "test"), exist_ok=True)

# 2. NGƯỠNG CÂN BẰNG
# Giới hạn số lượng ảnh MỖI LỚP được phép đưa vào tập huấn luyện (Giảm mẫu các lớp lớn)
MAX_IMAGES_PER_CLASS = 1500 

# 3. ĐỊNH NGHĨA AUGMENTATION (Tăng cường dữ liệu)
# Chỉ áp dụng lật ngang, lật dọc để giữ nguyên bản chất sinh học của lá
transform = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def process_and_balance():
    print("Đang phân tích dữ liệu gốc từ Master_Dataset...")
    label_files = glob.glob(os.path.join(master_labels, "*.txt"))
    
    # Phân loại ảnh theo lớp chiếm ưu thế nhất trong ảnh đó
    images_by_class = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    for lbl_path in label_files:
        filename = os.path.basename(lbl_path)
        img_path = os.path.join(master_images, filename.replace(".txt", ".jpg"))
        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg", ".png")
            if not os.path.exists(img_path):
                img_path = img_path.replace(".png", ".jpeg")
        
        if not os.path.exists(img_path):
            continue
            
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            if not lines: continue
            
            # Đếm số lượng nhãn trong ảnh
            class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in class_counts:
                        class_counts[cls_id] += 1
            
            # Gán ảnh vào lớp xuất hiện nhiều nhất (ưu tiên lớp 4 - healthy nếu có)
            if class_counts[4] > 0:
                images_by_class[4].append((img_path, lbl_path, lines))
            else:
                dominant_class = max(class_counts, key=class_counts.get)
                if class_counts[dominant_class] > 0:
                    images_by_class[dominant_class].append((img_path, lbl_path, lines))
                    
    selected_data = []
    
    # 4. UNDERSAMPLING (Giảm mẫu các lớp quá lớn)
    for cls in range(4): # Xử lý các lớp bệnh 0, 1, 2, 3
        random.shuffle(images_by_class[cls])
        # Chỉ lấy tối đa MAX_IMAGES_PER_CLASS ảnh cho mỗi lớp bệnh
        selected_data.extend(images_by_class[cls][:MAX_IMAGES_PER_CLASS])
        print(f"- Lớp bệnh {cls}: Chọn ngẫu nhiên {len(images_by_class[cls][:MAX_IMAGES_PER_CLASS])} ảnh từ {len(images_by_class[cls])} ảnh.")

    # 5. OVERSAMPLING (Tăng cường dữ liệu cho lớp Healthy - 4)
    healthy_data = images_by_class[4]
    print(f"- Lớp Healthy (4): Có {len(healthy_data)} ảnh gốc.")
    selected_data.extend(healthy_data) # Thêm bản gốc
    
    aug_count = 0
    # Nhân bản bằng Augmentation
    for img_p, lbl_p, lines in healthy_data:
        try:
            image = cv2.imread(img_p)
            if image is None: continue
            
            bboxes = []
            class_labels = []
            for line in lines:
                parts = line.strip().split()
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:5]]
                bboxes.append(coords)
                class_labels.append(cls_id)
                
            # Tạo ảnh augmentation
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']
            
            # Lưu ảnh mới (đặt tên có tiền tố aug_)
            base_name = os.path.basename(img_p)
            new_img_name = f"aug_{base_name}"
            new_lbl_name = f"aug_{os.path.basename(lbl_p)}"
            
            temp_img_path = os.path.join(master_images, new_img_name)
            temp_lbl_path = os.path.join(master_labels, new_lbl_name)
            
            cv2.imwrite(temp_img_path, transformed_image)
            
            with open(temp_lbl_path, 'w') as f:
                for cls, box in zip(transformed_class_labels, transformed_bboxes):
                    # Chỉ lưu định dạng YOLO với 6 số thập phân
                    f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
            
            # Thêm ảnh vừa nhân bản vào danh sách chọn lọc
            with open(temp_lbl_path, 'r') as f:
                new_lines = f.readlines()
            selected_data.append((temp_img_path, temp_lbl_path, new_lines))
            aug_count += 1
        except Exception as e:
            continue
            
    print(f"-> Đã tạo thêm {aug_count} ảnh Augmentation cho lớp Healthy.")
    
    return selected_data

def split_and_save(selected_data):
    # Chia theo tỷ lệ 80-10-10
    random.shuffle(selected_data)
    total = len(selected_data)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    splits = {
        'train': selected_data[:train_end],
        'val': selected_data[train_end:val_end],
        'test': selected_data[val_end:]
    }
    
    print(f"\nĐang chia tập và lưu vào Balanced_Dataset (Tổng {total} ảnh)...")
    for phase, items in splits.items():
        for img_path, lbl_path, _ in items:
            shutil.copy(img_path, os.path.join(balanced_dir, "images", phase, os.path.basename(img_path)))
            shutil.copy(lbl_path, os.path.join(balanced_dir, "labels", phase, os.path.basename(lbl_path)))
            
    print("HOÀN TẤT CÂN BẰNG DỮ LIỆU!")

if __name__ == '__main__':
    data = process_and_balance()
    split_and_save(data)