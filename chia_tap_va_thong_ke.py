import os
from collections import Counter

# Chỉ cần đọc thư mục Balanced_Dataset
balanced_dir = "Balanced_Dataset"
class_names = {0: 'gỉ sắt', 1: 'đốm mắt cua', 2: 'sâu đục lá', 3: 'nhện đỏ', 4: 'lá khoẻ mạnh'}

print("\n" + "="*50)
print("BẢNG PHÂN BỐ BOUNDING BOX TRONG BALANCED_DATASET")
print("="*50)
print(f"{'Lớp':<15} | {'Train (80%)':<12} | {'Val (10%)':<10} | {'Test (10%)':<10} | {'Tổng cộng'}")
print("-" * 50)

splits = ['train', 'val', 'test']
total_counts = {cls: 0 for cls in range(5)}
total_images = {'train': 0, 'val': 0, 'test': 0}

for cls_id in class_names.keys():
    row_counts = []
    cls_total = 0
    for split in splits:
        lbl_dir = os.path.join(balanced_dir, "labels", split)
        count = 0
        if os.path.exists(lbl_dir):
            for file in os.listdir(lbl_dir):
                if file.endswith(".txt"):
                    with open(os.path.join(lbl_dir, file), 'r') as f:
                        for line in f:
                            if line.startswith(f"{cls_id} "):
                                count += 1
                                
        row_counts.append(count)
        cls_total += count
        total_counts[cls_id] += cls_total

    print(f"{class_names[cls_id]:<15} | {row_counts[0]:<12} | {row_counts[1]:<10} | {row_counts[2]:<10} | {cls_total}")

print("="*50)

# Đếm tổng số ảnh
for split in splits:
    img_dir = os.path.join(balanced_dir, "images", split)
    if os.path.exists(img_dir):
         total_images[split] = len(os.listdir(img_dir))

print(f"Tổng số ảnh: Train={total_images['train']}, Val={total_images['val']}, Test={total_images['test']}")
print(f"Dữ liệu sẵn sàng huấn luyện nằm trong folder '{balanced_dir}'.\n")