import os
import shutil
import random

src_img_dir = "Balanced_Dataset/train/images"
kappa_dir = "Kappa_Test/images"
os.makedirs(kappa_dir, exist_ok=True)

# Lấy ngẫu nhiên 100 ảnh
all_imgs = [f for f in os.listdir(src_img_dir) if f.endswith(('.jpg', '.png'))]
random.seed(99) # Cố định seed
sample_imgs = random.sample(all_imgs, 100)

for img in sample_imgs:
    shutil.copy(os.path.join(src_img_dir, img), os.path.join(kappa_dir, img))

print(f"Đã copy {len(sample_imgs)} ảnh ra thư mục '{kappa_dir}' để chuẩn bị gán nhãn.")