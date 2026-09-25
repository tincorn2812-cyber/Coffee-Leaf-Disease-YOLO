import os
import shutil
import glob

# 1. KHỞI TẠO MASTER_DATASET MỚI
master_img_dir = "Master_Dataset/images"
master_lbl_dir = "Master_Dataset/labels"

if os.path.exists("Master_Dataset"):
    shutil.rmtree("Master_Dataset")

os.makedirs(master_img_dir, exist_ok=True)
os.makedirs(master_lbl_dir, exist_ok=True)

# 2. KHAI BÁO CÁC DẤU HIỆU NHẬN BIẾT BỘ DỮ LIỆU THÔNG QUA FILE DATA.YAML
rules = [
    {
        # Bộ 1: coffee-leaf-disease-jtif1
        'names': "['brown eye spot', 'healthy', 'leaf minor', 'leaf rust', 'red spider mite']",
        'mapping': {0: 1, 1: 4, 2: 2, 3: 0, 4: 3}
    },
    {
        # Bộ 2: coffee-leaves-detection
        'names': "['Miner', 'Rust']",
        'mapping': {0: 2, 1: 0}
    },
    {
        # Bộ 3: detection-diesease-coffee-leaf
        'names': "['Cercospora', 'Miner', 'Phoma', 'Rust']",
        'mapping': {0: 1, 1: 2, 3: 0}
    },
    {
        # Bộ 4: barakobama-coffee-leaf-disease
        'names': "['algal_growth', 'cercospora', 'leaf_miner', 'leaf_rust', 'red_spider_mite', 'sooty-mold']",
        'mapping': {1: 1, 2: 2, 3: 0, 4: 3}
    },
    {
        # Bộ 5: coffee-leaf-1jk2g (version 7) - Bổ sung cứu cánh
        'names': "['Leaf Rust Spots', 'Leaf-Rust-Severe', 'Malnutrition', 'Old Coffee Leaf', 'Pest', 'healthy']",
        'mapping': {5: 4} # Chỉ lấy healthy (ID gốc là 5) chuyển thành ID 4 chuẩn
    }
]

print("Đang quét tự động các thư mục chứa dữ liệu...")
total_images = 0

# Tìm tự động tất cả các file data.yaml trong thư mục làm việc
yaml_files = glob.glob("*/data.yaml")

for yaml_path in yaml_files:
    dataset_dir = os.path.dirname(yaml_path)
    if "Master_Dataset" in dataset_dir:
        continue
        
    # Đọc nội dung data.yaml
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Xác định đúng bộ dữ liệu và bản đồ ánh xạ
    current_mapping = None
    for rule in rules:
        if rule['names'] in content:
            current_mapping = rule['mapping']
            break
            
    if current_mapping is None:
        continue
        
    print(f"\nNhận diện thành công: {dataset_dir}")
    print("Đang lọc nhãn và gộp ảnh...")
    
    label_files = glob.glob(f"{dataset_dir}/**/labels/*.txt", recursive=True)
    count = 0
    
    for lbl_path in label_files:
        img_path = lbl_path.replace("labels", "images").replace(".txt", ".jpg")
        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg", ".png")
            if not os.path.exists(img_path):
                img_path = img_path.replace(".png", ".jpeg")
                
        if not os.path.exists(img_path):
            continue
            
        base_name = os.path.basename(lbl_path)
        new_name = f"{dataset_dir.replace(' ', '_')}_{base_name}"
        new_lbl_path = os.path.join(master_lbl_dir, new_name)
        new_img_path = os.path.join(master_img_dir, new_name.replace(".txt", os.path.splitext(img_path)[1]))

        valid_labels = []
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_id = int(parts[0])
                    if old_id in current_mapping:
                        new_id = current_mapping[old_id]
                        valid_labels.append(f"{new_id} {' '.join(parts[1:])}\n")
        
        if valid_labels:
            with open(new_lbl_path, 'w') as f:
                f.writelines(valid_labels)
            shutil.copy(img_path, new_img_path)
            count += 1
            total_images += 1
            
    print(f"-> Đã lấy {count} ảnh hợp lệ.")

print("\n" + "-" * 50)
print(f"GỘP XONG! Toàn bộ dữ liệu sạch nằm trong folder 'Master_Dataset'.")
print(f"Tổng số ảnh thu được từ 4 bộ: {total_images} ảnh.")