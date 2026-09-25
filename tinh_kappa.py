import os
import glob

# 1. Cấu hình đường dẫn
original_lbl_dir = "YOLO_Dataset/train/labels"
my_lbl_dir = "my_labels"

# 2. Hàm tính độ chồng lấp (Intersection over Union - IoU)
def bbox_iou(box1, box2):
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area)

# 3. Hàm tính Cohen's Kappa thuần Python (Bỏ qua sklearn)
def calculate_cohens_kappa(y1, y2):
    if len(y1) != len(y2) or len(y1) == 0:
        return 0.0
    
    classes = set(y1).union(set(y2))
    matrix = {c: {c_other: 0 for c_other in classes} for c in classes}
    
    for a, b in zip(y1, y2):
        matrix[a][b] += 1
        
    N = len(y1)
    po = sum(matrix[c][c] for c in classes) / N
    
    pe = 0
    for c in classes:
        sum_row = sum(matrix[c].values())
        sum_col = sum(matrix[r][c] for r in classes)
        pe += (sum_row * sum_col)
    pe /= (N * N)
    
    if pe == 1: return 1.0
    return (po - pe) / (1 - pe)

# 4. Đọc file nhãn
def read_labels(filepath):
    boxes = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    boxes.append((cls_id, coords))
    return boxes

# 5. So sánh và thu thập danh sách nhãn khớp nhau
y_original = []
y_my_label = []

my_txt_files = glob.glob(os.path.join(my_lbl_dir, "*.txt"))
matched_boxes = 0
iou_threshold = 0.45

for my_txt_path in my_txt_files:
    filename = os.path.basename(my_txt_path)
    orig_txt_path = os.path.join(original_lbl_dir, filename)
    
    my_boxes = read_labels(my_txt_path)
    orig_boxes = read_labels(orig_txt_path)
    
    for my_cls, my_coords in my_boxes:
        best_iou = 0
        best_orig_cls = -1
        
        for orig_cls, orig_coords in orig_boxes:
            iou = bbox_iou(my_coords, orig_coords)
            if iou > best_iou:
                best_iou = iou
                best_orig_cls = orig_cls
                
        if best_iou > iou_threshold:
            y_my_label.append(my_cls)
            y_original.append(best_orig_cls)
            matched_boxes += 1

# 6. Đánh giá kết quả
if matched_boxes > 0:
    kappa = calculate_cohens_kappa(y_original, y_my_label)
    print("-" * 50)
    print("KẾT QUẢ KIỂM CHỨNG TÍNH NHẤT QUÁN CỦA NHÃN:")
    print(f"- Số lượng bounding box khớp vị trí để đối chiếu: {matched_boxes}")
    print(f"- Hệ số Cohen's Kappa: {kappa:.4f}")
    
    if kappa > 0.8:
        print("-> Đánh giá: Rất xuất sắc (Almost perfect agreement). Dữ liệu hoàn toàn tin cậy!")
    elif kappa > 0.6:
        print("-> Đánh giá: Tốt (Substantial agreement). Dữ liệu đạt chuẩn để huấn luyện.")
    else:
        print("-> Đánh giá: Cần xem xét lại. Có sự sai lệch lớn giữa 2 vòng gán nhãn.")
    print("-" * 50)
else:
    print("Lỗi: Không tìm thấy bounding box nào khớp nhau. Hãy kiểm tra lại thư mục Kappa_Test/my_labels!")