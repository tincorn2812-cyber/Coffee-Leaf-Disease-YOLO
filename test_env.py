from ultralytics import YOLO

def main():
    print("--- KIỂM TRA MÔI TRƯỜNG YOLOV11 ---")
    
    model = YOLO("yolo11n.pt")
    
    model.info()

    # Chạy train thử nghiệm 1 epoch với tập dữ liệu coco8
    print("\n--- BẮT ĐẦU TRAIN THỬ NGHIỆM ---")
    results = model.train(
        data="coco8.yaml",
        epochs=1,          # Chỉ chạy 1 epoch để test môi trường
        imgsz=640,         # Kích thước ảnh chuẩn 640x640 như yêu cầu đề tài
        batch=4,
        device="cpu"       # Đổi thành device="0" nếu máy bạn có GPU NVIDIA
    )
    
    print("\n--- HOÀN TẤT KIỂM TRA MÔI TRƯỜNG ---")

if __name__ == "__main__":
    main()