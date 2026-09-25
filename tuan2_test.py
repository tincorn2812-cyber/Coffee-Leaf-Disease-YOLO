from ultralytics import YOLO
import time

def check_model_info():
    print("\n" + "="*60)
    print("PHẦN 1: CHẠY MODEL.INFO() TRÊN 5 WEIGHTS (N/S/M/L/X)")
    print("="*60)
    models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
    
    for m in models:
        print(f"\n--- Đang tải mô hình: {m} ---")
        model = YOLO(m)
        model.info()

def measure_latency(model_name, image_source):
    print(f"\n--- ĐANG ĐO LATENCY MÔ HÌNH: {model_name.upper()} ---")
    model = YOLO(model_name)
    
    # 1. Warm-up 10 lần 
    print(f"[*] Đang chạy Warm-up 10 lần cho {model_name}...")
    for _ in range(10):
        model.predict(image_source, device="cpu", verbose=False)
    print("[*] Hoàn tất Warm-up. Bắt đầu đo chính thức...")
    
    # 2. Đo 50 lần liên tục để lấy số liệu trung bình chính xác nhất
    num_tests = 50
    start_time = time.time()
    
    for _ in range(num_tests):
        model.predict(image_source, device="cpu", verbose=False)
        
    end_time = time.time()
    
    # 3. Tính toán kết quả
    avg_latency = ((end_time - start_time) / num_tests) * 1000
    fps = 1000 / avg_latency
    
    print(f">> KẾT QUẢ TRÊN CPU: Latency = {avg_latency:.2f} ms/ảnh | FPS = {fps:.2f} <<")

def main():
    # Thực thi Yêu cầu 1
    check_model_info()
    
    # Thực thi Yêu cầu 2
    print("\n" + "="*60)
    print("PHẦN 2: ĐO LATENCY THỰC TẾ (N/S/M) CÓ WARM-UP 10 LẦN")
    print("="*60)
    test_image = "https://ultralytics.com/images/bus.jpg" 
    models_to_test = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
    
    for m in models_to_test:
        measure_latency(m, test_image)

if __name__ == "__main__":
    main()