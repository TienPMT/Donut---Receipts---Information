import os
import json
from PIL import Image

REAL_IMG_DIR = r"D:\data\HUIT\Nam3\HK2\Deep Learning\TaiLieuThayBao\Project\Donut\Version2\Train_data\real"
OUTPUT_JSONL = r"D:\data\HUIT\Nam3\HK2\Deep Learning\TaiLieuThayBao\Project\Donut\Version2\Train_data\real_metadata.jsonl"

def create_label(file_name):
    print(f"\n--- ĐANG GÁN NHÃN CHO ẢNH: {file_name} ---")
    print("Nhập thông tin (để trống và Enter nếu không có/không thấy):")

    seller_name = input("1. Tên cửa hàng (seller_name): ").strip() or "N/A"
    address = input("2. Địa chỉ (address): ").strip() or "N/A"
    timestamp = input("3. Thời gian/Ngày tháng (timestamp): ").strip() or "N/A"
    total_cost = input("4. Tổng tiền (total_cost): ").strip() or "N/A"

    label_dict = {
        "seller": {
            "seller_name": seller_name,
            "address": address,
            "timestamp": timestamp,
            "total_cost": total_cost
        }
    }
    return json.dumps(label_dict, ensure_ascii=False)

def main():
    if not os.path.exists(REAL_IMG_DIR):
        print(f"Lỗi: Không tìm thấy thư mục {REAL_IMG_DIR}")
        return

    image_files = [f for f in os.listdir(REAL_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("Không tìm thấy ảnh nào trong thư mục real.")
        return

    print(f"Tìm thấy {len(image_files)} ảnh. Bắt đầu gán nhãn manual...")

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for file_name in image_files:

            img_path = os.path.join(REAL_IMG_DIR, file_name)
            os.startfile(img_path)

            ground_truth = create_label(file_name)

            line = {
                "file_name": file_name,
                "ground_truth": ground_truth
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            print(f"Đã lưu: {file_name}")

    print(f"\nHOÀN THÀNH! File metadata đã được lưu tại: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
