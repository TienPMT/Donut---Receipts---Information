import os

BASE_DIR = r"D:\data\HUIT\Nam3\HK2\Deep Learning\TaiLieuThayBao\Project\Donut\Version2\Train_data"
TRAIN_METADATA = os.path.join(BASE_DIR, "train_metadata.jsonl")
REAL_METADATA = os.path.join(BASE_DIR, "real_metadata.jsonl")
TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, "train")
REAL_IMAGES_DIR = os.path.join(BASE_DIR, "real")

def combine_data():
    print("--- ĐANG TRỘN DỮ LIỆU THỰC TẾ VÀO TẬP TRAIN ---")

    if not os.path.exists(REAL_METADATA):
        print(f"Lỗi: Không tìm thấy file {REAL_METADATA}")
        return

    with open(REAL_METADATA, 'r', encoding='utf-8') as f:
        real_lines = f.readlines()

    print(f"Tìm thấy {len(real_lines)} mẫu dữ liệu thực tế.")

    with open(TRAIN_METADATA, 'a', encoding='utf-8') as f:
        for line in real_lines:
            f.write(line)

    import shutil
    count_copy = 0
    for file_name in os.listdir(REAL_IMAGES_DIR):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(REAL_IMAGES_DIR, file_name)
            dst = os.path.join(TRAIN_IMAGES_DIR, file_name)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                count_copy += 1

    print(f"Đã copy {count_copy} ảnh thực tế vào thư mục {TRAIN_IMAGES_DIR}")
    print(f"Đã cập nhật xong file {TRAIN_METADATA}")

if __name__ == "__main__":
    combine_data()
