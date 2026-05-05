import os
import pandas as pd
import json
import shutil
import random
import unicodedata

# --- CẤU HÌNH ---
CSV_PATH = r"d:\data\HUIT\Nam3\HK2\Deep Learning\TaiLieuThayBao\Project\Donut\Version2\data1\mcocr_train_df.csv"
SRC_IMG_DIR = r"d:\data\HUIT\Nam3\HK2\Deep Learning\TaiLieuThayBao\Project\Donut\Version2\data1\train_images\train_images"
OUTPUT_BASE = r"d:\data\HUIT\Nam3\HK2\Deep Learning\TaiLieuThayBao\Project\Donut\Version2\Train_data"

def normalize_text(text):
    if not text: return ""
    text = unicodedata.normalize('NFC', str(text))
    return " ".join(text.split())

def extract_info_from_anno(row):
    """Bóc tách text từ anno_texts và anno_labels dùng dấu phân cách |||"""
    try:
        # Cắt chuỗi theo dấu |||
        texts = str(row['anno_texts']).split('|||')
        labels = str(row['anno_labels']).split('|||')
        
        info = {"seller": [], "address": [], "timestamp": [], "total_cost": []}
        
        # Ánh xạ nhãn chữ hoa sang nhãn chúng ta cần
        label_map = {
            "SELLER": "seller",
            "ADDRESS": "address",
            "TIMESTAMP": "timestamp",
            "TOTAL_COST": "total_cost"
        }
        
        for t, l in zip(texts, labels):
            l_upper = l.strip().upper()
            if l_upper in label_map:
                info[label_map[l_upper]].append(normalize_text(t))
        
        # Gộp các đoạn text lại
        return {k: " ".join(v) if v else "N/A" for k, v in info.items()}
    except Exception as e:
        return {"seller": "N/A", "address": "N/A", "timestamp": "N/A", "total_cost": "N/A"}

def prepare_dataset():
    if not os.path.exists(CSV_PATH):
        print(f"❌ LỖI: Không tìm thấy file CSV tại {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    data = []
    
    print(f"Đang xử lý {len(df)} dòng dữ liệu...")
    
    count_found = 0
    count_missing_img = 0

    for _, row in df.iterrows():
        img_name = row['img_id']
        img_path = os.path.join(SRC_IMG_DIR, img_name)
        
        if os.path.exists(img_path):
            gt_dict = extract_info_from_anno(row)
            
            # Chỉ lấy những mẫu có ít nhất 1 thông tin thực tế khác N/A
            if any(v != "N/A" for v in gt_dict.values()):
                data.append({
                    "file_name": img_name,
                    "ground_truth": json.dumps({"gt_parse": gt_dict}, ensure_ascii=False)
                })
                count_found += 1
        else:
            count_missing_img += 1

    if not data:
        print(f"❌ LỖI: Không trích xuất được dữ liệu.")
        print(f" - Số ảnh thiếu: {count_missing_img}")
        print(f" - Đường dẫn kiểm tra: {SRC_IMG_DIR}")
        return

    # Chia dữ liệu 90/10
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Lưu và copy file
    for mode, dataset in [("train", train_data), ("val", val_data)]:
        mode_dir = os.path.join(OUTPUT_BASE, mode)
        os.makedirs(mode_dir, exist_ok=True)
        metadata_path = os.path.join(OUTPUT_BASE, f"{mode}_metadata.jsonl")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                # Copy ảnh
                shutil.copy(os.path.join(SRC_IMG_DIR, item["file_name"]), os.path.join(mode_dir, item["file_name"]))
                # Ghi metadata
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ Hoàn thành!")
    print(f" - Tổng mẫu hợp lệ: {count_found}")
    print(f" - Số ảnh không tìm thấy: {count_missing_img}")
    print(f" - Metadata được lưu tại: {OUTPUT_BASE}")

if __name__ == "__main__":
    prepare_dataset()