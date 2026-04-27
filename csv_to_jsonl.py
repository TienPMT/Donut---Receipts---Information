import pandas as pd
import json
import os
import unicodedata

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # Chuyển về chuẩn NFC
    text = unicodedata.normalize('NFC', text)
    # Xóa khoảng trắng thừa
    text = " ".join(text.split())
    return text

def csv_to_jsonl(csv_path, output_path):
    print(f"Đang xử lý: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file {csv_path}")
        return
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Lỗi đọc file {csv_path}: {e}")
        return
        
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            if 'anno_texts' not in row or 'anno_labels' not in row or 'img_id' not in row:
                continue
                
            texts_str = str(row['anno_texts']) if pd.notna(row['anno_texts']) else ""
            labels_str = str(row['anno_labels']) if pd.notna(row['anno_labels']) else ""
            
            texts = texts_str.split('|||')
            labels = labels_str.split('|||')

            seller_items = []
            address_items = []
            timestamp_items = []
            total_items = []
            
            for label, text in zip(labels, texts):
                label = label.strip()
                text = normalize_text(text) 
                
                if label == 'SELLER':
                    seller_items.append(text)
                elif label == 'ADDRESS':
                    address_items.append(text)
                elif label == 'TIMESTAMP':
                    timestamp_items.append(text)
                elif label == 'TOTAL_COST':
                    total_items.append(text)
            
            # TẠO CHUỖI GROUND TRUTH SỬ DỤNG SPECIAL TOKENS
            # Định dạng CHUẨN: <s_seller>Giá trị</s_seller><s_address>Giá trị</s_address>...
            # Phù hợp với cấu hình decoder_start_token_id = <s_seller>
            ground_truth = (
                f"<s_seller>{' '.join(seller_items)}</s_seller>"
                f"<s_address>{' '.join(address_items)}</s_address>"
                f"<s_timestamp>{' '.join(timestamp_items)}</s_timestamp>"
                f"<s_total>{' '.join(total_items)}</s_total>"
            )
            
            line = {
                "file_name": row['img_id'],
                "ground_truth": ground_truth
            }
            
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
            count += 1

    print(f"=> Đã chuyển đổi thành công {count} dòng file {csv_path} sang: {output_path}")

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    # Đảm bảo file train_final.csv và val_final.csv đã tồn tại (chạy split_data.py trước nếu cần)
    if os.path.exists("./data/train_final.csv"):
        csv_to_jsonl("./data/train_final.csv", "./data/train_metadata.jsonl")
    
    if os.path.exists("./data/val_final.csv"):
        csv_to_jsonl("./data/val_final.csv", "./data/val_metadata.jsonl")