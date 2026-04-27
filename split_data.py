import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil

# --- CẤU HÌNH ---
input_csv = 'Usage_Data/mcocr_train_df.csv'
source_images_dir = os.path.join('Usage_Data', 'train_images', 'train_images') # Đường dẫn gốc chứa ảnh

train_output_csv = 'train_final.csv'
val_output_csv = 'val_final.csv'

# Folder đích để chứa ảnh sau khi chia
train_images_dest = 'train_images'
val_images_dest = 'val_images'

quality_threshold = 0.2  # Lọc bỏ ảnh có chất lượng dưới 0.2

def copy_images(image_list, source_dir, dest_dir):
    """
    Hàm sao chép danh sách ảnh từ thư mục nguồn sang thư mục đích
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Đã tạo thư mục: {dest_dir}")
    
    print(f"Bắt đầu sao chép {len(image_list)} ảnh vào {dest_dir}...")
    copied_count = 0
    missing_count = 0
    
    for img_name in image_list:        
        src_path = os.path.join(source_dir, img_name)
        dst_path = os.path.join(dest_dir, img_name)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            missing_count += 1
            print(f"Cảnh báo: Không tìm thấy ảnh {img_name}")
            
    print(f"Hoàn tất: Sao chép {copied_count} ảnh. (Thiếu: {missing_count})")

def split_and_clean():
    # 1. Đọc file CSV gốc
    if not os.path.exists(input_csv):
        print(f"Lỗi: Không tìm thấy file {input_csv}")
        return

    df = pd.read_csv(input_csv)
    initial_count = len(df)
    print(f"Tổng số dòng ban đầu: {initial_count}")

    # 2. Bước tiền xử lý: Lọc ảnh chất lượng thấp (Nếu có cột chất lượng)
    if 'anno_image_quality' in df.columns:
        df = df[df['anno_image_quality'] >= quality_threshold]
        print(f"Đã lọc bỏ {initial_count - len(df)} ảnh chất lượng thấp (Ngưỡng < {quality_threshold})")
    
    # 3. Thực hiện chia tập dữ liệu
    # test_size=300: Lấy đúng 300 mẫu cho tập Validation
    # shuffle=True: Xáo trộn dữ liệu trước khi chia
    train_df, val_df = train_test_split(
        df,
        test_size=300, 
        random_state=42, 
        shuffle=True
    )

    # 4. Lưu ra các file CSV mới
    train_df.to_csv(train_output_csv, index=False)
    val_df.to_csv(val_output_csv, index=False)

    print("-" * 30)
    print(f"KẾT QUẢ CHIA DỮ LIỆU:")
    print(f"- Tập Train mới: {len(train_df)} mẫu -> File: {train_output_csv}")
    print(f"- Tập Val mới: {len(val_df)} mẫu -> File: {val_output_csv}")
    
    # 5. Sao chép ảnh sang thư mục tương ứng
    print("-" * 30)
    print("Đang tiến hành tách ảnh sang các folder riêng biệt...")
    # Cột chứa tên ảnh là 'img_id'
    copy_images(train_df['img_id'], source_images_dir, train_images_dest)
    copy_images(val_df['img_id'], source_images_dir, val_images_dest)
    
    print("-" * 30)
    print("XONG!")

if __name__ == "__main__":
    split_and_clean()