import os
import numpy as np
import torch
from PIL import Image
from transformers import DonutProcessor
from model_config import DonutDataset, IMAGE_HEIGHT, IMAGE_WIDTH

def visualize():
    # Khởi tạo processor để test
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    
    # Path tới metadata train của Version 2
    metadata_path = r"d:\data\HUIT\Nam3\HK2\Deep Learning\TaiLieuThayBao\Project\Donut\Version2\Train_data\train_metadata.jsonl"
    
    if not os.path.exists(metadata_path):
        print(f"Không tìm thấy file metadata tại: {metadata_path}")
        return

    # Load dataset với split="train" để kích hoạt Augmentation
    dataset = DonutDataset(metadata_path, processor, split="train")
    
    # Tạo thư mục lưu ảnh test nếu chưa có
    save_dir = "./aug_preview"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Bắt đầu trích xuất 5 mẫu ảnh đã qua Augmentation vào thư mục {save_dir}...")

    for i in range(5):
        # Mỗi lần gọi __getitem__, một biến thể ngẫu nhiên sẽ được tạo ra
        sample = dataset[i]
        
        # Chuyển pixel_values (tensor) ngược lại thành ảnh để xem
        # Donut pixel_values thường được normalize, ta chỉ cần minh họa cơ chế
        # Để xem ảnh rõ nhất, ta sẽ can thiệp vào Dataset để lưu file PIL trực tiếp (tạm thời)
        
        # Lấy lại ảnh gốc và áp dụng aug bằng tay để lưu file xem thử
        item = dataset.dataset[i]
        img_path = os.path.join(dataset.img_dir, item["file_name"])
        image = Image.open(img_path).convert("RGB")
        
        # Chạy Augmentation
        image_np = np.array(image)
        augmented = dataset.aug(image=image_np)["image"]
        aug_img = Image.fromarray(augmented)
        
        # Lưu ảnh
        aug_img.save(f"{save_dir}/sample_{i}_augmented.jpg")
        print(f" - Đã lưu: sample_{i}_augmented.jpg")

    print("\nHãy kiểm tra thư mục 'aug_preview' để xem thành quả làm giàu dữ liệu!")

if __name__ == "__main__":
    visualize()
