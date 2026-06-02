import os
import numpy as np
import torch
from PIL import Image
from transformers import DonutProcessor
from model_config import DonutDataset, IMAGE_HEIGHT, IMAGE_WIDTH

def visualize():

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

    metadata_path = r"d:\data\HUIT\Nam3\HK2\Deep Learning\TaiLieuThayBao\Project\Donut\Version2\Train_data\train_metadata.jsonl"

    if not os.path.exists(metadata_path):
        print(f"Không tìm thấy file metadata tại: {metadata_path}")
        return

    dataset = DonutDataset(metadata_path, processor, split="train")

    save_dir = "./aug_preview"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Bắt đầu trích xuất 5 mẫu ảnh đã qua Augmentation vào thư mục {save_dir}...")

    for i in range(5):

        sample = dataset[i]

        item = dataset.dataset[i]
        img_path = os.path.join(dataset.img_dir, item["file_name"])
        image = Image.open(img_path).convert("RGB")

        image_np = np.array(image)
        augmented = dataset.aug(image=image_np)["image"]
        aug_img = Image.fromarray(augmented)

        aug_img.save(f"{save_dir}/sample_{i}_augmented.jpg")
        print(f" - Đã lưu: sample_{i}_augmented.jpg")

    print("\nHãy kiểm tra thư mục 'aug_preview' để xem thành quả làm giàu dữ liệu!")

if __name__ == "__main__":
    visualize()
