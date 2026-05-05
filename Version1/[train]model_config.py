import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    DonutProcessor, 
    VisionEncoderDecoderModel
)

# --- CẤU HÌNH ĐƯỜNG DẪN & HẰNG SỐ ---
TRAIN_IMG_PATH = "./data/train_resized"
METADATA_TRAIN = "./data/train_metadata.jsonl"
VAL_IMG_PATH = "./data/val_images"
METADATA_VAL = "./data/val_metadata.jsonl" # File metadata validation

MODEL_NAME = "naver-clova-ix/donut-base"

# Kích thước ảnh thực tế (Height, Width) - Lưu ý đảo ngược so với OpenCV (Width, Height)
# OpenCV/PIL: (960, 1280) -> (Width, Height)
# Donut Config: [1280, 960] -> [Height, Width]
IMAGE_HEIGHT = 1280
IMAGE_WIDTH = 960

def check_gpu():
    """Kiểm tra GPU và trả về thiết bị training. Kết thúc nếu không có GPU."""
    if not torch.cuda.is_available():
        print("Khong su dung GPU. Chuong trinh se ket thuc.")
        exit()
    
    device_name = torch.cuda.get_device_name(0)
    print(f"Đang sử dụng GPU: {device_name}")
    return "cuda"

class DonutDataset(Dataset):
    """Lớp Dataset để đọc dữ liệu ảnh và JSONL"""
    def __init__(self, dataset_path, metadata_path, processor, max_length=256):
        self.dataset_path = dataset_path
        self.processor = processor
        self.max_length = max_length
        self.metadata = []
        
        if not os.path.exists(metadata_path):
             print(f"Lỗi: Không tìm thấy file {metadata_path}")
             return

        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.metadata.append(json.loads(line))
                except Exception as e:
                    print(f"Lỗi parse JSON: {e}")
                    continue

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata[idx]
        img_path = os.path.join(self.dataset_path, sample["file_name"])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Lỗi đọc ảnh {img_path}: {e}")
            image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0))
            
        # Quan trọng: do_resize=True để đảm bảo ảnh đầu vào đúng kích thước mong muốn
        # Nếu ảnh đã resize chuẩn rồi thì có thể để False, nhưng để True cho an toàn
        pixel_values = self.processor(
            image, 
            return_tensors="pt", 
            do_resize=True, 
            size={"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH},
            do_align_long_axis=False
        ).pixel_values
        
        labels = self.processor.tokenizer(
            sample["ground_truth"],
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        labels = labels.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values.squeeze(), "labels": labels}

def setup_model_and_processor(device):
    """Khởi tạo model, processor và cấu hình các token đặc biệt"""
    print(f"Đang tải model base '{MODEL_NAME}'...")
    
    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    
    # --- CẤU HÌNH LẠI KÍCH THƯỚC MODEL ---
    # Giảm kích thước ảnh đầu vào của model từ 2560x1920 xuống 1280x960
    # Giúp giảm 4 lần lượng tính toán và VRAM
    model.config.encoder.image_size = [IMAGE_HEIGHT, IMAGE_WIDTH] 
    
    # Cập nhật processor config (để sau này save model nó nhớ config này)
    processor.image_processor.size = {"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH}
    processor.image_processor.do_align_long_axis = False

    model.to(device)

    # Thêm Special Tokens cho hóa đơn tiếng Việt
    new_tokens = ["<s_seller>", "<s_addr>", "<s_total>", "<s_timestamp>"]
    processor.tokenizer.add_tokens(new_tokens)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Cấu hình Token ID chuẩn cho quá trình decode
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_seller>")
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    
    return model, processor

def collate_fn(batch):
    """Hàm gom data thành batch cho DataLoader"""
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}