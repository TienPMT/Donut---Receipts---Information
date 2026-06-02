import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset
import json
import os
import numpy as np
from PIL import Image
import albumentations as A
from tokenizers import AddedToken

IMAGE_HEIGHT = 1280
IMAGE_WIDTH = 960

class DonutDataset(Dataset):
    def __init__(self, dataset_name_or_path, processor, max_length=768, split="train", duplication_factor=3):
        super().__init__()
        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.dataset = []

        self.duplication_factor = duplication_factor if split == "train" else 1

        with open(dataset_name_or_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.dataset.append(json.loads(line))

        folder_name = "train" if split == "train" else "validation"
        self.img_dir = os.path.join(os.path.dirname(dataset_name_or_path), folder_name)

        self.aug = None
        if split == "train":
            self.aug = A.Compose([
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        real_idx = idx % len(self.dataset)
        item = self.dataset[real_idx]

        image_path = os.path.join(self.img_dir, item["file_name"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Lỗi đọc ảnh {image_path}: {e}")

            image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))

        if self.aug:
            image_np = np.array(image)
            augmented = self.aug(image=image_np)["image"]
            image = Image.fromarray(augmented)

        pixel_values = self.processor(
            image,
            return_tensors="pt",
            do_resize=True,
            size={"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH},
            do_align_long_axis=False
        ).pixel_values.squeeze()

        try:
            gt_data = json.loads(item["ground_truth"])
            gt_dict = gt_data.get("gt_parse", gt_data)

            seller_val = gt_dict.get('SELLER', gt_dict.get('seller', ''))
            addr_val = gt_dict.get('ADDRESS', gt_dict.get('address', ''))
            time_val = gt_dict.get('TIMESTAMP', gt_dict.get('timestamp', ''))
            cost_val = gt_dict.get('TOTAL_COST', gt_dict.get('total_cost', ''))

            if isinstance(seller_val, dict):
                seller_val = seller_val.get('seller_name', '')
        except Exception as e:
            print(f"Lỗi parse Ground Truth {item['file_name']}: {e}")
            seller_val = addr_val = time_val = cost_val = ""

        gt_string = f"<s_seller><s_seller_name>{seller_val}</s_seller_name>"\
                    f"<s_address>{addr_val}</s_address>"\
                    f"<s_timestamp>{time_val}</s_timestamp>"\
                    f"<s_total_cost>{cost_val}</s_total_cost></s_seller>"

        labels = self.processor.tokenizer(
            gt_string,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}

def setup_model_and_processor(device):
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

    model.config.encoder.image_size = [IMAGE_HEIGHT, IMAGE_WIDTH]
    processor.image_processor.size = {"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH}
    processor.image_processor.do_align_long_axis = False

    new_tokens = [
        "<s_seller>", "</s_seller>",
        "<s_seller_name>", "</s_seller_name>",
        "<s_address>", "</s_address>",
        "<s_timestamp>", "</s_timestamp>",
        "<s_total_cost>", "</s_total_cost>"
    ]

    vietnamese_chars = list("áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ")

    vietnamese_tokens = [
        AddedToken(c, single_word=False, lstrip=False, rstrip=False, normalized=False)
        for c in vietnamese_chars
    ]

    processor.tokenizer.add_tokens(new_tokens + vietnamese_tokens)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_seller>")

    model.generation_config.max_length = 768
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_seller>")
    model.generation_config.eos_token_id = processor.tokenizer.eos_token_id

    if hasattr(model.config, "max_length"):
        model.config.max_length = None
    if hasattr(model.config.decoder, "max_length"):
        model.config.decoder.max_length = None

    return model.to(device), processor

def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    max_label_len = max([len(x["labels"]) for x in batch])

    padded_labels = []
    for x in batch:
        lab = x["labels"]
        pad_size = max_label_len - len(lab)
        padded_lab = torch.cat([lab, torch.full((pad_size,), -100, dtype=torch.long)])
        padded_labels.append(padded_lab)

    return {
        "pixel_values": pixel_values,
        "labels": torch.stack(padded_labels)
    }
