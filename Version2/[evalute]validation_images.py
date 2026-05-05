import os
import json
import random
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import re
from underthesea import text_normalize

# --- CẤU HÌNH ---
MODEL_PATH = r"D:\data\HUIT\Nam3\HK2\Deep Learning\TaiLieuThayBao\Project\Donut\Version2\donut_result"
VAL_METADATA = r"D:\data\HUIT\Nam3\HK2\Deep Learning\TaiLieuThayBao\Project\Donut\Version2\Train_data\val_metadata.jsonl"
VAL_IMG_DIR = r"D:\data\HUIT\Nam3\HK2\Deep Learning\TaiLieuThayBao\Project\Donut\Version2\Train_data\val"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_HEIGHT = 1280
IMAGE_WIDTH = 960

def clean_vietnamese_text(text):
    if not isinstance(text, str) or text.strip() == "N/A":
        return text
    text = text_normalize(text)
    text = re.sub(r'([áàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờớởỡợúùủũụưứừửữựýỳỷỹỵ])\s+([a-z])', r'\1\2', text)
    text = re.sub(r'([áàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờớởỡợúùủũụưứừửữựýỳỷỹỵ])n\s+g', r'\1ng', text)
    corrections = {"Ngàybán": "Ngày bán", "TỔNGTIỀN": "TỔNG TIỀN", "Thờigian": "Thời gian", "Địachỉ": "Địa chỉ", "khá ch": "khách", "tiề n": "tiền", "thà nh": "thành"}
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return re.sub(r'\s+', ' ', text).strip()

def run_prediction(model, processor, image):
    pixel_values = processor(image, return_tensors="pt", do_resize=True, size={"height": 1280, "width": 960}, do_align_long_axis=False).pixel_values.to(DEVICE)
    start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_seller>")
    outputs = model.generate(pixel_values, max_length=512, decoder_input_ids=torch.tensor([[start_token_id]]).to(DEVICE), pad_token_id=processor.tokenizer.pad_token_id, eos_token_id=processor.tokenizer.eos_token_id, use_cache=True)
    sequence = processor.batch_decode(outputs)[0]
    print(f"DEBUG - Raw sequence: {sequence}")
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<pad>", "", sequence)
    try:
        prediction = processor.token2json(sequence)
        if isinstance(prediction, dict):
            if "seller" in prediction and isinstance(prediction["seller"], dict):
                for key, value in prediction["seller"].items():
                    if isinstance(value, str): prediction["seller"][key] = clean_vietnamese_text(value)
            else:
                for key, value in prediction.items():
                    if isinstance(value, str): prediction[key] = clean_vietnamese_text(value)
    except:
        prediction = clean_vietnamese_text(sequence)
    return prediction

def main():
    print(f'Đang kiểm tra mô hình từ: {MODEL_PATH}')
    processor = DonutProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    with open(VAL_METADATA, "r", encoding="utf-8") as f:
        val_samples = [json.loads(line) for line in f]
    samples = random.sample(val_samples, min(5, len(val_samples)))
    for i, sample in enumerate(samples):
        img_path = os.path.join(VAL_IMG_DIR, sample["file_name"])
        if not os.path.exists(img_path): continue
        image = Image.open(img_path).convert("RGB")
        prediction = run_prediction(model, processor, image)
        print(f'\n[{i+1}] Ảnh: {sample["file_name"]}')
        print(f' - DỰ ĐOÁN: {json.dumps(prediction, indent=2, ensure_ascii=False)}')
        print("-" * 50)

if __name__ == "__main__":
    main()