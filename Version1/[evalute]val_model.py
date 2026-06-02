import os
import json
import torch
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import DonutProcessor, VisionEncoderDecoderModel
from difflib import SequenceMatcher

from model_config import (
    check_gpu,
    METADATA_VAL,
    VAL_IMG_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH
)

MODEL_PATH = "./donut_result"

def clean_text(text):

    text = re.sub(r"<[^>]+>", "", text)

    vowels_and_d = 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'

    text = re.sub(r'(?i)([' + vowels_and_d + r'])\s+([a-z]{1,2})\b', r'\1\2', text)

    text = re.sub(r'(\d+)([a-zA-ZđĐ])', r'\1 \2', text)

    allowed_chars = r'[^a-zA-Z0-9\s.,:\-\/()đĐàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]'
    text = re.sub(allowed_chars, '', text)

    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def crop_image_sliding_window(img, window_height=1200, overlap=400):
    """Cắt ảnh thành các cửa sổ trượt giữ nguyên độ nét"""
    width, height = img.size
    crops = []

    if height <= window_height:
        return [img]

    y_start = 0
    while y_start < height:
        y_end = y_start + window_height

        if y_end > height:
            y_end = height
            y_start = max(0, height - window_height)

        box = (0, y_start, width, y_end)
        crops.append(img.crop(box))

        y_start += (window_height - overlap)
        if y_end == height:
            break

    return crops

def merge_predictions(crop_preds):
    """Gộp kết quả từ các mảng cắt. Ưu tiên lấy giá trị có nội dung từ các thẻ XML"""

    merged = {"seller": "", "address": "", "timestamp": "", "total": ""}
    for field in merged.keys():
        values = []
        for seq in crop_preds:
            val = extract_field_seq(seq, field)
            if val and val not in values:
                values.append(val)

        merged[field] = " ".join(values).strip()
    return merged

def similarity_score(label, pred):
    return SequenceMatcher(None, label, pred).ratio()
def extract_field_seq(seq, tag):

    pattern = f"<s_{tag}>(.*?)</s_{tag}>"
    match = re.search(pattern, seq)
    if match:
        return clean_text(match.group(1))
    return ""

def main():
    device = check_gpu()
    print(f"--- ĐÁNH GIÁ MODEL CHÍNH THỨC: {MODEL_PATH} ---")
    print(f"Thiết bị: {device}")

    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy thư mục model tại {MODEL_PATH}")
        print("Bạn hãy move checkpoint tốt nhất vào đây trước khi chạy.")
        return

    try:

        print("Loading Processor & Model...")
        processor = DonutProcessor.from_pretrained(MODEL_PATH)

        processor.image_processor.size = {"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH}
        processor.image_processor.do_align_long_axis = False

        model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        print("-> Load thành công!")

    except Exception as e:
        print(f"Lỗi load model: {e}")
        return

    print(f"Đọc dữ liệu từ: {METADATA_VAL}")
    val_samples = []
    with open(METADATA_VAL, 'r', encoding='utf-8') as f:
        for line in f:
            val_samples.append(json.loads(line))

    print(f"-> Tổng số mẫu: {len(val_samples)}")

    results = {
        "seller": {"acc": [], "sim": []},
        "address": {"acc": [], "sim": []},
        "timestamp": {"acc": [], "sim": []},
        "total": {"acc": [], "sim": []}
    }

    total_val_loss = 0
    valid_loss_steps = 0
    print_count = 0

    print("\n>>> START EVALUATION...")
    for sample in tqdm(val_samples, desc="Processing"):
        img_name = sample["file_name"]
        ground_truth_str = sample["ground_truth"]

        img_path = os.path.join(VAL_IMG_PATH, img_name)

        if not os.path.exists(img_path):
            continue

        try:

            image = Image.open(img_path).convert("RGB")
            crops = crop_image_sliding_window(image, window_height=1200, overlap=400)

            crop_predictions = []

            for crop in crops:
                pixel_values = processor(
                    crop,
                    return_tensors="pt",
                    do_resize=True,
                    size={"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH},
                    do_align_long_axis=False
                ).pixel_values.to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        pixel_values,
                        decoder_start_token_id=model.config.decoder_start_token_id,
                        max_length=512,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True,
                    )

                pred_seq = processor.batch_decode(outputs)[0]
                pred_seq = pred_seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
                crop_predictions.append(pred_seq)

            merged_pred = merge_predictions(crop_predictions)

            orig_pixel_values = processor(image, return_tensors="pt", do_resize=True, size={"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH}, do_align_long_axis=False).pixel_values.to(device)
            decoder_input_ids = processor.tokenizer(ground_truth_str + processor.tokenizer.eos_token, add_special_tokens=False, max_length=512, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
            labels = decoder_input_ids.clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            with torch.no_grad():
                loss_output = model(pixel_values=orig_pixel_values, labels=labels)
                total_val_loss += loss_output.loss.item()
                valid_loss_steps += 1

            if print_count < 10:
                print(f"\n\n[Mẫu {print_count+1}] - Ảnh: {img_name} (Cắt thành {len(crops)} mảnh)")
                for field in ["seller", "address", "timestamp", "total"]:
                    pred_val = merged_pred[field]
                    gt_val = extract_field_seq(ground_truth_str, field)
                    print(f"  {field.upper():<10} | Thực tế: '{gt_val}'")
                    print(f"  {'':<10} | Dự đoán: '{pred_val}'")
                print_count += 1

            for field in ["seller", "address", "timestamp", "total"]:
                pred_val = merged_pred[field]
                gt_val = extract_field_seq(ground_truth_str, field)

                results[field]["acc"].append(1 if pred_val == gt_val else 0)
                results[field]["sim"].append(similarity_score(gt_val, pred_val))

        except Exception:
            pass

    avg_loss = total_val_loss / valid_loss_steps if valid_loss_steps > 0 else 0

    print("\n" + "="*80)
    print(f" FINAL REPORT FOR: {MODEL_PATH}")
    print(f" LOSS: {avg_loss:.4f}")
    print("="*80)
    print(f"{'FIELD':<15} | {'ACCURACY':<15} | {'SIMILARITY':<15}")
    print("-" * 80)

    final_acc, final_sim = [], []
    for field, data in results.items():
        if not data["acc"]: continue
        acc = np.mean(data["acc"])
        sim = np.mean(data["sim"])
        final_acc.append(acc)
        final_sim.append(sim)
        print(f"{field.upper():<15} | {acc:.2%} {'':<8} | {sim:.2%}")

    print("-" * 80)
    print(f"{'AVERAGE':<15} | {np.mean(final_acc):.2%} {'':<8} | {np.mean(final_sim):.2%}")
    print("="*80)

if __name__ == "__main__":
    main()
