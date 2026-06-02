import os
import json
import random
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, AutoTokenizer
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "donut_result")
VAL_METADATA = os.path.join(BASE_DIR, "data1", "validation_metadata.jsonl")
VAL_IMG_DIR = os.path.join(BASE_DIR, "data1", "validation")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_HEIGHT = 1280
IMAGE_WIDTH = 960

def run_prediction(model, processor, image):

    pixel_values = processor(
        image,
        return_tensors="pt",
        do_resize=True,
        size={"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH},
        do_align_long_axis=False
    ).pixel_values.to(DEVICE)

    start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_seller>")

    outputs = model.generate(
        pixel_values,
        max_length=768,
        decoder_input_ids=torch.tensor([[start_token_id]]).to(DEVICE),
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=None,
    )

    sequence = processor.batch_decode(outputs)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<pad>", "", sequence)

    sequence = re.sub(r'([,.;])([A-ZĐa-zà-ỹ])', r'\1 \2', sequence)

    toned_vowels = "áắấéếíóốớúứýàằầèềìòồờùừỳảẳẩẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹạặậẹệịọộợụựỵ"
    toned_vowels += toned_vowels.upper()
    sequence = re.sub(r'(["' + toned_vowels + r'])(["' + toned_vowels + r'])', r'\1 \2', sequence)

    all_vowels = "aăâeêioôơuưyáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵĐđ"
    all_vowels += all_vowels.upper()

    tails = ["c", "ch", "m", "n", "ng", "nh", "p", "t", "i", "y", "a", "o", "u"]
    tails += [t.upper() for t in tails]
    tail_pattern = "|".join(tails)

    pattern = r"([" + all_vowels + r"])\s+(" + tail_pattern + r")\b"

    old_sequence = ""
    while old_sequence != sequence:
        old_sequence = sequence
        sequence = re.sub(pattern, r"\1\2", sequence)

    try:
        prediction = processor.token2json(sequence)
    except Exception:
        prediction = sequence

    return prediction

def main():
    print(f'Đang kiểm tra mô hình từ: {MODEL_PATH}')
    if not os.path.exists(MODEL_PATH):
        print(f"Chưa có model tại {MODEL_PATH}. Hãy train trước hoặc trỏ MODEL_PATH tới folder checkpoint.")
        return

    try:
        processor = DonutProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    except OSError:
        print(f"[Cảnh báo] Thiếu file config của processor trong {MODEL_PATH}.")
        print("Đang tải Image Processor từ base model và Tokenizer từ local...")
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

        try:
            processor.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        except OSError:
            print("[Cảnh báo] Không tìm thấy tokenizer local, sẽ sử dụng tokenizer mặc định của base model.")

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    with open(VAL_METADATA, "r", encoding="utf-8") as f:
        val_samples = [json.loads(line) for line in f]

    samples = random.sample(val_samples, min(10, len(val_samples)))
    for i, sample in enumerate(samples):
        img_path = os.path.join(VAL_IMG_DIR, sample["file_name"])
        if not os.path.exists(img_path): continue
        image = Image.open(img_path).convert("RGB")
        prediction = run_prediction(model, processor, image)

        print(f'\n[{i+1}] Ảnh: {sample["file_name"]}')

        gt = json.loads(sample["ground_truth"]).get("gt_parse", {})
        print(f' - ĐÚNG: {json.dumps(gt, ensure_ascii=False)}')
        print(f' - DỰ ĐOÁN: {json.dumps(prediction, indent=2, ensure_ascii=False)}')
        print("-" * 50)

if __name__ == "__main__":
    main()
