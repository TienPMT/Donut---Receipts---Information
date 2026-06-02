import os
import json
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, AutoTokenizer
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "donut_result")

IMAGE_DIR = os.path.join(BASE_DIR, "data1", "train")

IMAGE_NAME = "mcocr_public_145013mqmzy.jpg"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 1280
IMAGE_WIDTH = 960

def run_prediction(model, processor, image):
    """Hàm xử lý ảnh và bóc tách thông tin qua mô hình Donut"""

    pixel_values = processor(
        image,
        return_tensors="pt",
        do_resize=True,
        size={"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH},
        do_align_long_axis=False
    ).pixel_values.to(DEVICE)

    start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_seller>")

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            max_length=768,
            decoder_input_ids=torch.tensor([[start_token_id]]).to(DEVICE),
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )

    sequence = processor.batch_decode(outputs)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<pad>", "", sequence)

    vowels = "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵĐđ"
    vowels += vowels.upper()
    tails = ["c", "ch", "m", "n", "ng", "nh", "p", "t", "i", "y", "a", "o", "u"]
    tails += [t.upper() for t in tails]
    tail_pattern = "|".join(tails)

    pattern = r"([" + vowels + r"])\s+(" + tail_pattern + r")\b"

    old_sequence = ""
    while old_sequence != sequence:
        old_sequence = sequence
        sequence = re.sub(pattern, r"\1\2", sequence)

    try:
        prediction_json = processor.token2json(sequence)

        if "seller" not in prediction_json:
            prediction_json = {"seller": prediction_json}
    except Exception:
        prediction_json = {"seller": {}}

    return prediction_json, sequence

def main():
    print("CÔNG CỤ TRÍCH XUẤT THÔNG TIN HÓA ĐƠN ĐƠN LẺ (DONUT V3)")

    image_path = os.path.join(IMAGE_DIR, IMAGE_NAME)
    if not os.path.exists(image_path):
        print(f"\n[LỖI] Không tìm thấy ảnh: {image_path}")
        print("Vui lòng kiểm tra lại biến IMAGE_DIR và IMAGE_NAME.")
        return

    print(f"\n Đang tải mô hình từ: {MODEL_PATH}...")
    try:
        processor = DonutProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    except OSError:
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        try:
            processor.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        except OSError:
            pass

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    print(f"Đã tải mô hình xong (Sử dụng {DEVICE.upper()}).")
    print(f"\nĐang phân tích ảnh: {IMAGE_NAME}...")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[LỖI] Không thể đọc được file ảnh: {e}")
        return

    prediction_json, raw_sequence = run_prediction(model, processor, image)

    print("KẾT QUẢ TRÍCH XUẤT")

    print(json.dumps(prediction_json, indent=4, ensure_ascii=False))

    print("CHUỖI VĂN BẢN THÔ MÔ HÌNH SINH RA")
    print(raw_sequence)

if __name__ == "__main__":
    main()
