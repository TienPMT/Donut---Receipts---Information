import os
import json
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, AutoTokenizer
import re

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "donut_result")
DEFAULT_IMG_DIR = os.path.join(BASE_DIR, "data1", "validation")
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
        max_length=768, # Đã tăng lên 768 để khớp với lúc Train
        decoder_input_ids=torch.tensor([[start_token_id]]).to(DEVICE),
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=None,
    )

    sequence = processor.batch_decode(outputs)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<pad>", "", sequence)
    
    # =====================================================================
    # THUẬT TOÁN HẬU XỬ LÝ KHOẢNG TRẮNG TIẾNG VIỆT THÔNG MINH
    # =====================================================================
    # BƯỚC 1: QUY TẮC DẤU CÂU (Tách chữ dính vào dấu chấm, dấu phẩy)
    # VD: "P.Cẩ m" -> "P. Cẩ m" | "6,P.Cẩm" -> "6, P. Cẩm"
    sequence = re.sub(r'([,.;])([A-ZĐa-zà-ỹ])', r'\1 \2', sequence)

    # BƯỚC 2: QUY TẮC 2 DẤU THANH (Tách 2 từ bị dính liền nhau)
    # Tiếng Việt không bao giờ có 2 chữ có dấu thanh đứng sát vách.
    # VD: "Dựá n" -> "Dự á n" | "thịxã" -> "thị xã"
    toned_vowels = "áắấéếíóốớúứýàằầèềìòồờùừỳảẳẩẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹạặậẹệịọộợụựỵ"
    toned_vowels += toned_vowels.upper()
    sequence = re.sub(r'(["' + toned_vowels + r'])(["' + toned_vowels + r'])', r'\1 \2', sequence)

    # BƯỚC 3: QUY TẮC NỐI ĐUÔI (Gộp các chữ cái đuôi bị mô hình tách rời)
    # VD: "Cẩ m" -> "Cẩm" | "Dự á n" -> "Dự án"
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
        
    return prediction, sequence

def main():
    print(f"\n{'='*50}")
    print(f"--- ĐÁNH GIÁ THỰC CHIẾN DONUT VERSION 3 ---")
    print(f"{'='*50}\n")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[THÔNG BÁO] Chưa tìm thấy model tại {MODEL_PATH}. Hãy train trước!")
        return

    # 1. NHẬP ĐƯỜNG DẪN LINH HOẠT TỪ NGƯỜI DÙNG
    print(f"Thư mục mặc định: {DEFAULT_IMG_DIR}")
    user_input = input("Nhập đường dẫn thư mục chứa ảnh test (Nhấn Enter để dùng mặc định): ").strip()
    
    real_img_dir = user_input if user_input else DEFAULT_IMG_DIR

    if not os.path.exists(real_img_dir):
        os.makedirs(real_img_dir, exist_ok=True)
        print(f"\n[LƯU Ý] Đã tạo thư mục {real_img_dir}.")
        print("Hãy copy ảnh thực tế vào thư mục này rồi chạy lại script nhé!")
        return

    # 2. TẢI MÔ HÌNH VÀ PROCESSOR (Có dự phòng lỗi)
    print(f"\nĐang tải model từ: {MODEL_PATH}...")
    try:
        processor = DonutProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    except OSError:
        print(f"[Cảnh báo] Thiếu file config của processor. Đang tải dự phòng...")
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        try:
            processor.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        except OSError:
            pass

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    # 3. QUÉT VÀ DỰ ĐOÁN ẢNH
    image_files = [f for f in os.listdir(real_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"-> Tìm thấy {len(image_files)} ảnh hợp lệ trong thư mục.\n")

    for i, file_name in enumerate(image_files):
        img_path = os.path.join(real_img_dir, file_name)
        image = Image.open(img_path).convert("RGB")
        
        prediction, raw_seq = run_prediction(model, processor, image)

        print(f"[{i+1}/{len(image_files)}] Ảnh: {file_name}")
        print(f"--- KẾT QUẢ TRÍCH XUẤT ---")
        print(json.dumps(prediction, indent=2, ensure_ascii=False))
        print("-" * 50)

if __name__ == "__main__":
    main()