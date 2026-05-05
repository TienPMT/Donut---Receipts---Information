import os
import random
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Tái sử dụng lại các thiết lập và cấu hình từ evaluate_model.py
from model_config import check_gpu, IMAGE_HEIGHT, IMAGE_WIDTH
from evalute_val_model import (
    crop_image_sliding_window, 
    merge_predictions, 
    extract_field_seq,
    MODEL_PATH
)

TEST_IMG_PATH = "./data/test_images"

def main():
    device = check_gpu()
    print(f"\n--- BẮT ĐẦU TEST ẢNH NGẪU NHIÊN ---")
    print(f"Thiết bị đang sử dụng: {device}")
    
    # 1. Trích xuất ngẫu nhiên 1 bức ảnh từ thư mục test
    if not os.path.exists(TEST_IMG_PATH):
        print(f"Lỗi: Không tìm thấy thư mục {TEST_IMG_PATH}")
        return
        
    test_images = [f for f in os.listdir(TEST_IMG_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not test_images:
        print(f"Lỗi: Không có ảnh nào trong thư mục {TEST_IMG_PATH}")
        return
        
    random_img_name = random.choice(test_images)
    img_path = os.path.join(TEST_IMG_PATH, random_img_name)
    
    print(f"\n[1] Đã chọn ngẫu nhiên ảnh: {random_img_name}")
    
    # 2. Hiển thị ảnh (Mở bằng trình xem ảnh mặc định của Windows)
    try:
        print("[2] Đang mở ảnh để bạn quan sát...")
        image = Image.open(img_path).convert("RGB")
        image.show()
    except Exception as e:
        print(f"Không thể hiển thị ảnh (Vẫn tiếp tục chạy dự đoán): {e}")
        return
        
    # 3. Load model và processor
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy model checkpoint tại {MODEL_PATH}")
        return
        
    print("[3] Đang tải Processor & Model Donut...")
    processor = DonutProcessor.from_pretrained(MODEL_PATH)
    processor.image_processor.size = {"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH}
    processor.image_processor.do_align_long_axis = False
    
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    
    print("[4] Đang dự đoán bằng thuật toán Cửa sổ trượt (Sliding Window)...")
    
    # Cắt ảnh thành các mảng
    crops = crop_image_sliding_window(image, window_height=1200, overlap=400)
    crop_predictions = []
    
    # Quét từng mảng cắt qua mô hình
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
        
        # Giải mã mảng cắt hiện tại
        pred_seq = processor.batch_decode(outputs)[0]
        pred_seq = pred_seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        crop_predictions.append(pred_seq)
        
    # Gộp toàn bộ kết quả từ các mảng (Đã tích hợp Regex Custom)
    merged_pred = merge_predictions(crop_predictions)
    
    # 4. In kết quả cuối cùng ra màn hình Console
    print("\n" + "="*60)
    print(f"KẾT QUẢ TRÍCH XUẤT CHO ẢNH: {random_img_name}")
    print("="*60)
    for field in ["seller", "address", "timestamp", "total"]:
        print(f"   {field.upper():<12}: {merged_pred[field]}")
    print("="*60)

if __name__ == "__main__":
    main()
