import os
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from model_config import check_gpu, IMAGE_HEIGHT, IMAGE_WIDTH
from evalute_val_model import crop_image_sliding_window, merge_predictions, MODEL_PATH

REAL_IMG_PATH = "./data/Tien_real_images"

def main():
    device = check_gpu()
    print(f"\n--- BẮT ĐẦU ĐÁNH GIÁ TRÊN ẢNH THỰC TẾ (Folder: {REAL_IMG_PATH}) ---")

    if not os.path.exists(REAL_IMG_PATH):
        print(f"Lỗi: Không tìm thấy thư mục '{REAL_IMG_PATH}'")
        print("Vui lòng đảm bảo folder 'Tien_real_images' nằm trong thu mục 'data'!")
        return

    real_images = [f for f in os.listdir(REAL_IMG_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not real_images:
        print(f"Thư mục {REAL_IMG_PATH} đang trống hoặc chưa có ảnh!")
        return

    print(f"-> Tìm thấy {len(real_images)} bức ảnh thực tế.")
    print("Loading Processor & Model Donut...")

    try:
        processor = DonutProcessor.from_pretrained(MODEL_PATH)
        processor.image_processor.size = {"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH}
        processor.image_processor.do_align_long_axis = False

        model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Lỗi khi tải checkpoint: {e}")
        return

    for img_name in real_images:
        print("\n" + "="*70)
        print(f" ĐANG XỬ LÝ ẢNH: {img_name}")
        img_path = os.path.join(REAL_IMG_PATH, img_name)

        try:
            image = Image.open(img_path).convert("RGB")

            w, h = image.size
            if w != IMAGE_WIDTH:
                new_h = int(h * (IMAGE_WIDTH / w))
                image = image.resize((IMAGE_WIDTH, new_h), Image.Resampling.LANCZOS)

            crops = crop_image_sliding_window(image, window_height=IMAGE_HEIGHT, overlap=400)
            crop_predictions = []

            for crop in crops:
                pixel_values = processor(
                    crop, return_tensors="pt", do_resize=True,
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
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=5,
                        use_cache=True,
                    )

                pred_seq = processor.batch_decode(outputs)[0]
                pred_seq = pred_seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
                crop_predictions.append(pred_seq)

            merged_pred = merge_predictions(crop_predictions)

            print("-" * 70)
            for field in ["seller", "address", "timestamp", "total"]:
                print(f"  {field.upper():<12} : {merged_pred.get(field, '')}")

        except Exception as e:
            print(f"Lỗi bỏ qua file {img_name}: {e}")

    print("\n" + "="*70)
    print("HOÀN TẤT KIỂM TRA ẢNH THỰC TẾ CỦA BẠN!")

if __name__ == '__main__':
    main()
