import torch
import gc
import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DonutProcessor, VisionEncoderDecoderModel

# --- CẤU HÌNH TỐI ƯU CHO RTX 5060 Ti ---
# Cho phép phân mảnh bộ nhớ thông minh để tránh OOM khi gần full VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import từ file cấu hình
from model_config import (
    check_gpu, 
    DonutDataset,
    collate_fn,
    TRAIN_IMG_PATH, 
    METADATA_TRAIN,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MODEL_NAME
)

# THƯ MỤC LƯU TRỮ TRƯỚC VÀ SAU KHI TRAIN
PRETRAINED_PATH = MODEL_NAME             # Model gốc từ HuggingFace (hoặc config)
OUTPUT_DIR = "./donut_checkpoints"       # Nơi lưu các file checkpoint trong lúc train
FINAL_SAVE_PATH = "./donut_result"       # Nơi lưu mô hình cuối cùng sau khi train xong

def setup_model_for_training(model_path, device):
    """Load model lên để train từ đầu"""
    print(f"Loading model từ: {model_path}...")
    
    # Load model & processor
    processor = DonutProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    
    # Ép lại config ảnh (quan trọng)
    model.config.encoder.image_size = [IMAGE_HEIGHT, IMAGE_WIDTH] 
    processor.image_processor.size = {"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH}
    processor.image_processor.do_align_long_axis = False

    model.to(device)
    
    # Đặt lại token ID để model biết bắt đầu từ đâu. 
    # Do bạn train từ đầu, đảm bảo tokenizer đã add thêm token '<s_seller>'
    # (Nếu chưa có, cần phải resize embedding và add thêm token cho processor, nhưng tạm thời 
    # code cũ của bạn có tokenizer đã được update hoặc mặc định, nếu có lỗi đoạn này báo lại tôi nhé)
    start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_seller>")
    if start_token_id is None or start_token_id == processor.tokenizer.unk_token_id:
        # Trường hợp train hoàn toàn từ HuggingFace nên token <s_seller> chưa có.
        # Ta add thêm token vào processor
        # Add special tokens
        special_tokens = ["<s_seller>", "</s_seller>", "<s_address>", "</s_address>", "<s_timestamp>", "</s_timestamp>", "<s_total>", "</s_total>"]
        
        # Add TIẾNG VIỆT TOKENS để không bị <unk> (Cả viết hoa & thường theo chuẩn NFC)
        vietnamese_tokens = [
            "à", "á", "ạ", "ả", "ã", "â", "ầ", "ấ", "ậ", "ẩ", "ẫ", "ă", "ằ", "ắ", "ặ", "ẳ", "ẵ", 
            "è", "é", "ẹ", "ẻ", "ẽ", "ê", "ề", "ế", "ệ", "ể", "ễ", 
            "ì", "í", "ị", "ỉ", "ĩ", 
            "ò", "ó", "ọ", "ỏ", "õ", "ô", "ồ", "ố", "ộ", "ổ", "ỗ", "ơ", "ờ", "ớ", "ợ", "ở", "ỡ", 
            "ù", "ú", "ụ", "ủ", "ũ", "ư", "ừ", "ứ", "ự", "ử", "ữ", 
            "ỳ", "ý", "ỵ", "ỷ", "ỹ", "đ",
            "À", "Á", "Ạ", "Ả", "Ã", "Â", "Ầ", "Ấ", "Ậ", "Ẩ", "Ẫ", "Ă", "Ằ", "Ắ", "Ặ", "Ẳ", "Ẵ", 
            "È", "É", "Ẹ", "Ẻ", "Ẽ", "Ê", "Ề", "Ế", "Ệ", "Ể", "Ễ", 
            "Ì", "Í", "Ị", "Ỉ", "Ĩ", 
            "Ò", "Ó", "Ọ", "Ỏ", "Õ", "Ô", "Ồ", "Ố", "Ộ", "Ổ", "Ỗ", "Ơ", "Ờ", "Ớ", "Ợ", "Ở", "Ỡ", 
            "Ù", "Ú", "Ụ", "Ủ", "Ũ", "Ư", "Ừ", "Ứ", "Ự", "Ử", "Ữ", 
            "Ỳ", "Ý", "Ỵ", "Ỷ", "Ỹ", "Đ"
        ]
        
        # Thêm toàn bộ các token mới vào tokenizer
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        processor.tokenizer.add_tokens(vietnamese_tokens) # Thêm ký tự Tiếng Việt dưới dạng token thường
        
        # BẮT BUỘC có dòng này để model thay đổi kích thước Embedding layer nhận các token mới
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_seller>")

    model.config.decoder_start_token_id = start_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    
    print("-> Model đã sẵn sàng để train!")
    return model, processor

def main():
    # 1. Kiểm tra GPU
    device = check_gpu()
    torch.cuda.empty_cache()
    gc.collect()

    # 2. Load Model & Data
    model, processor = setup_model_for_training(PRETRAINED_PATH, device)
    
    train_dataset = DonutDataset(
        dataset_path=TRAIN_IMG_PATH,
        metadata_path=METADATA_TRAIN,
        processor=processor,
        max_length=512 # Tăng lên 768 nếu hóa đơn của bạn rất dài
    )
    
    print(f"Tổng số ảnh train: {len(train_dataset)}")

    # 3. CẤU HÌNH TRAINING MẠNH HƠN CHO RTX 5060 (16GB VRAM)
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # --- TĂNG TỐC ĐỘ ---
        per_device_train_batch_size=4,  # Tăng lên 4 (hoặc 6 nếu vẫn còn VRAM)
        gradient_accumulation_steps=2,  # Giảm xuống 2 (Tổng batch ảo = 4x2 = 8)
        dataloader_num_workers=4,       # Tăng worker để load ảnh nhanh hơn
        
        # --- CHIẾN LƯỢC TRAIN ---
        num_train_epochs=50,            # Train thêm 50 epoch nữa (tổng cộng khá nhiều)
        learning_rate=2e-5,             # LR nhỏ để model học kỹ chi tiết (Seller)
        
        # --- TỐI ƯU PHẦN CỨNG ---
        bf16=True,                      # Bắt buộc bật cho RTX 50-series để chạy nhanh
        gradient_checkpointing=True,    # Vẫn nên bật để tiết kiệm VRAM cho Batch lớn
        
        # --- LOGGING & SAVE ---
        logging_steps=20,               # Log thường xuyên hơn
        save_strategy="steps",          # Chuyển sang lưu theo step thay vì mỗi epoch
        save_steps=3000,                # Lưu mỗi 1000 bước (Bạn có thể điều chỉnh con số này tùy theo độ lớn của 10 epoch)
        save_total_limit=2,             # Chỉ giữ 2 checkpoint mới nhất để tránh full ổ cứng
        remove_unused_columns=False,
        report_to="tensorboard",        # Ghi log để theo dõi được (cần cài tensorboard)
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    # 4. Bắt đầu Train
    print("\n>>> BẮT ĐẦU QUÁ TRÌNH TRAIN TỪ ĐẦU (FROM SCRATCH)...")
    print(f"Configs: Batch={training_args.per_device_train_batch_size}, Accum={training_args.gradient_accumulation_steps}, LR={training_args.learning_rate}")
    
    try:
        trainer.train()
    except Exception as e:
        print(f"\n[LỖI]: {e}")
        print("Mẹo: Nếu bị 'Out of Memory', hãy giảm 'per_device_train_batch_size' xuống 2.")
        return

    # 5. Lưu kết quả cuối cùng
    print("\n>>> ĐANG LƯU MODEL MỚI...")
    trainer.save_model(FINAL_SAVE_PATH)
    processor.save_pretrained(FINAL_SAVE_PATH)
    print(f"Hoàn tất! Model hoàn chỉnh đã lưu tại: {FINAL_SAVE_PATH}")

if __name__ == "__main__":
    main()