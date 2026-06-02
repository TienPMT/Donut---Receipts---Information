import torch
import gc
import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DonutProcessor, VisionEncoderDecoderModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from model_config import (
    check_gpu,
    DonutDataset,
    collate_fn,
    METADATA_TRAIN,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MODEL_NAME
)

PRETRAINED_PATH = MODEL_NAME
OUTPUT_DIR = "./donut_checkpoints"
FINAL_SAVE_PATH = "./donut_result"

def setup_model_for_training(model_path, device):
    """Load model lên để train từ đầu"""
    print(f"Loading model từ: {model_path}...")
    processor = DonutProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    model.config.encoder.image_size = [IMAGE_HEIGHT, IMAGE_WIDTH]
    processor.image_processor.size = {"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH}
    processor.image_processor.do_align_long_axis = False
    model.to(device)

    special_tokens = ["<s_seller>", "</s_seller>", "<s_address>", "</s_address>", "<s_timestamp>", "</s_timestamp>", "<s_total>", "</s_total>"]
    vietnamese_tokens = ["à", "á", "ạ", "ả", "ã", "â", "ầ", "ấ", "ậ", "ẩ", "ẫ", "ă", "ằ", "ắ", "ặ", "ẳ", "ẵ", "è", "é", "ẹ", "ẻ", "ẽ", "ê", "ề", "ế", "ệ", "ể", "ễ", "ì", "í", "ị", "ỉ", "ĩ", "ò", "ó", "ọ", "ỏ", "õ", "ô", "ồ", "ố", "ộ", "ổ", "ỗ", "ơ", "ờ", "ớ", "ợ", "ở", "ỡ", "ù", "ú", "ụ", "ủ", "ũ", "ư", "ừ", "ứ", "ự", "ử", "ữ", "ỳ", "ý", "ỵ", "ỷ", "ỹ", "đ", "À", "Á", "Ạ", "Ả", "Ã", "Â", "Ầ", "Ấ", "Ậ", "Ẩ", "Ẫ", "Ă", "Ằ", "Ắ", "Ặ", "Ẳ", "Ẵ", "È", "É", "Ẹ", "Ẻ", "Ẽ", "Ê", "Ề", "Ế", "Ệ", "Ể", "Ễ", "Ì", "Í", "Ị", "Ỉ", "Ĩ", "Ò", "Ó", "Ọ", "Ỏ", "Õ", "Ô", "Ồ", "Ố", "Ộ", "Ổ", "Ỗ", "Ơ", "Ờ", "Ớ", "Ợ", "Ở", "Ỡ", "Ù", "Ú", "Ụ", "Ủ", "Ũ", "Ư", "Ừ", "Ứ", "Ự", "Ử", "Ữ", "Ỳ", "Ý", "Ỵ", "Ỷ", "Ỹ", "Đ"]

    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    processor.tokenizer.add_tokens(vietnamese_tokens)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_seller>")
    model.config.decoder_start_token_id = start_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    print("-> Model đã sẵn sàng để train!")
    return model, processor

def main():

    device = check_gpu()
    torch.cuda.empty_cache()
    gc.collect()

    model, processor = setup_model_for_training(PRETRAINED_PATH, device)

    train_dataset = DonutDataset(
        dataset_path="data/train_images",
        metadata_path=METADATA_TRAIN,
        processor=processor,
        max_length=512
    )

    print(f"Tổng số ảnh train: {len(train_dataset)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        dataloader_num_workers=4,
        num_train_epochs=50,
        learning_rate=2e-5,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=3000,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="tensorboard",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    print("\n>>> BẮT ĐẦU QUÁ TRÌNH TRAIN TỪ ĐẦU (FROM SCRATCH)...")
    try:
        trainer.train()
    except Exception as e:
        print(f"\n[LỖI]: {e}")
        return

    print("\n>>> ĐANG LƯU MODEL MỚI...")
    trainer.save_model(FINAL_SAVE_PATH)
    processor.save_pretrained(FINAL_SAVE_PATH)
    print(f"Hoàn tất! Model hoàn chỉnh đã lưu tại: {FINAL_SAVE_PATH}")

if __name__ == "__main__":
    main()
