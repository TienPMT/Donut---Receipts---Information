import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from model_config import setup_model_and_processor, DonutDataset, collate_fn

warnings.filterwarnings("ignore")

def train():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = setup_model_and_processor(device)

    train_dataset = DonutDataset(
        r"d:\data\HUIT\Nam3\HK2\Deep Learning\TaiLieuThayBao\Project\Donut\Version2\Train_data\train_metadata_clean.jsonl",
        processor, split="train"
    )
    val_dataset = DonutDataset(
        r"d:\data\HUIT\Nam3\HK2\Deep Learning\TaiLieuThayBao\Project\Donut\Version2\Train_data\val_metadata_clean.jsonl",
        processor, split="val"
    )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "donut_checkpoints")
    RESULT_DIR = os.path.join(BASE_DIR, "donut_result")

    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=50,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=5,
        bf16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        report_to="none",
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )

    print(f"Bắt đầu Training Version 2... (Lưu tại: {CHECKPOINT_DIR})")
    trainer.train()

    model.save_pretrained(RESULT_DIR)
    processor.save_pretrained(RESULT_DIR)
    print(f"Đã lưu mô hình cuối cùng tại {RESULT_DIR}")

if __name__ == "__main__":
    train()
