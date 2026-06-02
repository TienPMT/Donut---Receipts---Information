import os
import torch
import warnings
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from model_config import setup_model_and_processor, DonutDataset, collate_fn

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

def train():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = setup_model_and_processor(device)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = r"D:\data\HUIT\Nam3\HK2\Deep Learning\TaiLieuThayBao\Project\Donut\Version3\data1"

    train_dataset = DonutDataset(
        os.path.join(DATA_DIR, "train_metadata.jsonl"),
        processor, max_length=768, split="train"
    )
    val_dataset = DonutDataset(
        os.path.join(DATA_DIR, "validation_metadata.jsonl"),
        processor, max_length=768, split="val"
    )

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
        generation_max_length=768,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print(f"Bắt đầu Training Version 3... (Lưu tại: {CHECKPOINT_DIR})")
    trainer.train()

    model.save_pretrained(RESULT_DIR)
    processor.save_pretrained(RESULT_DIR)
    print(f"Đã lưu mô hình cuối cùng tại {RESULT_DIR}")

if __name__ == "__main__":
    train()
