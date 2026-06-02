import argparse
import json
import os
import re
import unicodedata

V_CHARS = "aáàảãạâấầẩẫậăắằẳẵặeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵđ"
V_CHARS += V_CHARS.upper()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFC", text)

    text = re.sub(rf"([{V_CHARS}])\s+([a-z{V_CHARS.lower()}])", r"\1\2", text)

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_ground_truth(gt_obj: dict) -> dict:
    if "gt_parse" in gt_obj and isinstance(gt_obj["gt_parse"], dict):
        for k, v in gt_obj["gt_parse"].items():
            if isinstance(v, str):
                gt_obj["gt_parse"][k] = clean_text(v)
    elif "seller" in gt_obj and isinstance(gt_obj["seller"], dict):
        for k, v in gt_obj["seller"].items():
            if isinstance(v, str):
                gt_obj["seller"][k] = clean_text(v)
    else:
        for k, v in gt_obj.items():
            if isinstance(v, str):
                gt_obj[k] = clean_text(v)
    return gt_obj

def process_file(input_path: str, output_path: str) -> None:
    total = 0
    changed = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            record = json.loads(line)
            gt_raw = record.get("ground_truth", "{}")
            try:
                gt_obj = json.loads(gt_raw)
            except json.JSONDecodeError:
                gt_obj = {}

            before = json.dumps(gt_obj, ensure_ascii=False)
            gt_obj = clean_ground_truth(gt_obj)
            after = json.dumps(gt_obj, ensure_ascii=False)

            if before != after:
                changed += 1

            record["ground_truth"] = after
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[INFO] {input_path} -> {output_path}")
    print(f"[INFO] total={total} changed={changed}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Vietnamese spacing in JSONL ground_truth.")
    parser.add_argument("--data_dir", default="Train_data", help="Folder containing JSONL files")
    parser.add_argument("--suffix", default="_clean", help="Suffix for output JSONL")
    parser.add_argument("--files", nargs="+", default=["train_metadata.jsonl", "val_metadata.jsonl"])
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, args.data_dir)

    for name in args.files:
        input_path = os.path.join(data_dir, name)
        if not os.path.exists(input_path):
            print(f"[WARN] Missing: {input_path}")
            continue

        root, ext = os.path.splitext(name)
        output_name = f"{root}{args.suffix}{ext}"
        output_path = os.path.join(data_dir, output_name)
        process_file(input_path, output_path)

if __name__ == "__main__":
    main()
