import os
import json
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, AutoTokenizer
import re
from difflib import SequenceMatcher
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "donut_result")
VAL_METADATA = os.path.join(BASE_DIR, "data1", "validation_metadata.jsonl")
VAL_IMG_DIR = os.path.join(BASE_DIR, "data1", "validation")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_HEIGHT = 1280
IMAGE_WIDTH = 960
RELAXED_THRESHOLD = 0.80

def plot_loss_curve():
    print(f"\n{'='*50}")
    print("1. ĐANG VẼ BIỂU ĐỒ LOSS CURVE...")
    print(f"{'='*50}")

    state_path = os.path.join(MODEL_PATH, "trainer_state.json")
    if not os.path.exists(state_path):
        print(f"[Cảnh báo] Không tìm thấy file {state_path} để vẽ biểu đồ.")
        return

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])

    train_epochs, train_loss = [], []
    eval_epochs, eval_loss = [], []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            train_epochs.append(entry["epoch"])
            train_loss.append(entry["loss"])
        elif "eval_loss" in entry and "epoch" in entry:
            eval_epochs.append(entry["epoch"])
            eval_loss.append(entry["eval_loss"])

    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_loss, label='Training Loss (Huấn luyện)', color='blue', linewidth=2)
    if eval_epochs:
        plt.plot(eval_epochs, eval_loss, label='Validation Loss (Kiểm định)', color='red', marker='o', linewidth=2)

    plt.title('Biểu đồ Hội tụ của mô hình Donut (Loss Curve)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    if eval_loss:
        min_loss_idx = eval_loss.index(min(eval_loss))
        plt.annotate(f'Best Model\nEpoch {eval_epochs[min_loss_idx]:.2f}',
                     xy=(eval_epochs[min_loss_idx], eval_loss[min_loss_idx]),
                     xytext=(eval_epochs[min_loss_idx], eval_loss[min_loss_idx] + 0.5),
                     arrowprops=dict(facecolor='green', shrink=0.05),
                     fontsize=10, color='green', fontweight='bold')

    save_path = os.path.join(BASE_DIR, "loss_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"-> Đã lưu biểu đồ thành công tại: {save_path}")
    plt.close()

def compute_similarity(gt, pred):
    """Tính độ tương đồng giữa 2 chuỗi (từ 0.0 đến 1.0)"""
    gt_str = str(gt).strip().lower()
    pred_str = str(pred).strip().lower()
    if not gt_str and not pred_str:
        return 1.0
    return SequenceMatcher(None, gt_str, pred_str).ratio()

def extract_money_amount(text):
    """Trích xuất giá trị số học từ một chuỗi bất kỳ."""
    matches = re.findall(r'[\d.,]+', str(text))
    if not matches:
        return str(text).strip()

    max_val = -1
    best_str = str(text).strip()

    for m in matches:
        m = m.rstrip('.,')
        if m.endswith('.00') or m.endswith(',00'):
            m = m[:-3]

        clean_num = re.sub(r'[.,]', '', m)
        if clean_num.isdigit():
            val = int(clean_num)
            if val > max_val:
                max_val = val
                best_str = clean_num

    return best_str

def plot_confusion_matrix(confusion_counts):
    """Vẽ ma trận nhầm lẫn phân phối kết quả trích xuất dạng Heatmap phẳng chuẩn học thuật"""
    fields = ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]
    categories = ["Exact Match", "Relaxed Match", "Incorrect", "Missing"]

    matrix_data = [confusion_counts[f.lower()] for f in fields]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix_data, cmap="Blues", aspect="auto")

    plt.colorbar(im, ax=ax, label="Số lượng mẫu hóa đơn (Tần suất)")

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels(fields, fontsize=10, fontweight='bold')

    max_val = max(max(row) for row in matrix_data) if matrix_data else 1
    for i in range(len(fields)):
        for j in range(len(categories)):
            val = matrix_data[i][j]
            color = "white" if val > (max_val / 2) else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold", fontsize=12)

    ax.set_title("Ma Trận Phân Phối Trạng Thái Trích Xuất Hệ Thống (Confusion Matrix)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, "donut_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"-> Đã xuất và lưu Ma trận nhầm lẫn tại: {save_path}")
    plt.close()

def plot_performance_bars(metrics):
    """Vẽ biểu đồ dạng cột nhóm so sánh hiệu năng trực quan giữa các trường thông tin"""
    fields = ["seller", "address", "timestamp", "total_cost"]
    display_fields = ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]

    exacts, relaxeds, sims = [], [], []
    for f in fields:
        count = metrics[f]["count"]
        if count > 0:
            exacts.append((metrics[f]["exact"] / count) * 100)
            relaxeds.append((metrics[f]["relaxed"] / count) * 100)
            sims.append((metrics[f]["sim"] / count) * 100)
        else:
            exacts.append(0); relaxeds.append(0); sims.append(0)

    x = range(len(fields))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    rects1 = ax.bar([i - width for i in x], exacts, width, label='Exact Match (Khớp tuyệt đối)', color='#1f77b4')
    rects2 = ax.bar(x, relaxeds, width, label='Relaxed Match (Khớp nới lỏng @80%)', color='#ff7f0e')
    rects3 = ax.bar([i + width for i in x], sims, width, label='Similarity Score (Độ tương đồng)', color='#2ca02c')

    ax.set_ylabel('Tỷ lệ phần trăm (%)', fontsize=12, fontweight='bold')
    ax.set_title('Biểu đồ Hiệu năng Chi tiết theo từng Trường Thông tin (DONUT)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(display_fields, fontsize=10, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1); autolabel(rects2); autolabel(rects3)
    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, "donut_performance_bars.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"-> Đã xuất và lưu Biểu đồ hiệu năng dạng cột tại: {save_path}")
    plt.close()

def run_evaluation():
    print(f"\n{'='*50}")
    print("2. ĐANG ĐÁNH GIÁ CHỈ SỐ MÔ HÌNH TRÊN TẬP VALIDATION...")
    print(f"{'='*50}")

    if not os.path.exists(MODEL_PATH):
        print(f"Chưa có model tại {MODEL_PATH}.")
        return

    processor = DonutProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    with open(VAL_METADATA, "r", encoding="utf-8") as f:
        val_samples = [json.loads(line) for line in f]

    metrics = {
        "seller": {"exact": 0, "relaxed": 0, "sim": 0.0, "count": 0},
        "address": {"exact": 0, "relaxed": 0, "sim": 0.0, "count": 0},
        "timestamp": {"exact": 0, "relaxed": 0, "sim": 0.0, "count": 0},
        "total_cost": {"exact": 0, "relaxed": 0, "sim": 0.0, "count": 0}
    }

    confusion_counts = {
        "seller": [0, 0, 0, 0],
        "address": [0, 0, 0, 0],
        "timestamp": [0, 0, 0, 0],
        "total_cost": [0, 0, 0, 0]
    }

    start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_seller>")

    for sample in tqdm(val_samples, desc="Đang đánh giá"):
        img_path = os.path.join(VAL_IMG_DIR, sample["file_name"])
        if not os.path.exists(img_path): continue

        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt", do_resize=True,
                                 size={"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH},
                                 do_align_long_axis=False).pixel_values.to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values, max_length=768,
                decoder_input_ids=torch.tensor([[start_token_id]]).to(DEVICE),
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
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
            pred_dict = processor.token2json(sequence)
            if "seller" not in pred_dict:
                pred_dict = {"seller": pred_dict}
        except Exception:
            pred_dict = {"seller": {}}

        preds = pred_dict.get("seller", {})
        if not isinstance(preds, dict): preds = {}

        gt_data = json.loads(sample["ground_truth"]).get("gt_parse", {})

        field_mapping = [
            ("seller", ["SELLER", "seller", "seller_name"]),
            ("address", ["ADDRESS", "address"]),
            ("timestamp", ["TIMESTAMP", "timestamp"]),
            ("total_cost", ["TOTAL_COST", "total_cost"])
        ]

        for metric_key, gt_keys in field_mapping:
            gt_val = ""
            for k in gt_keys:
                if k in gt_data:
                    gt_val = gt_data[k]
                    break

            pred_val = ""
            for k in gt_keys:
                if k in preds:
                    pred_val = preds[k]
                    break

            if isinstance(pred_val, dict): pred_val = str(pred_val)
            if isinstance(gt_val, dict): gt_val = str(gt_val)

            gt_val = str(gt_val).strip()
            pred_val = str(pred_val).strip()

            if gt_val == "": continue
            metrics[metric_key]["count"] += 1

            is_exact = False
            if metric_key == "total_cost":
                gt_money = extract_money_amount(gt_val)
                pred_money = extract_money_amount(pred_val)
                sim_score = compute_similarity(gt_money, pred_money)

                if gt_money and pred_money and gt_money == pred_money:
                    is_exact = True
                elif gt_val.lower() == pred_val.lower():
                    is_exact = True
            else:
                sim_score = compute_similarity(gt_val, pred_val)
                if gt_val.lower() == pred_val.lower():
                    is_exact = True

            metrics[metric_key]["sim"] += sim_score

            if pred_val == "":
                confusion_counts[metric_key][3] += 1
            elif is_exact:
                confusion_counts[metric_key][0] += 1
                metrics[metric_key]["exact"] += 1
                metrics[metric_key]["relaxed"] += 1
            elif sim_score >= RELAXED_THRESHOLD:
                confusion_counts[metric_key][1] += 1
                metrics[metric_key]["relaxed"] += 1
            else:
                confusion_counts[metric_key][2] += 1

    print("\n\n" + "="*70)
    print(f"{'BÁO CÁO KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (DONUT OCR-FREE)':^70}")
    print("="*70)
    print(f"{'Trường thông tin':<16} | {'Exact Match':<13} | {'Match @ 80%':<13} | {'Similarity Score':<15}")
    print("-" * 70)

    total_exact, total_relaxed, total_sim, total_count = 0, 0, 0, 0

    for key, data in metrics.items():
        count = data["count"]
        if count == 0: continue

        exact_match = (data["exact"] / count) * 100
        relaxed_match = (data["relaxed"] / count) * 100
        avg_sim = (data["sim"] / count) * 100

        total_exact += data["exact"]
        total_relaxed += data["relaxed"]
        total_sim += data["sim"]
        total_count += count

        print(f"{key.upper():<16} | {exact_match:>11.2f}%  | {relaxed_match:>11.2f}%  | {avg_sim:>13.2f}%")

    print("-" * 70)
    overall_exact = (total_exact / total_count) * 100 if total_count > 0 else 0
    overall_relaxed = (total_relaxed / total_count) * 100 if total_count > 0 else 0
    overall_sim = (total_sim / total_count) * 100 if total_count > 0 else 0

    print(f"{'TRUNG BÌNH TOÀN CỤC':<16} | {overall_exact:>11.2f}%  | {overall_relaxed:>11.2f}%  | {overall_sim:>13.2f}%")
    print("="*70)

    plot_confusion_matrix(confusion_counts)
    plot_performance_bars(metrics)

if __name__ == "__main__":
    plot_loss_curve()
    run_evaluation()
