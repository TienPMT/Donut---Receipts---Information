import os
import json
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, AutoTokenizer
import re
from difflib import SequenceMatcher
from tqdm import tqdm

# --- CẤU HÌNH ---
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
    """
    Trích xuất giá trị số học từ một chuỗi bất kỳ.
    VD: "Tổng tiền : 125,000 đ" -> "125000"
    """
    # Tìm tất cả các cụm chứa số, phẩy và chấm
    matches = re.findall(r'[\d.,]+', str(text))
    if not matches:
        return str(text).strip()
        
    max_val = -1
    best_str = str(text).strip()
    
    for m in matches:
        m = m.rstrip('.,') # Cắt dấu chấm phẩy ở đuôi
        # Xử lý phần thập phân vô nghĩa .00 hoặc ,00
        if m.endswith('.00') or m.endswith(',00'):
            m = m[:-3]
            
        # Xóa các dấu phân cách hàng nghìn để quy về số nguyên
        clean_num = re.sub(r'[.,]', '', m)
        if clean_num.isdigit():
            val = int(clean_num)
            # Lấy con số lớn nhất trong câu (thường là tổng tiền)
            if val > max_val:
                max_val = val
                best_str = clean_num
                
    return best_str

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
        
        # =====================================================================
        # HẬU XỬ LÝ 3 BƯỚC (DẤU CÂU, DẤU THANH & ĐUÔI TỪ)
        # =====================================================================
        # B1: Tách dấu câu
        sequence = re.sub(r'([,.;])([A-ZĐa-zà-ỹ])', r'\1 \2', sequence)
        
        # B2: Tách 2 dấu thanh dính liền
        toned_vowels = "áắấéếíóốớúứýàằầèềìòồờùừỳảẳẩẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹạặậẹệịọộợụựỵ"
        toned_vowels += toned_vowels.upper()
        sequence = re.sub(r'(["' + toned_vowels + r'])(["' + toned_vowels + r'])', r'\1 \2', sequence)
        
        # B3: Nối các từ bị tách
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
        # =====================================================================
            
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

            # ========================================================
            # ĐÁNH GIÁ CHUYÊN SÂU DÀNH RIÊNG CHO TỔNG TIỀN (SỬ DỤNG HÀM MỚI)
            # ========================================================
            if metric_key == "total_cost":
                gt_money = extract_money_amount(gt_val)
                pred_money = extract_money_amount(pred_val)
                
                sim_score = compute_similarity(gt_money, pred_money)
                metrics[metric_key]["sim"] += sim_score
                
                # So sánh chính xác trên con số nguyên thủy
                if gt_money and pred_money and gt_money == pred_money:
                    metrics[metric_key]["exact"] += 1
                elif gt_val.lower() == pred_val.lower(): # Fallback đề phòng lỗi
                    metrics[metric_key]["exact"] += 1
                    
                if sim_score >= RELAXED_THRESHOLD:
                    metrics[metric_key]["relaxed"] += 1
            else:
                # Đánh giá tiêu chuẩn cho các trường còn lại
                sim_score = compute_similarity(gt_val, pred_val)
                metrics[metric_key]["sim"] += sim_score
                
                if gt_val.lower() == pred_val.lower():
                    metrics[metric_key]["exact"] += 1
                    
                if sim_score >= RELAXED_THRESHOLD:
                    metrics[metric_key]["relaxed"] += 1

    # --- IN KẾT QUẢ ---
    print("\n\n" + "="*70)
    print(f"{'BÁO CÁO KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (DONUT OCR-FREE)':^70}")
    print("="*70)
    # Đã sửa lại Match @ 80% cho chuẩn với biến RELAXED_THRESHOLD
    print(f"{'Trường thông tin':<16} | {'Exact Match':<13} | {'Match @ 80%':<13} | {'Similarity Score':<15}")
    print("-" * 70)
    
    total_exact = 0
    total_relaxed = 0
    total_sim = 0
    total_count = 0

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

if __name__ == "__main__":
    plot_loss_curve()
    run_evaluation()