# Dự án Donut - Trích xuất thông tin tài liệu

Dự án này sử dụng mô hình mã nguồn mở Donut (Document Understanding Transformer) để thực hiện bài toán nhận dạng và trích xuất thông tin (KIE - Key Information Extraction) từ hình ảnh hóa đơn / chứng từ (dựa trên bộ dữ liệu MCOCR).

## Cấu trúc Source Code

Dự án được chia thành nhiều phiên bản (Version) để theo dõi quá trình thử nghiệm và cải tiến:

- **Version1/**, **Version2/**, **Version3/**: Chứa các kịch bản (script) mã nguồn Python bao gồm:
  + `train_model.py`, `model_config.py`: Các script phục vụ mục đích thiết lập và huấn luyện mô hình.
  + `evalute/test_model.py`, `real_world_test.py`: Kịch bản đánh giá mô hình trên tập validation hoặc dữ liệu thực tế.
  + Các công cụ xử lý và chuẩn bị dữ liệu (`split_data.py`, `visualize_aug.py`, ...).

**Lưu ý:** Repository này chỉ lưu trữ mã nguồn thao tác (`.py`). Toàn bộ các dữ liệu như hình ảnh, file cấu trúc `.jsonl`, tệp CSV, và trọng số của mô hình (các thư mục `donut_checkpoints`, `donut_result`, `data`, v.v.) đều đã được loại trừ thông qua `.gitignore`.

## Yêu cầu môi trường
- Yêu cầu cấu hình Python với các thư viện cần toán Deep Learning như PyTorch, Transformers (Hugging Face), Datasets.
- Để sử dụng, vui lòng tự tải và tổ chức bộ dữ liệu vào các thư mục `data/` hoặc `dataset/` tương ứng theo định dạng yêu cầu của script trước khi chạy huấn luyện hoặc kiểm thử.
