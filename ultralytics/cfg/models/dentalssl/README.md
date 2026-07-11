# Self-Supervised Learning (SSL) Cấu Hình & Lộ Trình (YOLO26-SSL)

Thư mục này chứa các file cấu hình YAML dùng cho quá trình tiền huấn luyện tự giám sát (Self-Supervised Learning - SSL) cho dòng mô hình YOLO26 của dự án **DentalYOLO**.

---

## 1. Bảng Cấu Hình & Thông Số Kích Thước (Đầu vào: 640x640)

| Tên Phiên Bản | File Config | Số Lượng Params | Độ Phức Tạp (GFLOPs) | Kiểu Kiến Trúc | Đặc Tính Cho Ảnh Răng |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **YOLO26-Base-SSL** | [yolo26ssl.yaml](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/cfg/models/dentalssl/yolo26ssl.yaml) | 9.82 M (S) / 19.83 M (M) | 42.7 (S) | SimMIM Pixel Recon | Tái tạo trực tiếp ảnh thô bị che |
| **YOLO26-SSL1** | [yolo26ssl1.yaml](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/cfg/models/dentalssl/yolo26ssl1.yaml) | 9.82 M (S) | 42.7 (S) | Sobel Feature Recon | Tái tạo bản đồ biên Sobel (x, y) |
| **YOLO26-SSL2** | [yolo26ssl2.yaml](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/cfg/models/dentalssl/yolo26ssl2.yaml) | 9.82 M (S) | 42.7 (S) | Spark CNN MIM | Dùng tích chập thưa (MaskAwareConv) |
| **YOLO26-SSL3** | [yolo26ssl3.yaml](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/cfg/models/dentalssl/yolo26ssl3.yaml) | 9.36 M (S) | 20.1 (S) | DINOv2 Distillation | Chuyển giao tri thức ViT từ DINOv2 |
| **YOLO26-SSL4** | [yolo26ssl4.yaml](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/cfg/models/dentalssl/yolo26ssl4.yaml) | 9.44 M (S) | 20.8 (S) | **I-JEPA Latent Predictor** | Dự đoán trong không gian ẩn (SOTA) |

---

## 2. Mô Tả Chi Tiết 4 Phương Án SSL

### 🚀 Phương Án 1 (YOLO26-SSL1) — Feature-level SimMIM (Sobel Targets)
* **Khái niệm:** Thay vì ép mô hình tái tạo từng pixel thô của ảnh X-quang răng bị che (dễ bị nhiễu hạt phim), SSL1 bắt mô hình tái tạo **bản đồ biên cạnh Sobel** (bao gồm gradient ngang và dọc).
* **Hoạt động:** Sử dụng `SobelEdgeExtractor` trên GPU để sinh nhãn biên thời gian thực và huấn luyện bằng `FeatureReconstructionLoss` (L1 + SSIM tính trên bản đồ biên).

### 🚀 Phương Án 2 (YOLO26-SSL2) — Spark-style Partial Convolutions
* **Khái niệm:** Mạng tích chập (CNN) thông thường gặp hạn chế khi tính toán trên ảnh bị che vì các pixel trống (bằng 0) sẽ làm loãng đặc trưng ở các tầng tiếp theo. 
* **Hoạt động:** Thay thế các tầng `Conv` downsample ở backbone bằng `MaskAwareConv`. Lớp này sẽ thực hiện tích chập kèm theo cơ chế bù trừ tỷ lệ vùng bị che (Partial Conv) và tự động downsample nhãn mask thông qua `MaskRegistry` toàn cục.

### 🚀 Phương Án 3 (YOLO26-SSL3) — DINOv2 Knowledge Distillation
* **Khái niệm:** Học hỏi không gian ngữ nghĩa siêu mạnh của mô hình tự giám sát Vision Transformer (ViT) từ Meta AI.
* **Hoạt động:** YOLO26 đóng vai trò là Student, đi qua đầu chiếu `DINOv2DistillHead` để đưa đặc trưng P3, P4, P5 về 384 kênh. Teacher là mô hình DINOv2 (`vit_small_patch14_dinov2` load trực tiếp qua thư viện `timm`). Huấn luyện bằng `DistillationLoss` dựa trên Cosine Similarity.

### 🚀 Phương Án 4 (YOLO26-SSL4) — I-JEPA Latent Prediction (SOTA nhất)
* **Khái niệm:** Áp dụng kiến trúc I-JEPA của Yann LeCun. Thay vì cố tái cấu trúc hoặc so sánh trực tiếp, phương pháp này bắt mô hình dự đoán biểu diễn ngữ nghĩa trừu tượng trong không gian ẩn (latent space).
* **Hoạt động:** Gồm 2 nhánh backbone: **Context Encoder** (trainable) nhận ảnh che và **Target Encoder** (EMA frozen) nhận ảnh đầy đủ. Đầu chiếu predictor `LatentPredictor` dự đoán latent 256 chiều của vùng bị che và so khớp bằng `JEPALoss` (Smooth-L1 loss trên không gian ẩn).

---

## 3. Bản Đồ Source Code & Vị Trí Triển Khai

1. **Kiến trúc mô hình YAML:** Nằm trực tiếp tại thư mục này (`ultralytics/cfg/models/dentalssl/`).
2. **Loss & Module của SSL1 & SSL3:** [dental_ssl_modules.py](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/nn/modules/dental_ssl_modules.py).
3. **Module của SSL2 (Tích chập thưa):** `MaskAwareConv` và `MaskRegistry` tại [conv.py](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/nn/modules/conv.py).
4. **Module của SSL4 (JEPA):** [jepa_ssl_modules.py](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/nn/modules/jepa_ssl_modules.py).
5. **Đăng ký module:** Trình biên dịch YOLO nhận diện các lớp này tại [__init__.py](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/nn/modules/__init__.py) và [tasks.py](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/nn/tasks.py).
6. **Huấn luyện động:** Loop huấn luyện chính nằm tại [train.py](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/models/yolo/dental_ssl/train.py) (Tự động nhận cấu hình YAML để chuyển đổi Loss và luồng cập nhật trọng số EMA tương thích).

---

## 4. Hướng Dẫn Chạy Huấn Luyện

Chạy script huấn luyện SSL thông qua Python CLI từ thư mục gốc của dự án:

```bash
# 1. Chạy huấn luyện SSL mặc định (SimMIM pixel-level)
python -m ultralytics.models.yolo.dental_ssl.train \
    --model ultralytics/cfg/models/dentalssl/yolo26ssl.yaml \
    --data /path/to/opg_images --epochs 100 --imgsz 640 --batch 8 --device 0

# 2. Chạy huấn luyện SSL1 (Sobel Edge Reconstruction)
python -m ultralytics.models.yolo.dental_ssl.train \
    --model ultralytics/cfg/models/dentalssl/yolo26ssl1.yaml \
    --data /path/to/opg_images --epochs 100 --imgsz 640 --batch 8 --device 0

# 3. Chạy huấn luyện SSL2 (Spark MaskAwareConv)
python -m ultralytics.models.yolo.dental_ssl.train \
    --model ultralytics/cfg/models/dentalssl/yolo26ssl2.yaml \
    --data /path/to/opg_images --epochs 100 --imgsz 640 --batch 8 --device 0

# 4. Chạy huấn luyện SSL3 (DINOv2 Distillation)
python -m ultralytics.models.yolo.dental_ssl.train \
    --model ultralytics/cfg/models/dentalssl/yolo26ssl3.yaml \
    --data /path/to/opg_images --epochs 100 --imgsz 640 --batch 8 --device 0

# 5. Chạy huấn luyện SSL4 (I-JEPA Latent Predictor)
python -m ultralytics.models.yolo.dental_ssl.train \
    --model ultralytics/cfg/models/dentalssl/yolo26ssl4.yaml \
    --data /path/to/opg_images --epochs 100 --imgsz 640 --batch 8 --device 0
```
