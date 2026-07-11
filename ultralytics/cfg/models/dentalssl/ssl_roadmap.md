# Lộ Trình Triển Khai 4 Phương Án SSL Vào YOLO26 (SSL1, SSL2, SSL3, SSL4)

Tài liệu này đề xuất lộ trình và kế hoạch chi tiết để tích hợp và hiện thực hóa 4 phương pháp tự giám sát (SSL) hàng đầu hiện nay vào dòng mô hình YOLO26 của dự án.

---

## 1. Bản Đồ Tổng Quan 4 Phương Án

| Tên Phiên Bản | Phương Pháp SSL | Cơ Chế Học | Lợi Thế Cho Ảnh Răng (X-quang) | SOTA Level |
| :--- | :--- | :--- | :--- | :---: |
| **YOLO26-SSL1** | **Feature Reconstruction** (Tái tạo đặc trưng biên) | Tái tạo Sobel/HOG thay vì pixel thô. | Tránh nhiễu hạt phim, tập trung hình dáng cấu trúc. | ⭐⭐⭐ |
| **YOLO26-SSL2** | **Spark CNN Pretraining** (Autoencoder thưa) | Tích chập thưa (Sparse Conv) trên vùng unmasked. | CNN thuần học MIM không bị loãng đặc trưng. | ⭐⭐⭐⭐ |
| **YOLO26-SSL3** | **DINOv2 Distillation** (Học đối sánh) | Student YOLO26 khớp feature Teacher DINOv2. | Thừa kế không gian ngữ nghĩa ViT cực mạnh. | ⭐⭐⭐⭐ |
| **YOLO26-SSL4** | **I-JEPA Latent Prediction** (Dự đoán tiềm ẩn) | Dự đoán biểu diễn trừu tượng trong không gian ẩn (latent space), không tái tạo pixel. | **SOTA tuyệt đối** — học ngữ nghĩa cấp cao, bỏ qua hoàn toàn nhiễu pixel. | ⭐⭐⭐⭐⭐ |

---

## 2. Kế Hoạch Chi Tiết & Lộ Trình Triển Khai

### 🚀 Giai Đoạn 1: YOLO26-SSL1 (Feature-level SimMIM)
*Mục tiêu: Cải tiến trực tiếp từ cấu hình `yolo26ssl.yaml` hiện tại để tăng độ nhạy biên dạng.*

1. **Tạo file cấu hình YAML mới:**
   * Tạo [yolo26ssl1.yaml](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/cfg/models/dentalssl/yolo26ssl1.yaml) kế thừa nguyên bản `yolo26ssl.yaml` nhưng cấu hình decoder xuất ra đặc trưng kích thước tương đương với HOG descriptor hoặc Sobel map của ảnh gốc.
2. **Cập nhật Module Tái cấu trúc:**
   * Cập nhật loss function trong [dental_ssl_modules.py](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/nn/modules/dental_ssl_modules.py).
   * Thay thế việc tính loss trực tiếp trên pixel bằng cách:
     1. Chuyển ảnh gốc thành ảnh biên dạng (Sobel/Canny) hoặc trích xuất HOG trên GPU.
     2. Ép Decoder dự đoán trực tiếp ảnh biên dạng hoặc HOG này.
3. **Ưu điểm:** Độ ổn định cực cao khi hội tụ, mô hình tập trung 100% vào việc tìm hình khối cấu trúc răng.

---

### 🚀 Giai Đoạn 2: YOLO26-SSL2 (Tích hợp Spark / Sparse Conv)
*Mục tiêu: Đưa tích chập thưa vào YOLO để đạt hiệu năng MIM tối ưu nhất cho mạng tích chập.*

1. **Tạo file cấu hình YAML mới:**
   * Tạo [yolo26ssl2.yaml](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/cfg/models/dentalssl/yolo26ssl2.yaml).
   * Thay thế các lớp `Conv` mặc định ở Backbone bằng lớp `SparseConv` hoặc `MaskAwareConv`.
2. **Triển khai Code:**
   * Viết custom module `MaskAwareConv2d` trong `ultralytics/nn/modules/conv.py`. Lớp này sẽ nhận một nhãn vị trí mặt nạ (mask) đi kèm với Tensor ảnh để bỏ qua các vùng bị che (chỉ tích chập trên vùng có dữ liệu thực).
   * Cập nhật `parse_model()` trong `tasks.py` để xử lý đầu vào nhãn mặt nạ đi kèm.
3. **Ưu điểm:** Đưa hiệu quả học tự giám sát của mạng CNN tiệm cận với Vision Transformer (ViT).

---

### 🚀 Giai Đoạn 3: YOLO26-SSL3 (Chuyển giao tri thức từ DINOv2)
*Mục tiêu: Căn chỉnh biểu diễn đặc trưng của YOLO26 theo không gian ngữ nghĩa mạnh mẽ của DINOv2.*

1. **Tạo file cấu hình YAML mới:**
   * Tạo [yolo26ssl3.yaml](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/cfg/models/dentalssl/yolo26ssl3.yaml).
   * Cấu hình nhánh đầu ra của P3, P4, P5 sẽ đi qua các đầu chiếu (Projection Heads) để đưa về cùng chiều kích thước (channel dimension) với đặc trưng của DINOv2.
2. **Triển khai Code:**
   * Load mô hình DINOv2 đã pre-trained của Meta (`dinov2_vits14` hoặc `dinov2_vitb14`) làm nhánh giáo viên (Teacher) và đóng băng trọng số (không cập nhật gradient).
   * Viết hàm tính loss **Cosine Similarity Loss** (hoặc MSE Loss) giữa feature maps của YOLO26 (Student) và feature maps tương ứng của DINOv2 (Teacher).
3. **Ưu điểm:** YOLO26 sẽ học được cách nhìn và biểu diễn ngữ cảnh giống như một siêu mạng Transformer khổng lồ mà không làm tăng bất kỳ độ trễ nào lúc suy luận thực tế (Inference).

---

### 🚀 Giai Đoạn 4: YOLO26-SSL4 — I-JEPA Latent Prediction ⭐ (SOTA nhất hiện nay)
*Mục tiêu: Áp dụng phương pháp tiên tiến nhất (I-JEPA — Image-based Joint-Embedding Predictive Architecture, Meta AI / Yann LeCun, 2023-2026) để mô hình học ngữ nghĩa cấp cao trong không gian ẩn (latent space) mà hoàn toàn không cần tái tạo pixel.*

> [!IMPORTANT]
> **I-JEPA vượt trội MAE/SimMIM ở điểm gì?**
> - MAE/SimMIM (và SSL hiện tại của bạn): Che ảnh → tái tạo lại pixel gốc. Mô hình bị ép học các chi tiết thừa (nhiễu hạt phim, độ sáng không đều) → lãng phí capacity.
> - **I-JEPA:** Che ảnh → dự đoán **biểu diễn trừu tượng** (abstract representation) của vùng bị che trong không gian ẩn. Mô hình chỉ học **ngữ nghĩa cấp cao** (hình dáng răng, cấu trúc xương hàm, ranh giới mô mềm) → hiệu quả hơn rất nhiều cho downstream detection.

#### Kiến trúc I-JEPA cho YOLO26:

```text
                    Ảnh X-quang gốc
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
     [Context Encoder]      [Target Encoder (EMA)]
     (YOLO26 backbone)      (YOLO26 backbone copy)
     Nhận ảnh bị che         Nhận ảnh đầy đủ
              │                     │
              ▼                     ▼
     Feature P3/P4/P5      Feature P3/P4/P5
     (vùng visible)         (toàn bộ ảnh)
              │                     │
              ▼                     │
       [Predictor MLP]             │
       Dự đoán latent              │
       của vùng masked             │
              │                     │
              └──────── Loss ───────┘
                  Smooth-L1 Loss
                  trên latent space
```

#### Kế hoạch triển khai chi tiết:

1. **Tạo file YAML mới:**
   * Tạo [yolo26ssl4.yaml](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/cfg/models/dentalssl/yolo26ssl4.yaml).
   * YAML chỉ chứa backbone gốc YOLO26 (layers 0-10) KHÔNG có decoder tái tạo pixel.
   * Thêm một nhánh `LatentPredictor` gọn nhẹ (MLP 2-3 layers) ở cuối P5 để dự đoán latent.

2. **Tạo module mới:**
   * Viết file [jepa_ssl_modules.py](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/nn/modules/jepa_ssl_modules.py) chứa:

     **a) `LatentPredictor`** — MLP dự đoán biểu diễn ẩn:
     ```python
     class LatentPredictor(nn.Module):
         """Lightweight MLP that predicts target latent from context features."""
         def __init__(self, in_dim, hidden_dim=1024, out_dim=256, num_layers=2):
             # Linear → GELU → Linear → GELU → Linear
             # Nhận feature từ context encoder, dự đoán latent của target
     ```

     **b) `EMATargetEncoder`** — Bản sao momentum của backbone:
     ```python
     class EMATargetEncoder:
         """Exponential Moving Average copy of the context encoder."""
         def __init__(self, context_encoder, momentum=0.996):
             self.target = deepcopy(context_encoder)
             # Đóng băng hoàn toàn, không tính gradient
             # Cập nhật trọng số qua EMA: target = m * target + (1-m) * context

         @torch.no_grad()
         def update(self, context_encoder, momentum):
             for t_p, c_p in zip(self.target.parameters(), context_encoder.parameters()):
                 t_p.data.mul_(momentum).add_(c_p.data, alpha=1 - momentum)
     ```

     **c) `JEPALoss`** — Loss dự đoán latent:
     ```python
     class JEPALoss(nn.Module):
         """Smooth-L1 loss in latent space between predicted and target representations."""
         # Chỉ tính loss trên các patch bị che
         # KHÔNG sử dụng pixel-level loss (L1/SSIM)
     ```

3. **Sửa đổi trainer:**
   * Tạo file [jepa_train.py](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/models/yolo/dental_ssl/jepa_train.py) kế thừa logic từ `train.py` hiện tại nhưng:
     * Tạo 2 nhánh encoder: **Context Encoder** (trainable) + **Target Encoder** (EMA, frozen).
     * Masking strategy: Che nhiều vùng lớn liên tục (multi-block masking) thay vì random patches.
     * Sau mỗi step, cập nhật EMA: `target_params = 0.996 * target_params + 0.004 * context_params`.

4. **Điểm mấu chốt cho ảnh X-quang răng:**
   * **Multi-block masking đặc biệt:** Che từng vùng hàm (ví dụ: che toàn bộ vùng răng hàm dưới bên trái) → ép mô hình suy luận cấu trúc giải phẫu hoàn chỉnh từ ngữ cảnh xung quanh.
   * **Không có decoder nặng:** Tiết kiệm rất nhiều VRAM so với pixel reconstruction. Phù hợp train trên GPU consumer (RTX 3060/4060).
   * **Downstream transfer tốt nhất:** Các nghiên cứu gốc I-JEPA cho thấy vượt MAE/SimMIM 2-4% trên ImageNet linear probing. Hiệu ứng còn lớn hơn trên domain-specific data (như ảnh y khoa).

---

## 3. So Sánh 4 Phương Án

| Tiêu chí | SSL1 (Feature Recon) | SSL2 (Spark) | SSL3 (DINOv2 Distill) | SSL4 (I-JEPA) ⭐ |
| :--- | :---: | :---: | :---: | :---: |
| **Chất lượng biểu diễn** | Tốt | Rất tốt | Rất tốt | **Xuất sắc** |
| **VRAM cần thiết** | Trung bình | Trung bình | Cao (cần load DINOv2) | **Thấp** (không decoder) |
| **Độ phức tạp code** | Thấp | Cao (sparse conv) | Trung bình | Trung bình |
| **Phụ thuộc thư viện** | Không | `spconv` hoặc custom | `torch.hub` (DINOv2) | **Không** |
| **Tốc độ huấn luyện** | Nhanh | Trung bình | Chậm (teacher forward) | **Nhanh** |
| **SOTA ranking (2026)** | 3/4 | 2/4 | 2/4 | **1/4** |

> [!TIP]
> **Gợi ý chiến lược cho DentalYOLO:** Nên triển khai **SSL4 (I-JEPA)** trước vì nó vừa SOTA nhất, vừa nhẹ VRAM nhất, không phụ thuộc thư viện ngoài, và cực kỳ phù hợp cho domain X-quang nha khoa (nơi pixel-level detail không quan trọng bằng cấu trúc ngữ nghĩa).

---

## 4. Kế Hoạch Đánh Giá & So Sánh (Validation)

Sau khi hoàn thành huấn luyện SSL cho cả 4 phiên bản, lộ trình đánh giá bao gồm:
1. Trích xuất trọng số của Backbone từ 4 mô hình tiền huấn luyện.
2. Nạp vào mô hình phát hiện [dental-yolo26_v15.yaml](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/cfg/models/dental26/dental-yolo26_v15.yaml).
3. Huấn luyện với tập dữ liệu X-quang răng có nhãn thực tế để vẽ biểu đồ so sánh:
   * **mAP50 / mAP50-95** đạt được.
   * **Tốc độ hội tụ** (số epoch cần thiết để đạt độ chính xác mong muốn).
   * **VRAM peak** khi training.
