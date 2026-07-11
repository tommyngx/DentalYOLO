# Real-Time DEtection Transformer (RT-DETR) Model Architectures

Tài liệu này tổng hợp danh sách các cấu hình mô hình RT-DETR hiện có trong dự án, thông số kích thước (Parameters/FLOPs), và các cập nhật bổ sung gần đây.

---

## 1. Bảng thông số Parameters & FLOPs (Đầu vào: 640x640)

| Tên Mô Hình (Config) | Số lượng tham số (Parameters) | Độ phức tạp (GFLOPs) | Số Layer | Kiểu Backbone | Ghi chú |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **rtdetr-s.yaml** | 20,155,644 (~20.2M) | 60.3 | 431 | HGNetv2-S | Phiên bản siêu nhẹ cho thiết bị nhúng |
| **rtdetr-r18.yaml** | 23,832,816 (~23.8M) | 72.8 | 241 | ResNet18 Basic | Bản ResNet nhỏ nhất, tốc độ suy luận nhanh |
| **rtdetr-m.yaml** | 22,542,780 (~22.5M) | 69.7 | 431 | HGNetv2-M | Bản Medium chuẩn (nằm giữa S và L) |
| **rtdetr-l.yaml** | 32,970,476 (~33.0M) | 108.3 | 465 | HGNetv2-L | Phiên bản Large tiêu chuẩn gốc |
| **rtdetr-r34.yaml** | 37,715,556 (~37.7M) | 117.9 | 334 | ResNet34 Basic | Phiên bản ResNet mức độ Medium |
| **rtdetr-resnet50.yaml** | 42,925,132 (~42.9M) | 130.8 | 410 | ResNet50 BottleNeck | Phiên bản ResNet50 tiêu chuẩn gốc |
| **rtdetr-m2.yaml** | 44,959,900 (~45.0M) | 170.5 | 515 | HGNetv2-M2 | Bản trung bình lớn (nằm giữa L và X) |
| **rtdetr-resnet101.yaml** | 61,917,260 (~61.9M) | 191.8 | 546 | ResNet101 BottleNeck| Phiên bản ResNet101 tiêu chuẩn gốc |
| **rtdetr-x.yaml** | 67,467,852 (~67.5M) | 232.7 | 583 | HGNetv2-X | Phiên bản Extra Large gốc |
| **rtdetr-h.yaml** | 122,598,124 (~122.6M) | 486.6 | 683 | HGNetv2-H | Phiên bản Huge cho độ chính xác tối đa |

---

## 2. Chi Tiết Các Cập Nhật & Thay Đổi Mới

Chúng tôi đã bổ sung **6 phiên bản RT-DETR mới/hiệu chỉnh** (`s`, `m`, `m2`, `h`, `r18`, `r34`) bằng cách tối ưu hóa lại Backbone và cấu hình của Transformer Decoder để phân chia phân khúc mô hình tối ưu hơn cho các bài toán xử lý ảnh y tế (DentalYOLO).

### A. Tích hợp Module mới cho ResNet18 và ResNet34
* **Vấn đề của Code gốc:** Module `ResNetBlock` mặc định trong Ultralytics được xây dựng dưới dạng **BottleNeck** (độ giãn kênh `expansion = 4`, gồm 3 lớp conv). Tuy nhiên, các kiến trúc ResNet18/34 chính thống sử dụng block dạng **BasicBlock** (độ giãn kênh `expansion = 1`, gồm 2 lớp conv 3x3).
* **Giải pháp:** Đã phát triển thêm module độc lập [rtdetr_resnet_basic.py](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/nn/modules/rtdetr_resnet_basic.py) chứa `ResNetBasicBlock` và `ResNetBasicLayer` chuyên biệt cho ResNet18 và ResNet34.
* **Đăng ký module:** Hai module trên được import và xuất ra trong [__init__.py](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/nn/modules/__init__.py), đồng thời được thêm xử lý tính toán số kênh đầu ra (`c2 = args[1]`) trong bộ parser cấu trúc mạng của [tasks.py](file:///Users/francistommy/Desktop/BugHunter/Project/DentalYOLO/ultralytics/nn/tasks.py).

### B. Các Biến Thể Backbone HGNetv2 Mới
1. **rtdetr-s (Small):** Giảm số kênh ở lớp Stem (`[24, 32]`), giảm số kênh ở giữa & đầu ra của từng khối block, đồng thời giảm số block ở Stage 3 xuống còn 2 và hạ chiều ẩn (`hidden_dim`) của Decoder xuống 192.
2. **rtdetr-m (Medium):** Được tối ưu hóa có kích thước chuẩn nằm giữa bản S và bản L, sử dụng Stem (`[32, 40]`), giảm bớt số block ở Stage 3 xuống còn 2 và hạ chiều ẩn (`hidden_dim`) của Decoder xuống 224.
3. **rtdetr-m2 (Medium Large):** Bản trung bình lớn nằm giữa L và X, sử dụng Stem (`[32, 56]`) và số block trung bình, `hidden_dim = 320`.
4. **rtdetr-h (Huge):** Sử dụng cấu hình Backbone HGNetv2-H cực lớn tham chiếu từ repo RT-DETRv2 PyTorch gốc, tăng số block xử lý sâu trên từng stage và sử dụng `hidden_dim = 512` cho chất lượng phát hiện tốt nhất.

---

## 3. Cách Sử Dụng Trong Code

Bạn có thể dễ dàng khởi tạo hoặc train bất kỳ phiên bản mô hình mới nào bằng cách truyền đường dẫn file YAML cấu hình:

```python
from ultralytics import RTDETR

# Khởi tạo mô hình RT-DETR phiên bản siêu nhẹ (Small)
model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-s.yaml")

# Hoặc khởi tạo phiên bản ResNet18
model_r18 = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml")

# Tiến hành huấn luyện hoặc dự đoán bình thường
# model.train(data="dental_data.yaml", epochs=100, imgsz=640)
```
