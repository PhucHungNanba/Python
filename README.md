# Đồ án & Bài tập môn Ngôn ngữ Lập trình Python

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab)

Mục đích tạo Repo này lưu trữ quá trình học tập và nghiên cứu.

## Thông tin sinh viên
* **Họ và tên:** Huỳnh Phúc Hưng
* **MSSV:** 3122411073
* **Lớp:** DCT122C3
* **Giảng viên hướng dẫn:** ThS.NCS Phùng Thái Thiên Trang

---

## Cấu trúc Repository

Repository này bao gồm 2 phần chính: Các bài tập quá trình/Seminar và Đồ án cuối kỳ.

| Tên File/Thư mục | Mô tả |
| :--- | :--- |
| `Báo cáo đồ án...docx` | Báo cáo chi tiết toàn văn của Nhóm 1 |
| `mobilenet_huynhphuchung.ipynb` | **Source code chính của đồ án cá nhân (MobileNetV2)** |
| `Resnet.ipynb` | Bài tập tìm hiểu về mạng ResNet |
| `K_means.ipynb` | Bài tập thuật toán phân cụm K-Means |
| `numpy_và_pandas.ipynb` | Bài tập xử lý dữ liệu với NumPy và Pandas |
| `matplotlib.ipynb` | Bài tập trực quan hóa dữ liệu |
| `OOP.ipynb` | Bài tập Lập trình hướng đối tượng trong Python |
| `vidu1.ipynb`, `vidu2.ipynb` | Các bài tập ví dụ demo seminar |

Cùng một số file bài tập, tài liệu khác đã bị mất do bị Memory leak

---

## Đồ án Cuối kỳ: FaceFusion System
**Đề tài:** Hệ thống học sâu hợp nhất nhận diện Sắc tộc, Độ tuổi và Giới tính.

### 1. Tổng quan dự án (Project Overview)
Trong kỷ nguyên số hóa, phân tích khuôn mặt (Face Analysis) là bài toán trọng tâm. Nhóm 1 đã xây dựng hệ thống **FaceFusion** - một hệ thống học sâu đa nhiệm (Multi-Task Learning) có khả năng giải quyết đồng thời 3 bài toán chỉ với một lần suy diễn:
1.  **Age Estimation:** Phân loại nhóm tuổi.
2.  **Gender Classification:** Nhận diện giới tính (Nam/Nữ).
3.  **Ethnicity Recognition:** Phân loại chủng tộc (5 nhóm).

**Dữ liệu sử dụng:** Bộ dữ liệu UTKFace (hơn 20.000 ảnh).

### 2. Đóng góp cá nhân (My Contribution) - MobileNetV2
Trong khi các thành viên khác tập trung vào độ chính xác với các mô hình lớn (VGG16, ResNet, EfficientNet), tôi chọn hướng tiếp cận **tối ưu hóa tốc độ và tài nguyên** để hướng tới các ứng dụng thời gian thực (Real-time) trên thiết bị di động.

![Sơ đồ kiến trúc MobileNetV2](https://private-user-images.githubusercontent.com/230809903/531786136-fd199403-9d07-407b-94dc-e5461da8fb2e.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg5MjMzOTUsIm5iZiI6MTc2ODkyMzA5NSwicGF0aCI6Ii8yMzA4MDk5MDMvNTMxNzg2MTM2LWZkMTk5NDAzLTlkMDctNDA3Yi05NGRjLWU1NDYxZGE4ZmIyZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTIwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEyMFQxNTMxMzVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hMWZmNTUyNjk5NTg0ZDA2OTk3YmZlZGRiNjFlMjYwNDQ3MGRhYzhjZjE1YTYyYjA3YWNjNTM1ZDFmMGY3YmVhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.hAmgV8Rl7nGql0KYieKYmRwBBZvFisaOth1SgXQQBvI)

![Sơ đồ kiến trúctổng quát của các mô hình](https://private-user-images.githubusercontent.com/230809903/531786699-15b528ae-6bba-4973-aa95-aa70a256b101.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg5MjMzOTUsIm5iZiI6MTc2ODkyMzA5NSwicGF0aCI6Ii8yMzA4MDk5MDMvNTMxNzg2Njk5LTE1YjUyOGFlLTZiYmEtNDk3My1hYTk1LWFhNzBhMjU2YjEwMS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTIwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEyMFQxNTMxMzVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lZmE3Y2I5ZWU3Yzg2NjM0MmVmYmJiODYwMGJjMTg4ZjQxNWUwOTA0ZjQ4MjIyMTRmMzJjMGYzNzQzNmFmNjZlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.iTVdl32cy5yIzQ5Htgo7-JMR6iXMq43fA-fd3pSDpXg)

#### Phương pháp thực hiện (Methodology)
* **Kiến trúc (Architecture):** Sử dụng **MobileNetV2** (pre-trained trên ImageNet) làm backbone trích xuất đặc trưng.
* **Cơ chế Multi-Head:** Từ vector đặc trưng chung, mạng rẽ nhánh thành 3 đầu ra riêng biệt (3 fully connected layers) cho Tuổi, Giới tính và Chủng tộc.
* **Hàm mất mát (Loss Function):** Tổng hợp CrossEntropyLoss của 3 nhánh: $L_{total} = L_{age} + L_{gender} + L_{race}$.

#### Kết quả thực nghiệm (Results)
Sau quá trình huấn luyện 15 epochs trên Google Colab (T4 GPU), mô hình đạt được kết quả khả quan, cân bằng giữa tốc độ và độ chính xác:

| Thuộc tính (Task) | Độ chính xác (Accuracy) | Nhận xét |
| :--- | :--- | :--- |
| **Giới tính (Gender)** | **~ 88.0%** | Cao nhất, đặc trưng rõ ràng |
| **Chủng tộc (Race)** | **~ 83.0%** | Ổn định đối với một mạng nhẹ (Lightweight) |
| **Độ tuổi (Age)** | **~ 77.0%** | Thấp hơn do sự giao thoa giữa các nhóm tuổi |

**Demo nhận diện:**
Hệ thống có khả năng hiển thị bounding box kèm nhãn và độ tin cậy cho từng thuộc tính.

*(Ví dụ kết quả chạy thực tế)*

![Kết quả thực tế đầu tiên:](https://private-user-images.githubusercontent.com/230809903/531790920-a949434b-bad8-4e93-aa90-fcca9ba1bc11.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg5MjMzOTUsIm5iZiI6MTc2ODkyMzA5NSwicGF0aCI6Ii8yMzA4MDk5MDMvNTMxNzkwOTIwLWE5NDk0MzRiLWJhZDgtNGU5My1hYTkwLWZjY2E5YmExYmMxMS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTIwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEyMFQxNTMxMzVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00YmRlY2VmNDFiMzUxNzM5MDkwODJlZmUwYTk5N2Y5MzY1OGU5MjEyNGU0MjA2YTBmZjM3NjFmMzQ0OGU3ZTQ5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.9iDL0UTYjJST4IvnQBfvySXU6_QDIc4_9wXabmkJ-ic)
> *Output: Asian (56.6%), Male (74.1%), Age 0-14 (72.5%)*

![Kết quả thực tế thứ hai: ](https://private-user-images.githubusercontent.com/230809903/531790926-83c5898c-6354-4d1e-835b-ccacce1d9789.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg5MjMzOTUsIm5iZiI6MTc2ODkyMzA5NSwicGF0aCI6Ii8yMzA4MDk5MDMvNTMxNzkwOTI2LTgzYzU4OThjLTYzNTQtNGQxZS04MzViLWNjYWNjZTFkOTc4OS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTIwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEyMFQxNTMxMzVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0wZGNiYWY3N2Q4ZTZkYWZlYzU2ODFkYTI5MDVlM2MwNDNmZTk0MzgyOWYxMWY4M2U2YWE0MWUwZWRlMmJjNmE5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.vVO5Ti6zXbuqKoaH6x0jCSt5CMT9CM31j_KPIfjZrO8)
> *Output: Asian (66.3%), Male (85.6%), Age 0-14 (88.6%)*

#### Ưu điểm & Hạn chế
* **Ưu điểm:** Tốc độ suy diễn cực nhanh, số lượng tham số ít, chạy mượt mà trên CPU/Colab free, phù hợp triển khai mobile.
* **Hạn chế:** Còn hiện tượng thiên kiến dữ liệu (bias) với người da trắng, độ chính xác tuổi cần cải thiện thêm.

---

## Hướng dẫn cài đặt & Chạy thử

Để chạy file `mobilenet_huynhphuchung.ipynb` hoặc các file bài tập khác:

1.  **Môi trường:** Khuyến khích sử dụng **Google Colab** để tận dụng GPU miễn phí.
2.  **Dữ liệu:**
    * Cần có file `kaggle.json` để tải dataset UTKFace nếu muốn train lại mô hình.
    * Mount Google Drive để lưu/load model checkpoint.
3.  **Thư viện:**
    ```python
    import torch
    import torchvision
    import cv2
    import numpy as np
    # ... (xem chi tiết trong notebook)
    ```
4.  **Chạy Demo:**
    * Tải file trọng số `mobilenet_multihead_best.pth` (nếu có).
    * Chạy cell `Inference` và trỏ đường dẫn tới ảnh cần test.

---

## Tài liệu tham khảo
1.  Sandler, M., et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks". CVPR 2018.
2.  UTKFace Dataset - Kaggle.
3.  Báo cáo đồ án cuối kỳ môn Python - Nhóm 1, ĐH Sài Gòn.

---
*Developed by Huỳnh Phúc Hưng - 2025*
