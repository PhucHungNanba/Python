# Đồ án & Bài tập môn Ngôn ngữ Lập trình Python

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)x
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab)

Chào mừng đến với kho lưu trữ các bài tập và đồ án cuối kỳ môn **Ngôn ngữ Lập trình Python**. Repository này lưu trữ quá trình học tập và nghiên cứu của tôi tại Đại học Sài Gòn.

## Thông tin sinh viên
* [cite_start]**Họ và tên:** Huỳnh Phúc Hưng [cite: 8]
* [cite_start]**MSSV:** 3122411073 [cite: 8]
* [cite_start]**Lớp:** DCT122C3 [cite: 6]
* **Giảng viên hướng dẫn:** ThS.NCS [cite_start]Phùng Thái Thiên Trang [cite: 5]

---

## Cấu trúc Repository

Repository này bao gồm 2 phần chính: Các bài tập quá trình/Seminar và Đồ án cuối kỳ.

| Tên File/Thư mục | Mô tả |
| :--- | :--- |
| `Báo cáo đồ án...docx` | [cite_start]Báo cáo chi tiết toàn văn của Nhóm 1 [cite: 1] |
| `mobilenet_huynhphuchung.ipynb` | [cite_start]**Source code chính của đồ án cá nhân (MobileNetV2)** [cite: 632] |
| `Resnet.ipynb` | Bài tập tìm hiểu về mạng ResNet |
| `K_means.ipynb` | Bài tập thuật toán phân cụm K-Means |
| `numpy_và_pandas.ipynb` | Bài tập xử lý dữ liệu với NumPy và Pandas |
| `matplotlib.ipynb` | Bài tập trực quan hóa dữ liệu |
| `OOP.ipynb` | Bài tập Lập trình hướng đối tượng trong Python |
| `vidu1.ipynb`, `vidu2.ipynb` | Các bài tập ví dụ demo seminar |

Cùng một số file bài tập, tài liệu khác đã bị mất do bị Memory leak
---

## Đồ án Cuối kỳ: FaceFusion System
[cite_start]**Đề tài:** Hệ thống học sâu hợp nhất nhận diện Sắc tộc, Độ tuổi và Giới tính[cite: 4, 71].

### 1. Tổng quan dự án (Project Overview)
Trong kỷ nguyên số hóa, phân tích khuôn mặt (Face Analysis) là bài toán trọng tâm. [cite_start]Nhóm 1 đã xây dựng hệ thống **FaceFusion** - một hệ thống học sâu đa nhiệm (Multi-Task Learning) có khả năng giải quyết đồng thời 3 bài toán chỉ với một lần suy diễn[cite: 72]:
1.  **Age Estimation:** Phân loại nhóm tuổi.
2.  **Gender Classification:** Nhận diện giới tính (Nam/Nữ).
3.  **Ethnicity Recognition:** Phân loại chủng tộc (5 nhóm).

[cite_start]**Dữ liệu sử dụng:** Bộ dữ liệu UTKFace (hơn 20.000 ảnh)[cite: 77, 117].

### 2. Đóng góp cá nhân (My Contribution) - MobileNetV2 [cite_start]Trong khi các thành viên khác tập trung vào độ chính xác với các mô hình lớn (VGG16, ResNet, EfficientNet), tôi chọn hướng tiếp cận **tối ưu hóa tốc độ và tài nguyên** để hướng tới các ứng dụng thời gian thực (Real-time) trên thiết bị di động[cite: 544, 545, 546].

![Sơ đồ kiến trúc MobileNetV2](https://private-user-images.githubusercontent.com/230809903/531786136-fd199403-9d07-407b-94dc-e5461da8fb2e.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc1ODI0MzksIm5iZiI6MTc2NzU4MjEzOSwicGF0aCI6Ii8yMzA4MDk5MDMvNTMxNzg2MTM2LWZkMTk5NDAzLTlkMDctNDA3Yi05NGRjLWU1NDYxZGE4ZmIyZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEwNVQwMzAyMTlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02NWVhNzcyOThjYWVlNWVkNTkzMmViNGIyY2QxMjBiNTJhMGQ4YmQxYmUwNDAxZmVkYjUyZDVmNDQyMjg2N2VjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.cLeLvRZYCUMpeT41ctE1_790AxZ7JJczwoNhQl4SsKc)

![Sơ đồ kiến trúctổng quát của các mô hình](https://private-user-images.githubusercontent.com/230809903/531786699-15b528ae-6bba-4973-aa95-aa70a256b101.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc1ODI0MzksIm5iZiI6MTc2NzU4MjEzOSwicGF0aCI6Ii8yMzA4MDk5MDMvNTMxNzg2Njk5LTE1YjUyOGFlLTZiYmEtNDk3My1hYTk1LWFhNzBhMjU2YjEwMS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEwNVQwMzAyMTlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00NmZmMzNmMzc1N2QxYmY2YjM2ODRlZWY1Y2ZhZGZjZDIyZmU1N2UxNTA0NmNjZGRlZjk4NTJjZjQ5YjNlZDhiJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.xGMvLXOsKN4Q-mat1IYH4T16tHCNe8XZjXyBI_JbEHw) 

#### Phương pháp thực hiện (Methodology)
* [cite_start]**Kiến trúc (Architecture):** Sử dụng **MobileNetV2** (pre-trained trên ImageNet) làm backbone trích xuất đặc trưng[cite: 551].
* [cite_start]**Cơ chế Multi-Head:** Từ vector đặc trưng chung, mạng rẽ nhánh thành 3 đầu ra riêng biệt (3 fully connected layers) cho Tuổi, Giới tính và Chủng tộc[cite: 553].
* [cite_start]**Hàm mất mát (Loss Function):** Tổng hợp CrossEntropyLoss của 3 nhánh: $L_{total} = L_{age} + L_{gender} + L_{race}$[cite: 558, 559].

#### Kết quả thực nghiệm (Results)
[cite_start]Sau quá trình huấn luyện 1 ppp[ơ
5 epochs trên Google Colab (T4 GPU), mô hình đạt được kết quả khả quan, cân bằng giữa tốc độ và độ chính xác[cite: 573, 574]:

| Thuộc tính (Task) | Độ chính xác (Accuracy) | Nhận xét |
| :--- | :--- | :--- |
| **Giới tính (Gender)** | **~ 88.0%** | [cite_start]Cao nhất, đặc trưng rõ ràng [cite: 575] |
| **Chủng tộc (Race)** | **~ 83.0%** | [cite_start]Ổn định đối với một mạng nhẹ (Lightweight) [cite: 577] |
| **Độ tuổi (Age)** | **~ 77.0%** | [cite_start]Thấp hơn do sự giao thoa giữa các nhóm tuổi [cite: 579] |

**Demo nhận diện:**
[cite_start]Hệ thống có khả năng hiển thị bounding box kèm nhãn và độ tin cậy cho từng thuộc tính[cite: 612].

*(Ví dụ kết quả chạy thực tế)*

![Kết quả thực tế đầu tiên:](https://private-user-images.githubusercontent.com/230809903/531790920-a949434b-bad8-4e93-aa90-fcca9ba1bc11.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc1ODI0MzksIm5iZiI6MTc2NzU4MjEzOSwicGF0aCI6Ii8yMzA4MDk5MDMvNTMxNzkwOTIwLWE5NDk0MzRiLWJhZDgtNGU5My1hYTkwLWZjY2E5YmExYmMxMS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEwNVQwMzAyMTlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lZmZiMjU0NDliMWFmODc1MDMzNjUwMWExZDRkYjA1ZTAxZmIxZGQyNTU4ODYxYjQwZDY4MGU4MGRmZWFkZjQ0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.Uk2b69KO-6-XSNeXyzsZshXaRqE-flF-LPHQnPvuB-4)
> [cite_start]*Output: Asian (56.6%), Male (74.1%), Age 0-14 (72.5%)*

![Kết quả thực tế thứ haihttps://private-user-images.githubusercontent.com/230809903/531790926-83c5898c-6354-4d1e-835b-ccacce1d9789.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc1ODI0MzksIm5iZiI6MTc2NzU4MjEzOSwicGF0aCI6Ii8yMzA4MDk5MDMvNTMxNzkwOTI2LTgzYzU4OThjLTYzNTQtNGQxZS04MzViLWNjYWNjZTFkOTc4OS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEwNVQwMzAyMTlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yMzRkMGUxZGRiNGMwNGYxMzRjZGY0NWY3YWZmNzE5N2UzMmRkNGM4YTBjMDU1ZTFlMzAyZjkyM2E0YmRkYTBlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.tVT1o_gpheKM5JX0exlTx8csJEu_WkX7hjmbtzQSSmc):](
> [cite_start]*Output: Asian (66.3%), Male (85.6%), Age 0-14 (88.6%)* 

#### Ưu điểm & Hạn chế
* [cite_start]**Ưu điểm:** Tốc độ suy diễn cực nhanh, số lượng tham số ít, chạy mượt mà trên CPU/Colab free, phù hợp triển khai mobile[cite: 619, 620].
* [cite_start]**Hạn chế:** Còn hiện tượng thiên kiến dữ liệu (bias) với người da trắng, độ chính xác tuổi cần cải thiện thêm[cite: 615, 617].

---

## Hướng dẫn cài đặt & Chạy thử

Để chạy file `mobilenet_huynhphuchung.ipynb` hoặc các file bài tập khác:

1.  [cite_start]**Môi trường:** Khuyến khích sử dụng **Google Colab** để tận dụng GPU miễn phí[cite: 594].
2.  **Dữ liệu:**
    * [cite_start]Cần có file `kaggle.json` để tải dataset UTKFace nếu muốn train lại mô hình[cite: 596].
    * [cite_start]Mount Google Drive để lưu/load model checkpoint[cite: 591].
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
    * [cite_start]Chạy cell `Inference` và trỏ đường dẫn tới ảnh cần test[cite: 601, 604].

---

## Tài liệu tham khảo
1.  Sandler, M., et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks". [cite_start]CVPR 2018[cite: 638].
2.  [cite_start]UTKFace Dataset - Kaggle[cite: 637].
3.  [cite_start]Báo cáo đồ án cuối kỳ môn Python - Nhóm 1, ĐH Sài Gòn[cite: 1].

---
*Developed by Huỳnh Phúc Hưng - 2025*
