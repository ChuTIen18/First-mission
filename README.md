URL_DATA_VSC: https://drive.google.com/file/d/1Cy5-SI85uwugsjF2Z7hiTMHqsGq5ESDX/view?usp=sharing
Histogram_SampleData: data_flow_2023.ipynb

#  Data Export Structure – AIS Vessel Dataset (2023–2024)

## 1. Tổng quan

Dataset được trích xuất từ SQL Server (Docker container) và chia theo:

- Năm dữ liệu: 2023, 2024
- Khu vực hoạt động:
  - Bay (tàu trong vịnh)
  - Offshore (tàu ngoài khơi)

Dữ liệu được export từ các bảng:

- `training_bay`
- `training_offshore`

Mỗi bảng được export theo dạng chunk (50,000 rows/file) để tối ưu RAM và tránh crash hệ thống.

---
