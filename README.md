# Đồ án Data Mining: Dự đoán nguy cơ mắc bệnh Parkinson

Bộ khung source code cho đồ án khai thác dữ liệu. Các notebook được tạo ở trạng thái **chưa chạy**; cần đặt dữ liệu gốc vào `data/` trước khi chạy trên Jupyter Notebook hoặc VS Code.

> **Lưu ý về phạm vi:** bộ dữ liệu Parkinsons phổ biến thường có cột `status` (1 = có Parkinson, 0 = không Parkinson) và các đặc trưng đo từ giọng nói. Mô hình trên bộ dữ liệu này phân loại nhãn trong dữ liệu, không phải công cụ chẩn đoán y khoa và không tự chứng minh khả năng dự đoán nguy cơ mắc bệnh trong tương lai.

## 1. Cấu trúc thư mục

```text
DoAn_DataMining_Parkinson/
├── data/
│   ├── D07 - parkinsons.data.xlsx   # Đặt file dữ liệu gốc tại đây
│   └── README.md
├── notebooks/
│   ├── 01_KhamPha_TienXuLy.ipynb
│   └── 02_ChonDacTrung_MoHinh.ipynb
├── src/
│   ├── __init__.py
│   └── utils.py
├── tests/
│   └── test_utils.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 2. Chuẩn bị dữ liệu

Đặt file Excel đã chọn vào đúng đường dẫn:

`data/D07 - parkinsons.data.xlsx`

Notebook cũng nhận `data/parkinsons_processed.csv` nếu file này đã được tạo ở notebook 01. Không ghi đè file gốc; giữ nguyên file ban đầu để đối chiếu.

Notebook mặc định nhận diện cột mục tiêu tên `status`, `target` hoặc `label`. Nếu file sử dụng tên cột khác, sửa biến `TARGET_COL` ở notebook 01 và 02 cho phù hợp. Cột `name`/ID được xem là định danh và không đưa vào mô hình.

## 3. Cài đặt môi trường

Mở terminal tại thư mục gốc của dự án, tạo môi trường ảo rồi cài thư viện:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Mở thư mục dự án bằng VS Code, chọn Python interpreter của `.venv`, sau đó mở notebook trong thư mục `notebooks/`.

## 4. Thứ tự làm việc

1. `01_KhamPha_TienXuLy.ipynb`: đọc dữ liệu, lập hồ sơ chất lượng, kiểm tra thiếu/trùng/kiểu dữ liệu, khảo sát ngoại lai bằng IQR và MAD, minh họa Scaling/Normalization và tùy chọn lưu CSV.
2. `02_ChonDacTrung_MoHinh.ipynb`: tách train/test có stratify, chọn đặc trưng theo Filter và Wrapper, huấn luyện cây quyết định CART, đánh giá Accuracy/Precision/Recall/F1/ROC-AUC, vẽ ROC và trích luật IF–THEN.

Các cell trong notebook cố ý không có output. Chạy lần lượt từ trên xuống sau khi đã đặt dữ liệu vào `data/`.

## 5. Ghi chú phương pháp

- Tiền xử lý học tham số từ toàn bộ dữ liệu có thể gây rò rỉ dữ liệu. Notebook mô hình thực hiện chọn đặc trưng trên tập train; không dùng test để chọn đặc trưng.
- Việc đánh dấu ngoại lai không đồng nghĩa phải xóa hàng loạt. Cần kiểm tra ý nghĩa của điểm dữ liệu và ghi rõ lý do nếu loại bỏ.
- `DecisionTreeClassifier` được dùng cho CART. `criterion='entropy'` là một biến thể dựa trên entropy, không nên ghi rằng sklearn đã triển khai nguyên bản ID3/C4.5.
- Recall của lớp Parkinson thường đáng quan tâm, nhưng cần trình bày cùng Precision, ROC-AUC, ma trận nhầm lẫn và giới hạn của bộ dữ liệu.
- Mục checklist 14 bước trong notebook là checklist kỹ thuật đề xuất. Đối chiếu lại tên bước với checklist chính thức của giảng viên nếu tài liệu môn học có quy định riêng.
