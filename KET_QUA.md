# Kết quả — ACS-ECG-AI

Metric trên **tập validation** (15%, chia theo bệnh nhân), chạy ở `RUN_MODE = "full"` với toàn
bộ 17.960 bản ghi có nhãn. Phần cứng: NVIDIA RTX 2000 Ada Laptop GPU.

Số liệu **trích trực tiếp từ output đã nhúng trong 4 notebook đã chạy**, không train lại. Mọi
metric dẫn xuất (F1, Precision, Sensitivity, Specificity, Accuracy) đã được tính lại độc lập từ
TP/TN/FP/FN và **khớp tuyệt đối** với bảng mà notebook in ra.

Ngưỡng phân loại của mỗi mô hình chọn theo **Youden's J** trên tập validation của chính nó, nên
cột `Ngưỡng` khác nhau giữa các dòng là bình thường. **AUROC và AUPRC là hai metric duy nhất
không phụ thuộc ngưỡng** — dùng chúng khi so sánh mô hình.

> Hai bài toán độc lập, báo cáo tách riêng. **Đừng so sánh chéo số giữa hai bài:** tỷ lệ dương
> tính của chúng khác nhau (8,05% so với 49,63%), nên cùng một giá trị AUPRC mang ý nghĩa hoàn
> toàn khác nhau.

---

## Bài toán 2 — Phát hiện STEMI

**Nhãn:** `STEMI` · **Tỷ lệ dương tính trên validation:** 8.05% (217 / 2,696 bản ghi)

Mốc ngẫu nhiên để đối chiếu: **AUROC 0.5000** và **AUPRC 0.0805** (bằng đúng tỷ lệ dương tính).

### Mô hình đơn — ResNet1D · `01_stemi_classifier.ipynb`

| Mô hình | AUROC | AUPRC | F1 | Precision | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|
| ResNet1D | 0.9182 | 0.6519 | 0.4928 | 0.3436 | 0.8710 | 0.8544 | 0.8557 |

Checkpoint tốt nhất ở epoch 22, ngưỡng 0.3367. AUPRC gấp **8.10×** mức ngẫu nhiên. Confusion: TP 189 · TN 2,118 · FP 361 · FN 28 · NPV 0.9870.

Cùng mô hình đó, so hai cách chọn ngưỡng:

| Ngưỡng | F1 | Precision | Sensitivity | Specificity | NPV |
|---|---|---|---|---|---|
| 0.5000 (mặc định) | 0.5240 | 0.3880 | 0.8065 | 0.8887 | 0.9813 |
| 0.3367 (Youden's J) | 0.4928 | 0.3436 | 0.8710 | 0.8544 | 0.9870 |

### So sánh 4 kiến trúc · `03_stemi_model_comparison.ipynb`

Cùng pipeline tiền xử lý, cùng phép chia train/val, cùng seed — nên chênh lệch phản ánh
kiến trúc chứ không phải may rủi của dữ liệu. Xếp theo AUPRC giảm dần.

| Mô hình | AUROC | AUPRC | F1 | Precision | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|
| PlainCNN | 0.9345 | 0.6736 | 0.4581 | 0.3064 | 0.9078 | 0.8201 | 0.8272 |
| CNN+BiLSTM | 0.9295 | 0.6632 | 0.4491 | 0.2998 | 0.8940 | 0.8173 | 0.8234 |
| InceptionTime1D | 0.9256 | 0.6426 | 0.4636 | 0.3147 | 0.8802 | 0.8322 | 0.8361 |
| ResNet1D | 0.9258 | 0.6282 | 0.5191 | 0.3806 | 0.8157 | 0.8838 | 0.8783 |

Confusion matrix và chi phí huấn luyện:

| Mô hình | Ngưỡng | TP | TN | FP | FN | NPV | Tham số | Thời gian |
|---|---|---|---|---|---|---|---|---|
| PlainCNN | 0.0673 | 197 | 2,033 | 446 | 20 | 0.9903 | 304,961 | 229 s |
| CNN+BiLSTM | 0.0744 | 194 | 2,026 | 453 | 23 | 0.9888 | 339,265 | 211 s |
| InceptionTime1D | 0.3238 | 191 | 2,063 | 416 | 26 | 0.9876 | 483,137 | 292 s |
| ResNet1D | 0.3647 | 177 | 2,191 | 288 | 40 | 0.9821 | 986,753 | 274 s |

Khoảng AUROC giữa 4 kiến trúc: **0.9256 – 0.9345** (chênh 0.0089).

---

## Bài toán 3 — Phát hiện dấu hiệu thiếu máu cơ tim cấp (ACS)

**Nhãn:** `STEMI | NSTEMI | UA` · **Tỷ lệ dương tính trên validation:** 49.63% (1,337 / 2,694 bản ghi)

Mốc ngẫu nhiên để đối chiếu: **AUROC 0.5000** và **AUPRC 0.4963** (bằng đúng tỷ lệ dương tính).

### Mô hình đơn — ResNet1D · `02_ischemia_acs_classifier.ipynb`

| Mô hình | AUROC | AUPRC | F1 | Precision | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|
| ResNet1D | 0.6767 | 0.6833 | 0.5866 | 0.6598 | 0.5280 | 0.7318 | 0.6307 |

Checkpoint tốt nhất ở epoch 20, ngưỡng 0.4911. AUPRC gấp **1.38×** mức ngẫu nhiên. Confusion: TP 706 · TN 993 · FP 364 · FN 631 · NPV 0.6115.

Cùng mô hình đó, so hai cách chọn ngưỡng:

| Ngưỡng | F1 | Precision | Sensitivity | Specificity | NPV |
|---|---|---|---|---|---|
| 0.5000 (mặc định) | 0.5684 | 0.6684 | 0.4944 | 0.7583 | 0.6035 |
| 0.4911 (Youden's J) | 0.5866 | 0.6598 | 0.5280 | 0.7318 | 0.6115 |

### So sánh 4 kiến trúc · `04_acs_model_comparison.ipynb`

Cùng pipeline tiền xử lý, cùng phép chia train/val, cùng seed — nên chênh lệch phản ánh
kiến trúc chứ không phải may rủi của dữ liệu. Xếp theo AUPRC giảm dần.

| Mô hình | AUROC | AUPRC | F1 | Precision | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|
| PlainCNN | 0.6970 | 0.6975 | 0.6301 | 0.6462 | 0.6148 | 0.6684 | 0.6418 |
| CNN+BiLSTM | 0.6860 | 0.6909 | 0.5899 | 0.6578 | 0.5348 | 0.7259 | 0.6310 |
| InceptionTime1D | 0.6847 | 0.6885 | 0.6058 | 0.6426 | 0.5729 | 0.6861 | 0.6299 |
| ResNet1D | 0.6819 | 0.6815 | 0.5851 | 0.6715 | 0.5183 | 0.7502 | 0.6351 |

Confusion matrix và chi phí huấn luyện:

| Mô hình | Ngưỡng | TP | TN | FP | FN | NPV | Tham số | Thời gian |
|---|---|---|---|---|---|---|---|---|
| PlainCNN | 0.4516 | 822 | 907 | 450 | 515 | 0.6378 | 304,961 | 222 s |
| CNN+BiLSTM | 0.5376 | 715 | 985 | 372 | 622 | 0.6129 | 339,265 | 187 s |
| InceptionTime1D | 0.4719 | 766 | 931 | 426 | 571 | 0.6198 | 483,137 | 332 s |
| ResNet1D | 0.5032 | 693 | 1,018 | 339 | 644 | 0.6125 | 986,753 | 244 s |

Khoảng AUROC giữa 4 kiến trúc: **0.6819 – 0.6970** (chênh 0.0151).

---

## Cách đọc các con số này

**Chênh lệch giữa 4 kiến trúc nằm trong nhiễu.** Khoảng cách AUROC giữa mô hình cao nhất và thấp
nhất là 0,0089 (STEMI) và 0,0151 (ACS). Trong khi đó, chạy lại **cùng một notebook với cùng
seed** đã cho AUROC STEMI dao động khoảng 0,015 — đo được qua bốn lần chạy: 0,9232 → 0,9254 →
0,9293 → 0,9362. Ba nguyên nhân cộng dồn: `cudnn.benchmark` chọn thuật toán convolution không
tất định, AMP làm phép cộng dấu phẩy động không kết hợp được, và early stopping chốt ở epoch
khác nhau giữa các lần nên checkpoint tốt nhất là model khác.

Vì vậy **không kết luận được kiến trúc nào tốt hơn** từ một seed. Thứ đọc được chắc chắn là:
**tăng độ phức tạp kiến trúc không mang lại lợi ích đo được**. ResNet1D nhiều hơn PlainCNN gấp
3,2 lần tham số mà không hơn ở cả hai bài toán; InceptionTime1D tốn nhiều thời gian nhất cũng
vậy. Nút thắt nằm ở dữ liệu và cách đặt bài toán, không ở sức chứa của mô hình.

**STEMI: Precision thấp là do chủ đích, không phải lỗi.** `pos_weight = 11,46` cố tình đánh đổi
Precision lấy Sensitivity. Kết quả là mô hình bắt được khoảng 87% số ca STEMI nhưng kèm nhiều báo
động giả, trong khi NPV đạt 0,987. Đây là đặc tính của một công cụ **sàng lọc loại trừ** (âm tính
thì khá yên tâm) hơn là công cụ khẳng định chẩn đoán. Cần Precision cao hơn thì chỉ việc nâng
ngưỡng — không phải train lại.

**ACS yếu hơn hẳn, và điều đó hợp lý về mặt y học.** Nhãn này gộp cả UA (6.213 ca, khoảng 70% số
ca dương tính), mà đau thắt ngực không ổn định theo định nghĩa thường **không có thay đổi đặc
trưng trên ECG**. Mô hình đang được yêu cầu tìm thứ phần lớn không hiện diện trong tín hiệu. Đây
là giới hạn của bài toán, không phải của mô hình.

---

## Điều kiện thí nghiệm

- **Dữ liệu:** Du et al., *Sci Data* 13:1009 (2026), DOI dữ liệu `10.6084/m9.figshare.29925314`.
  Chỉ dùng phần train 90% có nhãn: 17.960 bản ghi / 17.018 bệnh nhân.
- **Tập test ẩn 10% không được đọc ở bất kỳ bước nào** — cả việc chọn mô hình lẫn chọn ngưỡng đều
  làm trên validation.
- **Chia theo bệnh nhân** 85/15, stratify ở cấp bệnh nhân. Đã assert giao nhau `patient_id` rỗng.
- **Tiền xử lý:** WFDB thô 12×5000 @ 500 Hz → NaN→0 → cắt/đệm về 5000 mẫu → bandpass Butterworth
  0,5–40 Hz zero-phase → z-score theo chuyển đạo, mean/std fit **riêng trên tập train**.
- **Huấn luyện:** `BCEWithLogitsLoss(pos_weight)`, AdamW (lr 1e-3, weight decay 1e-4),
  ReduceLROnPlateau theo val AUPRC, early stopping patience 10, tối đa 30 epoch, seed 42.

## Hạn chế

- Chỉ một lần chia train/val, chưa cross-validation — chưa có khoảng tin cậy cho các metric.
- Một seed cho mỗi kiến trúc: đủ để loại bỏ kiến trúc kém hẳn, chưa đủ để xếp hạng những cái sát
  nhau (xem phần trên).
- Xác suất đầu ra chưa được hiệu chuẩn (ECE khoảng 0,10 ở bài STEMI). Muốn dùng xác suất tuyệt
  đối thì cần hiệu chuẩn lại (Platt/isotonic) trên một tập riêng.
- Chưa dùng biến lâm sàng (`age`, `gender`, `Time_Interval`) — mô hình thuần tín hiệu.
- Chưa đánh giá trên tập test ẩn. Muốn kiểm chứng độc lập cần nộp dự đoán lên nền tảng của nhóm
  tác giả bộ dữ liệu.
