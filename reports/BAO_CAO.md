# Báo cáo kết quả — Phát hiện STEMI và OMI từ ECG 12 chuyển đạo

Dự án ACS-ECG-AI (Vinmec). Báo cáo tổng hợp kết quả **đã chạy thực tế** của bốn notebook:

| notebook | bài toán | nội dung |
|---|---|---|
| `03_stemi_model_comparison.ipynb` | Phát hiện **STEMI** | so sánh 10 kiến trúc + 2 ensemble, một lần chia 85/15 |
| `06_omi_model_comparison.ipynb` | Phát hiện **OMI** | so sánh 11 kiến trúc + 2 ensemble, một lần chia 85/15 |
| `08_stemi_stability_calibration.ipynb` | Phát hiện **STEMI** | 5-fold theo bệnh nhân cho 3 ứng viên, kiểm định ghép cặp, hiệu chuẩn, CI |
| `09_omi_stability_calibration.ipynb` | Phát hiện **OMI** | như trên |

**Đọc §2 và §3 kèm §4.** §2/§3 là kết quả của *một* lần chia dữ liệu; §4 kiểm định lại và **đảo ngược
kết luận về model tốt nhất** ở cả hai bài toán.

Cả hai chạy ở `RUN_MODE="full"` trên GPU NVIDIA A100-80GB (Google Colab), PyTorch 2.11.

---

## 1. Mô tả bộ dữ liệu

**Nguồn:** *A large-scale 12-lead electrocardiogram dataset for acute coronary syndrome prediction
containing 19,955 ECGs* — Du et al., *Scientific Data* 13:1009 (2026).
DOI paper `10.1038/s41597-026-07278-0` · dữ liệu `10.6084/m9.figshare.29925314`.

### Quy mô

| | giá trị |
|---|---|
| Tổng bản ghi công bố | 19.955 ECG |
| Phần có nhãn (`train.csv`, 90%) | **17.960 bản ghi / 17.018 bệnh nhân** |
| Phần test ẩn (`test.csv`, 10%) | 1.995 bản ghi — **không có nhãn, không đụng tới** |
| Định dạng tín hiệu | WFDB, 12 chuyển đạo × 10 giây @ 500 Hz → ma trận 12 × 5000 |
| Kèm theo | file `.med` (median beat 12×500), tuổi, giới tính, `Patient_id` |

Nhãn được gán từ **báo cáo chụp mạch vành (DSA)** trong vòng 7 ngày sau ECG — không suy ra từ hình
dạng sóng, nên đây là nhãn "ground truth" giải phẫu chứ không phải nhãn đọc ECG.

### Phân bố nhãn (trên 17.960 bản ghi có nhãn)

| nhãn | số ca | tỷ lệ | ý nghĩa |
|---|---|---|---|
| STEMI | 1.442 | 8,03% | nhồi máu có ST chênh lên |
| NSTEMI | 1.235 | 6,88% | nhồi máu không ST chênh lên |
| UA | 6.213 | 34,6% | đau thắt ngực không ổn định |
| AMI | 2.679 | 14,9% | nhồi máu cấp (STEMI + NSTEMI) |
| **OMI** | **1.151** | **6,41%** | tắc mạch hoàn toàn TIMI 0–1 trước PCI |
| CTO | 957 | 5,33% | tắc mạn tính |
| PCI | 4.541 | 25,3% | có can thiệp mạch vành |

Sanity check số ca so với paper: notebook 03 so trực tiếp (lệch −4,7% đến +0,3%, ngưỡng ±10%),
notebook 06 so với số paper × 0,9 vì `train.csv` chỉ giữ 90% (lệch +0,4% đến +11,4%, ngưỡng ±12%).
Cả hai đều đạt.

### OMI chồng lấn với cách phân loại kinh điển

| nhóm | n | trong đó OMI | % của nhóm | % tổng OMI |
|---|---|---|---|---|
| STEMI | 1.442 | 727 | 50,4% | 63,2% |
| **NSTEMI** | 1.235 | **423** | **34,3%** | **36,8%** |
| UA | 6.213 | 0 | 0% | 0% |
| CTO | 957 | 0 | 0% | 0% |

**423/1.151 ca OMI (36,8%) nằm trong nhóm NSTEMI** — động mạch tắc hoàn toàn nhưng tiêu chí ST chênh
lên không bắt được, nên theo phác đồ kinh điển họ không được ưu tiên can thiệp cấp cứu. Đây là lý do
bài toán OMI đáng làm riêng thay vì dùng lại model STEMI.

### Tiền xử lý (dùng chung cho cả hai bài toán)

```
WFDB thô (12 × 5000 @ 500 Hz)
  → NaN→0 → cắt/đệm về đúng 5000 mẫu
  → bandpass Butterworth 0,5–40 Hz (filtfilt, zero-phase)
  → cache float16 (~2,2 GB)
  → z-score theo chuyển đạo, mean/std fit RIÊNG trên tập train (chống rò rỉ)
  → chia train/val 85/15 THEO BỆNH NHÂN, stratify ở cấp bệnh nhân
```

Chia theo `Patient_id` là bắt buộc: một bệnh nhân có thể có nhiều ECG trong 7 ngày trước DSA. Đã
assert giao nhau `Patient_id` giữa train/val là rỗng.

Hai bản ghi lỗi trong dữ liệu gốc (`03228`, `14262`) có `.dat` chỉ chứa 3.500/5.000 mẫu — được bắt
lỗi và zero-pad bù.

### Cấu hình huấn luyện chung

`BCEWithLogitsLoss(pos_weight)` · AdamW · `ReduceLROnPlateau` theo val AUPRC · early stopping
patience 10 · gradient clipping max-norm 1.0 · batch 64 · tối đa 30 epoch · SEED = 42 ·
`Transformer1D` dùng LR riêng 3e-4 (self-attention hội tụ kém ổn định ở LR chung 1e-3).

Ngưỡng quyết định của mỗi model chọn theo **Youden's J** trên validation của chính nó.

---

## 2. Bài toán 1 — Phát hiện STEMI (notebook 03)

### Thiết lập

| | |
|---|---|
| Nhãn đích | cột `STEMI` (lấy trực tiếp) |
| Dữ liệu | 17.960 bản ghi, 1.442 dương (8,03%) |
| Train | 14.465 bệnh nhân / 15.264 bản ghi / 1.225 dương (8,03%) |
| Validation | 2.553 bệnh nhân / **2.696 bản ghi** / 217 dương (8,05%) |
| `pos_weight` | 11,46 |
| Nguồn dữ liệu | chỉ ACS-ECG 2026 (không trộn PTB-XL, để nhãn STEMI đúng nghĩa) |

### Các mô hình

10 kiến trúc 1D: `PlainCNN` (mốc sàn), `ResNet1D`, `InceptionTime1D`, `CNN+BiLSTM`, `SEResNet1D`,
`Transformer1D`, `TCN1D`, `XResNet1D`, `ConvNeXtV2_1D`, `AiTiAMI` *(tái hiện theo mô tả kiến trúc
trong Lee et al., Eur Heart J 2025 — **không phải model/trọng số gốc**)*.

Cộng thêm 2 ensemble hợp nhất xác suất: **WBF-ensemble** (trọng số = `max(AUROC − 0.5, ε)`, chuẩn
hoá tổng 1) và **Average-ensemble** (trung bình đều).

### Bảng metric — validation 2.696 bản ghi (xếp theo AUPRC)

⚠️ Bảng này dựa trên **một** lần chia 85/15. §4 cho thấy thứ hạng giữa các model sát nhau ở đây không
tái lập được qua 5 fold — xem §4.1 và §4.3 trước khi trích dẫn thứ hạng.

| Mô hình | AUROC | AUPRC | F1 | Precision | Sensitivity | Specificity | Accuracy | Ngưỡng |
|---|---|---|---|---|---|---|---|---|
| **PlainCNN** | 0,9295 | **0,6675** | 0,5115 | 0,3621 | 0,8710 | 0,8657 | 0,8661 | 0,365 |
| **WBF-ensemble** | **0,9343** | 0,6648 | 0,5087 | 0,3593 | 0,8710 | 0,8641 | 0,8646 | 0,445 |
| Average-ensemble | 0,9336 | 0,6644 | 0,4578 | 0,3056 | 0,9124 | 0,8185 | 0,8260 | 0,345 |
| ResNet1D | 0,9231 | 0,6530 | **0,5550** | **0,4244** | 0,8018 | **0,9048** | **0,8965** | 0,770 |
| XResNet1D | 0,9297 | 0,6505 | 0,5268 | 0,3840 | 0,8387 | 0,8822 | 0,8787 | 0,649 |
| ConvNeXtV2_1D | 0,9283 | 0,6482 | 0,5353 | 0,3973 | 0,8203 | 0,8911 | 0,8854 | 0,495 |
| InceptionTime1D | 0,9218 | 0,6440 | 0,4804 | 0,3352 | 0,8479 | 0,8528 | 0,8524 | 0,598 |
| TCN1D | 0,9128 | 0,6395 | 0,4845 | 0,3422 | 0,8295 | 0,8604 | 0,8579 | 0,598 |
| AiTiAMI *(tái hiện)* | 0,9258 | 0,6298 | 0,5207 | 0,3776 | 0,8387 | 0,8790 | 0,8757 | 0,429 |
| CNN+BiLSTM | 0,9224 | 0,6284 | 0,4501 | 0,3008 | 0,8940 | 0,8181 | 0,8242 | 0,369 |
| SEResNet1D | 0,9236 | 0,6228 | 0,4685 | 0,3224 | 0,8571 | 0,8423 | 0,8435 | 0,186 |
| Transformer1D | 0,7962 | 0,3184 | 0,3089 | 0,1960 | 0,7281 | 0,7386 | 0,7378 | 0,559 |

*Prevalence = 0,0805 — đó là mức AUPRC của một model đoán ngẫu nhiên. AUPRC 0,667 tức gấp ~8,3 lần.*

### Metric tách theo lớp — PlainCNN (model tốt nhất)

| Lớp | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative (0) — không STEMI | 0,9871 | 0,8657 | 0,9224 | 2.479 |
| **Positive (1) — STEMI** | 0,3621 | 0,8710 | 0,5115 | 217 |
| macro avg | 0,6746 | 0,8683 | 0,7170 | 2.696 |
| weighted avg | 0,9368 | 0,8661 | 0,8893 | 2.696 |

Confusion matrix: TP 189 · FN 28 · TN 2.146 · FP 333 · **NPV 0,9871**.

### Phân tầng nguy cơ 3 mức (theo phương pháp ROMIAE)

WBF-ensemble, ngưỡng thấp 0,352 · ngưỡng cao 0,980:

| Mức | n | tỷ lệ | ca STEMI | tỷ lệ dương trong nhóm |
|---|---|---|---|---|
| Thấp | 2.061 | 76,4% | 20 | **0,97%** (NPV 99,0%) |
| Trung bình | 604 | 22,4% | 168 | 27,8% |
| Cao | 31 | 1,1% | 29 | **93,5%** (PPV) |

Ba phần tư số ca được loại an toàn với NPV ~99%, còn nhóm "cao" gần như chắc chắn là STEMI.

### Chi phí huấn luyện

| Model | Tham số | Thời gian (s) | Epoch | Best epoch |
|---|---|---|---|---|
| PlainCNN | 304.961 | 139,3 | 27 | 17 |
| ResNet1D | 986.753 | 160,6 | 28 | 18 |
| Transformer1D | 1.180.417 | 55,7 | 11 | 1 |
| TCN1D | 2.116.929 | 168,1 | 30 | 20 |
| AiTiAMI | 2.116.481 | 176,4 | 30 | 22 |

*(các model còn lại 104–179 s, xem notebook)*

### Nhận xét

- **Bài toán STEMI đã giải khá tốt: AUROC ~0,93 cho gần như mọi kiến trúc CNN.**
- **`PlainCNN` 305 K tham số ngang bằng mọi kiến trúc phức tạp hơn** (kể cả TCN1D 2,1 M) — với dữ
  liệu và độ dài tín hiệu này, độ phức tạp thêm vào không mang lại gì.
- **Ensemble WBF không thắng rõ ràng** model tốt nhất đơn lẻ: +0,005 AUROC, −0,003 AUPRC. Các model
  mắc cùng loại lỗi nên hợp nhất xác suất không thêm thông tin mới.
- **`Transformer1D` thất bại rõ** (AUROC 0,796, dừng ở epoch 1 best) — self-attention trên chuỗi
  5.000 mẫu không phù hợp ở quy mô dữ liệu này.
- Precision thấp (~0,36) là hệ quả trực tiếp của ngưỡng Youden's J trên tập mất cân bằng 8% — muốn
  PPV cao hơn thì phải đẩy ngưỡng lên, đổi lại mất độ nhạy.

---

## 3. Bài toán 2 — Phát hiện OMI (notebook 06)

Đây là **bài toán mà paper gốc của bộ dữ liệu đặt ra**: baseline deep learning duy nhất mà Du et al.
công bố làm đúng OMI/non-OMI, không phải STEMI.

### Thiết lập

| | |
|---|---|
| Nhãn đích | cột `OMI` — tắc cấp hoàn toàn TIMI 0–1 theo báo cáo chụp mạch, trước PCI |
| Dữ liệu | 17.960 bản ghi, 1.151 dương (6,41%) |
| Train | 14.465 bệnh nhân / 15.256 bản ghi / 977 dương (6,40%) |
| Validation | 2.553 bệnh nhân / **2.704 bản ghi** / 174 dương (6,43%) |
| `pos_weight` | 14,62 |

### Các mô hình

11 kiến trúc: 10 kiến trúc y hệt notebook 03, **cộng thêm `PaperBaselineCNN`** — tái hiện sơ đồ
baseline ở Fig. 9 của chính paper (10 lớp conv kernel 71 chia 3 khối, nối dày đặc kiểu DenseNet, số
kênh khớp đúng sơ đồ). Ba điểm cố ý làm khác paper: input raw 12×5000 thay vì median beat 12×500,
thêm BatchNorm sau mỗi conv, head 1 logit thay vì fc 2 lớp.

### Bảng metric — validation 2.704 bản ghi (xếp theo AUPRC)

⚠️ Bảng này dựa trên **một** lần chia 85/15, và AUPRC ở đây **cao hơn cả năm fold** đo được ở §4.2 —
không dùng làm hiệu năng kỳ vọng của OMI.

| Mô hình | AUROC | AUPRC | F1 | Precision | Sensitivity | Specificity | Accuracy | Ngưỡng |
|---|---|---|---|---|---|---|---|---|
| **PlainCNN** | 0,9201 | **0,5219** | 0,3948 | 0,2546 | 0,8793 | 0,8229 | 0,8266 | 0,518 |
| TCN1D | 0,9135 | 0,5160 | **0,4237** | **0,2854** | 0,8218 | **0,8585** | **0,8561** | 0,633 |
| ResNet1D | 0,9122 | 0,5134 | 0,3885 | 0,2517 | 0,8506 | 0,8261 | 0,8277 | 0,528 |
| **WBF-ensemble** | **0,9296** | 0,5134 | 0,4125 | 0,2663 | **0,9138** | 0,8269 | 0,8325 | 0,348 |
| Average-ensemble | 0,9292 | 0,5080 | 0,4114 | 0,2654 | 0,9138 | 0,8261 | 0,8317 | 0,342 |
| InceptionTime1D | 0,9117 | 0,4961 | 0,3943 | 0,2551 | 0,8678 | 0,8257 | 0,8284 | 0,521 |
| SEResNet1D | 0,9139 | 0,4942 | 0,3365 | 0,2062 | 0,9138 | 0,7581 | 0,7681 | 0,086 |
| CNN+BiLSTM | 0,8945 | 0,4938 | 0,3821 | 0,2479 | 0,8333 | 0,8261 | 0,8266 | 0,070 |
| XResNet1D | 0,9077 | 0,4737 | 0,3359 | 0,2069 | 0,8908 | 0,7652 | 0,7733 | 0,302 |
| AiTiAMI *(tái hiện)* | 0,9092 | 0,4722 | 0,3794 | 0,2428 | 0,8678 | 0,8138 | 0,8173 | 0,459 |
| ConvNeXtV2_1D | 0,9164 | 0,4695 | 0,4170 | 0,2739 | 0,8736 | 0,8407 | 0,8428 | 0,204 |
| PaperBaselineCNN *(tái hiện)* | 0,8896 | 0,4402 | 0,3653 | 0,2358 | 0,8103 | 0,8194 | 0,8188 | 0,326 |
| Transformer1D | 0,8493 | 0,3240 | 0,2970 | 0,1824 | 0,7989 | 0,7538 | 0,7567 | 0,035 |

*Prevalence = 0,0643 — AUPRC 0,522 tức gấp ~8,1 lần mức ngẫu nhiên.*

### Metric tách theo lớp — WBF-ensemble

| Lớp | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative (0) — không OMI | 0,9929 | 0,8269 | 0,9023 | 2.530 |
| **Positive (1) — OMI** | 0,2663 | 0,9138 | 0,4125 | 174 |
| macro avg | 0,6296 | 0,8703 | 0,6574 | 2.704 |
| weighted avg | 0,9461 | 0,8325 | 0,8708 | 2.704 |

Confusion matrix: TP 159 · **FN chỉ 15** · TN 2.092 · FP 438 · **NPV 0,9929**.

### Đối chiếu với baseline paper và chuyên gia đọc ECG

⚠️ **Không so trực tiếp được**: paper đo trên **test ẩn 1.995 bản ghi** với input **median beat
12×500**; ở đây đo trên **validation tách từ train** với input **raw 12×5000**.

| Nguồn | AUROC | Sens | Spec | PPV | NPV | F1 | Acc |
|---|---|---|---|---|---|---|---|
| WBF-ensemble *(ta)* | 0,9296 | 0,9138 | 0,8269 | 0,2663 | 0,9929 | 0,4125 | 0,8325 |
| PlainCNN *(ta)* | 0,9201 | 0,8793 | 0,8229 | 0,2546 | 0,9900 | 0,3948 | 0,8266 |
| PaperBaselineCNN *(ta tái hiện)* | 0,8896 | 0,8103 | 0,8194 | 0,2358 | 0,9843 | 0,3653 | 0,8188 |
| [paper] Baseline CNN *(test ẩn)* | 0,900 | 0,697 | 0,873 | 0,277 | 0,976 | 0,396 | 0,861 |
| [paper] Chuyên gia ECG #1 *(đọc mù)* | — | 0,277 | 0,972 | 0,407 | 0,951 | 0,330 | 0,927 |
| [paper] Chuyên gia ECG #2 *(đọc mù)* | — | 0,429 | 0,941 | 0,338 | 0,959 | 0,378 | 0,908 |

**So công bằng hơn — ép mọi model về đúng độ nhạy 0,697 của baseline paper** rồi mới so Spec/PPV:

| Model | Ngưỡng | Sens | Spec | PPV | NPV | F1 | Acc |
|---|---|---|---|---|---|---|---|
| **AiTiAMI** *(tái hiện)* | 0,721 | 0,7011 | **0,9221** | **0,3824** | 0,9782 | **0,4949** | **0,9079** |
| ConvNeXtV2_1D | 0,565 | 0,7011 | 0,9186 | 0,3720 | 0,9781 | 0,4861 | 0,9046 |
| XResNet1D | 0,808 | 0,7011 | 0,9174 | 0,3686 | 0,9781 | 0,4832 | 0,9035 |
| WBF-ensemble | 0,577 | 0,7011 | 0,9162 | 0,3653 | 0,9781 | 0,4803 | 0,9024 |
| PlainCNN | 0,765 | 0,7011 | 0,9099 | 0,3486 | 0,9779 | 0,4656 | 0,8964 |
| PaperBaselineCNN | 0,525 | 0,7011 | 0,8783 | 0,2837 | 0,9771 | 0,4040 | 0,8669 |
| **[paper] Baseline CNN** *(test ẩn)* | — | 0,697 | 0,873 | 0,277 | — | 0,396 | 0,861 |

Tại cùng độ nhạy, model tốt nhất ở đây đạt **Specificity 0,922 / PPV 0,382** so với **0,873 / 0,277**
của baseline paper — nhưng đo trên tập khác nên đây là tín hiệu, không phải bằng chứng.

### Phân nhóm quan trọng nhất — OMI ẩn trong NSTEMI

Đây là chỗ model OMI tạo giá trị lâm sàng vượt lên cách phân loại cũ. Metric tính lại **chỉ trên các
ca NSTEMI của validation**, giữ nguyên ngưỡng toàn cục.

Kích thước phân nhóm: NSTEMI **n = 176, OMI = 49 (27,8%)** · STEMI n = 226, OMI = 124 (54,9%).

| Model | AUROC | AUPRC | Sens | Spec | PPV | NPV | F1 |
|---|---|---|---|---|---|---|---|
| **WBF-ensemble** | 0,7098 | **0,5213** | **0,7755** | 0,5276 | 0,3878 | 0,8590 | 0,5170 |
| Average-ensemble | 0,7090 | 0,5198 | 0,7755 | 0,5276 | 0,3878 | 0,8590 | 0,5170 |
| **AiTiAMI** *(tái hiện)* | **0,7276** | 0,5189 | 0,7551 | **0,5984** | **0,4205** | **0,8636** | **0,5401** |
| InceptionTime1D | 0,6934 | 0,5108 | 0,7143 | 0,5354 | 0,3723 | 0,8293 | 0,4895 |
| TCN1D | 0,7008 | 0,5069 | 0,6327 | 0,6378 | 0,4026 | 0,8182 | 0,4921 |
| PlainCNN | 0,6708 | 0,4652 | 0,7347 | 0,5433 | 0,3830 | 0,8415 | 0,5035 |
| PaperBaselineCNN | 0,6481 | 0,4226 | 0,6939 | 0,5512 | 0,3736 | 0,8235 | 0,4857 |

**Diễn giải lâm sàng:** trong 100 ca NSTEMI thực chất có mạch tắc hoàn toàn, WBF-ensemble bắt được
khoảng **78 ca mà tiêu chí ST chênh lên bỏ sót hoàn toàn**.

⚠️ Cỡ mẫu chỉ 176 bản ghi / 49 ca dương — đủ thấy xu hướng, **không đủ để xếp hạng** các model sát
nhau (chênh dưới ~0,05 AUROC là nhiễu).

**Cập nhật — đã xác nhận lại bằng 5-fold + bootstrap (notebook `09` mục 22).** Cảnh báo ở trên đã
được xử lý: dùng lại OOF predictions sẵn có ở mục 14, tính lại hiệu năng trên đúng phân nhóm NSTEMI
của mỗi fold (n = 1.235 bản ghi / 423 ca OMI — gấp ~7 lần cỡ mẫu cũ), giữ nguyên ngưỡng Youden's J đã
chốt cho toàn fold (đúng nguyên tắc triển khai thật: không biết trước bệnh nhân nào là NSTEMI để đổi
ngưỡng riêng).

| | Notebook `06` (1 lần chia) | Mục 22 — 5-fold + bootstrap |
|---|---|---|
| n / số ca dương | 176 / 49 | 1.235 / 423 |
| AUROC | ~0,71 (điểm đơn) | **0,648**, 95% CI [0,616; 0,679] |
| AUPRC | ~0,52 (điểm đơn) | **0,471**, 95% CI [0,423; 0,520] |

Con số cũ (AUROC 0,71) **nằm ngoài cận trên của CI mới** — ước lượng một-lần-chia hơi lạc quan do may
rủi dữ liệu. Con số 5-fold thấp hơn nhưng có khoảng tin cậy chặt, **dùng trích dẫn được**: mô hình vẫn
bắt được tín hiệu thật ở phân nhóm NSTEMI-ẩn-OMI (AUROC 0,65 rõ ràng hơn ngẫu nhiên), chỉ là hiệu năng
khiêm tốn hơn số đơn lẻ ban đầu. `WBF-3` vs `Average-3` trên đúng phân nhóm này: không phân biệt được
(bootstrap CI của Δ chứa 0) — dùng ensemble nào cũng như nhau ở đây.

### Phân tầng nguy cơ 3 mức

WBF-ensemble, ngưỡng thấp 0,395 · ngưỡng cao 0,931:

| Mức | n | tỷ lệ | ca OMI | tỷ lệ dương trong nhóm |
|---|---|---|---|---|
| Thấp | 2.166 | 80,1% | 21 | **0,97%** (NPV 99,0%) |
| Trung bình | 509 | 18,8% | 133 | 26,1% |
| Cao | 29 | 1,1% | 20 | **69,0%** (PPV) |

Với OMI, nhóm "thấp" mới là nhóm có giá trị vận hành nhất: loại 80% bệnh nhân khỏi diện nghi ngờ tắc
mạch với NPV ≥ 99% là căn cứ để **không kích hoạt phòng thông tim cấp cứu**.

### Nhận xét

- **`PlainCNN` lại đứng đầu theo AUPRC (0,5219)**, đúng như ở bài toán STEMI — mốc sàn không bị đánh
  bại bởi kiến trúc nào phức tạp hơn.
- **`WBF-ensemble` đứng đầu theo AUROC (0,9296) và cho độ nhạy cao nhất (0,914) với chỉ 15 FN** —
  ở bài toán OMI (bỏ sót = mất cơ hội tái tưới máu), đây là điểm vận hành đáng chọn hơn.
- **`PaperBaselineCNN` xếp gần cuối (AUROC 0,8896)** — nhưng vì input khác (raw thay vì median beat)
  nên **không kết luận được** kiến trúc của paper kém; notebook 07 mới là chỗ so cặp raw ↔ median.
- **OMI khó hơn STEMI về bản chất** — F1 của lớp dương 0,41 so với 0,51, AUROC 0,92 so với 0,93,
  Precision 0,27 so với 0,36 (AUPRC 0,52 vs 0,67 nhưng prevalence khác nhau nên không so thẳng được):
  ST chênh
  lên là mẫu hình thị giác rõ, tắc mạch thì không phải lúc nào cũng để lại dấu trên một bản ECG 10
  giây. Ngay cả chuyên gia đọc mù cũng chỉ đạt độ nhạy 0,277–0,429.
- Model đạt **độ nhạy cao gấp 2–3 lần chuyên gia đọc mù** — đây chính là lý do AI có giá trị ở bài
  toán này.

---

## 4. Kiểm định độ ổn định và hiệu chuẩn xác suất (notebook 08 & 09)

Hai notebook `03`/`06` xếp hạng model trên **một** lần chia train/val duy nhất, trong khi chính chúng
ghi nhận dao động ±0,015 AUROC khi chạy lại — lớn hơn khoảng cách giữa các model đầu bảng. Notebook
`08`/`09` kiểm định lại phần thứ hạng đó bằng **5-fold chia theo bệnh nhân**, dùng **cùng một bộ fold
cho mọi model** nên so sánh là *paired* (so từng fold một).

⚠️ **Cập nhật phạm vi (chưa chạy lại):** ban đầu chỉ chọn 3 ứng viên top-điểm ở `03`/`06` — nhưng đó
lại chính là xếp hạng một-lần-chia mà `08`/`09` đang chứng minh không đáng tin, nên chọn theo top-3
điểm số là tuỳ ý, không nhất quán với ngưỡng nhiễu ±0,015 AUROC mà chính `03`/`06` tự đặt ra. Đã sửa:
`CANDIDATES` trong `08`/`09` mở rộng thành **8 model mỗi bài toán** — mọi kiến trúc nằm trong vùng
nhiễu ±0,015 AUROC so với model đơn tốt nhất (xem danh sách ở bảng dưới). **Toàn bộ số liệu trong §4
này vẫn là kết quả của lần chạy 3 ứng viên cũ** — notebook đã sửa code, chưa chạy lại trên GPU. Kết
quả cho 8 ứng viên sẽ cập nhật sau khi chạy Colab.

### Thiết lập

| | |
|---|---|
| Chia fold | `StratifiedKFold` 5 fold trên **bảng cấp bệnh nhân**, map ngược về bản ghi |
| Kiểm tra rò rỉ | không bệnh nhân nào ở hai fold; tỷ lệ dương mỗi fold lệch tối đa **0,04 điểm %** |
| Ứng viên STEMI (đã sửa, **chưa chạy**) | `PlainCNN`, `XResNet1D`, `ConvNeXtV2_1D`, `AiTiAMI`, `ResNet1D`, `SEResNet1D`, `CNN+BiLSTM`, `InceptionTime1D` — 8 model trong vùng ±0,015 AUROC quanh `XResNet1D` (0,9297) |
| Ứng viên OMI (đã sửa, **chưa chạy**) | `PlainCNN`, `ConvNeXtV2_1D`, `TCN1D`, `SEResNet1D`, `ResNet1D`, `InceptionTime1D`, `AiTiAMI`, `XResNet1D` — 8 model trong vùng ±0,015 AUROC quanh `PlainCNN` (0,9201) |
| *Số liệu §4 dưới đây* | *vẫn của lần chạy cũ: STEMI = `PlainCNN`+`ResNet1D`+`XResNet1D`, OMI = `PlainCNN`+`TCN1D`+`ResNet1D`* |
| Mỗi fold | train ~14.360 bản ghi / val ~3.590 bản ghi (so với val 2.696 của lần chia 85/15 gốc) |
| Không đổi | kiến trúc, `BCEWithLogitsLoss(pos_weight)`, AdamW, `ReduceLROnPlateau`, early stopping |
| Chi phí | 3 model × 5 fold × 2 bài toán = 30 lần train, tổng **71 phút** GPU A100 (STEMI 2.174 s, OMI 2.086 s) |

Bảy/tám kiến trúc còn lại **không** được huấn luyện lại; chúng chỉ được chấm lại bằng checkpoint cũ ở
bảng sàng matched-threshold.

### 4.1 STEMI — AUPRC qua 5 fold (trung bình ± độ lệch chuẩn)

| Mô hình | AUROC | AUPRC | Sensitivity | Specificity | F1 | NPV |
|---|---|---|---|---|---|---|
| **Average-3** | **0,9451 ± 0,0036** | **0,7157 ± 0,0104** | 0,8842 | 0,8818 | 0,5476 | 0,9887 |
| WBF-3 | 0,9451 ± 0,0036 | 0,7156 ± 0,0103 | 0,8842 | 0,8820 | 0,5479 | 0,9887 |
| PlainCNN | 0,9344 ± 0,0066 | 0,6951 ± 0,0114 | 0,8655 | 0,8779 | 0,5356 | 0,9869 |
| XResNet1D | 0,9374 ± 0,0047 | 0,6849 ± 0,0098 | 0,8634 | 0,8833 | 0,5417 | 0,9867 |
| ResNet1D | 0,9342 ± 0,0031 | 0,6804 ± 0,0173 | 0,8515 | 0,8857 | 0,5427 | 0,9857 |

*Ngưỡng của mỗi (model, fold) lấy theo Youden's J trong chính fold đó — vì thế cột Sensitivity ở đây
**không** dùng để so model với nhau; việc đó dành cho §4.4.*

### 4.2 OMI — AUPRC qua 5 fold

| Mô hình | AUROC | AUPRC | Sensitivity | Specificity | F1 | NPV |
|---|---|---|---|---|---|---|
| **Average-3** | **0,8991 ± 0,0140** | **0,4374 ± 0,0557** | 0,8558 | 0,8174 | 0,3818 | 0,9881 |
| WBF-3 | 0,8991 ± 0,0140 | 0,4373 ± 0,0556 | 0,8558 | 0,8180 | 0,3824 | 0,9881 |
| PlainCNN | 0,8902 ± 0,0129 | 0,4218 ± 0,0441 | 0,8227 | 0,8245 | 0,3775 | 0,9855 |
| ResNet1D | 0,8893 ± 0,0141 | 0,4174 ± 0,0480 | 0,8404 | 0,8108 | 0,3700 | 0,9869 |
| TCN1D | 0,8787 ± 0,0224 | 0,4144 ± 0,0518 | 0,8036 | 0,8261 | 0,3723 | 0,9840 |

**Độ lệch chuẩn giữa các fold ở OMI (±0,05 AUPRC) lớn gấp năm lần ở STEMI (±0,01).** AUPRC của
`PlainCNN` dao động 0,3579–0,4718 tuỳ fold. Con số 0,5219 báo cáo ở §3 (lần chia 85/15) **cao hơn cả
năm fold** — tức lần chia đó rơi vào phía may mắn. Hai thiết lập không giống hệt nhau (val 2.696 so
với ~3.590 bản ghi, train 85% so với 80%), nên không quy toàn bộ chênh lệch cho may rủi, nhưng hướng
lệch đủ rõ để **không dùng số của §3 làm hiệu năng kỳ vọng của OMI**. Ở STEMI thì ngược lại: cả năm
fold đều cao hơn con số 0,6675 của §2.

### 4.3 Kiểm định ghép cặp — chênh lệch có thật hay là nhiễu?

Wilcoxon signed-rank ở n = 5 cặp **không thể** cho p hai phía < 0,05 (giá trị nhỏ nhất là 2/2⁵ =
0,0625). Vì giả thuyết có hướng, bảng dưới báo cáo **Wilcoxon một phía** (nhỏ nhất 0,03125) kèm
**bootstrap CI 95% của ΔAUPRC** lấy mẫu lại theo bệnh nhân — cái sau là bằng chứng chính vì không bị
chặn bởi số fold.

**STEMI**

| So sánh | Thắng | Δ AUPRC | Wilcoxon 1 phía | Bootstrap 95% CI của Δ | Kết luận |
|---|---|---|---|---|---|
| Average-3 vs PlainCNN | 5/5 | +0,0206 | **0,0312** | [+0,0286; +0,0463] | ensemble hơn, **có ý nghĩa** |
| Average-3 vs XResNet1D | 5/5 | +0,0308 | **0,0312** | [+0,0212; +0,0439] | ensemble hơn, **có ý nghĩa** |
| Average-3 vs ResNet1D | 5/5 | +0,0352 | **0,0312** | [+0,0386; +0,0639] | ensemble hơn, **có ý nghĩa** |
| Average-3 vs WBF-3 | 2/5 | +0,0000 | 0,4062 | [−0,0002; +0,0000] | **ngang nhau** |
| PlainCNN vs XResNet1D | 4/5 | +0,0102 | 0,0938 | — | **không phân biệt được** |
| PlainCNN vs ResNet1D | 4/5 | +0,0147 | 0,0625 | — | **không phân biệt được** |
| XResNet1D vs ResNet1D | 3/5 | +0,0045 | 0,4062 | — | **ngang nhau** |

**OMI**

| So sánh | Thắng | Δ AUPRC | Wilcoxon 1 phía | Bootstrap 95% CI của Δ | Kết luận |
|---|---|---|---|---|---|
| Average-3 vs TCN1D | 5/5 | +0,0230 | **0,0312** | [+0,0154; +0,0403] | ensemble hơn, **có ý nghĩa** |
| Average-3 vs ResNet1D | 4/5 | +0,0200 | 0,0625 | [+0,0117; +0,0364] | nghiêng về ensemble, chưa chắc ở mức 5% |
| Average-3 vs PlainCNN | 4/5 | +0,0156 | 0,0938 | [+0,0131; +0,0359] | nghiêng về ensemble, chưa chắc ở mức 5% |
| Average-3 vs WBF-3 | 3/5 | +0,0002 | 0,2188 | [−0,0003; +0,0002] | **ngang nhau** |
| PlainCNN vs ResNet1D | 4/5 | +0,0044 | 0,2188 | — | **không phân biệt được** |
| PlainCNN vs TCN1D | 4/5 | +0,0074 | 0,1562 | — | **không phân biệt được** |
| ResNet1D vs TCN1D | 3/5 | +0,0030 | 0,3125 | — | **ngang nhau** |

*Bốn dòng cuối mỗi bảng (so hai model đơn với nhau) được tính thêm ngoài notebook, dùng chính bộ
AUPRC 5 fold ở §4.1/§4.2 — notebook chỉ so model hạng nhất với phần còn lại.*

### 4.4 So ở cùng một điểm vận hành — Specificity = 0,85

Cột `Sensitivity` ở §2/§3 đo tại ngưỡng Youden's J **riêng của từng model**, nên hai model "recall
khác nhau" có thể chỉ vì ngưỡng khác nhau. Bảng dưới ép mọi model về đúng Specificity 0,85 (nội suy
trên ROC) rồi mới so, tính trên dự đoán out-of-fold của cả 17.960 bản ghi.

| STEMI | Sens | PPV | NPV | Ngưỡng rule-out (NPV≥99%) | % loại được |
|---|---|---|---|---|---|
| **Average-3** | **0,8967** | **0,3430** | **0,9895** | 0,199 | **77,9%** |
| WBF-3 | 0,8967 | 0,3431 | 0,9895 | 0,199 | 77,9% |
| XResNet1D | 0,8842 | 0,3398 | 0,9882 | 0,119 | 75,0% |
| ResNet1D | 0,8696 | 0,3362 | 0,9868 | 0,097 | 72,7% |
| PlainCNN | 0,8509 | 0,3314 | 0,9849 | 0,056 | 68,1% |

| OMI | Sens | PPV | NPV | Ngưỡng rule-out (NPV≥99%) | % loại được |
|---|---|---|---|---|---|
| **WBF-3** | **0,7976** | **0,2672** | **0,9840** | 0,199 | **69,9%** |
| Average-3 | 0,7967 | 0,2670 | 0,9839 | 0,196 | 69,8% |
| ResNet1D | 0,7750 | 0,2614 | 0,9822 | 0,143 | 66,1% |
| PlainCNN | 0,7593 | 0,2576 | 0,9810 | 0,114 | 61,1% |
| TCN1D | 0,7385 | 0,2521 | 0,9794 | 0,019 | 54,1% |

**Đây là kết quả đảo ngược kết luận cũ:** ở cả hai bài toán, khi so tại cùng một điểm vận hành,
`PlainCNN` là model **kém nhất** trong ba ứng viên — dù nó đứng đầu bảng AUPRC ở §2/§3. Độ nhạy cao
mà §2/§3 gán cho nó đến từ việc Youden's J chọn cho nó một ngưỡng thấp hơn, không phải từ chất lượng
phân biệt.

⚠️ AUPRC trong bảng này thấp hơn ở §4.1/§4.2 vì nó gộp xác suất của **năm model khác nhau** (mỗi fold
một model) vào chung một thang; số dùng để đánh giá hiệu năng vẫn là trung bình từng fold.

### 4.5 Hiệu chuẩn xác suất (cross-fit 5-fold)

Xác suất của fold *i* được hiệu chuẩn bằng bộ hiệu chuẩn fit trên out-of-fold của **bốn fold còn
lại** — không bản ghi nào tham gia hiệu chuẩn chính nó, mà vẫn dùng hết dữ liệu cho bước chọn ngưỡng.

| Bài toán | ECE thô | ECE sau Platt | ECE sau Isotonic | Brier thô → sau | Chọn |
|---|---|---|---|---|---|
| STEMI | 0,0791 | **0,0025** | 0,0042 | 0,0624 → 0,0395 | Platt |
| OMI | 0,1292 | 0,0038 | **0,0037** | 0,0893 → 0,0460 | Isotonic |

Hiệu chuẩn giảm ECE khoảng **30 lần** ở cả hai bài toán, đổi lại AUPRC giảm không đáng kể (STEMI
0,7062 → 0,7010; OMI 0,4164 → 0,4037). Đây là phần trả lời trực tiếp cho hạn chế số 6 của §5: sau
hiệu chuẩn, xác suất đọc được như xác suất thật.

⚠️ Riêng OMI, hai bin cao nhất sau hiệu chuẩn vẫn lệch mạnh (dự đoán 0,83–0,90 nhưng tỷ lệ dương thật
chỉ 0,47–0,53) — chỉ 51 bản ghi rơi vào đó, quá ít để hiệu chuẩn đáng tin ở vùng xác suất cao.

### 4.6 Ngưỡng vận hành ba mức kèm khoảng tin cậy

Bootstrap 1.000 lần, **lấy mẫu lại theo bệnh nhân** (không theo bản ghi — 855 bệnh nhân có nhiều ECG,
lấy mẫu theo bản ghi sẽ cho CI hẹp giả tạo). Ngưỡng giữ cố định, tính trên xác suất đã hiệu chuẩn.

| STEMI — `Average-3` + Platt | n | tỷ lệ | ca dương | chỉ số | 95% CI |
|---|---|---|---|---|---|
| Thấp (< 0,057) | 13.892 | 77,4% | 138 | NPV **0,9901** | [0,9885; 0,9916] |
| Trung bình | 3.886 | 21,6% | 1.131 | tỷ lệ dương 29,1% | — |
| Cao (≥ 0,881) | 182 | 1,0% | 173 | PPV **0,9505** | [0,9152; 0,9788] |

| OMI — `Average-3` + Isotonic | n | tỷ lệ | ca dương | chỉ số | 95% CI |
|---|---|---|---|---|---|
| Thấp (< 0,034) | 12.176 | 67,8% | 121 | NPV **0,9901** | [0,9884; 0,9918] |
| Trung bình | 5.589 | 31,1% | 915 | tỷ lệ dương 16,4% | — |
| Cao (≥ 0,619) | 195 | 1,1% | 115 | PPV **0,5897** | [0,5178; 0,6622] |

Ngưỡng rule-out là con số vững nhất trong cả báo cáo: NPV 99,0% với CI rộng chỉ ±0,17 điểm % ở cả hai
bài toán. Ngược lại PPV nhóm "cao" của OMI có CI rộng **±7 điểm %** (0,52–0,66) vì nhóm này chỉ có
195 bản ghi — **không được báo cáo PPV của OMI như một con số điểm**.

### 4.7 Kết luận

1. **Mô hình nên chọn ở cả hai bài toán là `Average-3` — trung bình cộng xác suất của ba model.**
   Ở STEMI nó thắng cả ba model đơn 5/5 fold với p một phía 0,031 và CI của Δ nằm trọn bên dương. Ở
   OMI bằng chứng yếu hơn (thắng có ý nghĩa trước `TCN1D`, chỉ "nghiêng về" trước `PlainCNN` và
   `ResNet1D`) nhưng không có bằng chứng nào chống lại nó.
2. **Dùng `Average-3` chứ không phải `WBF-3`.** Hai cái khác nhau ở mức 0,0000–0,0002 AUPRC, CI của Δ
   chứa 0 → ngang nhau về thống kê. Trung bình cộng không có trọng số cần ước lượng nên đơn giản hơn
   và không có chỗ để rò rỉ.
3. **Không có model đơn lẻ nào thắng.** Mọi cặp model đơn ở cả hai bài toán đều p > 0,05. Kết luận
   "`PlainCNN` tốt nhất" ở §2/§3 **không tái lập được** — nó là hệ quả của một lần chia dữ liệu.
4. **`PlainCNN` thực ra là ứng viên yếu nhất khi so công bằng** (§4.4): tại Spec = 0,85 nó xếp cuối
   trong ba ứng viên ở cả STEMI (Sens 0,851 so với 0,884 của `XResNet1D`) lẫn OMI (0,759 so với 0,775
   của `ResNet1D`).
5. **STEMI ổn định, OMI thì không.** ±0,01 AUPRC so với ±0,05 giữa các fold. Mọi con số OMI phải kèm
   khoảng dao động; một lần chạy đơn lẻ ở bài toán này không nói lên điều gì.
6. **Xác suất sau hiệu chuẩn dùng được**, và ngưỡng rule-out (loại 68–78% bệnh nhân với NPV 99%) là
   kết quả có giá trị vận hành nhất, với CI đủ hẹp để báo cáo.

---

## 5. Machine learning cổ điển trên đặc trưng lâm sàng (notebook 10)

Tám notebook trước đều dùng deep learning — mạng 1D học trực tiếp trên tín hiệu thô. Notebook `10`
trả lời câu hỏi ngược lại: **nếu trích đặc trưng ECG kiểu lâm sàng kinh điển rồi đưa vào ba mô hình
học máy cổ điển, kết quả thua deep learning bao nhiêu?**

### Thiết lập

| | |
|---|---|
| Đặc trưng | `neurokit2` dò đỉnh R trên chuyển đạo II, đo biên độ tại 4 cửa sổ tương đối R-peak (đường đẳng điện, đỉnh R, điểm J+60–80ms — cách đo ST chênh kinh điển, đỉnh T) trên **cả 12 chuyển đạo** + nhịp tim, HRV. Không dùng `ecg_delineate` đầy đủ (tỷ lệ lỗi cao trên tín hiệu chỉ lọc băng thông). **64 đặc trưng/bản ghi**, NaN 0,04%, chỉ 8/17.960 bản ghi không dò được nhịp nào. |
| Model | Logistic Regression, Random Forest, XGBoost — `class_weight`/`scale_pos_weight` bù lệch lớp, `SimpleImputer(median)` fit trên train từng fold |
| Đánh giá | **5-fold theo bệnh nhân, cùng công thức chia (cùng seed) như notebook `08`/`09`** — để so với `Average-3` (deep learning) là so ghép cặp, không phải so hai mẫu độc lập |
| Chi phí | Toàn bộ (trích đặc trưng + 30 lần fit) chạy trên **CPU, không cần GPU** — dưới 2 phút |

### Kết quả

| STEMI | AUROC (TB±SD) | AUPRC (TB±SD) |
|---|---|---|
| Logistic Regression | 0,7926 ± 0,0168 | 0,2886 ± 0,0181 |
| Random Forest | 0,8967 ± 0,0054 | 0,5420 ± 0,0434 |
| **XGBoost (tốt nhất)** | **0,8985 ± 0,0045** | **0,5662 ± 0,0249** |
| *Deep learning `Average-3` (tham chiếu, §4)* | *0,9451 ± 0,0036* | *0,7157 ± 0,0104* |

| OMI | AUROC (TB±SD) | AUPRC (TB±SD) |
|---|---|---|
| Logistic Regression | 0,7513 ± 0,0145 | 0,1718 ± 0,0213 |
| Random Forest | 0,8445 ± 0,0105 | 0,2922 ± 0,0362 |
| **XGBoost (tốt nhất)** | **0,8429 ± 0,0103** | **0,3131 ± 0,0395** |
| *Deep learning `Average-3` (tham chiếu, §4)* | *0,8991 ± 0,0140* | *0,4374 ± 0,0557* |

**XGBoost tốt nhất ở cả hai bài toán**, thắng Logistic Regression rõ rệt (5/5 fold, p=0,031 cả hai
bài) — mô hình tuyến tính không đủ để nắm bắt quan hệ giữa các đặc trưng ST/T. So với Random Forest,
XGBoost chỉ nhỉnh hơn (4/5 fold, p=0,063, chưa đạt 0,05) — coi hai model này là ngang nhau.

### So với Deep Learning

| Bài toán | Δ AUPRC (DL − ML) | DL thắng | Wilcoxon 1 phía |
|---|---|---|---|
| STEMI | **+0,1494** | 5/5 fold | 0,031 |
| OMI | **+0,1243** | 5/5 fold | 0,031 |

**Deep learning thắng rõ rệt và nhất quán ở cả hai bài toán** — chênh lệch AUPRC ~0,12–0,15, lớn hơn
nhiều so với mức nhiễu ±0,015 AUROC đã ghi nhận trước đó. Nhưng **ML cổ điển không hề vô dụng**:
XGBoost đạt AUROC 0,898 (STEMI) / 0,843 (OMI) chỉ bằng vài chục đặc trưng thủ công — khoảng cách
AUROC với deep learning (~0,05–0,06) hẹp hơn nhiều so với khoảng cách AUPRC (~0,12–0,15). Điều này gợi
ý: deep learning học được thêm tín hiệu tinh vi nằm trong **hình dạng** sóng ECG (không chỉ biên độ
tại vài điểm cố định) — hình dạng đoạn ST cong lên/nằm ngang/chênh xuống, thứ mà 4 cửa sổ biên độ cố
định không nắm bắt được.

⚠️ **So sánh này giả định fold ở notebook `10` trùng khớp fold ở `08`/`09`** (cùng `train.csv`, cùng
mã chia fold, cùng seed) — không có cách xác minh trực tiếp trong notebook `10` vì không nạp lại
checkpoint deep learning. Nếu giả định đúng thì đây là so sánh ghép cặp hợp lệ; nếu không, nên coi là
ước lượng gần đúng.

### Đặc trưng nào quan trọng nhất

`feature_importances_` của XGBoost (trung bình 5 fold, tính lại độc lập để lấy số cụ thể — không chỉ
xem biểu đồ trong notebook):

- **STEMI**: dẫn đầu là biên độ tại cửa sổ J+60ms (đúng điểm đo ST chênh kinh điển) ở `aVF`, `V5`,
  `II`, `V4`, `V3`, `V2` — tức các chuyển đạo đọc thành dưới (`aVF`/`II`/`III`) và thành trước/bên
  (`V2`–`V5`), đúng hai vùng STEMI hay gặp nhất. `ST_elev` (đã trừ baseline) ở `aVL`/`III` cũng lọt
  top 10.
- **OMI**: `V6_baseline`, `V6_t`, `aVR_t` dẫn đầu — khác STEMI, không dồn hẳn vào một điểm đo ST kinh
  điển mà trải ra nhiều đặc trưng khác nhau (baseline, T-wave, ST) — khớp với việc OMI **không có tiêu
  chí ST chênh lên rõ ràng** như STEMI, XGBoost phải "gộp" nhiều tín hiệu yếu hơn thay vì dựa vào một
  đặc trưng chủ đạo.

Model học đúng thứ bác sĩ nhìn vào khi đọc ECG — củng cố độ tin cậy của cách trích đặc trưng, dù kết
quả tổng thể vẫn thua deep learning.


## 6. Hạn chế chung (áp dụng cho cả hai bài toán)

1. ~~Chỉ một lần chia train/val~~ — **đã xử lý ở §4** cho ba ứng viên đầu bảng mỗi bài toán (5-fold
   theo bệnh nhân + bootstrap CI). Bảy/tám kiến trúc còn lại vẫn **chỉ có một lần chia**, nên thứ
   hạng của chúng ở §2/§3 vẫn chưa có khoảng tin cậy.
2. **Dao động chạy lại ±0,015 AUROC** do `cudnn.benchmark` + AMP dù đã cố định seed. Hai model cách
   nhau dưới ~0,015 AUROC nên coi là **ngang nhau** — §4.3 lượng hoá lại điều này bằng kiểm định
   ghép cặp thay vì ước lượng bằng mắt.
3. **Tập test ẩn 10% chưa được đụng tới** ở bất kỳ bước nào — cả chọn model lẫn chọn ngưỡng đều làm
   trên validation. Muốn có số so trực tiếp với paper phải nộp dự đoán lên nền tảng chấm điểm của tác giả.
4. **`AiTiAMI` là tái hiện theo mô tả kiến trúc**, không phải model/trọng số gốc của Medical AI Co.
   Không dùng kết quả của nó để phát biểu về sản phẩm thương mại thật.
5. **Nhãn OMI có false negative nằm sẵn**: các ca tự tái tưới máu trước khi chụp (TIMI 2–3 kèm
   troponin đỉnh cao) bị xếp âm tính dù bản chất là OMI — chính tác giả thừa nhận ở phần Limitation.
   Điều này kéo trần hiệu năng đo được xuống thấp hơn thực tế.
6. ~~Xác suất chưa hiệu chuẩn~~ — **đã xử lý ở §4.5**: cross-fit Platt/isotonic đưa ECE từ 0,079
   xuống 0,0025 (STEMI) và từ 0,129 xuống 0,0037 (OMI). Còn lại một điểm chưa xử lý: ở OMI, vùng xác
   suất > 0,8 vẫn quá thưa (51 bản ghi) để hiệu chuẩn đáng tin.
7. **Ngưỡng vận hành ở §4.6 chốt trên out-of-fold, chưa phải trên dữ liệu độc lập.** Trước khi dùng
   lâm sàng phải xác nhận lại trên một tập ngoài.
8. **Chưa kiểm tra ngoài phân phối** (Georgia / Chapman-Shaoxing) và **chưa phân tích lỗi false
   negative** — hai việc còn nợ, không nằm trong `08`/`09`.
9. **So sánh ML vs DL ở §5 dựa trên giả định fold trùng khớp** giữa notebook `10` và `08`/`09` —
   không xác minh trực tiếp được (không nạp lại checkpoint DL trong `10`). Kiểm định ghép cặp ở cả
   §4 lẫn §5 đều dùng p-value trên 5 fold **không hoàn toàn độc lập** (train set các fold chồng lấn
   ≥60%) — coi p-value là tham khảo, ưu tiên đọc theo số fold thắng liên tục + khoảng tin cậy bootstrap.
10. **`CANDIDATES` ở `08`/`09` đã mở rộng từ 3 lên 8 model/bài toán nhưng CHƯA chạy lại** (cần GPU
    thật trên Colab, checkpoint 3 model cũ được resume tự động, chỉ cần train thêm 5 model mới —
    ước tính +60-70 phút GPU mỗi bài toán). Số liệu §4 trong báo cáo này vẫn của cấu hình 3 ứng viên
    cũ cho tới khi chạy lại. Nếu `Average`/`WBF` đổi model champion sau khi mở rộng, §5 (so ML vs DL)
    và notebook `10`'s `DL_REFERENCE` cũng cần cập nhật theo.
