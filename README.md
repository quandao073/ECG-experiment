# ACS-ECG-AI — Phát hiện hội chứng mạch vành cấp từ ECG 12 chuyển đạo

Dự án Vinmec. Huấn luyện mô hình học sâu 1D trên tín hiệu ECG thô để giải ba bài toán phân loại
nhị phân độc lập.

**Bộ dữ liệu:** *A large-scale 12-lead electrocardiogram dataset for acute coronary syndrome
prediction containing 19,955 ECGs* (Du et al., Sci Data 13:1009, 2026)
· paper `10.1038/s41597-026-07278-0` · dữ liệu `10.6084/m9.figshare.29925314`

## Ba bài toán

| | nhãn | tỷ lệ dương | notebook |
|---|---|---|---|
| **Bài toán 2** | `STEMI` | 8,0% | `03`, `05` |
| **Bài toán 3** | `STEMI \| NSTEMI \| UA` (proxy ACS) | 49,5% | `04` |
| **Bài toán 4** | `OMI` — tắc mạch hoàn toàn TIMI 0–1 theo chụp mạch | 6,4% | `06` |

Ba bài toán được giữ **tách biệt hoàn toàn** — mỗi bài có notebook và kết luận riêng.

Bài toán 4 là **bài toán mà paper gốc của bộ dữ liệu đặt ra**: baseline deep learning duy nhất mà
Du *et al.* công bố và mở mã nguồn làm đúng OMI/non-OMI, không phải STEMI. Đáng chú ý, **423/1.151
ca OMI (37%) nằm trong nhóm NSTEMI** — mạch tắc hoàn toàn nhưng tiêu chí ST chênh lên không bắt được.

## Notebook

| file | nội dung |
|---|---|
| [`notebooks/03_stemi_model_comparison.ipynb`](notebooks/03_stemi_model_comparison.ipynb) | STEMI — so sánh 10 kiến trúc + ensemble WBF + phân tầng nguy cơ, kèm EDA/xem tín hiệu và hiệu chuẩn xác suất (ECE) cho model tốt nhất |
| [`notebooks/04_acs_model_comparison.ipynb`](notebooks/04_acs_model_comparison.ipynb) | ACS — so sánh 4 kiến trúc, kèm EDA/xem tín hiệu và hiệu chuẩn xác suất (ECE) cho model tốt nhất |
| [`notebooks/05_stemi_ensemble_ptbxl.ipynb`](notebooks/05_stemi_ensemble_ptbxl.ipynb) | STEMI — 9 kiến trúc trên dữ liệu gộp ACS-ECG + PTB-XL (nhãn proxy) |
| [`notebooks/06_omi_model_comparison.ipynb`](notebooks/06_omi_model_comparison.ipynb) | OMI — so sánh 11 kiến trúc, đối chiếu baseline paper & chuyên gia, phân nhóm "OMI ẩn trong NSTEMI" |
| [`notebooks/07_omi_median_beat.ipynb`](notebooks/07_omi_median_beat.ipynb) | OMI — **cùng 11 kiến trúc, đổi input sang nhịp trung bình `.med` 12×500**, so cặp raw ↔ median trên cùng tập validation |
| [`notebooks/08_stemi_stability_calibration.ipynb`](notebooks/08_stemi_stability_calibration.ipynb) | STEMI — **5-fold theo bệnh nhân** cho 3 ứng viên đầu bảng, kiểm định ghép cặp, so khớp ngưỡng ở cùng Specificity, hiệu chuẩn cross-fit, bootstrap CI cho ngưỡng 3 mức |
| [`notebooks/09_omi_stability_calibration.ipynb`](notebooks/09_omi_stability_calibration.ipynb) | OMI — như `08`, ứng viên `PlainCNN` / `TCN1D` / `ResNet1D` |
| [`notebooks/10_classical_ml_comparison.ipynb`](notebooks/10_classical_ml_comparison.ipynb) | STEMI + OMI — đặc trưng lâm sàng (`neurokit2`) + Logistic Regression/Random Forest/XGBoost, so ghép cặp với deep learning (`08`/`09`) trên cùng 5-fold, chạy CPU không cần GPU |

Các notebook so sánh dùng **cùng pipeline, cùng split, cùng seed** nên chênh lệch phản ánh kiến trúc
chứ không phải may rủi dữ liệu — nhưng vẫn là **một** lần chia và **một** seed, nên thứ hạng giữa các
model sát nhau chưa có căn cứ; `08`/`09` mới là nơi kiểm định lại bằng 5-fold và khoảng tin cậy.
Riêng `06` bổ sung **`PaperBaselineCNN`** — tái hiện sơ đồ kiến trúc
baseline ở Fig. 9 của paper (10 lớp conv kernel 71, nối dày đặc kiểu DenseNet).

Mỗi notebook **tự chứa** và chạy được trên cả **Google Colab** lẫn **máy cá nhân**, tự nhận diện môi
trường. Kết quả hiển thị ngay trong notebook, không sinh file báo cáo.

## Chạy

**Google Colab:** xem [`HUONG_DAN_COLAB.md`](HUONG_DAN_COLAB.md) — hướng dẫn đầy đủ từ nén dữ liệu,
đưa lên Drive, tới xử lý sự cố.

**Máy cá nhân:** đặt dữ liệu ở `datasets/`, sửa `LOCAL_DATA_ROOT` ở mục 2 của notebook, rồi Run all.

Cả hai môi trường đều mặc định `RUN_MODE = "debug"` (200 bản ghi, 3 epoch) để kiểm tra pipeline
trước. Đổi thành `"full"` khi chạy thật.

## Pipeline

```
WFDB thô (12 × 5000 @ 500 Hz)
  → NaN→0 → cắt/đệm về 5000 mẫu
  → bandpass Butterworth 0,5–40 Hz (filtfilt, zero-phase)
  → cache float16 (~2,2 GB, dùng lại được giữa các notebook cùng pipeline tiền xử lý)
  → z-score theo chuyển đạo, mean/std fit RIÊNG trên tập train
  → chia train/val 85/15 THEO BỆNH NHÂN, stratify ở cấp bệnh nhân
  → 1D CNN, BCEWithLogitsLoss(pos_weight), AdamW, early stopping theo val AUPRC
```

Hai điểm dễ làm sai đã được xử lý và kiểm tra bằng assert trong notebook:

- **Chia theo bệnh nhân, không theo bản ghi.** Một bệnh nhân có thể có nhiều ECG trong 7 ngày trước
  DSA; 855 bệnh nhân có >1 bản ghi và 121 trong số đó mang nhãn STEMI không đồng nhất giữa các bản
  ghi của chính họ.
- **Tập test ẩn 10% không được đọc ở bất kỳ bước nào.** `test.csv` không có cột nhãn.

## Ghi chú về dữ liệu

- Hai bản ghi `03228` và `14262` có file `.dat` chỉ chứa 3.500/5.000 mẫu, khiến `wfdb.rdrecord()`
  báo lỗi. Notebook bắt lỗi này, đọc phần thực có rồi zero-pad phần thiếu.
- File `.med` là nhịp trung bình **12 × 500** mẫu (1 giây) — một nhịp đại diện lấy trung vị từ ~15
  nhịp của bản ghi 10 giây, **không** phải bản nén của tín hiệu 12 × 5000. Baseline của paper huấn
  luyện trên chính input này; `07_omi_median_beat.ipynb` dùng nó. Định dạng đã xác minh trên toàn bộ
  dữ liệu: đúng 12.000 byte mỗi file = 500 mẫu × 12 chuyển đạo, `int16` little-endian, **xen kẽ theo
  mẫu** (vị trí `i` và `i+12` là hai mẫu liên tiếp của cùng một chuyển đạo), gain 1000/mV như `.hea`:
  ```python
  sig = np.fromfile(p, dtype="<i2").reshape(500, 12).T.astype(np.float32) / 1000.0
  ```
  Cả 19.955 file đều nguyên vẹn — nhánh `.med` không gặp lỗi bản ghi cắt ngắn như nhánh raw.
- Ba nhãn `STEMI` / `NSTEMI` / `UA` loại trừ nhau, không bản ghi nào mang hai nhãn.

## Cấu trúc

```
datasets/           dữ liệu gốc (CSV/, row_data/, med_data/) — không commit
notebooks/          notebook so sánh kiến trúc, chạy trực tiếp ở đây
outputs/
  ├── models/       checkpoint
  └── cache/        cache tín hiệu (~2,2 GB, sinh lại được)
HUONG_DAN_COLAB.md  hướng dẫn chạy trên Colab
```
