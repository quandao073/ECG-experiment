# Thiết kế — Kiểm định độ ổn định (K-Fold) + Hiệu chuẩn xác suất (STEMI & OMI)

Ngày 15/08/2026 · nguồn yêu cầu: TIP "Kiểm định độ ổn định + Hiệu chuẩn xác suất" (P0)

## Mục tiêu

Trả lời ba câu hỏi mà `03_stemi_model_comparison.ipynb` và `06_omi_model_comparison.ipynb` chưa trả
lời được vì chỉ có **một lần chia train/val, một seed**:

1. `PlainCNN` thực sự tốt hơn nhóm bám sát, hay chênh lệch chỉ là nhiễu ±0,015 AUROC của một lần chạy?
2. Khi ép về **cùng một điểm vận hành** (Specificity = 0,85), model nào thật sự nhỉnh hơn về
   Sensitivity/NPV — thay vì so recall tại ngưỡng Youden's J riêng của từng model?
3. Sau khi hiệu chuẩn xác suất, ngưỡng vận hành 3 mức có khoảng tin cậy rộng bao nhiêu?

## Deliverable

- `notebooks/08_stemi_stability_calibration.ipynb`
- `notebooks/09_omi_stability_calibration.ipynb`

Đánh số 08/09 thay vì 07/08 như TIP đề xuất: README (bản đang sửa) đã dành số `07` cho
`07_omi_median_beat.ipynb`.

## Quyết định đã chốt với người yêu cầu

Bốn mâu thuẫn nội tại của TIP, đã hỏi và chốt:

| Mâu thuẫn | Quyết định |
|---|---|
| Bước 3b cần OOF của cả 10–11 model, nhưng phạm vi chỉ cho train 3 | Hai bảng tách bạch: (a) **inference-only** nạp lại checkpoint 10–11 model của `03`/`06`, chạy trên đúng val 85/15 gốc; (b) OOF 5-fold cho 3 ứng viên. Không train thêm model nào. |
| STEMI ứng viên #3 là `WBF-ensemble` — không train độc lập được | Train `PlainCNN`, `ResNet1D`, `XResNet1D`. Mỗi fold tính thêm WBF/Average **trên đúng 3 model này** (miễn phí, không train thêm). |
| Bước 4 cấm fit calibration trên tập chọn ngưỡng ở bước 5, bước 5 lại bootstrap trên toàn bộ OOF | **Cross-fit calibration 5-fold**: fold *i* được hiệu chuẩn bằng bộ hiệu chuẩn fit trên OOF của 4 fold còn lại. Ghép lại → mọi bản ghi đều có xác suất hiệu chuẩn "honest", dùng hết dữ liệu, không rò rỉ. |
| Nền tảng chạy | Giữ nguyên nhánh Colab + local như `03`/`06`, không thêm nhánh Kaggle. |

## Lỗi thống kê trong TIP và cách xử lý

TIP yêu cầu Wilcoxon signed-rank kết luận "p < 0,05". **Với n = 5 cặp, p hai phía nhỏ nhất có thể
đạt là 2/2⁵ = 0,0625** — AC đó bất khả thi về mặt toán học, dù model thắng cả 5/5 fold.

Xử lý: giả thuyết vốn có hướng ("PlainCNN tốt hơn"), nên báo cáo

- **Wilcoxon một phía** (n=5 → p nhỏ nhất 0,03125, *có thể* < 0,05) — giữ đúng tinh thần AC;
- **paired t-test** trên ΔAUPRC;
- **bootstrap CI 95% của ΔAUPRC** — bằng chứng chính, không bị chặn bởi n = 5.

Muốn Wilcoxon hai phía đủ lực cần 5-fold × 2 lần lặp (10 cặp) → gấp đôi chi phí GPU. Notebook để
`N_REPEATS = 1` mặc định, có ghi chú cách bật.

## Kiến trúc notebook

### §1 Bộ khung

Copy **verbatim** từ `03`/`06` các cell: cài `wfdb`, cấu hình đường dẫn, staging dữ liệu Colab,
seed/GPU, đọc CSV + gán nhãn, lấy mẫu debug, tiền xử lý + cache memmap, chia 85/15 gốc, metric +
`style_df`, toàn bộ class kiến trúc. Cache `sig_full_n17960_*.npy` dùng lại nguyên si (hash record
giống hệt).

Hai chỗ **buộc phải sửa**, không copy nguyên được:

- `ECGDataset.__init__` nhận thêm `mean`/`std` (mặc định là `LEAD_MEAN`/`LEAD_STD` toàn cục) — vì
  z-score phải fit lại trên train của **từng fold**.
- `train_one()` nhận thêm `fold`, `train_loader`, `val_loader`, `pos_weight` làm tham số thay vì đọc
  biến toàn cục.

Kiến trúc model, `BCEWithLogitsLoss(pos_weight)`, AdamW, `ReduceLROnPlateau` theo val AUPRC, early
stopping, gradient clipping, AMP: **giữ nguyên tuyệt đối**.

CONFIG bổ sung: `K_FOLDS=5`, `N_REPEATS=1`, `CANDIDATES`, `SPEC_TARGET=0.85`, `N_BOOTSTRAP=1000`,
`RUN_SWEEP=False`. `MODEL_DIR` trỏ sang thư mục riêng (`models/stemi_kfold`, `models/omi_kfold`) để
không đụng checkpoint của `03`/`06`; `ORIG_MODEL_DIR` trỏ về thư mục cũ để đọc checkpoint inference-only.

`RUN_MODE="debug"` = 200 bản ghi, **1 fold**, 3 epoch.

### §2 Chia fold & huấn luyện

`StratifiedKFold(K=5, shuffle=True, random_state=SEED)` chạy trên **bảng cấp bệnh nhân** (nhãn bệnh
nhân = `max` nhãn các bản ghi của họ — đúng convention mục 9 hiện có), rồi map ngược về bản ghi.

Assert: (a) không bệnh nhân nào ở hai fold; (b) tỷ lệ dương mỗi fold lệch ≤ 2 điểm % so với toàn tập.

Cùng một bộ 5 fold dùng cho cả 3 model → so sánh là *paired*. `pos_weight` và `LEAD_MEAN/STD` tính
lại trên train của từng fold.

Checkpoint `{model}_fold{k}_best.pt`, fingerprint `(run_mode, target, fold, k_folds, seed, epochs,
n_train)`. Sau mỗi fold ghi OOF probability ra `oof_{target}.npz` → phần thống kê / calibration /
bootstrap chạy lại được **không cần GPU**.

### §3 Thống kê

Bảng `model × fold` với `{auroc, auprc, sensitivity, specificity, f1, npv}`; bảng mean±std; Wilcoxon
một phía + paired t-test + bootstrap CI của ΔAUPRC; boxplot + stripplot AUPRC theo fold cho cả 3
model (và 2 ensemble). Kết luận in ra theo đúng ba khả năng: thắng có ý nghĩa / không đủ bằng chứng
để phân biệt / thua.

### §4 Matched-threshold (3b)

Hai bảng, ghi rõ khác nguồn dữ liệu:

- **(a) inference-only** — nạp checkpoint 10–11 model từ `ORIG_MODEL_DIR`, chạy trên val 85/15 gốc.
  Không tìm thấy checkpoint → in cảnh báo, bỏ qua, không crash.
- **(b) OOF 5-fold** — 3 ứng viên + 2 ensemble.

Ngưỡng nội suy trên ROC tại Spec = 0,85; thêm biến thể ngưỡng rule-out NPV ≥ 99%. Đề xuất ứng viên
thứ 4 chỉ khi chênh > 0,03 Sensitivity so với cả 3 ứng viên hiện tại.

### §5 Hiệu chuẩn & bootstrap

Cross-fit 5-fold cho **Platt scaling** (`LogisticRegression` trên logit) và **isotonic regression**.
ECE (10 bin) trước/sau + reliability diagram chồng ba đường (thô / Platt / isotonic).

Phân tầng 3 mức bằng `find_low_threshold` / `find_high_threshold` copy nguyên từ §14c/15c, chạy trên
xác suất đã cross-fit. Bootstrap 1.000 lần **theo bệnh nhân** (cluster bootstrap — 855 bệnh nhân có
nhiều bản ghi, resample theo bản ghi sẽ cho CI hẹp giả tạo), ngưỡng giữ cố định, báo NPV nhóm thấp và
PPV nhóm cao dạng 95% CI (percentile 2,5–97,5).

### §6 Bonus

- Task 6 (light sweep LR → dropout/focal) viết sẵn, khoá sau `RUN_SWEEP = False`.
- Task 7 (OOD Georgia / Chapman-Shaoxing) **BLOCKED**: `datasets/` không có dữ liệu CinC2021.
- Task 8 (error analysis + Grad-CAM) để notebook riêng, không nhét vào đây.

## Ràng buộc giữ nguyên

Không đổi kiến trúc; không thêm model ngoài danh sách ứng viên; không chạm `test.csv` (tập test ẩn);
tái sử dụng cache memmap float16 hiện có.
