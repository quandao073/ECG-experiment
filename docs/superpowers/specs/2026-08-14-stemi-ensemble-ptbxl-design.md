# 05 — STEMI ensemble trên PTB-XL + ACS-ECG (WBF)

## Mục tiêu

Notebook mới `notebooks/05_stemi_ensemble_ptbxl.ipynb` huấn luyện phát hiện STEMI (nhị phân)
trên dữ liệu **gộp** từ hai nguồn — ACS-ECG Dataset 2026 (đã dùng ở 01/03) và PTB-XL (tải qua
Kaggle) — với 9 kiến trúc mô hình, rồi hợp nhất bằng ensemble kiểu WBF (weighted-average theo
AUROC validation của từng model). Không sửa 01–04.

## Ngoài phạm vi

- Không đổi bài toán/dữ liệu của 01–04.
- Không đụng tập test ẩn `test.csv` của ACS-ECG ở bất kỳ bước nào.
- Không làm sliding-window/detection thật theo thời gian — "WBF" ở đây là ẩn dụ: mỗi model là
  một nguồn xác suất, hợp nhất bằng trung bình có trọng số (không phải WBF gốc cho bounding box).
- Không cố gắng hiệu chỉnh (calibrate) nhãn proxy của PTB-XL thành STEMI thật — chấp nhận giới
  hạn khoa học và nói rõ trong notebook.

## Nguồn dữ liệu

### ACS-ECG Dataset 2026
Tái dùng nguyên staging code từ 01/03 (mount Drive → copy zip → giải nén → `row_data/`, `CSV/`).
Nhãn `STEMI` lấy trực tiếp từ `train.csv`.

### PTB-XL (Kaggle: `m0hamedyousry/ptb-xl-a-large-publicly-available-ecg-dataset`)
- Cài `kagglehub` nếu thiếu (cùng pattern kiểm tra `importlib.util.find_spec` như `wfdb`).
- Cần credential Kaggle: kiểm tra theo thứ tự — biến môi trường `KAGGLE_USERNAME`/`KAGGLE_KEY`,
  `~/.kaggle/kaggle.json`, rồi Colab `userdata` secrets (`KAGGLE_KEY` nếu chạy trên Colab). Thiếu
  cả ba thì in hướng dẫn cụ thể rồi dừng bằng `assert` rõ ràng — không đoán mò.
- `kagglehub.dataset_download(...)` trả về đường dẫn cache cục bộ (không cần bước Drive/zip thủ
  công như ACS-ECG vì dataset nhỏ và ít file hơn nhiều).
- Dò tìm `records500/`, `ptbxl_database.csv`, `scp_statements.csv` trong cây thư mục tải về bằng
  hàm kiểu `_locate_data_root` đã có (chịu được việc bị bọc thêm lớp thư mục).
- Đọc `ptbxl_database.csv`, parse cột `scp_codes` (chuỗi dict Python) bằng `ast.literal_eval`.
- Đọc `scp_statements.csv`, lọc `diagnostic == 1`, lấy `diagnostic_class` theo SCP code.
- `aggregate_diagnostic(scp_dict)`: với mỗi code có likelihood > 0 trong `scp_codes`, tra
  `diagnostic_class`; label proxy `MI_PROXY = 1` nếu `"MI"` xuất hiện trong tập class thu được.
- Đường dẫn file: cột `filename_hr` trong `ptbxl_database.csv` trỏ tới bản ghi 500Hz.

### Cảnh báo bắt buộc trong notebook (markdown, ngay chỗ gán nhãn PTB-XL)
Nhãn proxy PTB-XL (`MI` superclass) **không tương đương** STEMI cấp cứu định nghĩa theo DSA của
ACS-ECG — gồm cả nhồi máu cũ, không phân biệt có/không ST chênh. Trộn 2 định nghĩa vào một nhãn
nhị phân là **domain/label shift có chủ đích**, không phải chuẩn vàng. Bảng breakdown theo
`source` ở phần đánh giá tồn tại chính để lộ rõ ảnh hưởng của việc này.

## Hoà trộn & schema chung

DataFrame gộp: `source` (`"ptbxl"` / `"acs_ecg"`), `record_id`, `record_path` (đường dẫn tuyệt đối
không đuôi mở rộng, dùng thẳng cho `wfdb.rdrecord`), `patient_key` (= `f"{source}:{patient_id}"`
để không lẫn bệnh nhân giữa 2 nguồn dù trùng số ID), `age`, `gender`, `label` (STEMI hoặc
MI_PROXY tuỳ nguồn).

## Chia tập & cache

- Patient-level split trên `patient_key`, stratify theo `(source, label)` gộp thành 1 cột khoá
  (ví dụ `f"{source}_{label}"`) để giữ tỷ lệ 2 nguồn cân đối ở cả train lẫn val.
- `RUN_MODE debug/full` giữ nguyên quy ước 01/03 (debug lấy mẫu stratified theo cùng khoá đó).
- Tiền xử lý: bandpass Butterworth 0.5–40Hz `filtfilt`, pad/truncate về 5000 mẫu — dùng chung
  hàm nhưng cache file riêng (tên cache gồm hash danh sách `record_id`, không đụng cache của
  01–04).
- Chuẩn hoá z-score: mean/std tính trên **toàn bộ train gộp** (không tách riêng theo nguồn) —
  đơn giản và nhất quán với 01/03; nếu domain gap lớn có thể xem lại ở vòng lặp sau, không phải
  scope của notebook này.
- Hai bản ghi lỗi `.dat` ngắn hơn header đã biết ở ACS-ECG (`03228`, `14262`) — tái dùng cơ chế
  bắt lỗi `ValueError` như 01/03. PTB-XL không có ca lỗi nào được biết trước; nếu phát sinh, cùng
  cơ chế fallback áp dụng chung.

## 9 kiến trúc

Tái dùng nguyên class từ 03: `PlainCNN`, `ResNet1D`, `InceptionTime1D`, `CNN+BiLSTM`. Thêm 5 lớp
mới, cùng interface `(batch, 12, 5000) → logit`:

1. **SEResNet1D** — `ResidualBlock1D` + khối squeeze-excitation (global-avg-pool theo kênh → FC
   giảm chiều → FC khôi phục → sigmoid → nhân lại vào feature map) sau mỗi block.
2. **Transformer1D** — cắt tín hiệu thành patch không chồng lấn theo trục thời gian (patch dài đủ
   để bao vài chu kỳ tim, ví dụ 250 mẫu = 0.5s → 20 patch), chiếu tuyến tính + positional
   embedding học được, vài lớp `TransformerEncoderLayer`, token `[CLS]` học được đưa vào head.
3. **TCN1D** — chồng nhiều khối Conv1D giãn nở (dilation tăng dần theo luỹ thừa 2), causal hoặc
   non-causal padding, residual trong mỗi khối, GlobalAvgPool → head.
4. **XResNet1D** — biến thể `ResNet1D`: stem 3 lớp conv nhỏ xếp chồng (deep stem) thay vì 1 lớp
   conv to, downsample nhánh tắt bằng AvgPool+1×1 conv thay vì strided conv, activation SiLU.
5. **ConvNeXtV2_1D** — khối: depthwise Conv1D kernel lớn → LayerNorm (channel-last) → pointwise
   MLP mở rộng 4x với GELU → Global Response Normalization (GRN) → pointwise co lại → residual.
   Vài khối xếp chồng xen kẽ downsample.

Mỗi kiến trúc thêm một mục vào bảng markdown mô tả ý tưởng chính, theo đúng format bảng đã có ở
đầu notebook 03.

## Huấn luyện

Tái dùng nguyên `train_one(name, cls)` + checkpoint-fingerprint-resume từ 03/04: `BCEWithLogitsLoss`
với `pos_weight`, AdamW, `ReduceLROnPlateau` theo val AUPRC, early stopping patience 10, seed cố
định trước mỗi model. `MODELS` registry dict với 9 entry, vòng lặp train tuần tự như cũ.

**Cảnh báo thời gian chạy** (markdown, đầu mục huấn luyện): 9 model × ~35–40k bản ghi lớn hơn
đáng kể so với 03/04 (4 model × 18k) — full run trên Colab free T4 có thể mất vài giờ. Vẫn giữ
checkpoint-resume nên đứt phiên giữa chừng chỉ mất phần đang train dở, Run all lại sẽ tiếp tục
đúng chỗ.

## Ensemble WBF

Sau khi có `RESULTS[name] = {"y":..., "p":...}` cho cả 9 model trên **cùng** tập validation:

```
w_i = max(AUROC_i - 0.5, eps)          # model tệ hơn ngẫu nhiên gần như không đóng góp
w_i = w_i / sum(w_i)                   # chuẩn hoá tổng = 1
p_wbf = sum(w_i * p_i)                 # xác suất hợp nhất
```

So sánh `p_wbf` với: (a) trung bình đều không trọng số, (b) model tốt nhất đơn lẻ theo AUROC —
để trả lời câu hỏi ensemble có thực sự lợi hay chỉ pha loãng model tốt nhất.

## Đánh giá

- Bảng so sánh 9 model + WBF + baseline (đều/best-single) — tái dùng `style_df`/`compare_df`
  pattern từ 03, cột rút gọn giống cell tóm tắt mới thêm ở 01–04 (AUROC/AUPRC/F1/Precision/
  Sensitivity/Specificity/Accuracy).
- **Bảng breakdown theo `source`**: cùng bộ metric nhưng lọc `source == "ptbxl"` và
  `source == "acs_ecg"` riêng trên cùng tập val — đây là phần lộ rõ domain/label gap giữa nhãn
  proxy PTB-XL và nhãn STEMI thật của ACS-ECG.
- Không đánh giá trên `test.csv` ẩn của ACS-ECG.

## File/thư mục

- `notebooks/05_stemi_ensemble_ptbxl.ipynb` — notebook duy nhất, tự chứa (Colab + máy cá nhân),
  theo đúng quy ước Colab/local, `WORK_DIR`/`CACHE_DIR`/`MODEL_DIR` như 03/04 nhưng thư mục model
  riêng (`models/stemi_ensemble_ptbxl`) để không đụng checkpoint của 03.
- Cache tín hiệu PTB-XL và ACS-ECG **tách hai memmap riêng** rồi ghép index khi tạo Dataset, để
  tái dùng cache ACS-ECG đã build ở 01/03 nếu có (không bắt buộc build lại), tránh lãng phí thời
  gian tiền xử lý ACS-ECG lần nữa.

## Rủi ro đã biết, không phải bug

1. Nhãn proxy PTB-XL yếu hơn ACS-ECG (đã nêu ở trên).
2. Kagglehub cần credential — bước thủ công một lần, không tránh được.
3. Runtime dài hơn 03/04 nhiều — chấp nhận đổi lấy độ đa dạng ensemble.
4. Domain gap có thể khiến ensemble không thắng model tốt nhất đơn lẻ — đó là kết quả hợp lệ cần
   báo cáo trung thực, không phải lý do chỉnh nhãn/dữ liệu để "ép" kết quả đẹp hơn.
