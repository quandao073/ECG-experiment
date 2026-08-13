# Hướng dẫn train trên Google Colab — ACS-ECG-AI

Hướng dẫn chạy bốn notebook của dự án trên Google Colab, từ lúc chưa có gì tới lúc có kết quả.

> **Trạng thái kiểm chứng.** Cả bốn notebook đã chạy thật và chạy thông trên máy cá nhân
> (Windows + RTX 2000 Ada), ở cả chế độ `debug` lẫn `full`. Riêng **nhánh Colab chưa được thực thi
> trên Colab thật** — phần logic không cần Colab (dò cấu trúc zip, đồng bộ cache) đã được kiểm thử
> riêng, nhưng bước mount Drive và tốc độ thực tế chỉ xác nhận được khi bạn chạy. Các con số thời
> gian ở mục 6 là **ước tính ngoại suy** từ số đo trên máy cá nhân.

## Bốn notebook, hai bài toán

Hai bài toán được giữ **hoàn toàn tách biệt** — mỗi bài có notebook riêng và kết luận riêng.

| notebook | bài toán | nội dung |
|---|---|---|
| `01_stemi_classifier.ipynb` | **Bài toán 2** — STEMI | Một mô hình (ResNet1D), kèm EDA, hiệu chuẩn xác suất, tối ưu ngưỡng |
| `02_ischemia_acs_classifier.ipynb` | **Bài toán 3** — ACS | Như trên, nhãn `STEMI \| NSTEMI \| UA` |
| `03_stemi_model_comparison.ipynb` | **Bài toán 2** — STEMI | So sánh 4 kiến trúc: PlainCNN, ResNet1D, InceptionTime1D, CNN+BiLSTM |
| `04_acs_model_comparison.ipynb` | **Bài toán 3** — ACS | So sánh 4 kiến trúc, cùng bộ như trên |

Cả bốn dùng **chung một pipeline tiền xử lý và chung file cache**, nên notebook thứ hai trở đi trong
cùng một phiên bỏ qua hoàn toàn bước tiền xử lý.

Kết quả (bảng metric, biểu đồ) **hiển thị ngay trong notebook**, không ghi ra file báo cáo. Thư mục
`outputs/` chỉ giữ hai thứ: **checkpoint mô hình** và **cache tín hiệu**.

Mỗi notebook chạy được ở hai môi trường: **Google Colab** và **máy cá nhân**, tự nhận diện.

---

## 1. Tổng quan quy trình

| bước | làm ở đâu | mất bao lâu | lặp lại mỗi phiên? |
|---|---|---|---|
| Nén dữ liệu thành 1 file zip | máy cá nhân | ~3–5 phút | không |
| Upload zip + 4 notebook lên Drive | máy cá nhân | tuỳ mạng (~2,4 GB) | không |
| Mở Colab, bật GPU, kiểm tra đường dẫn | Colab | ~2 phút | chỉ lần đầu |
| Run all | Colab | xem mục 6 | có |

**Điểm mấu chốt:** bộ dữ liệu có **59.867 file**. Google Drive tính mỗi lần mở file là một lệnh gọi
mạng, nên đọc trực tiếp 39.910 file `.dat` từ Drive sẽ chậm hơn hàng chục lần so với đọc từ đĩa
local của Colab. Vì vậy quy trình bắt buộc đi qua **một file zip duy nhất**: Drive giữ 1 file lớn,
Colab copy về `/content` rồi giải nén ra đó. **Đừng upload thư mục dữ liệu đã giải nén lên Drive.**

---

## 2. Bước 1 — Nén dữ liệu thành một file zip

Trên máy cá nhân, nén thư mục `datasets/`:

```
datasets.zip
└── (có thể có hoặc không có lớp thư mục bọc ngoài)
    ├── CSV/
    │   ├── train.csv
    │   └── test.csv
    ├── row_data/          ← 39.910 file .dat + .hea
    └── med_data/          ← 19.955 file .med (notebook không dùng)
```

Notebook **tự dò** thư mục thật chứa `CSV/` và `row_data/`, nên zip có bọc thêm một hay hai lớp thư
mục đều được. Đã kiểm thử với: không bọc, bọc 1 lớp, bọc 2 lớp, và zip do macOS tạo (có `__MACOSX`).

```powershell
cd C:\Users\anhquan\Workspace\AI_THUC_CHIEN\VSF_Projects\ECG-experiment
Compress-Archive -Path datasets\* -DestinationPath datasets.zip -CompressionLevel Fastest
```

`Compress-Archive` chậm với số lượng file lớn. Có 7-Zip thì nhanh hơn nhiều:

```powershell
& "C:\Program Files\7-Zip\7z.exe" a -tzip -mx=1 datasets.zip datasets\*
```

Mức nén thấp nhất là cố ý: tín hiệu ECG là `int16` gần như không nén được, ép nén cao chỉ tốn thời
gian mà giảm chẳng bao nhiêu.

**Có thể bỏ `med_data/` để tiết kiệm ~240 MB** — notebook không dùng tới nó (file `.med` chỉ chứa
12 × 500 mẫu, tức nhịp trung bình 1 giây, không phải bản thay thế cho tín hiệu 12 × 5000):

```powershell
& "C:\Program Files\7-Zip\7z.exe" a -tzip -mx=1 datasets.zip datasets\CSV datasets\row_data
```

Kích thước zip: **~2,4 GB** (có `med_data`) hoặc **~2,1 GB** (không có).

---

## 3. Bước 2 — Đưa lên Google Drive

Tạo thư mục **`ACS-ECG-AI`** ngay ở gốc "My Drive":

```
My Drive/
└── ACS-ECG-AI/
    ├── datasets.zip
    ├── 01_stemi_classifier.ipynb
    ├── 02_ischemia_acs_classifier.ipynb
    ├── 03_stemi_model_comparison.ipynb
    └── 04_acs_model_comparison.ipynb
```

Notebook sẽ tự tạo thêm `outputs/` ở đây.

### Dung lượng Drive cần có

| thứ | dung lượng |
|---|---|
| `datasets.zip` | ~2,4 GB |
| cache tín hiệu (dùng chung cho cả 4 notebook) | ~2,2 GB |
| checkpoint (2 mô hình đơn + 2 × 4 mô hình so sánh) | ~60 MB |
| **tổng** | **~4,7 GB** |

Drive miễn phí 15 GB là đủ. Nếu tài khoản gần đầy thì dọn trước — lỗi hết dung lượng giữa lúc train
rất khó chịu.

---

## 4. Bước 3 — Mở notebook và bật GPU

1. Trên Drive, nháy đúp `01_stemi_classifier.ipynb` → **Open with** → **Google Colaboratory**.
   (Chưa thấy tuỳ chọn này: **Open with** → **Connect more apps** → cài *Colaboratory*.)
2. **Runtime** → **Change runtime type** → **Hardware accelerator: GPU** (thường là T4) → **Save**.

**Đừng bỏ qua bước 2.** Chạy nhầm runtime CPU thì một epoch ở chế độ `full` mất hàng chục phút thay
vì khoảng 10 giây. Notebook có kiểm tra: nếu đang ở Colab mà không có GPU, **mục 3** in cảnh báo lớn
— nhưng nó **không tự dừng**, nên bạn vẫn nên tự xác nhận bằng dòng `GPU:` mà mục 3 in ra.

---

## 5. Bước 4 — Kiểm tra đường dẫn

Ở **mục 2 (Cấu hình)** của notebook:

```python
if IS_COLAB:
    DRIVE_PROJECT = Path("/content/drive/MyDrive/ACS-ECG-AI")   # <<< SỬA nếu đặt tên khác
    DRIVE_DATA_ZIP = DRIVE_PROJECT / "datasets.zip"             # <<< SỬA nếu tên zip khác
```

Làm y hệt mục 3 ở trên thì **không cần sửa gì cả**.

> ### ⚠️ Lỗi hay gặp nhất: dán link chia sẻ Drive vào đây
>
> `DRIVE_PROJECT` phải là **đường dẫn trong hệ thống file sau khi mount**, không phải URL:
>
> ```python
> # SAI — đây là link chia sẻ
> DRIVE_PROJECT = Path("https://drive.google.com/drive/folders/1-w0Jnio...")
>
> # ĐÚNG — /content/drive/MyDrive/ + TÊN thư mục
> DRIVE_PROJECT = Path("/content/drive/MyDrive/ACS-ECG-AI")
> ```
>
> Sau khi Colab mount Drive, toàn bộ "My Drive" xuất hiện thành thư mục thật tại
> `/content/drive/MyDrive/`. Notebook cần **tên thư mục**, không phải link.
>
> Notebook bắt lỗi này: gặp URL thì dừng ngay kèm hướng dẫn; đường dẫn đúng dạng nhưng sai tên thì
> **in ra danh sách thư mục đang có trong MyDrive** để bạn chọn.
>
> **Đừng xoá nhánh `else:`** (phần dành cho máy cá nhân). Cả hai nhánh `if IS_COLAB` / `else` phải
> còn nguyên; xoá `else` sẽ gây `NameError: WORK_DIR is not defined` khi chạy ở máy cá nhân.

---

## 6. Bước 5 — Chạy

**Runtime → Run all.**

### Lần đầu để nguyên `RUN_MODE = "debug"`

Notebook được giao ở chế độ `debug` (200 bản ghi, 3 epoch) để xác nhận đường dẫn, quyền Drive, GPU
và pipeline chạy thông **trước khi** bỏ ra hàng chục phút cho lần chạy thật. Metric ở chế độ debug
không có giá trị khoa học.

Trong lúc chạy, Colab hiện **popup xin quyền truy cập Google Drive** — bấm đồng ý. Đây là thao tác
thủ công duy nhất.

### Chuyển sang chạy thật

Ở **mục 2**, đổi `RUN_MODE = "debug"` thành `RUN_MODE = "full"`, rồi **Run all** lại. Không cần sửa
gì khác — số epoch, batch size và tập dữ liệu tự đổi theo.

### Thời gian dự kiến

| việc | máy cá nhân (RTX 2000 Ada) | ước tính Colab T4 |
|---|---|---|
| Chuẩn bị phiên (cài wfdb, mount Drive, copy + giải nén zip) | — | ~5–7 phút |
| Build cache tín hiệu — **một lần cho cả 4 notebook** | ~105 giây | ~4–6 phút |
| `01_stemi_classifier` | ~4 phút | ~8–10 phút |
| `02_ischemia_acs_classifier` | ~4 phút | ~8–10 phút |
| `03_stemi_model_comparison` (4 mô hình) | ~15 phút | ~35–45 phút |
| `04_acs_model_comparison` (4 mô hình) | ~15 phút | ~35–45 phút |

Colab chậm hơn vì máy ảo chỉ có 2 vCPU (giải nén và tiền xử lý đều là việc của CPU) và phải chuyển
2,4 GB qua mạng Drive.

**Chạy notebook 01 trước** — nó build cache, ba notebook còn lại dùng lại.

**Với Colab miễn phí (~4h/phiên, tự ngắt sau ~90 phút không tương tác), đừng chạy cả 4 trong một
phiên.** Nên chia: phiên 1 chạy 01 + 02, phiên 2 chạy 03, phiên 3 chạy 04. Cache đã sao lưu lên
Drive nên phiên sau không phải tiền xử lý lại.

---

## 7. Khi phiên Colab bị ngắt

Chuyện bình thường, không phải sự cố. **Cần làm gì: mở lại notebook, chọn lại GPU, Run all. Hết.**

| thứ | ở đâu | sau khi mất phiên |
|---|---|---|
| Dữ liệu đã giải nén | `/content` | mất → tự giải nén lại |
| Cache tín hiệu | `/content` + bản sao trên Drive | **lấy lại từ Drive**, bỏ qua tiền xử lý |
| Checkpoint | Drive | **giữ nguyên** |

- **Notebook 01/02:** mục 13 tự nạp `checkpoint_last.pt` và train tiếp từ đúng epoch đang dở.
- **Notebook 03/04:** mô hình nào đã train xong thì được nạp từ checkpoint và bỏ qua, chỉ train tiếp
  những mô hình còn thiếu.
- **Build cache** cũng nối tiếp được: bị ngắt giữa chừng thì lần sau chạy tiếp từ bản ghi đang dở.

Việc nối tiếp chỉ xảy ra khi checkpoint khớp đúng `RUN_MODE`, nhãn đích và kích thước tập train — nên
checkpoint của lần chạy `debug` không bị nhặt nhầm vào lần chạy `full`.

---

## 8. Lấy kết quả về

Bảng metric và biểu đồ nằm ngay trong notebook. Muốn giữ lại: **File → Save a copy in Drive**, hoặc
**File → Download → Download .ipynb** — bản tải về giữ nguyên toàn bộ output.

Trên Drive chỉ còn `ACS-ECG-AI/outputs/` với `models/` (checkpoint) và `cache/`. **Xoá được sau khi
xong:** `outputs/cache/` (~2,2 GB) và `datasets.zip` (~2,4 GB), nếu không định train lại.

---

## 9. Xử lý sự cố

| triệu chứng | nguyên nhân | cách xử lý |
|---|---|---|
| `FileNotFoundError: Không thấy /content/drive/MyDrive/...` | Sai tên thư mục | Xem danh sách thư mục mà thông báo lỗi in ra, sửa `DRIVE_PROJECT` cho khớp |
| `ValueError: DRIVE_PROJECT đang là URL chia sẻ Drive` | Dán link thay vì đường dẫn | Xem mục 5 |
| `Không thấy .../datasets.zip` | Sai tên file zip | Thông báo lỗi liệt kê file đang có trong thư mục — sửa `DRIVE_DATA_ZIP` |
| Cảnh báo "COLAB KHÔNG CÓ GPU" | Runtime đang là CPU | Runtime → Change runtime type → GPU → Run all lại |
| Popup Drive không hiện | Trình duyệt chặn popup | Cho phép popup cho `colab.research.google.com`, chạy lại cell mục 2b |
| `assert RAW_DIR and CSV_DIR` ở mục 4 | Zip thiếu `row_data/` hoặc giải nén hỏng | Xoá `/content/_dataset.zip` rồi chạy lại mục 2b để copy lại |
| Sanity check mục 4 báo lệch quá ±10% | CSV không phải bản gốc | Đối chiếu lại nguồn Figshare — **đừng bỏ qua cảnh báo này** |
| Hết dung lượng Drive giữa chừng | Drive gần đầy | Dọn Drive rồi Run all lại — checkpoint đã ghi vẫn còn |
| `NameError: WORK_DIR is not defined` | Đã xoá nhánh `else:` ở mục 2 | Khôi phục lại nhánh đó (xem mục 5) |

---

## 10. Kết quả tham chiếu

Đo thật trên máy cá nhân với `RUN_MODE = "full"`, validation 15% chia theo bệnh nhân. Dùng để đối
chiếu khi bạn chạy trên Colab.

### Notebook 01 / 02 — mô hình đơn (ResNet1D)

| | 01 STEMI | 02 ACS |
|---|---|---|
| AUROC | 0,9182 | 0,6767 |
| AUPRC (mức ngẫu nhiên) | 0,6519 (0,0805) | 0,6833 (0,4963) |
| Recall lớp dương @ Youden | 0,8710 | 0,5280 |
| Precision lớp dương @ Youden | 0,3436 | 0,6598 |
| NPV | 0,9870 | 0,6115 |
| Thời gian (30 epoch) | ~190 s | ~216 s |

### Notebook 03 / 04 — so sánh 4 kiến trúc (xếp theo AUPRC)

**STEMI:**

| Model | AUROC | AUPRC | Tham số | Thời gian |
|---|---|---|---|---|
| PlainCNN | 0,9345 | 0,6736 | 304.961 | 229 s |
| CNN+BiLSTM | 0,9295 | 0,6632 | 339.265 | 211 s |
| InceptionTime1D | 0,9256 | 0,6426 | 483.137 | 292 s |
| ResNet1D | 0,9258 | 0,6282 | 986.753 | 274 s |

**ACS:**

| Model | AUROC | AUPRC | Tham số | Thời gian |
|---|---|---|---|---|
| PlainCNN | 0,6970 | 0,6975 | 304.961 | 222 s |
| CNN+BiLSTM | 0,6860 | 0,6909 | 339.265 | 187 s |
| InceptionTime1D | 0,6847 | 0,6885 | 483.137 | 332 s |
| ResNet1D | 0,6819 | 0,6815 | 986.753 | 244 s |

> **Đừng vội kết luận PlainCNN là kiến trúc tốt nhất.** Bốn mô hình STEMI nằm trong khoảng AUROC
> 0,9256–0,9345, tức chênh nhau **0,009** — nhỏ hơn hẳn biên độ dao động giữa các lần chạy (~0,015,
> xem cảnh báo ngay dưới). Bài ACS còn sát hơn: 0,6819–0,6970.
>
> Điều đọc được một cách chắc chắn từ hai bảng này là: **tăng độ phức tạp kiến trúc không mang lại
> lợi ích đo được trên bài toán này.** ResNet1D nhiều hơn PlainCNN gấp 3 lần tham số mà không hơn.
> Đó là một kết luận có giá trị — nó nói rằng nút thắt nằm ở dữ liệu và cách đặt bài toán, không
> phải ở sức chứa của mô hình.

> ### ⚠️ Về tính lặp lại — đọc kỹ trước khi so sánh số
>
> Seed cố định (42) nhưng kết quả **vẫn không lặp lại chính xác**. Bốn lần chạy `full` liên tiếp cho
> AUROC STEMI lần lượt **0,9232 → 0,9254 → 0,9293 → 0,9362** (biên độ ~0,013).
>
> Ba nguyên nhân cộng dồn: `cudnn.benchmark = True` chọn thuật toán convolution không tất định, AMP
> làm phép cộng dấu phẩy động không kết hợp được, và early stopping chốt ở epoch khác nhau giữa các
> lần nên checkpoint tốt nhất cũng là model khác.
>
> **Hệ quả:** đừng diễn giải chênh lệch dưới ~0,015 AUROC như một cải thiện thật. Điều này áp dụng
> trực tiếp cho notebook 03/04: hai kiến trúc cách nhau dưới ngưỡng đó nên coi là **ngang nhau**.
> Muốn xếp hạng chắc chắn thì phải chạy nhiều seed rồi so trung bình ± độ lệch chuẩn.
>
> Cần tái lập chính xác thì đặt `torch.backends.cudnn.benchmark = False` và
> `torch.use_deterministic_algorithms(True)` ở mục 3 — đổi lại chậm hơn.

---

## 11. Notebook làm gì khác đi khi chạy trên Colab

Ghi lại để bạn biết chính xác chuyện gì đang diễn ra, không phải để thao tác.

1. **Tách đôi nơi lưu trữ.** Cache đặt ở `/content` (nhanh, mất khi hết phiên) và được sao lưu lên
   Drive; checkpoint ghi thẳng lên Drive để không mất khi đứt phiên.
2. **`num_workers = 2`** cho DataLoader. Trên Windows để 0 vì Windows tạo worker bằng `spawn`, mỗi
   worker phải nạp lại toàn bộ notebook; Linux dùng `fork` nên worker rẻ và có lợi thật.
3. **`cudnn.benchmark = True`.** Input cố định (12 × 5000) nên cuDNN chọn thuật toán nhanh nhất một
   lần rồi giữ nguyên. Đổi lại epoch đầu chậm hơn vài giây — bình thường, không phải trục trặc.

---

## 12. Hạn chế đã biết

- **Nhánh Colab chưa chạy thử trên Colab thật** (xem ghi chú đầu tài liệu).
- Mỗi phiên vẫn phải giải nén lại dữ liệu thô (~3–5 phút) kể cả khi cache đã có trên Drive, vì mục 4
  và mục 8 cần đọc file WFDB gốc.
- Chỉ một lần chia train/val, chưa cross-validation, nên chưa có khoảng tin cậy cho các metric.
- Notebook 03/04 chỉ chạy **một seed** cho mỗi kiến trúc — đủ để loại bỏ kiến trúc kém hẳn, chưa đủ
  để xếp hạng những cái sát nhau.
- Xác suất đầu ra chưa được hiệu chuẩn. `pos_weight` cố tình đẩy xác suất về phía lớp dương để bù
  mất cân bằng, nên muốn dùng xác suất tuyệt đối thì cần hiệu chuẩn lại (Platt/isotonic) trên một
  tập riêng.
- Chưa dùng biến lâm sàng (`age`, `gender`, `Time_Interval`) — mô hình thuần tín hiệu.
- Tập test ẩn 10% không được đụng tới ở bất kỳ bước nào.
