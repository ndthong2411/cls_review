# Progress Tracking Guide

## Quick Start

Khi chạy `full_comparison.py`, bạn sẽ thấy 2 levels của progress tracking:

### 1. Overall Progress (Main Loop)
```
Overall Progress:  40%|##########          | 2/5 [00:01<00:02, 1.08exp/s]
```
- Hiển thị: Tổng số experiments đã hoàn thành
- Unit: exp/s (experiments per second)
- ETA: Ước tính thời gian còn lại

### 2. CV Folds Progress (Nested)
```
  CV Folds: 60%|####################    | 3/5 [00:05<00:03, 0.67it/s]
```
- Hiển thị: Cross-validation folds đang chạy
- Unit: it/s (iterations per second)
- Tự động ẩn khi hoàn thành (leave=False)

## Example Output

```
Overall Progress:  10%|######                              | 15/150 [05:23<48:35, 21.60s/exp]

[15/150] Gen3_XGBoost | Scale: standard | Imb: smote | FeatSel: mutual_info_5
  CV Folds: 40%|############            | 2/5 [00:45<01:07, 22.50s/it]
  [OK] PR-AUC: 0.8234 | Sens: 0.7654 | Spec: 0.8912 | F1: 0.7234 | Time: 112.3s

[16/150] Gen3_LightGBM | Scale: none | Imb: smote_enn | FeatSel: select_k_best_12
  [CACHE] Loaded from cache!
  [OK] PR-AUC: 0.8156 | Sens: 0.7523 | Spec: 0.8845 | F1: 0.7145 | Time: 0.0s
```

## Features

✅ **Real-time Progress**: Biết chính xác model nào đang train
✅ **Time Estimation**: ETA cho từng level
✅ **Cache Detection**: Hiển thị khi load từ cache
✅ **Clean Output**: Không break progress bars
✅ **Windows Compatible**: Không dùng Unicode characters

## Cache Behavior

Khi experiments được load từ cache:
- CV Folds progress bar không xuất hiện
- Thời gian training = 0.0s
- Hiển thị `[CACHE] Loaded from cache!`
- Overall progress vẫn update bình thường

## Running the Script

```bash
# Standard run
python full_comparison.py

# With specific dataset
python full_comparison.py --data data/raw/creditcard.csv

# Without cache (full rerun)
python full_comparison.py --no-cache

# Check cached experiments
python full_comparison.py --list-cache
```

## Interpreting Progress

### Progress Bar Format
```
Description: XX%|###########     | current/total [elapsed<remaining, speed]
```

- **Description**: "Overall Progress" hoặc "CV Folds"
- **XX%**: Phần trăm hoàn thành
- **###|###**: Visual progress bar
- **current/total**: Số hiện tại / tổng số
- **[elapsed<remaining]**: Thời gian đã chạy < thời gian còn lại
- **speed**: Tốc độ (exp/s hoặc it/s)

### Example Interpretation
```
Overall Progress:  25%|#######            | 37/150 [18:45<56:15, 29.87s/exp]
```
- Đã hoàn thành: 37/150 experiments (25%)
- Thời gian đã chạy: 18 phút 45 giây
- Thời gian còn lại ước tính: 56 phút 15 giây
- Tốc độ trung bình: 29.87 giây/experiment

## Troubleshooting

### Progress bars không hiển thị đúng?
- Kiểm tra terminal hỗ trợ ANSI escape codes
- Thử chạy trên console/terminal khác
- Progress bars hoạt động tốt trên: CMD, PowerShell, VSCode Terminal

### Output bị overlap?
- Đảm bảo terminal window đủ rộng (ít nhất 120 characters)
- Progress bars tự động adjust nếu window nhỏ hơn

### Muốn tắt progress bars?
Nếu muốn chạy không có progress bars (ví dụ: redirect output):
```python
# Trong full_comparison.py, comment out tqdm:
# for idx, exp in enumerate(tqdm(experiments, ...), 1):
for idx, exp in enumerate(experiments, 1):
```

## Technical Details

### Implementation
- Library: `tqdm >= 4.66`
- Nested progress bars với `leave=False`
- Output via `tqdm.write()` để không break progress
- ASCII-only characters cho Windows compatibility

### Performance Impact
- Negligible (<0.1% overhead)
- Progress updates every ~0.1s
- No impact on model training speed

## Test Script

Chạy test để xem demo:
```bash
python test_progress.py
```

Test này sẽ simulate 5 experiments với 3 CV folds mỗi cái, giúp bạn thấy progress bars hoạt động như thế nào.
