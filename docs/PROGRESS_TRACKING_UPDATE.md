# Progress Tracking Update - full_comparison.py

## Summary
Đã thêm tqdm progress bars vào `full_comparison.py` để tracking real-time progress của model training.

## Changes Made

### 1. Import tqdm
```python
from tqdm import tqdm
```

### 2. Main Experiment Loop Progress
- Thêm progress bar cho vòng lặp chính (Overall Progress)
- Hiển thị số experiments đã hoàn thành / tổng số
- Tốc độ: exp/s (experiments per second)
- Width: 120 characters

```python
for idx, exp in enumerate(tqdm(experiments, desc="Overall Progress", ncols=120, unit="exp"), 1):
    tqdm.write(f"\n[{idx}/{len(experiments)}] {exp['model_name']} | "
               f"Scale: {exp['scaler']} | Imb: {exp['imbalance']} | FeatSel: {exp['feature_selection']}")
```

### 3. Cross-Validation Folds Progress
- Thêm nested progress bar cho CV folds
- Progress bar tự động biến mất sau khi hoàn thành (leave=False)
- Hiển thị tốc độ: it/s (iterations per second)
- Width: 100 characters

```python
cv_splits = list(cv.split(X_train, y_train))
for fold, (train_idx, val_idx) in enumerate(tqdm(cv_splits, desc=f"  CV Folds", leave=False, ncols=100), 1):
    # Training code...
```

### 4. Output Formatting
- Sử dụng `tqdm.write()` thay vì `print()` để tránh breaking progress bars
- Thay thế Unicode characters (✓) bằng ASCII ([OK], [CACHE]) để tương thích với Windows

```python
tqdm.write(f"  [OK] PR-AUC: {result['pr_auc']:.4f} | ...")
tqdm.write("  [CACHE] Loaded from cache!")
```

## Visual Example

```
Overall Progress:  40%|##########                    | 2/5 [00:01<00:02, 1.08exp/s]

[2/5] Gen2_RandomForest | Scale: standard | Imb: smote | FeatSel: select_k_best_5
  CV Folds: 60%|########################      | 3/5 [00:05<00:03, 0.67it/s]
```

## Benefits

1. **Real-time Progress Tracking**: Biết được model đang chạy và còn bao lâu nữa
2. **Nested Progress**: Tracking cả overall progress và CV fold progress
3. **Time Estimation**: Hiển thị ETA (estimated time of arrival)
4. **Clean Output**: Output không bị break progress bars nhờ tqdm.write()
5. **Windows Compatible**: Không dùng Unicode characters

## Testing

Chạy test script:
```bash
python test_progress.py
```

## No Breaking Changes

- Tất cả functionality cũ vẫn hoạt động bình thường
- Không làm thay đổi logic của bất kỳ model nào
- Cache functionality vẫn hoạt động
- Log files vẫn được tạo đúng format

## Files Modified

1. `full_comparison.py`:
   - Added tqdm import (line 34)
   - Added progress bar for experiments loop (line 1169)
   - Added progress bar for CV folds (line 648)
   - Changed print to tqdm.write in cache loading (line 625)
   - Changed print to tqdm.write in results output (line 1186)

2. New test file: `test_progress.py` - Demonstrates progress bars functionality

## Notes

- Progress bars work in both console and IDE terminals
- Progress bars automatically hide when experiments are loaded from cache
- ETA updates dynamically based on training speed
- Nested progress bars don't interfere with each other
