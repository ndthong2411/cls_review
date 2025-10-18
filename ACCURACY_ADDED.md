# Cập Nhật: Thêm Accuracy vào Full Comparison

## Thay Đổi

Đã thêm **Accuracy** metric vào tất cả output của `full_comparison.py`

### 1. Output Mỗi Experiment
**Trước:**
```
[OK] PR-AUC: 0.7580 | Sens: 0.6525 | Spec: 0.7960 | F1: 0.7028 | Time: 0.2s
```

**Sau:**
```
[OK] PR-AUC: 0.7580 | Acc: 0.7243 | Sens: 0.6525 | Spec: 0.7960 | F1: 0.7028 | Time: 0.2s
```

### 2. Top 10 Configurations
**Thêm dòng:**
```
  Accuracy:    0.7243 ± 0.0012
```

### 3. Generation Comparison
**Trước:**
```
  Sensitivity: 0.6525 | Specificity: 0.7960 | F1: 0.7028
```

**Sau:**
```
  Accuracy: 0.7243 | Sensitivity: 0.6525 | Specificity: 0.7960 | F1: 0.7028
```

### 4. Final Test Metrics
**Thêm:**
```
Accuracy:         0.7243
Balanced Acc:     0.7242
```

### 5. Prediction Script Header
**Thêm:**
```
Accuracy (Test): 0.7243
```

## Metric Đã Có Sẵn

Accuracy đã được tính trong `calculate_metrics()` từ trước:
```python
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    ...
}
```

Chỉ cần thêm vào output display thôi.

## Ví Dụ Output Mới

```
[1/270] Gen1_LogisticRegression | Scale: standard | Imb: none | FeatSel: none
  [CACHE] Loaded from cache!
  [OK] PR-AUC: 0.7580 | Acc: 0.7243 | Sens: 0.6525 | Spec: 0.7960 | F1: 0.7028 | Time: 0.2s
```

## Lưu Ý

- **Accuracy** sẽ cao khi dataset cân bằng (cardio_train ~72%)
- **Accuracy** sẽ misleading khi imbalanced (creditcard có thể ~99% nhưng vô dụng)
- Vẫn ưu tiên **PR-AUC** làm metric chính cho ranking
- **Balanced Accuracy** hữu ích hơn cho imbalanced data

## Test

```bash
python full_comparison.py --data data/raw/cardio_train.csv
```

Output sẽ hiển thị accuracy ở mọi vị trí quan trọng.
