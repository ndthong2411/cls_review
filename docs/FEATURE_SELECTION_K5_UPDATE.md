# Cập Nhật: Thêm Feature Selection k=5

## Tổng Quan

Đã cập nhật `full_comparison.py` để thêm cấu hình feature selection với **k=5** (chọn 5 features quan trọng nhất) vào experiment matrix. Điều này cho phép so sánh hiệu suất giữa:
- Sử dụng tất cả features (none)
- Sử dụng 5 features tốt nhất (select_k_best_5) ⭐ MỚI
- Sử dụng 12 features tốt nhất (select_k_best_12)

## Thay Đổi Code

### File: `full_comparison.py` (dòng 1140)

**Trước:**
```python
for scaler, imbalance, feat_sel in product(
    scalers,
    ['none', 'smote', 'smote_enn'],
    ['none', 'select_k_best_12']  # Chỉ có 2 options
):
```

**Sau:**
```python
for scaler, imbalance, feat_sel in product(
    scalers,
    ['none', 'smote', 'smote_enn'],
    ['none', 'select_k_best_5', 'select_k_best_12']  # Thêm select_k_best_5
):
```

## So Sánh Số Lượng Experiments

| Metric | Trước | Sau | Thay Đổi |
|--------|-------|-----|----------|
| **Models cần scaling** | 5 models | 5 models | - |
| **Scalers** (cho models cần scaling) | 2 (standard, robust) | 2 | - |
| **Imbalance methods** | 3 (none, smote, smote_enn) | 3 | - |
| **Feature selection** | 2 (none, k=12) | 3 (none, k=5, k=12) | +1 ⭐ |
| | | | |
| **Experiments (scaling models)** | 5×2×3×2 = 60 | 5×2×3×3 = 90 | +30 |
| **Experiments (non-scaling models)** | 8×1×3×2 = 48 | 8×1×3×3 = 72 | +24 |
| **TỔNG EXPERIMENTS** | **108** | **162** | **+54 (+50%)** |

## Thời Gian Ước Tính

Với thời gian trung bình ~45 giây mỗi experiment:

| | Trước | Sau | Thêm |
|---|-------|-----|------|
| **Tổng thời gian** | ~81 phút | ~122 phút | +40 phút |
| **Tính theo giờ** | ~1.4 giờ | ~2.0 giờ | +0.6 giờ |

**Lưu ý:** Thời gian thực tế có thể thay đổi tùy thuộc vào:
- Dataset size (cardio: 70k rows, creditcard: 285k rows)
- Model complexity (Gen4 models chậm hơn Gen1)
- Hardware (GPU availability)
- Cache hits (cached experiments không cần train lại)

## Feature Selection Methods

### 1. `none` - Không Feature Selection
- Sử dụng **TẤT CẢ** features
- Cardio dataset: 15 features
- Creditcard dataset: 30 features
- **Ưu điểm:** Không mất thông tin
- **Nhược điểm:** Có thể overfitting, chậm hơn

### 2. `select_k_best_5` - Top 5 Features ⭐ MỚI
- Chọn **5 features quan trọng nhất** dựa trên F-statistic
- **Ưu điểm:** 
  - Rất nhanh (ít features)
  - Giảm overfitting
  - Dễ interpret
  - Tốt cho models đơn giản
- **Nhược điểm:** 
  - Có thể mất thông tin quan trọng
  - Không tốt cho dataset phức tạp

### 3. `select_k_best_12` - Top 12 Features
- Chọn **12 features quan trọng nhất**
- **Ưu điểm:**
  - Balance giữa performance và simplicity
  - Giữ được nhiều thông tin
  - Vẫn giảm được noise
- **Nhược điểm:**
  - Có thể vẫn hơi phức tạp cho một số models

## Lợi Ích

### 1. 🔍 Tìm Số Lượng Features Tối Ưu
So sánh trực tiếp hiệu suất với 3 mức độ feature selection khác nhau để tìm balance tốt nhất.

### 2. 📊 Hiểu Feature Importance
- Features nào được chọn trong top 5?
- Features nào trong top 12 nhưng không trong top 5?
- Liệu 5 features có đủ để đạt performance tốt?

### 3. ⚡ Performance vs Complexity
- Model với 5 features: nhanh, đơn giản, dễ deploy
- Model với 12 features: mạnh hơn nhưng phức tạp hơn
- Model với all features: mạnh nhất nhưng có thể overfit

### 4. 🎯 Tối Ưu Cho Use Case
- **Production deployment:** Ưu tiên k=5 (nhanh, đơn giản)
- **Research/Analysis:** Ưu tiên all features (đầy đủ thông tin)
- **Balance:** Ưu tiên k=12 (compromise tốt)

## Ví Dụ Experiments Mới

### Cardio Dataset
```
Gen1_LogisticRegression | Scale=standard | Imb=none | FeatSel=select_k_best_5
Gen1_LogisticRegression | Scale=standard | Imb=smote | FeatSel=select_k_best_5
Gen1_LogisticRegression | Scale=standard | Imb=smote_enn | FeatSel=select_k_best_5
Gen1_LogisticRegression | Scale=robust | Imb=none | FeatSel=select_k_best_5
...
Gen1_KNN | Scale=standard | Imb=none | FeatSel=select_k_best_5
...
Gen4_TabNet | Scale=none | Imb=smote_enn | FeatSel=select_k_best_5
```

Mỗi model sẽ có thêm 54 configurations với `select_k_best_5`.

## Kết Quả Mong Đợi

### Top Performers (Dự Đoán)

**Scenario 1: k=5 Wins**
```
Model: Gen1_KNN
Config: robust / smote_enn / select_k_best_5
PR-AUC: 0.8150
→ Chứng tỏ 5 features quan trọng là đủ!
```

**Scenario 2: k=12 Wins**
```
Model: Gen3_XGBoost
Config: none / smote / select_k_best_12
PR-AUC: 0.8200
→ Cần nhiều features hơn cho performance tốt nhất
```

**Scenario 3: All Features Wins**
```
Model: Gen4_PyTorch_MLP
Config: standard / none / none
PR-AUC: 0.8250
→ Deep learning tận dụng tốt nhiều features
```

## So Sánh Chi Tiết

### Expected Feature Selection Impact

| Dataset | Total Features | k=5 | k=12 | All |
|---------|---------------|-----|------|-----|
| **Cardio** | 15 | 33% | 80% | 100% |
| **Creditcard** | 30 | 17% | 40% | 100% |

### Model Type Predictions

| Model Type | Best k | Lý Do |
|------------|--------|-------|
| **Simple (Gen1)** | k=5 hoặc k=12 | Tránh overfitting |
| **Ensemble (Gen2)** | k=12 hoặc all | Handle được nhiều features |
| **Boosting (Gen3)** | k=12 hoặc all | Feature importance tự động |
| **Deep Learning (Gen4)** | all | Học được feature interactions |

## Cách Phân Tích Kết Quả

### 1. So Sánh Trong Cùng Model
```python
# Ví dụ: Gen1_LogisticRegression
df[df['model'] == 'Gen1_LogisticRegression'].groupby('feature_selection')['pr_auc'].mean()

# Output:
# none               0.7580
# select_k_best_5    0.7650  ← Tốt hơn với k=5!
# select_k_best_12   0.7620
```

### 2. So Sánh Across Models
```python
# Model nào benefit nhất từ feature selection?
best_by_feat_sel = df.groupby(['model', 'feature_selection'])['pr_auc'].max().unstack()
best_by_feat_sel['best_k'] = best_by_feat_sel.idxmax(axis=1)
```

### 3. Tìm Sweet Spot
```python
# Số features tối ưu cho từng generation?
df.groupby(['generation', 'feature_selection'])['pr_auc'].mean()
```

## Running the Experiments

### Chạy Đầy Đủ
```bash
# Cardio dataset
python full_comparison.py --data data/raw/cardio_train.csv

# Creditcard dataset
python full_comparison.py --data data/raw/creditcard.csv
```

### Chạy Với Cache
```bash
# Nếu đã chạy trước đó, cached experiments sẽ được load
python full_comparison.py --data data/raw/cardio_train.csv
# Output: "✓ Loaded from cache!" cho các experiments đã train
```

### Chạy Không Cache (Train Lại Tất Cả)
```bash
python full_comparison.py --data data/raw/cardio_train.csv --no-cache
```

## Cache Organization

Cache directory structure giờ sẽ bao gồm cả k=5:

```
experiments/model_cache/
├── cardio_train/
│   ├── Gen1_KNN_standard_smote_none_abc123.pkl
│   ├── Gen1_KNN_standard_smote_select_k_best_5_def456.pkl  ⭐ NEW
│   ├── Gen1_KNN_standard_smote_select_k_best_12_ghi789.pkl
│   └── ...
└── creditcard/
    ├── Gen1_KNN_standard_smote_none_xyz123.pkl
    ├── Gen1_KNN_standard_smote_select_k_best_5_uvw456.pkl  ⭐ NEW
    └── ...
```

## Troubleshooting

### Q: "ValueError: k should be >=0 and <= n_features"
**A:** Xảy ra khi k > số features. 
- Cardio: 15 features → k=5 OK, k=12 OK
- Creditcard: 30 features → k=5 OK, k=12 OK
- Nếu dataset có < 5 features, cần giảm k

### Q: Thời gian chạy quá lâu?
**A:** Một số tùy chọn:
- Chạy với cache enabled (mặc định)
- Chỉ chạy một subset models
- Sử dụng GPU (nếu có)

### Q: Làm sao biết k nào tốt nhất?
**A:** Xem TOP 10 trong results:
```bash
python analyze_results.py experiments/full_comparison/cardio_train/full_comparison_*.csv
```

## Tóm Tắt

✅ **Đã thêm:** `select_k_best_5` vào experiment matrix  
✅ **Tổng experiments:** 108 → 162 (+54)  
✅ **Thời gian thêm:** ~40 phút  
✅ **Lợi ích:** So sánh 3 levels của feature selection  
✅ **Compatible:** Hoạt động với cả cardio và creditcard datasets  
✅ **Cache-friendly:** Chỉ train experiments mới, reuse cache cũ  

---

**Date:** 2025-10-16  
**File Modified:** `full_comparison.py`  
**Lines Changed:** 1140  
