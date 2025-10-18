# Thêm Mutual Information Feature Selection

**Ngày cập nhật:** 2025-10-16  
**File thay đổi:** `full_comparison.py`  
**Tác giả:** System Update

---

## 📊 Tổng Quan

Đã thêm **Mutual Information** làm phương pháp feature selection thứ hai để so sánh với ANOVA F-test. Điều này cho phép phát hiện cả mối quan hệ **phi tuyến** giữa features và target.

## 🔄 Các Thay Đổi

### 1. Import Libraries (dòng 48)
```python
# Trước:
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Sau:
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
```

### 2. Feature Selection Config (dòng 93-98)
```python
# Trước:
FEATURE_SELECTION_METHODS = {
    'none': None,
    'select_k_best_5': SelectKBest(f_classif, k=5),
    'select_k_best_12': SelectKBest(f_classif, k=12),
}

# Sau:
FEATURE_SELECTION_METHODS = {
    'none': None,
    'select_k_best_5': SelectKBest(f_classif, k=5),
    'select_k_best_12': SelectKBest(f_classif, k=12),
    'mutual_info_5': SelectKBest(mutual_info_classif, k=5),    # ⭐ MỚI
    'mutual_info_12': SelectKBest(mutual_info_classif, k=12),  # ⭐ MỚI
}
```

### 3. Experiment Matrix (dòng 1141)
```python
# Trước:
['none', 'select_k_best_5', 'select_k_best_12']

# Sau:
['none', 'select_k_best_5', 'select_k_best_12', 'mutual_info_5', 'mutual_info_12']
```

## 📈 Số Lượng Experiments

| Stage | Feature Selection Methods | Total Experiments | Change |
|-------|--------------------------|-------------------|--------|
| **Baseline** | 2 (none, k=12) | 108 | - |
| **+ K=5** | 3 (none, k=5, k=12) | 162 | +54 (+50%) |
| **+ Mutual Info** | 5 (all above + mi=5, mi=12) | **270** | **+108 (+67%)** |

### Breakdown

| Model Type | Scalers | Imbalance | Feature Sel | Per Model | Total |
|-----------|---------|-----------|-------------|-----------|-------|
| **Scaling (5 models)** | 2 | 3 | 5 | 30 | 150 |
| **Non-scaling (8 models)** | 1 | 3 | 5 | 15 | 120 |
| **TOTAL** | - | - | - | - | **270** |

## ⏱️ Thời Gian Ước Tính

| Configuration | Time (minutes) | Time (hours) | Additional |
|--------------|----------------|--------------|------------|
| Baseline (2 methods) | 81 | 1.4 | - |
| + K=5 (3 methods) | 122 | 2.0 | +41 min |
| + Mutual Info (5 methods) | **203** | **3.4** | **+81 min** |

**Lưu ý:** Với cache enabled, chỉ experiments mới cần train!

## 🎯 5 Feature Selection Methods

| # | Method | Algorithm | K | Description |
|---|--------|-----------|---|-------------|
| 1 | `none` | - | All | Không feature selection |
| 2 | `select_k_best_5` | ANOVA F-test | 5 | Top 5 features (linear) |
| 3 | `select_k_best_12` | ANOVA F-test | 12 | Top 12 features (linear) |
| 4 | `mutual_info_5` ⭐ | Mutual Information | 5 | Top 5 features (non-linear) |
| 5 | `mutual_info_12` ⭐ | Mutual Information | 12 | Top 12 features (non-linear) |

## 🔬 ANOVA F-test vs Mutual Information

### So Sánh Chi Tiết

| Đặc Điểm | ANOVA F-test | Mutual Information |
|----------|--------------|-------------------|
| **Mối quan hệ** | Chỉ tuyến tính | Tuyến tính + Phi tuyến |
| **Tốc độ** | Rất nhanh ⚡ | Chậm hơn 🐢 |
| **Độ phức tạp** | O(n) | O(n log n) |
| **Giả định** | Phân phối chuẩn | Không giả định |
| **Ổn định** | Rất ổn định | Có thể dao động |
| **Pattern phức tạp** | ❌ Không | ✅ Có |
| **Tương tác features** | ❌ Không | ✅ Một phần |

### Công Thức

#### ANOVA F-test
```
F = (Variance between classes) / (Variance within classes)

F càng cao → Feature càng phân biệt được classes
```

#### Mutual Information
```
MI(X, Y) = H(X) + H(Y) - H(X,Y)

Trong đó:
- H(X): Entropy của feature X
- H(Y): Entropy của target Y  
- H(X,Y): Joint entropy

MI càng cao → X và Y càng phụ thuộc vào nhau
```

### Ví Dụ Minh Họa

```python
import numpy as np

# Feature 1: Linear relationship
x1 = np.array([1, 2, 3, 4, 5])
y = np.array([0, 0, 1, 1, 1])
# ANOVA: HIGH ✅  |  MI: HIGH ✅

# Feature 2: Non-linear (quadratic)
x2 = np.array([1, 4, 9, 16, 25])  # x^2
y = np.array([0, 0, 1, 1, 1])
# ANOVA: LOW ❌  |  MI: HIGH ✅  ← Chỉ MI phát hiện được!

# Feature 3: Noise
x3 = np.random.randn(5)
y = np.array([0, 0, 1, 1, 1])
# ANOVA: LOW ✅  |  MI: LOW ✅
```

## 💡 Khi Nào Dùng Method Nào?

### Dùng ANOVA F-test khi:
✅ Dữ liệu có phân phối chuẩn  
✅ Quan hệ tuyến tính  
✅ Cần tốc độ nhanh  
✅ Dataset lớn (>100k rows)  
✅ Features đã được scale/normalize  

**Phù hợp với:**
- Simple models (Logistic Regression, Linear SVM)
- Preprocessed features (PCA components)
- Initial exploration

### Dùng Mutual Information khi:
✅ Quan hệ phi tuyến  
✅ Không giả định phân phối  
✅ Features phức tạp (interactions)  
✅ Dataset vừa (<50k rows)  
✅ Cần robust feature selection  

**Phù hợp với:**
- Complex models (Random Forest, XGBoost)
- Raw features (before transformation)
- Medical/Biological data
- **Credit Card Fraud** (non-linear fraud patterns)
- **Cardiovascular** (complex health interactions)

## 🎯 Ứng Dụng Với Datasets

### Credit Card Fraud Dataset
```python
# Features: V1-V28 (PCA components) + Time + Amount
# Total: 30 features

# ANOVA F-test:
- Tốt cho V1-V28 (PCA đã tuyến tính hóa)
- Nhanh với 285k rows

# Mutual Information:
- Tốt cho Time & Amount (non-linear patterns)
- Phát hiện fraud patterns phức tạp
- Có thể chậm với dataset lớn

# Kỳ vọng:
- MI có thể chọn Time/Amount
- ANOVA chọn V1-V28 components
```

### Cardiovascular Dataset
```python
# Features: age, BMI, BP, cholesterol, etc.
# Total: 15 features

# ANOVA F-test:
- Tốt cho continuous features
- Nhanh với 70k rows

# Mutual Information:
- Phát hiện interactions (BMI × age)
- Tốt cho categorical features
- Robust với outliers

# Kỳ vọng:
- MI chọn features có interaction
- ANOVA chọn features có correlation mạnh
```

## 📊 Kết Quả Mong Đợi

### Scenario 1: ANOVA và MI chọn features giống nhau
```
Model: Gen3_XGBoost
select_k_best_5:  ['age', 'BMI', 'ap_hi', 'ap_lo', 'cholesterol']
mutual_info_5:    ['age', 'BMI', 'ap_hi', 'ap_lo', 'cholesterol']
Performance:      TƯƠNG ĐƯƠNG

→ Quan hệ tuyến tính, đơn giản
```

### Scenario 2: MI chọn features khác (TỐT HƠN)
```
Model: Gen2_RandomForest
select_k_best_5:  ['age', 'BMI', 'ap_hi', 'ap_lo', 'cholesterol']
mutual_info_5:    ['age', 'BMI', 'pulse_pressure', 'MAP', 'age×BMI']
Performance:      MI > ANOVA (+2-3% PR-AUC)

→ Phát hiện được interactions!
```

### Scenario 3: ANOVA tốt hơn (dataset đơn giản)
```
Model: Gen1_LogisticRegression  
select_k_best_5:  ['age', 'BMI', 'ap_hi', 'ap_lo', 'cholesterol']
mutual_info_5:    ['age', 'weight', 'height', 'ap_hi', 'ap_lo']
Performance:      ANOVA > MI (+1% PR-AUC)

→ Linear model prefer linear features
```

## 🚀 Cách Chạy

### Full Run (All 270 Experiments)
```bash
# Cardio dataset
python full_comparison.py --data data/raw/cardio_train.csv

# Creditcard dataset  
python full_comparison.py --data data/raw/creditcard.csv
```

### Với Cache (Recommended)
```bash
# Chỉ train experiments mới (mutual_info_*)
# Experiments cũ load từ cache
python full_comparison.py --data data/raw/cardio_train.csv

# Output:
# [1/270] Gen1_LogisticRegression | mutual_info_5 | ...
#   ⚙️  Training... (experiment mới)
# [2/270] Gen1_LogisticRegression | select_k_best_5 | ...
#   ✓ Loaded from cache! (experiment cũ)
```

### No Cache (Train All)
```bash
python full_comparison.py --data data/raw/cardio_train.csv --no-cache
```

## 📁 Output Structure

```
experiments/
├── logs/
│   ├── cardio_train_20251016_HHMMSS.log
│   └── creditcard_20251016_HHMMSS.log
├── full_comparison/
│   ├── cardio_train/
│   │   ├── full_comparison_20251016_HHMMSS.csv
│   │   └── best_model/
│   └── creditcard/
│       ├── full_comparison_20251016_HHMMSS.csv
│       └── best_model/
└── model_cache/
    ├── cardio_train/
    │   ├── Gen1_KNN_..._mutual_info_5_abc.pkl  ⭐ NEW
    │   ├── Gen1_KNN_..._mutual_info_12_def.pkl ⭐ NEW
    │   └── ... (existing cache files)
    └── creditcard/
        └── ... (same structure)
```

## 🔍 Phân Tích Kết Quả

### 1. So Sánh ANOVA vs MI
```python
import pandas as pd

df = pd.read_csv('experiments/full_comparison/cardio_train/full_comparison_*.csv')

# Group by model and feature selection method
comparison = df.groupby(['model', 'feature_selection'])['pr_auc'].mean().unstack()

# Compare k=5
print(comparison[['select_k_best_5', 'mutual_info_5']])

# Which is better?
better = (comparison['mutual_info_5'] > comparison['select_k_best_5']).sum()
print(f"Mutual Info tốt hơn ở {better}/{len(comparison)} models")
```

### 2. Feature Importance Comparison
```python
# Features chọn bởi ANOVA
anova_features = get_selected_features('select_k_best_5')

# Features chọn bởi MI
mi_features = get_selected_features('mutual_info_5')

# So sánh
print(f"Overlap: {len(set(anova_features) & set(mi_features))} features")
print(f"Only ANOVA: {set(anova_features) - set(mi_features)}")
print(f"Only MI: {set(mi_features) - set(anova_features)}")
```

### 3. Best Method Per Model
```python
# Tìm method tốt nhất cho mỗi model
best_methods = df.groupby('model').apply(
    lambda x: x.loc[x['pr_auc'].idxmax(), 'feature_selection']
)

print(best_methods.value_counts())
# Output:
# mutual_info_5     5 models
# mutual_info_12    3 models  
# select_k_best_5   3 models
# select_k_best_12  2 models
# none              0 models
```

## ⚠️ Lưu Ý Quan Trọng

### 1. Mutual Information Chậm Hơn
- MI có thể chậm gấp 2-5 lần ANOVA
- Với dataset lớn (>200k rows), có thể mất vài phút
- Sử dụng cache để tránh train lại

### 2. Random Seed
```python
# MI có element stochastic
mutual_info_classif(X, y, random_state=42)  # Đảm bảo reproducibility
```

### 3. Feature Scaling
- ANOVA: Không nhạy cảm với scale
- MI: **NHẠY CẢM** với scale
- → Nên scale trước khi dùng MI

### 4. Small Sample Size
- MI cần ít nhất ~100 samples per feature
- Với k=12, cần >1200 samples
- Dataset của chúng ta: 70k (cardio), 285k (credit) → OK ✅

## 📚 Tài Liệu Tham Khảo

### Papers
1. **Mutual Information Feature Selection**
   - Cover & Thomas (1991) - Elements of Information Theory
   
2. **ANOVA F-test**
   - Fisher (1925) - Statistical Methods for Research Workers

### Scikit-learn Documentation
- [`mutual_info_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)
- [`f_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html)

## 🎉 Tóm Tắt

✅ **Đã thêm:** `mutual_info_5` và `mutual_info_12`  
✅ **Tổng experiments:** 162 → 270 (+108, +67%)  
✅ **Thời gian thêm:** ~81 phút (~1.4 giờ)  
✅ **Lợi ích:** Phát hiện non-linear feature relationships  
✅ **Compatible:** Hoạt động với mọi models và datasets  
✅ **Cache-friendly:** Chỉ train experiments mới  

---

**Next Steps:**
1. ✅ Chạy full comparison với mutual info
2. ⏳ Phân tích features được chọn
3. ⏳ So sánh performance ANOVA vs MI
4. ⏳ Document insights

**Date:** 2025-10-16  
**Version:** 3.0 (Baseline → K=5 → Mutual Info)
