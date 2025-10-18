# Báo Cáo Phân Tích Experiment Logs

## Tổng Quan

Phân tích chi tiết 2 experiment logs từ nghiên cứu so sánh các mô hình machine learning trên 2 bộ dữ liệu:
- **cardio_train**: Dữ liệu bệnh tim mạch (cân bằng class)
- **creditcard**: Dữ liệu phát hiện gian lận thẻ tín dụng (mất cân bằng nghiêm trọng)

---

## 1. CARDIO_TRAIN Dataset

### Thông tin Dataset
- **Tên file**: `cardio_train_20251018_022847.log`
- **Thời gian chạy**: 2025-10-18 02:28:47
- **Kích thước**: 70,000 mẫu × 13 cột (sử dụng 15 features sau feature engineering)

### Phân Bố Class
- **Positive**: 34,979 (50.0%)
- **Negative**: 35,021 (50.0%)
- **Tỷ lệ mất cân bằng**: 1:1.0 ✅ (Cân bằng hoàn hảo)

### Train/Test Split
- **Train**: 56,000 mẫu (27,983 positive)
- **Test**: 14,000 mẫu (6,996 positive)

### Experiments
- **Tổng số thí nghiệm**: 270
- **Đã cache**: 270 (100%)
- **Parse thành công**: 270

### Kết Quả Thống Kê
| Metric | Giá trị |
|--------|---------|
| Average PR-AUC | 0.7642 |
| **Best PR-AUC** | **0.8023** |
| Worst PR-AUC | 0.7068 |
| Average F1 | 0.6996 |
| Avg time/experiment | 238.37s |
| **Total runtime** | **17.9 giờ** |

### 🏆 Top 5 Configurations (theo PR-AUC)

1. **Gen1_DecisionTree** - PR-AUC: 0.8023 | F1: 0.7172
   - Config: `none | smote_enn | mutual_info_12`

2. **Gen1_KNN** - PR-AUC: 0.8022 | F1: 0.7148
   - Config: `robust | smote_enn | select_k_best_12`

3. **Gen1_KNN** - PR-AUC: 0.8011 | F1: 0.7153
   - Config: `robust | smote_enn | mutual_info_12`

4. **Gen1_KNN** - PR-AUC: 0.8003 | F1: 0.7151
   - Config: `robust | smote_enn | none`

5. **Gen1_DecisionTree** - PR-AUC: 0.8000 | F1: 0.7149
   - Config: `none | smote_enn | select_k_best_12`

### 🔝 Best Model Per Generation

| Generation | Model | PR-AUC | F1 | Best Config |
|------------|-------|--------|-----|-------------|
| **Gen1** | DecisionTree | **0.8023** | 0.7172 | none \| smote_enn \| mutual_info_12 |
| Gen2 | GradientBoosting | 0.7865 | 0.7247 | none \| none \| none |
| Gen3 | CatBoost | 0.7864 | 0.7253 | none \| none \| mutual_info_12 |
| Gen4 | PyTorch_MLP | 0.7839 | 0.7237 | standard \| smote \| select_k_best_12 |

### 💡 Insights - Cardio Train
- ✅ **Model tốt nhất**: Gen1 Decision Tree đơn giản với SMOTE-ENN
- ✅ **SMOTE-ENN** rất hiệu quả cho dataset cân bằng này
- ✅ Feature selection (mutual_info) cải thiện performance
- ⚠️ Các model phức tạp (Gen3, Gen4) KHÔNG tốt hơn model đơn giản
- ⚠️ Overfitting risk với deep learning models

---

## 2. CREDITCARD Dataset

### Thông tin Dataset
- **Tên file**: `creditcard_20251018_204737.log`
- **Thời gian chạy**: 2025-10-18 20:47:37
- **Kích thước**: 284,807 mẫu × 31 cột (30 features)

### Phân Bố Class
- **Positive (Fraud)**: 492 (0.2%)
- **Negative (Normal)**: 284,315 (99.8%)
- **Tỷ lệ mất cân bằng**: 1:577.9 ⚠️ (Mất cân bằng NGHIÊM TRỌNG)

### Train/Test Split
- **Train**: 227,845 mẫu (394 positive)
- **Test**: 56,962 mẫu (98 positive)

### Experiments
- **Tổng số thí nghiệm**: 270
- **Đã cache**: 164 (60.7%)
- **Parse thành công**: 261

### Kết Quả Thống Kê
| Metric | Giá trị |
|--------|---------|
| Average PR-AUC | 0.6944 |
| **Best PR-AUC** | **0.8693** |
| Worst PR-AUC | 0.0778 |
| Average F1 | 0.4363 |
| Avg time/experiment | 613.82s |
| **Total runtime** | **44.5 giờ** |

### 🏆 Top 5 Configurations (theo PR-AUC)

1. **Gen1_KNN** - PR-AUC: 0.8693 | F1: 0.8595
   - Config: `standard | none | select_k_best_12`

2. **Gen1_KNN** - PR-AUC: 0.8689 | F1: 0.8593
   - Config: `robust | none | select_k_best_12`

3. **Gen1_KNN** - PR-AUC: 0.8689 | F1: 0.8615
   - Config: `robust | none | mutual_info_12`

4. **Gen1_KNN** - PR-AUC: 0.8670 | F1: 0.8608
   - Config: `standard | none | mutual_info_12`

5. **Gen1_KNN** - PR-AUC: 0.8632 | F1: 0.8403
   - Config: `standard | none | none`

### 🔝 Best Model Per Generation

| Generation | Model | PR-AUC | F1 | Best Config |
|------------|-------|--------|-----|-------------|
| **Gen1** | KNN | **0.8693** | **0.8595** | standard \| none \| select_k_best_12 |
| Gen2 | RandomForest | 0.8425 | 0.7961 | none \| smote \| none |
| Gen3 | XGBoost | 0.8359 | 0.8429 | none \| none \| none |
| Gen4 | PyTorch_MLP | 0.7882 | 0.7625 | robust \| smote_enn \| none |

### 💡 Insights - Credit Card
- ✅ **Model tốt nhất**: Gen1 KNN với feature selection
- ✅ **Feature selection QUAN TRỌNG** - K=12 features tối ưu
- ⚠️ **KHÔNG nên dùng SMOTE** cho top performers (none imbalance handling)
- ⚠️ Deep learning (Gen4) performance kém hơn nhiều
- ✅ XGBoost (Gen3) balanced tốt giữa PR-AUC và F1

---

## 3. So Sánh 2 Datasets

| Metric | Cardio Train | Credit Card | Winner |
|--------|--------------|-------------|--------|
| Class Imbalance | 1:1.0 | 1:577.9 | Cardio ✅ |
| **Best PR-AUC** | 0.8023 | **0.8693** | **Credit ✅** |
| Avg PR-AUC | 0.7642 | 0.6944 | Cardio ✅ |
| Best Model | DecisionTree | KNN | - |
| Total Runtime | 17.9 giờ | 44.5 giờ | Cardio ✅ |
| Variance (max-min) | 0.0955 | 0.7915 | Cardio ✅ |

### Phân Tích Chuyên Sâu

#### Cardio Train (Balanced Dataset)
- ✅ **Model stability cao**: Variance thấp (0.0955)
- ✅ **Preprocessing quan trọng**: SMOTE-ENN cải thiện rõ rệt
- ✅ **Simple is better**: Gen1 models outperform Gen3/Gen4
- ⏱️ **Runtime hợp lý**: Trung bình ~4 phút/experiment

#### Credit Card (Highly Imbalanced)
- ⚠️ **High variance**: (0.7915) - một số config thất bại hoàn toàn
- ✅ **Feature selection critical**: K=12 là sweet spot
- ⚠️ **SMOTE có thể phản tác dụng**: Top models không dùng SMOTE
- ⏱️ **Expensive**: Trung bình ~10 phút/experiment (2.5x chậm hơn)
- 🎯 **Best strategy**: KNN + feature selection, NO imbalance handling

---

## 4. Kết Luận & Khuyến Nghị

### 🎯 Key Findings

1. **Simple Models Win**
   - Gen1 models (DecisionTree, KNN) đạt best performance trên CẢ HAI datasets
   - Gen3/Gen4 (boosting, deep learning) không justify complexity

2. **Preprocessing Strategy Phụ Thuộc Dataset**
   - **Balanced data** (Cardio): SMOTE-ENN hiệu quả
   - **Imbalanced data** (Credit): Không dùng SMOTE cho top performance

3. **Feature Selection Quan Trọng**
   - K=12 features là optimal cho cả 2 datasets
   - Mutual information và SelectKBest đều hiệu quả

4. **Computational Cost**
   - Credit Card dataset tốn 2.5x thời gian hơn
   - Imbalance ratio cao → training time tăng

### 📊 Recommendations

#### Cho Cardio Dataset:
```python
best_config = {
    'model': 'DecisionTree',
    'scaler': 'none',
    'imbalance': 'smote_enn',
    'feature_selection': 'mutual_info_12',
    'expected_pr_auc': 0.8023
}
```

#### Cho Credit Card Dataset:
```python
best_config = {
    'model': 'KNN',
    'scaler': 'standard',
    'imbalance': 'none',
    'feature_selection': 'select_k_best_12',
    'expected_pr_auc': 0.8693
}
```

### 🔬 Future Work
1. ❓ Tại sao Gen1 models outperform advanced models?
2. ❓ Investigate SMOTE failure trên highly imbalanced data
3. ❓ Ensemble Gen1 models thay vì dùng Gen3/Gen4?
4. ✅ K=12 features có phải là universal optimal?

---

## 5. Visualizations

Đã tạo các biểu đồ phân tích tại:
- `analysis_output/cardio_train/`
  - `cardio_train_pr_auc_by_model.png`
  - `cardio_train_preprocessing_impact.png`
  - `cardio_train_pr_auc_vs_f1.png`
  - `cardio_train_heatmap_model_scaler.png`

- `analysis_output/creditcard/`
  - `creditcard_pr_auc_by_model.png`
  - `creditcard_preprocessing_impact.png`
  - `creditcard_pr_auc_vs_f1.png`
  - `creditcard_heatmap_model_scaler.png`

---

**Generated by**: `analyze_log_files.py`
**Analysis date**: 2025-10-19
