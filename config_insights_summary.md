# 📊 PHÂN TÍCH ẢNH HƯỞNG CỦA CONFIG LÊN PERFORMANCE

## 🎯 TÓM TẮT TỔNG QUAN

| Dataset | Total Experiments | Best F1 | Best Model | Best Config |
|---------|------------------|---------|------------|-------------|
| **Cardio** | 270 | **0.7260** | LightGBM/GradientBoosting | none + select_k_best_12 + smote |
| **CreditCard** | 270 | **0.8615** | KNN | robust + mutual_info_12 + none |

---

## 1️⃣ SCALING (Chuẩn hóa dữ liệu)

### **CARDIO Dataset** (Balanced - 50/50)

| Scaler | F1 Mean | ROC-AUC Mean | Kết luận |
|--------|---------|--------------|----------|
| **none** ✅ | **0.7065** | **0.7826** | TỐT NHẤT |
| standard | 0.6969 | 0.7674 | Kém hơn 1.4% |
| robust | 0.6911 | 0.7713 | Kém hơn 2.2% |

**💡 INSIGHT:**
- Tree-based models (LightGBM, RandomForest, XGBoost) **KHÔNG CẦN** scaling
- Scaling thậm chí làm **GIẢM** performance vì tree-based models chiếm đa số
- **Khuyến nghị:** Dùng `scaler=none` cho Cardio dataset

---

### **CREDITCARD Dataset** (Imbalanced - 578:1)

| Scaler | F1 Mean | ROC-AUC Mean | Kết luận |
|--------|---------|--------------|----------|
| **none** ✅ | **0.5284** | 0.9321 | TỐT NHẤT cho F1 |
| standard | 0.3675 | **0.9484** | Tốt cho ROC-AUC nhưng F1 thấp |
| robust | 0.3585 | 0.9431 | Tương tự standard |

**💡 INSIGHT:**
- **NGHỊCH LÝ:** `none` cho F1 cao nhất (0.5284) nhưng ROC-AUC thấp hơn
- **LÝ DO:** Tree-based models vẫn chiếm đa số → không cần scaling
- **Scaling có ích cho:** KNN, SVM, Neural Networks (cần chuẩn hóa)
- **Top model (KNN)** cần `robust/standard` → ROC-AUC = 0.92

**⚠️ QUAN TRỌNG:**
```
none scaler + tree-based → F1 trung bình cao
robust scaler + KNN → F1 BEST (0.8615)
```

**Khuyến nghị:**
- Tree-based models: `scaler=none`
- Distance-based (KNN): `scaler=robust` hoặc `standard`

---

## 2️⃣ FEATURE SELECTION (Chọn features quan trọng)

### **CARDIO Dataset**

| Feature Selection | F1 Mean | ROC-AUC Mean | Features giữ lại |
|-------------------|---------|--------------|------------------|
| **mutual_info_12** ✅ | **0.7102** | 0.7869 | 12/15 (80%) |
| none | 0.7101 | 0.7868 | 15/15 (100%) |
| select_k_best_12 | 0.7095 | **0.7870** | 12/15 (80%) |
| select_k_best_5 | 0.7080 | 0.7696 | 5/15 (33%) |
| mutual_info_5 | 0.6600 | 0.7461 | 5/15 (33%) |

**💡 INSIGHT:**
- **Giữ 12 features (80%):** Gần như tốt bằng dùng ALL features
- **Chỉ giữ 5 features (33%):** Giảm 5-7% performance
- **Mutual Info vs SelectKBest:** Khác biệt KHÔNG đáng kể (+0.0007)

**Variance Analysis:**
```
Feature Selection có ảnh hưởng LỚN NHẤT (std = 0.0221)
> Scaler (0.0077) > Model (0.0115) > Imbalance (0.0002)
```

**Khuyến nghị:**
- Production: Dùng `mutual_info_12` hoặc `select_k_best_12` (giảm 20% features, performance giống nhau)
- Fast inference: Dùng `select_k_best_5` (giảm 67% features, chấp nhận mất 5% F1)

---

### **CREDITCARD Dataset**

| Feature Selection | F1 Mean | ROC-AUC Mean | Features giữ lại |
|-------------------|---------|--------------|------------------|
| **none** ✅ | **0.5728** | 0.9495 | 30/30 (100%) |
| mutual_info_12 | 0.4952 | **0.9523** | 12/30 (40%) |
| select_k_best_12 | 0.4850 | 0.9468 | 12/30 (40%) |
| select_k_best_5 | 0.3166 | 0.9259 | 5/30 (17%) |
| mutual_info_5 | 0.3131 | 0.9239 | 5/30 (17%) |

**💡 INSIGHT:**
- **Dữ liệu imbalanced ĐẶC BIỆT:** Feature selection làm **GIẢM MẠNH** F1-score
- Giảm từ 30 → 12 features: **-15% F1** (0.5728 → 0.4952)
- Giảm từ 30 → 5 features: **-45% F1** (0.5728 → 0.3166)

**⚠️ TẠI SAO?**
```
Credit Card Fraud có rất ít positive samples (0.17%)
→ Mỗi feature đều chứa thông tin quý giá
→ Loại bỏ features = Mất thông tin phát hiện fraud
```

**Khuyến nghị:** Dùng **ALL features (`none`)** cho imbalanced dataset

---

## 3️⃣ IMBALANCE HANDLING (Xử lý mất cân bằng)

### **CARDIO Dataset** (Đã cân bằng 50/50)

| Imbalance Handler | F1 Mean | Precision | Recall (Sensitivity) | Samples thay đổi |
|-------------------|---------|-----------|----------------------|------------------|
| smote ✅ | 0.6997 | 0.7511 | 0.6585 | +34 (+0.1%) |
| none | 0.6996 | 0.7512 | 0.6583 | baseline |
| smote_enn | 0.6994 | 0.7585 | 0.6545 | -37,481 (-67%) |

**💡 INSIGHT:**
- Data đã cân bằng → SMOTE/SMOTEENN **KHÔNG CẢI THIỆN** nhiều
- Variance nhỏ nhất (std = 0.0002) → Imbalance handler ít ảnh hưởng nhất
- SMOTEENN tăng Precision (+0.7%) nhưng giảm Recall và **mất 67% data**

**Khuyến nghị:**
- Dùng `none` (đơn giản, nhanh) hoặc `smote` (cải thiện nhẹ 0.01%)
- **TRÁNH** `smote_enn` (mất quá nhiều data)

---

### **CREDITCARD Dataset** (Cực kỳ imbalanced 578:1)

| Imbalance Handler | F1 Mean | Precision | Recall (Sensitivity) | ROC-AUC |
|-------------------|---------|-----------|----------------------|---------|
| **none** ✅ | **0.5063** | 0.4880 | **0.7841** | 0.9288 |
| smote | 0.4137 | 0.3318 | **0.8302** | **0.9451** |
| smote_enn | 0.3896 | 0.3033 | **0.8383** | 0.9452 |

**💡 NGHỊCH LÝ QUAN TRỌNG:**

```
❌ Suy nghĩ sai: "Data imbalanced → PHẢI dùng SMOTE"
✅ Thực tế: none cho F1 tốt nhất (0.5063 vs 0.4137)
```

**PHÂN TÍCH CHI TIẾT:**

| Method | Precision | Recall | F1 | Trade-off |
|--------|-----------|--------|-----|-----------|
| **none** | 0.4880 | 0.7841 | **0.5063** | Balanced |
| smote | 0.3318 ↓32% | 0.8302 ↑6% | 0.4137 ↓18% | Recall tăng nhưng Precision **SỤT MẠNH** |
| smote_enn | 0.3033 ↓38% | 0.8383 ↑7% | 0.3896 ↓23% | Tồi nhất |

**⚠️ TẠI SAO SMOTE LẠI TỆ HƠN?**

1. **Synthetic samples không tốt:**
   - SMOTE tạo 227,057 fake fraud samples từ chỉ 394 real samples
   - Tỷ lệ 577:1 → Model học từ **99.8% data giả**
   - Samples giả không phản ánh đúng pattern thực tế

2. **Precision sụt mạnh:**
   - SMOTE → Precision = 0.3318 (70% predictions là FALSE POSITIVE!)
   - Trong fraud detection: False Positive = Khách hàng tốt bị chặn → MẤT KHÁCH

3. **ROC-AUC cao hơn nhưng F1 thấp hơn:**
   - SMOTE: ROC-AUC = 0.9451 (cao hơn) nhưng F1 = 0.4137 (thấp hơn)
   - → Model phân biệt class tốt nhưng threshold không tối ưu cho F1

**Imbalance by Generation:**

| Generation | none | smote | smote_enn | Best choice |
|------------|------|-------|-----------|-------------|
| Gen1 (Basic) | 0.4372 | 0.2497 | 0.2276 | **none** |
| Gen2 (Ensemble) | 0.6151 | 0.4310 | 0.4126 | **none** |
| Gen3 (Boosting) | 0.6096 | **0.6366** | 0.5872 | **smote** (duy nhất!) |
| Gen4 (Deep Learning) | 0.2646 | 0.4239 | 0.4083 | smote |

**💡 KEY FINDING:**
- Gen3 Boosting (XGBoost, LightGBM) **có lợi từ SMOTE**
- Các thế hệ khác: `none` tốt hơn

**Khuyến nghị:**
- Tree-based (Gen1, Gen2): Dùng `none`
- Boosting (Gen3): Dùng `smote`
- Deep Learning (Gen4): Dùng `smote` (nhưng performance vẫn thấp)

---

## 4️⃣ MODEL PERFORMANCE

### **CARDIO Dataset**

| Rank | Model | F1 Mean | F1 Max | Speed | Khuyến nghị |
|------|-------|---------|--------|-------|-------------|
| 1 | Gen2_MLP_Sklearn | 0.7109 | 0.7251 | Fast | ✅ Tốt, nhanh |
| 2 | Gen2_GradientBoosting | 0.7107 | 0.7260 | Medium | ✅ BEST F1 max |
| 3 | Gen3_CatBoost | 0.7105 | 0.7253 | **Slow (513s)** | ❌ Quá chậm |
| 4 | Gen2_RandomForest | 0.7102 | 0.7252 | **Fast (6s)** | ✅ Nhanh nhất |
| 5 | Gen3_LightGBM | 0.7096 | **0.7260** | Fast | ✅ BEST balance |

**💡 INSIGHT:**
- **Gen3_LightGBM:** BEST choice (F1 cao, nhanh 26s)
- **Gen2_RandomForest:** Best cho production (F1 tương đương, chỉ 6s)
- **Gen3_CatBoost:** Tránh (chậm gấp 86x so với RF, performance tương tự)

---

### **CREDITCARD Dataset**

| Rank | Model | F1 Mean | F1 Max | ROC-AUC | Đặc điểm |
|------|-------|---------|--------|---------|----------|
| 1 | **Gen3_XGBoost** ✅ | 0.6716 | 0.8429 | **0.9747** | Tốt nhất cho tree-based |
| 2 | Gen2_MLP_Sklearn | 0.6538 | 0.8285 | 0.9467 | Tốt, cần scaling |
| 3 | Gen2_RandomForest | 0.6276 | 0.8224 | 0.9623 | Ổn định |
| 4 | Gen2_ExtraTrees | 0.6028 | 0.7721 | 0.9589 | Nhanh |
| 5 | Gen3_LightGBM | 0.5916 | 0.8312 | 0.9731 | Nhanh |
| ... | ... | ... | ... | ... | ... |
| 🏆 | **Gen1_KNN** | 0.5592 | **0.8615** | 0.9225 | **BEST F1 max** với scaling |

**💡 NGHỊCH LÝ:**
```
XGBoost: F1_mean cao (0.6716) nhưng F1_max không phải cao nhất
KNN: F1_mean thấp (0.5592) nhưng F1_max CAO NHẤT (0.8615)
```

**GIẢI THÍCH:**
- **KNN rất nhạy với config:**
  - Config tốt (robust + mutual_info_12 + none): F1 = 0.8615 🏆
  - Config xấu (none + none + smote): F1 = 0.1065 😱
  - → Variance LỚN

- **XGBoost ổn định hơn:**
  - F1 luôn trong khoảng 0.5-0.68
  - Ít phụ thuộc config

**Khuyến nghị:**
- **Production (ổn định):** XGBoost (F1 = 0.6716, ROC-AUC = 0.9747)
- **Best performance:** KNN với config tối ưu (F1 = 0.8615)

---

## 5️⃣ TOP CONFIGURATIONS

### **CARDIO Dataset - Top 3 Configs**

```
🥇 RANK #1: Gen3_LightGBM
   Config: scaler=none + fs=select_k_best_12 + imb=smote
   F1=0.7260, ROC-AUC=0.8018, Accuracy=0.7366, Time=25.90s

🥈 RANK #2: Gen2_GradientBoosting
   Config: scaler=none + fs=select_k_best_12 + imb=smote
   F1=0.7260, ROC-AUC=0.8029, Accuracy=0.7372, Time=16.30s
   → Nhanh hơn LightGBM 37%!

🥉 RANK #3: Gen2_GradientBoosting
   Config: scaler=none + fs=mutual_info_12 + imb=smote
   F1=0.7256, ROC-AUC=0.8025, Accuracy=0.7369, Time=24.86s
```

**Pattern nhận ra:**
- `scaler=none` xuất hiện trong **100% top 10** configs
- `fs=select_k_best_12` hoặc `mutual_info_12` (giữ 12 features)
- `imb=smote` hoặc `none` (khác biệt nhỏ)

---

### **CREDITCARD Dataset - Top 3 Configs**

```
🥇 RANK #1: Gen1_KNN
   Config: scaler=robust + fs=mutual_info_12 + imb=none
   F1=0.8615, ROC-AUC=0.9225, Accuracy=0.9996, Time=111.84s

🥈 RANK #2: Gen1_KNN
   Config: scaler=standard + fs=mutual_info_12 + imb=none
   F1=0.8608, ROC-AUC=0.9212, Accuracy=0.9996, Time=114.23s

🥉 RANK #3: Gen1_KNN
   Config: scaler=standard + fs=select_k_best_12 + imb=none
   F1=0.8595, ROC-AUC=0.9212, Accuracy=0.9996, Time=40.63s
```

**Pattern nhận ra:**
- **KNN chiếm 7/10 top configs** (cực kỳ mạnh cho fraud detection!)
- `scaler=robust/standard` (BẮT BUỘC cho KNN)
- `fs=mutual_info_12` hoặc `select_k_best_12` (giữ 12/30 features)
- `imb=none` (KHÔNG dùng SMOTE!)

**⚠️ Best tree-based:**
```
RANK #8: Gen3_XGBoost
Config: scaler=none + fs=none + imb=none
F1=0.8429, ROC-AUC=0.9747, Accuracy=0.9995, Time=1.46s
→ NHANH GẤP 76X so với KNN! (1.46s vs 111.84s)
```

---

## 6️⃣ INTERACTION ANALYSIS (Tương tác giữa configs)

### **CARDIO: Scaler × Feature Selection**

|  | mutual_info_12 | none | select_k_best_12 | mutual_info_5 | select_k_best_5 |
|---|---|---|---|---|---|
| **none** | **0.7187** ✅ | 0.7178 | 0.7179 | 0.6663 | 0.7118 |
| robust | 0.6987 | 0.6935 | 0.6925 | 0.6516 | 0.7045 |
| standard | 0.7080 | 0.6981 | 0.6980 | 0.6620 | 0.7055 |

**💡 INSIGHT:**
- **Best combo:** `none + mutual_info_12` (F1 = 0.7187)
- **Scaling LÀM HẠI** performance với mọi feature selector
- Giữ 5 features → Performance giảm mạnh

---

### **CREDITCARD: Scaler × Feature Selection**

|  | mutual_info_12 | none | select_k_best_12 | mutual_info_5 | select_k_best_5 |
|---|---|---|---|---|---|
| **none** | 0.6030 | **0.6945** ✅ | 0.6050 | 0.5933 | 0.3885 |
| robust | 0.4052 | 0.3756 | 0.3760 | 0.2990 | 0.2574 |
| standard | 0.4126 | 0.3713 | 0.3806 | 0.2849 | 0.2609 |

**💡 INSIGHT:**
- **Best combo:** `none + none` (F1 = 0.6945) - Không scaling, không feature selection
- **Nhưng:** Top config thực tế là `robust + mutual_info_12` (F1 = 0.8615) với KNN!
- → Average cao ≠ Peak performance cao

---

## 7️⃣ STATISTICAL INSIGHTS

### **Config Impact Ranking**

| Dataset | Rank 1 | Rank 2 | Rank 3 | Rank 4 |
|---------|--------|--------|--------|--------|
| **Cardio** | **Feature_Selector** (0.0221) | Model (0.0115) | Scaler (0.0077) | Imbalance (0.0002) |
| **CreditCard** | **Model** (0.1971) | Feature_Selector (0.1162) | Scaler (0.0956) | Imbalance (0.0616) |

**💡 INSIGHT:**

**CARDIO (Balanced data):**
- **Feature Selection ảnh hưởng NHẤT** (std = 0.0221)
- Imbalance handling hầu như KHÔNG ảnh hưởng (std = 0.0002)
- → Tập trung vào chọn features đúng!

**CREDITCARD (Imbalanced data):**
- **Model choice ảnh hưởng NHẤT** (std = 0.1971)
- Feature Selection và Scaler cũng quan trọng
- → Chọn model phù hợp (KNN cho peak, XGBoost cho stable)

---

## 🎓 LESSONS LEARNED

### 1. **Scaling không phải lúc nào cũng tốt**
```
❌ Sai lầm: "Phải scaling trước khi train model"
✅ Đúng: Tree-based models KHÔNG cần scaling
         Distance-based models (KNN, SVM) CẦN scaling
```

### 2. **SMOTE không phải giải pháp vạn năng**
```
❌ Sai lầm: "Data imbalanced → dùng SMOTE"
✅ Đúng: SMOTE có thể LÀM HẠI performance
         → Test cả none, smote, smote_enn và chọn tốt nhất
```

### 3. **Feature Selection phụ thuộc vào dataset**
```
Cardio (balanced): Giữ 80% features (12/15) → performance tốt
CreditCard (imbalanced): Giữ 100% features → performance tốt nhất
```

### 4. **Average performance ≠ Peak performance**
```
XGBoost: F1_mean cao, ổn định → Production
KNN: F1_max cao nhất nhưng không ổn định → Research
```

### 5. **Speed vs Performance trade-off**
```
RandomForest: 6s, F1=0.7252 → Production
CatBoost: 513s, F1=0.7253 → Không đáng
```

---

## 📋 KHUYẾN NGHỊ CUỐI CÙNG

### **CARDIO Dataset (Balanced, Medical diagnosis)**

#### Production Model:
```python
Model: Gen2_RandomForest
Config: scaler=none, feature_selection=select_k_best_12, imbalance=none
Performance: F1=0.7252, ROC-AUC=0.8016, Time=5.95s
Lý do: Nhanh, ổn định, không cần hyperparameter tuning phức tạp
```

#### Best Performance:
```python
Model: Gen3_LightGBM
Config: scaler=none, feature_selection=select_k_best_12, imbalance=smote
Performance: F1=0.7260, ROC-AUC=0.8018, Time=25.90s
Lý do: Performance cao nhất, tốc độ chấp nhận được
```

---

### **CREDITCARD Dataset (Imbalanced, Fraud detection)**

#### Production Model (Stable):
```python
Model: Gen3_XGBoost
Config: scaler=none, feature_selection=none, imbalance=none
Performance: F1=0.8429, ROC-AUC=0.9747, Time=1.46s
Lý do: Ổn định, NHANH, ROC-AUC cao, không cần preprocessing phức tạp
```

#### Best Performance (Peak):
```python
Model: Gen1_KNN
Config: scaler=robust, feature_selection=mutual_info_12, imbalance=none
Performance: F1=0.8615, ROC-AUC=0.9225, Time=111.84s
Lý do: F1 cao nhất, accuracy 99.96%
Nhược điểm: Chậm gấp 76x, cần tuning cẩn thận
```

#### Business Decision:
```
Nếu ưu tiên Precision (ít False Positive):
  → Dùng none imbalance (Precision = 0.4880)

Nếu ưu tiên Recall (bắt hết Fraud):
  → Dùng smote_enn (Recall = 0.8383, nhưng mất Precision)

Trade-off tốt nhất: none (F1 = 0.5063)
```

---

## 🔥 KEY TAKEAWAYS

1. **Dataset characteristics quyết định config strategy**
   - Balanced data: Feature selection quan trọng nhất
   - Imbalanced data: Model choice quan trọng nhất

2. **Tree-based models ≠ Distance-based models**
   - Trees: Không cần scaling, ổn định
   - KNN/SVM: CẦN scaling, nhạy cảm với config

3. **SMOTE không phải silver bullet**
   - Test cả 3 options: none, smote, smote_enn
   - none có thể tốt hơn SMOTE với tree-based models

4. **Production vs Research có config khác nhau**
   - Production: Ổn định, nhanh (RandomForest, XGBoost)
   - Research: Peak performance (LightGBM, KNN tuned)

5. **Variance analysis giúp ưu tiên effort**
   - Cardio: Tập trung vào feature selection
   - CreditCard: Tập trung vào model selection
