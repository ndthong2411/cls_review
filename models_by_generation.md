# 🤖 MODELS THEO TỪNG GENERATION

## 📋 TỔNG QUAN

Bạn đang sử dụng **4 Generations** với tổng cộng **18 models** (nếu đủ dependencies):

| Generation | Số Models | Loại | Đặc điểm chính |
|------------|-----------|------|----------------|
| **Gen 1** | 3 | Baseline (Classical) | Đơn giản, nhanh, dễ hiểu |
| **Gen 2** | 5 | Intermediate (Ensemble) | Mạnh hơn, chậm hơn, ensemble learning |
| **Gen 3** | 3 | Advanced (Gradient Boosting) | State-of-the-art, GPU support |
| **Gen 4** | 2 | Deep Learning (SOTA) | Neural networks, attention mechanism |

---

## 🎯 GENERATION 1: BASELINE (Classical ML)

> **Mục đích:** Baseline models để so sánh, dễ hiểu, nhanh train

### 1. **Gen1_LogisticRegression**
```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',  # Xử lý imbalance
    solver='lbfgs'
)
```

**📊 Đặc điểm:**
- **Loại:** Linear classifier
- **Needs scaling:** ✅ YES (BẮT BUỘC)
- **Complexity:** Thấp
- **Training speed:** ⚡ Rất nhanh
- **Interpretability:** ⭐⭐⭐⭐⭐ (Dễ hiểu nhất)

**💡 Khi nào dùng:**
- Baseline để so sánh
- Dữ liệu linearly separable
- Cần interpretability cao (xem feature importance)

**⚠️ Hạn chế:**
- Không bắt được non-linear relationships
- Performance thấp hơn ensemble models

---

### 2. **Gen1_DecisionTree**
```python
DecisionTreeClassifier(
    max_depth=10,           # Giới hạn độ sâu để tránh overfit
    min_samples_leaf=20,    # Tối thiểu 20 samples/leaf
    min_samples_split=40,   # Tối thiểu 40 samples để split
    random_state=42,
    class_weight='balanced'
)
```

**📊 Đặc điểm:**
- **Loại:** Tree-based classifier
- **Needs scaling:** ❌ NO
- **Complexity:** Trung bình
- **Training speed:** ⚡⚡ Nhanh
- **Interpretability:** ⭐⭐⭐⭐ (Dễ visualize)

**💡 Khi nào dùng:**
- Cần model đơn giản, dễ hiểu
- Visualize decision rules
- Không cần scaling data

**⚠️ Hạn chế:**
- Dễ overfit (đã set max_depth=10 để giảm)
- Performance thấp hơn ensemble

---

### 3. **Gen1_KNN** (K-Nearest Neighbors)
```python
KNeighborsClassifier(
    n_neighbors=5,          # Xem 5 láng giềng gần nhất
    weights='distance',     # Láng giềng gần có trọng số cao hơn
    metric='minkowski',     # Euclidean distance
    n_jobs=-1              # Dùng tất cả CPU cores
)
```

**📊 Đặc điểm:**
- **Loại:** Distance-based classifier (lazy learning)
- **Needs scaling:** ✅ YES (BẮT BUỘC)
- **Complexity:** Cao (lưu toàn bộ training data)
- **Training speed:** ⚡⚡⚡ Instant (không train)
- **Prediction speed:** 🐌 CHẬM (phải tính distance với tất cả training samples)
- **Interpretability:** ⭐⭐⭐ (Hiểu được logic nhưng khó giải thích)

**💡 Khi nào dùng:**
- Dữ liệu có pattern cục bộ rõ ràng
- **TUYỆT VỜI cho fraud detection** (như kết quả bạn thấy: F1=0.8615 🏆)
- Không cần training time (chỉ store data)

**⚠️ Hạn chế:**
- RẤT CHẬM khi predict (với CreditCard: 111s vs XGBoost: 1.46s)
- Tốn memory (lưu toàn bộ training set)
- **CỰC KỲ NHẠY CẢM** với scaling và feature selection

**🔥 Insight từ kết quả thực tế:**
```
CreditCard dataset: KNN đạt F1 CAO NHẤT (0.8615)
Với config: scaler=robust + fs=mutual_info_12 + imb=none
→ Fraud patterns có locality tốt!
```

---

## 🎯 GENERATION 2: INTERMEDIATE (Ensemble Learning)

> **Mục đích:** Ensemble methods để tăng performance, giảm variance

### 4. **Gen2_RandomForest**
```python
RandomForestClassifier(
    n_estimators=300,       # 300 trees (tăng từ 100)
    max_depth=15,           # Độ sâu mỗi tree
    min_samples_leaf=10,    # Tối thiểu 10 samples/leaf
    random_state=42,
    class_weight='balanced',
    n_jobs=-1               # Parallel training
)
```

**📊 Đặc điểm:**
- **Loại:** Ensemble (Bagging - Bootstrap Aggregating)
- **Needs scaling:** ❌ NO
- **Complexity:** Cao
- **Training speed:** ⚡⚡ Nhanh (parallel)
- **Prediction speed:** ⚡⚡⚡ Rất nhanh
- **Interpretability:** ⭐⭐ (Feature importance available)

**💡 Khi nào dùng:**
- **PRODUCTION FAVORITE** (ổn định, nhanh, không cần tuning nhiều)
- Cardio: F1=0.7252, chỉ 5.95s 🚀
- Không cần preprocessing phức tạp

**⚙️ Cách hoạt động:**
```
1. Tạo 300 decision trees
2. Mỗi tree train trên subset ngẫu nhiên của data (bootstrap)
3. Mỗi split chỉ xét subset ngẫu nhiên của features
4. Voting: Lấy majority vote của 300 trees
```

**🔥 Ưu điểm:**
- Giảm overfitting (so với single tree)
- Robust với outliers
- Tự động handle non-linear relationships

---

### 5. **Gen2_ExtraTrees** (Extremely Randomized Trees)
```python
ExtraTreesClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
```

**📊 Đặc điểm:**
- **Loại:** Ensemble (Extra randomization)
- **Needs scaling:** ❌ NO
- **Training speed:** ⚡⚡⚡ NHANH HƠN RandomForest
- **Prediction speed:** ⚡⚡⚡ Rất nhanh

**💡 Khác với RandomForest:**
```
RandomForest: Tìm best split cho mỗi feature
ExtraTrees:   Random split cho mỗi feature → NHANH HƠN
```

**🔥 Trade-off:**
- Training nhanh hơn RF (~30-40%)
- Variance giảm nhiều hơn
- Nhưng bias có thể tăng → Performance đôi khi thấp hơn RF

---

### 6. **Gen2_GradientBoosting**
```python
GradientBoostingClassifier(
    n_estimators=200,          # 200 boosting stages
    max_depth=5,               # Shallow trees (weak learners)
    learning_rate=0.1,         # Shrinkage
    random_state=42,
    subsample=0.8,             # Stochastic boosting (80% data/iteration)
    validation_fraction=0.1,   # 10% for early stopping
    n_iter_no_change=20        # Stop if no improve in 20 iterations
)
```

**📊 Đặc điểm:**
- **Loại:** Ensemble (Boosting - Sequential learning)
- **Needs scaling:** ❌ NO
- **Training speed:** 🐌🐌 Chậm (sequential, không parallel được)
- **Prediction speed:** ⚡⚡ Nhanh
- **Performance:** ⭐⭐⭐⭐ Cao

**💡 Cách hoạt động:**
```
1. Train tree đầu tiên
2. Tính errors (residuals)
3. Train tree tiếp theo để FIX errors của tree trước
4. Lặp lại 200 lần
5. Final prediction = sum(all trees × learning_rate)
```

**🔥 Ưu điểm:**
- Performance cao hơn RandomForest
- Cardio: F1=0.7260 (rank #2)

**⚠️ Nhược điểm:**
- Training CHẬM (sequential)
- Dễ overfit nếu không có early stopping
- Không scale tốt với big data

---

### 7. **Gen2_SVM_RBF** (Support Vector Machine)
```python
SVC(
    kernel='rbf',              # Radial Basis Function kernel
    C=1.0,                     # Regularization
    gamma='scale',             # Kernel coefficient
    random_state=42,
    class_weight='balanced',
    probability=True           # Enable probability estimates
)
```

**📊 Đặc điểm:**
- **Loại:** Kernel-based classifier
- **Needs scaling:** ✅ YES (BẮT BUỘC)
- **Training speed:** 🐌🐌🐌 RẤT CHẬM với big data
- **Complexity:** Cao (O(n²) to O(n³))
- **Memory:** Tốn nhiều

**💡 Cách hoạt động:**
```
1. Map data lên không gian cao chiều (kernel trick)
2. Tìm hyperplane tách class với margin tối đa
3. Support vectors: Samples gần decision boundary
```

**⚠️ Khi nào dùng:**
- Small to medium datasets (< 10k samples)
- **TRÁNH** với big data (CreditCard: 227k samples → RẤT CHẬM)

**🔥 Kết quả thực tế:**
- Cardio: Performance kém, training chậm
- → Không recommend

---

### 8. **Gen2_MLP_Sklearn** (Multi-Layer Perceptron)
```python
MLPClassifier(
    hidden_layer_sizes=(128, 64),    # 2 hidden layers: 128 → 64 neurons
    activation='relu',               # ReLU activation
    solver='adam',                   # Adam optimizer
    alpha=0.0001,                    # L2 regularization
    learning_rate_init=0.001,
    max_iter=500,                    # Max 500 epochs
    early_stopping=True,             # Stop khi validation loss không giảm
    validation_fraction=0.1,         # 10% for validation
    n_iter_no_change=20,             # Early stopping patience
    random_state=42
)
```

**📊 Đặc điểm:**
- **Loại:** Neural Network (2 hidden layers)
- **Needs scaling:** ✅ YES (BẮT BUỘC)
- **Training speed:** 🐌 Chậm (~37s)
- **Performance:** ⭐⭐⭐⭐ Tốt

**🏗️ Architecture:**
```
Input (15 features) → 128 neurons → 64 neurons → Output (2 classes)
                      [ReLU]         [ReLU]       [Softmax]
```

**🔥 Kết quả thực tế:**
```
Cardio: F1=0.7109 (RANK #1 by average)
→ Mạnh và ổn định!
CreditCard: F1=0.6538 (RANK #2)
```

**💡 Khi nào dùng:**
- Cần model linh hoạt học non-linear patterns
- Có GPU tốt hơn (nhưng sklearn không dùng GPU)
- Better choice: Gen4_PyTorch_MLP (có GPU support)

---

## 🎯 GENERATION 3: ADVANCED (Gradient Boosting SOTA)

> **Mục đích:** State-of-the-art boosting algorithms với GPU acceleration

### 9. **Gen3_XGBoost** 🚀 GPU
```python
xgb.XGBClassifier(
    n_estimators=2000,          # 2000 trees (tăng mạnh từ 500)
    max_depth=10,               # Deep trees (từ 6 → 10)
    learning_rate=0.03,         # Slow learning (0.1 → 0.03)
    min_child_weight=3,         # Prevent overfitting
    subsample=0.9,              # 90% data per tree
    colsample_bytree=0.9,       # 90% features per tree
    gamma=0,                    # Min loss reduction for split
    early_stopping_rounds=100,  # Stop nếu 100 rounds không improve
    tree_method='gpu_hist',     # 🚀 GPU ACCELERATION
    gpu_id=0,
    predictor='gpu_predictor',
    eval_metric='logloss'
)
```

**📊 Đặc điểm:**
- **Loại:** Extreme Gradient Boosting
- **Needs scaling:** ❌ NO
- **Training speed:** ⚡⚡⚡ NHANH (GPU) hoặc 🐌 Chậm (CPU)
- **Performance:** ⭐⭐⭐⭐⭐ XUẤT SẮC
- **GPU Support:** ✅ YES

**🔥 Ưu điểm:**
```
✅ Regularization mạnh (L1, L2, gamma)
✅ Handle missing values tốt
✅ Built-in early stopping
✅ GPU acceleration → Nhanh gấp 10-100x CPU
✅ Production-ready
```

**🏆 Kết quả thực tế:**
```
CreditCard: F1=0.8429, ROC-AUC=0.9747, Time=1.46s
→ BEST production model cho CreditCard!

Cardio: F1=0.7083 (rank #7)
→ Tốt nhưng không phải best
```

**💡 Hyperparameter tuning quan trọng:**
```python
# Tăng n_estimators, giảm learning_rate → Tốt hơn
n_estimators=2000, learning_rate=0.03  # Slow & Steady wins
vs
n_estimators=500, learning_rate=0.1    # Fast but may underfit
```

---

### 10. **Gen3_LightGBM** 🚀 GPU
```python
lgb.LGBMClassifier(
    n_estimators=2000,
    max_depth=10,
    num_leaves=100,             # 2^max_depth rule: 2^10 = 1024, giới hạn 100
    learning_rate=0.03,
    min_child_samples=30,       # Prevent overfitting
    subsample=0.9,
    colsample_bytree=0.9,
    subsample_freq=1,           # Enable bagging
    is_unbalance=True,          # Auto handle imbalance
    device='gpu',               # 🚀 GPU ACCELERATION
    gpu_platform_id=0,
    gpu_device_id=0
)
```

**📊 Đặc điểm:**
- **Loại:** Light Gradient Boosting (histogram-based)
- **Needs scaling:** ❌ NO
- **Training speed:** ⚡⚡⚡⚡ NHANH NHẤT trong Gen3
- **Memory:** Tiết kiệm hơn XGBoost
- **Performance:** ⭐⭐⭐⭐⭐ XUẤT SẮC

**💡 Khác với XGBoost:**
```
LightGBM: Leaf-wise tree growth → Faster, deeper trees
XGBoost:  Level-wise tree growth → Slower, balanced trees

LightGBM: Histogram-based splits → Memory efficient
XGBoost:  Pre-sorted based → More memory
```

**🏆 Kết quả thực tế:**
```
Cardio: F1=0.7260 (RANK #1 🏆)
Time: 25.90s

→ BEST model cho Cardio dataset!
```

**🔥 Khi nào dùng:**
- Large datasets (> 10k rows)
- Cần training nhanh
- Memory hạn chế
- **Production favorite cho balanced data**

---

### 11. **Gen3_CatBoost** 🚀 GPU
```python
cb.CatBoostClassifier(
    iterations=2000,
    depth=10,
    learning_rate=0.03,
    l2_leaf_reg=3,              # L2 regularization
    border_count=254,           # Số lượng splits (precision cao)
    random_strength=1,          # Randomness for robustness
    bagging_temperature=1,      # Bayesian bootstrap
    auto_class_weights='Balanced',
    early_stopping_rounds=100,
    od_type='Iter',
    task_type='GPU',            # 🚀 GPU ACCELERATION
    devices='0'
)
```

**📊 Đặc điểm:**
- **Loại:** Categorical Boosting
- **Needs scaling:** ❌ NO
- **Training speed:** 🐌🐌🐌 CHẬM NHẤT (513s vs XGBoost 1.46s!)
- **Performance:** ⭐⭐⭐⭐⭐ Tốt nhưng không đáng với thời gian training
- **Special:** XỬ LÝ CATEGORICAL FEATURES CỰC TỐT

**💡 Điểm mạnh:**
```
✅ Xử lý categorical features NATIVE (không cần encoding)
✅ Robust với overfitting (ordered boosting)
✅ Default parameters tốt (ít cần tuning)
✅ Symmetric trees → Nhanh khi predict
```

**⚠️ Kết quả thực tế:**
```
Cardio: F1=0.7253, Time=513.54s ← CHẬM GẤP 86X RandomForest!
RandomForest: F1=0.7252, Time=5.95s

→ Performance gần bằng nhau nhưng CatBoost CHẬM KHỦNG KHIẾP
→ KHÔNG RECOMMEND cho production
```

**💡 Khi nào dùng:**
- Dataset có NHIỀU categorical features
- Có thời gian training (offline)
- **KHÔNG DÙNG** cho datasets như Cardio (ít categorical, training quá chậm)

---

## 🎯 GENERATION 4: DEEP LEARNING (State-of-the-Art)

> **Mục đích:** Deep neural networks cho tabular data

### 12. **Gen4_PyTorch_MLP** 🧠 GPU
```python
PyTorchMLPClassifier(
    hidden_dims=[256, 128, 64, 32],      # 4 hidden layers
    dropout_rates=[0.4, 0.3, 0.2, 0.1],  # Progressive dropout
    use_batch_norm=True,                  # Batch normalization
    learning_rate=0.001,
    batch_size=128,
    epochs=200,
    optimizer_name='adamw',               # AdamW optimizer
    weight_decay=1e-4,                    # L2 regularization
    scheduler='plateau',                  # ReduceLROnPlateau
    early_stopping_patience=30,
    class_weight='balanced',
    device=None,                          # Auto-detect GPU
    random_state=42
)
```

**📊 Đặc điểm:**
- **Loại:** Deep Neural Network (4 hidden layers)
- **Needs scaling:** ✅ YES (BẮT BUỘC)
- **Training speed:** 🐌 Chậm
- **GPU Support:** ✅ YES (auto-detect)
- **Performance:** ⭐⭐⭐⭐ Tốt

**🏗️ Architecture:**
```
Input (15) → 256 → 128 → 64 → 32 → Output (2)
   ↓         ↓     ↓      ↓     ↓
[BatchNorm] [BN]  [BN]  [BN]  [BN]
[Dropout40%][30%] [20%] [10%]
[ReLU]      [ReLU][ReLU][ReLU]
```

**🔥 Features:**
```
✅ Batch Normalization → Stable training
✅ Progressive Dropout → Prevent overfitting (40% → 10%)
✅ AdamW Optimizer → Better regularization than Adam
✅ Learning Rate Scheduler → Adaptive learning
✅ Early Stopping → Prevent overfitting
✅ GPU Acceleration
```

**🏆 Kết quả thực tế:**
```
Cardio: F1=0.7044 (rank #9)
→ Tốt nhưng không bằng LightGBM/GradientBoosting

CreditCard: F1=0.3019 (rank #10) 😱
→ PERFORMANCE THẤP! Có thể do:
   - Deep networks cần MORE DATA
   - Imbalanced data (0.17% fraud) khó học
   - Tree-based models TỐT HƠN cho tabular data
```

**💡 Khi nào dùng:**
- Dataset LỚN (> 100k samples)
- Có GPU mạnh
- Cần học complex non-linear patterns
- **NHƯNG:** Tree-based vẫn thường tốt hơn cho tabular data!

---

### 13. **Gen4_TabNet** 🔍 Attention-based
```python
TabNetClassifier(
    n_d=64,                    # Decision prediction layer width
    n_a=64,                    # Attention embedding width
    n_steps=5,                 # Sequential attention steps
    gamma=1.5,                 # Feature reusage coefficient
    n_independent=2,
    n_shared=2,
    lambda_sparse=1e-4,        # Sparsity regularization
    momentum=0.3,
    clip_value=2.0,
    optimizer_params={'lr': 2e-2},
    mask_type='sparsemax',     # Attention mechanism
    seed=42,
    device_name='auto',        # Auto-detect GPU
    max_epochs=200,
    patience=30,
    batch_size=256,
    virtual_batch_size=128
)
```

**📊 Đặc điểm:**
- **Loại:** Attention-based Deep Learning
- **Needs scaling:** ❌ NO (xử lý raw features như tree-based)
- **Training speed:** 🐌 Chậm
- **GPU Support:** ✅ YES
- **Interpretability:** ⭐⭐⭐⭐ (Attention masks → feature importance)
- **Performance:** ⭐⭐⭐ Trung bình

**🔥 Điểm đặc biệt:**
```
✅ INTERPRETABLE như Decision Trees
✅ POWERFUL như Neural Networks
✅ Attention Mechanism → Tự động chọn features quan trọng
✅ Không cần scaling (handle raw features)
✅ Sparse predictions → Regularization tốt
```

**💡 Cách hoạt động:**
```
1. Sequential attention (5 steps)
2. Mỗi step chọn features quan trọng (attention mask)
3. Feature reusage với coefficient gamma=1.5
4. Sparsemax activation → Chỉ activate vài features quan trọng nhất
```

**🏆 Kết quả thực tế:**
```
Cardio: F1=0.7085 (rank #6)
CreditCard: F1=0.4930 (rank #6)

→ Trung bình, không outstanding
→ Có lẽ do dataset không đủ lớn để TabNet phát huy
```

**💡 Khi nào dùng:**
- Cần INTERPRETABILITY (visualize attention masks)
- Dataset lớn (> 100k samples)
- Quan tâm đến feature selection tự động
- **Research purposes** (TabNet mới, chưa proven trong production)

---

## 📊 SO SÁNH TỔNG QUAN

### **Performance Ranking (F1-Score)**

#### **CARDIO Dataset:**
| Rank | Model | Generation | F1 Mean | Needs Scaling | Speed |
|------|-------|------------|---------|---------------|-------|
| 🥇 1 | Gen2_MLP_Sklearn | 2 | 0.7109 | ✅ | Medium |
| 🥈 2 | Gen2_GradientBoosting | 2 | 0.7107 | ❌ | Slow |
| 🥉 3 | Gen3_CatBoost | 3 | 0.7105 | ❌ | Very Slow |
| 4 | Gen2_RandomForest | 2 | 0.7102 | ❌ | **Fast** ⚡ |
| 5 | Gen3_LightGBM | 3 | 0.7096 | ❌ | Fast |

**💡 Recommendation:**
- **Production:** RandomForest (F1=0.7102, 6s)
- **Best Performance:** LightGBM (F1=0.7260 max)

---

#### **CREDITCARD Dataset:**
| Rank | Model | Generation | F1 Mean | Needs Scaling | Speed |
|------|-------|------------|---------|---------------|-------|
| 🥇 1 | Gen3_XGBoost | 3 | 0.6716 | ❌ | **Very Fast** ⚡ |
| 🥈 2 | Gen2_MLP_Sklearn | 2 | 0.6538 | ✅ | Medium |
| 🥉 3 | Gen2_RandomForest | 2 | 0.6276 | ❌ | Fast |
| ... | Gen1_KNN | 1 | 0.5592 | ✅ | Very Slow |

**⚠️ Nhưng F1 MAX:**
- **Gen1_KNN:** 0.8615 🏆 (với config tối ưu)
- **Gen3_XGBoost:** 0.8429

**💡 Recommendation:**
- **Production (Stable):** XGBoost (F1=0.6716, stable)
- **Best Peak:** KNN (F1=0.8615, unstable)

---

### **Scaling Requirements**

| Needs Scaling | Models |
|---------------|--------|
| ✅ **YES** | LogisticRegression, KNN, SVM_RBF, MLP_Sklearn, PyTorch_MLP |
| ❌ **NO** | DecisionTree, RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost, TabNet |

**💡 Rule of thumb:**
- Distance-based & Linear models: CẦN scaling
- Tree-based models: KHÔNG CẦN scaling

---

### **GPU Support**

| GPU Support | Models |
|-------------|--------|
| 🚀 **YES** | Gen3_XGBoost, Gen3_LightGBM, Gen3_CatBoost, Gen4_PyTorch_MLP, Gen4_TabNet |
| ❌ **NO** | All Gen1, Gen2 models |

**💡 GPU Speedup:**
- XGBoost: 10-100x faster
- LightGBM: 5-20x faster
- CatBoost: 3-10x faster (nhưng vẫn chậm nhất)

---

## 🎯 KHUYẾN NGHỊ CUỐI CÙNG

### **Cho người mới:**
```python
Start with: Gen2_RandomForest
Why: Không cần scaling, fast, ổn định, ít hyperparameters
```

### **Cho Production:**
```python
Cardio: Gen2_RandomForest hoặc Gen3_LightGBM
CreditCard: Gen3_XGBoost

Why: Nhanh, ổn định, không cần preprocessing phức tạp
```

### **Cho Research (Best Performance):**
```python
Cardio: Gen3_LightGBM (F1=0.7260)
CreditCard: Gen1_KNN với tuning (F1=0.8615)

Why: Peak performance cao nhất
```

### **Khi nào DÙNG Deep Learning (Gen4)?**
```python
❌ TRÁNH nếu:
   - Dataset nhỏ (< 100k samples)
   - Tabular data đơn giản
   - Cần production stable model

✅ DÙNG nếu:
   - Dataset CỰC LỚN (> 1M samples)
   - Có GPU mạnh
   - Research/Experiment
   - Cần interpretability (TabNet)
```

---

## 🔥 KEY TAKEAWAYS

1. **Tree-based models chiếm ưu thế cho tabular data**
   - Top 5 models đều là tree-based hoặc ensemble

2. **Deep Learning KHÔNG PHẢI lúc nào cũng tốt**
   - Gen4 không outperform Gen2/Gen3 trên tabular data

3. **GPU chỉ có ích với dataset lớn**
   - Cardio (70k): GPU không tăng tốc đáng kể
   - CreditCard (280k): GPU giúp XGBoost nhanh hơn

4. **Simplicity wins**
   - RandomForest đơn giản nhưng competitive với models phức tạp hơn

5. **Config quan trọng hơn model choice**
   - KNN: F1 từ 0.1065 → 0.8615 chỉ bằng cách đổi config!
