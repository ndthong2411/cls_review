# Kế hoạch dự án: Pipeline huấn luyện và so sánh phương pháp dự đoán bệnh tim mạch (Cardiovascular Disease)

Dữ liệu: Kaggle "Cardiovascular Disease Dataset" (Sulianova)
Định dạng: CSV; Target: `cardio` (0/1)
Mục tiêu: Xây dựng pipeline tái lập, chạy toàn bộ biến thể tiền xử lý + xử lý mất cân bằng + lựa chọn đặc trưng + nhiều mô hình (ML & DL), kèm đánh giá thống kê, để tạo bảng so sánh ảnh hưởng của từng “impact” lên kết quả cuối cùng.

---

## 1) Kiến trúc dự án và công cụ

- Ngôn ngữ: Python 3.10+
- Thư viện chính:
  - Data/Preprocess: pandas, numpy, scikit-learn, imbalanced-learn, category_encoders
  - Imputation nâng cao: fancyimpute (MICE), sklearn.experimental IterativeImputer, KNNImputer
  - Feature selection: sklearn.feature_selection, sklearn.inspection, xgboost, shap
  - Imbalance: imbalanced-learn (SMOTE, ADASYN, SMOTEENN, Tomek, ENN, BalancedRandomForest, EasyEnsemble)
  - Models: scikit-learn (LR/SVM/Tree/RF), xgboost, lightgbm, catboost
  - Deep Learning: PyTorch hoặc TensorFlow/Keras (chọn 1; mặc định Keras)
  - Tối ưu tham số: scikit-optimize hoặc Optuna (khuyến nghị Optuna)
  - Tracking & cấu hình: Hydra + MLflow (hoặc Weights & Biases); mặc định MLflow + Hydra
  - Trực quan hóa: matplotlib, seaborn, plotly (tuỳ chọn)
- Cấu trúc thư mục (đề xuất):
  ```
  cls_review/
  ├─ data/
  │  ├─ raw/               # CSV gốc (không sửa)
  │  ├─ interim/           # Sau làm sạch cơ bản
  │  └─ processed/         # Sau pipeline (train/val/test, folds)
  ├─ notebooks/            # EDA, prototype
  ├─ src/
  │  ├─ configs/           # Hydra configs (YAML)
  │  ├─ data/              # tải/đọc/validate dữ liệu
  │  ├─ preprocessing/     # missing/outliers/scale/encode
  │  ├─ features/          # feature eng & selection & PCA
  │  ├─ imbalance/         # sampling & class-weights
  │  ├─ models/            # định nghĩa model wrappers
  │  ├─ training/          # vòng lặp train, CV, tuning
  │  ├─ evaluation/        # metrics, thống kê, plots
  │  ├─ utils/             # seed, logging, helpers
  │  └─ experiment/        # orchestrator chạy toàn bộ
  ├─ experiments/          # kết quả, figures, tables
  ├─ mlruns/               # MLflow tracking (local)
  ├─ requirements.txt
  ├─ README.md
  └─ claude.md             # file này
  ```

---

## 2) Chuẩn hóa dữ liệu đầu vào

- CSV parsing best practices:
  - strip whitespace ở header/values
  - options: `skipinitialspace=True`, `na_values=["", "NA", "N/A", "null"]`
  - kiểm tra delimiter (`,` hoặc `;`), encoding
- Kiểm tra schema: cột dự kiến: `age`, `height`, `weight`, `ap_hi`, `ap_lo`, `cholesterol`, `gluc`, `smoke`, `alco`, `active`, `gender`, `cardio`.
- Tạo `bmi = weight / (height/100)^2`, kiểm tra các đơn vị `age` (thường là ngày → đổi sang năm: age_years = age/365).

---

## 3) Tiền xử lý (Preprocessing) – Biến thể có thể bật/tắt

Triển khai dưới dạng Hydra configs để chạy matrix experiments.

3.1 Missing Values
- Đo tỉ lệ missing theo feature.
- Chiến lược theo mức độ missing (configurable per feature group):
  - <5%: listwise deletion (xóa hàng) hoặc mean/median imputation
  - 5–20%: IterativeImputer (MICE), KNNImputer, model-based
  - >20%: drop feature hoặc thêm missing-indicator
- Lưu ý: fit imputer trên train fold, transform trên val/test để tránh leakage.

3.2 Outliers
- Các phương pháp: IQR, Z-score, IsolationForest (tuỳ chọn)
- Chiến lược: cap (winsorize), transform (RobustScaler/yeo-johnson), hoặc flag outlier features.
- Cảnh báo: medical data → không remove mù quáng; cung cấp option chỉ “clip” IQR thay vì drop.

3.3 Scaling/Normalization
- Lựa chọn: StandardScaler, MinMaxScaler, RobustScaler (per numeric group)
- Thực hiện sau imputation, fit theo train fold.

3.4 Encoding Categorical
- Nominal: One-Hot
- Ordinal: LabelEncoder/OrdinalEncoder (vd cholesterol, gluc có thang 1-3)
- High cardinality (nếu có): Target encoding (cross-fold scheme để tránh leakage)

3.5 Feature Transformations
- Log/BoxCox/YeoJohnson cho skewed
- Polynomial features (tuỳ chọn, với regularization)
- Binarize thresholds (y học: ví dụ huyết áp ≥140/90) để tạo feature domain-knowledge.

---

## 4) Feature Engineering & Selection

4.1 Filter methods
- Correlation matrix; remove highly correlated (>|0.9|) – configurable
- Statistical tests: ANOVA F-test cho continuous, Chi-square cho categorical, Mutual Information cho non-linear

4.2 Wrapper methods
- RFE với base estimators: Logistic, KNN, DecisionTree, SVM, RF (chọn n_features_to_select via CV)

4.3 Embedded methods
- L1 (Lasso) Logistic/LinearSVC, tree-based importance (RF/XGB/LGBM), SHAP for post-hoc explainability

4.4 Dimensionality reduction
- PCA: fit trên train; giữ 85–95% variance (config `pca.explained_variance=0.9`)
- LDA (supervised) – tùy chọn; t-SNE/UMAP cho visualization (không dùng train)

---

## 5) Xử lý mất cân bằng lớp (Imbalance)

- Data-level:
  - RandomOverSampler, SMOTE, ADASYN
  - Undersampling: RandomUnderSampler, TomekLinks, ENN
  - Hybrid: SMOTEENN, SMOTETomek (RECOMMENDED)
  - Áp dụng CHỈ trên train fold
- Algorithm-level:
  - class_weight cho LR/SVM/Tree/RF/XGB(LGB: is_unbalance/scale_pos_weight)
  - BalancedRandomForest, EasyEnsemble, BalancedBagging

- Thiết kế so sánh:
  - Baseline (no sampling) vs class_weight vs SMOTE vs SMOTE+ENN
  - Ghi nhận ảnh hưởng đến Recall, PR-AUC, ROC-AUC, F1

---

## 6) Tập mô hình (Model Zoo) - Tiến hóa từ Baseline đến SOTA

### **THIẾT KẾ TIẾN HOÁ (Progressive Model Evolution)**

Pipeline này được thiết kế để so sánh performance qua **4 thế hệ mô hình**, từ đơn giản đến tiên tiến, giúp đánh giá **trade-off giữa độ phức tạp và hiệu suất**:

```
Generation 1 (BASELINE) → Generation 2 (INTERMEDIATE) → Generation 3 (ADVANCED) → Generation 4 (SOTA)
     ↓                           ↓                              ↓                        ↓
  Nhanh, giải thích được    Cân bằng tốt               Cần compute lớn          Cutting-edge
  Thời gian: giây/phút      Thời gian: phút             Thời gian: giờ          Thời gian: giờ-ngày
```

---

### **6.1 Generation 1: BASELINE Models (Classical ML)**

**Mục đích**: Thiết lập baseline nhanh, dễ giải thích, ít tài nguyên

| Model | Sklearn API | Ưu điểm | Nhược điểm | Hyperparams chính |
|-------|-------------|---------|------------|-------------------|
| **Logistic Regression** | `LogisticRegression` | Nhanh, interpretable, probabilistic | Chỉ tuyến tính | `C` (0.001-10), `penalty` (l1/l2/elasticnet), `class_weight` |
| **Decision Tree** | `DecisionTreeClassifier` | Giải thích tốt, no scaling | Overfitting dễ | `max_depth` (3-15), `min_samples_leaf` (10-50) |
| **K-Nearest Neighbors** | `KNeighborsClassifier` | Phi tham số, đơn giản | Nhạy scaling, chậm predict | `n_neighbors` (3-15), `weights` (uniform/distance) |

**Expected Performance**: Accuracy 70-82%, PR-AUC 0.72-0.85

---

### **6.2 Generation 2: INTERMEDIATE Models (Ensemble ML)**

**Mục đích**: Nâng cao hiệu suất với ensemble, vẫn giữ tốc độ và khả năng giải thích

| Model | Library | Ưu điểm | Nhược điểm | Hyperparams chính |
|-------|---------|---------|------------|-------------------|
| **Random Forest** | `sklearn.ensemble.RandomForestClassifier` | Robust, feature importance, ít overfit | Chậm với data lớn | `n_estimators` (100-500), `max_depth` (10-30), `max_features` (sqrt/log2) |
| **Extra Trees** | `sklearn.ensemble.ExtraTreesClassifier` | Nhanh hơn RF, variance thấp | Feature importance ít ổn định | `n_estimators` (100-500), `max_features` |
| **Gradient Boosting** | `sklearn.ensemble.GradientBoostingClassifier` | Sequential learning mạnh | Dễ overfit, chậm | `learning_rate` (0.01-0.1), `n_estimators` (100-500), `max_depth` (3-7) |
| **SVM (RBF)** | `sklearn.svm.SVC` | Tốt với biên phi tuyến | Chậm, cần scaling | `C` (0.1-100), `gamma` (scale/auto), `class_weight` |

**Expected Performance**: Accuracy 83-90%, PR-AUC 0.86-0.92

---

### **6.3 Generation 3: ADVANCED Models (Gradient Boosting SOTA + Deep Learning)**

**Mục đích**: Đạt performance cao nhất cho tabular data với modern boosting và deep learning

#### **A. Modern Gradient Boosting (RECOMMENDED cho tabular)**

| Model | Library | Ưu điểm | Nhược điểm | Hyperparams chính |
|-------|---------|---------|------------|-------------------|
| **XGBoost** | `xgboost.XGBClassifier` | SOTA cho tabular, regularization tốt | Cần tuning kỹ | `learning_rate` (0.01-0.1), `max_depth` (3-10), `n_estimators` (100-1000), `subsample` (0.6-1.0), `colsample_bytree` (0.6-1.0), `scale_pos_weight` |
| **LightGBM** | `lightgbm.LGBMClassifier` | Nhanh nhất, hiệu quả bộ nhớ | Overfit với small data | `num_leaves` (20-150), `learning_rate`, `feature_fraction` (0.5-1.0), `bagging_fraction`, `is_unbalance=True` |
| **CatBoost** | `catboost.CatBoostClassifier` | Categorical tự động, ít overfit | Chậm hơn LGBM | `depth` (4-10), `learning_rate`, `iterations`, `class_weights`, `auto_class_weights` |

**Expected Performance**: Accuracy 88-95%, PR-AUC 0.90-0.96

#### **B. Deep Learning cho Tabular**

| Model | Framework | Kiến trúc | Khi nào dùng |
|-------|-----------|-----------|--------------|
| **MLP (PyTorch)** | PyTorch | Input→Dense(128)→BN→Dropout(0.3)→Dense(64)→BN→Dropout(0.2)→Dense(32)→Dropout(0.1)→Dense(1, sigmoid) | Dataset lớn (>10k), muốn học non-linear phức tạp |
| **TabNet** | `pytorch-tabnet` | Attention-based tabular DL với feature selection | Cần interpretability + DL performance |

**Hyperparams (MLP PyTorch)**:
- Optimizer: Adam, SGD+momentum
- Learning rate: 1e-4 to 1e-2 (với scheduler: ReduceLROnPlateau)
- Batch size: 32-128
- Weight decay: 1e-5 to 1e-3
- Class weight cho imbalance
- Early stopping patience: 20-50 epochs

**Expected Performance**: Accuracy 85-93%, PR-AUC 0.88-0.94 (có thể thấp hơn XGBoost với tabular data)

---

### **6.4 Generation 4: STATE-OF-THE-ART (Experimental/Future)**

**Lưu ý**: Chỉ áp dụng nếu mở rộng sang **multi-modal data** (ECG signal + tabular clinical) hoặc **distributed settings**.

#### **A. Cho Time Series / ECG (nếu có tín hiệu)**

| Model | Framework | Kiến trúc | Use Case |
|-------|-----------|-----------|----------|
| **1D CNN** | PyTorch | Conv1D(64,k=3)→BN→ReLU→MaxPool→Conv1D(128)→...→Dense | ECG raw signal |
| **LSTM/BiLSTM** | PyTorch | LSTM(128)→LSTM(64)→Dense | Sequential clinical records |
| **CNN-LSTM Hybrid** | PyTorch | TimeDistributed(Conv1D)→LSTM→Dense | ECG với temporal context |
| **Transformer** | PyTorch | Multi-head self-attention | Long sequences, pattern phức tạp |

**Expected Performance (ECG)**: Accuracy 95-99%, AUC 0.96-0.99

#### **B. Federated Learning (Multi-site)**

| Approach | Library | Mục đích |
|----------|---------|----------|
| **Federated Averaging** | PySyft, Flower | Train trên nhiều bệnh viện không share data |
| **Differential Privacy** | Opacus | Privacy-preserving ML |

**Expected Performance**: Comparable với centralized (92-96% accuracy) nhưng bảo mật dữ liệu

#### **C. Quantum ML (Experimental)**

| Model | Library | Status |
|-------|---------|--------|
| **QSVC** | Qiskit | Proof-of-concept; accuracy ~82% |
| **VQC** | PennyLane | Research stage |

---

### **6.5 Bảng so sánh tổng quan (Summary Table)**

| Generation | Models | Training Time | Interpretability | Expected PR-AUC | Best For |
|------------|--------|---------------|------------------|-----------------|----------|
| **Gen 1 (Baseline)** | LR, DT, KNN | Giây-Phút | ⭐⭐⭐⭐⭐ | 0.72-0.85 | Quick baseline, clinical explanation |
| **Gen 2 (Intermediate)** | RF, ExtraTrees, GB, SVM | Phút-Giờ | ⭐⭐⭐⭐ | 0.86-0.92 | Balanced performance/speed |
| **Gen 3 (Advanced)** | XGB, LGBM, CatBoost, MLP | Giờ | ⭐⭐⭐ | 0.90-0.96 | Maximum tabular performance |
| **Gen 4 (SOTA)** | CNN-LSTM, Transformers, Federated | Giờ-Ngày | ⭐⭐ | 0.95-0.99 | Multi-modal, distributed, cutting-edge |

---

### **6.6 Chiến lược chạy thí nghiệm theo Generation**

**Phase 1: Baseline Sweep (1-2 giờ)**
```yaml
models: [lr, dt, knn]
preprocessing: [standard, robust]  # 2 variants
imbalance: [class_weight, smote]   # 2 variants
Total runs: 3 models × 2 preprocessing × 2 imbalance = 12 runs
```

**Phase 2: Intermediate Sweep (3-6 giờ)**
```yaml
models: [rf, extratrees, gb, svm]
preprocessing: [standard, robust]
imbalance: [class_weight, smote, smoteenn]
feature_selection: [none, rfe]
Total runs: 4 × 2 × 3 × 2 = 48 runs
```

**Phase 3: Advanced Tuning (6-24 giờ)**
```yaml
models: [xgb, lgbm, catboost, mlp_torch]
optuna_trials: 50 per model  # Bayesian optimization
imbalance: [smoteenn]  # Best từ Phase 2
preprocessing: [robust]  # Best từ Phase 2
Total: 4 models × 50 trials = 200 runs
```

**Phase 4: SOTA Experimental (tuỳ chọn)**
- Chỉ khi có ECG data hoặc yêu cầu federated learning
- Benchmark với top-3 từ Phase 3

---

### **6.7 Output: Progressive Performance Report**

**Bảng kết quả mẫu** (tự động generate từ MLflow):

| Generation | Model | Preprocessing | Imbalance | Recall | Precision | F1 | PR-AUC | ROC-AUC | Train Time |
|------------|-------|---------------|-----------|--------|-----------|----|---------|---------| |
| Gen 1 | LogisticReg | Standard | class_weight | 0.78 | 0.76 | 0.77 | 0.82 | 0.84 | 2s |
| Gen 1 | DecisionTree | None | SMOTE | 0.81 | 0.73 | 0.77 | 0.79 | 0.82 | 5s |
| Gen 2 | RandomForest | Robust | SMOTE-ENN | 0.86 | 0.84 | 0.85 | 0.89 | 0.91 | 3m |
| Gen 2 | SVM-RBF | Standard | class_weight | 0.84 | 0.82 | 0.83 | 0.88 | 0.90 | 8m |
| **Gen 3** | **XGBoost** | **Robust** | **SMOTE-ENN** | **0.91** | **0.89** | **0.90** | **0.94** | **0.96** | **45m** |
| Gen 3 | LightGBM | Robust | SMOTE-ENN | 0.90 | 0.88 | 0.89 | 0.93 | 0.95 | 22m |
| Gen 3 | CatBoost | Robust | SMOTE-ENN | 0.90 | 0.89 | 0.90 | 0.93 | 0.95 | 1h 5m |
| Gen 3 | MLP-PyTorch | Standard | class_weight | 0.87 | 0.85 | 0.86 | 0.90 | 0.92 | 38m |

**Visualization outputs**:
1. **Generation comparison boxplot**: PR-AUC distribution per generation
2. **Training time vs Performance scatter**: Efficiency frontier
3. **McNemar test matrix**: Statistical significance between top models
4. **Feature importance evolution**: Xem features quan trọng thay đổi qua generations

---

---

## 7) Chiến lược đánh giá & thí nghiệm

7.1 Splitting & CV
- Hold-out: 70/15/15 (train/val/test) hoặc 60/20/20
- Cross-Validation: StratifiedKFold k=5 (default) – mọi preprocessing phải nằm trong pipeline và fit theo fold
- Early stopping dùng val fold, không đụng test cho đến cuối.

7.2 Metrics (imbalanced-first)
- Chính: Recall (Sensitivity), PR-AUC, F1
- Phụ: ROC-AUC, Precision, Specificity, Balanced Accuracy
- Báo cáo confusion matrix, ROC/PR curves; threshold tuning theo mục tiêu Recall≥90% nếu là screening.

7.3 Thống kê so sánh
- Report mean±std qua folds
- McNemar test cho khác biệt tỷ lệ lỗi giữa 2 models (hold-out predictions)
- DeLong test cho AUC (so sánh ROC-AUC)

7.4 Kiểm soát rò rỉ dữ liệu
- Tất cả bước: impute, scale, encode, select, PCA, SMOTE… đều nằm trong sklearn Pipeline/ColumnTransformer; fit trên train của từng fold

7.5 Reproducibility
- `random_state=42`, seed toàn cục (numpy, torch, TF); lưu versions libs; log cấu hình bằng Hydra/MLflow

---

## 8) Orchestration & Tracking - Progressive Experiment Design

### **8.1 Phased Experiment Strategy**

Thay vì chạy toàn bộ ma trận một lúc, chia thành **3 phases tuần tự** để tối ưu tài nguyên và học từ kết quả trước:

```
Phase 1 (BASELINE) → Identify best preprocessing
         ↓
Phase 2 (INTERMEDIATE) → Identify best imbalance strategy + feature selection
         ↓
Phase 3 (ADVANCED) → Hyperparameter tuning intensive với best configs
```

---

### **8.2 Phase 1: Baseline Exploration (Budget: 1-2 giờ)**

**Mục tiêu**: Xác định preprocessing tốt nhất với models đơn giản

**Hydra config matrix**:
```yaml
phase: baseline
models: [lr, dt, knn]
preprocessing:
  missing: [median, knn]  # 2 variants
  outliers: [none, iqr_clip]  # 2 variants
  scale: [standard, robust]  # 2 variants
  encode: [onehot]  # fixed
imbalance:
  method: [class_weight, smote]  # 2 variants
features:
  selector: [none]  # no selection yet
optuna:
  n_trials: 5  # quick search per model
```

**Total runs**: 3 models × 2 missing × 2 outliers × 2 scale × 2 imbalance × 5 trials = **240 runs**

**Output**: Best preprocessing combo (e.g., `median + iqr_clip + robust`)

**Run command**:
```powershell
python -m src.experiment.run_phase --phase=baseline --multirun
```

---

### **8.3 Phase 2: Intermediate Optimization (Budget: 3-6 giờ)**

**Mục tiêu**: Test ensemble methods với best preprocessing từ Phase 1, explore imbalance & feature selection

**Hydra config** (sử dụng best từ Phase 1):
```yaml
phase: intermediate
models: [rf, extratrees, gb, svm]
preprocessing:
  missing: median  # FIXED từ Phase 1
  outliers: iqr_clip  # FIXED
  scale: robust  # FIXED
  encode: onehot
imbalance:
  method: [class_weight, smote, adasyn, smoteenn, smotetomek]  # 5 variants
features:
  selector: [none, rfe, l1, tree_importance]  # 4 variants
  correlation_threshold: 0.9
optuna:
  n_trials: 10  # moderate search
```

**Total runs**: 4 models × 5 imbalance × 4 selectors × 10 trials = **800 runs**

**Output**: 
- Best imbalance method (e.g., `smoteenn`)
- Best feature selector (e.g., `tree_importance`)

**Run command**:
```powershell
python -m src.experiment.run_phase --phase=intermediate --config_path=experiments/phase1_best.yaml
```

---

### **8.4 Phase 3: Advanced Tuning (Budget: 6-24 giờ)**

**Mục tiêu**: Intensive hyperparameter tuning cho SOTA models với best pipeline từ Phase 2

**Hydra config** (frozen pipeline):
```yaml
phase: advanced
models: [xgb, lgbm, catboost, mlp_torch]
preprocessing:  # ALL FIXED từ Phase 2
  missing: median
  outliers: iqr_clip
  scale: robust
  encode: onehot
imbalance:
  method: smoteenn  # FIXED từ Phase 2
features:
  selector: tree_importance  # FIXED
  correlation_threshold: 0.9
optuna:
  n_trials: 100  # intensive search
  timeout: 21600  # 6 hours per model
  pruner: HyperbandPruner  # early stop bad trials
```

**Total runs**: 4 models × 100 trials = **400 runs** (với pruning, ~250 completed)

**Optuna objectives per model**:
- Primary: `pr_auc` (maximize)
- Secondary constraints: `recall >= 0.85`, `train_time < 3600s`

**Run command** (per model, parallel nếu có GPU):
```powershell
# Sequential
python -m src.experiment.run_phase --phase=advanced model.name=xgb
python -m src.experiment.run_phase --phase=advanced model.name=lgbm
python -m src.experiment.run_phase --phase=advanced model.name=catboost
python -m src.experiment.run_phase --phase=advanced model.name=mlp_torch

# Hoặc parallel (nếu multi-GPU)
python -m src.experiment.run_phase --phase=advanced --multirun model.name=xgb,lgbm,catboost,mlp_torch
```

---

### **8.5 Optuna Study Configuration**

**Search spaces per model generation**:

**XGBoost**:
```python
{
    'learning_rate': ('loguniform', 0.01, 0.3),
    'max_depth': ('int', 3, 10),
    'n_estimators': ('int', 100, 1000),
    'subsample': ('uniform', 0.5, 1.0),
    'colsample_bytree': ('uniform', 0.5, 1.0),
    'min_child_weight': ('int', 1, 10),
    'gamma': ('loguniform', 1e-8, 1.0),
    'reg_alpha': ('loguniform', 1e-8, 1.0),
    'reg_lambda': ('loguniform', 1e-8, 10.0),
    'scale_pos_weight': ('categorical', [None, 'auto'])  # auto = n_neg/n_pos
}
```

**LightGBM**:
```python
{
    'num_leaves': ('int', 20, 150),
    'learning_rate': ('loguniform', 0.01, 0.3),
    'n_estimators': ('int', 100, 1000),
    'max_depth': ('int', -1, 15),  # -1 = no limit
    'feature_fraction': ('uniform', 0.4, 1.0),
    'bagging_fraction': ('uniform', 0.4, 1.0),
    'bagging_freq': ('int', 1, 7),
    'min_child_samples': ('int', 5, 100),
    'reg_alpha': ('loguniform', 1e-8, 10.0),
    'reg_lambda': ('loguniform', 1e-8, 10.0),
    'is_unbalance': True  # FIXED
}
```

**MLP PyTorch**:
```python
{
    'hidden1': ('categorical', [64, 128, 256]),
    'hidden2': ('categorical', [32, 64, 128]),
    'hidden3': ('categorical', [16, 32, 64]),
    'dropout1': ('uniform', 0.2, 0.5),
    'dropout2': ('uniform', 0.1, 0.4),
    'dropout3': ('uniform', 0.0, 0.3),
    'lr': ('loguniform', 1e-4, 1e-2),
    'weight_decay': ('loguniform', 1e-6, 1e-3),
    'batch_size': ('categorical', [32, 64, 128, 256]),
    'optimizer': ('categorical', ['adam', 'adamw', 'sgd']),
    'scheduler': ('categorical', ['plateau', 'cosine', 'step'])
}
```

**Optuna pruning strategy**:
```python
study = optuna.create_study(
    direction='maximize',
    study_name=f'{model_name}_phase{phase}',
    storage=f'sqlite:///optuna_{model_name}.db',
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1,  # min CV folds
        max_resource=5,  # max CV folds
        reduction_factor=3
    ),
    sampler=optuna.samplers.TPESampler(seed=42)
)
```

---

### **8.6 MLflow Tracking Structure**

**Experiment hierarchy**:
```
mlruns/
├─ cardio_phase1_baseline/
│  ├─ run_lr_001/
│  ├─ run_lr_002/
│  └─ ...
├─ cardio_phase2_intermediate/
│  ├─ run_rf_001/
│  └─ ...
└─ cardio_phase3_advanced/
   ├─ run_xgb_001/
   ├─ run_xgb_002/  # Optuna trial 2
   └─ ...
```

**Logged parameters per run**:
```python
mlflow.log_params({
    # Pipeline config
    'phase': cfg.phase,
    'model': cfg.model.name,
    'missing_strategy': cfg.preprocessing.missing,
    'scaler': cfg.preprocessing.scale,
    'outlier_method': cfg.preprocessing.outliers,
    'imbalance_method': cfg.imbalance.method,
    'feature_selector': cfg.features.selector,
    'n_features_original': X.shape[1],
    'n_features_selected': X_selected.shape[1],
    
    # Model hyperparams (dynamic per model)
    **model.get_params(),
    
    # CV config
    'cv_folds': cfg.cv.n_splits,
    'test_size': cfg.cv.test_size,
    'random_seed': cfg.seed
})
```

**Logged metrics** (per fold + aggregated):
```python
# Per fold
for fold in range(n_folds):
    mlflow.log_metrics({
        f'fold{fold}_recall': ...,
        f'fold{fold}_precision': ...,
        f'fold{fold}_pr_auc': ...,
        # ... all metrics
    }, step=fold)

# Aggregated
mlflow.log_metrics({
    'mean_recall': np.mean(recalls),
    'std_recall': np.std(recalls),
    'mean_pr_auc': np.mean(pr_aucs),
    'std_pr_auc': np.std(pr_aucs),
    'train_time_total': total_time,
    'train_time_per_fold': total_time / n_folds
})
```

**Logged artifacts**:
```python
mlflow.log_artifacts('figures/', artifact_path='plots')  # ROC/PR curves
mlflow.log_artifact('models/best_model.pkl')
mlflow.log_artifact('pipelines/full_pipeline.pkl')
mlflow.sklearn.log_model(pipeline, 'model', signature=signature)
```

---

### **8.7 Progress Monitoring Dashboard**

**Real-time tracking** với MLflow UI:
```powershell
mlflow ui --backend-store-uri .\mlruns --port 5000
```

**Custom dashboard queries** (via MLflow API):
```python
# Get best run per phase
from mlflow.tracking import MlflowClient
client = MlflowClient()

phases = ['baseline', 'intermediate', 'advanced']
for phase in phases:
    exp = client.get_experiment_by_name(f'cardio_phase_{phase}')
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=['metrics.mean_pr_auc DESC'],
        max_results=1
    )
    best = runs[0]
    print(f"{phase}: {best.data.params['model']} PR-AUC={best.data.metrics['mean_pr_auc']:.4f}")
```

---

### **8.8 Automated Report Generation**

Sau mỗi phase, tự động generate summary:

```python
# src/experiment/phase_report.py
def generate_phase_report(phase: str):
    """Generate markdown + plots summarizing phase results"""
    
    # 1. Load all runs from phase
    runs_df = mlflow.search_runs(experiment_names=[f'cardio_phase_{phase}'])
    
    # 2. Best model per metric
    best_pr_auc = runs_df.loc[runs_df['metrics.mean_pr_auc'].idxmax()]
    best_recall = runs_df.loc[runs_df['metrics.mean_recall'].idxmax()]
    
    # 3. Statistical comparison (top 5)
    top5 = runs_df.nlargest(5, 'metrics.mean_pr_auc')
    mcnemar_matrix = compute_mcnemar_tests(top5)
    
    # 4. Visualization
    plot_phase_comparison(runs_df, phase)
    plot_hyperparameter_importance(runs_df, phase)
    
    # 5. Markdown report
    report = f"""
    # Phase {phase} Summary
    
    ## Best Models
    - **PR-AUC**: {best_pr_auc['params.model']} ({best_pr_auc['metrics.mean_pr_auc']:.4f})
    - **Recall**: {best_recall['params.model']} ({best_recall['metrics.mean_recall']:.4f})
    
    ## Key Findings
    {generate_insights(runs_df)}
    
    ## Next Phase Recommendations
    {recommend_next_phase_config(runs_df)}
    """
    
    save_report(report, f'experiments/reports/phase_{phase}_summary.md')
```

**Run after each phase**:
```powershell
python -m src.experiment.phase_report --phase=baseline
python -m src.experiment.phase_report --phase=intermediate
python -m src.experiment.phase_report --phase=advanced
```

---

## 9) Sản phẩm đầu ra

### **9.1 Bảng tổng hợp chính (experiments/results_summary.csv)**

Gồm các cột:
- **Metadata**: `experiment_id`, `timestamp`, `generation` (1-4), `model_name`
- **Pipeline config**: `missing_strategy`, `scaler`, `encoder`, `outlier_method`, `imbalance_method`, `feature_selector`, `n_features_final`
- **Metrics**: `recall`, `precision`, `f1`, `pr_auc`, `roc_auc`, `specificity`, `balanced_acc`, `brier_score`
- **Performance**: `train_time_sec`, `inference_time_ms`, `cv_std` (stability metric)
- **Statistical**: `mcnemar_p_vs_baseline`, `delong_p_vs_baseline`

### **9.2 Báo cáo hình ảnh (experiments/figures/)**

1. **Progressive Performance Evolution**:
   - `gen_comparison_boxplot.png`: PR-AUC distribution cho mỗi generation (Gen1-Gen4)
   - `time_vs_performance.png`: Scatter plot training time vs PR-AUC (Pareto frontier)
   - `generation_heatmap.png`: Heatmap all metrics across generations

2. **Model-specific Analysis**:
   - `roc_curves_top5.png`: ROC curves cho top 5 models
   - `pr_curves_top5.png`: Precision-Recall curves cho top 5 models
   - `confusion_matrices_grid.png`: 2×2 grid confusion matrices (best per generation)

3. **Feature Analysis**:
   - `feature_importance_evolution.png`: Bar chart feature importance qua generations
   - `shap_summary_top_model.png`: SHAP summary plot cho best model
   - `correlation_heatmap.png`: Feature correlation matrix

4. **Statistical Comparison**:
   - `mcnemar_matrix.png`: Heatmap McNemar test p-values (model pairwise)
   - `delong_comparison.png`: DeLong test results cho AUC comparison

### **9.3 Báo cáo thống kê (experiments/reports/)**

1. **progressive_report.md**: Markdown report tóm tắt findings, bao gồm:
   ```markdown
   # Progressive Model Evolution Report
   
   ## Executive Summary
   - Best Model: [XGBoost with SMOTE-ENN]
   - Best PR-AUC: [0.94 ± 0.02]
   - Performance gain vs Baseline: [+15% PR-AUC]
   - Optimal generation: [Gen 3 - Advanced]
   
   ## Generation Analysis
   ### Gen 1 (Baseline): Mean PR-AUC = 0.79
   ### Gen 2 (Intermediate): Mean PR-AUC = 0.88 (+11%)
   ### Gen 3 (Advanced): Mean PR-AUC = 0.93 (+6%)
   ### Gen 4 (SOTA): [Not applicable for tabular data]
   
   ## Key Findings
   1. SMOTE-ENN consistently outperforms class_weight (+3-5% Recall)
   2. RobustScaler better than StandardScaler for this dataset (+2% metrics)
   3. XGBoost marginally beats LightGBM (0.94 vs 0.93 PR-AUC, p=0.12)
   4. MLP underperforms tree-based methods (-4% PR-AUC) for tabular data
   
   ## Statistical Tests
   - McNemar: XGBoost vs RandomForest (p < 0.01, significant)
   - DeLong: XGBoost vs CatBoost AUC (p = 0.08, not significant)
   ```

2. **statistical_tests.csv**: Chi tiết p-values cho tất cả pairwise comparisons

3. **hyperparameter_importance.json**: Optuna feature importance cho hyperparameters

### **9.4 MLflow Artifacts**

Mỗi run log:
- `params/`: All hyperparameters + pipeline configs
- `metrics/`: Time-series metrics (cho DL: loss/epoch curves)
- `artifacts/`:
  - `model.pkl` hoặc `model.pth`: Serialized model
  - `confusion_matrix.png`
  - `roc_curve.png`, `pr_curve.png`
  - `feature_importance.csv` (nếu có)
  - `pipeline.pkl`: Full sklearn pipeline (reproducibility)

### **9.5 Reproducibility Package**

- `experiments/config_best_model.yaml`: Hydra config của best model
- `experiments/requirements_frozen.txt`: Pinned versions (`pip freeze`)
- `experiments/seed_info.json`: Random seeds used
- `experiments/run_best_model.sh`: Script để reproduce exact best run

---

## 10) Lộ trình triển khai (Milestones)

1. Khởi tạo repo + requirements + skeleton src
2. EDA + data validation + chuẩn hóa schema (age_years, bmi)
3. Xây dựng các transformer: Missing, Outlier, Scaling, Encoding, FeatureSelect, PCA
4. Thiết lập imbalance module (SMOTE, SMOTEENN, class_weight)
5. Gói các models + search spaces; thiết lập Optuna objective
6. Cross-validation loop với Hydra + MLflow logging
7. Tổng hợp bảng kết quả và vẽ biểu đồ; thống kê so sánh
8. Viết README + hướng dẫn chạy; đóng gói scripts

---

## 11) Chi tiết triển khai (API đề xuất)

- `src/data/dataset.py`
  - load_csv(path, parse_config) → DataFrame
  - validate_schema(df) → df_clean, report
- `src/preprocessing/pipeline.py`
  - build_preprocess(cfg) → sklearn Pipeline/ColumnTransformer
- `src/features/selection.py`
  - get_selector(cfg, estimator) → selector step
- `src/imbalance/sampler.py`
  - get_sampler(cfg) → imblearn Pipeline step
- `src/models/zoo.py`
  - get_model(cfg) → estimator & param_space
- `src/training/cv_runner.py`
  - run_cv(cfg, X, y) → metrics per fold, logs MLflow
- `src/evaluation/metrics.py`
  - compute_metrics(y_true, y_proba, threshold)
  - plot_roc_pr(...)
- `src/experiment/run.py`
  - Hydra main: iterate configs or accept overrides; orchestrate Optuna + CV

---

## 12) Cấu hình mẫu Hydra (rút gọn)

```yaml
# src/configs/config.yaml
seed: 42
cv:
  n_splits: 5
  shuffle: true
  stratified: true
  test_size: 0.2
preprocessing:
  missing: mice   # [delete, mean, median, knn, mice]
  outliers: iqr_clip  # [none, iqr_clip, zscore_clip]
  scale: standard     # [standard, minmax, robust, none]
  encode: onehot      # [onehot, ordinal, target]
features:
  correlation_threshold: 0.9
  selector: rfe        # [none, rfe, l1, tree_importance]
  pca:
    use: false
    explained_variance: 0.9
imbalance:
  method: smoteenn     # [none, class_weight, smote, smoteenn]
model:
  name: xgb            # [lr, dt, rf, xgb, lgbm, cat, mlp]
  optimize: pr_auc     # metric objective
optuna:
  n_trials: 50
tracking:
  mlflow_uri: ./mlruns
  experiment_name: cardio_baselines
```

---

## 13) Hướng dẫn chạy (sau khi code hoàn tất)

- Tải dữ liệu Kaggle, đặt vào `data/raw/cardio_train.csv` (hoặc tên tương ứng)
- Cài đặt phụ thuộc và chạy experiments:

```powershell
# Windows PowerShell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.experiment.run +model.name=rf preprocessing.missing=median imbalance.method=smote
python -m src.experiment.run +model.name=xgb optuna.n_trials=30
```

- Xem kết quả trong `experiments/` và giao diện MLflow:

```powershell
mlflow ui --backend-store-uri .\mlruns
```

---

## 14) Ghi chú & rủi ro

- Tránh data leakage bằng mọi giá (encoding/target, scaling, SMOTE) – chỉ fit trên train fold.
- Với dataset tabular này, DL có thể không vượt cây tăng cường; vẫn đưa để so sánh.
- Cân nhắc calibrate threshold theo mục tiêu lâm sàng (Recall ưu tiên) và báo cáo PPV/NPV, LR+/LR-.
- Nếu thiếu tài nguyên, ưu tiên RF, XGB, CatBoost + SMOTE-ENN.

---

## 15) Tiếp theo (sau kế hoạch)

- Khởi tạo `requirements.txt`, scaffold `src/` và file cấu hình tối thiểu.
- Viết script import dữ liệu + validate schema, tạo `bmi` và `age_years`.
- Lập danh sách ma trận thí nghiệm “nhỏ” để chạy sanity check (ví dụ 12 run) trước khi mở rộng Optuna.
