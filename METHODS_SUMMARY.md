# TÓM TẮT CÁC METHODS ĐANG SỬ DỤNG

## 📊 **1. MACHINE LEARNING MODELS** (13 models, 4 thế hệ)

### **Generation 1 - Baseline Models** (3 models)

1. **Logistic Regression**
   - Linear classifier với L2 regularization
   - Solver: LBFGS
   - Class weight: Balanced
   - Max iterations: 1000

2. **Decision Tree**
   - Tree-based với pruning
   - Max depth: 10
   - Min samples leaf: 20
   - Min samples split: 40
   - Class weight: Balanced

3. **K-Nearest Neighbors (KNN)**
   - Distance-based classifier
   - k=5 neighbors
   - Weights: Distance-weighted
   - Metric: Minkowski

---

### **Generation 2 - Intermediate/Ensemble** (5 models)

4. **Random Forest**
   - Ensemble of 300 decision trees
   - Bagging strategy
   - Max depth: 15
   - Min samples leaf: 10
   - Class weight: Balanced

5. **Extra Trees**
   - 300 extremely randomized trees
   - Variance reduction strategy
   - Max depth: 15
   - Min samples leaf: 10
   - Class weight: Balanced

6. **Gradient Boosting**
   - Sequential boosting với gradient descent
   - 200 estimators
   - Learning rate: 0.1
   - Max depth: 5
   - Early stopping: patience=20

7. **SVM (RBF kernel)**
   - Support Vector Machine
   - Kernel: Radial Basis Function
   - C: 1.0
   - Gamma: scale
   - Class weight: Balanced

8. **MLP (Sklearn)**
   - Multi-layer perceptron neural network
   - Architecture: 128 → 64
   - Activation: ReLU
   - Solver: Adam
   - Learning rate: 0.001
   - Max iterations: 500
   - Early stopping: validation_fraction=0.1, patience=20

---

### **Generation 3 - Advanced Boosting** (3 models, GPU-accelerated)

9. **XGBoost** 🚀 GPU
   - eXtreme Gradient Boosting
   - 2000 estimators
   - Max depth: 10
   - Learning rate: 0.03
   - Subsample: 0.9
   - Colsample bytree: 0.9
   - GPU acceleration: `tree_method='gpu_hist'`
   - Early stopping: rounds=100

10. **LightGBM** 🚀 GPU
    - Light Gradient Boosting Machine
    - 2000 estimators
    - Max depth: 10
    - Num leaves: 100
    - Learning rate: 0.03
    - Subsample: 0.9
    - GPU acceleration: `device='gpu'`
    - Early stopping via callbacks

11. **CatBoost** 🚀 GPU
    - Categorical Boosting
    - 2000 iterations
    - Depth: 10
    - Learning rate: 0.03
    - L2 leaf reg: 3
    - Auto class weights: Balanced
    - GPU acceleration: `task_type='GPU'`
    - Early stopping: rounds=100

---

### **Generation 4 - Deep Learning** (2 models)

12. **PyTorch MLP**
    - Deep Multi-Layer Perceptron
    - Architecture: [256, 128, 64, 32]
    - Dropout: [0.4, 0.3, 0.2, 0.1] (progressive)
    - Batch normalization: Yes
    - Optimizer: AdamW
    - Learning rate: 0.001
    - Weight decay: 1e-4
    - Batch size: 128
    - Epochs: 200
    - Scheduler: ReduceLROnPlateau
    - Early stopping: patience=30
    - Auto GPU detection

13. **TabNet**
    - Attention-based tabular learning
    - Interpretable architecture
    - n_d / n_a: 64
    - n_steps: 5 (attention steps)
    - Gamma: 1.5 (feature reusage)
    - Lambda sparse: 1e-4
    - Batch size: 256
    - Virtual batch size: 128
    - Max epochs: 200
    - Patience: 30
    - Auto GPU detection

---

## ⚙️ **2. PREPROCESSING METHODS**

### **Scaling** (4 options)

| Method | Description | Use Case |
|--------|-------------|----------|
| **None** | No scaling | Tree-based models (RF, XGBoost, etc.) |
| **StandardScaler** | Z-score: (x - μ) / σ | Linear models, Neural networks |
| **MinMaxScaler** | Min-max: (x - min) / (max - min) → [0,1] | When bounded range needed |
| **RobustScaler** | Median + IQR scaling | Robust to outliers |

### **Imbalance Handling** (4 options)

| Method | Description | Algorithm |
|--------|-------------|-----------|
| **None** | No resampling | Use original distribution |
| **SMOTE** | Synthetic Minority Oversampling | Generate synthetic minority samples |
| **ADASYN** | Adaptive Synthetic Sampling | Adaptive density-based oversampling |
| **SMOTE-ENN** | Hybrid method | SMOTE + Edited Nearest Neighbors cleaning |

### **Feature Selection** (5 options)

| Method | Description | K features |
|--------|-------------|------------|
| **None** | Use all features | All (15 or 30) |
| **SelectKBest (k=5)** | ANOVA F-test | Top 5 |
| **SelectKBest (k=12)** | ANOVA F-test | Top 12 |
| **Mutual Info (k=5)** | Mutual information score | Top 5 |
| **Mutual Info (k=12)** | Mutual information score | Top 12 |

---

## 🔬 **3. FEATURE ENGINEERING**

### **Cardio Dataset** (15 features total)

#### Engineered Features:
```python
# Age conversion
age_years = age / 365.25

# Body Mass Index
bmi = weight / (height/100)²

# Blood pressure metrics
pulse_pressure = ap_hi - ap_lo
map = (ap_hi + 2 × ap_lo) / 3  # Mean Arterial Pressure
is_hypertension = (ap_hi ≥ 140) | (ap_lo ≥ 90)

# BMI categorization
bmi_category = pd.cut(bmi, bins=[0, 18.5, 25, 30, 100])
```

#### Original Features:
- `age_years`, `gender`, `height`, `weight`, `bmi`
- `ap_hi`, `ap_lo`, `pulse_pressure`, `map`, `is_hypertension`
- `cholesterol`, `gluc`, `smoke`, `alco`, `active`

### **Credit Card Dataset** (30 features total)

#### No Engineering - Raw Features:
```python
Features used:
- V1, V2, V3, ..., V28  # PCA-transformed (anonymized)
- Time                   # Seconds since first transaction
- Amount                 # Transaction amount
```

---

## 📏 **4. EVALUATION METRICS**

### **Primary Metrics**

| Metric | Description | Priority |
|--------|-------------|----------|
| **PR-AUC** | Precision-Recall Area Under Curve | 🥇 Primary ranking metric |
| **Accuracy** | Overall correctness | Standard metric |
| **Balanced Accuracy** | Avg of sensitivity & specificity | For imbalanced data |

### **Medical/Classification Metrics**

| Metric | Formula | Description |
|--------|---------|-------------|
| **Sensitivity (Recall)** | TP / (TP + FN) | True Positive Rate |
| **Specificity** | TN / (TN + FP) | True Negative Rate |
| **Precision (PPV)** | TP / (TP + FP) | Positive Predictive Value |
| **NPV** | TN / (TN + FN) | Negative Predictive Value |
| **F1-Score** | 2 × (Prec × Rec) / (Prec + Rec) | Harmonic mean |
| **ROC-AUC** | Area Under ROC Curve | Classification performance |
| **MCC** | Matthews Correlation | Balanced metric [-1, 1] |

### **Confusion Matrix**
```
                 Predicted
               Neg    Pos
Actual  Neg    TN     FP
        Pos    FN     TP
```

---

## 🔄 **5. EXPERIMENT DESIGN**

### **Grid Search Strategy**

```
Total combinations = Models × Scalers × Imbalance × FeatSel
```

#### Theoretical Maximum:
```
= 13 models × (3 scalers + none) × 4 imbalance × 5 feat_sel
= 13 × 4 × 4 × 5
= 1,040 potential configurations
```

#### Actually Run (filtered):
```
Cardio Dataset:     270 experiments
Credit Card:        270 experiments

Filtering logic:
- Skip scaling for models with needs_scaling=False
- Skip incompatible combinations
```

### **Experiment Matrix Example**

| Model | Scaler | Imbalance | FeatSel | Total Configs |
|-------|--------|-----------|---------|---------------|
| LogisticRegression | standard, minmax, robust | none, smote, adasyn, smote_enn | none, k5, k12, mi5, mi12 | 3 × 4 × 5 = 60 |
| DecisionTree | none only | none, smote, adasyn, smote_enn | none, k5, k12, mi5, mi12 | 1 × 4 × 5 = 20 |
| ... | ... | ... | ... | ... |

### **Train/Test Split**

```python
Train: 80% (stratified)
Test:  20% (stratified)
Random state: 42
```

**Cardio Dataset:**
- Train: 56,000 samples (27,983 positive)
- Test: 14,000 samples (6,996 positive)

**Credit Card Dataset:**
- Train: 227,845 samples (394 positive)
- Test: 56,962 samples (98 positive)

### **Cross-Validation**

```python
CV Folds: 1 (single train/val split in fast mode)
Strategy: Stratified split to maintain class distribution
```

### **Model Caching**

```python
Cache enabled: Yes
Cache directory: experiments/model_cache/{dataset_name}/
Cache key: MD5 hash of {model}_{scaler}_{imbalance}_{featsel}
Cache format: Pickle (.pkl)
Cache includes: Model, scaler, feature_selector, metrics, timestamp
```

---

## 🎯 **6. OPTIMIZATION TECHNIQUES**

### **Regularization**

| Model | Regularization Technique |
|-------|-------------------------|
| Logistic Regression | L2 regularization |
| SVM | C parameter (regularization strength) |
| PyTorch MLP | Dropout (0.4→0.3→0.2→0.1) + Weight decay |
| XGBoost | L2 reg on weights (via min_child_weight) |
| LightGBM | L2 reg (via min_child_samples) |
| CatBoost | L2 leaf reg = 3 |
| TabNet | Lambda sparse = 1e-4 |

### **Class Balancing**

```python
Applied to:
- Logistic Regression: class_weight='balanced'
- Decision Tree: class_weight='balanced'
- Random Forest: class_weight='balanced'
- Extra Trees: class_weight='balanced'
- SVM: class_weight='balanced'
- LightGBM: is_unbalance=True
- CatBoost: auto_class_weights='Balanced'
- PyTorch MLP: class_weight='balanced'
```

### **Early Stopping**

| Model | Strategy | Patience |
|-------|----------|----------|
| Gradient Boosting | validation_fraction=0.1 | 20 iterations |
| MLP Sklearn | validation_fraction=0.1 | 20 iterations |
| XGBoost | eval_set monitoring | 100 rounds |
| LightGBM | eval_set monitoring | 100 rounds |
| CatBoost | eval_set monitoring | 100 rounds |
| PyTorch MLP | validation loss monitoring | 30 epochs |
| TabNet | validation loss monitoring | 30 epochs |

### **GPU Acceleration** 🚀

```python
XGBoost:
  tree_method: 'gpu_hist'
  gpu_id: 0
  predictor: 'gpu_predictor'

LightGBM:
  device: 'gpu'
  gpu_platform_id: 0
  gpu_device_id: 0

CatBoost:
  task_type: 'GPU'
  devices: '0'

PyTorch MLP:
  device: Auto-detect CUDA

TabNet:
  device_name: 'auto'
```

### **Learning Rate Scheduling**

```python
PyTorch MLP:
  Scheduler: ReduceLROnPlateau
  Mode: min (reduce on loss plateau)
  Factor: 0.5
  Patience: 10

Boosting models:
  Fixed learning rate: 0.03
  (Compensated by large n_estimators + early stopping)
```

---

## 📈 **7. HYPERPARAMETER TUNING**

### **XGBoost / LightGBM / CatBoost**

```python
Common hyperparameters:
  n_estimators/iterations: 2000  # Large with early stopping
  max_depth/depth: 10            # Deep trees
  learning_rate: 0.03            # Conservative for stability
  subsample: 0.9                 # Bagging 90% of data
  colsample_bytree: 0.9          # Use 90% of features

XGBoost specific:
  min_child_weight: 3
  gamma: 0
  eval_metric: 'logloss'

LightGBM specific:
  num_leaves: 100                # 2^max_depth rule
  min_child_samples: 30
  subsample_freq: 1

CatBoost specific:
  border_count: 254
  random_strength: 1
  bagging_temperature: 1
  od_type: 'Iter'
```

### **Random Forest / Extra Trees**

```python
Hyperparameters:
  n_estimators: 300
  max_depth: 15
  min_samples_leaf: 10
  min_samples_split: 40
  class_weight: 'balanced'
  n_jobs: -1                     # Use all CPU cores
```

### **Gradient Boosting (Sklearn)**

```python
Hyperparameters:
  n_estimators: 200
  max_depth: 5
  learning_rate: 0.1
  subsample: 0.8
  validation_fraction: 0.1
  n_iter_no_change: 20
```

### **PyTorch MLP**

```python
Architecture:
  hidden_dims: [256, 128, 64, 32]
  dropout_rates: [0.4, 0.3, 0.2, 0.1]
  use_batch_norm: True

Training:
  optimizer: AdamW
  learning_rate: 0.001
  weight_decay: 1e-4
  batch_size: 128
  epochs: 200

Scheduler:
  type: ReduceLROnPlateau
  mode: 'min'
  factor: 0.5
  patience: 10

Early Stopping:
  patience: 30
  monitor: validation_loss
```

### **TabNet**

```python
Architecture:
  n_d: 64                        # Decision prediction width
  n_a: 64                        # Attention embedding width
  n_steps: 5                     # Sequential attention steps
  gamma: 1.5                     # Feature reusage coefficient
  n_independent: 2
  n_shared: 2

Regularization:
  lambda_sparse: 1e-4            # Sparsity penalty
  momentum: 0.3
  clip_value: 2.0

Training:
  optimizer_params: {'lr': 0.02}
  batch_size: 256
  virtual_batch_size: 128
  max_epochs: 200
  patience: 30
  mask_type: 'sparsemax'
```

---

## 📊 **8. DATASET STATISTICS**

### **Cardio Train Dataset**

```
Size: 70,000 samples × 13 columns (15 after engineering)
Target: 'cardio' (cardiovascular disease)

Class Distribution:
  Positive (Disease):  34,979 (50.0%)
  Negative (Healthy):  35,021 (50.0%)
  Imbalance Ratio: 1:1.0 ✅ BALANCED

Train/Test Split (80/20):
  Train: 56,000 samples (27,983 positive)
  Test:  14,000 samples (6,996 positive)
```

### **Credit Card Dataset**

```
Size: 284,807 samples × 31 columns (30 features)
Target: 'Class' (fraud detection)

Class Distribution:
  Positive (Fraud):    492 (0.2%)
  Negative (Normal):   284,315 (99.8%)
  Imbalance Ratio: 1:577.9 ⚠️ HIGHLY IMBALANCED

Train/Test Split (80/20):
  Train: 227,845 samples (394 positive)
  Test:  56,962 samples (98 positive)
```

---

## 🏆 **9. RESULTS SUMMARY**

### **Cardio Train - Best Results**

| Generation | Best Model | PR-AUC | Accuracy | Config |
|------------|-----------|--------|----------|--------|
| Gen1 | DecisionTree | 0.8023 | ~0.72 | none \| smote_enn \| mutual_info_12 |
| Gen2 | GradientBoosting | 0.7865 | ~0.73 | none \| none \| none |
| Gen3 | CatBoost | 0.7864 | ~0.73 | none \| none \| mutual_info_12 |
| Gen4 | PyTorch_MLP | 0.7839 | ~0.72 | standard \| smote \| select_k_best_12 |

**Key Insights:**
- ✅ Simple Gen1 models outperform advanced Gen3/Gen4
- ✅ SMOTE-ENN very effective for balanced dataset
- ✅ Feature selection (k=12) improves performance
- ⚠️ Overfitting risk with deep learning models

### **Credit Card - Best Results**

| Generation | Best Model | PR-AUC | F1-Score | Config |
|------------|-----------|--------|----------|--------|
| Gen1 | KNN | 0.8693 | 0.8595 | standard \| none \| select_k_best_12 |
| Gen2 | RandomForest | 0.8425 | 0.7961 | none \| smote \| none |
| Gen3 | XGBoost | 0.8359 | 0.8429 | none \| none \| none |
| Gen4 | PyTorch_MLP | 0.7882 | 0.7625 | robust \| smote_enn \| none |

**Key Insights:**
- ✅ KNN + feature selection = best performance
- ⚠️ SMOTE may hurt performance (top model uses none)
- ✅ Feature selection critical (k=12 optimal)
- ⚠️ Deep learning underperforms on this dataset

---

## 💡 **10. KEY FINDINGS & RECOMMENDATIONS**

### **General Findings**

1. **Simpler is Better**
   - Gen1 models (DecisionTree, KNN) achieve best results on BOTH datasets
   - Gen3/Gen4 complexity doesn't justify performance gain
   - Occam's Razor principle applies

2. **Preprocessing Depends on Dataset**
   - **Balanced data** (Cardio): SMOTE-ENN effective
   - **Imbalanced data** (Credit): Avoid SMOTE for top performance
   - Feature selection helps both datasets (k=12 sweet spot)

3. **Computational Efficiency**
   - Cardio: ~18 hours total (238s/experiment)
   - Credit Card: ~45 hours total (614s/experiment)
   - Imbalance ratio affects training time significantly

### **Best Practices**

#### For Balanced Datasets (like Cardio):
```python
recommended_config = {
    'model': 'DecisionTree',
    'scaler': 'none',
    'imbalance': 'smote_enn',
    'feature_selection': 'mutual_info_12',
    'expected_pr_auc': 0.80+
}
```

#### For Imbalanced Datasets (like Credit Card):
```python
recommended_config = {
    'model': 'KNN',
    'scaler': 'standard',
    'imbalance': 'none',  # Important!
    'feature_selection': 'select_k_best_12',
    'expected_pr_auc': 0.86+
}
```

### **Future Research Questions**

1. ❓ Why do Gen1 models outperform advanced models?
2. ❓ Is SMOTE harmful for highly imbalanced data?
3. ❓ Would ensemble of Gen1 models beat Gen3/Gen4?
4. ❓ Is k=12 universally optimal across datasets?
5. ❓ Can we explain the feature importance patterns?

---

## 📁 **11. PROJECT STRUCTURE**

```
cls_review/
├── data/
│   └── raw/
│       ├── cardio_train.csv
│       └── creditcard.csv
│
├── experiments/
│   ├── full_comparison/
│   │   ├── cardio_train/
│   │   │   ├── full_comparison_20251018_022851.csv
│   │   │   └── best_model/
│   │   └── creditcard/
│   │       ├── full_comparison_20251018_204737.csv
│   │       └── best_model/
│   │
│   ├── model_cache/
│   │   ├── cardio_train/
│   │   └── creditcard/
│   │
│   └── logs/
│       ├── cardio_train_*.log
│       └── creditcard_*.log
│
├── src/
│   └── models/
│       ├── pytorch_mlp.py
│       └── tabnet_model.py
│
├── full_comparison.py          # Main experiment script
├── analyze_log_files.py        # Log analysis tool
├── LOG_ANALYSIS_REPORT.md      # Detailed results report
└── METHODS_SUMMARY.md          # This file
```

---

## 🔧 **12. USAGE**

### **Run Full Comparison**

```bash
# Cardio dataset
python full_comparison.py --data data/raw/cardio_train.csv

# Credit card dataset
python full_comparison.py --data data/raw/creditcard.csv

# Disable cache (rerun all)
python full_comparison.py --data data/raw/cardio_train.csv --no-cache

# Clear cache
python full_comparison.py --clear-cache

# List cached experiments
python full_comparison.py --list-cache
```

### **Analyze Logs**

```bash
# Run log analysis
python analyze_log_files.py
```

This generates:
- Console output with detailed statistics
- Visualizations in `analysis_output/{dataset}/`
- Comparative analysis between datasets

---

## 📚 **13. REFERENCES**

### **Libraries Used**

- **scikit-learn** - Classical ML algorithms
- **XGBoost** - Gradient boosting
- **LightGBM** - Light gradient boosting
- **CatBoost** - Categorical boosting
- **PyTorch** - Deep learning framework
- **TabNet** - Attention-based tabular learning
- **imbalanced-learn** - SMOTE, ADASYN, SMOTE-ENN
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib/seaborn** - Visualization

### **Algorithms**

1. Logistic Regression - Linear classification
2. Decision Tree (CART) - Tree-based classification
3. K-Nearest Neighbors - Instance-based learning
4. Random Forest (Breiman, 2001) - Ensemble bagging
5. Extra Trees (Geurts et al., 2006) - Randomized trees
6. Gradient Boosting (Friedman, 2001) - Sequential boosting
7. SVM with RBF kernel (Cortes & Vapnik, 1995)
8. Multi-Layer Perceptron - Feed-forward neural network
9. XGBoost (Chen & Guestrin, 2016) - Extreme gradient boosting
10. LightGBM (Ke et al., 2017) - Histogram-based boosting
11. CatBoost (Prokhorenkova et al., 2018) - Ordered boosting
12. TabNet (Arik & Pfister, 2019) - Attention mechanisms

---

**Document Version:** 1.0
**Last Updated:** 2025-10-19
**Author:** Generated from full_comparison.py analysis
