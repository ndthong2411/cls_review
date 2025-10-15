# Full Comparison Script Guide

## 📊 Overview

`full_comparison.py` là script toàn diện để so sánh **TẤT CẢ** các models và preprocessing strategies theo đúng plan trong `25_10_15_PROJECT_PLAN.md`.

---

## 🎯 What It Does

### Models Tested (4 Generations)

#### **Generation 1: Baseline** (Expected PR-AUC: 0.72-0.85)
- ✅ Logistic Regression
- ✅ Decision Tree
- ✅ K-Nearest Neighbors

#### **Generation 2: Intermediate** (Expected PR-AUC: 0.86-0.92)
- ✅ Random Forest
- ✅ Extra Trees
- ✅ Gradient Boosting
- ✅ SVM (RBF kernel)
- ✅ MLP (Sklearn)

#### **Generation 3: Advanced** (Expected PR-AUC: 0.90-0.96)
- ✅ XGBoost
- ✅ LightGBM
- ✅ CatBoost

#### **Generation 4: SOTA** (Not in this script)
- ⚠️ CNN-LSTM, Transformers - Requires multi-modal data (ECG signals)
- ⚠️ Federated Learning - Requires distributed setup

---

## 🔧 Preprocessing Strategies Tested

### 1. **Scaling Methods**
- `standard`: StandardScaler (mean=0, std=1)
- `minmax`: MinMaxScaler (range 0-1)
- `robust`: RobustScaler (median-based, outlier-resistant)
- `none`: No scaling (for tree-based models)

### 2. **Imbalance Handling**
- `none`: No resampling (use class weights)
- `smote`: Synthetic Minority Over-sampling
- `adasyn`: Adaptive Synthetic Sampling
- `smote_enn`: SMOTE + Edited Nearest Neighbors (hybrid)

### 3. **Feature Selection**
- `none`: Use all 15 features
- `select_k_best_10`: Top 10 features by ANOVA F-test
- `select_k_best_12`: Top 12 features by ANOVA F-test

---

## 📈 Evaluation Metrics (Medical Focus)

### Primary Metrics
- **PR-AUC**: Precision-Recall AUC (best for imbalanced data)
- **Sensitivity**: True Positive Rate (critical for screening)
- **Specificity**: True Negative Rate (avoid false alarms)
- **F1-Score**: Harmonic mean of precision/recall

### Secondary Metrics
- **ROC-AUC**: Receiver Operating Characteristic AUC
- **Precision**: Positive Predictive Value (PPV)
- **NPV**: Negative Predictive Value
- **MCC**: Matthews Correlation Coefficient
- **Balanced Accuracy**: Average of sensitivity + specificity

---

## 🚀 Usage

### Basic Run
```bash
python full_comparison.py
```

### Expected Output
```
experiments/full_comparison/
└── full_comparison_YYYYMMDD_HHMMSS.csv
```

---

## 📊 Experiment Matrix

### Total Experiments Calculation

For **each model**:
- If model needs scaling: 2 scalers (standard, robust) × 3 imbalance methods × 2 feature selections = **12 configs**
- If model doesn't need scaling: 1 scaler (none) × 3 imbalance methods × 2 feature selections = **6 configs**

**Models that need scaling** (6): LR, KNN, SVM, MLP
**Models that don't need scaling** (7): DT, RF, ExtraTrees, GB, XGB, LGBM, CatBoost

**Total**: 6 × 12 + 7 × 6 = 72 + 42 = **114 experiments**

---

## ⏱️ Estimated Runtime

| Phase | Models | Time per Config | Total Time |
|-------|--------|-----------------|------------|
| Gen 1 (3 models) | LR, DT, KNN | 10-30s | 5-15 min |
| Gen 2 (5 models) | RF, ET, GB, SVM, MLP | 30-120s | 15-60 min |
| Gen 3 (3 models) | XGB, LGBM, CatBoost | 60-180s | 30-90 min |
| **TOTAL** | **11 models** | **114 configs** | **50-165 min** |

**Recommendation**: Run overnight or on powerful machine with `n_jobs=-1`.

---

## 📋 Output Format

### CSV Columns

```csv
model,generation,scaler,imbalance,feature_selection,
accuracy,balanced_accuracy,sensitivity,specificity,precision,npv,f1,roc_auc,pr_auc,mcc,
accuracy_std,balanced_accuracy_std,sensitivity_std,specificity_std,precision_std,npv_std,f1_std,roc_auc_std,pr_auc_std,mcc_std,
train_time_sec
```

### Example Row
```csv
Gen3_XGBoost,3,none,smote_enn,select_k_best_12,0.7234,0.7245,0.9145,0.5346,0.6789,0.8567,0.7812,0.8234,0.9456,0.4567,0.0123,0.0145,...
```

---

## 📊 Analysis & Visualization

After running, use this code to analyze results:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('experiments/full_comparison/full_comparison_XXXXXX.csv')

# 1. Top 10 configurations
top10 = df.nlargest(10, 'pr_auc')
print(top10[['model', 'generation', 'scaler', 'imbalance', 'pr_auc', 'sensitivity', 'f1']])

# 2. Best per generation
for gen in [1, 2, 3]:
    best = df[df['generation'] == gen].nlargest(1, 'pr_auc').iloc[0]
    print(f"Gen {gen}: {best['model']} - PR-AUC={best['pr_auc']:.4f}")

# 3. Scaling impact (for models that need it)
scaling_models = df[df['scaler'].isin(['standard', 'robust'])]
print(scaling_models.groupby('scaler')['pr_auc'].mean())

# 4. Imbalance strategy impact
print(df.groupby('imbalance')['pr_auc'].mean().sort_values(ascending=False))

# 5. Feature selection impact
print(df.groupby('feature_selection')['pr_auc'].mean().sort_values(ascending=False))

# 6. Generation comparison boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='generation', y='pr_auc')
plt.title('PR-AUC Distribution by Generation')
plt.xlabel('Model Generation')
plt.ylabel('PR-AUC')
plt.savefig('experiments/full_comparison/generation_comparison.png', dpi=300)
plt.show()

# 7. Training time vs Performance
plt.figure(figsize=(12, 6))
scatter = plt.scatter(df['train_time_sec'], df['pr_auc'], 
                      c=df['generation'], cmap='viridis', s=100, alpha=0.6)
plt.colorbar(scatter, label='Generation')
plt.xlabel('Training Time (seconds)')
plt.ylabel('PR-AUC')
plt.title('Efficiency Frontier: Training Time vs Performance')
plt.grid(True, alpha=0.3)
plt.savefig('experiments/full_comparison/efficiency_frontier.png', dpi=300)
plt.show()

# 8. Heatmap: Model × Imbalance Strategy
pivot = df.pivot_table(values='pr_auc', index='model', columns='imbalance', aggfunc='mean')
plt.figure(figsize=(10, 8))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.85)
plt.title('PR-AUC: Model × Imbalance Strategy')
plt.tight_layout()
plt.savefig('experiments/full_comparison/model_imbalance_heatmap.png', dpi=300)
plt.show()
```

---

## 🎯 Key Research Questions Answered

### 1. **Which generation performs best?**
→ Compare mean PR-AUC across Gen 1/2/3

### 2. **Is scaling necessary for tree-based models?**
→ Compare models with/without scaling (should be similar for RF/GB/XGB)

### 3. **Which imbalance strategy is best?**
→ Compare SMOTE vs ADASYN vs SMOTE-ENN across models

### 4. **Does feature selection help?**
→ Compare performance with all features vs SelectKBest

### 5. **Training time vs Performance tradeoff?**
→ Identify models with best PR-AUC/time ratio

### 6. **Sensitivity vs Specificity tradeoff?**
→ Medical screening needs high sensitivity (detect disease)
→ Confirmatory tests need high specificity (avoid false positives)

---

## 📊 Expected Results (Based on Plan)

### Generation 1 (Baseline)
```
Logistic Regression: PR-AUC ~0.78-0.82
Decision Tree:       PR-AUC ~0.72-0.76
KNN:                 PR-AUC ~0.74-0.78
```

### Generation 2 (Intermediate)
```
Random Forest:       PR-AUC ~0.86-0.90
Extra Trees:         PR-AUC ~0.85-0.89
Gradient Boosting:   PR-AUC ~0.87-0.91
SVM-RBF:             PR-AUC ~0.84-0.88
MLP:                 PR-AUC ~0.83-0.87
```

### Generation 3 (Advanced)
```
XGBoost:    PR-AUC ~0.92-0.96 ⭐ (Expected best)
LightGBM:   PR-AUC ~0.91-0.95
CatBoost:   PR-AUC ~0.91-0.95
```

---

## 🔍 Debugging

### Issue: Import errors (XGBoost/LightGBM/CatBoost)
**Solution**: Install missing packages
```bash
pip install xgboost lightgbm catboost
```

### Issue: Memory error
**Solution**: Reduce CV folds or run on subset
```python
# In CONFIG at top of file
CONFIG = {
    'cv_folds': 3,  # Instead of 5
}
```

### Issue: Too slow
**Solution**: Test on smaller experiment matrix first
```python
# In main(), change product() to:
for scaler, imbalance, feat_sel in product(
    ['robust'],  # Only 1 scaler
    ['smote'],   # Only 1 imbalance method
    ['none']     # Only 1 feature selection
):
```

---

## 🎓 Learning Insights

### Why This Approach?
1. **Comprehensive**: Tests ALL combinations systematically
2. **Fair Comparison**: Same CV splits, same preprocessing order
3. **No Data Leakage**: Fit transformers on each fold separately
4. **Reproducible**: Fixed random seeds throughout
5. **Medical Focus**: Metrics prioritize sensitivity/specificity

### Preprocessing Order (CRITICAL)
```
Original Data
    ↓
Feature Engineering (age_years, BMI, etc.)
    ↓
Train/Test Split (stratified)
    ↓
CV Fold Split
    ↓
Feature Selection (fit on fold train, transform fold val)
    ↓
Imbalance Handling (ONLY on fold train)
    ↓
Scaling (fit on resampled fold train, transform fold val)
    ↓
Model Training
```

**Why this order?**
- Feature selection BEFORE imbalance → Avoid selecting synthetic features
- Imbalance BEFORE scaling → Scale real + synthetic data together
- All fitting on fold train → No leakage to validation fold

---

## 📚 Next Steps

### 1. **Run Full Comparison**
```bash
python full_comparison.py
```

### 2. **Analyze Results**
Use the analysis code above to create visualizations

### 3. **Hyperparameter Tuning** (Top 3 models)
```bash
# Use Optuna for top 3 models from full_comparison
python advanced_tuning.py --models XGBoost,LightGBM,CatBoost
```

### 4. **Update Streamlit App**
Load `full_comparison_XXXXXX.csv` into `app.py` for visualization

### 5. **Statistical Testing**
- McNemar test for model comparison
- DeLong test for ROC-AUC comparison
- Wilcoxon signed-rank test for CV fold differences

---

## 📌 Quick Reference

| Command | Purpose |
|---------|---------|
| `python full_comparison.py` | Run all 114 experiments |
| `ls experiments/full_comparison/` | Check results |
| `head -n 20 experiments/full_comparison/*.csv` | Preview results |

| Config File | Purpose |
|-------------|---------|
| `full_comparison.py` | Main script |
| `requirements.txt` | Dependencies |
| `docs/25_10_15_PROJECT_PLAN.md` | Original methodology plan |

---

**Created**: 2025-10-15  
**Author**: CVD Pipeline Project  
**Status**: Ready to run  
**Estimated Runtime**: 50-165 minutes (114 experiments)
