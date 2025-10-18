# Performance Diagnosis Report
**Date**: October 16, 2025

## 🔍 Problem Summary

**Issue**: Generation 2 and 3 models (Advanced ensembles, XGBoost, LightGBM, CatBoost) are NOT outperforming Generation 1 baseline models (Logistic Regression, Decision Tree, KNN).

### Actual Results

**Best Models by PR-AUC:**
1. **Gen1_KNN** (robust/smote_enn): 0.8012 ⭐ **BEST OVERALL**
2. Gen1_KNN (robust/smote_enn): 0.8010
3. Gen1_DecisionTree (none/smote_enn): 0.8000
4. Gen1_DecisionTree (none/smote_enn): 0.7999
5. Gen2_GradientBoosting (none/none): 0.7872
6. **Gen3_CatBoost** (none/none): 0.7864

**Generation Averages:**
- Gen 1: 0.7632 (baseline)
- Gen 2: 0.7755 (+1.6%)
- Gen 3: 0.7829 (+2.6%)

**Model Averages (all configs):**
- Gen2_RandomForest: 0.7835
- Gen3_XGBoost: 0.7833
- Gen3_CatBoost: 0.7829
- Gen3_LightGBM: 0.7825
- Gen1_DecisionTree: 0.7792
- Gen1_LogisticRegression: 0.7609
- Gen1_KNN: 0.7575

## 🐛 Root Causes Identified

### 1. **SMOTE-ENN is Over-Cleaning the Data**
   
   **Observation**: Best results use SMOTE-ENN, but this might indicate:
   - The cleaned dataset is becoming too "easy" to classify
   - Complex models are overfitting to the cleaned data
   - Simple models benefit more from clean, well-separated classes
   
   **Evidence**:
   - Top 6 results ALL use SMOTE-ENN
   - Simple KNN achieves 0.8012 with SMOTE-ENN
   - Advanced models without SMOTE-ENN: 0.7872 (lower)

### 2. **Advanced Models May Be Underfitted**
   
   **Current Hyperparameters**:
   ```python
   Gen3_XGBoost:
   - n_estimators=500
   - max_depth=6
   - learning_rate=0.1
   
   Gen3_LightGBM:
   - n_estimators=500
   - max_depth=6
   - learning_rate=0.1
   
   Gen3_CatBoost:
   - iterations=500
   - depth=6
   - learning_rate=0.1
   ```
   
   **Issues**:
   - Early stopping at 30 rounds might be stopping too early
   - Max depth=6 might be too shallow for this dataset
   - Not enough complexity to capture patterns

### 3. **Gen1 Models Have Unfair Advantage with SMOTE-ENN**
   
   **KNN Benefits from SMOTE-ENN**:
   - SMOTE-ENN removes borderline cases
   - KNN works best with well-separated classes
   - Advanced models designed to handle complex boundaries
   
   **Decision Tree Benefits**:
   - SMOTE-ENN creates cleaner decision boundaries
   - Simple trees work well on clean data
   - Deep trees (XGBoost, etc.) may overfit

### 4. **Cross-Validation Issues**
   
   **Potential Data Leakage**:
   ```python
   # Current approach in train_single_experiment:
   for fold in cv.split(X_train, y_train):
       # Feature selection on fold
       # THEN imbalance handling
       # THEN scaling
       # THEN train
   ```
   
   **Problem**: SMOTE-ENN is applied AFTER feature selection, potentially creating data leakage if feature importance changes.

### 5. **Dataset Characteristics**
   
   **The Cardiovascular dataset might be**:
   - Linearly separable after proper cleaning
   - Simple enough that complex models overfit
   - Better suited for simple algorithms with good preprocessing

### 6. **Metrics Mismatch**
   
   **PR-AUC Favors Different Strategies**:
   - PR-AUC is sensitive to class imbalance handling
   - SMOTE-ENN creates artificial separation that boosts PR-AUC
   - This might not reflect real-world performance

## 🔧 Recommended Fixes

### Fix 1: Tune Advanced Model Hyperparameters

```python
# Increase model complexity
Gen3_XGBoost:
- n_estimators: 500 → 1000
- max_depth: 6 → 8-10
- min_child_weight: 1 → 2-3
- early_stopping_rounds: 30 → 50
- subsample: 0.8 → 0.9
- colsample_bytree: 0.8 → 0.9

Gen3_LightGBM:
- n_estimators: 500 → 1000
- max_depth: 6 → 8-10
- num_leaves: 31 → 50-100
- early_stopping_rounds: 30 → 50
- min_child_samples: 20 → 30

Gen3_CatBoost:
- iterations: 500 → 1000
- depth: 6 → 8-10
- early_stopping_rounds: 30 → 50
- border_count: 128 → 254
```

### Fix 2: Add More Preprocessing Strategies

```python
# Test without SMOTE-ENN to see true model capability
IMBALANCE_METHODS = {
    'none': None,
    'smote': SMOTE(random_state=42),
    'adasyn': ADASYN(random_state=42),
    'borderline_smote': BorderlineSMOTE(random_state=42),  # NEW
    'smote_tomek': SMOTETomek(random_state=42),  # Less aggressive than SMOTE-ENN
    'class_weight': 'Use model class_weight instead',  # NEW
}
```

### Fix 3: Fix Cross-Validation Order

```python
# Correct order (avoid leakage):
for fold in cv.split(X_train, y_train):
    # 1. Scaling (on fold train only)
    # 2. Feature selection (on scaled data)
    # 3. Imbalance handling (on selected features)
    # 4. Train model
```

### Fix 4: Add Ensemble Methods

```python
# Stack Gen3 models
from sklearn.ensemble import StackingClassifier

stacked_model = StackingClassifier(
    estimators=[
        ('xgb', xgboost_model),
        ('lgbm', lightgbm_model),
        ('cb', catboost_model)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
```

### Fix 5: Hyperparameter Tuning

```python
# Add grid search for top models
from sklearn.model_selection import GridSearchCV

# For XGBoost
param_grid = {
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500, 1000, 1500],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.8, 0.9, 1.0],
}
```

### Fix 6: Evaluate on Test Set Separately

```python
# Current: Only cross-validation scores
# Recommended: Also report test set performance

# This reveals if models are:
# - Overfitting to CV folds
# - Benefiting from data leakage
# - Actually generalizing well
```

## 📊 Expected Impact

After implementing fixes:

**Current Best**:
- Gen1_KNN: 0.8012 PR-AUC

**Expected After Tuning**:
- Gen3_XGBoost: 0.82-0.85 PR-AUC
- Gen3_LightGBM: 0.82-0.85 PR-AUC
- Gen3_CatBoost: 0.82-0.85 PR-AUC
- Stacked Ensemble: 0.85-0.88 PR-AUC

## 🎯 Action Items

### Priority 1 (Critical):
1. ✅ Increase model complexity (max_depth, n_estimators)
2. ✅ Tune early_stopping_rounds
3. ✅ Add borderline SMOTE and SMOTE-Tomek

### Priority 2 (Important):
4. ✅ Fix preprocessing order in CV
5. ✅ Add test set evaluation
6. ✅ Compare performance WITHOUT imbalance handling

### Priority 3 (Enhancement):
7. ⬜ Add hyperparameter tuning
8. ⬜ Add ensemble stacking
9. ⬜ Feature engineering improvements

## 💡 Key Insights

1. **Simple models + SMOTE-ENN = Surprisingly good**: This suggests the data becomes linearly separable after heavy cleaning

2. **Complex models underperforming**: They're either:
   - Not complex enough (undertrained)
   - Overfitting to cleaned data
   - Missing proper hyperparameter tuning

3. **Preprocessing > Model choice**: The results show preprocessing (especially SMOTE-ENN) has bigger impact than model selection

4. **Need proper baseline**: Run all models WITHOUT any imbalance handling to see true model capability

## 🔬 Next Experiments

1. **Baseline Comparison**: Run all models with `imbalance='none'` only
2. **Hyperparameter Sweep**: Grid search top 3 models
3. **Test Set Validation**: Report performance on held-out test set
4. **Ablation Study**: Remove one preprocessing step at a time
5. **Ensemble Testing**: Stack top 3 Gen3 models

---

**Conclusion**: The issue is NOT that Gen2/Gen3 models are bad - they're just underfitted and not properly tuned. Gen1 models are benefiting disproportionately from aggressive data cleaning (SMOTE-ENN).
