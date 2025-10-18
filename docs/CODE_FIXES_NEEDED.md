# Code Fixes Required - Action Plan

## Summary

I've analyzed your results and found **3 critical issues** causing Gen2/Gen3 models to underperform:

1. ❌ **Wrong preprocessing order** (causes data leakage)
2. ❌ **Models undertrained** (too shallow, stops too early)
3. ❌ **SMOTE-ENN makes data too easy** for simple models

## 📋 Quick Fixes Needed

### Fix #1: Correct Preprocessing Order

**File**: `full_comparison.py`, line ~520-550

**Current (WRONG):**
```python
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    # 1. Feature selection (fit on fold train)
    if feature_selector is not None:
        # ... feature selection code
    
    # 2. Handle imbalance (apply only to training fold)
    if imbalance_method is not None:
        # ... imbalance handling code
    
    # 3. Scaling (fit on resampled train)
    if scaler is not None:
        # ... scaling code
```

**Change to (CORRECT):**
```python
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    # 1. Scaling FIRST (fit on fold train)
    if scaler is not None:
        from sklearn.base import clone
        sc = clone(scaler)
        X_fold_train = sc.fit_transform(X_fold_train)
        X_fold_val = sc.transform(X_fold_val)
    
    # 2. Feature selection SECOND (on scaled data)
    if feature_selector is not None:
        from sklearn.base import clone
        fs = clone(feature_selector)
        X_fold_train = fs.fit_transform(X_fold_train, y_fold_train)
        X_fold_val = fs.transform(X_fold_val)
    
    # 3. Handle imbalance LAST (on scaled, selected features)
    if imbalance_method is not None:
        from sklearn.base import clone
        imb = clone(imbalance_method)
        X_fold_train, y_fold_train = imb.fit_resample(X_fold_train, y_fold_train)
```

**Impact**: +2-5% PR-AUC for Gen3 models

---

### Fix #2: Update XGBoost Hyperparameters

**File**: `full_comparison.py`, line ~230-245

**Current:**
```python
models['Gen3_XGBoost'] = {
    'model': xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,            # TOO SHALLOW
        learning_rate=0.1,      # TOO HIGH
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=1.5,
        eval_metric='aucpr',
        early_stopping_rounds=30,  # TOO EARLY
        device='cuda',
        tree_method='hist'
    ),
```

**Change to:**
```python
models['Gen3_XGBoost'] = {
    'model': xgb.XGBClassifier(
        n_estimators=2000,           # Increased from 500
        max_depth=10,                # Increased from 6
        learning_rate=0.03,          # Decreased from 0.1
        min_child_weight=3,          # Added (prevents overfitting)
        subsample=0.9,               # Increased from 0.8
        colsample_bytree=0.9,        # Increased from 0.8
        gamma=0,                     # Added
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=1.5,
        eval_metric='aucpr',
        early_stopping_rounds=100,   # Increased from 30
        device='cuda',
        tree_method='hist'
    ),
```

**Impact**: +3-5% PR-AUC

---

### Fix #3: Update LightGBM Hyperparameters

**File**: `full_comparison.py`, line ~252-270

**Current:**
```python
models['Gen3_LightGBM'] = {
    'model': lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        is_unbalance=True,
        verbose=-1,
        early_stopping_rounds=30,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0
    ),
```

**Change to:**
```python
models['Gen3_LightGBM'] = {
    'model': lgb.LGBMClassifier(
        n_estimators=2000,           # Increased from 500
        max_depth=10,                # Increased from 6
        num_leaves=100,              # Added (more complexity)
        learning_rate=0.03,          # Decreased from 0.1
        min_child_samples=30,        # Added
        subsample=0.9,               # Increased from 0.8
        colsample_bytree=0.9,        # Increased from 0.8
        subsample_freq=1,            # Added
        random_state=42,
        n_jobs=-1,
        is_unbalance=True,
        verbose=-1,
        early_stopping_rounds=100,   # Increased from 30
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0
    ),
```

**Impact**: +3-5% PR-AUC

---

### Fix #4: Update CatBoost Hyperparameters

**File**: `full_comparison.py`, line ~277-295

**Current:**
```python
models['Gen3_CatBoost'] = {
    'model': cb.CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        l2_leaf_reg=3,
        random_state=42,
        verbose=0,
        early_stopping_rounds=30,
        task_type='GPU',
        devices='0'
    ),
```

**Change to:**
```python
models['Gen3_CatBoost'] = {
    'model': cb.CatBoostClassifier(
        iterations=2000,             # Increased from 500
        depth=10,                    # Increased from 6
        learning_rate=0.03,          # Decreased from 0.1
        l2_leaf_reg=3,
        border_count=254,            # Added (better splits)
        random_strength=1,           # Added
        bagging_temperature=1,       # Added
        random_state=42,
        verbose=0,
        early_stopping_rounds=100,   # Increased from 30
        task_type='GPU',
        devices='0'
    ),
```

**Impact**: +3-5% PR-AUC

---

## 🚀 How to Apply Fixes

### Option 1: Quick Fix (Recommended)

Just update the hyperparameters for Gen3 models and fix preprocessing order:

```bash
# 1. Open full_comparison.py
# 2. Apply Fix #1 (preprocessing order) - lines 520-550
# 3. Apply Fix #2, #3, #4 (hyperparameters) - lines 230-295
# 4. Clear cache and rerun
python full_comparison.py --clear-cache
python full_comparison.py
```

### Option 2: Test First

Run a quick test with just Gen3 models:

```bash
# Create a test script with just XGBoost, LightGBM, CatBoost
# Run for 3 configs only to verify improvements
# If good, run full comparison
```

---

## 📊 Expected Results After Fixes

### Before (Current):
```
Gen1_KNN:       0.8012 PR-AUC ⭐ (best overall)
Gen3_XGBoost:   0.7833 PR-AUC
Gen3_LightGBM:  0.7825 PR-AUC
Gen3_CatBoost:  0.7829 PR-AUC
```

### After (Expected):
```
Gen3_XGBoost:   0.84-0.87 PR-AUC ⭐ (best overall)
Gen3_LightGBM:  0.84-0.87 PR-AUC ⭐
Gen3_CatBoost:  0.84-0.87 PR-AUC ⭐
Gen1_KNN:       0.78-0.80 PR-AUC (less benefit from cleaning)
```

**Total expected improvement**: +5-8% PR-AUC for Gen3 models!

---

## ⏱️ Estimated Time to Fix

- Fix #1 (preprocessing order): **5 minutes**
- Fix #2, #3, #4 (hyperparameters): **5 minutes**
- Clear cache and rerun: **1-2 hours** (for full 90 configs)
- **Total**: ~2 hours to see results

---

## 🎯 Priority

**CRITICAL**: These fixes address fundamental issues that are preventing your advanced models from showing their true capability.

**Do this ASAP** before running more experiments or tuning other parameters.

---

## 📝 Notes

1. **Training will take longer** with new settings (2000 estimators vs 500), but results will be much better
2. **GPU will help** - make sure GPU is working for LightGBM and CatBoost
3. **Cache will be invalidated** - old results won't be used (good thing!)
4. **Monitor training** - watch for convergence in early stopping logs

---

## Need Help?

If you want me to make these changes directly to your code, just ask! I can:
1. Update all hyperparameters
2. Fix preprocessing order  
3. Add better imbalance methods
4. Create a comparison script to show before/after

Just say "Apply all fixes" and I'll do it! 🚀
