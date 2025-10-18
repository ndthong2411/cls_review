# Why Gen2/Gen3 Models Underperform: Root Cause Analysis

## 📊 Current Results

**Top 10 Models (by PR-AUC):**
1. Gen1_KNN + SMOTE-ENN: **0.8012** ⭐
2. Gen1_KNN + SMOTE-ENN: 0.8010
3. Gen1_DecisionTree + SMOTE-ENN: 0.8000
4. Gen1_DecisionTree + SMOTE-ENN: 0.7999
5. Gen2_GradientBoosting: 0.7872
6. Gen3_CatBoost: 0.7864

**Generation Averages:**
- Gen 1: 0.7632
- Gen 2: 0.7755 *(+1.6%)*
- Gen 3: 0.7829 *(+2.6%)*

**Expected** Gen 3 should be 5-10% better than Gen 1, not just 2.6%!

---

## 🐛 CRITICAL ISSUES FOUND

### Issue #1: ❌ **PREPROCESSING ORDER IS WRONG** (DATA LEAKAGE)

**Current order in `train_single_experiment()`:**
```python
for fold in cv.split(X_train, y_train):
    # 1. Feature selection
    # 2. Imbalance handling (SMOTE/ADASYN)
    # 3. Scaling
    # 4. Train model
```

**Why this is WRONG:**
- Feature selection should happen on **scaled** data (for distance-based methods)
- SMOTE-ENN should be last (to avoid creating synthetic samples that leak information)
- Scaling should be first to normalize features before any operations

**Correct order should be:**
```python
for fold in cv.split(X_train, y_train):
    # 1. Scaling (normalize features first)
    # 2. Feature selection (on scaled data)
    # 3. Imbalance handling (last, on clean features)
    # 4. Train model
```

**Impact:** This alone could improve Gen3 models by 2-5% PR-AUC!

---

### Issue #2: 🎛️ **ADVANCED MODELS ARE UNDERTRAINED**

**Current Hyperparameters:**
```python
Gen3_XGBoost:
    n_estimators=500
    max_depth=6          # TOO SHALLOW!
    learning_rate=0.1    # TOO HIGH!
    early_stopping_rounds=30  # STOPS TOO EARLY!

Gen3_LightGBM:
    n_estimators=500
    max_depth=6          # TOO SHALLOW!
    learning_rate=0.1
    early_stopping_rounds=30

Gen3_CatBoost:
    iterations=500
    depth=6              # TOO SHALLOW!
    learning_rate=0.1
```

**Problems:**
1. **max_depth=6** is too shallow for complex patterns
2. **early_stopping=30** stops before convergence
3. **learning_rate=0.1** is too high, causes oscillation

**Recommended settings:**
```python
Gen3_XGBoost:
    n_estimators=2000         # More trees
    max_depth=10             # Deeper trees
    learning_rate=0.03       # Slower, more stable
    early_stopping_rounds=100 # More patience
    min_child_weight=3       # Prevent overfitting
    subsample=0.9            # More data per tree
    colsample_bytree=0.9

Gen3_LightGBM:
    n_estimators=2000
    max_depth=10
    num_leaves=100           # More complexity
    learning_rate=0.03
    early_stopping_rounds=100
    min_child_samples=30

Gen3_CatBoost:
    iterations=2000
    depth=10
    learning_rate=0.03
    early_stopping_rounds=100
    border_count=254         # Better splits
```

---

### Issue #3: 🧹 **SMOTE-ENN MAKES DATA TOO EASY**

**Observation:**
- ALL top 6 results use SMOTE-ENN
- Gen1 models (KNN, Decision Tree) benefit disproportionately
- Complex models don't show their strength on "cleaned" data

**Why SMOTE-ENN helps simple models:**
- Removes borderline/ambiguous cases
- Creates well-separated classes
- Simple models (KNN) excel on clean, separated data
- Complex models designed to handle messy, complex boundaries

**Solution:**
1. Test models WITHOUT imbalance handling to see true capability
2. Use less aggressive methods (SMOTE-Tomek, BorderlineSMOTE)
3. Rely more on `class_weight` parameter in models

---

### Issue #4: 📉 **EARLY STOPPING TOO AGGRESSIVE**

**Current:** `early_stopping_rounds=30`

**Problem:**
- Validation loss can fluctuate for 50-100 rounds before stabilizing
- Stopping at 30 rounds means models are undertrained
- Gen3 models need more iterations to find optimal patterns

**Evidence from training logs:**
- Models stop around iteration 200-300
- Could likely improve if allowed to run to 800-1200

**Fix:** Increase to `early_stopping_rounds=100`

---

### Issue #5: 🔢 **CROSS-VALIDATION LEAK**

**Current approach:**
```python
# Inside CV fold:
X_train_fit, X_val_fit = train_test_split(X_fold_train, ...)
model.fit(X_train_fit, y_train_fit, eval_set=[(X_val_fit, y_val_fit)])
```

**Problem:**
- Creates nested validation inside CV fold
- Reduces training data further (90% → 81%)
- Gen3 models need MORE data, not less

**Better approach:**
```python
# Use the CV fold itself for validation
model.fit(X_fold_train, y_fold_train, 
         eval_set=[(X_fold_val, y_fold_val)])
```

---

## 🔧 RECOMMENDED FIXES (Priority Order)

### Priority 1: CRITICAL (Do First)

1. **Fix preprocessing order** ✅
   - Change to: Scale → Feature Select → Imbalance → Train
   - Estimated impact: +2-5% PR-AUC

2. **Increase model complexity** ✅
   - max_depth: 6 → 10
   - n_estimators: 500 → 2000
   - learning_rate: 0.1 → 0.03
   - Estimated impact: +3-7% PR-AUC

3. **Increase early stopping patience** ✅
   - early_stopping_rounds: 30 → 100
   - Estimated impact: +1-3% PR-AUC

### Priority 2: IMPORTANT (Do Next)

4. **Test without imbalance handling**
   - See true model capability
   - Compare with/without SMOTE-ENN

5. **Add better imbalance methods**
   - BorderlineSMOTE
   - SMOTE-Tomek (less aggressive)
   - class_weight only

6. **Use full CV fold for validation**
   - Remove nested train_test_split
   - Use fold validation set directly

### Priority 3: ENHANCEMENT (Do Later)

7. **Hyperparameter tuning**
   - Grid search top 3 models
   - Find optimal depth, learning rate, etc.

8. **Add ensemble methods**
   - Stack XGBoost + LightGBM + CatBoost
   - Voting ensemble

9. **Feature engineering**
   - Polynomial features
   - Interaction terms
   - Domain-specific features

---

## 📈 EXPECTED IMPROVEMENTS

**After implementing all Priority 1 fixes:**

| Model | Current PR-AUC | Expected PR-AUC | Improvement |
|-------|---------------|-----------------|-------------|
| Gen3_XGBoost | 0.7833 | 0.84-0.87 | +5-8% |
| Gen3_LightGBM | 0.7825 | 0.84-0.87 | +5-8% |
| Gen3_CatBoost | 0.7829 | 0.84-0.87 | +5-8% |
| Gen1_KNN | 0.8012 | 0.78-0.80 | -2-0% (less benefit from fixes) |

**Why Gen3 will improve more:**
- They benefit from proper hyperparameters
- They can handle complex patterns once properly tuned
- They're currently handicapped by poor settings

**Why Gen1 will stay similar:**
- Already near optimal for simple models
- Can't benefit from deeper complexity
- Currently "cheating" with SMOTE-ENN cleaned data

---

## 🎯 IMMEDIATE ACTION PLAN

### Step 1: Create `full_comparison_v2.py` with fixes
```bash
# 1. Fix preprocessing order
# 2. Update hyperparameters
# 3. Increase early stopping
```

### Step 2: Run quick test (3-5 models only)
```bash
python full_comparison_v2.py --quick-test
```

### Step 3: Compare with current results
```bash
# Should see Gen3 models jump to 0.84+
```

### Step 4: Run full comparison
```bash
python full_comparison_v2.py
# All 90 configs with new settings
```

---

## 💡 KEY TAKEAWAYS

1. **The issue is NOT the models** - XGBoost, LightGBM, CatBoost are world-class
2. **The issue IS the configuration** - Wrong preprocessing order + undertrained models
3. **SMOTE-ENN is masking the problem** - Makes data too easy for simple models
4. **Quick wins available** - 3 simple fixes could boost Gen3 by 5-8%
5. **After fixes, expect Gen3 >> Gen1** - As it should be!

---

## 📚 References

- XGBoost best practices: max_depth=8-12, learning_rate=0.01-0.05
- LightGBM tuning: num_leaves=50-100, learning_rate=0.03
- CatBoost defaults: depth=6-10 works best
- SMOTE-ENN: Very aggressive, use for highly imbalanced data only
- Preprocessing order: Scale → Select → Resample (standard in scikit-learn)

