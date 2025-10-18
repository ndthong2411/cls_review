"""Calculate experiments with Mutual Information added"""

# Model counts
models_needing_scaling = 5  # LogisticRegression, KNN, SVM_RBF, MLP_Sklearn, PyTorch_MLP
models_not_needing_scaling = 8  # DecisionTree, RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost, TabNet

# Configuration options
scalers_for_scaling = 2  # standard, robust
scalers_for_non_scaling = 1  # none
imbalance_methods = 3  # none, smote, smote_enn

# Feature selection progression
feat_sel_OLD = 2  # none, select_k_best_12
feat_sel_WITH_K5 = 3  # none, select_k_best_5, select_k_best_12
feat_sel_WITH_MUTUAL_INFO = 5  # none, select_k_best_5, select_k_best_12, mutual_info_5, mutual_info_12

print("="*80)
print("PROGRESSION: FEATURE SELECTION METHODS")
print("="*80)

# Original (baseline)
old_scaling = models_needing_scaling * scalers_for_scaling * imbalance_methods * feat_sel_OLD
old_non_scaling = models_not_needing_scaling * scalers_for_non_scaling * imbalance_methods * feat_sel_OLD
old_total = old_scaling + old_non_scaling

print(f"\n📊 BASELINE (none + select_k_best_12):")
print(f"   Feature selection: 2 methods")
print(f"   Scaling models:     {models_needing_scaling} × {scalers_for_scaling} × {imbalance_methods} × {feat_sel_OLD} = {old_scaling}")
print(f"   Non-scaling models: {models_not_needing_scaling} × {scalers_for_non_scaling} × {imbalance_methods} × {feat_sel_OLD} = {old_non_scaling}")
print(f"   TOTAL: {old_total} experiments")

# With k=5 added
k5_scaling = models_needing_scaling * scalers_for_scaling * imbalance_methods * feat_sel_WITH_K5
k5_non_scaling = models_not_needing_scaling * scalers_for_non_scaling * imbalance_methods * feat_sel_WITH_K5
k5_total = k5_scaling + k5_non_scaling

print(f"\n📊 WITH K=5 ADDED (none + k=5 + k=12):")
print(f"   Feature selection: 3 methods")
print(f"   Scaling models:     {models_needing_scaling} × {scalers_for_scaling} × {imbalance_methods} × {feat_sel_WITH_K5} = {k5_scaling}")
print(f"   Non-scaling models: {models_not_needing_scaling} × {scalers_for_non_scaling} × {imbalance_methods} × {feat_sel_WITH_K5} = {k5_non_scaling}")
print(f"   TOTAL: {k5_total} experiments (+{k5_total - old_total})")

# With mutual_info added
mi_scaling = models_needing_scaling * scalers_for_scaling * imbalance_methods * feat_sel_WITH_MUTUAL_INFO
mi_non_scaling = models_not_needing_scaling * scalers_for_non_scaling * imbalance_methods * feat_sel_WITH_MUTUAL_INFO
mi_total = mi_scaling + mi_non_scaling

print(f"\n📊 WITH MUTUAL INFO (none + k=5 + k=12 + mi=5 + mi=12): ⭐ NEW")
print(f"   Feature selection: 5 methods")
print(f"   Scaling models:     {models_needing_scaling} × {scalers_for_scaling} × {imbalance_methods} × {feat_sel_WITH_MUTUAL_INFO} = {mi_scaling}")
print(f"   Non-scaling models: {models_not_needing_scaling} × {scalers_for_non_scaling} × {imbalance_methods} × {feat_sel_WITH_MUTUAL_INFO} = {mi_non_scaling}")
print(f"   TOTAL: {mi_total} experiments (+{mi_total - k5_total} from k=5, +{mi_total - old_total} from baseline)")

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"   Baseline → K=5:       {old_total} → {k5_total} (+{k5_total - old_total}, +{(k5_total - old_total) / old_total * 100:.0f}%)")
print(f"   K=5 → Mutual Info:    {k5_total} → {mi_total} (+{mi_total - k5_total}, +{(mi_total - k5_total) / k5_total * 100:.0f}%)")
print(f"   Baseline → Final:     {old_total} → {mi_total} (+{mi_total - old_total}, +{(mi_total - old_total) / old_total * 100:.0f}%)")

# Time estimates
avg_time = 45  # seconds per experiment
baseline_time = old_total * avg_time / 60
k5_time = k5_total * avg_time / 60
mi_time = mi_total * avg_time / 60

print(f"\n{'='*80}")
print(f"TIME ESTIMATES (avg {avg_time}s per experiment)")
print(f"{'='*80}")
print(f"   Baseline:      {baseline_time:.1f} minutes ({baseline_time/60:.1f} hours)")
print(f"   With K=5:      {k5_time:.1f} minutes ({k5_time/60:.1f} hours) [+{k5_time - baseline_time:.1f} min]")
print(f"   With Mutual:   {mi_time:.1f} minutes ({mi_time/60:.1f} hours) [+{mi_time - k5_time:.1f} min]")

print(f"\n{'='*80}")
print(f"FEATURE SELECTION METHODS")
print(f"{'='*80}")
print(f"   1. none              : All features (no selection)")
print(f"   2. select_k_best_5   : Top 5 features (ANOVA F-test)")
print(f"   3. select_k_best_12  : Top 12 features (ANOVA F-test)")
print(f"   4. mutual_info_5     : Top 5 features (Mutual Information) ⭐ NEW")
print(f"   5. mutual_info_12    : Top 12 features (Mutual Information) ⭐ NEW")

print(f"\n{'='*80}")
print(f"KEY DIFFERENCES: ANOVA vs MUTUAL INFORMATION")
print(f"{'='*80}")
print(f"   ANOVA F-test:")
print(f"     ✓ Detects LINEAR relationships only")
print(f"     ✓ Very FAST")
print(f"     ✓ Assumes normal distribution")
print(f"     ✓ Univariate (independent features)")
print(f"")
print(f"   Mutual Information:")
print(f"     ✓ Detects LINEAR + NON-LINEAR relationships")
print(f"     ✓ SLOWER (but more powerful)")
print(f"     ✓ No distribution assumptions")
print(f"     ✓ Captures complex dependencies")

print(f"\n{'='*80}")
print(f"WHY ADD MUTUAL INFORMATION?")
print(f"{'='*80}")
print(f"   1. Credit Card Fraud: May have non-linear patterns")
print(f"   2. Cardiovascular: Complex interactions (BMI × age × BP)")
print(f"   3. Compare: Which method selects better features?")
print(f"   4. Robustness: Test if results are consistent")
print()
