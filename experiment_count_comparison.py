"""Calculate the number of experiments with the new configuration"""

# Count models by type
models_needing_scaling = 5  # LogisticRegression, KNN, SVM_RBF, MLP_Sklearn, PyTorch_MLP
models_not_needing_scaling = 8  # DecisionTree, RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost, TabNet

# Configuration options
scalers_for_scaling_models = 2  # standard, robust
scalers_for_non_scaling_models = 1  # none
imbalance_methods = 3  # none, smote, smote_enn
feature_selection_methods_OLD = 2  # none, select_k_best_12
feature_selection_methods_NEW = 3  # none, select_k_best_5, select_k_best_12

print("="*80)
print("EXPERIMENT COUNT COMPARISON")
print("="*80)

# OLD configuration
old_experiments_scaling = models_needing_scaling * scalers_for_scaling_models * imbalance_methods * feature_selection_methods_OLD
old_experiments_non_scaling = models_not_needing_scaling * scalers_for_non_scaling_models * imbalance_methods * feature_selection_methods_OLD
old_total = old_experiments_scaling + old_experiments_non_scaling

print(f"\n📊 OLD Configuration (without select_k_best_5):")
print(f"   Models needing scaling:     {models_needing_scaling} × {scalers_for_scaling_models} × {imbalance_methods} × {feature_selection_methods_OLD} = {old_experiments_scaling}")
print(f"   Models not needing scaling: {models_not_needing_scaling} × {scalers_for_non_scaling_models} × {imbalance_methods} × {feature_selection_methods_OLD} = {old_experiments_non_scaling}")
print(f"   TOTAL: {old_total} experiments")

# NEW configuration
new_experiments_scaling = models_needing_scaling * scalers_for_scaling_models * imbalance_methods * feature_selection_methods_NEW
new_experiments_non_scaling = models_not_needing_scaling * scalers_for_non_scaling_models * imbalance_methods * feature_selection_methods_NEW
new_total = new_experiments_scaling + new_experiments_non_scaling

print(f"\n📊 NEW Configuration (with select_k_best_5):")
print(f"   Models needing scaling:     {models_needing_scaling} × {scalers_for_scaling_models} × {imbalance_methods} × {feature_selection_methods_NEW} = {new_experiments_scaling}")
print(f"   Models not needing scaling: {models_not_needing_scaling} × {scalers_for_non_scaling_models} × {imbalance_methods} × {feature_selection_methods_NEW} = {new_experiments_non_scaling}")
print(f"   TOTAL: {new_total} experiments")

print(f"\n📈 DIFFERENCE:")
print(f"   Additional experiments: {new_total - old_total}")
print(f"   Increase: {((new_total - old_total) / old_total * 100):.1f}%")

# Estimated time
avg_time_per_exp = 45  # seconds (average)
old_time_minutes = old_total * avg_time_per_exp / 60
new_time_minutes = new_total * avg_time_per_exp / 60

print(f"\n⏱️  ESTIMATED TIME (avg {avg_time_per_exp}s per experiment):")
print(f"   OLD: {old_time_minutes:.1f} minutes ({old_time_minutes/60:.1f} hours)")
print(f"   NEW: {new_time_minutes:.1f} minutes ({new_time_minutes/60:.1f} hours)")
print(f"   Additional time: {new_time_minutes - old_time_minutes:.1f} minutes")

print(f"\n✅ BENEFITS of adding select_k_best_5:")
print(f"   - Compare 5-feature vs 12-feature models")
print(f"   - Find optimal feature subset size")
print(f"   - Potentially better performance with fewer features")
print(f"   - More comprehensive comparison")

print(f"\n📝 Feature Selection Configurations:")
print(f"   none            : Use all features")
print(f"   select_k_best_5 : Select top 5 features (NEW!)")
print(f"   select_k_best_12: Select top 12 features")
