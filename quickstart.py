"""
Quick Start Training Script

A simplified version to get started quickly without full Hydra setup.
Train baseline models and save results for the Streamlit demo.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import time
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CVD PREDICTION - QUICK START TRAINING")
print("="*70)

# Load data
print("\n[1/6] Loading data...")
data_path = Path("data/raw/cardio_train.csv")

if not data_path.exists():
    print(f"❌ Error: Dataset not found at {data_path}")
    print("   Please download 'cardio_train.csv' from Kaggle:")
    print("   https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset")
    print(f"   And place it in: {data_path.parent.absolute()}")
    exit(1)

df = pd.read_csv(data_path, sep=';')
df.columns = df.columns.str.strip()

# Feature engineering
print("[2/6] Engineering features...")
df['age_years'] = df['age'] / 365.25
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3

# Select features
feature_cols = [
    'age_years', 'gender', 'height', 'weight', 'bmi',
    'ap_hi', 'ap_lo', 'pulse_pressure', 'map',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active'
]

X = df[feature_cols].values
y = df['cardio'].values

print(f"   Dataset shape: {X.shape}")
print(f"   Positive class: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

# Train/test split
print("[3/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
print("[4/6] Setting up models...")
models = {
    # Generation 1: Baseline
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'generation': 1,
        'needs_scaling': True
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=42, class_weight='balanced'),
        'generation': 1,
        'needs_scaling': False
    },
    
    # Generation 2: Intermediate
    'Random Forest': {
        'model': RandomForestClassifier(
            n_estimators=300,  # Increased from 100
            max_depth=15, 
            random_state=42, 
            class_weight='balanced', 
            n_jobs=-1
        ),
        'generation': 2,
        'needs_scaling': False
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(
            n_estimators=300,  # Increased from 200
            max_depth=5, 
            learning_rate=0.1,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=30  # Increased patience from 20
        ),
        'generation': 2,
        'needs_scaling': False
    },
}

# Try to import advanced models
try:
    import xgboost as xgb
    models['XGBoost'] = {
        'model': xgb.XGBClassifier(
            n_estimators=500,  # Increased for more capacity
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
            early_stopping_rounds=30,  # Increased patience
            eval_metric='logloss',
            tree_method='gpu_hist',  # 🚀 GPU ACCELERATION!
            gpu_id=0,
            predictor='gpu_predictor',
            verbosity=0
        ),
        'generation': 3,
        'needs_scaling': False,
        'use_eval_set': True
    }
    print("   ✓ XGBoost available with GPU support")
except ImportError:
    print("   ⚠ XGBoost not available (install with: pip install xgboost)")

try:
    import lightgbm as lgb
    models['LightGBM'] = {
        'model': lgb.LGBMClassifier(
            n_estimators=500,  # Increased for more capacity
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            is_unbalance=True,
            verbose=-1,
            early_stopping_rounds=30,  # Increased patience
            device='gpu',  # 🚀 GPU ACCELERATION!
            gpu_platform_id=0,
            gpu_device_id=0
        ),
        'generation': 3,
        'needs_scaling': False,
        'use_eval_set': True
    }
    print("   ✓ LightGBM available with GPU support")
except ImportError:
    print("   ⚠ LightGBM not available (install with: pip install lightgbm)")

try:
    import catboost as cb
    models['CatBoost'] = {
        'model': cb.CatBoostClassifier(
            iterations=500,  # Increased for more capacity
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False,
            auto_class_weights='Balanced',
            early_stopping_rounds=30,  # Increased patience
            od_type='Iter',
            task_type='GPU',  # 🚀 GPU ACCELERATION!
            devices='0'
        ),
        'generation': 3,
        'needs_scaling': False,
        'use_eval_set': True
    }
    print("   ✓ CatBoost available with GPU support")
except ImportError:
    print("   ⚠ CatBoost not available (install with: pip install catboost)")

# Training with cross-validation
print(f"\n[5/6] Training {len(models)} models with 5-fold CV...")
results = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model_info in models.items():
    print(f"\n  Training: {model_name} (Generation {model_info['generation']})")
    
    fold_results = []
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Apply SMOTE for imbalance
        smote = SMOTE(random_state=42)
        X_fold_train_res, y_fold_train_res = smote.fit_resample(X_fold_train, y_fold_train)
        
        # Scaling if needed
        if model_info['needs_scaling']:
            scaler = StandardScaler()
            X_fold_train_res = scaler.fit_transform(X_fold_train_res)
            X_fold_val = scaler.transform(X_fold_val)
        
        # Train with early stopping support
        model = model_info['model']
        use_eval_set = model_info.get('use_eval_set', False)
        
        if use_eval_set:
            # Split training data for early stopping validation
            from sklearn.model_selection import train_test_split as split
            X_tr, X_val_es, y_tr, y_val_es = split(
                X_fold_train_res, y_fold_train_res,
                test_size=0.1,
                random_state=42,
                stratify=y_fold_train_res
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val_es, y_val_es)], verbose=False)
        else:
            model.fit(X_fold_train_res, y_fold_train_res)
        
        # Predict
        y_pred = model.predict(X_fold_val)
        y_proba = model.predict_proba(X_fold_val)[:, 1]
        
        # Calculate confusion matrix for specificity
        tn, fp, fn, tp = confusion_matrix(y_fold_val, y_pred).ravel()
        
        # Metrics
        precision, recall, _ = precision_recall_curve(y_fold_val, y_proba)
        pr_auc = auc(recall, precision)
        
        fold_metrics = {
            'accuracy': accuracy_score(y_fold_val, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_fold_val, y_pred),
            'sensitivity': recall_score(y_fold_val, y_pred),  # Same as recall, but clearer for medical context
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # True Negative Rate
            'precision': precision_score(y_fold_val, y_pred),
            'f1': f1_score(y_fold_val, y_pred),
            'roc_auc': roc_auc_score(y_fold_val, y_proba),
            'pr_auc': pr_auc,
            'mcc': matthews_corrcoef(y_fold_val, y_pred),  # Matthews Correlation Coefficient
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0,  # Negative Predictive Value
        }
        fold_results.append(fold_metrics)
        
        print(f"    Fold {fold}: PR-AUC={pr_auc:.4f}, Sensitivity={fold_metrics['sensitivity']:.4f}, "
              f"Specificity={fold_metrics['specificity']:.4f}, F1={fold_metrics['f1']:.4f}")
    
    train_time = time.time() - start_time
    
    # Average metrics
    avg_metrics = {
        'model_name': model_name,
        'generation': model_info['generation'],
        'accuracy': np.mean([f['accuracy'] for f in fold_results]),
        'balanced_accuracy': np.mean([f['balanced_accuracy'] for f in fold_results]),
        'sensitivity': np.mean([f['sensitivity'] for f in fold_results]),
        'specificity': np.mean([f['specificity'] for f in fold_results]),
        'precision': np.mean([f['precision'] for f in fold_results]),
        'f1': np.mean([f['f1'] for f in fold_results]),
        'roc_auc': np.mean([f['roc_auc'] for f in fold_results]),
        'pr_auc': np.mean([f['pr_auc'] for f in fold_results]),
        'mcc': np.mean([f['mcc'] for f in fold_results]),
        'npv': np.mean([f['npv'] for f in fold_results]),
        'train_time_sec': train_time,
        'cv_std_pr_auc': np.std([f['pr_auc'] for f in fold_results]),
        'cv_std_sensitivity': np.std([f['sensitivity'] for f in fold_results]),
        'cv_std_f1': np.std([f['f1'] for f in fold_results])
    }
    
    results.append(avg_metrics)
    
    print(f"    ✓ Avg Metrics - PR-AUC: {avg_metrics['pr_auc']:.4f}, Sensitivity: {avg_metrics['sensitivity']:.4f}, "
          f"Specificity: {avg_metrics['specificity']:.4f}, F1: {avg_metrics['f1']:.4f}")
    print(f"    Training time: {train_time:.1f}s")

# Save results
print("\n[6/6] Saving results...")
results_df = pd.DataFrame(results)

# Create output directories
output_dir = Path("experiments")
output_dir.mkdir(exist_ok=True)

results_path = output_dir / "results_summary.csv"
results_df.to_csv(results_path, index=False)
print(f"   ✓ Saved results to: {results_path}")

# Summary
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\n📊 Top 3 Models by PR-AUC:")
top3 = results_df.nlargest(3, 'pr_auc')
for idx, row in top3.iterrows():
    print(f"\n  {row['model_name']} (Gen {row['generation']})")
    print(f"    PR-AUC:      {row['pr_auc']:.4f} ± {row['cv_std_pr_auc']:.4f}")
    print(f"    ROC-AUC:     {row['roc_auc']:.4f}")
    print(f"    Sensitivity: {row['sensitivity']:.4f} ± {row['cv_std_sensitivity']:.4f}")
    print(f"    Specificity: {row['specificity']:.4f}")
    print(f"    F1-Score:    {row['f1']:.4f} ± {row['cv_std_f1']:.4f}")
    print(f"    Precision:   {row['precision']:.4f}")
    print(f"    MCC:         {row['mcc']:.4f}")

print("\n📋 All Models Summary:")
print(results_df[['model_name', 'pr_auc', 'sensitivity', 'specificity', 'f1']].to_string(index=False))

print("\n" + "="*70)
print("Next steps:")
print("  1. View results: streamlit run app.py")
print("  2. Run full pipeline: python -m src.experiment.run_phase --phase=advanced")
print("="*70)
