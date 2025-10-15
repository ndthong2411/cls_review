"""
Save Best Model Script

After running quickstart.py or full_comparison.py, this script:
1. Identifies the best model from results
2. Retrains it on full training data
3. Saves the model to disk with metadata
4. Creates a prediction function for deployment
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

print("="*80)
print("SAVE BEST MODEL SCRIPT")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_FILE = "experiments/results_summary.csv"  # From quickstart.py
OUTPUT_DIR = Path("experiments/best_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD RESULTS
# ============================================================================

print("\n[1/5] Loading results...")
if not Path(RESULTS_FILE).exists():
    print(f"❌ Results file not found: {RESULTS_FILE}")
    print("   Please run 'python quickstart.py' first!")
    exit(1)

results_df = pd.read_csv(RESULTS_FILE)
print(f"   Found {len(results_df)} models")

# Find best model by PR-AUC
best_idx = results_df['pr_auc'].idxmax()
best_model_info = results_df.iloc[best_idx]

print(f"\n   🏆 Best Model: {best_model_info['model_name']}")
print(f"   Generation: {best_model_info['generation']}")
print(f"   PR-AUC: {best_model_info['pr_auc']:.4f}")
print(f"   Sensitivity: {best_model_info['sensitivity']:.4f}")
print(f"   Specificity: {best_model_info['specificity']:.4f}")
print(f"   F1-Score: {best_model_info['f1']:.4f}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[2/5] Loading data...")
data_path = Path("data/raw/cardio_train.csv")

if not data_path.exists():
    print(f"❌ Dataset not found: {data_path}")
    exit(1)

df = pd.read_csv(data_path, sep=';')
df.columns = df.columns.str.strip()

# Feature engineering (same as quickstart.py)
df['age_years'] = df['age'] / 365.25
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3

feature_cols = [
    'age_years', 'gender', 'height', 'weight', 'bmi',
    'ap_hi', 'ap_lo', 'pulse_pressure', 'map',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active'
]

X = df[feature_cols].values
y = df['cardio'].values

# Train/test split (same as quickstart.py)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# ============================================================================
# RECREATE AND TRAIN BEST MODEL
# ============================================================================

print(f"\n[3/5] Training {best_model_info['model_name']} on full training data...")

model_name = best_model_info['model_name']

# Import and create model based on name
if model_name == 'Logistic Regression':
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    needs_scaling = True
    
elif model_name == 'Decision Tree':
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=42, class_weight='balanced')
    needs_scaling = False
    
elif model_name == 'Random Forest':
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1)
    needs_scaling = False
    
elif model_name == 'Gradient Boosting':
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42, validation_fraction=0.1, n_iter_no_change=30)
    needs_scaling = False
    
elif model_name == 'XGBoost':
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor',
        early_stopping_rounds=30, eval_metric='logloss', verbosity=0
    )
    needs_scaling = False
    use_eval_set = True
    
elif model_name == 'LightGBM':
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1,
        is_unbalance=True, device='gpu', early_stopping_rounds=30, verbose=-1
    )
    needs_scaling = False
    use_eval_set = True
    
elif model_name == 'CatBoost':
    import catboost as cb
    model = cb.CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.1, random_state=42,
        auto_class_weights='Balanced', task_type='GPU', devices='0',
        early_stopping_rounds=30, od_type='Iter', verbose=False
    )
    needs_scaling = False
    use_eval_set = True
    
else:
    print(f"❌ Unknown model: {model_name}")
    exit(1)

# Apply SMOTE
print("   Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Scaling if needed
scaler = None
if needs_scaling:
    print("   Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
else:
    X_test_scaled = X_test

# Train with early stopping if supported
use_eval_set = 'use_eval_set' in locals() and use_eval_set

if use_eval_set:
    print("   Training with early stopping...")
    from sklearn.model_selection import train_test_split as split
    X_tr, X_val, y_tr, y_val = split(X_train_res, y_train_res, test_size=0.1, random_state=42, stratify=y_train_res)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    # Get best iteration info
    if hasattr(model, 'best_iteration_'):
        print(f"   ✓ Best iteration: {model.best_iteration_}")
    elif hasattr(model, 'best_iteration'):
        print(f"   ✓ Best iteration: {model.best_iteration}")
else:
    print("   Training...")
    model.fit(X_train_res, y_train_res)

print("   ✓ Training complete!")

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================

print("\n[4/5] Evaluating on test set...")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef
)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall_curve, precision_curve)

test_metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
    'sensitivity': recall_score(y_test, y_pred),
    'specificity': tn / (tn + fp),
    'precision': precision_score(y_test, y_pred),
    'npv': tn / (tn + fn),
    'f1': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_proba),
    'pr_auc': pr_auc,
    'mcc': matthews_corrcoef(y_test, y_pred)
}

print(f"   Test PR-AUC:      {test_metrics['pr_auc']:.4f}")
print(f"   Test Sensitivity: {test_metrics['sensitivity']:.4f}")
print(f"   Test Specificity: {test_metrics['specificity']:.4f}")
print(f"   Test F1-Score:    {test_metrics['f1']:.4f}")
print(f"   Test ROC-AUC:     {test_metrics['roc_auc']:.4f}")

# ============================================================================
# SAVE MODEL AND METADATA
# ============================================================================

print(f"\n[5/5] Saving model and metadata...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f"best_model_{model_name.replace(' ', '_').lower()}_{timestamp}.pkl"
metadata_filename = f"best_model_{model_name.replace(' ', '_').lower()}_{timestamp}_metadata.json"
scaler_filename = f"scaler_{model_name.replace(' ', '_').lower()}_{timestamp}.pkl" if scaler else None

# Save model
model_path = OUTPUT_DIR / model_filename
joblib.dump(model, model_path)
print(f"   ✓ Model saved to: {model_path}")

# Save scaler if exists
if scaler:
    scaler_path = OUTPUT_DIR / scaler_filename
    joblib.dump(scaler, scaler_path)
    print(f"   ✓ Scaler saved to: {scaler_path}")

# Save metadata
metadata = {
    'model_name': model_name,
    'generation': int(best_model_info['generation']),
    'timestamp': timestamp,
    'training_info': {
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'features': feature_cols,
        'n_features': len(feature_cols),
        'preprocessing': {
            'smote': True,
            'scaling': needs_scaling,
            'scaler_type': 'StandardScaler' if needs_scaling else None
        }
    },
    'cv_metrics': {
        'pr_auc': float(best_model_info['pr_auc']),
        'sensitivity': float(best_model_info['sensitivity']),
        'specificity': float(best_model_info['specificity']),
        'f1': float(best_model_info['f1']),
        'roc_auc': float(best_model_info['roc_auc'])
    },
    'test_metrics': {k: float(v) for k, v in test_metrics.items()},
    'model_params': model.get_params(),
    'files': {
        'model': model_filename,
        'scaler': scaler_filename,
        'metadata': metadata_filename
    }
}

metadata_path = OUTPUT_DIR / metadata_filename
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Metadata saved to: {metadata_path}")

# ============================================================================
# CREATE PREDICTION FUNCTION
# ============================================================================

# Create a simple prediction script
prediction_script = f"""
# Prediction Script for Best Model
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import joblib
import numpy as np

# Load model
model = joblib.load('{model_path}')
{f"scaler = joblib.load('{OUTPUT_DIR / scaler_filename}')" if scaler else "# No scaler needed"}

def predict_cvd_risk(age_years, gender, height, weight, ap_hi, ap_lo, 
                     cholesterol, gluc, smoke, alco, active):
    \"\"\"
    Predict cardiovascular disease risk
    
    Parameters:
    -----------
    age_years : float - Age in years
    gender : int - 1=Female, 2=Male
    height : float - Height in cm
    weight : float - Weight in kg
    ap_hi : int - Systolic blood pressure
    ap_lo : int - Diastolic blood pressure
    cholesterol : int - 1=Normal, 2=Above normal, 3=Well above normal
    gluc : int - 1=Normal, 2=Above normal, 3=Well above normal
    smoke : int - 0=No, 1=Yes
    alco : int - 0=No, 1=Yes
    active : int - 0=No, 1=Yes
    
    Returns:
    --------
    risk_probability : float - Probability of CVD (0-1)
    risk_class : str - "Low", "Medium", or "High"
    \"\"\"
    # Calculate derived features
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo
    map_value = (ap_hi + 2 * ap_lo) / 3
    
    # Create feature vector
    features = np.array([[
        age_years, gender, height, weight, bmi,
        ap_hi, ap_lo, pulse_pressure, map_value,
        cholesterol, gluc, smoke, alco, active
    ]])
    
    {f"# Apply scaling\\n    features = scaler.transform(features)" if scaler else "# No scaling needed"}
    
    # Predict
    probability = model.predict_proba(features)[0, 1]
    
    # Classify risk
    if probability < 0.3:
        risk_class = "Low"
    elif probability < 0.7:
        risk_class = "Medium"
    else:
        risk_class = "High"
    
    return probability, risk_class

# Example usage
if __name__ == "__main__":
    prob, risk = predict_cvd_risk(
        age_years=55,
        gender=1,  # Female
        height=165,
        weight=70,
        ap_hi=140,
        ap_lo=90,
        cholesterol=2,
        gluc=1,
        smoke=0,
        alco=0,
        active=1
    )
    
    print(f"CVD Risk Probability: {{prob:.2%}}")
    print(f"Risk Category: {{risk}}")
"""

prediction_script_path = OUTPUT_DIR / f"predict_{model_name.replace(' ', '_').lower()}.py"
with open(prediction_script_path, 'w') as f:
    f.write(prediction_script)
print(f"   ✓ Prediction script saved to: {prediction_script_path}")

print("\n" + "="*80)
print("✅ BEST MODEL SAVED SUCCESSFULLY!")
print("="*80)
print(f"\nModel: {model_name}")
print(f"Location: {OUTPUT_DIR}")
print(f"\nFiles created:")
print(f"  1. {model_filename} - Trained model")
if scaler_filename:
    print(f"  2. {scaler_filename} - Scaler")
print(f"  3. {metadata_filename} - Metadata")
print(f"  4. predict_{model_name.replace(' ', '_').lower()}.py - Prediction script")

print(f"\nTo use the model:")
print(f"  python {prediction_script_path}")

print("\n" + "="*80)
