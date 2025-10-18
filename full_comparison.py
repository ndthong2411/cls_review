"""
Comprehensive Model & Preprocessing Comparison Script

Implements all 4 generations of models with multiple preprocessing strategies:
- Generation 1 (Baseline): LR, DT, KNN
- Generation 2 (Intermediate): RF, ExtraTrees, GB, SVM
- Generation 3 (Advanced): XGBoost, LightGBM, CatBoost
- Generation 4 (Deep Learning): MLP-PyTorch, TabNet

Preprocessing strategies:
- Scaling: Standard, MinMax, Robust
- Encoding: OneHot, Ordinal
- Imbalance: None, SMOTE, ADASYN, SMOTE-ENN
- Feature Selection: None, SelectKBest, RFE

Evaluation metrics (Medical Focus):
- PR-AUC (Primary for imbalanced data)
- Sensitivity (Recall) - Critical for medical screening
- Specificity - True Negative Rate
- F1-Score - Harmonic mean
- ROC-AUC
- Matthews Correlation Coefficient (MCC)
- NPV (Negative Predictive Value)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import warnings
from datetime import datetime
from itertools import product
from tqdm import tqdm

# Sklearn imports
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef
)

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_path': 'data/raw/cardio_train.csv',
    'output_dir': 'experiments/full_comparison',
    'cv_folds': 1,  # Set to 1 for single train/val split (fast mode for all models)
    'random_state': 42,
    'test_size': 0.2,
    'n_jobs': -1,
    'use_cache': True,  # Enable model caching to avoid retraining
    'cache_dir': 'experiments/model_cache',  # Directory for cached models
}

# ============================================================================
# PREPROCESSING STRATEGIES
# ============================================================================

SCALING_METHODS = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
    'robust': RobustScaler(),
}

IMBALANCE_METHODS = {
    'none': None,
    'smote': SMOTE(random_state=42),
    'adasyn': ADASYN(random_state=42),
    'smote_enn': SMOTEENN(random_state=42),
}

FEATURE_SELECTION_METHODS = {
    'none': None,
    'select_k_best_5': SelectKBest(f_classif, k=5),
    'select_k_best_12': SelectKBest(f_classif, k=12),
    'mutual_info_5': SelectKBest(mutual_info_classif, k=5),
    'mutual_info_12': SelectKBest(mutual_info_classif, k=12),
}

# ============================================================================
# MODEL DEFINITIONS BY GENERATION
# ============================================================================

def get_models():
    """Returns all models organized by generation"""
    
    models = {}
    
    # ========== GENERATION 1: BASELINE ==========
    models['Gen1_LogisticRegression'] = {
        'model': LogisticRegression(
            max_iter=1000, 
            random_state=42, 
            class_weight='balanced',
            solver='lbfgs'
        ),
        'generation': 1,
        'needs_scaling': True,
        'description': 'Simple linear classifier with regularization'
    }
    
    models['Gen1_DecisionTree'] = {
        'model': DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=20,
            min_samples_split=40,
            random_state=42,
            class_weight='balanced'
        ),
        'generation': 1,
        'needs_scaling': False,
        'description': 'Tree-based classifier with pruning'
    }
    
    models['Gen1_KNN'] = {
        'model': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='minkowski',
            n_jobs=-1
        ),
        'generation': 1,
        'needs_scaling': True,
        'description': 'Distance-based classifier'
    }
    
    # ========== GENERATION 2: INTERMEDIATE ==========
    models['Gen2_RandomForest'] = {
        'model': RandomForestClassifier(
            n_estimators=300,  # Increased from 100
            max_depth=15,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'generation': 2,
        'needs_scaling': False,
        'description': 'Ensemble of decision trees with bagging'
    }
    
    models['Gen2_ExtraTrees'] = {
        'model': ExtraTreesClassifier(
            n_estimators=300,  # Increased from 100
            max_depth=15,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'generation': 2,
        'needs_scaling': False,
        'description': 'Extra randomized trees for variance reduction'
    }
    
    models['Gen2_GradientBoosting'] = {
        'model': GradientBoostingClassifier(
            n_estimators=200,  # Increased from 100
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8,
            validation_fraction=0.1,  # For early stopping monitoring
            n_iter_no_change=20  # Early stopping
        ),
        'generation': 2,
        'needs_scaling': False,
        'description': 'Sequential boosting with gradient descent and early stopping'
    }
    
    models['Gen2_SVM_RBF'] = {
        'model': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42,
            class_weight='balanced',
            probability=True
        ),
        'generation': 2,
        'needs_scaling': True,
        'description': 'Support Vector Machine with RBF kernel'
    }
    
    models['Gen2_MLP_Sklearn'] = {
        'model': MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=500,  # Increased from 200
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,  # Early stopping patience
            verbose=False
        ),
        'generation': 2,
        'needs_scaling': True,
        'description': 'Multi-layer perceptron neural network with early stopping'
    }
    
    # ========== GENERATION 3: ADVANCED ==========
    # Try to import advanced models
    try:
        import xgboost as xgb
        models['Gen3_XGBoost'] = {
            'model': xgb.XGBClassifier(
                n_estimators=2000,  # Increased from 500 for better convergence
                max_depth=10,  # Increased from 6 for more complexity
                learning_rate=0.03,  # Decreased from 0.1 for stable learning
                min_child_weight=3,  # Added to prevent overfitting
                subsample=0.9,  # Increased from 0.8 for more data per tree
                colsample_bytree=0.9,  # Increased from 0.8
                gamma=0,  # Minimum loss reduction for split
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                early_stopping_rounds=100,  # Increased from 30 for more patience
                tree_method='gpu_hist',  # 🚀 GPU ACCELERATION!
                gpu_id=0,
                predictor='gpu_predictor',
                verbosity=0
            ),
            'generation': 3,
            'needs_scaling': False,
            'description': 'eXtreme Gradient Boosting with GPU acceleration and optimized hyperparameters',
            'use_eval_set': True
        }
        print("   ✓ XGBoost available with GPU support")
    except ImportError:
        print("   ⚠️  XGBoost not available")
    
    try:
        import lightgbm as lgb
        models['Gen3_LightGBM'] = {
            'model': lgb.LGBMClassifier(
                n_estimators=2000,  # Increased from 500 for better convergence
                max_depth=10,  # Increased from 6 for more complexity
                num_leaves=100,  # Added for more complexity (2^max_depth rule)
                learning_rate=0.03,  # Decreased from 0.1 for stable learning
                min_child_samples=30,  # Added to prevent overfitting
                subsample=0.9,  # Increased from 0.8
                colsample_bytree=0.9,  # Increased from 0.8
                subsample_freq=1,  # Enable bagging
                random_state=42,
                n_jobs=-1,
                is_unbalance=True,
                verbose=-1,
                # Note: early_stopping_rounds is now passed via callbacks in fit()
                device='gpu',  # 🚀 GPU ACCELERATION!
                gpu_platform_id=0,
                gpu_device_id=0
            ),
            'generation': 3,
            'needs_scaling': False,
            'description': 'Light Gradient Boosting Machine with GPU acceleration and optimized hyperparameters',
            'use_eval_set': True
        }
        print("   ✓ LightGBM available with GPU support")
    except ImportError:
        print("   ⚠️  LightGBM not available")
    
    try:
        import catboost as cb
        models['Gen3_CatBoost'] = {
            'model': cb.CatBoostClassifier(
                iterations=2000,  # Increased from 500 for better convergence
                depth=10,  # Increased from 6 for more complexity
                learning_rate=0.03,  # Decreased from 0.1 for stable learning
                l2_leaf_reg=3,  # L2 regularization
                border_count=254,  # Increased for better splits (more precise boundaries)
                random_strength=1,  # Randomness for scoring splits
                bagging_temperature=1,  # Bayesian bootstrap intensity
                random_state=42,
                verbose=False,
                auto_class_weights='Balanced',
                early_stopping_rounds=100,  # Increased from 30 for more patience
                od_type='Iter',
                task_type='GPU',  # 🚀 GPU ACCELERATION!
                devices='0'
            ),
            'generation': 3,
            'needs_scaling': False,
            'description': 'Categorical Boosting with GPU acceleration and optimized hyperparameters',
            'use_eval_set': True
        }
        print("   ✓ CatBoost available with GPU support")
    except ImportError:
        print("   ⚠️  CatBoost not available")

    # ========== GENERATION 4: DEEP LEARNING (SOTA) ==========
    try:
        from src.models.pytorch_mlp import PyTorchMLPClassifier
        models['Gen4_PyTorch_MLP'] = {
            'model': PyTorchMLPClassifier(
                hidden_dims=[256, 128, 64, 32],  # Deep architecture
                dropout_rates=[0.4, 0.3, 0.2, 0.1],  # Progressive dropout
                use_batch_norm=True,
                learning_rate=0.001,
                batch_size=128,
                epochs=200,  # More epochs with early stopping
                optimizer_name='adamw',  # AdamW for better regularization
                weight_decay=1e-4,
                scheduler='plateau',  # Reduce LR on plateau
                early_stopping_patience=30,
                class_weight='balanced',
                device=None,  # Auto-detect GPU
                random_state=42,
                verbose=False
            ),
            'generation': 4,
            'needs_scaling': True,  # Neural networks benefit from scaling
            'description': 'Deep MLP with batch norm, dropout, and adaptive learning rate',
            'use_eval_set': True
        }
        print("   ✓ PyTorch MLP available")
    except ImportError as e:
        print(f"   ⚠️  PyTorch MLP not available: {e}")

    try:
        from src.models.tabnet_model import TabNetClassifier
        models['Gen4_TabNet'] = {
            'model': TabNetClassifier(
                n_d=64,  # Width of decision prediction layer
                n_a=64,  # Width of attention embedding
                n_steps=5,  # Number of steps in architecture
                gamma=1.5,  # Feature reusage coefficient
                n_independent=2,
                n_shared=2,
                lambda_sparse=1e-4,  # Sparsity regularization
                momentum=0.3,
                clip_value=2.0,
                optimizer_params={'lr': 2e-2},
                mask_type='sparsemax',
                seed=42,
                verbose=0,
                device_name='auto',  # Auto-detect GPU
                max_epochs=200,
                patience=30,
                batch_size=256,
                virtual_batch_size=128
            ),
            'generation': 4,
            'needs_scaling': False,  # TabNet handles raw features
            'description': 'Attention-based interpretable tabular learning with feature selection',
            'use_eval_set': True
        }
        print("   ✓ TabNet available")
    except ImportError as e:
        print(f"   ⚠️  TabNet not available: {e}")

    return models

# ============================================================================
# DATA LOADING & FEATURE ENGINEERING
# ============================================================================

def load_and_engineer_data(data_path):
    """Load data and engineer features - supports both cardio and creditcard datasets"""
    print(f"\n{'='*80}")
    print("LOADING & FEATURE ENGINEERING")
    print(f"{'='*80}")
    print(f"Dataset: {data_path}")
    
    # Try to detect the delimiter by reading first line
    with open(data_path, 'r') as f:
        first_line = f.readline()
    
    # Detect delimiter
    if '","' in first_line or first_line.count(',') > first_line.count(';'):
        delimiter = ','
    else:
        delimiter = ';'
    
    print(f"Detected delimiter: '{delimiter}'")
    
    df = pd.read_csv(data_path, sep=delimiter)
    df.columns = df.columns.str.strip().str.strip('"')  # Remove quotes from column names
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
    
    # Detect dataset type and target column
    if 'cardio' in df.columns:
        dataset_type = 'cardio'
        target_col = 'cardio'
        print(f"  → Detected: Cardiovascular dataset")
    elif 'Class' in df.columns:
        dataset_type = 'creditcard'
        target_col = 'Class'
        print(f"  → Detected: Credit Card Fraud dataset")
    else:
        # Auto-detect target (last column)
        dataset_type = 'generic'
        target_col = df.columns[-1]
        print(f"  ⚠️  Target column not auto-detected, using last column: '{target_col}'")
    
    print(f"Target column: '{target_col}'")
    
    # Feature engineering based on dataset type
    if dataset_type == 'cardio':
        # Cardiovascular dataset feature engineering
        df['age_years'] = df['age'] / 365.25
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
        df['is_hypertension'] = ((df['ap_hi'] >= 140) | (df['ap_lo'] >= 90)).astype(int)
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
        
        feature_cols = [
            'age_years', 'gender', 'height', 'weight', 'bmi',
            'ap_hi', 'ap_lo', 'pulse_pressure', 'map', 'is_hypertension',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active'
        ]
    elif dataset_type == 'creditcard':
        # Credit card fraud dataset - use all V1-V28 features + Time + Amount
        feature_cols = [col for col in df.columns if col not in [target_col]]
        print(f"  Using all {len(feature_cols)} features (V1-V28, Time, Amount)")
    else:
        # Generic dataset - use all columns except target
        feature_cols = [col for col in df.columns if col != target_col]
        print(f"  Using all {len(feature_cols)} available features")
    
    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values
    
    print(f"\nFinal preprocessing:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")
    print(f"  Positive class: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"  Negative class: {(1-y).sum()} ({(1-y).sum()/len(y)*100:.1f}%)")
    
    return X, y, feature_cols

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive medical metrics"""
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred),  # True Positive Rate
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # True Negative Rate
        'precision': precision_score(y_true, y_pred),  # Positive Predictive Value
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0,  # Negative Predictive Value
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': pr_auc,
        'mcc': matthews_corrcoef(y_true, y_pred),
    }
    
    return metrics

# ============================================================================
# MODEL CACHING FUNCTIONS
# ============================================================================

def get_experiment_hash(model_name, scaler_name, imbalance_name, feature_selector_name):
    """Generate unique hash for experiment configuration"""
    import hashlib
    config_str = f"{model_name}_{scaler_name}_{imbalance_name}_{feature_selector_name}"
    return hashlib.md5(config_str.encode()).hexdigest()[:12]

def get_cache_path(model_name, scaler_name, imbalance_name, feature_selector_name):
    """Get cache file path for experiment"""
    cache_dir = Path(CONFIG['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    exp_hash = get_experiment_hash(model_name, scaler_name, imbalance_name, feature_selector_name)
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    cache_file = cache_dir / f"{safe_model_name}_{exp_hash}.pkl"
    
    return cache_file

def save_experiment_to_cache(model_name, scaler_name, imbalance_name, feature_selector_name, results):
    """Save experiment results to cache"""
    import joblib
    
    cache_file = get_cache_path(model_name, scaler_name, imbalance_name, feature_selector_name)
    
    cache_data = {
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model': model_name,
            'scaler': scaler_name,
            'imbalance': imbalance_name,
            'feature_selection': feature_selector_name,
        }
    }
    
    joblib.dump(cache_data, cache_file)
    return cache_file

def load_experiment_from_cache(model_name, scaler_name, imbalance_name, feature_selector_name):
    """Load experiment results from cache if available"""
    import joblib
    
    cache_file = get_cache_path(model_name, scaler_name, imbalance_name, feature_selector_name)
    
    if cache_file.exists():
        try:
            cache_data = joblib.load(cache_file)
            return cache_data['results']
        except Exception as e:
            print(f"  ⚠️  Cache load failed: {e}")
            return None
    
    return None

def clear_cache():
    """Clear all cached experiments"""
    cache_dir = Path(CONFIG['cache_dir'])
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print(f"✓ Cache cleared: {cache_dir}")
    else:
        print("ℹ️  No cache directory found")

def list_cached_experiments():
    """List all cached experiments"""
    import joblib
    
    cache_dir = Path(CONFIG['cache_dir'])
    if not cache_dir.exists():
        print("ℹ️  No cache directory found")
        return []
    
    cached_files = list(cache_dir.glob("*.pkl"))
    
    if not cached_files:
        print("ℹ️  No cached experiments found")
        return []
    
    print(f"\n{'='*80}")
    print(f"CACHED EXPERIMENTS ({len(cached_files)} found)")
    print(f"{'='*80}")
    
    cached_experiments = []
    for cache_file in cached_files:
        try:
            cache_data = joblib.load(cache_file)
            config = cache_data['config']
            results = cache_data['results']
            
            cached_experiments.append({
                'file': cache_file.name,
                'model': config['model'],
                'scaler': config['scaler'],
                'imbalance': config['imbalance'],
                'feature_selection': config['feature_selection'],
                'pr_auc': results.get('pr_auc', 'N/A'),
                'timestamp': cache_data['timestamp'],
            })
            
            print(f"\n{config['model']:30s}")
            print(f"  Config: {config['scaler']:8s} | {config['imbalance']:12s} | {config['feature_selection']}")
            prauc = results.get('pr_auc', None)
            if isinstance(prauc, (int, float)):
                print(f"  PR-AUC: {prauc:.4f}")
            else:
                print("  PR-AUC: N/A")
            print(f"  Cached: {cache_data['timestamp']}")
            
        except Exception as e:
            print(f"  ⚠️  Error loading {cache_file.name}: {e}")
    
    return cached_experiments

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_single_experiment(
    model_name, model_info,
    X_train, y_train, X_test, y_test,
    scaler_name, imbalance_name, feature_selector_name,
    cv_folds=5
):
    """Train a single experiment with specific configuration"""

    # Check cache first if enabled
    if CONFIG.get('use_cache', True):
        cached_results = load_experiment_from_cache(
            model_name, scaler_name, imbalance_name, feature_selector_name
        )
        if cached_results is not None:
            print("  [CACHE] Loaded from cache!", flush=True)
            return cached_results

    results = {
        'model': model_name,
        'generation': model_info['generation'],
        'scaler': scaler_name,
        'imbalance': imbalance_name,
        'feature_selection': feature_selector_name,
    }

    # Setup preprocessing
    scaler = SCALING_METHODS[scaler_name] if model_info['needs_scaling'] else None
    imbalance_method = IMBALANCE_METHODS[imbalance_name]
    feature_selector = FEATURE_SELECTION_METHODS[feature_selector_name]

    # Check if model has fast_cv_folds override (for slow models)
    actual_cv_folds = model_info.get('fast_cv_folds', cv_folds)
    if actual_cv_folds != cv_folds:
        print(f"  [FAST MODE] Using {actual_cv_folds} fold(s) instead of {cv_folds} (slow model)", flush=True)

    fold_metrics = []
    fold_times = []  # Track time for each fold for ETA estimation

    start_time = time.time()

    # Handle single fold case (no CV, just train/val split)
    if actual_cv_folds == 1:
        print(f"  [SINGLE SPLIT] Using 80/20 train/val split (no cross-validation)", flush=True)
        # Create single train/val split
        train_idx, val_idx = train_test_split(
            np.arange(len(X_train)),
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )
        cv_splits = [(train_idx, val_idx)]
    else:
        # Normal cross-validation with StratifiedKFold
        cv = StratifiedKFold(n_splits=actual_cv_folds, shuffle=True, random_state=42)
        cv_splits = list(cv.split(X_train, y_train))

    # Progress bar for CV folds/splits
    for fold, (train_idx, val_idx) in enumerate(tqdm(cv_splits, desc=f"  CV Folds", leave=False, ncols=100), 1):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # Print fold start with ETA if we have previous fold times
        fold_start = time.time()
        if fold_times:
            avg_fold_time = np.mean(fold_times)
            remaining_folds = actual_cv_folds - fold + 1
            eta_seconds = avg_fold_time * remaining_folds
            eta_str = f" (ETA all: {eta_seconds/60:.1f}m, ~{avg_fold_time:.0f}s/fold)" if eta_seconds > 60 else f" (ETA: ~{avg_fold_time:.0f}s)"
        else:
            eta_str = ""
        print(f"    Fold {fold}/{actual_cv_folds}{eta_str} - Preprocessing...", end='', flush=True)

        # Preprocessing phase
        preproc_start = time.time()

        # 1. Scaling FIRST (fit on fold train, normalize features)
        if scaler is not None:
            from sklearn.base import clone
            sc = clone(scaler)
            X_fold_train = sc.fit_transform(X_fold_train)
            X_fold_val = sc.transform(X_fold_val)

        # 2. Feature selection SECOND (on scaled data, avoid leakage)
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

        preproc_time = time.time() - preproc_start
        print(f" Done ({preproc_time:.1f}s). Training model...", end='', flush=True)

        # Train model - NOW START TIMING FOR MODEL TRAINING ONLY
        model_train_start = time.time()
        from sklearn.base import clone
        mdl = clone(model_info['model'])

        # Start periodic progress updates for slow models (in a separate thread)
        import threading
        stop_progress = threading.Event()

        def print_progress():
            """Print elapsed time and progress % every 10 seconds during training"""
            spinner = ['|', '/', '-', '\\']
            spin_idx = 0
            while not stop_progress.is_set():
                stop_progress.wait(10)  # Update every 10 seconds
                if not stop_progress.is_set():
                    elapsed_total = time.time() - fold_start
                    elapsed_training = time.time() - model_train_start
                    # Estimate progress based on previous fold times
                    if fold_times:
                        avg_time = np.mean(fold_times)
                        progress_pct = min(100, (elapsed_total / avg_time) * 100)
                        eta_remaining = max(0, avg_time - elapsed_total)
                        print(f" {spinner[spin_idx]} [train: {elapsed_training:.0f}s, total: {elapsed_total:.0f}s, ~{progress_pct:.0f}%, ETA {eta_remaining:.0f}s]", end='', flush=True)
                    else:
                        print(f" {spinner[spin_idx]} [train: {elapsed_training:.0f}s]", end='', flush=True)
                    spin_idx = (spin_idx + 1) % 4

        progress_thread = threading.Thread(target=print_progress, daemon=True)
        progress_thread.start()

        # Check if model supports early stopping with eval_set
        use_eval_set = model_info.get('use_eval_set', False)
        
        if use_eval_set:
            # For XGBoost, LightGBM, CatBoost - use validation set for early stopping
            # Need to create a small validation split from training data
            X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(
                X_fold_train, y_fold_train, 
                test_size=0.1, 
                random_state=42, 
                stratify=y_fold_train
            )
            
            # Fit with early stopping
            # Different models handle verbose parameter differently
            if 'LightGBM' in model_name or 'LGBMClassifier' in str(type(mdl)):
                # LightGBM: use callbacks for early stopping (new API)
                import lightgbm as lgb
                mdl.fit(
                    X_train_fit, y_train_fit,
                    eval_set=[(X_val_fit, y_val_fit)],
                    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
                )
            elif 'PyTorch' in model_name or 'TabNet' in model_name:
                # PyTorch models and TabNet handle verbosity in constructor
                mdl.fit(
                    X_train_fit, y_train_fit,
                    eval_set=[(X_val_fit, y_val_fit)]
                )
            else:
                # XGBoost and CatBoost accept verbose parameter
                mdl.fit(
                    X_train_fit, y_train_fit,
                    eval_set=[(X_val_fit, y_val_fit)],
                    verbose=False
                )
        else:
            # Standard fit for other models
            mdl.fit(X_fold_train, y_fold_train)
        
        # Predict
        y_pred = mdl.predict(X_fold_val)
        y_proba = mdl.predict_proba(X_fold_val)[:, 1]
        
        # Stop progress updates
        stop_progress.set()

        # Calculate metrics
        metrics = calculate_metrics(y_fold_val, y_pred, y_proba)
        fold_metrics.append(metrics)

        # Print fold completion time and add to tracking
        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        print(f" Done ({fold_time:.1f}s)", flush=True)
    
    train_time = time.time() - start_time
    
    # Average metrics across folds
    avg_metrics = {}
    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)
    
    results.update(avg_metrics)
    results['train_time_sec'] = train_time
    
    # Save to cache if enabled
    if CONFIG.get('use_cache', True):
        save_experiment_to_cache(
            model_name, scaler_name, imbalance_name, feature_selector_name, results
        )
    
    return results

# ============================================================================
# BEST MODEL TRAINING & SAVING
# ============================================================================

def train_and_save_best_model(best_config, X_train, y_train, X_test, y_test, feature_names, output_dir):
    """Train best model on full training data and save everything"""
    import joblib
    
    print(f"\n{'='*80}")
    print("TRAINING BEST MODEL ON FULL TRAINING DATA")
    print(f"{'='*80}")
    
    # Extract configuration
    model_name = best_config['model']
    scaler_name = best_config['scaler']
    imbalance_name = best_config['imbalance']
    feature_selector_name = best_config['feature_selection']
    
    print(f"Model: {model_name}")
    print(f"Scaler: {scaler_name}")
    print(f"Imbalance: {imbalance_name}")
    print(f"Feature Selection: {feature_selector_name}")
    
    # Get fresh model instance
    models = get_models()
    model_info = models[model_name]
    
    # Setup preprocessing pipeline
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    y_train_processed = y_train.copy()
    
    # Align preprocessing order with CV: Scaling -> Feature Selection -> Imbalance
    # 1. Scaling (if model needs it)
    scaler = None
    if scaler_name != 'none' and model_info['needs_scaling']:
        scaler = SCALING_METHODS[scaler_name]
        X_train_processed = scaler.fit_transform(X_train_processed)
        X_test_processed = scaler.transform(X_test_processed)
        print(f"✓ Scaling applied: {scaler_name}")

    # 2. Feature Selection
    feature_selector = None
    if feature_selector_name != 'none':
        feature_selector = FEATURE_SELECTION_METHODS[feature_selector_name]
        X_train_processed = feature_selector.fit_transform(X_train_processed, y_train_processed)
        X_test_processed = feature_selector.transform(X_test_processed)
        print(f"✓ Feature selection: {X_train_processed.shape[1]} features selected")

    # 3. Handle Imbalance (train set only)
    imbalance_handler = None
    if imbalance_name != 'none':
        imbalance_handler = IMBALANCE_METHODS[imbalance_name]
        X_train_processed, y_train_processed = imbalance_handler.fit_resample(X_train_processed, y_train_processed)
        print(f"✓ Imbalance handling: {len(X_train_processed)} samples after {imbalance_name}")
    
    # 4. Train final model
    print(f"\n{'='*80}")
    print("TRAINING FINAL MODEL...")
    print(f"{'='*80}")
    
    from sklearn.base import clone
    final_model = clone(model_info['model'])
    
    # Check if model supports early stopping
    use_eval_set = model_info.get('use_eval_set', False)
    
    if use_eval_set:
        # Split for validation during training
        X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(
            X_train_processed, y_train_processed,
            test_size=0.1,
            random_state=42,
            stratify=y_train_processed
        )
        
        print(f"Training with early stopping (validation: {len(X_val_fit)} samples)")
        start_time = time.time()

        # Handle verbose parameter based on model type
        if 'LightGBM' in model_name or 'LGBMClassifier' in str(type(final_model)):
            # Pass early_stopping_rounds via fit for LightGBM
            final_model.fit(
                X_train_fit, y_train_fit,
                eval_set=[(X_val_fit, y_val_fit)],
                early_stopping_rounds=100
            )
        elif 'PyTorch' in model_name or 'TabNet' in model_name:
            # PyTorch models and TabNet handle verbosity in constructor
            final_model.fit(
                X_train_fit, y_train_fit,
                eval_set=[(X_val_fit, y_val_fit)]
            )
        else:
            # XGBoost and CatBoost accept verbose parameter
            final_model.fit(
                X_train_fit, y_train_fit,
                eval_set=[(X_val_fit, y_val_fit)],
                verbose=False
            )

        train_time = time.time() - start_time
        
        # Get best iteration info
        # Best iteration info (handles LightGBM/XGBoost/CatBoost)
        best_iter = None
        if hasattr(final_model, 'best_iteration'):
            best_iter = final_model.best_iteration
        elif hasattr(final_model, 'best_iteration_'):
            best_iter = final_model.best_iteration_
        if best_iter is not None:
            print(f"✓ Best iteration: {best_iter}")
    else:
        print(f"Training on full data ({len(X_train_processed)} samples)")
        start_time = time.time()
        final_model.fit(X_train_processed, y_train_processed)
        train_time = time.time() - start_time
    
    print(f"✓ Training completed in {train_time:.1f}s")
    
    # 5. Evaluate on test set
    print(f"\n{'='*80}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*80}")
    
    y_pred = final_model.predict(X_test_processed)
    y_proba = final_model.predict_proba(X_test_processed)[:, 1]
    
    final_metrics = calculate_metrics(y_test, y_pred, y_proba)

    print(f"PR-AUC:           {final_metrics['pr_auc']:.4f}")
    print(f"Accuracy:         {final_metrics['accuracy']:.4f}")
    print(f"Balanced Acc:     {final_metrics['balanced_accuracy']:.4f}")
    print(f"ROC-AUC:          {final_metrics['roc_auc']:.4f}")
    print(f"Sensitivity:      {final_metrics['sensitivity']:.4f}")
    print(f"Specificity:      {final_metrics['specificity']:.4f}")
    print(f"F1-Score:         {final_metrics['f1']:.4f}")
    print(f"Precision:        {final_metrics['precision']:.4f}")
    print(f"MCC:              {final_metrics['mcc']:.4f}")
    
    # 6. Save everything
    print(f"\n{'='*80}")
    print("SAVING MODEL & ARTIFACTS")
    print(f"{'='*80}")
    
    save_dir = output_dir / "best_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = save_dir / f"best_model_{timestamp}.pkl"
    joblib.dump(final_model, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Save scaler (if used)
    if scaler is not None:
        scaler_path = save_dir / f"scaler_{timestamp}.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"✓ Scaler saved: {scaler_path}")
    
    # Save feature selector (if used)
    if feature_selector is not None:
        fs_path = save_dir / f"feature_selector_{timestamp}.pkl"
        joblib.dump(feature_selector, fs_path)
        print(f"✓ Feature selector saved: {fs_path}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_name': model_name,
        'generation': int(best_config['generation']),
        'configuration': {
            'scaler': scaler_name,
            'imbalance': imbalance_name,
            'feature_selection': feature_selector_name,
        },
        'cv_metrics': {
            'pr_auc': float(best_config['pr_auc']),
            'pr_auc_std': float(best_config['pr_auc_std']),
            'sensitivity': float(best_config['sensitivity']),
            'specificity': float(best_config['specificity']),
            'f1': float(best_config['f1']),
        },
        'test_metrics': {k: float(v) for k, v in final_metrics.items()},
        'training_info': {
            'train_samples': int(len(X_train_processed)),
            'test_samples': int(len(X_test)),
            'features': int(X_train_processed.shape[1]),
            'train_time_sec': float(train_time),
        },
        'feature_names': feature_names,
        'best_iteration': int(final_model.best_iteration) if hasattr(final_model, 'best_iteration') else None,
    }
    
    metadata_path = save_dir / f"metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")
    
    # Save prediction script
    prediction_script = f'''"""
Best Model Prediction Script
Generated: {timestamp}
Model: {model_name}
PR-AUC (CV): {best_config['pr_auc']:.4f}
PR-AUC (Test): {final_metrics['pr_auc']:.4f}
Accuracy (Test): {final_metrics['accuracy']:.4f}
"""

import joblib
import numpy as np
import pandas as pd

# Load model and preprocessing
model = joblib.load("{model_path.name}")
{"scaler = joblib.load('" + f"scaler_{timestamp}.pkl" + "')" if scaler is not None else "# No scaler used"}
{"feature_selector = joblib.load('" + f"feature_selector_{timestamp}.pkl" + "')" if feature_selector is not None else "# No feature selection used"}

def predict(X):
    """
    Predict cardiovascular disease probability
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, {len(feature_names)})
        Input features: {feature_names}
    
    Returns:
    --------
    predictions : dict
        - 'class': predicted class (0 or 1)
        - 'probability': probability of positive class
        - 'risk_level': risk interpretation
    """
    X = np.array(X).reshape(1, -1) if len(np.array(X).shape) == 1 else np.array(X)
    
    # Apply preprocessing (order: scale -> feature select)
    {"X = scaler.transform(X)" if scaler is not None else "# No scaling"}
    {"X = feature_selector.transform(X)" if feature_selector is not None else "# No feature selection"}
    
    # Predict
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    results = []
    for pred, proba in zip(y_pred, y_proba):
        risk_level = "HIGH" if proba >= 0.7 else "MODERATE" if proba >= 0.4 else "LOW"
        results.append({{
            'class': int(pred),
            'probability': float(proba),
            'risk_level': risk_level
        }})
    
    return results[0] if len(results) == 1 else results

# Example usage
if __name__ == "__main__":
    # Sample patient data (replace with actual values)
    sample = [
        # age_years, gender, height, weight, bmi, ap_hi, ap_lo, 
        # pulse_pressure, map, is_hypertension, cholesterol, gluc, smoke, alco, active
        52.5, 1, 165, 75, 27.5, 130, 85, 45, 100, 0, 2, 1, 0, 0, 1
    ]
    
    result = predict(sample)
    print(f"Prediction: {{result['class']}}")
    print(f"Probability: {{result['probability']:.2%}}")
    print(f"Risk Level: {{result['risk_level']}}")
'''
    
    script_path = save_dir / f"predict_{timestamp}.py"
    with open(script_path, 'w') as f:
        f.write(prediction_script)
    print(f"✓ Prediction script saved: {script_path}")
    
    print(f"\n{'='*80}")
    print("✅ BEST MODEL PACKAGE SAVED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Location: {save_dir}")
    print(f"\nTo use the model:")
    print(f"  python {script_path.name}")
    
    return final_model, final_metrics

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Detect dataset name from path for organizing outputs
    data_path = Path(CONFIG['data_path'])
    dataset_name = data_path.stem  # e.g., 'cardio_train' or 'creditcard'
    
    # Setup dataset-specific directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('experiments/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'{dataset_name}_{timestamp}.log'
    
    # Update output directory to be dataset-specific
    base_output_dir = Path('experiments/full_comparison')
    output_dir = base_output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    CONFIG['output_dir'] = str(output_dir)
    
    # Update cache directory to be dataset-specific
    base_cache_dir = Path('experiments/model_cache')
    cache_dir = base_cache_dir / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    CONFIG['cache_dir'] = str(cache_dir)
    
    # Tee class to write to both file and console
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for f in self.files:
                f.write(text)
                f.flush()  # Ensure immediate write
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Redirect stdout to both console and file
    import sys
    log_fp = open(log_file, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_fp)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL & PREPROCESSING COMPARISON")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir}")
    
    # Load data
    data_path = Path(CONFIG['data_path'])
    if not data_path.exists():
        print(f"\n❌ Dataset not found at {data_path}")
        print("   Download from: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset")
        return
    
    X, y, feature_names = load_and_engineer_data(data_path)
    
    # Train/test split
    print(f"\n{'='*80}")
    print("TRAIN/TEST SPLIT")
    print(f"{'='*80}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'], 
        stratify=y
    )
    print(f"Train: {len(X_train)} samples ({y_train.sum()} positive)")
    print(f"Test:  {len(X_test)} samples ({y_test.sum()} positive)")
    
    # Cache status
    if CONFIG.get('use_cache', True):
        cache_dir = Path(CONFIG['cache_dir'])
        if cache_dir.exists():
            cached_count = len(list(cache_dir.glob("*.pkl")))
            print(f"\n{'='*80}")
            print(f"CACHE STATUS")
            print(f"{'='*80}")
            print(f"Cache enabled: ✓")
            print(f"Cache directory: {cache_dir}")
            print(f"Cached experiments: {cached_count}")
            if cached_count > 0:
                print(f"💡 Tip: Use --list-cache to see cached experiments")
                print(f"💡 Tip: Use --clear-cache to reset cache")
        else:
            print(f"\n💡 Cache enabled but no cached experiments yet")
    else:
        print(f"\n⚠️  Cache disabled - all experiments will be run")
    
    # Get all models
    models = get_models()
    print(f"\n{'='*80}")
    print(f"LOADED {len(models)} MODELS")
    print(f"{'='*80}")
    for name, info in models.items():
        print(f"  {name:30s} - Gen {info['generation']} - {info['description']}")
    
    # Generate experiment combinations
    print(f"\n{'='*80}")
    print("EXPERIMENT MATRIX")
    print(f"{'='*80}")
    
    # Smart configuration: Only scale for models that need it
    experiments = []
    for model_name, model_info in models.items():
        scalers = ['standard', 'robust'] if model_info['needs_scaling'] else ['none']
        
        for scaler, imbalance, feat_sel in product(
            scalers,
            ['none', 'smote', 'smote_enn'],
            ['none', 'select_k_best_5', 'select_k_best_12', 'mutual_info_5', 'mutual_info_12']
        ):
            experiments.append({
                'model_name': model_name,
                'model_info': model_info,
                'scaler': scaler,
                'imbalance': imbalance,
                'feature_selection': feat_sel
            })
    
    print(f"Total experiments: {len(experiments)}")
    print(f"Estimated time: {len(experiments) * 30 / 60:.1f} - {len(experiments) * 60 / 60:.1f} minutes")
    
    # Add scalers for models that don't need scaling
    SCALING_METHODS['none'] = None
    
    # Run experiments
    print(f"\n{'='*80}")
    print("RUNNING EXPERIMENTS")
    print(f"{'='*80}")

    all_results = []

    # Main progress bar for all experiments
    pbar = tqdm(total=len(experiments), desc="Overall Progress", unit="exp",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for idx, exp in enumerate(experiments, 1):
        # Print experiment details
        print(f"\n[{idx}/{len(experiments)}] {exp['model_name']} | "
              f"Scale: {exp['scaler']} | Imb: {exp['imbalance']} | FeatSel: {exp['feature_selection']}", flush=True)

        result = train_single_experiment(
            exp['model_name'], exp['model_info'],
            X_train, y_train, X_test, y_test,
            exp['scaler'], exp['imbalance'], exp['feature_selection'],
            cv_folds=CONFIG['cv_folds']
        )

        all_results.append(result)

        # Print quick summary
        print(f"  [OK] PR-AUC: {result['pr_auc']:.4f} | "
              f"Acc: {result['accuracy']:.4f} | "
              f"Sens: {result['sensitivity']:.4f} | "
              f"Spec: {result['specificity']:.4f} | "
              f"F1: {result['f1']:.4f} | "
              f"Time: {result['train_time_sec']:.1f}s", flush=True)

        # Update progress bar
        pbar.update(1)

    pbar.close()
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = output_dir / f"full_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Results saved to: {results_path}")
    
    # Top 10 by PR-AUC
    print(f"\n{'='*80}")
    print("TOP 10 CONFIGURATIONS BY PR-AUC")
    print(f"{'='*80}")
    top10 = results_df.nlargest(10, 'pr_auc')
    
    for idx, row in top10.iterrows():
        print(f"\n{row['model']:30s} (Gen {row['generation']})")
        print(f"  Config: Scale={row['scaler']:8s} | Imb={row['imbalance']:12s} | FeatSel={row['feature_selection']}")
        print(f"  PR-AUC:      {row['pr_auc']:.4f} ± {row['pr_auc_std']:.4f}")
        print(f"  Accuracy:    {row['accuracy']:.4f} ± {row['accuracy_std']:.4f}")
        print(f"  ROC-AUC:     {row['roc_auc']:.4f} ± {row['roc_auc_std']:.4f}")
        print(f"  Sensitivity: {row['sensitivity']:.4f} ± {row['sensitivity_std']:.4f}")
        print(f"  Specificity: {row['specificity']:.4f} ± {row['specificity_std']:.4f}")
        print(f"  F1-Score:    {row['f1']:.4f} ± {row['f1_std']:.4f}")
        print(f"  Precision:   {row['precision']:.4f}")
        print(f"  MCC:         {row['mcc']:.4f}")
        print(f"  Train Time:  {row['train_time_sec']:.1f}s")
    
    # Generation comparison
    print(f"\n{'='*80}")
    print("GENERATION COMPARISON (BEST CONFIG PER GENERATION)")
    print(f"{'='*80}")
    for gen in sorted(results_df['generation'].unique()):
        gen_df = results_df[results_df['generation'] == gen]
        best = gen_df.nlargest(1, 'pr_auc').iloc[0]
        print(f"\nGeneration {gen}: {best['model']}")
        print(f"  Best PR-AUC: {best['pr_auc']:.4f} (Config: {best['scaler']}/{best['imbalance']}/{best['feature_selection']})")
        print(f"  Accuracy: {best['accuracy']:.4f} | Sensitivity: {best['sensitivity']:.4f} | Specificity: {best['specificity']:.4f} | F1: {best['f1']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"COMPLETE! Total time: {results_df['train_time_sec'].sum() / 60:.1f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Close log file
    import sys
    sys.stdout.flush()
    if hasattr(sys.stdout, 'files'):
        for f in sys.stdout.files:
            if f != original_stdout:
                f.close()
        sys.stdout = original_stdout
    print(f"\n✅ Log saved to: {log_file}")
    
    # ========================================================================
    # TRAIN AND SAVE BEST MODEL
    # ========================================================================
    print(f"\n{'='*80}")
    print("🏆 IDENTIFYING & SAVING BEST MODEL")
    print(f"{'='*80}")
    
    # Find best configuration by PR-AUC
    best_idx = results_df['pr_auc'].idxmax()
    best_config = results_df.loc[best_idx]
    
    print(f"\nBest Configuration:")
    print(f"  Model: {best_config['model']}")
    print(f"  PR-AUC (CV): {best_config['pr_auc']:.4f} ± {best_config['pr_auc_std']:.4f}")
    print(f"  Config: {best_config['scaler']} / {best_config['imbalance']} / {best_config['feature_selection']}")
    
    # Train and save
    final_model, final_metrics = train_and_save_best_model(
        best_config, X_train, y_train, X_test, y_test, 
        feature_names, output_dir
    )
    
    print(f"\n{'='*80}")
    print("🎉 ALL DONE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    import argparse
    
    # Argument parser
    parser = argparse.ArgumentParser(
        description='Comprehensive Model & Preprocessing Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python full_comparison.py --data data/raw/creditcard.csv
  python full_comparison.py --data data/raw/cardio_train.csv --no-cache
  python full_comparison.py --list-cache
  python full_comparison.py --clear-cache
        """
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default=CONFIG['data_path'],
        help=f'Path to dataset CSV file (default: {CONFIG["data_path"]})'
    )
    parser.add_argument(
        '--no-cache', 
        action='store_true',
        help='Disable caching and rerun all experiments'
    )
    parser.add_argument(
        '--clear-cache', 
        action='store_true',
        help='Clear all cached experiments and exit'
    )
    parser.add_argument(
        '--list-cache', 
        action='store_true',
        help='List all cached experiments and exit'
    )
    parser.add_argument(
        '--n-trials', 
        type=int,
        help='(Ignored - kept for compatibility)'
    )
    parser.add_argument(
        '--timeout', 
        type=int,
        help='(Ignored - kept for compatibility)'
    )
    
    args = parser.parse_args()
    
    # Handle cache management commands
    if args.clear_cache:
        print("Clearing experiment cache...")
        clear_cache()
        import sys
        sys.exit(0)
    
    if args.list_cache:
        list_cached_experiments()
        import sys
        sys.exit(0)
    
    # Update config
    if args.no_cache:
        print("⚠️  Running WITHOUT cache")
        CONFIG['use_cache'] = False
    
    if args.data:
        CONFIG['data_path'] = args.data
        print(f"📁 Using dataset: {args.data}")
    
    main()
