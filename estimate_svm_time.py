"""
Estimate training time for SVM on creditcard dataset
Config: Gen2_SVM_RBF | Scale: standard | Imb: smote | FeatSel: none
"""
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

print("="*80)
print("SVM Training Time Estimation")
print("="*80)

# Load data
print("\n1. Loading creditcard dataset...")
df = pd.read_csv('data/raw/creditcard.csv')
X = df.drop('Class', axis=1).values
y = df['Class'].values

print(f"   Dataset shape: {X.shape}")
print(f"   Positive class: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")

# Train/test split (80/20 as in config)
print("\n2. Train/test split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# Simulate one fold of CV
print("\n3. Simulating ONE fold (1/5) of cross-validation...")
print("   Config: Scale=standard | Imb=smote | FeatSel=none")

# Take 80% of train for this fold (mimicking 5-fold CV)
fold_train_size = int(len(X_train) * 0.8)
X_fold_train = X_train[:fold_train_size]
y_fold_train = y_train[:fold_train_size]
X_fold_val = X_train[fold_train_size:]
y_fold_val = y_train[fold_train_size:]

print(f"   Fold train size: {len(X_fold_train)}")
print(f"   Fold val size: {len(X_fold_val)}")

# Step 1: Scaling
print("\n4. Preprocessing - Scaling...")
preproc_start = time.time()
scaler = StandardScaler()
X_fold_train_scaled = scaler.fit_transform(X_fold_train)
X_fold_val_scaled = scaler.transform(X_fold_val)
scaling_time = time.time() - preproc_start
print(f"   Scaling time: {scaling_time:.2f}s")

# Step 2: SMOTE (this is the slow part!)
print("\n5. Preprocessing - SMOTE (oversampling)...")
smote_start = time.time()
smote = SMOTE(random_state=42)
X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(
    X_fold_train_scaled, y_fold_train
)
smote_time = time.time() - smote_start
print(f"   SMOTE time: {smote_time:.2f}s")
print(f"   Before SMOTE: {len(X_fold_train_scaled)} samples")
print(f"   After SMOTE: {len(X_fold_train_resampled)} samples (class balanced)")
print(f"   Total preprocessing: {scaling_time + smote_time:.2f}s")

# Step 3: Train SVM (THIS IS THE VERY SLOW PART!)
print("\n6. Training SVM-RBF...")
print("   This will take several minutes...")
print("   SVM params: kernel=rbf, C=1.0, gamma=scale, class_weight=balanced")

svm_start = time.time()
svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42,
    class_weight='balanced',
    probability=True
)

# Show progress during training
import threading
stop_flag = threading.Event()

def show_progress():
    spinner = ['|', '/', '-', '\\']
    idx = 0
    while not stop_flag.is_set():
        elapsed = time.time() - svm_start
        print(f"\r   Training... {spinner[idx]} [{elapsed:.0f}s elapsed]", end='', flush=True)
        idx = (idx + 1) % 4
        stop_flag.wait(2)

progress_thread = threading.Thread(target=show_progress, daemon=True)
progress_thread.start()

# Actual training
svm.fit(X_fold_train_resampled, y_fold_train_resampled)

stop_flag.set()
svm_time = time.time() - svm_start
print(f"\r   Training... Done!                                ")
print(f"   SVM training time: {svm_time:.2f}s ({svm_time/60:.2f} minutes)")

# Total time for ONE fold
total_fold_time = scaling_time + smote_time + svm_time
print(f"\n   Total time for ONE fold: {total_fold_time:.2f}s ({total_fold_time/60:.2f} minutes)")

# Extrapolate to 5 folds
print("\n" + "="*80)
print("ESTIMATION FOR FULL 5-FOLD CV")
print("="*80)
total_cv_time = total_fold_time * 5
print(f"Total time for 5 folds: {total_cv_time:.2f}s ({total_cv_time/60:.2f} minutes)")

# Breakdown
print(f"\nBreakdown per fold:")
print(f"  - Preprocessing: {(scaling_time + smote_time):.2f}s")
print(f"  - SVM training: {svm_time:.2f}s")
print(f"  - Total: {total_fold_time:.2f}s")

print(f"\nWith 5 folds:")
print(f"  - Total preprocessing: {(scaling_time + smote_time) * 5:.2f}s ({(scaling_time + smote_time) * 5 / 60:.2f} min)")
print(f"  - Total SVM training: {svm_time * 5:.2f}s ({svm_time * 5 / 60:.2f} min)")
print(f"  - Grand total: {total_cv_time:.2f}s ({total_cv_time/60:.2f} min)")

print("\n" + "="*80)
print("ANSWER TO YOUR QUESTION")
print("="*80)
print(f"Experiment [126/270] Gen2_SVM_RBF | Scale: standard | Imb: smote | FeatSel: none")
print(f"Expected time: ~{total_cv_time/60:.1f} minutes ({total_cv_time/60/60:.2f} hours)")
print(f"\nWith your progress tracking, you'll see updates every 10 seconds!")
print("="*80)
