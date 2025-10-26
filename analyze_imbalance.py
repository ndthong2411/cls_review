import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# Load datasets
cardio = pd.read_csv('data/raw/cardio_train.csv', sep=';')
creditcard = pd.read_csv('data/raw/creditcard.csv')

print('='*100)
print('PHAN TICH CLASS DISTRIBUTION - 2 DATASETS CUA BAN')
print('='*100)

# ============= CARDIO DATASET =============
print('\n[1] CARDIO_TRAIN DATASET:')
print(f'Total samples: {len(cardio):,}')
cardio_dist = cardio['cardio'].value_counts().sort_index()
print(f'Class 0 (Healthy): {cardio_dist[0]:,} samples ({cardio_dist[0]/len(cardio)*100:.1f}%)')
print(f'Class 1 (Disease): {cardio_dist[1]:,} samples ({cardio_dist[1]/len(cardio)*100:.1f}%)')
ratio_cardio = cardio_dist[0]/cardio_dist[1]
print(f'Imbalance Ratio: {ratio_cardio:.2f}:1')
if ratio_cardio < 1.5:
    print('=> BALANCED - Khong can imbalance handling')
else:
    print('=> IMBALANCED - Nen dung SMOTE/SMOTEENN')

# ============= CREDITCARD DATASET =============
print('\n[2] CREDITCARD DATASET:')
print(f'Total samples: {len(creditcard):,}')
credit_dist = creditcard['Class'].value_counts().sort_index()
print(f'Class 0 (Normal): {credit_dist[0]:,} samples ({credit_dist[0]/len(creditcard)*100:.2f}%)')
print(f'Class 1 (Fraud): {credit_dist[1]:,} samples ({credit_dist[1]/len(creditcard)*100:.2f}%)')
ratio_credit = credit_dist[0]/credit_dist[1]
print(f'Imbalance Ratio: {ratio_credit:.2f}:1')
print('=> SEVERELY IMBALANCED!')

# ============= TRAIN/TEST SPLIT =============
print('\n' + '='*100)
print('TRAIN/TEST SPLIT (80/20)')
print('='*100)

# Prepare Cardio data
cardio_features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                   'cholesterol', 'gluc', 'smoke', 'alco', 'active']
X_cardio = cardio[cardio_features]
y_cardio = cardio['cardio']

X_cardio_train, X_cardio_test, y_cardio_train, y_cardio_test = train_test_split(
    X_cardio, y_cardio, test_size=0.2, random_state=42, stratify=y_cardio
)

cardio_train_dist = y_cardio_train.value_counts().sort_index()
print('\n[1] CARDIO TRAIN SET (before imbalance handling):')
print(f'Total: {len(y_cardio_train):,} samples')
print(f'Class 0: {cardio_train_dist[0]:,} samples ({cardio_train_dist[0]/len(y_cardio_train)*100:.1f}%)')
print(f'Class 1: {cardio_train_dist[1]:,} samples ({cardio_train_dist[1]/len(y_cardio_train)*100:.1f}%)')

# Prepare CreditCard data
X_credit = creditcard.drop('Class', axis=1)
y_credit = creditcard['Class']

X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(
    X_credit, y_credit, test_size=0.2, random_state=42, stratify=y_credit
)

credit_train_dist = y_credit_train.value_counts().sort_index()
print('\n[2] CREDITCARD TRAIN SET (before imbalance handling):')
print(f'Total: {len(y_credit_train):,} samples')
print(f'Class 0: {credit_train_dist[0]:,} samples ({credit_train_dist[0]/len(y_credit_train)*100:.2f}%)')
print(f'Class 1: {credit_train_dist[1]:,} samples ({credit_train_dist[1]/len(y_credit_train)*100:.2f}%)')

# ============= APPLY SMOTE =============
print('\n' + '='*100)
print('AFTER SMOTE (Synthetic Minority Over-sampling)')
print('='*100)

# SMOTE on Cardio
smote = SMOTE(random_state=42)
X_cardio_smote, y_cardio_smote = smote.fit_resample(X_cardio_train, y_cardio_train)
cardio_smote_dist = pd.Series(y_cardio_smote).value_counts().sort_index()

print('\n[1] CARDIO + SMOTE:')
print(f'Total: {len(y_cardio_smote):,} samples (was {len(y_cardio_train):,})')
print(f'Class 0: {cardio_smote_dist[0]:,} samples ({cardio_smote_dist[0]/len(y_cardio_smote)*100:.1f}%)')
print(f'Class 1: {cardio_smote_dist[1]:,} samples ({cardio_smote_dist[1]/len(y_cardio_smote)*100:.1f}%)')
print(f'Change: +{len(y_cardio_smote) - len(y_cardio_train):,} samples ({(len(y_cardio_smote)/len(y_cardio_train) - 1)*100:.1f}% increase)')

# SMOTE on CreditCard
X_credit_smote, y_credit_smote = smote.fit_resample(X_credit_train, y_credit_train)
credit_smote_dist = pd.Series(y_credit_smote).value_counts().sort_index()

print('\n[2] CREDITCARD + SMOTE:')
print(f'Total: {len(y_credit_smote):,} samples (was {len(y_credit_train):,})')
print(f'Class 0: {credit_smote_dist[0]:,} samples ({credit_smote_dist[0]/len(y_credit_smote)*100:.2f}%)')
print(f'Class 1: {credit_smote_dist[1]:,} samples ({credit_smote_dist[1]/len(y_credit_smote)*100:.2f}%)')
print(f'Change: +{len(y_credit_smote) - len(y_credit_train):,} samples ({(len(y_credit_smote)/len(y_credit_train) - 1)*100:.1f}% increase)')

# ============= APPLY SMOTEENN =============
print('\n' + '='*100)
print('AFTER SMOTEENN (SMOTE + Edited Nearest Neighbors)')
print('='*100)

# SMOTEENN on Cardio
smoteenn = SMOTEENN(random_state=42)
X_cardio_enn, y_cardio_enn = smoteenn.fit_resample(X_cardio_train, y_cardio_train)
cardio_enn_dist = pd.Series(y_cardio_enn).value_counts().sort_index()

print('\n[1] CARDIO + SMOTEENN:')
print(f'Total: {len(y_cardio_enn):,} samples (was {len(y_cardio_train):,})')
print(f'Class 0: {cardio_enn_dist[0]:,} samples ({cardio_enn_dist[0]/len(y_cardio_enn)*100:.1f}%)')
print(f'Class 1: {cardio_enn_dist[1]:,} samples ({cardio_enn_dist[1]/len(y_cardio_enn)*100:.1f}%)')
change_cardio = len(y_cardio_enn) - len(y_cardio_train)
print(f'Change: {change_cardio:+,} samples ({(len(y_cardio_enn)/len(y_cardio_train) - 1)*100:.1f}% change)')
print(f'Note: SMOTEENN removed {len(y_cardio_smote) - len(y_cardio_enn):,} noisy samples after SMOTE')

# SMOTEENN on CreditCard
X_credit_enn, y_credit_enn = smoteenn.fit_resample(X_credit_train, y_credit_train)
credit_enn_dist = pd.Series(y_credit_enn).value_counts().sort_index()

print('\n[2] CREDITCARD + SMOTEENN:')
print(f'Total: {len(y_credit_enn):,} samples (was {len(y_credit_train):,})')
print(f'Class 0: {credit_enn_dist[0]:,} samples ({credit_enn_dist[0]/len(y_credit_enn)*100:.2f}%)')
print(f'Class 1: {credit_enn_dist[1]:,} samples ({credit_enn_dist[1]/len(y_credit_enn)*100:.2f}%)')
change_credit = len(y_credit_enn) - len(y_credit_train)
print(f'Change: {change_credit:+,} samples ({(len(y_credit_enn)/len(y_credit_train) - 1)*100:.1f}% change)')
print(f'Note: SMOTEENN removed {len(y_credit_smote) - len(y_credit_enn):,} noisy samples after SMOTE')

# ============= SUMMARY TABLE =============
print('\n' + '='*100)
print('BANG TONG HOP: ANH HUONG CUA IMBALANCE HANDLING')
print('='*100)

print('\nCARDIO_TRAIN DATASET:')
print('-' * 100)
print(f'{"Method":<20} {"Total Samples":<15} {"Class 0":<15} {"Class 1":<15} {"Change":<20}')
print('-' * 100)
print(f'{"Original Train":<20} {len(y_cardio_train):>14,} {cardio_train_dist[0]:>14,} {cardio_train_dist[1]:>14,} {"baseline":<20}')
print(f'{"+ SMOTE":<20} {len(y_cardio_smote):>14,} {cardio_smote_dist[0]:>14,} {cardio_smote_dist[1]:>14,} {f"+{len(y_cardio_smote)-len(y_cardio_train):,}":<20}')
print(f'{"+ SMOTEENN":<20} {len(y_cardio_enn):>14,} {cardio_enn_dist[0]:>14,} {cardio_enn_dist[1]:>14,} {f"{len(y_cardio_enn)-len(y_cardio_train):+,}":<20}')

print('\nCREDITCARD DATASET:')
print('-' * 100)
print(f'{"Method":<20} {"Total Samples":<15} {"Class 0":<15} {"Class 1":<15} {"Change":<20}')
print('-' * 100)
print(f'{"Original Train":<20} {len(y_credit_train):>14,} {credit_train_dist[0]:>14,} {credit_train_dist[1]:>14,} {"baseline":<20}')
print(f'{"+ SMOTE":<20} {len(y_credit_smote):>14,} {credit_smote_dist[0]:>14,} {credit_smote_dist[1]:>14,} {f"+{len(y_credit_smote)-len(y_credit_train):,}":<20}')
print(f'{"+ SMOTEENN":<20} {len(y_credit_enn):>14,} {credit_enn_dist[0]:>14,} {credit_enn_dist[1]:>14,} {f"+{len(y_credit_enn)-len(y_credit_train):,}":<20}')

print('\n' + '='*100)
print('KET LUAN:')
print('='*100)
print('\n1. CARDIO DATASET:')
print('   - Da can bang san (50/50), SMOTE/SMOTEENN khong tang so luong nhieu')
print('   - SMOTEENN co the giam samples vi loai bo noise')
print('   - Nen dung "none" hoac "smoteenn" de clean data')

print('\n2. CREDITCARD DATASET:')
print(f'   - Rat mat can bang ({ratio_credit:.0f}:1), SMOTE tang MANH Class 1')
print(f'   - SMOTE tang tu {credit_train_dist[1]:,} len {credit_smote_dist[1]:,} samples (+{credit_smote_dist[1]-credit_train_dist[1]:,})')
print('   - BAT BUOC phai dung SMOTE hoac SMOTEENN!')
print('='*100)
