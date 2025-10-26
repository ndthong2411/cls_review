import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
cardio_df = pd.read_csv('experiments/full_comparison/cardio_train/full_comparison_20251019_081844.csv')
credit_df = pd.read_csv('experiments/full_comparison/creditcard/full_comparison_20251019_163508.csv')

print('='*120)
print('PHAN TICH ANH HUONG CUA CONFIG LEN PERFORMANCE')
print('='*120)

# ============= OVERVIEW =============
print('\n[1] TONG QUAN DATA:')
print(f'\nCardio Dataset: {len(cardio_df)} experiments')
print(f'Credit Dataset: {len(credit_df)} experiments')

print('\nCardio columns:', cardio_df.columns.tolist())
print('\nCredit columns:', credit_df.columns.tolist())

# ============= ANALYZE SCALING IMPACT =============
print('\n' + '='*120)
print('[2] ANH HUONG CUA SCALING (Phuong phap chuan hoa)')
print('='*120)

for dataset_name, df in [('CARDIO', cardio_df), ('CREDITCARD', credit_df)]:
    print(f'\n{dataset_name} DATASET:')
    print('-'*120)

    print('\nF1-Score by Scaler:')
    f1_by_scaler = df.groupby('scaler')['f1'].agg(['mean', 'std', 'max', 'min', 'count'])
    print(f1_by_scaler.sort_values('mean', ascending=False))

    print('\nROC-AUC by Scaler:')
    auc_by_scaler = df.groupby('scaler')['roc_auc'].agg(['mean', 'std', 'max', 'min', 'count'])
    print(auc_by_scaler.sort_values('mean', ascending=False))

    # Best scaler
    best_scaler = f1_by_scaler['mean'].idxmax()
    print(f'\n=> BEST SCALER: {best_scaler} (F1 = {f1_by_scaler.loc[best_scaler, "mean"]:.4f})')

# ============= ANALYZE FEATURE SELECTION IMPACT =============
print('\n' + '='*120)
print('[3] ANH HUONG CUA FEATURE SELECTION (Chon features quan trong)')
print('='*120)

for dataset_name, df in [('CARDIO', cardio_df), ('CREDITCARD', credit_df)]:
    print(f'\n{dataset_name} DATASET:')
    print('-'*120)

    print('\nF1-Score by Feature Selector:')
    f1_by_fs = df.groupby('feature_selection')['f1'].agg(['mean', 'std', 'max', 'min', 'count'])
    print(f1_by_fs.sort_values('mean', ascending=False))

    print('\nROC-AUC by Feature Selector:')
    auc_by_fs = df.groupby('feature_selection')['roc_auc'].agg(['mean', 'std', 'max', 'min', 'count'])
    print(auc_by_fs.sort_values('mean', ascending=False))

    best_fs = f1_by_fs['mean'].idxmax()
    print(f'\n=> BEST FEATURE SELECTOR: {best_fs} (F1 = {f1_by_fs.loc[best_fs, "mean"]:.4f})')

# ============= ANALYZE IMBALANCE IMPACT =============
print('\n' + '='*120)
print('[4] ANH HUONG CUA IMBALANCE HANDLING')
print('='*120)

for dataset_name, df in [('CARDIO', cardio_df), ('CREDITCARD', credit_df)]:
    print(f'\n{dataset_name} DATASET:')
    print('-'*120)

    print('\nF1-Score by Imbalance Handler:')
    f1_by_imb = df.groupby('imbalance')['f1'].agg(['mean', 'std', 'max', 'min', 'count'])
    print(f1_by_imb.sort_values('mean', ascending=False))

    print('\nROC-AUC by Imbalance Handler:')
    auc_by_imb = df.groupby('imbalance')['roc_auc'].agg(['mean', 'std', 'max', 'min', 'count'])
    print(auc_by_imb.sort_values('mean', ascending=False))

    # Precision/Recall trade-off
    print('\nPrecision/Recall by Imbalance Handler:')
    prec_recall = df.groupby('imbalance').agg({
        'precision': 'mean',
        'sensitivity': 'mean',
        'f1': 'mean'
    }).round(4)
    print(prec_recall.sort_values('f1', ascending=False))

    best_imb = f1_by_imb['mean'].idxmax()
    print(f'\n=> BEST IMBALANCE HANDLER: {best_imb} (F1 = {f1_by_imb.loc[best_imb, "mean"]:.4f})')

# ============= ANALYZE MODEL PERFORMANCE =============
print('\n' + '='*120)
print('[5] PERFORMANCE BY MODEL TYPE')
print('='*120)

for dataset_name, df in [('CARDIO', cardio_df), ('CREDITCARD', credit_df)]:
    print(f'\n{dataset_name} DATASET:')
    print('-'*120)

    print('\nTop 10 Models by F1-Score:')
    top_models = df.groupby('model')['f1'].agg(['mean', 'max', 'count']).sort_values('mean', ascending=False).head(10)
    print(top_models)

# ============= BEST CONFIGURATIONS =============
print('\n' + '='*120)
print('[6] TOP 10 BEST CONFIGURATIONS (Theo F1-Score)')
print('='*120)

for dataset_name, df in [('CARDIO', cardio_df), ('CREDITCARD', credit_df)]:
    print(f'\n{dataset_name} DATASET:')
    print('-'*120)

    top_configs = df.nlargest(10, 'f1')[
        ['model', 'scaler', 'feature_selection', 'imbalance',
         'f1', 'roc_auc', 'accuracy', 'train_time_sec']
    ]

    for idx, row in top_configs.iterrows():
        print(f"\nRank #{top_configs.index.get_loc(idx) + 1}:")
        print(f"  Model: {row['model']}")
        print(f"  Config: scaler={row['scaler']}, fs={row['feature_selection']}, imb={row['imbalance']}")
        print(f"  F1={row['f1']:.4f}, ROC-AUC={row['roc_auc']:.4f}, Acc={row['accuracy']:.4f}, Time={row['train_time_sec']:.2f}s")

# ============= INTERACTION ANALYSIS =============
print('\n' + '='*120)
print('[7] PHAN TICH TUONG TAC GIUA CAC CONFIG')
print('='*120)

for dataset_name, df in [('CARDIO', cardio_df), ('CREDITCARD', credit_df)]:
    print(f'\n{dataset_name} DATASET:')
    print('-'*120)

    # Scaler + Feature Selector
    print('\nF1-Score by (Scaler + Feature Selector):')
    scaler_fs = df.groupby(['scaler', 'feature_selection'])['f1'].mean().unstack().round(4)
    print(scaler_fs)

    # Find best combination
    best_combo = df.groupby(['scaler', 'feature_selection'])['f1'].mean().idxmax()
    best_f1 = df.groupby(['scaler', 'feature_selection'])['f1'].mean().max()
    print(f'\n=> BEST (Scaler + FS): {best_combo[0]} + {best_combo[1]} (F1 = {best_f1:.4f})')

    # Imbalance impact by generation
    print('\nImbalance Handler by Generation:')
    imb_by_gen = df.groupby(['generation', 'imbalance'])['f1'].mean().unstack().round(4)
    print(imb_by_gen)

# ============= STATISTICAL INSIGHTS =============
print('\n' + '='*120)
print('[8] THONG KE VA INSIGHTS')
print('='*120)

for dataset_name, df in [('CARDIO', cardio_df), ('CREDITCARD', credit_df)]:
    print(f'\n{dataset_name} DATASET:')
    print('-'*120)

    # Variance analysis
    print('\nVariance by config type:')
    print(f"  Scaler variance: {df.groupby('scaler')['f1'].mean().std():.4f}")
    print(f"  Feature Selector variance: {df.groupby('feature_selection')['f1'].mean().std():.4f}")
    print(f"  Imbalance Handler variance: {df.groupby('imbalance')['f1'].mean().std():.4f}")
    print(f"  Model variance: {df.groupby('model')['f1'].mean().std():.4f}")

    print('\n=> Config co anh huong LON NHAT: ', end='')
    variances = {
        'Scaler': df.groupby('scaler')['f1'].mean().std(),
        'Feature_Selector': df.groupby('feature_selection')['f1'].mean().std(),
        'Imbalance': df.groupby('imbalance')['f1'].mean().std(),
        'Model': df.groupby('model')['f1'].mean().std()
    }
    max_var = max(variances, key=variances.get)
    print(f"{max_var} (std = {variances[max_var]:.4f})")

print('\n' + '='*120)
print('HOAN THANH PHAN TICH!')
print('='*120)
