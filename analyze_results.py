"""Quick analysis of model comparison results"""
import pandas as pd

# Load results
df = pd.read_csv('experiments/full_comparison/full_comparison_20251016_094856.csv')

print("="*80)
print("TOP 10 MODELS BY PR-AUC")
print("="*80)
top10 = df.nlargest(10, 'pr_auc')[['model', 'generation', 'scaler', 'imbalance', 'pr_auc', 'sensitivity', 'specificity', 'f1']]
for idx, row in top10.iterrows():
    print(f"\n{row['model']:30s} (Gen {row['generation']})")
    print(f"  Config: {row['scaler']:8s} | {row['imbalance']:12s}")
    print(f"  PR-AUC: {row['pr_auc']:.4f} | Sens: {row['sensitivity']:.4f} | Spec: {row['specificity']:.4f} | F1: {row['f1']:.4f}")

print("\n" + "="*80)
print("BEST MODEL PER GENERATION")
print("="*80)
for gen in sorted(df['generation'].unique()):
    gen_df = df[df['generation'] == gen]
    best = gen_df.nlargest(1, 'pr_auc').iloc[0]
    worst = gen_df.nsmallest(1, 'pr_auc').iloc[0]
    avg_prauc = gen_df['pr_auc'].mean()
    
    print(f"\nGeneration {gen}:")
    print(f"  Best:  {best['model']:30s} | PR-AUC: {best['pr_auc']:.4f} ({best['scaler']}/{best['imbalance']})")
    print(f"  Worst: {worst['model']:30s} | PR-AUC: {worst['pr_auc']:.4f} ({worst['scaler']}/{worst['imbalance']})")
    print(f"  Average PR-AUC: {avg_prauc:.4f}")
    print(f"  Models in Gen {gen}: {gen_df['model'].nunique()}")

print("\n" + "="*80)
print("GENERATION STATISTICS")
print("="*80)
stats = df.groupby('generation').agg({
    'pr_auc': ['mean', 'std', 'min', 'max'],
    'roc_auc': ['mean', 'max'],
    'f1': ['mean', 'max'],
    'train_time_sec': ['mean', 'sum']
}).round(4)
print(stats)

print("\n" + "="*80)
print("MODEL COMPARISON (Average across all configs)")
print("="*80)
model_stats = df.groupby('model').agg({
    'pr_auc': 'mean',
    'roc_auc': 'mean',
    'f1': 'mean',
    'train_time_sec': 'mean',
    'generation': 'first'
}).sort_values('pr_auc', ascending=False)
print(model_stats.to_string())
