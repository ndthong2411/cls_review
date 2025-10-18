# Quick Reference: Feature Selection Methods

## 📋 5 Methods Available

| # | Name | Algorithm | K | Speed | Detects |
|---|------|-----------|---|-------|---------|
| 1 | `none` | - | All | ⚡⚡⚡ | - |
| 2 | `select_k_best_5` | ANOVA F-test | 5 | ⚡⚡⚡ | Linear |
| 3 | `select_k_best_12` | ANOVA F-test | 12 | ⚡⚡⚡ | Linear |
| 4 | `mutual_info_5` | Mutual Information | 5 | ⚡⚡ | Linear + Non-linear |
| 5 | `mutual_info_12` | Mutual Information | 12 | ⚡⚡ | Linear + Non-linear |

## 🎯 Quick Decision Guide

```
Need speed? → Use ANOVA (select_k_best_*)
Need power? → Use Mutual Info (mutual_info_*)

Linear data? → ANOVA good enough
Non-linear? → Must use Mutual Info

Large dataset (>100k)? → ANOVA faster
Small dataset (<50k)? → Both OK

Simple models? → ANOVA
Complex models? → Mutual Info
```

## 📊 Experiment Count

- **Per model (needs scaling):** 2 scalers × 3 imbalance × 5 feat_sel = 30 experiments
- **Per model (no scaling):** 1 scaler × 3 imbalance × 5 feat_sel = 15 experiments
- **Total:** (5×30) + (8×15) = **270 experiments**

## ⏱️ Time

- **Baseline (2 methods):** ~81 min (1.4 hrs)
- **+ K=5 (3 methods):** ~122 min (2.0 hrs)
- **+ Mutual Info (5 methods):** ~203 min (3.4 hrs)

## 🚀 Commands

```bash
# Run all experiments
python full_comparison.py --data data/raw/cardio_train.csv

# With cache (only train new experiments)
python full_comparison.py --data data/raw/cardio_train.csv  # Default

# Without cache (train all)
python full_comparison.py --data data/raw/cardio_train.csv --no-cache
```

## 📁 Files Modified

- `full_comparison.py` (lines 48, 93-98, 1141)

## 📚 Documentation

- `docs/MUTUAL_INFO_FEATURE_SELECTION.md` - Full details
- `docs/FEATURE_SELECTION_K5_UPDATE.md` - K=5 update
- `docs/DATASET_SPECIFIC_ORGANIZATION.md` - Dataset organization
