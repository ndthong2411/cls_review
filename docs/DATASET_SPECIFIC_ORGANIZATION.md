# Dataset-Specific Organization Guide

## Overview

The `full_comparison.py` script now automatically organizes all outputs, logs, cache, and results by dataset name. This allows you to run experiments on different datasets (e.g., cardiovascular, creditcard fraud) without any conflicts.

## How It Works

When you run the script with a specific dataset, the system:

1. **Extracts the dataset name** from the file path (e.g., `cardio_train.csv` → `cardio_train`)
2. **Creates dataset-specific directories** for all outputs
3. **Uses dataset-specific cache** to avoid conflicts
4. **Generates dataset-specific log files** with clear naming

## Directory Structure

### Before (Old Behavior)
```
experiments/
├── logs/
│   └── training_20251016_120000.log          # All datasets mixed
├── full_comparison/
│   ├── full_comparison_20251016_120000.csv   # All results mixed
│   └── best_model/                           # Could be from any dataset
└── model_cache/
    └── *.pkl                                 # Cache conflicts!
```

### After (New Behavior)
```
experiments/
├── logs/
│   ├── cardio_train_20251016_120000.log     # ✅ Cardio logs
│   └── creditcard_20251016_130000.log       # ✅ Creditcard logs
├── full_comparison/
│   ├── cardio_train/                         # ✅ All cardio outputs
│   │   ├── full_comparison_20251016_120000.csv
│   │   └── best_model/
│   │       ├── best_model_20251016_120000.pkl
│   │       ├── metadata_20251016_120000.json
│   │       └── predict_20251016_120000.py
│   └── creditcard/                           # ✅ All creditcard outputs
│       ├── full_comparison_20251016_130000.csv
│       └── best_model/
│           ├── best_model_20251016_130000.pkl
│           ├── metadata_20251016_130000.json
│           └── predict_20251016_130000.py
└── model_cache/
    ├── cardio_train/                         # ✅ Cardio cache
    │   └── *.pkl
    └── creditcard/                           # ✅ Creditcard cache
        └── *.pkl
```

## Usage Examples

### Run on Cardiovascular Dataset
```bash
python full_comparison.py --data data/raw/cardio_train.csv
```

**Outputs:**
- Log: `experiments/logs/cardio_train_YYYYMMDD_HHMMSS.log`
- Results: `experiments/full_comparison/cardio_train/`
- Cache: `experiments/model_cache/cardio_train/`

### Run on Credit Card Fraud Dataset
```bash
python full_comparison.py --data data/raw/creditcard.csv
```

**Outputs:**
- Log: `experiments/logs/creditcard_YYYYMMDD_HHMMSS.log`
- Results: `experiments/full_comparison/creditcard/`
- Cache: `experiments/model_cache/creditcard/`

### Run Multiple Datasets (No Conflicts!)
```bash
# Run cardio in one terminal
python full_comparison.py --data data/raw/cardio_train.csv

# Run creditcard in another terminal (parallel execution OK!)
python full_comparison.py --data data/raw/creditcard.csv
```

## Benefits

### 1. **No Cache Conflicts**
Each dataset has its own cache directory, so trained models from one dataset won't be confused with another.

### 2. **Clear Organization**
Find all results for a specific dataset in one place:
```bash
experiments/full_comparison/creditcard/  # All creditcard results here
```

### 3. **Easy Comparison**
Compare results across datasets:
```bash
experiments/full_comparison/
├── cardio_train/
│   └── full_comparison_20251016_120000.csv
└── creditcard/
    └── full_comparison_20251016_130000.csv
```

### 4. **Parallel Execution**
Run experiments on different datasets simultaneously without any interference.

### 5. **Clean Logs**
Logs are clearly labeled by dataset:
```bash
experiments/logs/
├── cardio_train_20251016_120000.log    # Clear which dataset
├── creditcard_20251016_130000.log      # No confusion
```

## File Naming Convention

| Component | Format | Example |
|-----------|--------|---------|
| Log File | `{dataset_name}_{timestamp}.log` | `creditcard_20251016_120000.log` |
| Results CSV | `full_comparison_{timestamp}.csv` | `full_comparison_20251016_120000.csv` |
| Best Model | `best_model_{timestamp}.pkl` | `best_model_20251016_120000.pkl` |
| Cache Files | `{model}_{config_hash}.pkl` | `Gen1_KNN_abc123.pkl` |

## Cache Management Per Dataset

### List Cache for Specific Dataset
The cache is automatically separated by dataset. When you run:
```bash
python full_comparison.py --data data/raw/creditcard.csv --list-cache
```
It will only show cache for the creditcard dataset.

### Clear Cache for Specific Dataset
To clear cache for a specific dataset, delete that dataset's cache directory:
```bash
# Windows PowerShell
Remove-Item -Recurse -Force experiments/model_cache/creditcard/

# Linux/Mac
rm -rf experiments/model_cache/creditcard/
```

Or use the global clear (clears ALL datasets):
```bash
python full_comparison.py --clear-cache
```

## Dataset Auto-Detection

The script automatically detects the dataset type based on column names:

### Cardiovascular Dataset
- **Detected by:** `cardio` column present
- **Features:** Creates medical features (BMI, pulse pressure, etc.)
- **Target:** `cardio` column

### Credit Card Fraud Dataset
- **Detected by:** `Class` column present
- **Features:** Uses all V1-V28 + Time + Amount
- **Target:** `Class` column

### Generic Dataset
- **Detected by:** No specific columns
- **Features:** All columns except last
- **Target:** Last column

## Migration from Old Structure

If you have existing results in the old flat structure, you can organize them manually:

```bash
# Create dataset directories
mkdir -p experiments/full_comparison/cardio_train
mkdir -p experiments/full_comparison/creditcard

# Move cardio results
mv experiments/full_comparison/full_comparison_*_cardio*.csv experiments/full_comparison/cardio_train/

# Move creditcard results  
mv experiments/full_comparison/full_comparison_*_credit*.csv experiments/full_comparison/creditcard/
```

## Troubleshooting

### Q: My results are still in the old location
**A:** Make sure you're using the updated `full_comparison.py` script. The new organization is automatic.

### Q: Can I change the dataset-specific directory names?
**A:** Yes! The directory name comes from the dataset filename. If you rename `creditcard.csv` to `fraud_data.csv`, it will create `experiments/full_comparison/fraud_data/`.

### Q: How do I combine results from multiple runs?
**A:** All CSV results for a dataset are in the same directory. Use pandas to combine:
```python
import pandas as pd
from pathlib import Path

# Combine all cardio results
cardio_dir = Path('experiments/full_comparison/cardio_train')
all_results = pd.concat([
    pd.read_csv(f) for f in cardio_dir.glob('full_comparison_*.csv')
])
```

## See Also

- [Getting Started Guide](25_10_15_GETTING_STARTED.md)
- [Full Comparison Guide](25_10_15_FULL_COMPARISON_GUIDE.md)
- [Model Caching Guide](MODEL_CACHING_GUIDE.md)
