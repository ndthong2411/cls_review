# Model Caching Guide

## Overview

The `full_comparison.py` script now includes an intelligent caching system that saves trained model results to avoid retraining the same configurations repeatedly. This can save hours of computation time when running experiments.

## How It Works

### Automatic Caching

When you run an experiment, the system:
1. **Checks cache**: Before training, looks for cached results with the same configuration
2. **Loads if found**: If a matching cache exists, loads results instantly (< 1 second)
3. **Trains if not found**: If no cache exists, trains the model normally
4. **Saves to cache**: After training, saves the results for future use

### Cache Key

Each experiment is uniquely identified by:
- Model name (e.g., `Gen3_LightGBM`)
- Scaler method (e.g., `standard`, `robust`, `none`)
- Imbalance handling (e.g., `smote`, `none`)
- Feature selection (e.g., `select_k_best_12`, `none`)

The system generates a hash from these parameters to create a unique cache filename.

## Usage

### Basic Usage (with cache)

```bash
python full_comparison.py
```

This runs all experiments and uses cached results when available.

### Run Without Cache

```bash
python full_comparison.py --no-cache
```

Forces retraining of all models, ignoring cached results.

### List Cached Experiments

```bash
python full_comparison.py --list-cache
```

Shows all cached experiments with their configurations and metrics.

### Clear Cache

```bash
python full_comparison.py --clear-cache
```

Deletes all cached experiments. Use this if you:
- Changed the dataset
- Modified preprocessing code
- Want to ensure fresh training

### Help

```bash
python full_comparison.py --help
```

## Cache Location

Cached experiments are stored in:
```
experiments/model_cache/
```

Each cache file is a `.pkl` file containing:
- Experiment results (all metrics)
- Configuration (model, scaler, imbalance, feature selection)
- Timestamp

## When to Clear Cache

Clear the cache when:
- ✅ Dataset has changed
- ✅ Preprocessing code has been modified
- ✅ Model hyperparameters have changed
- ✅ You want to ensure reproducibility with fresh training
- ✅ Cache is taking too much disk space

## Benefits

### Time Savings

Example: Running 90 experiments
- **Without cache**: ~45-90 minutes (all models trained)
- **With cache (second run)**: ~5-10 seconds (all loaded from cache)
- **With partial cache**: Variable (only new configs are trained)

### Iterative Development

Perfect for:
- Testing different model configurations
- Adding new models to comparison
- Debugging specific model configurations
- Interrupted training sessions (resume where you left off)

## Configuration

In `full_comparison.py`, you can configure:

```python
CONFIG = {
    'use_cache': True,  # Enable/disable caching
    'cache_dir': 'experiments/model_cache',  # Cache directory
    # ... other settings
}
```

## Technical Details

### Cache Structure

```
experiments/model_cache/
├── Gen1_LogisticRegression_a1b2c3d4e5f6.pkl
├── Gen3_LightGBM_x9y8z7w6v5u4.pkl
└── ...
```

### Cache File Contents

```python
{
    'results': {
        'model': 'Gen3_LightGBM',
        'generation': 3,
        'pr_auc': 0.7234,
        'sensitivity': 0.8123,
        # ... all metrics
    },
    'config': {
        'model': 'Gen3_LightGBM',
        'scaler': 'standard',
        'imbalance': 'smote',
        'feature_selection': 'none'
    },
    'timestamp': '2025-10-15T14:30:00'
}
```

## Best Practices

1. **Clear cache after major changes**: Dataset, preprocessing, or hyperparameters
2. **Keep cache for parameter tuning**: When only adjusting experiment matrix
3. **Use --list-cache regularly**: To see what's cached and avoid confusion
4. **Backup important caches**: If you want to preserve specific experiment runs

## Troubleshooting

### Cache Not Loading

If cached experiments aren't loading:
1. Check cache directory exists
2. Verify configuration matches exactly
3. Clear cache if you suspect corruption

### Cache Taking Too Much Space

```bash
# Check cache size
du -sh experiments/model_cache/

# Clear cache
python full_comparison.py --clear-cache
```

### Stale Cache Results

If you've modified code but cache still loads old results:
```bash
python full_comparison.py --clear-cache
python full_comparison.py
```

## Example Workflow

```bash
# First run - trains all models (takes ~1 hour)
python full_comparison.py

# Add a new model to comparison
# Edit full_comparison.py to add Gen4_NewModel

# Second run - only trains new model, loads rest from cache (takes ~2 minutes)
python full_comparison.py

# Check what's cached
python full_comparison.py --list-cache

# Dataset updated - clear cache and retrain
python full_comparison.py --clear-cache
python full_comparison.py
```

## Notes

- Cache files are platform-independent (can share between Windows/Linux/Mac)
- Cache includes only metrics, not the actual trained model objects
- For final model deployment, use the `best_model/` directory (separate from cache)
- Cache is safe to delete at any time - it's purely for convenience
