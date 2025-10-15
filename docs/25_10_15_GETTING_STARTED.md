# Getting Started Guide

## Complete Setup Walkthrough

### Step 1: Clone and Setup Environment

```powershell
# Navigate to the project directory
cd e:\thong\code\cls_review

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```powershell
python check_install.py
```

This will check:
- Python version (3.10+ recommended)
- All required packages
- Optional packages (XGBoost, LightGBM, CatBoost)
- PyTorch GPU availability
- Dataset presence
- Directory structure

### Step 3: Download Dataset

1. Visit: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
2. Download `cardio_train.csv`
3. Place in: `data/raw/cardio_train.csv`

Or see detailed instructions in `data/raw/README.md`

### Step 4: Run Quick Training

```powershell
python quickstart.py
```

**What this does:**
- Loads and preprocesses the CVD dataset
- Engineers features (age_years, BMI, pulse pressure, MAP)
- Trains 4-6 models with 5-fold cross-validation
- Applies SMOTE for class imbalance
- Computes comprehensive metrics (Recall, Precision, F1, PR-AUC, ROC-AUC)
- Saves results to `experiments/results_summary.csv`

**Expected time:** 5-15 minutes depending on your CPU

**Expected output:**
```
[1/6] Loading data...
   Dataset shape: (70000, 14)
   Positive class: 35021 (50.0%)

[2/6] Engineering features...
[3/6] Splitting data...
[4/6] Setting up models...
   ✓ XGBoost available
   ✓ LightGBM available
   ✓ CatBoost available

[5/6] Training 7 models with 5-fold CV...

  Training: Logistic Regression (Generation 1)
    Fold 1: PR-AUC=0.8234, Recall=0.8567
    ...
    ✓ Average PR-AUC: 0.8245 ± 0.0123

  Training: XGBoost (Generation 3)
    Fold 1: PR-AUC=0.9123, Recall=0.9234
    ...
    ✓ Average PR-AUC: 0.9145 ± 0.0089

[6/6] Saving results...
   ✓ Saved results to: experiments/results_summary.csv

Top 3 Models by PR-AUC:
  XGBoost              | Gen 3 | PR-AUC: 0.9145 | Recall: 0.9156
  LightGBM             | Gen 3 | PR-AUC: 0.9089 | Recall: 0.9078
  Random Forest        | Gen 2 | PR-AUC: 0.8867 | Recall: 0.8934
```

### Step 5: Launch Streamlit Demo

```powershell
streamlit run app.py
```

This opens an interactive web app at `http://localhost:8501` with:

**Tab 1: Data Explorer**
- Dataset statistics and sample data
- Target distribution pie chart
- Age distribution by CVD status
- Feature correlation heatmap

**Tab 2: Model Training**
- Configure preprocessing options
- Select models by generation
- Set cross-validation parameters
- Generate training commands

**Tab 3: Results Comparison**
- View trained model performance
- Compare metrics across generations
- Performance vs training time plots
- Detailed results table
- Filter by generation and metric

**Tab 4: Make Predictions**
- Input patient data
- Get CVD risk predictions
- (Available after full pipeline training)

## Advanced Usage

### Full Experiment Pipeline with Hydra

For comprehensive experiments with hyperparameter tuning:

```powershell
# Phase 1: Baseline exploration (1-2 hours)
python -m src.experiment.run_phase --phase=baseline

# Phase 2: Intermediate optimization (3-6 hours)
python -m src.experiment.run_phase --phase=intermediate

# Phase 3: Advanced tuning (6-24 hours)
python -m src.experiment.run_phase --phase=advanced
```

### Custom Configuration

Override any config parameter:

```powershell
python quickstart.py \
  --preprocessing.missing=knn \
  --preprocessing.scale=robust \
  --imbalance.method=smoteenn \
  --cv.n_splits=10
```

### View MLflow Experiments

```powershell
mlflow ui --backend-store-uri .\mlruns
```

Visit `http://localhost:5000` to see:
- All experiment runs
- Parameters and metrics
- Artifacts (models, plots)
- Compare runs side-by-side

## Troubleshooting

### Issue: "Dataset not found"
**Solution:** Download `cardio_train.csv` from Kaggle and place in `data/raw/`

### Issue: "Import error: xgboost/lightgbm/catboost"
**Solution:** These are optional. Install with:
```powershell
pip install xgboost lightgbm catboost
```

### Issue: PyTorch CUDA not available
**Solution:** For GPU support, uninstall PyTorch and reinstall with CUDA:
```powershell
pip uninstall torch torchvision
# Visit https://pytorch.org/get-started/locally/ for correct command
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "Permission denied" when activating venv
**Solution:** Run PowerShell as Administrator or set execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Streamlit won't start
**Solution:** Check if port 8501 is in use:
```powershell
streamlit run app.py --server.port 8502
```

## Performance Tips

### Speed up training:
1. **Use GPU**: Install CUDA-enabled PyTorch for MLP training
2. **Parallel jobs**: XGBoost, LightGBM, CatBoost use `n_jobs=-1` by default
3. **Reduce CV folds**: Use `--cv.n_splits=3` instead of 5
4. **Sample data**: For testing, use a subset: `df.sample(10000)`

### Save resources:
1. **Skip optional models**: Comment out XGBoost/LightGBM/CatBoost if not needed
2. **Reduce Optuna trials**: Use `--optuna.n_trials=20` instead of 100
3. **Use early stopping**: Already enabled for gradient boosting models

## What's Next?

After running quickstart.py and exploring the demo:

1. **Experiment with preprocessing**: Try different imputation, scaling, and imbalance methods
2. **Feature engineering**: Add domain-specific features in `src/data/dataset.py`
3. **Hyperparameter tuning**: Run full Optuna optimization for top models
4. **Custom models**: Add your own models to `src/models/zoo.py`
5. **Analysis**: Generate detailed reports with SHAP explanations
6. **Deployment**: Export best model for production use

## Getting Help

- **Documentation**: See `claude.md` for detailed methodology
- **Code structure**: Check `src/` for modular components
- **Examples**: Look at `quickstart.py` for simple usage patterns
- **Configuration**: Explore `src/configs/` for all options

## Contributing

Found a bug or want to add a feature?
1. Check existing issues
2. Create a new branch
3. Make your changes
4. Test with `check_install.py` and `quickstart.py`
5. Submit a pull request

---

Happy experimenting! 🚀
