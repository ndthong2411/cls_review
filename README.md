# cls_review - Cardiovascular Disease Prediction Pipeline

Progressive model comparison framework for cardiovascular disease prediction using the Kaggle CVD dataset.

## 🚀 Quick Start

```powershell
# 1. Setup environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Verify installation
python check_install.py

# 3. Download dataset from Kaggle
# Place cardio_train.csv in: data/raw/

# 4. Quick training (6+ models, 2-3 minutes)
python quickstart.py

# 5. Full comparison (11 models, 114 configs, 1-3 hours)
python full_comparison.py

# 6. Launch demo
streamlit run app.py
```

Visit: http://localhost:8501

## 📊 Features

- **4 Model Generations**: Baseline → Intermediate → Advanced → SOTA (11+ models)
- **Comprehensive Preprocessing**: Missing values, outliers, scaling, encoding
- **Imbalance Handling**: SMOTE, ADASYN, SMOTE-ENN, class weights
- **Cross-Validation**: 5-fold stratified with proper pipeline (no data leakage)
- **Interactive Demo**: Streamlit app with data exploration and model comparison
- **Medical Metrics**: PR-AUC, Sensitivity, Specificity, F1, ROC-AUC, MCC, NPV
- **Full Comparison**: 114 experiment configurations testing all combinations

## 📚 Documentation

All documentation is in the `docs/` folder:

- **[Getting Started Guide](docs/25_10_15_GETTING_STARTED.md)** - Complete setup walkthrough
- **[Full Comparison Guide](docs/25_10_15_FULL_COMPARISON_GUIDE.md)** - Run 114 experiments
- **[Project Summary](docs/25_10_15_PROJECT_SUMMARY.md)** - What's built and how to use
- **[Project Plan](docs/25_10_15_PROJECT_PLAN.md)** - Full methodology and implementation
- **[Dataset Info](docs/25_10_15_DATASET_INFO.md)** - Dataset download instructions
- **[Main README](docs/25_10_15_README.md)** - Detailed project documentation
- **[Index](docs/INDEX.md)** - Navigate all documentation

## 🎯 Model Performance (Expected)

### Quick Start (quickstart.py - 6 models)
| Model | Generation | PR-AUC | Sensitivity | F1 | Time |
|-------|-----------|--------|-------------|-----|------|
| XGBoost | 3 | 0.914 | 0.912 | 0.895 | 45s |
| LightGBM | 3 | 0.909 | 0.908 | 0.891 | 22s |
| CatBoost | 3 | 0.908 | 0.907 | 0.890 | 65s |
| Random Forest | 2 | 0.887 | 0.863 | 0.852 | 180s |
| Logistic Reg | 1 | 0.825 | 0.782 | 0.769 | 2s |

### Full Comparison (full_comparison.py - 11 models × 114 configs)
**Best Expected**: XGBoost with SMOTE-ENN + Robust Scaling
- **PR-AUC**: 0.92-0.96
- **Sensitivity**: 0.89-0.94
- **Specificity**: 0.86-0.91
- **F1-Score**: 0.88-0.92

## 📁 Project Structure

```
cls_review/
├── quickstart.py          # Quick training (6 models, 2-3 min)
├── full_comparison.py     # Comprehensive comparison (114 configs)
├── app.py                 # Streamlit demo app
├── check_install.py       # Installation checker
├── requirements.txt       # Dependencies
├── docs/                  # All documentation (180+ pages)
├── src/                   # Source code
│   ├── configs/          # Hydra configurations
│   ├── data/             # Data loading
│   ├── preprocessing/    # Preprocessing pipeline
│   ├── models/           # Model definitions
│   └── ...
├── data/
│   └── raw/              # Place cardio_train.csv here
└── experiments/          # Results and outputs
```

## 🔗 Links

- **Dataset**: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
- **Documentation**: [docs/](docs/)
- **Source Code**: [src/](src/)

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

**Built with**: Python • PyTorch • scikit-learn • XGBoost • Streamlit • MLflow
