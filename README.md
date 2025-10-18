# cls_review - Cardiovascular Disease Prediction Pipeline

Progressive model comparison framework for cardiovascular disease prediction using the Kaggle CVD dataset.

## 🚀 Quick Start

```powershell
# 1. Setup environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run full comparison training
python full_comparison.py

# 3. 🎨 Launch Interactive Dashboard (NEW!)
streamlit run app_streamlit.py

# 4. (Optional) Analyze results in terminal
python analyze_results.py

# 5. Cache management
python full_comparison.py --list-cache   # View cached experiments
python full_comparison.py --clear-cache  # Clear cache
```

**Dashboard**: http://localhost:8501  
**Docs**: See `QUICK_COMMANDS.md` for all commands

## 📊 Features

- **🎨 Interactive Dashboard**: Streamlit web app with visualizations, filters, comparisons
- **4 Model Generations**: Baseline → Intermediate → Advanced (Gen1-Gen4)
- **13 Models**: LogReg, DT, KNN, RF, ET, GB, SVM, MLP, XGBoost, LightGBM, CatBoost, PyTorch MLP, TabNet
- **Comprehensive Preprocessing**: Scaling, imbalance handling, feature selection
- **108 Experiments**: All preprocessing combinations tested
- **GPU Acceleration**: XGBoost, LightGBM, CatBoost on CUDA
- **Smart Caching**: Avoid retraining, saves hours!
- **Medical Metrics**: PR-AUC (primary), Sensitivity, Specificity, F1, ROC-AUC, MCC, NPV
- **Training Logs**: Auto-saved to `experiments/logs/`

## 📚 Documentation

- **[Quick Commands](QUICK_COMMANDS.md)** - ⭐ Bảng lệnh tổng hợp (START HERE!)
- **[Streamlit Dashboard Guide](docs/STREAMLIT_DASHBOARD_GUIDE.md)** - 🎨 Dashboard usage
- **[Logging Guide](docs/LOGGING_GUIDE.md)** - Training logs location
- **[Changelog](docs/CHANGELOG_OCT16.md)** - Recent fixes & improvements
- **[Index](docs/INDEX.md)** - All documentation index

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
