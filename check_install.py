"""
Installation Verification Script

Checks if all required dependencies are installed correctly.
"""

import sys
from pathlib import Path

print("="*70)
print("CHECKING INSTALLATION")
print("="*70)

# Check Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    print("  ⚠ Warning: Python 3.10+ recommended")

# Check required packages
required_packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',
    'imblearn': 'imbalanced-learn',
    'optuna': 'optuna',
    'mlflow': 'mlflow',
    'hydra': 'hydra-core',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'shap': 'shap',
    'streamlit': 'streamlit',
    'plotly': 'plotly',
    'torch': 'PyTorch',
}

optional_packages = {
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'catboost': 'CatBoost',
}

print("\n[Required Packages]")
missing = []
for module, name in required_packages.items():
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} - MISSING")
        missing.append(name)

print("\n[Optional Packages - Advanced Models]")
for module, name in optional_packages.items():
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ⚠ {name} - Not installed (optional)")

if missing:
    print(f"\n❌ Missing required packages: {', '.join(missing)}")
    print("\nInstall with:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Check PyTorch GPU availability
print("\n[PyTorch Configuration]")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("  Running on CPU (for GPU support, install CUDA-enabled PyTorch)")
except Exception as e:
    print(f"  Error checking PyTorch: {e}")

# Check data directory
print("\n[Data Directory]")
data_path = Path("data/raw/cardio_train.csv")
if data_path.exists():
    import pandas as pd
    df = pd.read_csv(data_path, sep=';', nrows=5)
    print(f"  ✓ Dataset found: {data_path}")
    print(f"  ✓ Can read CSV (detected {len(df.columns)} columns)")
else:
    print(f"  ✗ Dataset NOT found at: {data_path}")
    print("\n  Download instructions:")
    print("    1. Go to: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset")
    print("    2. Download 'cardio_train.csv'")
    print(f"    3. Place it in: {data_path.parent.absolute()}")

# Check output directories
print("\n[Output Directories]")
for dir_name in ['experiments', 'experiments/reports', 'experiments/figures']:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"  ✓ {dir_name}/")
    else:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {dir_name}/")

print("\n" + "="*70)
if not missing and data_path.exists():
    print("✓ ALL CHECKS PASSED - Ready to run!")
    print("\nNext steps:")
    print("  1. Quick training:  python quickstart.py")
    print("  2. Launch demo:     streamlit run app.py")
elif not missing:
    print("✓ Installation OK - Please download dataset")
    print("\nSee: data/raw/README.md for download instructions")
else:
    print("❌ Please install missing packages first")
    print("\nRun: pip install -r requirements.txt")
print("="*70)
