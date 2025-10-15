# Cardiovascular Disease Dataset - Download Instructions

## Dataset Information

**Source**: Kaggle  
**URL**: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset  
**File**: cardio_train.csv  
**Size**: ~70,000 records  
**License**: CC0: Public Domain

## Features

The dataset contains the following features:

### Objective Features:
- `age`: Age in days
- `height`: Height in cm
- `weight`: Weight in kg
- `gender`: 1 - women, 2 - men
- `ap_hi`: Systolic blood pressure
- `ap_lo`: Diastolic blood pressure

### Examination Features:
- `cholesterol`: 1 - normal, 2 - above normal, 3 - well above normal
- `gluc`: Glucose level - 1: normal, 2: above normal, 3: well above normal

### Subjective Features:
- `smoke`: Whether patient smokes (binary)
- `alco`: Alcohol intake (binary)
- `active`: Physical activity (binary)

### Target Variable:
- `cardio`: Presence or absence of cardiovascular disease (binary: 0 or 1)

## How to Download

### Option 1: Manual Download (Recommended)

1. Go to: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
2. Click "Download" button (requires Kaggle account)
3. Extract the ZIP file
4. Move `cardio_train.csv` to this project's `data/raw/` folder

### Option 2: Kaggle API (Advanced)

If you have Kaggle API configured:

```bash
# Install Kaggle API
pip install kaggle

# Configure API token (follow Kaggle documentation)
# Download dataset
kaggle datasets download -d sulianova/cardiovascular-disease-dataset

# Extract
unzip cardiovascular-disease-dataset.zip -d data/raw/
```

## After Download

Your directory structure should look like:

```
cls_review/
└── data/
    └── raw/
        └── cardio_train.csv  ← Place file here
```

Then run:

```powershell
python quickstart.py
```

## Dataset Statistics

- Total samples: ~70,000
- Features: 11 input features + 1 target
- Missing values: None (clean dataset)
- Class distribution: Nearly balanced (~50/50)
- Data quality: High quality, well-maintained dataset

## Citation

If you use this dataset, please cite:

```
Ulianova, S. (2019). Cardiovascular Disease dataset. 
Kaggle. https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
```
