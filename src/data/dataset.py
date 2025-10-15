"""Dataset loading, validation, and feature engineering for CVD data."""

from pathlib import Path
from typing import Tuple, Dict, Optional
import pandas as pd
import numpy as np
from ..utils import get_logger

logger = get_logger(__name__)


def load_cardio_data(
    file_path: str | Path,
    parse_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Load cardiovascular disease dataset from CSV.
    
    Args:
        file_path: Path to CSV file
        parse_config: Optional parsing configuration
    
    Returns:
        Loaded DataFrame
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            f"Please download from Kaggle and place in {file_path.parent}"
        )
    
    # Default parsing config
    default_config = {
        'sep': ';',
        'skipinitialspace': True,
        'na_values': ['', 'NA', 'N/A', 'null', 'NaN']
    }
    
    if parse_config:
        default_config.update(parse_config)
    
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, **default_config)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


def validate_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate dataset schema and data quality.
    
    Expected columns for Kaggle CVD dataset:
    - id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc,
      smoke, alco, active, cardio
    
    Args:
        df: Input DataFrame
    
    Returns:
        Tuple of (validated DataFrame, validation report dict)
    """
    report = {
        'original_shape': df.shape,
        'missing_values': {},
        'outliers': {},
        'data_types': {},
        'issues': []
    }
    
    # Expected columns
    expected_cols = [
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio'
    ]
    
    # Check for required columns
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove 'id' column if exists
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        logger.info("Removed 'id' column")
    
    # Check missing values
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df) * 100
        report['missing_values'][col] = {
            'count': int(missing_count),
            'percentage': round(missing_pct, 2)
        }
        
        if missing_pct > 0:
            logger.warning(f"Column '{col}' has {missing_pct:.2f}% missing values")
    
    # Data type validation
    numeric_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    categorical_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
    
    for col in numeric_cols:
        if col in df.columns:
            report['data_types'][col] = str(df[col].dtype)
            if not pd.api.types.is_numeric_dtype(df[col]):
                report['issues'].append(f"{col} should be numeric but is {df[col].dtype}")
    
    # Check for obvious outliers (medical domain knowledge)
    outlier_checks = {
        'age': (0, 365 * 120),  # 0-120 years in days
        'height': (100, 250),  # cm
        'weight': (30, 300),  # kg
        'ap_hi': (50, 300),  # systolic blood pressure
        'ap_lo': (30, 200),  # diastolic blood pressure
    }
    
    for col, (min_val, max_val) in outlier_checks.items():
        if col in df.columns:
            outliers = ((df[col] < min_val) | (df[col] > max_val)).sum()
            outlier_pct = outliers / len(df) * 100
            report['outliers'][col] = {
                'count': int(outliers),
                'percentage': round(outlier_pct, 2),
                'range': (min_val, max_val)
            }
            
            if outlier_pct > 1:
                logger.warning(f"Column '{col}' has {outlier_pct:.2f}% outliers outside [{min_val}, {max_val}]")
    
    # Check target distribution
    if 'cardio' in df.columns:
        target_dist = df['cardio'].value_counts()
        report['target_distribution'] = target_dist.to_dict()
        imbalance_ratio = target_dist.max() / target_dist.min()
        logger.info(f"Target distribution: {target_dist.to_dict()}, imbalance ratio: {imbalance_ratio:.2f}")
    
    report['final_shape'] = df.shape
    
    return df, report


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from existing ones.
    
    Creates:
    - age_years: Convert age from days to years
    - bmi: Body Mass Index
    - pulse_pressure: Difference between systolic and diastolic BP
    - map: Mean Arterial Pressure
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Age in years (original is in days)
    if 'age' in df.columns:
        df['age_years'] = df['age'] / 365.25
        logger.info("Created 'age_years' feature")
    
    # BMI
    if 'weight' in df.columns and 'height' in df.columns:
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        logger.info("Created 'bmi' feature")
    
    # Pulse pressure (systolic - diastolic)
    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        logger.info("Created 'pulse_pressure' feature")
    
    # Mean Arterial Pressure
    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
        logger.info("Created 'map' (Mean Arterial Pressure) feature")
    
    # Age groups (clinical relevance)
    if 'age_years' in df.columns:
        df['age_group'] = pd.cut(
            df['age_years'],
            bins=[0, 40, 50, 60, 70, 120],
            labels=['<40', '40-50', '50-60', '60-70', '70+']
        )
        logger.info("Created 'age_group' feature")
    
    # BMI categories
    if 'bmi' in df.columns:
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['underweight', 'normal', 'overweight', 'obese']
        )
        logger.info("Created 'bmi_category' feature")
    
    # Hypertension flags (based on clinical thresholds)
    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        df['hypertension'] = ((df['ap_hi'] >= 140) | (df['ap_lo'] >= 90)).astype(int)
        logger.info("Created 'hypertension' binary feature")
    
    return df


def get_feature_groups(df: pd.DataFrame) -> Dict[str, list]:
    """
    Categorize features into groups for preprocessing.
    
    Returns:
        Dictionary with feature groups
    """
    numeric_continuous = ['age', 'age_years', 'height', 'weight', 'bmi', 
                         'ap_hi', 'ap_lo', 'pulse_pressure', 'map']
    
    categorical_nominal = ['gender']
    
    categorical_ordinal = ['cholesterol', 'gluc']  # 1: normal, 2: above normal, 3: well above normal
    
    binary = ['smoke', 'alco', 'active', 'hypertension']
    
    categorical_derived = ['age_group', 'bmi_category']
    
    target = ['cardio']
    
    # Filter only existing columns
    return {
        'numeric_continuous': [c for c in numeric_continuous if c in df.columns],
        'categorical_nominal': [c for c in categorical_nominal if c in df.columns],
        'categorical_ordinal': [c for c in categorical_ordinal if c in df.columns],
        'binary': [c for c in binary if c in df.columns],
        'categorical_derived': [c for c in categorical_derived if c in df.columns],
        'target': [c for c in target if c in df.columns]
    }
