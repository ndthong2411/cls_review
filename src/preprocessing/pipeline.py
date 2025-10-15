"""Build sklearn Pipeline for preprocessing."""

from typing import Dict, List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from omegaconf import DictConfig

from .transformers import (
    MissingValueHandler,
    OutlierHandler,
    get_scaler,
    get_encoder
)
from ..utils import get_logger

logger = get_logger(__name__)


def build_preprocessing_pipeline(
    cfg: DictConfig,
    feature_groups: Dict[str, List[str]]
) -> Pipeline:
    """
    Build preprocessing pipeline from configuration.
    
    Args:
        cfg: Hydra configuration
        feature_groups: Dict with 'numeric_continuous', 'categorical_nominal', etc.
    
    Returns:
        sklearn Pipeline
    """
    numeric_features = feature_groups.get('numeric_continuous', [])
    categorical_nominal = feature_groups.get('categorical_nominal', [])
    categorical_ordinal = feature_groups.get('categorical_ordinal', [])
    binary_features = feature_groups.get('binary', [])
    
    # Build numeric pipeline
    numeric_steps = []
    
    # Missing values
    if cfg.preprocessing.missing != 'none':
        numeric_steps.append((
            'missing',
            MissingValueHandler(strategy=cfg.preprocessing.missing)
        ))
    
    # Outliers
    if cfg.preprocessing.outliers != 'none':
        numeric_steps.append((
            'outliers',
            OutlierHandler(
                method=cfg.preprocessing.outliers,
                iqr_factor=cfg.preprocessing.outlier_params.iqr_factor,
                zscore_threshold=cfg.preprocessing.outlier_params.zscore_threshold
            )
        ))
    
    # Scaling
    if cfg.preprocessing.scale != 'none':
        scaler = get_scaler(cfg.preprocessing.scale)
        if scaler is not None:
            numeric_steps.append(('scaler', scaler))
    
    # Build categorical pipeline
    categorical_steps = []
    
    # Missing values for categorical
    if cfg.preprocessing.missing != 'delete':
        categorical_steps.append((
            'missing',
            MissingValueHandler(strategy='median')  # Use median for categorical (encoded as numbers)
        ))
    
    # Encoding
    encoder = get_encoder(cfg.preprocessing.encode)
    categorical_steps.append(('encoder', encoder))
    
    # Combine all feature types
    transformers = []
    
    if numeric_features:
        transformers.append((
            'numeric',
            Pipeline(numeric_steps) if numeric_steps else 'passthrough',
            numeric_features
        ))
    
    if categorical_nominal:
        transformers.append((
            'categorical_nominal',
            Pipeline(categorical_steps) if categorical_steps else 'passthrough',
            categorical_nominal
        ))
    
    if categorical_ordinal:
        # Ordinal uses ordinal encoder
        ordinal_steps = categorical_steps.copy()
        ordinal_steps[-1] = ('encoder', get_encoder('ordinal'))
        transformers.append((
            'categorical_ordinal',
            Pipeline(ordinal_steps) if ordinal_steps else 'passthrough',
            categorical_ordinal
        ))
    
    if binary_features:
        # Binary features usually don't need encoding
        transformers.append((
            'binary',
            'passthrough',
            binary_features
        ))
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Drop features not specified
        verbose_feature_names_out=False
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    
    logger.info(f"Built preprocessing pipeline with {len(transformers)} feature groups")
    
    return pipeline
