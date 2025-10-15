"""Custom transformers for preprocessing pipeline."""

from typing import Optional, Literal
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)
from category_encoders import TargetEncoder


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handle missing values based on strategy.
    
    Supports: delete, mean, median, knn, mice
    """
    
    def __init__(
        self,
        strategy: Literal['delete', 'mean', 'median', 'knn', 'mice'] = 'median',
        threshold: float = 0.2,
        knn_neighbors: int = 5
    ):
        self.strategy = strategy
        self.threshold = threshold
        self.knn_neighbors = knn_neighbors
        self.imputer_ = None
    
    def fit(self, X, y=None):
        if self.strategy == 'delete':
            return self
        elif self.strategy in ['mean', 'median']:
            self.imputer_ = SimpleImputer(strategy=self.strategy)
        elif self.strategy == 'knn':
            self.imputer_ = KNNImputer(n_neighbors=self.knn_neighbors)
        elif self.strategy == 'mice':
            self.imputer_ = IterativeImputer(random_state=42, max_iter=10)
        
        self.imputer_.fit(X)
        return self
    
    def transform(self, X):
        if self.strategy == 'delete':
            # Remove rows with any missing values
            if isinstance(X, pd.DataFrame):
                return X.dropna()
            else:
                mask = ~np.isnan(X).any(axis=1)
                return X[mask]
        else:
            return self.imputer_.transform(X)


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Detect and handle outliers.
    
    Supports: none, iqr_clip, zscore_clip
    """
    
    def __init__(
        self,
        method: Literal['none', 'iqr_clip', 'zscore_clip'] = 'none',
        iqr_factor: float = 1.5,
        zscore_threshold: float = 3.0
    ):
        self.method = method
        self.iqr_factor = iqr_factor
        self.zscore_threshold = zscore_threshold
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X, y=None):
        if self.method == 'none':
            return self
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        if self.method == 'iqr_clip':
            q1 = np.percentile(X_array, 25, axis=0)
            q3 = np.percentile(X_array, 75, axis=0)
            iqr = q3 - q1
            self.lower_bounds_ = q1 - self.iqr_factor * iqr
            self.upper_bounds_ = q3 + self.iqr_factor * iqr
        
        elif self.method == 'zscore_clip':
            mean = np.mean(X_array, axis=0)
            std = np.std(X_array, axis=0)
            self.lower_bounds_ = mean - self.zscore_threshold * std
            self.upper_bounds_ = mean + self.zscore_threshold * std
        
        return self
    
    def transform(self, X):
        if self.method == 'none':
            return X
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_clipped = np.clip(X_array, self.lower_bounds_, self.upper_bounds_)
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_clipped, columns=X.columns, index=X.index)
        return X_clipped


def get_scaler(name: Literal['standard', 'minmax', 'robust', 'none']):
    """Get scaler instance by name."""
    if name == 'standard':
        return StandardScaler()
    elif name == 'minmax':
        return MinMaxScaler()
    elif name == 'robust':
        return RobustScaler()
    elif name == 'none':
        return None
    else:
        raise ValueError(f"Unknown scaler: {name}")


def get_encoder(name: Literal['onehot', 'ordinal', 'target'], **kwargs):
    """Get encoder instance by name."""
    if name == 'onehot':
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    elif name == 'ordinal':
        return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    elif name == 'target':
        return TargetEncoder()
    else:
        raise ValueError(f"Unknown encoder: {name}")
