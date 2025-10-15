"""Model zoo - wrapper classes for all models."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from omegaconf import DictConfig

from ..utils import get_logger

logger = get_logger(__name__)


def get_model(cfg: DictConfig, n_pos: Optional[int] = None, n_neg: Optional[int] = None):
    """
    Get model instance from configuration.
    
    Args:
        cfg: Model configuration
        n_pos: Number of positive samples (for class weight calculation)
        n_neg: Number of negative samples
    
    Returns:
        Model instance
    """
    model_name = cfg.model.name
    params = dict(cfg.model.params)
    
    # Calculate scale_pos_weight if needed
    if n_pos is not None and n_neg is not None:
        scale_pos_weight = n_neg / n_pos
    else:
        scale_pos_weight = 1.0
    
    if model_name == 'lr':
        return LogisticRegression(**params)
    
    elif model_name == 'dt':
        return DecisionTreeClassifier(**params)
    
    elif model_name == 'knn':
        return KNeighborsClassifier(**params)
    
    elif model_name == 'rf':
        return RandomForestClassifier(**params)
    
    elif model_name == 'gb':
        return GradientBoostingClassifier(**params)
    
    elif model_name == 'svm':
        return SVC(**params, probability=True)
    
    elif model_name == 'xgb':
        if 'scale_pos_weight' in params and params['scale_pos_weight'] == 1:
            params['scale_pos_weight'] = scale_pos_weight
        return xgb.XGBClassifier(**params)
    
    elif model_name == 'lgbm':
        if 'scale_pos_weight' in params and params['scale_pos_weight'] == 1:
            params['scale_pos_weight'] = scale_pos_weight
        return lgb.LGBMClassifier(**params)
    
    elif model_name == 'catboost':
        return cb.CatBoostClassifier(**params, verbose=False)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_search_space(model_name: str, cfg: DictConfig) -> Dict[str, Any]:
    """
    Get Optuna search space for a model.
    
    Args:
        model_name: Model name
        cfg: Configuration with optuna.params
    
    Returns:
        Dictionary mapping param names to Optuna suggest functions
    """
    search_space = {}
    
    if not hasattr(cfg.model, 'optuna') or not hasattr(cfg.model.optuna, 'params'):
        return search_space
    
    for param_name, param_config in cfg.model.optuna.params.items():
        param_type = param_config.get('type')
        
        if param_type == 'int':
            search_space[param_name] = {
                'type': 'int',
                'low': param_config.low,
                'high': param_config.high,
                'step': param_config.get('step', 1)
            }
        elif param_type == 'uniform':
            search_space[param_name] = {
                'type': 'uniform',
                'low': param_config.low,
                'high': param_config.high
            }
        elif param_type == 'loguniform':
            search_space[param_name] = {
                'type': 'loguniform',
                'low': param_config.low,
                'high': param_config.high
            }
        elif param_type == 'categorical':
            search_space[param_name] = {
                'type': 'categorical',
                'choices': param_config.choices
            }
    
    return search_space
