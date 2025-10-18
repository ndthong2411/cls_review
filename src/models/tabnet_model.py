"""
TabNet Model for Cardiovascular Disease Classification

TabNet: Attentive Interpretable Tabular Learning
- Attention-based feature selection
- Interpretability through attention masks
- Sequential multi-step architecture
- Self-supervised pre-training support
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional, Tuple
import warnings
import torch  # Needed for default optimizer reference

try:
    from pytorch_tabnet.tab_model import TabNetClassifier as _TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    _TabNetClassifier = None


class TabNetClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible TabNet Classifier wrapper

    Parameters:
    -----------
    n_d : int
        Width of the decision prediction layer
    n_a : int
        Width of the attention embedding for each mask
    n_steps : int
        Number of steps in the architecture (usually 3-10)
    gamma : float
        Coefficient for feature reusage in the masks (1.0-2.0)
    n_independent : int
        Number of independent Gated Linear Units layers at each step
    n_shared : int
        Number of shared Gated Linear Units at each step
    lambda_sparse : float
        Extra sparsity loss coefficient
    optimizer_fn : torch.optim
        Optimizer to use
    optimizer_params : dict
        Optimizer parameters
    scheduler_fn : torch.optim.lr_scheduler
        Learning rate scheduler
    scheduler_params : dict
        Scheduler parameters
    mask_type : str
        Either 'sparsemax' or 'entmax'
    seed : int
        Random seed for reproducibility
    verbose : int
        Verbosity level
    device_name : str
        Device to use: 'cuda' or 'auto'
    """

    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-4,
        momentum: float = 0.3,
        clip_value: float = 2.0,
        optimizer_fn=None,
        optimizer_params: dict = None,
        scheduler_fn=None,
        scheduler_params: dict = None,
        mask_type: str = 'sparsemax',
        seed: int = 42,
        verbose: int = 0,
        device_name: str = 'auto',
        max_epochs: int = 100,
        patience: int = 20,
        batch_size: int = 256,
        virtual_batch_size: int = 128,
    ):
        if not TABNET_AVAILABLE:
            raise ImportError(
                "pytorch-tabnet is not installed. "
                "Install it with: pip install pytorch-tabnet"
            )

        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.momentum = momentum
        self.clip_value = clip_value
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params or {'lr': 2e-2}
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params
        self.mask_type = mask_type
        self.seed = seed
        self.verbose = verbose
        self.device_name = device_name
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size

        self.model_ = None
        self.classes_ = None

    def fit(self, X, y, eval_set=None):
        """
        Fit TabNet model

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training labels
        eval_set : list of tuples, optional
            Validation set as [(X_val, y_val)]
        """
        import torch

        # Convert to numpy
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.classes_ = np.unique(y)

        # Setup optimizer if not provided
        if self.optimizer_fn is None:
            self.optimizer_fn = torch.optim.Adam

        # Create TabNet model
        self.model_ = _TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            lambda_sparse=self.lambda_sparse,
            momentum=self.momentum,
            clip_value=self.clip_value,
            optimizer_fn=self.optimizer_fn,
            optimizer_params=self.optimizer_params,
            scheduler_fn=self.scheduler_fn,
            scheduler_params=self.scheduler_params,
            mask_type=self.mask_type,
            seed=self.seed,
            verbose=self.verbose,
            device_name=self.device_name,
        )

        # Prepare evaluation set
        eval_set_formatted = None
        if eval_set is not None:
            X_val, y_val = eval_set[0]
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64)
            eval_set_formatted = [(X_val, y_val)]

        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=self.classes_,
            y=y
        )
        weights = {i: w for i, w in enumerate(class_weights)}

        # Fit model
        self.model_.fit(
            X_train=X,
            y_train=y,
            eval_set=eval_set_formatted,
            eval_metric=['auc', 'accuracy'],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            weights=weights,
            drop_last=False
        )

        return self

    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet")

        X = np.asarray(X, dtype=np.float32)
        return self.model_.predict_proba(X)

    def predict(self, X):
        """Predict class labels"""
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet")

        X = np.asarray(X, dtype=np.float32)
        return self.model_.predict(X)

    def get_feature_importance(self, X=None):
        """
        Get global feature importance

        Parameters:
        -----------
        X : array-like, optional
            Data to compute importance on. If None, uses training data importance.

        Returns:
        --------
        feature_importances_ : array
            Feature importance scores
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet")

        if X is not None:
            X = np.asarray(X, dtype=np.float32)
            importance = self.model_.feature_importances_
        else:
            importance = self.model_.feature_importances_

        return importance

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {
            'n_d': self.n_d,
            'n_a': self.n_a,
            'n_steps': self.n_steps,
            'gamma': self.gamma,
            'n_independent': self.n_independent,
            'n_shared': self.n_shared,
            'lambda_sparse': self.lambda_sparse,
            'momentum': self.momentum,
            'clip_value': self.clip_value,
            'optimizer_fn': self.optimizer_fn,
            'optimizer_params': self.optimizer_params,
            'scheduler_fn': self.scheduler_fn,
            'scheduler_params': self.scheduler_params,
            'mask_type': self.mask_type,
            'seed': self.seed,
            'verbose': self.verbose,
            'device_name': self.device_name,
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'batch_size': self.batch_size,
            'virtual_batch_size': self.virtual_batch_size,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
