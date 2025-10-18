"""
PyTorch MLP Model for Cardiovascular Disease Classification

Advanced deep learning model with:
- Customizable architecture (hidden layers, dropout, batch norm)
- GPU acceleration support
- Early stopping
- Learning rate scheduling
- Sklearn-compatible API
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from typing import Optional, Tuple, List
import warnings
from tqdm import tqdm


class MLPNet(nn.Module):
    """Multi-layer Perceptron network"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rates: List[float] = [0.3, 0.2, 0.1],
        use_batch_norm: bool = True
    ):
        super(MLPNet, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.ReLU())

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer (logits). Apply sigmoid only at inference.
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible PyTorch MLP Classifier

    Parameters:
    -----------
    hidden_dims : list of int
        Number of units in each hidden layer
    dropout_rates : list of float
        Dropout rate for each hidden layer
    use_batch_norm : bool
        Whether to use batch normalization
    learning_rate : float
        Learning rate for optimizer
    batch_size : int
        Training batch size
    epochs : int
        Maximum number of training epochs
    optimizer_name : str
        Optimizer to use: 'adam', 'adamw', 'sgd'
    weight_decay : float
        L2 regularization coefficient
    scheduler : str or None
        Learning rate scheduler: 'plateau', 'cosine', 'step', None
    early_stopping_patience : int
        Number of epochs without improvement before stopping
    class_weight : str or None
        Class weighting strategy: 'balanced', None
    device : str or None
        Device to use: 'cuda', 'cpu', or None (auto-detect)
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print training progress
    """

    def __init__(
        self,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rates: List[float] = [0.3, 0.2, 0.1],
        use_batch_norm: bool = True,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 100,
        optimizer_name: str = 'adam',
        weight_decay: float = 1e-5,
        scheduler: Optional[str] = 'plateau',
        early_stopping_patience: int = 20,
        class_weight: Optional[str] = 'balanced',
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False
    ):
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.class_weight = class_weight
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        # Will be set during fit
        self.model_ = None
        self.input_dim_ = None
        self.classes_ = None
        self.device_ = None
        self.best_epoch_ = None
        self.best_score_ = None

    def _setup_device(self):
        """Setup computation device"""
        if self.device is not None:
            self.device_ = torch.device(self.device)
        else:
            self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.verbose:
            print(f"Using device: {self.device_}")

    def _setup_model(self, input_dim: int):
        """Initialize model architecture"""
        self.input_dim_ = input_dim
        self.model_ = MLPNet(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rates=self.dropout_rates,
            use_batch_norm=self.use_batch_norm
        ).to(self.device_)

        if self.verbose:
            total_params = sum(p.numel() for p in self.model_.parameters())
            trainable_params = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
            print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")

    def _setup_optimizer(self):
        """Initialize optimizer"""
        if self.optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(
                self.model_.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model_.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model_.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        return optimizer

    def _setup_scheduler(self, optimizer):
        """Initialize learning rate scheduler"""
        if self.scheduler is None:
            return None

        if self.scheduler.lower() == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=self.verbose
            )
        elif self.scheduler.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs
            )
        elif self.scheduler.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")

        return scheduler

    def _setup_criterion(self, y_train):
        """Initialize loss function with class weights if needed"""
        if self.class_weight == 'balanced':
            # Calculate class weights
            n_samples = len(y_train)
            n_classes = 2
            class_counts = np.bincount(y_train)
            # Avoid division by zero if any class missing
            class_counts = np.maximum(class_counts, 1)
            weights = n_samples / (n_classes * class_counts)
            # Weight for positive class (class 1)
            pos_weight = torch.tensor([weights[1] / weights[0]], dtype=torch.float32).to(self.device_)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            if self.verbose:
                print(f"Using class weights (pos_weight): {pos_weight.item():.4f}")
        else:
            criterion = nn.BCEWithLogitsLoss()

        return criterion

    def fit(self, X, y, eval_set=None):
        """
        Fit the model to training data

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training labels
        eval_set : list of tuples, optional
            Validation set as [(X_val, y_val)]
        """
        # Set random seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # Store classes
        self.classes_ = np.unique(y)

        # Setup device
        self._setup_device()

        # Create validation split if not provided
        if eval_set is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, random_state=self.random_state, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = eval_set[0]
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64)

        # Setup model
        self._setup_model(X_train.shape[1])

        # Setup optimizer and criterion
        optimizer = self._setup_optimizer()
        scheduler = self._setup_scheduler(optimizer)
        criterion = self._setup_criterion(y_train)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True  # Fix: Drop last incomplete batch to avoid BatchNorm error
        )

        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Training loop
        best_val_score = -np.inf
        patience_counter = 0
        self.best_epoch_ = 0

        # Progress bar for epochs
        epoch_pbar = tqdm(range(self.epochs), desc="Training PyTorch MLP", unit="epoch")
        
        for epoch in epoch_pbar:
            # Training phase
            self.model_.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device_)
                batch_y = batch_y.to(self.device_).unsqueeze(1)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                # BCEWithLogitsLoss expects logits
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            self.model_.eval()
            val_loss = 0.0
            val_preds = []
            val_true = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device_)
                    batch_y = batch_y.to(self.device_).unsqueeze(1)

                    outputs = self.model_(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()
                    # Convert logits to probabilities for metrics
                    val_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                    val_true.extend(batch_y.cpu().numpy().flatten())

            val_loss /= len(val_loader)

            # Calculate validation score (ROC-AUC)
            try:
                val_score = roc_auc_score(val_true, val_preds)
            except:
                val_score = 0.0

            # Learning rate scheduling
            if scheduler is not None:
                if self.scheduler.lower() == 'plateau':
                    scheduler.step(val_score)
                else:
                    scheduler.step()

            # Update progress bar
            lr = optimizer.param_groups[0]['lr']
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_auc': f'{val_score:.4f}',
                'lr': f'{lr:.6f}'
            })

            # Print progress (less frequent to avoid clutter)
            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"\nEpoch {epoch+1}/{self.epochs}: "
                      f"Train Loss={train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, "
                      f"Val ROC-AUC={val_score:.4f}, "
                      f"LR={lr:.6f}")

            # Early stopping
            if val_score > best_val_score:
                best_val_score = val_score
                self.best_epoch_ = epoch + 1
                self.best_score_ = val_score
                patience_counter = 0

                # Save best model state
                self.best_state_dict_ = self.model_.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if hasattr(self, 'best_state_dict_'):
            self.model_.load_state_dict(self.best_state_dict_)

        if self.verbose:
            print(f"\nTraining completed!")
            print(f"Best epoch: {self.best_epoch_}")
            print(f"Best validation ROC-AUC: {self.best_score_:.4f}")

        return self

    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet")

        X = np.asarray(X, dtype=np.float32)

        self.model_.eval()

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device_)
                logits = self.model_(batch_X)
                probs = torch.sigmoid(logits)
                predictions.extend(probs.cpu().numpy().flatten())

        predictions = np.array(predictions)

        # Return probabilities for both classes
        return np.vstack([1 - predictions, predictions]).T

    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {
            'hidden_dims': self.hidden_dims,
            'dropout_rates': self.dropout_rates,
            'use_batch_norm': self.use_batch_norm,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'optimizer_name': self.optimizer_name,
            'weight_decay': self.weight_decay,
            'scheduler': self.scheduler,
            'early_stopping_patience': self.early_stopping_patience,
            'class_weight': self.class_weight,
            'device': self.device,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
