"""Models module - All machine learning models"""

from .zoo import get_model, get_search_space

# Generation 4 Deep Learning models
try:
    from .pytorch_mlp import PyTorchMLPClassifier
    PYTORCH_MLP_AVAILABLE = True
except ImportError:
    PYTORCH_MLP_AVAILABLE = False
    PyTorchMLPClassifier = None

try:
    from .tabnet_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    TabNetClassifier = None

__all__ = [
    'get_model',
    'get_search_space',
    'PyTorchMLPClassifier',
    'TabNetClassifier',
    'PYTORCH_MLP_AVAILABLE',
    'TABNET_AVAILABLE',
]
