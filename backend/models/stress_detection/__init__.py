"""
Stress Detection Models Package
- LSTM models for physiological signals
- CNN models for facial images
- Hybrid multimodal models
- Baseline models for comparison
"""

from .lstm_model import (
    LSTMStressDetector,
    LSTMStressRegressor,
    create_lstm_classifier,
    create_lstm_regressor
)

from .cnn_model import (
    SimpleCNNStressDetector,
    ResNetStressDetector,
    EfficientNetStressDetector,
    CNNFeatureExtractor,
    create_simple_cnn,
    create_resnet_detector,
    create_feature_extractor
)

from .hybrid_model import (
    HybridStressDetector,
    PhysiologicalBranch,
    FacialBranch,
    MultimodalFusion,
    create_hybrid_detector,
    create_physio_branch,
    create_facial_branch
)

from .baseline_models import (
    SimpleMLPBaseline,
    ConvolutionalBaseline,
    SKLearnWrapper,
    EnsembleBaseline,
    create_mlp_baseline,
    create_cnn1d_baseline,
    create_random_forest,
    create_svm,
    create_logistic_regression,
    create_ensemble
)

__all__ = [
    # LSTM
    'LSTMStressDetector',
    'LSTMStressRegressor',
    'create_lstm_classifier',
    'create_lstm_regressor',
    
    # CNN
    'SimpleCNNStressDetector',
    'ResNetStressDetector',
    'EfficientNetStressDetector',
    'CNNFeatureExtractor',
    'create_simple_cnn',
    'create_resnet_detector',
    'create_feature_extractor',
    
    # Hybrid
    'HybridStressDetector',
    'PhysiologicalBranch',
    'FacialBranch',
    'MultimodalFusion',
    'create_hybrid_detector',
    'create_physio_branch',
    'create_facial_branch',
    
    # Baseline
    'SimpleMLPBaseline',
    'ConvolutionalBaseline',
    'SKLearnWrapper',
    'EnsembleBaseline',
    'create_mlp_baseline',
    'create_cnn1d_baseline',
    'create_random_forest',
    'create_svm',
    'create_logistic_regression',
    'create_ensemble'
]
