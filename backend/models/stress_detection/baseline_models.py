"""
Baseline Models for Stress Detection
- Machine learning baselines (Random Forest, SVM, Logistic Regression)
- Simple neural network baselines
- For comparison with deep learning models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle


class SimpleMLPBaseline(nn.Module):
    """
    Simple Multi-Layer Perceptron baseline.
    Takes flattened/averaged features and classifies.
    """
    
    def __init__(
        self,
        input_features,
        hidden_size=128,
        num_classes=4,
        dropout=0.3
    ):
        """
        Args:
            input_features: Number of input features
            hidden_size: Hidden layer dimension
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, input_features)
        
        Returns:
            logits: (batch, num_classes)
        """
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout(x)
        
        logits = self.fc4(x)
        return logits


class ConvolutionalBaseline(nn.Module):
    """
    Simple 1D CNN baseline for time-series physiological signals.
    """
    
    def __init__(
        self,
        seq_length=256,
        num_channels=3,
        num_classes=4,
        dropout=0.3
    ):
        """
        Args:
            seq_length: Sequence length
            num_channels: Number of input channels (3: ECG, EDA, Temp)
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate flattened size
        seq_after_pool = seq_length // (2**3)  # 3 pooling layers
        self.flatten_size = 128 * seq_after_pool
        
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, num_channels, seq_length)
        
        Returns:
            logits: (batch, num_classes)
        """
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        x = x.reshape(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        
        return logits


class SKLearnWrapper:
    """
    Wrapper for scikit-learn models for consistent API.
    """
    
    def __init__(self, model_type='random_forest', **kwargs):
        """
        Args:
            model_type: 'random_forest', 'svm', 'logistic_regression'
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 15),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                gamma=kwargs.get('gamma', 'scale'),
                probability=True,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                C=kwargs.get('C', 1.0),
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X, y):
        """
        Train the model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predictions (n_samples,)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Get class probabilities.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Probabilities (n_samples, n_classes)
        """
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # Random Forest has this, but as fallback compute manually
            predictions = self.predict(X)
            proba = np.zeros((X.shape[0], 4))
            for i, pred in enumerate(predictions):
                proba[i, pred] = 1.0
            return proba
    
    def score(self, X, y):
        """
        Evaluate accuracy.
        
        Args:
            X: Test features
            y: Test labels
        
        Returns:
            Accuracy score
        """
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
        
        Returns:
            Cross-validation scores
        """
        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv)
        return scores
    
    def save(self, filepath):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
    
    def load(self, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']


class EnsembleBaseline:
    """
    Ensemble of multiple baseline models.
    """
    
    def __init__(self):
        """Initialize ensemble with multiple models."""
        self.models = {
            'rf': SKLearnWrapper('random_forest', n_estimators=100),
            'svm': SKLearnWrapper('svm', kernel='rbf', C=1.0),
            'lr': SKLearnWrapper('logistic_regression', max_iter=1000)
        }
    
    def fit(self, X, y):
        """Train all models."""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X, y)
    
    def predict(self, X):
        """
        Ensemble prediction by voting.
        
        Args:
            X: Features
        
        Returns:
            Predictions
        """
        predictions = []
        for model in self.models.values():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Voting
        predictions = np.array(predictions)
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int), minlength=4).argmax(),
            axis=0,
            arr=predictions
        )
        
        return ensemble_pred
    
    def predict_proba(self, X):
        """
        Ensemble probability by averaging.
        
        Args:
            X: Features
        
        Returns:
            Probability predictions
        """
        all_proba = []
        for model in self.models.values():
            proba = model.predict_proba(X)
            all_proba.append(proba)
        
        # Average probabilities
        ensemble_proba = np.mean(all_proba, axis=0)
        
        return ensemble_proba
    
    def score(self, X, y):
        """Evaluate all models."""
        scores = {}
        for name, model in self.models.items():
            scores[name] = model.score(X, y)
        
        return scores


def create_mlp_baseline(input_features, num_classes=4):
    """Create MLP baseline model."""
    return SimpleMLPBaseline(
        input_features=input_features,
        hidden_size=128,
        num_classes=num_classes,
        dropout=0.3
    )


def create_cnn1d_baseline(seq_length=256, num_channels=3, num_classes=4):
    """Create 1D CNN baseline model."""
    return ConvolutionalBaseline(
        seq_length=seq_length,
        num_channels=num_channels,
        num_classes=num_classes,
        dropout=0.3
    )


def create_random_forest(n_estimators=100):
    """Create Random Forest baseline."""
    return SKLearnWrapper('random_forest', n_estimators=n_estimators)


def create_svm(kernel='rbf'):
    """Create SVM baseline."""
    return SKLearnWrapper('svm', kernel=kernel)


def create_logistic_regression():
    """Create Logistic Regression baseline."""
    return SKLearnWrapper('logistic_regression')


def create_ensemble():
    """Create ensemble baseline."""
    return EnsembleBaseline()


if __name__ == "__main__":
    print("Testing Baseline Models\n")
    
    # Test MLP Baseline
    print("="*50)
    print("MLP Baseline")
    print("="*50)
    input_features = 42  # From feature engineering
    model = create_mlp_baseline(input_features, num_classes=4)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(4, input_features)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Predicted classes: {torch.argmax(logits, dim=1)}")
    
    # Test CNN1D Baseline
    print("\n" + "="*50)
    print("1D CNN Baseline")
    print("="*50)
    model = create_cnn1d_baseline(seq_length=256, num_channels=3, num_classes=4)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(4, 3, 256)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Predicted classes: {torch.argmax(logits, dim=1)}")
    
    # Test scikit-learn models
    print("\n" + "="*50)
    print("Scikit-learn Baselines")
    print("="*50)
    
    X_train = np.random.randn(100, 42)
    y_train = np.random.randint(0, 4, 100)
    X_test = np.random.randn(20, 42)
    y_test = np.random.randint(0, 4, 20)
    
    rf = create_random_forest(n_estimators=50)
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    print(f"Random Forest accuracy: {rf_score:.3f}")
    
    svm = create_svm(kernel='rbf')
    svm.fit(X_train, y_train)
    svm_score = svm.score(X_test, y_test)
    print(f"SVM accuracy: {svm_score:.3f}")
    
    lr = create_logistic_regression()
    lr.fit(X_train, y_train)
    lr_score = lr.score(X_test, y_test)
    print(f"Logistic Regression accuracy: {lr_score:.3f}")
    
    # Test Ensemble
    print("\n" + "="*50)
    print("Ensemble Baseline")
    print("="*50)
    ensemble = create_ensemble()
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_acc = np.mean(ensemble_pred == y_test)
    print(f"Ensemble accuracy: {ensemble_acc:.3f}")
    
    print("\n✓ All baseline models created successfully!")
