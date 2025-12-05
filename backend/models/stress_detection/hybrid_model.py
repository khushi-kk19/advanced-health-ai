"""
Hybrid CNN-LSTM Model combining temporal physiological and spatial facial features
- Physiological branch: LSTM for time-series ECG, EDA, Temperature
- Facial branch: CNN for facial emotion recognition
- Fusion: Concatenate and classify
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .lstm_model import LSTMStressDetector, AttentionLayer
from .cnn_model import ResNetStressDetector, CNNFeatureExtractor


class HybridStressDetector(nn.Module):
    """
    Hybrid model combining physiological (LSTM) and facial (CNN) modalities.
    
    Architecture:
    - Physiological branch: LSTM with attention on ECG, EDA, Temperature
    - Facial branch: CNN feature extractor
    - Fusion: Concatenation + MLP classifier
    """
    
    def __init__(
        self,
        num_classes=4,
        lstm_hidden=128,
        cnn_features=512,
        fusion_hidden=256,
        dropout=0.3
    ):
        """
        Args:
            num_classes: Number of output classes (4: stress levels)
            lstm_hidden: LSTM hidden dimension
            cnn_features: CNN feature dimension
            fusion_hidden: Fusion MLP hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden
        self.cnn_features = cnn_features
        self.fusion_hidden = fusion_hidden
        
        # ===== PHYSIOLOGICAL BRANCH (LSTM) =====
        self.physio_embedding = nn.Linear(3, lstm_hidden)  # 3 signals: ECG, EDA, Temp
        
        self.physio_lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        lstm_output_size = lstm_hidden * 2  # Bidirectional
        
        self.physio_attention = AttentionLayer(lstm_output_size, num_heads=4)
        
        self.physio_fc = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden)
        )
        
        # ===== FACIAL BRANCH (CNN) =====
        self.facial_cnn = CNNFeatureExtractor(
            model_name='resnet18',
            pretrained=True,
            feature_dim=cnn_features
        )
        
        # ===== FUSION =====
        fusion_input_size = lstm_hidden + cnn_features
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fusion_hidden, num_classes)
        )
        
        # Normalization layers
        self.physio_norm = nn.LayerNorm(lstm_hidden)
        self.facial_norm = nn.LayerNorm(cnn_features)
    
    def forward(self, physio_signal, facial_image):
        """
        Args:
            physio_signal: (batch, seq_len, 3) - ECG, EDA, Temperature
            facial_image: (batch, 3, 224, 224) - Facial image
        
        Returns:
            logits: (batch, num_classes)
            physio_features: (batch, lstm_hidden) - Physiological features
            facial_features: (batch, cnn_features) - Facial features
        """
        # ===== PHYSIOLOGICAL BRANCH =====
        physio = self.physio_embedding(physio_signal)  # (batch, seq_len, lstm_hidden)
        physio = F.relu(physio)
        
        physio_lstm_out, _ = self.physio_lstm(physio)  # (batch, seq_len, lstm_hidden*2)
        
        # Attention
        physio_att, _ = self.physio_attention(
            physio_lstm_out, physio_lstm_out, physio_lstm_out
        )
        
        # Global average pooling
        physio_features = torch.mean(physio_att, dim=1)  # (batch, lstm_hidden*2)
        
        # Process physiological features
        physio_features = self.physio_fc(physio_features)  # (batch, lstm_hidden)
        physio_features = self.physio_norm(physio_features)
        
        # ===== FACIAL BRANCH =====
        facial_features = self.facial_cnn(facial_image)  # (batch, cnn_features)
        facial_features = self.facial_norm(facial_features)
        
        # ===== FUSION =====
        fused = torch.cat([physio_features, facial_features], dim=1)  # (batch, lstm_hidden + cnn_features)
        logits = self.fusion(fused)  # (batch, num_classes)
        
        return logits, physio_features, facial_features


class PhysiologicalBranch(nn.Module):
    """
    Standalone physiological processing branch.
    Can be used for ablation studies or physiological-only inference.
    """
    
    def __init__(
        self,
        num_classes=4,
        hidden_size=128,
        dropout=0.3
    ):
        """
        Args:
            num_classes: Number of output classes
            hidden_size: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding = nn.Linear(3, hidden_size)
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.attention = AttentionLayer(hidden_size * 2, num_heads=4)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 3) - ECG, EDA, Temperature
        
        Returns:
            logits: (batch, num_classes)
        """
        x = self.embedding(x)
        x = F.relu(x)
        
        x, _ = self.lstm(x)
        x, _ = self.attention(x, x, x)
        
        x = torch.mean(x, dim=1)
        logits = self.fc(x)
        
        return logits


class FacialBranch(nn.Module):
    """
    Standalone facial processing branch.
    Can be used for ablation studies or facial-only inference.
    """
    
    def __init__(
        self,
        num_classes=4,
        model_name='resnet18',
        dropout=0.3
    ):
        """
        Args:
            num_classes: Number of output classes
            model_name: ResNet variant
            dropout: Dropout rate
        """
        super().__init__()
        
        self.cnn = ResNetStressDetector(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=True
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224) - Facial image
        
        Returns:
            logits: (batch, num_classes)
        """
        return self.cnn(x)


class MultimodalFusion(nn.Module):
    """
    Advanced multimodal fusion strategies.
    """
    
    @staticmethod
    def early_fusion(physio_features, facial_features):
        """Concatenate features early."""
        return torch.cat([physio_features, facial_features], dim=1)
    
    @staticmethod
    def late_fusion(physio_logits, facial_logits, alpha=0.5):
        """Weighted average of predictions."""
        return alpha * physio_logits + (1 - alpha) * facial_logits
    
    @staticmethod
    def attention_fusion(physio_features, facial_features, hidden_dim=256):
        """
        Attention-based fusion.
        Returns: Fused features with attention weights.
        """
        # Compute attention weights
        physio_attention = torch.sigmoid(torch.sum(physio_features * facial_features, dim=1, keepdim=True))
        facial_attention = torch.sigmoid(torch.sum(facial_features * physio_features, dim=1, keepdim=True))
        
        # Normalize
        total_attention = physio_attention + facial_attention
        physio_attention = physio_attention / (total_attention + 1e-8)
        facial_attention = facial_attention / (total_attention + 1e-8)
        
        # Fuse
        fused = physio_features * physio_attention + facial_features * facial_attention
        
        return fused, {'physio': physio_attention, 'facial': facial_attention}


def create_hybrid_detector(num_classes=4, pretrained_cnn=True):
    """Create hybrid stress detector."""
    return HybridStressDetector(
        num_classes=num_classes,
        lstm_hidden=128,
        cnn_features=512,
        fusion_hidden=256,
        dropout=0.3
    )


def create_physio_branch(num_classes=4):
    """Create physiological branch only."""
    return PhysiologicalBranch(
        num_classes=num_classes,
        hidden_size=128,
        dropout=0.3
    )


def create_facial_branch(num_classes=4):
    """Create facial branch only."""
    return FacialBranch(
        num_classes=num_classes,
        model_name='resnet18',
        dropout=0.3
    )


if __name__ == "__main__":
    print("Testing Hybrid Stress Detection Models\n")
    
    # Test input
    batch_size = 4
    seq_len = 256  # Physiological sequence length
    
    physio_signal = torch.randn(batch_size, seq_len, 3)  # ECG, EDA, Temp
    facial_image = torch.randn(batch_size, 3, 224, 224)
    
    # Test Hybrid Model
    print("="*50)
    print("Hybrid Stress Detector")
    print("="*50)
    model = create_hybrid_detector(num_classes=4, pretrained_cnn=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    logits, physio_feat, facial_feat = model(physio_signal, facial_image)
    print(f"Physio input shape: {physio_signal.shape}")
    print(f"Facial input shape: {facial_image.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Physio features shape: {physio_feat.shape}")
    print(f"Facial features shape: {facial_feat.shape}")
    print(f"Predicted classes: {torch.argmax(logits, dim=1)}")
    
    # Test Physiological Branch
    print("\n" + "="*50)
    print("Physiological Branch (LSTM)")
    print("="*50)
    physio_model = create_physio_branch(num_classes=4)
    print(f"Parameters: {sum(p.numel() for p in physio_model.parameters()):,}")
    physio_logits = physio_model(physio_signal)
    print(f"Output shape: {physio_logits.shape}")
    print(f"Predicted classes: {torch.argmax(physio_logits, dim=1)}")
    
    # Test Facial Branch
    print("\n" + "="*50)
    print("Facial Branch (CNN)")
    print("="*50)
    facial_model = create_facial_branch(num_classes=4)
    print(f"Parameters: {sum(p.numel() for p in facial_model.parameters()):,}")
    facial_logits = facial_model(facial_image)
    print(f"Output shape: {facial_logits.shape}")
    print(f"Predicted classes: {torch.argmax(facial_logits, dim=1)}")
    
    # Test Multimodal Fusion
    print("\n" + "="*50)
    print("Fusion Strategies")
    print("="*50)
    
    # Extract features
    with torch.no_grad():
        _, physio_feat, facial_feat = model(physio_signal, facial_image)
        
        # Late fusion
        physio_logits = physio_model(physio_signal)
        facial_logits = facial_model(facial_image)
        late_fused = MultimodalFusion.late_fusion(physio_logits, facial_logits, alpha=0.6)
        print(f"Late fusion shape: {late_fused.shape}")
    
    print("\n✓ All models created successfully!")
