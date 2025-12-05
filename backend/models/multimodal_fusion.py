"""
MULTIMODAL FUSION: Advanced techniques for combining physiological and facial modalities
- Early fusion: Combine signals before processing
- Late fusion: Combine predictions after processing
- Intermediate fusion: Combine learned representations
- Attention-based fusion: Learned fusion with attention weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EarlyFusion(nn.Module):
    """
    Early fusion: Concatenate raw signals and process jointly.
    Not typically used for multimodal learning but useful for analysis.
    """
    
    def __init__(self, num_classes=4):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(3 + 224*224*3, 512),  # Flatten image
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, physio_signal, facial_image):
        """
        Args:
            physio_signal: (batch, seq_len, 3)
            facial_image: (batch, 3, 224, 224)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Average physiological signal
        physio_avg = torch.mean(physio_signal, dim=1)  # (batch, 3)
        
        # Flatten image
        facial_flat = facial_image.reshape(facial_image.size(0), -1)  # (batch, 150528)
        
        # Concatenate
        combined = torch.cat([physio_avg, facial_flat], dim=1)
        
        logits = self.fc(combined)
        return logits


class LateFusion(nn.Module):
    """
    Late fusion: Process modalities separately and combine predictions.
    """
    
    def __init__(self, physio_model, facial_model, fusion_type='weighted', alpha=0.5):
        """
        Args:
            physio_model: Trained physiological model
            facial_model: Trained facial model
            fusion_type: 'weighted', 'voting', 'product'
            alpha: Weight for physiological model (if weighted)
        """
        super().__init__()
        
        self.physio_model = physio_model
        self.facial_model = facial_model
        self.fusion_type = fusion_type
        self.alpha = alpha
    
    def forward(self, physio_signal, facial_image):
        """
        Args:
            physio_signal: (batch, seq_len, 3)
            facial_image: (batch, 3, 224, 224)
        
        Returns:
            fused_logits: (batch, num_classes)
        """
        with torch.no_grad():
            physio_logits = self.physio_model(physio_signal)
            facial_logits = self.facial_model(facial_image)
        
        if self.fusion_type == 'weighted':
            fused_logits = self.alpha * physio_logits + (1 - self.alpha) * facial_logits
        
        elif self.fusion_type == 'voting':
            physio_pred = torch.argmax(F.softmax(physio_logits, dim=1), dim=1)
            facial_pred = torch.argmax(F.softmax(facial_logits, dim=1), dim=1)
            
            # Average probabilities and take max
            fused_proba = (F.softmax(physio_logits, dim=1) + F.softmax(facial_logits, dim=1)) / 2
            fused_logits = torch.log(fused_proba + 1e-8)  # Log for log_softmax compatibility
        
        elif self.fusion_type == 'product':
            physio_proba = F.softmax(physio_logits, dim=1)
            facial_proba = F.softmax(facial_logits, dim=1)
            
            # Product rule
            product_proba = physio_proba * facial_proba
            product_proba = product_proba / (product_proba.sum(dim=1, keepdim=True) + 1e-8)
            fused_logits = torch.log(product_proba + 1e-8)
        
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        return fused_logits


class IntermediateFusion(nn.Module):
    """
    Intermediate fusion: Combine feature representations before classification.
    """
    
    def __init__(
        self,
        physio_feature_dim=128,
        facial_feature_dim=512,
        fusion_hidden=256,
        num_classes=4,
        dropout=0.3
    ):
        """
        Args:
            physio_feature_dim: Physiological feature dimension
            facial_feature_dim: Facial feature dimension
            fusion_hidden: Fusion hidden dimension
            num_classes: Number of classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.physio_dim = physio_feature_dim
        self.facial_dim = facial_feature_dim
        
        # Feature projections
        self.physio_projection = nn.Sequential(
            nn.Linear(physio_feature_dim, fusion_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_hidden)
        )
        
        self.facial_projection = nn.Sequential(
            nn.Linear(facial_feature_dim, fusion_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_hidden)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(2 * fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(fusion_hidden),
            
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(fusion_hidden // 2),
            
            nn.Linear(fusion_hidden // 2, num_classes)
        )
    
    def forward(self, physio_features, facial_features):
        """
        Args:
            physio_features: (batch, physio_feature_dim)
            facial_features: (batch, facial_feature_dim)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Project features
        physio_proj = self.physio_projection(physio_features)
        facial_proj = self.facial_projection(facial_features)
        
        # Concatenate
        fused = torch.cat([physio_proj, facial_proj], dim=1)
        
        # Classify
        logits = self.fusion(fused)
        
        return logits


class AttentionFusion(nn.Module):
    """
    Attention-based fusion: Learn fusion weights via attention mechanism.
    """
    
    def __init__(
        self,
        physio_feature_dim=128,
        facial_feature_dim=512,
        fusion_hidden=256,
        num_classes=4,
        dropout=0.3,
        num_heads=4
    ):
        """
        Args:
            physio_feature_dim: Physiological feature dimension
            facial_feature_dim: Facial feature dimension
            fusion_hidden: Fusion hidden dimension
            num_classes: Number of classes
            dropout: Dropout rate
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.physio_dim = physio_feature_dim
        self.facial_dim = facial_feature_dim
        
        # Project to common dimension
        self.common_dim = fusion_hidden
        
        self.physio_projection = nn.Linear(physio_feature_dim, self.common_dim)
        self.facial_projection = nn.Linear(facial_feature_dim, self.common_dim)
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.common_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fusion_hidden // 2, num_classes)
        )
    
    def forward(self, physio_features, facial_features):
        """
        Args:
            physio_features: (batch, physio_feature_dim)
            facial_features: (batch, facial_feature_dim)
        
        Returns:
            logits: (batch, num_classes)
            attention_weights: Dictionary of attention weights
        """
        # Project features
        physio = self.physio_projection(physio_features)  # (batch, common_dim)
        facial = self.facial_projection(facial_features)  # (batch, common_dim)
        
        # Add batch dimension for attention (they expect (batch, seq_len, dim))
        physio = physio.unsqueeze(1)  # (batch, 1, common_dim)
        facial = facial.unsqueeze(1)  # (batch, 1, common_dim)
        
        # Cross-attention: Physiological attends to facial
        physio_attended, _ = self.cross_attention(physio, facial, facial)
        
        # Cross-attention: Facial attends to physiological
        facial_attended, _ = self.cross_attention(facial, physio, physio)
        
        # Self-attention for refinement
        physio_refined, _ = self.self_attention(physio_attended, physio_attended, physio_attended)
        facial_refined, _ = self.self_attention(facial_attended, facial_attended, facial_attended)
        
        # Remove sequence dimension and concatenate
        physio_final = physio_refined.squeeze(1)  # (batch, common_dim)
        facial_final = facial_refined.squeeze(1)  # (batch, common_dim)
        
        fused = torch.cat([physio_final, facial_final], dim=1)  # (batch, 2*common_dim)
        
        logits = self.classifier(fused)
        
        attention_weights = {
            'cross_physio': physio_attended,
            'cross_facial': facial_attended
        }
        
        return logits, attention_weights


class BilinearFusion(nn.Module):
    """
    Bilinear fusion: Learns interactions between modalities.
    """
    
    def __init__(
        self,
        physio_feature_dim=128,
        facial_feature_dim=512,
        fusion_rank=128,
        num_classes=4,
        dropout=0.3
    ):
        """
        Args:
            physio_feature_dim: Physiological feature dimension
            facial_feature_dim: Facial feature dimension
            fusion_rank: Rank of bilinear fusion
            num_classes: Number of classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.physio_dim = physio_feature_dim
        self.facial_dim = facial_feature_dim
        self.rank = fusion_rank
        
        # Bilinear layer
        self.bilinear = nn.Bilinear(
            physio_feature_dim,
            facial_feature_dim,
            fusion_rank
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_rank + physio_feature_dim + facial_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, physio_features, facial_features):
        """
        Args:
            physio_features: (batch, physio_feature_dim)
            facial_features: (batch, facial_feature_dim)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Bilinear interaction
        interaction = self.bilinear(physio_features, facial_features)  # (batch, rank)
        
        # Concatenate all features
        fused = torch.cat([interaction, physio_features, facial_features], dim=1)
        
        logits = self.classifier(fused)
        
        return logits


class GatedFusion(nn.Module):
    """
    Gated fusion: Uses learned gates to control information flow.
    """
    
    def __init__(
        self,
        physio_feature_dim=128,
        facial_feature_dim=512,
        fusion_hidden=256,
        num_classes=4,
        dropout=0.3
    ):
        """
        Args:
            physio_feature_dim: Physiological feature dimension
            facial_feature_dim: Facial feature dimension
            fusion_hidden: Fusion hidden dimension
            num_classes: Number of classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.physio_dim = physio_feature_dim
        self.facial_dim = facial_feature_dim
        
        # Gates
        self.physio_gate = nn.Sequential(
            nn.Linear(physio_feature_dim + facial_feature_dim, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, physio_feature_dim),
            nn.Sigmoid()
        )
        
        self.facial_gate = nn.Sequential(
            nn.Linear(physio_feature_dim + facial_feature_dim, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, facial_feature_dim),
            nn.Sigmoid()
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(physio_feature_dim + facial_feature_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fusion_hidden // 2, num_classes)
        )
    
    def forward(self, physio_features, facial_features):
        """
        Args:
            physio_features: (batch, physio_feature_dim)
            facial_features: (batch, facial_feature_dim)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Concatenate for gate input
        concat = torch.cat([physio_features, facial_features], dim=1)
        
        # Compute gates
        physio_gate = self.physio_gate(concat)
        facial_gate = self.facial_gate(concat)
        
        # Apply gates
        physio_gated = physio_features * physio_gate
        facial_gated = facial_features * facial_gate
        
        # Concatenate gated features
        fused = torch.cat([physio_gated, facial_gated], dim=1)
        
        logits = self.classifier(fused)
        
        return logits


def create_early_fusion(num_classes=4):
    """Create early fusion model."""
    return EarlyFusion(num_classes=num_classes)


def create_intermediate_fusion(num_classes=4):
    """Create intermediate fusion model."""
    return IntermediateFusion(
        physio_feature_dim=128,
        facial_feature_dim=512,
        fusion_hidden=256,
        num_classes=num_classes,
        dropout=0.3
    )


def create_attention_fusion(num_classes=4):
    """Create attention-based fusion model."""
    return AttentionFusion(
        physio_feature_dim=128,
        facial_feature_dim=512,
        fusion_hidden=256,
        num_classes=num_classes,
        dropout=0.3,
        num_heads=4
    )


def create_bilinear_fusion(num_classes=4):
    """Create bilinear fusion model."""
    return BilinearFusion(
        physio_feature_dim=128,
        facial_feature_dim=512,
        fusion_rank=128,
        num_classes=num_classes,
        dropout=0.3
    )


def create_gated_fusion(num_classes=4):
    """Create gated fusion model."""
    return GatedFusion(
        physio_feature_dim=128,
        facial_feature_dim=512,
        fusion_hidden=256,
        num_classes=num_classes,
        dropout=0.3
    )


if __name__ == "__main__":
    print("Testing Multimodal Fusion Strategies\n")
    
    batch_size = 4
    physio_features = torch.randn(batch_size, 128)
    facial_features = torch.randn(batch_size, 512)
    
    fusion_methods = [
        ("Intermediate Fusion", create_intermediate_fusion()),
        ("Attention Fusion", create_attention_fusion()),
        ("Bilinear Fusion", create_bilinear_fusion()),
        ("Gated Fusion", create_gated_fusion()),
    ]
    
    for name, model in fusion_methods:
        print("="*50)
        print(name)
        print("="*50)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if "Attention" in name:
            logits, attn = model(physio_features, facial_features)
            print(f"Output shape: {logits.shape}")
        else:
            logits = model(physio_features, facial_features)
            print(f"Output shape: {logits.shape}")
        
        print(f"Predicted classes: {torch.argmax(logits, dim=1)}")
        print()
    
    print("✓ All fusion strategies created successfully!")