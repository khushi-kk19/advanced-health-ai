"""
CNN Model for Facial Emotion Recognition & Stress Detection
- Input: Facial images (224x224x3)
- Output: Emotion classification or stress levels
- Architecture: ResNet-based or custom CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNNStressDetector(nn.Module):
    """
    Simple CNN for facial stress detection.
    """
    
    def __init__(self, num_classes=4, pretrained=False):
        """
        Args:
            num_classes: Number of output classes (4: stress levels)
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
        )
        
        # Global average pooling + classification
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, 3, 224, 224)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global average pooling
        x = self.gap(x)  # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 256)
        
        # Classification head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


class ResNetStressDetector(nn.Module):
    """
    ResNet-based facial stress detection using pretrained features.
    """
    
    def __init__(self, num_classes=4, model_name='resnet18', pretrained=True):
        """
        Args:
            num_classes: Number of output classes
            model_name: ResNet variant ('resnet18', 'resnet34', 'resnet50')
            pretrained: Use pretrained ImageNet weights
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Load pretrained ResNet
        if model_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Get feature dimension
        if model_name in ['resnet18', 'resnet34']:
            feature_dim = 512
        else:  # resnet50
            feature_dim = 2048
        
        # Custom classification head
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Batch normalization
        self.norm1 = nn.BatchNorm1d(256)
        self.norm2 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, 3, 224, 224)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Backbone features
        x = self.backbone(x)  # (batch, feature_dim, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, feature_dim)
        
        # Classification head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.norm2(x)
        x = self.dropout(x)
        logits = self.fc3(x)
        
        return logits


class EfficientNetStressDetector(nn.Module):
    """
    EfficientNet-based facial stress detection.
    More efficient than ResNet with better accuracy.
    """
    
    def __init__(self, num_classes=4, pretrained=True):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Use pretrained ImageNet weights
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Load pretrained EfficientNet-B0
        try:
            from torchvision.models import efficientnet_b0
            backbone = efficientnet_b0(pretrained=pretrained)
        except:
            print("EfficientNet not available, using ResNet18 instead")
            backbone = models.resnet18(pretrained=pretrained)
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Custom classification head
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1280, 256)  # EfficientNet-B0 outputs 1280 features
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.norm1 = nn.BatchNorm1d(256)
        self.norm2 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, 3, 224, 224)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Backbone features
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        
        # Classification head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.norm2(x)
        x = self.dropout(x)
        logits = self.fc3(x)
        
        return logits


class CNNFeatureExtractor(nn.Module):
    """
    CNN for extracting facial features (for multimodal fusion).
    Returns feature vector instead of classification.
    """
    
    def __init__(self, model_name='resnet18', pretrained=True, feature_dim=512):
        """
        Args:
            model_name: Backbone model
            pretrained: Use pretrained weights
            feature_dim: Output feature dimension
        """
        super().__init__()
        
        # Load pretrained model
        if model_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            base_features = 512
        elif model_name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
            base_features = 512
        elif model_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            base_features = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Backbone
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Feature projection
        self.fc = nn.Linear(base_features, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, 3, 224, 224)
        
        Returns:
            features: (batch, feature_dim)
        """
        # Extract features
        x = self.backbone(x)  # (batch, base_features, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, base_features)
        
        # Project to feature dimension
        x = self.fc(x)
        x = self.norm(x)
        
        return x


def create_simple_cnn(num_classes=4):
    """Create simple CNN model."""
    return SimpleCNNStressDetector(num_classes=num_classes, pretrained=False)


def create_resnet_detector(num_classes=4, model_name='resnet18', pretrained=True):
    """Create ResNet-based detector."""
    return ResNetStressDetector(num_classes=num_classes, model_name=model_name, pretrained=pretrained)


def create_feature_extractor(model_name='resnet18', feature_dim=512):
    """Create feature extractor for multimodal fusion."""
    return CNNFeatureExtractor(model_name=model_name, pretrained=True, feature_dim=feature_dim)


if __name__ == "__main__":
    print("Testing CNN Stress Detection Models\n")
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Test Simple CNN
    print("="*50)
    print("Simple CNN")
    print("="*50)
    model = create_simple_cnn(num_classes=4)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Predicted classes: {torch.argmax(logits, dim=1)}")
    
    # Test ResNet Detector
    print("\n" + "="*50)
    print("ResNet Detector (resnet18)")
    print("="*50)
    model = create_resnet_detector(num_classes=4, model_name='resnet18', pretrained=False)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    print(f"Predicted classes: {torch.argmax(logits, dim=1)}")
    
    # Test Feature Extractor
    print("\n" + "="*50)
    print("Feature Extractor")
    print("="*50)
    model = create_feature_extractor(model_name='resnet18', feature_dim=512)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    features = model(x)
    print(f"Feature shape: {features.shape}")
    print(f"Feature vector range: [{features.min():.3f}, {features.max():.3f}]")
    
    print("\n✓ All models created successfully!")
