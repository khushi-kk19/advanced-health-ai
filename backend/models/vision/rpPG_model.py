"""
Deep learning model for rPPG heart rate prediction.
Architecture: CNN + LSTM + Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class rPPGCNNLSTM(nn.Module):
    """
    CNN-LSTM-Attention network for rPPG signal processing.
    
    Architecture:
    1. Conv1D layers to extract temporal patterns
    2. LSTM to capture long-term dependencies
    3. Attention layer to focus on important frequencies
    4. Fully connected for heart rate prediction
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, 
                 num_filters=32, kernel_size=3, dropout=0.2):
        super(rPPGCNNLSTM, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, 
                              padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size, 
                              padding=kernel_size//2)
        self.pool = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        
        # LSTM layers
        self.lstm = nn.LSTM(num_filters*2, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(hidden_size*2, num_heads=4, 
                                              dropout=dropout, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: Input shape (batch, 1, seq_len)
        
        Returns:
            heart_rate: Predicted heart rate (batch, 1)
            attention_weights: Attention weights for visualization
        """
        # CNN feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Reshape for LSTM: (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        x = torch.mean(attn_out, dim=1)
        
        # Fully connected
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, attn_weights


class rPPGResNet1D(nn.Module):
    """
    ResNet-1D for rPPG signal processing.
    Simplified architecture with residual connections.
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_blocks=4):
        super(rPPGResNet1D, self).__init__()
        
        self.conv_in = nn.Conv1d(input_size, hidden_size, kernel_size=7, 
                                padding=3)
        self.bn_in = nn.BatchNorm1d(hidden_size)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._residual_block(hidden_size, hidden_size, 3)
            for _ in range(num_blocks)
        ])
        
        # Global average pooling + FC
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def _residual_block(self, in_channels, out_channels, kernel_size):
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input shape (batch, 1, seq_len)
        
        Returns:
            heart_rate: Predicted heart rate (batch, 1)
        """
        x = self.relu(self.bn_in(self.conv_in(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            identity = x
            x = block(x)
            x = x + identity
            x = self.relu(x)
        
        # Global average pooling
        x = torch.mean(x, dim=2)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, None


if __name__ == "__main__":
    # Test models
    batch_size = 16
    seq_len = 30
    
    x = torch.randn(batch_size, 1, seq_len)
    
    # Test CNN-LSTM
    model1 = rPPGCNNLSTM(input_size=1, hidden_size=64)
    out1, attn1 = model1(x)
    print(f"CNN-LSTM output shape: {out1.shape}")
    print(f"Attention weights shape: {attn1.shape}")
    
    # Test ResNet
    model2 = rPPGResNet1D(input_size=1, hidden_size=64)
    out2, _ = model2(x)
    print(f"ResNet output shape: {out2.shape}")