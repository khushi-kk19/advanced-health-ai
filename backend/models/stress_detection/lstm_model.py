"""
LSTM Model for Physiological Stress Detection
- Input: Preprocessed ECG, EDA, Temperature signals
- Output: Stress classification (0-3) or regression (0-1)
- Architecture: Multi-layer LSTM with attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionLayer(nn.Module):
    """Self-attention mechanism for temporal sequences."""
    
    def __init__(self, hidden_size, num_heads=4):
        """
        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.head_dim = hidden_size // num_heads
        self.scale = np.sqrt(self.head_dim)
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, values, keys, query, mask=None):
        """
        Args:
            values: (batch, seq_len, hidden_size)
            keys: (batch, seq_len, hidden_size)
            query: (batch, seq_len, hidden_size)
            mask: Optional mask
        
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        batch_size = query.shape[0]
        
        # Linear transformations
        Q = self.query(query)  # (batch, seq_len, hidden)
        K = self.key(keys)     # (batch, seq_len, hidden)
        V = self.value(values) # (batch, seq_len, hidden)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, -1, self.hidden_size)
        
        # Final linear layer
        output = self.fc_out(context)
        
        return output, attention_weights


class LSTMStressDetector(nn.Module):
    """
    LSTM-based stress detection from physiological signals.
    
    Input: (batch_size, seq_len, 3) - ECG, EDA, Temperature
    Output: (batch_size, num_classes) - Stress classification
    """
    
    def __init__(
        self,
        input_size=3,
        hidden_size=128,
        num_layers=2,
        num_classes=4,
        dropout=0.3,
        use_attention=True,
        bidirectional=True
    ):
        """
        Args:
            input_size: Number of input features (3: ECG, EDA, Temp)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (4: baseline, stress, amusement, meditation)
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        
        # Embedding/projection layer
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(lstm_output_size, num_heads=4)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(lstm_output_size)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
        
        Returns:
            logits: (batch_size, num_classes)
            attention_weights: Attention weights if use_attention=True
        """
        batch_size = x.shape[0]
        
        # Embed input
        x = self.embedding(x)  # (batch, seq_len, hidden_size)
        x = self.norm1(x)
        x = F.relu(x)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq_len, lstm_out_size)
        lstm_out = self.norm2(lstm_out)
        
        attention_weights = None
        
        # Attention (optional)
        if self.use_attention:
            lstm_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        out = torch.mean(lstm_out, dim=1)  # (batch, lstm_out_size)
        
        # Classification head
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        logits = self.fc3(out)
        
        return logits, attention_weights


class LSTMStressRegressor(nn.Module):
    """
    LSTM-based stress regression (continuous stress level 0-1).
    Similar architecture but outputs continuous value.
    """
    
    def __init__(
        self,
        input_size=3,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        use_attention=True,
        bidirectional=True
    ):
        """
        Args:
            input_size: Number of input features (3: ECG, EDA, Temp)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        
        # Embedding
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention
        if use_attention:
            self.attention = AttentionLayer(lstm_output_size, num_heads=4)
        
        # Regression head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        # Normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(lstm_output_size)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
        
        Returns:
            stress_level: (batch_size, 1) in range [0, 1]
        """
        # Embed
        x = self.embedding(x)
        x = self.norm1(x)
        x = F.relu(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.norm2(lstm_out)
        
        # Attention (optional)
        if self.use_attention:
            lstm_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        out = torch.mean(lstm_out, dim=1)
        
        # Regression head
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        stress_level = torch.sigmoid(self.fc3(out))  # Sigmoid to [0, 1]
        
        return stress_level.squeeze()


def create_lstm_classifier(num_classes=4, use_attention=True):
    """Create LSTM classifier model."""
    return LSTMStressDetector(
        input_size=3,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
        use_attention=use_attention,
        bidirectional=True
    )


def create_lstm_regressor(use_attention=True):
    """Create LSTM regressor model."""
    return LSTMStressRegressor(
        input_size=3,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        use_attention=use_attention,
        bidirectional=True
    )


if __name__ == "__main__":
    print("Testing LSTM Stress Detection Models\n")
    
    # Test classifier
    print("="*50)
    print("LSTM Classifier")
    print("="*50)
    model_clf = create_lstm_classifier(num_classes=4, use_attention=True)
    print(f"Model parameters: {sum(p.numel() for p in model_clf.parameters()):,}")
    
    # Test input
    batch_size = 4
    seq_len = 256  # Window size
    input_size = 3  # ECG, EDA, Temperature
    
    x = torch.randn(batch_size, seq_len, input_size)
    logits, attn = model_clf(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    if attn is not None:
        print(f"Attention weights shape: {attn.shape}")
    print(f"Predicted classes: {torch.argmax(logits, dim=1)}")
    
    # Test regressor
    print("\n" + "="*50)
    print("LSTM Regressor")
    print("="*50)
    model_reg = create_lstm_regressor(use_attention=True)
    print(f"Model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
    
    stress_level = model_reg(x)
    print(f"Input shape: {x.shape}")
    print(f"Stress level shape: {stress_level.shape}")
    print(f"Stress levels (0-1): {stress_level}")
    
    print("\n✓ Models created successfully!")
