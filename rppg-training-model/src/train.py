import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.model import rPPGModel  # Assuming a model class is defined in model.py
from src.utils import save_model, log_training  # Assuming utility functions are defined in utils.py
import json

class rPPGDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        # Assuming the last column is the target variable
        features = sample[:-1].values.astype(np.float32)
        target = sample[-1].astype(np.float32)
        return torch.tensor(features), torch.tensor(target)

def train_model(train_file, val_file, config):
    # Load datasets
    train_dataset = rPPGDataset(train_file)
    val_dataset = rPPGDataset(val_file)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = rPPGModel(config['input_size'], config['hidden_size'], config['output_size'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        for features, target in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, target in val_loader:
                outputs = model(features)
                val_loss += criterion(outputs, target).item()
        
        val_loss /= len(val_loader)
        log_training(epoch, loss.item(), val_loss)  # Log training progress
    
    save_model(model, config['model_save_path'])  # Save the trained model

if __name__ == "__main__":
    with open('configs/hyperparams.json') as f:
        config = json.load(f)
    
    train_file = 'data/splits/train.csv'
    val_file = 'data/splits/val.csv'
    
    train_model(train_file, val_file, config)