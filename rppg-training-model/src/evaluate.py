import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in val_loader:
            inputs, targets = data
            outputs = model(inputs)
            all_preds.append(outputs.numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)

    return {
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'predictions': all_preds,
        'targets': all_targets
    }

if __name__ == "__main__":
    import torch
    from data_loader import get_data_loaders
    from model import MyModel  # Replace with your actual model class

    val_loader = get_data_loaders('val')  # Assuming a function to get validation data loader
    model = MyModel()  # Initialize your model
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))  # Load the best model

    results = evaluate_model(model, val_loader)
    print("Evaluation Results:")
    print(f"Mean Squared Error: {results['mean_squared_error']:.4f}")
    print(f"Mean Absolute Error: {results['mean_absolute_error']:.4f}")