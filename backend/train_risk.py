"""
Train Random Forest classifier for disease risk prediction.
Supports any CSV with features + binary target.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import argparse
import os
import json

def validate_and_clean(df, feature_cols, target_col):
    """Validate and clean data before training."""
    print(f"\nValidating data...")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Check features exist
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"\n❌ Missing features: {missing_features}")
        print(f"Available columns: {df.columns.tolist()}")
        raise KeyError(f"Features not found: {missing_features}")
    
    # Check target exists
    if target_col not in df.columns:
        print(f"❌ Target '{target_col}' not found in CSV")
        print(f"Available columns: {df.columns.tolist()}")
        raise KeyError(f"Target not found: {target_col}")
    
    # Convert to numeric
    print(f"\nConverting to numeric...")
    for col in feature_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check missing values
    missing_counts = df[feature_cols + [target_col]].isna().sum()
    print(f"Missing values per column:")
    for col, count in missing_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
    
    # Drop rows with any missing values
    df_clean = df[feature_cols + [target_col]].dropna()
    dropped = len(df) - len(df_clean)
    print(f"\n✓ Dropped {dropped} rows with missing values")
    print(f"✓ Cleaned dataset: {len(df_clean)} rows")
    
    if len(df_clean) == 0:
        raise ValueError("No valid data after cleaning!")
    
    return df_clean

def train(csv_path, feature_cols, target_col, output_path="data/models/risk.pkl", test_size=0.2):
    """Train risk prediction model."""
    
    print(f"\n{'='*60}")
    print(f"DISEASE RISK PREDICTION - RANDOM FOREST")
    print(f"{'='*60}")
    print(f"CSV: {csv_path}")
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        raise
    
    # Validate and clean
    df_clean = validate_and_clean(df, feature_cols, target_col)
    
    # Prepare features and target
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        pct = (c / len(y)) * 100
        print(f"  Class {int(u)}: {c} samples ({pct:.1f}%)")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n✓ Train/test split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Build pipeline
    print(f"\nTraining Random Forest classifier...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle imbalanced data
        ))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION")
    print(f"{'='*60}")
    
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC-ROC: {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"\n✓ Model saved to: {output_path}")
    
    # Save metadata
    metadata = {
        'features': feature_cols,
        'target': target_col,
        'n_samples': len(df_clean),
        'test_accuracy': float(accuracy),
        'test_auc': float(auc),
        'n_estimators': 200
    }
    
    metadata_path = output_path.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")
    
    return pipeline, accuracy, auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train risk prediction model")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--features", required=True, help="Comma-separated feature names (no spaces!)")
    parser.add_argument("--target", required=True, help="Target column name (binary: 0/1)")
    parser.add_argument("--out", default="data/models/risk.pkl", help="Output model path")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction")
    
    args = parser.parse_args()
    
    # Clean feature names (remove spaces)
    features = [f.strip() for f in args.features.split(",")]
    
    train(args.csv, features, args.target, args.out, args.test_size)
