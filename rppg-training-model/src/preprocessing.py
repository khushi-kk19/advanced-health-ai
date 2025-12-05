import pandas as pd
import numpy as np
import os
import json

def load_metadata(file_path):
    with open(file_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def load_rppg_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Normalize the rPPG signal
    data['normalized'] = (data['signal'] - data['signal'].mean()) / data['signal'].std()
    
    # Apply a simple moving average filter
    data['filtered'] = data['normalized'].rolling(window=5).mean()
    
    return data

def save_processed_data(data, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    data.to_csv(output_path, index=False)

def process_subject_data(subject_id):
    meta_file = f'data/raw/ubfc_rppg/subject{subject_id}_meta.json'
    data_file = f'data/raw/ubfc_rppg/subject{subject_id}.csv'
    
    metadata = load_metadata(meta_file)
    rppg_data = load_rppg_data(data_file)
    
    processed_data = preprocess_data(rppg_data)
    
    output_file = f'data/processed/subject{subject_id}_processed.csv'
    save_processed_data(processed_data, output_file)
    
    return metadata, processed_data

if __name__ == "__main__":
    for subject_id in [41, 42, 46]:
        metadata, processed_data = process_subject_data(subject_id)
        print(f"Processed data for subject {subject_id}:")
        print(metadata)
        print(processed_data.head())