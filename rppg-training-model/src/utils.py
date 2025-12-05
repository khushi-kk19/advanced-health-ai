import os
import json
import numpy as np
import pandas as pd
import logging

def setup_logging(log_file='rppg_training.log'):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_csv(file_path):
    return pd.read_csv(file_path)

def save_csv(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)

def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)