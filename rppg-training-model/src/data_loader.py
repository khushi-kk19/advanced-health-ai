import os
import pandas as pd
import json

class RPPGDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.subjects = ['subject41', 'subject42', 'subject46']

    def load_metadata(self):
        metadata = {}
        for subject in self.subjects:
            meta_path = os.path.join(self.data_dir, f"{subject}_meta.json")
            with open(meta_path, 'r') as f:
                metadata[subject] = json.load(f)
        return metadata

    def load_rppg_data(self):
        rppg_data = {}
        for subject in self.subjects:
            data_path = os.path.join(self.data_dir, f"{subject}.csv")
            rppg_data[subject] = pd.read_csv(data_path)
        return rppg_data

    def get_data(self):
        metadata = self.load_metadata()
        rppg_data = self.load_rppg_data()
        return metadata, rppg_data

if __name__ == "__main__":
    data_loader = RPPGDataLoader(data_dir='data/raw/ubfc_rppg')
    metadata, rppg_data = data_loader.get_data()
    print(metadata)
    print(rppg_data)