# Remote Photoplethysmography (rPPG) Training Model

This project aims to develop a training model for Remote Photoplethysmography (rPPG) analysis using a dataset of facial videos. The model will extract heart rate information from the videos, leveraging deep learning techniques.

## Project Structure

```
rppg-training-model
├── data
│   ├── raw
│   │   └── ubfc_rppg
│   │       ├── subject41_meta.json
│   │       ├── subject41.csv
│   │       ├── subject42_meta.json
│   │       ├── subject42.csv
│   │       ├── subject46_meta.json
│   │       └── subject46.csv
│   ├── processed
│   │   └── .gitkeep
│   └── splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── src
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks
│   ├── 01_eda.ipynb
│   └── 02_model_analysis.ipynb
├── configs
│   ├── config.yaml
│   └── hyperparams.json
├── logs
│   └── .gitkeep
├── checkpoints
│   └── .gitkeep
├── requirements.txt
├── setup.py
└── README.md
```

## Dataset

The dataset used for training consists of raw rPPG data and metadata for multiple subjects. The data is organized as follows:

- **Raw Data**: Contains CSV files with rPPG data and JSON files with metadata for each subject.
- **Processed Data**: Placeholder for processed data files.
- **Data Splits**: Contains CSV files for training, validation, and testing.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd rppg-training-model
pip install -r requirements.txt
```

## Usage

1. **Data Loading**: Use `src/data_loader.py` to load the dataset.
2. **Preprocessing**: Preprocess the data using functions in `src/preprocessing.py`.
3. **Model Training**: Train the model using `src/train.py`.
4. **Evaluation**: Evaluate the model's performance with `src/evaluate.py`.
5. **Notebooks**: Explore the data and analyze the model using Jupyter notebooks in the `notebooks` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.