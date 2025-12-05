import pandas as pd
from pathlib import Path

RAW = Path("data/processed/ohiot1dm")   # adjust if your folder name differs
OUT = RAW / "ohio_merged.csv"

all_rows = []

for csv_file in RAW.glob("*.csv"):
    name = csv_file.stem
    # detect patient id by splitting before "-ws"
    patient_id = name.split("-ws")[0]

    df = pd.read_csv(csv_file)

    # add column to identify patient
    df["patient_id"] = patient_id
    df["source_file"] = csv_file.name

    all_rows.append(df)

merged = pd.concat(all_rows, ignore_index=True)

# sort by timestamp if exists
if "timestamp" in merged.columns:
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged = merged.sort_values(["patient_id", "timestamp"])

merged.to_csv(OUT, index=False)

print("\nMERGED CSV CREATED:")
print(OUT)
print("Shape:", merged.shape)
print("Columns:", merged.columns.tolist())
