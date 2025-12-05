# scripts/preprocess_uci_heart.py
import pandas as pd
from pathlib import Path
import json

RAW = Path("data/raw/heart_ucirepo")
OUT = Path("data/processed/heart_ucirepo")
OUT.mkdir(parents=True, exist_ok=True)

# column names from UCI heart-disease processed.cleveland.data (common mapping)
cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]

src = RAW / "processed.cleveland.data"
if not src.exists():
    # try other common names
    candidates = list(RAW.glob("*"))
    if candidates:
        src = candidates[0]
    else:
        raise FileNotFoundError(src)

# read; file may be comma separated, may have ? for missing
df = pd.read_csv(src, header=None, na_values='?')
if df.shape[1] == len(cols):
    df.columns = cols
else:
    # try to infer, otherwise keep generic names
    df.columns = [f"col{i}" for i in range(df.shape[1])]

# basic cleaning
df = df.drop_duplicates().reset_index(drop=True)
# convert numeric columns where possible
for c in df.columns:
    try:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    except:
        pass

out_csv = OUT / "heart_disease.csv"
df.to_csv(out_csv, index=False)

manifest = {"rows": len(df), "cols": list(df.columns)}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("Wrote", out_csv)
