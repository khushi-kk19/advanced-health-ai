# scripts/preprocess_framingham.py
import pandas as pd
from pathlib import Path
import json
OUT = Path("data/processed/framingham"); OUT.mkdir(parents=True, exist_ok=True)
RAW = Path("data/raw/framingham")
# common file names from Kaggle mirror
cands = list(RAW.glob("*.csv")) + list(RAW.rglob("*.csv"))
if not cands:
    raise FileNotFoundError("No CSV found in data/raw/framingham")
# pick largest csv (likely the main one)
src = max(cands, key=lambda p: p.stat().st_size)
df = pd.read_csv(src)
# basic standardization: lowercase columns, strip spaces
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
# drop duplicates and save
df = df.drop_duplicates().reset_index(drop=True)
out_csv = OUT / "framingham.csv"
df.to_csv(out_csv, index=False)
manifest = {"source": str(src), "rows": len(df), "cols": list(df.columns)}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("Wrote", out_csv)
