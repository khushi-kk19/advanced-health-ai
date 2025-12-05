# scripts/preprocess_ppg_bp.py
from pathlib import Path
import json, shutil
import pandas as pd
try:
    import wfdb
except Exception:
    wfdb = None

RAW = Path("data/raw/ppg_bp")
OUT = Path("data/processed/ppg_bp"); OUT.mkdir(parents=True, exist_ok=True)
# find .hea files (WFDB records)
hea_files = list(RAW.rglob("*.hea"))
if wfdb and hea_files:
    for hea in hea_files:
        rec_dir = hea.parent
        rec_name = hea.stem
        try:
            record = wfdb.rdsamp(str(rec_dir / rec_name))
            samples, meta = record
            fs = meta.get('fs', None)
            df = pd.DataFrame(samples)
            if fs:
                df.insert(0, "time", (pd.Series(range(len(df))) / float(fs)))
            out_file = OUT / f"{rec_name}.csv"
            df.to_csv(out_file, index=False)
            print("Converted WFDB record", rec_name, "->", out_file)
        except Exception as e:
            print("WFDB read failed for", rec_name, e)
else:
    # fallback: copy CSVs or extracted folder
    csvs = list(RAW.rglob("*.csv"))
    if csvs:
        for c in csvs:
            df = pd.read_csv(c)
            out_file = OUT / c.name
            df.to_csv(out_file, index=False)
            print("Copied CSV", c, "->", out_file)
    else:
        # copy raw for manual inspection
        fallback = OUT / "raw"
        if fallback.exists():
            shutil.rmtree(fallback)
        shutil.copytree(RAW, fallback)
        print("No CSV/WFDB found; raw files copied to", fallback)

(OUT / "manifest.json").write_text(json.dumps({"note":"Converted or copied raw files", "files":[p.name for p in OUT.glob("*")]} , indent=2))
