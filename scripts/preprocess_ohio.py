# scripts/preprocess_ohio.py
from pathlib import Path
import pandas as pd
import json
RAW = Path("data/raw/ohiot1dm")
OUT = Path("data/processed/ohiot1dm"); OUT.mkdir(parents=True, exist_ok=True)

# convert xlsx to csv if necessary and then process
files = list(RAW.rglob("*"))
# find spreadsheets or CSVs
xls = [p for p in files if p.suffix.lower() in (".xlsx", ".xls")]
csvs = [p for p in files if p.suffix.lower()==".csv"]

# convert xlsx to csv
for x in xls:
    try:
        df = pd.read_excel(x)
        outp = RAW / (x.stem + ".csv")
        df.to_csv(outp, index=False)
        csvs.append(outp)
        print("Converted", x, "->", outp)
    except Exception as e:
        print("Failed converting", x, e)

if not csvs:
    raise FileNotFoundError("No CSVs found in data/raw/ohiot1dm")

# naive grouping by filename => per-patient
for c in csvs:
    try:
        df = pd.read_csv(c)
        # attempt to find timestamp column or time-like col
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower()]
        if time_cols:
            tcol = time_cols[0]
            df[tcol] = pd.to_datetime(df[tcol], errors='coerce')
            df = df.sort_values(tcol).reset_index(drop=True)
            # resample to 5-min by forward-fill if timestamp granularity OK
            df = df.set_index(tcol).asfreq('1min', method='ffill').reset_index().rename(columns={df.index.name:'timestamp'})
        # normalize CGM column name guess
        cols_lower = [c.lower() for c in df.columns]
        cgm_col = None
        for name in ['cgm','glucose','sensor_glucose']:
            for col in df.columns:
                if name in col.lower():
                    cgm_col = col; break
            if cgm_col: break
        # keep a minimal set
        keep = []
        if cgm_col: keep.append(cgm_col)
        for cand in ['insulin','carb','basal','bolus','meal','hr','heart']:
            for col in df.columns:
                if cand in col.lower() and col not in keep:
                    keep.append(col)
        # always keep timestamp
        if 'timestamp' not in df.columns:
            # ensure we have a timestamp index? if not, add sequence index
            df = df.reset_index().rename(columns={'index':'timestamp'})
        out_file = OUT / (c.stem + ".csv")
        df.to_csv(out_file, index=False)
        print("Processed", c, "->", out_file)
    except Exception as e:
        print("Failed processing", c, e)

(OUT / "manifest.json").write_text(json.dumps({"files":[p.name for p in OUT.glob("*.csv")]}, indent=2))
