# scripts/ohio_prepare_for_training.py
import pandas as pd
from pathlib import Path

ROOT = Path('.').resolve()
IN = ROOT / 'data' / 'processed' / 'ohiot1dm' / 'ohio_merged.csv'
OUTDIR = IN.parent
OUT = OUTDIR / 'ohio_merged_clean.csv'

print("Reading:", IN)
df = pd.read_csv(IN)

# parse timestamp reliably
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# selected features and target
features = ['basal','basis_heart_rate','basis_gsr','basis_skin_temperature','basis_steps','bolus','meal','exercise','temp_basal']
target = 'glucose_level'

# keep patient id, timestamp, features, target
keep = ['patient_id','timestamp'] + features + [target]
df = df[[c for c in keep if c in df.columns]]

# Drop rows with missing target or all-feature-missing
df = df.dropna(subset=[target]).reset_index(drop=True)

# Optionally forward-fill a few sensor columns per patient to handle small gaps
df = df.sort_values(['patient_id','timestamp'])
df[features] = df.groupby('patient_id')[features].ffill().bfill()

# Remove any remaining rows with missing feature values (simple)
df = df.dropna(subset=features).reset_index(drop=True)

# Write cleaned file
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print("Wrote cleaned CSV:", OUT)
print("Shape:", df.shape)
