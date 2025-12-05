"""
Data validation & quality assessment for health timeseries.
Checks schema, data types, missing values, outliers, temporal consistency.
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

class HealthDataValidator:
    def __init__(self, csv_path, patient_col='patient_id', timestamp_col='timestamp', target_col=None):
        self.csv_path = csv_path
        self.patient_col = patient_col
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.df = None
        self.report = {}
        
    def load_data(self):
        """Load CSV safely"""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df.columns = [c.strip() for c in self.df.columns]
            self.report['load_status'] = 'SUCCESS'
            self.report['total_rows'] = len(self.df)
            self.report['total_columns'] = len(self.df.columns)
            print(f"✓ Loaded {len(self.df)} rows × {len(self.df.columns)} columns")
            return True
        except Exception as e:
            self.report['load_status'] = f'FAILED: {str(e)}'
            print(f"✗ Load failed: {e}")
            return False
    
    def validate_schema(self, required_cols):
        """Check required columns exist"""
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            self.report['schema_status'] = 'FAILED'
            self.report['missing_columns'] = missing
            print(f"✗ Missing columns: {missing}")
            return False
        self.report['schema_status'] = 'SUCCESS'
        print(f"✓ All required columns present")
        return True
    
    def check_data_types(self, feature_cols, target_col):
        """Validate numeric columns"""
        issues = []
        
        # Try converting to numeric
        for col in feature_cols + [target_col]:
            non_numeric = pd.to_numeric(self.df[col], errors='coerce').isna() & self.df[col].notna()
            n_non_numeric = non_numeric.sum()
            if n_non_numeric > 0:
                issues.append(f"{col}: {n_non_numeric} non-numeric values")
        
        if issues:
            self.report['type_check'] = issues
            print(f"⚠ Type issues: {issues}")
            return False
        
        self.report['type_check'] = 'SUCCESS'
        print(f"✓ All columns can be coerced to numeric")
        return True
    
    def check_missing_values(self, feature_cols, target_col):
        """Report NaN statistics"""
        all_cols = feature_cols + [target_col]
        missing_stats = {}
        
        for col in all_cols:
            n_missing = self.df[col].isna().sum()
            pct_missing = (n_missing / len(self.df)) * 100
            missing_stats[col] = {'count': n_missing, 'percent': pct_missing}
            
            if pct_missing > 50:
                print(f"⚠ {col}: {pct_missing:.1f}% missing (HIGH)")
            elif pct_missing > 20:
                print(f"⚠ {col}: {pct_missing:.1f}% missing")
            else:
                print(f"✓ {col}: {pct_missing:.1f}% missing")
        
        self.report['missing_values'] = missing_stats
        return True
    
    def check_temporal_consistency(self):
        """Validate patient timestamps"""
        if self.timestamp_col not in self.df.columns:
            print("ⓘ No timestamp column, skipping temporal checks")
            return True
        
        self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col], errors='coerce')
        temporal_issues = []
        
        if self.patient_col in self.df.columns:
            for pid, group in self.df.groupby(self.patient_col):
                group_sorted = group.sort_values(self.timestamp_col)
                diffs = group_sorted[self.timestamp_col].diff().dt.total_seconds()
                
                # Check for backwards timestamps
                if (diffs < 0).any():
                    temporal_issues.append(f"Patient {pid}: timestamps not monotonic")
                
                # Check for large gaps
                if (diffs > 3600).any():  # > 1 hour
                    gap_count = (diffs > 3600).sum()
                    temporal_issues.append(f"Patient {pid}: {gap_count} gaps > 1 hour")
        
        if temporal_issues:
            self.report['temporal_status'] = temporal_issues
            for issue in temporal_issues[:5]:  # Show first 5
                print(f"⚠ {issue}")
            if len(temporal_issues) > 5:
                print(f"... and {len(temporal_issues)-5} more issues")
            return False
        
        self.report['temporal_status'] = 'SUCCESS'
        print("✓ Temporal consistency OK")
        return True
    
    def check_patient_coverage(self):
        """Analyze per-patient data density"""
        if self.patient_col not in self.df.columns:
            print("ⓘ No patient column, skipping coverage analysis")
            return True
        
        coverage = self.df.groupby(self.patient_col).size()
        self.report['patient_coverage'] = {
            'n_patients': len(coverage),
            'min_records': int(coverage.min()),
            'max_records': int(coverage.max()),
            'mean_records': float(coverage.mean()),
            'median_records': float(coverage.median())
        }
        
        print(f"✓ {len(coverage)} patients")
        print(f"  - Records per patient: min={coverage.min()}, max={coverage.max()}, avg={coverage.mean():.0f}")
        
        # Warn if many patients have few records
        few_records = (coverage < 50).sum()
        if few_records > 0:
            print(f"⚠ {few_records} patients have < 50 records (may not form sequences)")
        
        return True
    
    def check_outliers(self, feature_cols, z_threshold=3):
        """Detect statistical outliers"""
        outlier_counts = {}
        
        for col in feature_cols:
            numeric_col = pd.to_numeric(self.df[col], errors='coerce')
            z_scores = np.abs((numeric_col - numeric_col.mean()) / numeric_col.std())
            n_outliers = (z_scores > z_threshold).sum()
            outlier_counts[col] = n_outliers
            
            if n_outliers > 0:
                pct = (n_outliers / len(self.df)) * 100
                print(f"⚠ {col}: {n_outliers} outliers ({pct:.1f}%)")
        
        self.report['outliers'] = outlier_counts
        return True
    
    def estimate_sequences(self, feature_cols, target_col, seq_len=30):
        """Estimate how many sequences can be built"""
        if self.patient_col not in self.df.columns:
            print("ⓘ No patient column, skipping sequence estimation")
            return 0
        
        total_sequences = 0
        skipped_patients = 0
        
        for pid, group in self.df.groupby(self.patient_col):
            # Apply same cleaning as train_lstm.py
            g = group.reset_index(drop=True)
            g_feat = g[feature_cols].apply(pd.to_numeric, errors='coerce')
            g_target = pd.to_numeric(g[target_col], errors='coerce')
            valid_mask = (~g_feat.isna().any(axis=1)) & (~g_target.isna())
            g = g.loc[valid_mask].reset_index(drop=True)
            
            if len(g) <= seq_len:
                skipped_patients += 1
            else:
                n_seqs = len(g) - seq_len
                total_sequences += n_seqs
        
        self.report['sequence_estimate'] = {
            'total_sequences': total_sequences,
            'skipped_patients': skipped_patients,
            'seq_len': seq_len
        }
        
        print(f"✓ Sequence estimation (seq_len={seq_len}):")
        print(f"  - Estimated sequences: {total_sequences}")
        print(f"  - Skipped patients (too few records): {skipped_patients}")
        
        if total_sequences == 0:
            print("✗ ERROR: No sequences can be built! Data quality too low.")
            return 0
        
        print(f"  → Safe to train with seq_len={seq_len}")
        return total_sequences
    
    def generate_report(self):
        """Print summary report"""
        print("\n" + "="*60)
        print("DATA VALIDATION REPORT")
        print("="*60)
        import json
        print(json.dumps(self.report, indent=2, default=str))
        print("="*60 + "\n")
        
        # Save report
        report_path = Path(self.csv_path).parent / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
        print(f"Report saved to: {report_path}")
    
    def validate_all(self, feature_cols, target_col, seq_len=30):
        """Run full validation pipeline"""
        print(f"\nValidating: {self.csv_path}\n")
        
        if not self.load_data():
            return False
        
        if not self.validate_schema(feature_cols + [target_col]):
            return False
        
        self.check_data_types(feature_cols, target_col)
        self.check_missing_values(feature_cols, target_col)
        self.check_temporal_consistency()
        self.check_patient_coverage()
        self.check_outliers(feature_cols)
        
        n_seqs = self.estimate_sequences(feature_cols, target_col, seq_len=seq_len)
        
        self.generate_report()
        
        return n_seqs > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate health timeseries data")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--features", required=True, help="Comma-separated feature names")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length for estimation")
    parser.add_argument("--patient_col", default="patient_id", help="Patient ID column")
    parser.add_argument("--timestamp_col", default="timestamp", help="Timestamp column")
    
    args = parser.parse_args()
    feature_cols = [f.strip() for f in args.features.split(",")]
    
    validator = HealthDataValidator(
        args.csv,
        patient_col=args.patient_col,
        timestamp_col=args.timestamp_col,
        target_col=args.target
    )
    
    success = validator.validate_all(feature_cols, args.target, seq_len=args.seq_len)
    exit(0 if success else 1)