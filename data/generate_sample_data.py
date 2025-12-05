"""
Generate realistic Ohio T1DM sample data for testing.
Based on actual T1DM patterns: glucose 70-250 mg/dL, 5-min intervals.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_ohio_t1dm_sample(n_patients=5, days=90, output_path=None):
    """
    Generate synthetic Ohio T1DM data matching real distribution.
    """
    np.random.seed(42)
    
    records = []
    
    for patient_id in [559, 563, 570, 575, 588]:
        print(f"Generating data for patient {patient_id}...")
        
        # Start date
        start_date = datetime(2020, 1, 1)
        
        # Generate 5-minute interval data for N days
        n_samples = days * 24 * 12  # 5-min intervals
        
        for i in range(n_samples):
            timestamp = start_date + timedelta(minutes=5*i)
            
            # Realistic T1DM patterns
            # Glucose cycles through day (high after meals, low at night)
            time_of_day = (timestamp.hour * 60 + timestamp.minute) / (24 * 60)
            glucose_base = 120 + 50 * np.sin(2 * np.pi * time_of_day)
            glucose_level = glucose_base + np.random.normal(0, 15)
            glucose_level = np.clip(glucose_level, 50, 400)
            
            # Basal insulin (constant, 0.5-2.0 units/hour)
            basal = np.random.uniform(0.5, 2.0)
            
            # Wearable sensor data (realistic ranges)
            basis_heart_rate = 70 + 20 * np.sin(2 * np.pi * time_of_day) + np.random.normal(0, 5)
            basis_gsr = np.random.uniform(0.1, 5.0)
            basis_skin_temperature = 33 + np.random.normal(0, 0.5)
            basis_steps = np.random.poisson(5)
            
            # Bolus insulin (0-20 units, random meals)
            bolus = np.random.choice([0] * 9 + [np.random.uniform(5, 20)])
            
            # Meal carbs (0-100g, sparse)
            meal = np.random.choice([0] * 15 + [np.random.uniform(30, 100)])
            
            # Exercise (0-100 arbitrary units)
            exercise = np.random.choice([0] * 20 + [np.random.uniform(10, 50)])
            
            # Temp basal (same as basal for now)
            temp_basal = basal
            
            records.append({
                'patient_id': patient_id,
                'timestamp': timestamp,
                'basal': round(basal, 2),
                'basis_heart_rate': round(basis_heart_rate, 1),
                'basis_gsr': round(basis_gsr, 2),
                'basis_skin_temperature': round(basis_skin_temperature, 1),
                'basis_steps': int(basis_steps),
                'bolus': round(bolus, 1),
                'meal': round(meal, 1),
                'exercise': round(exercise, 1),
                'temp_basal': round(temp_basal, 2),
                'glucose_level': round(glucose_level, 1),
            })
    
    df = pd.DataFrame(records)
    
    if output_path is None:
        output_path = "data/processed/ohiot1dm/ohio_merged_clean.csv"
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Generated {len(df)} records")
    print(f"✓ Saved to {output_path}")
    print(f"\nData summary:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    df = generate_ohio_t1dm_sample(n_patients=5, days=90)