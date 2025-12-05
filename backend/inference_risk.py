"""
Load risk model and make predictions on new patients.
"""
import joblib
import numpy as np
import json

class RiskPredictor:
    def __init__(self, model_path="data/models/risk.pkl"):
        self.model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.features = self.metadata['features']
        print(f"✓ Loaded risk model from {model_path}")
        print(f"✓ Expected features: {self.features}")
    
    def predict(self, **kwargs):
        """
        Predict risk for a patient.
        
        Dynamic - accepts any features the model was trained on.
        """
        # Validate all required features provided
        missing = [f for f in self.features if f not in kwargs]
        if missing:
            print(f"\n❌ ERROR: Missing required features: {missing}")
            print(f"\nExpected features:")
            for f in self.features:
                print(f"  - {f}")
            raise KeyError(f"Missing features: {missing}")
        
        # Build feature vector in correct order
        X = np.array([[kwargs[f] for f in self.features]])
        
        # Predict
        pred_class = self.model.predict(X)[0]
        pred_prob = self.model.predict_proba(X)[0]
        
        risk_negative = pred_prob[0]
        risk_positive = pred_prob[1]
        
        print(f"\n{'='*60}")
        print(f"PATIENT RISK PREDICTION")
        print(f"{'='*60}")
        print(f"Input features:")
        for f in self.features:
            print(f"  {f}: {kwargs[f]}")
        
        print(f"\nPrediction:")
        print(f"  Class: {'POSITIVE' if pred_class == 1 else 'NEGATIVE'}")
        print(f"  Positive Risk: {risk_positive:.1%}")
        print(f"  Negative Risk: {risk_negative:.1%}")
        
        if risk_positive > 0.5:
            print(f"\n⚠️  HIGH RISK - Recommend clinical evaluation")
        elif risk_positive > 0.3:
            print(f"\n⚠️  MODERATE RISK - Recommend lifestyle changes")
        else:
            print(f"\n✓ LOW RISK")
        
        return {
            'prediction': int(pred_class),
            'risk_positive': float(risk_positive),
            'risk_negative': float(risk_negative)
        }

if __name__ == "__main__":
    # Load predictor first to see what features it needs
    predictor = RiskPredictor()
    
    # Example: Predict for a patient with ALL required features
    # (Model was trained on real Framingham data with these features)
    result = predictor.predict(
        age=55,
        bmi=28,
        currentsmoker=1,
        cigsperday=15,
        bpmeds=0,
        prevalentstroke=0,
        prevalenthyp=0,
        diabetes=0,
        totchol=220,
        sysbp=130
    )
    
    print(f"\nResult: {result}")