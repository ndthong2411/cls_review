"""
Best Model Prediction Script
Generated: 20251019_163520
Model: Gen1_KNN
PR-AUC (CV): 0.8693
PR-AUC (Test): 0.8897
"""

import joblib
import numpy as np
import pandas as pd

# Load model and preprocessing
model = joblib.load("best_model_20251019_163520.pkl")
scaler = joblib.load('scaler_20251019_163520.pkl')
feature_selector = joblib.load('feature_selector_20251019_163520.pkl')

def predict(X):
    """
    Predict cardiovascular disease probability
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 30)
        Input features: ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    Returns:
    --------
    predictions : dict
        - 'class': predicted class (0 or 1)
        - 'probability': probability of positive class
        - 'risk_level': risk interpretation
    """
    X = np.array(X).reshape(1, -1) if len(np.array(X).shape) == 1 else np.array(X)
    
    # Apply preprocessing (order: scale -> feature select)
    X = scaler.transform(X)
    X = feature_selector.transform(X)
    
    # Predict
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    results = []
    for pred, proba in zip(y_pred, y_proba):
        risk_level = "HIGH" if proba >= 0.7 else "MODERATE" if proba >= 0.4 else "LOW"
        results.append({
            'class': int(pred),
            'probability': float(proba),
            'risk_level': risk_level
        })
    
    return results[0] if len(results) == 1 else results

# Example usage
if __name__ == "__main__":
    # Sample patient data (replace with actual values)
    sample = [
        # age_years, gender, height, weight, bmi, ap_hi, ap_lo, 
        # pulse_pressure, map, is_hypertension, cholesterol, gluc, smoke, alco, active
        52.5, 1, 165, 75, 27.5, 130, 85, 45, 100, 0, 2, 1, 0, 0, 1
    ]
    
    result = predict(sample)
    print(f"Prediction: {result['class']}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
