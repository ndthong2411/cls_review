"""
Best Model Prediction Script
Generated: 20251019_081848
Model: Gen1_DecisionTree
PR-AUC (CV): 0.8023
PR-AUC (Test): 0.7979
Accuracy (Test): 0.7238
"""

import joblib
import numpy as np
import pandas as pd

# Load model and preprocessing
model = joblib.load("best_model_20251019_081848.pkl")
# No scaler used
feature_selector = joblib.load('feature_selector_20251019_081848.pkl')

def predict(X):
    """
    Predict cardiovascular disease probability
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 15)
        Input features: ['age_years', 'gender', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo', 'pulse_pressure', 'map', 'is_hypertension', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    
    Returns:
    --------
    predictions : dict
        - 'class': predicted class (0 or 1)
        - 'probability': probability of positive class
        - 'risk_level': risk interpretation
    """
    X = np.array(X).reshape(1, -1) if len(np.array(X).shape) == 1 else np.array(X)
    
    # Apply preprocessing (order: scale -> feature select)
    # No scaling
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
