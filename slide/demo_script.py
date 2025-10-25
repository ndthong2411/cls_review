#!/usr/bin/env python3
"""
🎭 LIVE DEMO SCRIPT FOR PRESENTATION
Run this during presentation to show real-time model predictions
"""

import pandas as pd
import numpy as np
import joblib
import streamlit as st
from pathlib import Path
import time

# Configuration
DEMO_DATA_PATH = "data/raw/cardio_train.csv"
MODEL_CACHE_PATH = "experiments/model_cache/cardio_train"
BEST_MODEL_CONFIG = {
    "model": "Gen3_XGBoost",
    "scaler": "robust",
    "imbalance": "smote_enn", 
    "feature_selection": "select_k_best_12"
}

def load_demo_model():
    """Load the best model for demo"""
    try:
        # Try to find the best cached model
        cache_dir = Path(MODEL_CACHE_PATH)
        model_files = list(cache_dir.glob("Gen3_XGBoost_*.pkl"))
        
        if model_files:
            # Load the most recent model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            model = joblib.load(latest_model)
            print(f"✅ Loaded model: {latest_model.name}")
            return model
        else:
            print("❌ No cached model found, training demo model...")
            return train_demo_model()
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def train_demo_model():
    """Train a quick demo model if no cached model available"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import RobustScaler
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Load sample data
        df = pd.read_csv(DEMO_DATA_PATH, sep=';')
        
        # Feature engineering
        df['age_years'] = df['age'] / 365.25
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
        
        feature_cols = [
            'age_years', 'gender', 'height', 'weight', 'bmi',
            'ap_hi', 'ap_lo', 'pulse_pressure', 'map',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active'
        ]
        
        X = df[feature_cols].values
        y = df['cardio'].values
        
        # Quick training on subset
        sample_size = min(5000, len(X))
        X_sample = X[:sample_size]
        y_sample = y[:sample_size]
        
        # Preprocessing
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_sample)
        
        selector = SelectKBest(f_classif, k=12)
        X_selected = selector.fit_transform(X_scaled, y_sample)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_selected, y_sample)
        
        print("✅ Trained demo model successfully")
        return model, scaler, selector
        
    except Exception as e:
        print(f"❌ Error training demo model: {e}")
        return None

def create_demo_patients():
    """Create demo patient profiles for presentation"""
    patients = [
        {
            "name": "Bệnh nhân A - Nguyễn Văn An",
            "age": 67,
            "gender": 1,
            "height": 165,
            "weight": 75,
            "ap_hi": 165,
            "ap_lo": 95,
            "cholesterol": 3,
            "gluc": 1,
            "smoke": 1,
            "alco": 0,
            "active": 1,
            "story": "Bệnh nhân 67 tuổi, nam, đến khám vì tức ngực khi gắng sức. Có tiền sử hút thuốc 20 năm."
        },
        {
            "name": "Bệnh nhân B - Trần Thị Mai",
            "age": 45,
            "gender": 0,
            "height": 158,
            "weight": 62,
            "ap_hi": 120,
            "ap_lo": 80,
            "cholesterol": 2,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1,
            "story": "Bệnh nhân 45 tuổi, nữ, khám sức khỏe định kỳ. Không có triệu chứng cụ thể."
        },
        {
            "name": "Bệnh nhân C - Lê Văn Hùng",
            "age": 55,
            "gender": 1,
            "height": 170,
            "weight": 85,
            "ap_hi": 145,
            "ap_lo": 90,
            "cholesterol": 3,
            "gluc": 2,
            "smoke": 1,
            "alco": 1,
            "active": 0,
            "story": "Bệnh nhân 55 tuổi, nam, tiền sử đái tháo đường type 2, hay uống rượu bia."
        }
    ]
    return patients

def preprocess_patient(patient_data):
    """Preprocess patient data for model prediction"""
    # Feature engineering
    age_years = patient_data['age']
    bmi = patient_data['weight'] / ((patient_data['height'] / 100) ** 2)
    pulse_pressure = patient_data['ap_hi'] - patient_data['ap_lo']
    map_pressure = (patient_data['ap_hi'] + 2 * patient_data['ap_lo']) / 3
    
    features = [
        age_years, patient_data['gender'], patient_data['height'], patient_data['weight'], bmi,
        patient_data['ap_hi'], patient_data['ap_lo'], pulse_pressure, map_pressure,
        patient_data['cholesterol'], patient_data['gluc'], patient_data['smoke'], 
        patient_data['alco'], patient_data['active']
    ]
    
    return np.array(features).reshape(1, -1)

def predict_risk(model, patient_data, scaler=None, selector=None):
    """Make prediction for a patient"""
    try:
        # Preprocess
        features = preprocess_patient(patient_data)
        
        # Apply preprocessing if available
        if scaler is not None:
            features = scaler.transform(features)
        
        if selector is not None:
            features = selector.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0, 1]
        
        return prediction, probability
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, None

def interpret_risk(probability, patient_data):
    """Interpret risk level and provide recommendations"""
    if probability < 0.3:
        risk_level = "THẤP"
        color = "🟢"
        recommendations = [
            "Duy trì lối sống lành mạnh",
            "Khám sức khỏe định kỳ 6 tháng/lần",
            "Tập thể dục đều đặn"
        ]
    elif probability < 0.7:
        risk_level = "TRUNG BÌNH"
        color = "🟡"
        recommendations = [
            "Cần theo dõi chặt chẽ các yếu tố nguy cơ",
            "Kiểm tra huyết áp, mỡ máu định kỳ",
            "Thay đổi lối sống: ăn uống lành mạnh, tập thể dục",
            "Cân nhắc tư vấn chuyên gia tim mạch"
        ]
    else:
        risk_level = "CAO"
        color = "🔴"
        recommendations = [
            "Cần tư vấn chuyên gia tim mạch NGAY LẬP TỨC",
            "Kiểm tra sâu: ECG, X-quang, xét nghiệm mỡ máu",
            "Cân nhắc bắt đầu điều trị phòng ngừa",
            "Theo dõi y tế chặt chẽ"
        ]
    
    # Identify key risk factors
    risk_factors = []
    if patient_data['age'] > 60:
        risk_factors.append(f"Tuổi cao ({patient_data['age']} tuổi)")
    if patient_data['ap_hi'] > 140 or patient_data['ap_lo'] > 90:
        risk_factors.append(f"Huyết áp cao ({patient_data['ap_hi']}/{patient_data['ap_lo']} mmHg)")
    if patient_data['cholesterol'] >= 3:
        risk_factors.append("Mỡ máu cao")
    if patient_data['smoke'] == 1:
        risk_factors.append("Hút thuốc")
    if patient_data['bmi'] > 25:
        risk_factors.append(f"Thừa cân (BMI: {patient_data['bmi']:.1f})")
    
    return risk_level, color, recommendations, risk_factors

def run_demo():
    """Run live demo"""
    print("🎭 STARTING LIVE DEMO...")
    print("=" * 50)
    
    # Load model
    model_components = load_demo_model()
    if model_components is None:
        print("❌ Cannot load model. Demo cancelled.")
        return
    
    if isinstance(model_components, tuple):
        model, scaler, selector = model_components
    else:
        model = model_components
        scaler = None
        selector = None
    
    # Get demo patients
    patients = create_demo_patients()
    
    print(f"📋 Loaded {len(patients)} demo patients")
    print("=" * 50)
    
    for i, patient in enumerate(patients, 1):
        print(f"\n🏥 BỆNH NHÂN {i}: {patient['name']}")
        print(f"📖 Tiền sử: {patient['story']}")
        print(f"📊 Dữ liệu:")
        print(f"   - Tuổi: {patient['age']} tuổi")
        print(f"   - Giới tính: {'Nam' if patient['gender'] == 1 else 'Nữ'}")
        print(f"   - BMI: {patient['weight'] / ((patient['height'] / 100) ** 2):.1f}")
        print(f"   - Huyết áp: {patient['ap_hi']}/{patient['ap_lo']} mmHg")
        print(f"   - Mỡ máu: {['Bình thường', 'Trên bình thường', 'Cao'][patient['cholesterol']-1]}")
        print(f"   - Hút thuốc: {'Có' if patient['smoke'] == 1 else 'Không'}")
        
        # Make prediction
        print(f"\n🤖 AI đang phân tích...")
        time.sleep(2)  # Dramatic pause
        
        prediction, probability = predict_risk(model, patient, scaler, selector)
        
        if prediction is not None:
            risk_level, color, recommendations, risk_factors = interpret_risk(probability, patient)
            
            print(f"\n{color} KẾT QUẢ DỰ ĐOÁN:")
            print(f"   - Rủi ro tim mạch: {risk_level}")
            print(f"   - Xác suất: {probability:.1%}")
            
            if risk_factors:
                print(f"   - Các yếu tố nguy cơ chính: {', '.join(risk_factors)}")
            
            print(f"\n💡 KHUYẾN NGHỊ:")
            for rec in recommendations:
                print(f"   • {rec}")
        else:
            print("❌ Không thể dự đoán cho bệnh nhân này")
        
        print("\n" + "=" * 50)
        
        if i < len(patients):
            input("Nhấn Enter để tiếp tục bệnh nhân tiếp theo...")

def main():
    """Main function"""
    print("🎯 CARDIOVASCULAR DISEASE PREDICTION DEMO")
    print("=" * 50)
    print("Chọn chế độ demo:")
    print("1. Demo với bệnh nhân mẫu")
    print("2. Demo tương tác (nhập dữ liệu thủ công)")
    print("3. Streamlit demo (web interface)")
    
    choice = input("\nNhập lựa chọn (1-3): ").strip()
    
    if choice == "1":
        run_demo()
    elif choice == "2":
        interactive_demo()
    elif choice == "3":
        print("🌐 Mở Streamlit demo...")
        print("Chạy lệnh: streamlit run app_streamlit.py")
    else:
        print("❌ Lựa chọn không hợp lệ")

def interactive_demo():
    """Interactive demo with manual input"""
    print("\n🎮 DEMO TƯƠNG TÁC")
    print("=" * 30)
    
    try:
        age = int(input("Tuổi: "))
        gender = int(input("Giới tính (1=Nam, 0=Nữ): "))
        height = int(input("Chiều cao (cm): "))
        weight = int(input("Cân nặng (kg): "))
        ap_hi = int(input("Huyết áp tâm thu: "))
        ap_lo = int(input("Huyết áp tâm trương: "))
        cholesterol = int(input("Mỡ máu (1=Bình thường, 2=Trên BT, 3=Cao): "))
        gluc = int(input("Đường huyết (1=Bình thường, 2=Trên BT, 3=Cao): "))
        smoke = int(input("Hút thuốc (1=Có, 0=Không): "))
        alco = int(input("Uống rượu (1=Có, 0=Không): "))
        active = int(input("Vận động (1=Có, 0=Không): "))
        
        patient = {
            'age': age, 'gender': gender, 'height': height, 'weight': weight,
            'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol,
            'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active
        }
        
        # Load model and predict
        model_components = load_demo_model()
        if model_components:
            if isinstance(model_components, tuple):
                model, scaler, selector = model_components
            else:
                model = model_components
                scaler = None
                selector = None
            
            prediction, probability = predict_risk(model, patient, scaler, selector)
            
            if prediction is not None:
                risk_level, color, recommendations, risk_factors = interpret_risk(probability, patient)
                
                print(f"\n{color} KẾT QUẢ:")
                print(f"Rủi ro: {risk_level} ({probability:.1%})")
                print("Khuyến nghị:")
                for rec in recommendations:
                    print(f"• {rec}")
            else:
                print("❌ Không thể dự đoán")
        
    except ValueError:
        print("❌ Dữ liệu nhập không hợp lệ")

if __name__ == "__main__":
    main()
