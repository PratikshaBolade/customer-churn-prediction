import joblib
import numpy as np

model = joblib.load("../models/xgb_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

def predict(data):
    data_scaled = scaler.transform([data])
    return model.predict_proba(data_scaled)[0][1]