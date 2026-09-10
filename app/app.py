import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "xgb_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "models", "features.pkl"))

st.title("💳 Customer Churn Prediction System")

# -----------------------
# INPUTS
# -----------------------
st.sidebar.header("Customer Details")

def user_input():
    Customer_Age = st.sidebar.slider("Age", 18, 80, 40)
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    Dependent_count = st.sidebar.slider("Dependents", 0, 5, 1)

    Education_Level = st.sidebar.selectbox("Education",
        ["High School", "Graduate", "Post-Graduate", "Doctorate", "Uneducated"])

    Marital_Status = st.sidebar.selectbox("Marital Status",
        ["Single", "Married", "Divorced"])

    Income_Category = st.sidebar.selectbox("Income",
        ["Less than $40K", "$40K - $60K", "$60K - $80K",
         "$80K - $120K", "$120K +"])

    Card_Category = st.sidebar.selectbox("Card Type",
        ["Blue", "Silver", "Gold", "Platinum"])

    # 🔥 Strong features
    Total_Trans_Ct = st.sidebar.slider("Transaction Count", 0, 150, 50)
    Total_Trans_Amt = st.sidebar.number_input("Transaction Amount", 0, 20000, 3000)
    Months_Inactive_12_mon = st.sidebar.slider("Inactive Months", 0, 12, 2)
    Total_Relationship_Count = st.sidebar.slider("Bank Products", 1, 6, 3)

    # Encoding
    Gender = 1 if Gender == "Male" else 0

    enc = {
        "Education_Level": {"High School":0, "Graduate":1, "Post-Graduate":2, "Doctorate":3, "Uneducated":4},
        "Marital_Status": {"Single":0, "Married":1, "Divorced":2},
        "Income_Category": {"Less than $40K":0, "$40K - $60K":1, "$60K - $80K":2, "$80K - $120K":3, "$120K +":4},
        "Card_Category": {"Blue":0, "Silver":1, "Gold":2, "Platinum":3}
    }

    data = [
        Customer_Age, Gender, Dependent_count,
        enc["Education_Level"][Education_Level],
        enc["Marital_Status"][Marital_Status],
        enc["Income_Category"][Income_Category],
        enc["Card_Category"][Card_Category],
        Total_Trans_Ct, Total_Trans_Amt,
        Months_Inactive_12_mon,
        Total_Relationship_Count
    ]

    return np.array(data)

input_data = user_input()

# -----------------------
# PREDICTION
# -----------------------
if st.button("Predict"):
    prob = model.predict_proba(scaler.transform([input_data]))[0][1]

    st.subheader(f"Churn Probability: {prob:.2f}")

    if prob >= 0.7:
        st.error("⚠️ HIGH RISK")
    elif prob >= 0.4:
        st.warning("⚠️ MEDIUM RISK")
    else:
        st.success("LOW RISK")

# -----------------------
# HISTOGRAM
# -----------------------
st.subheader("📊 Churn Probability Distribution")

probs = pd.read_csv(os.path.join(BASE_DIR, "models", "probs.csv")).values.flatten()

bins = [0, 0.2, 0.4, 0.6, 0.8, 1]

fig, ax = plt.subplots()
counts, bins, patches = ax.hist(probs, bins=bins)

for i, patch in enumerate(patches):
    if bins[i] >= 0.8:
        patch.set_facecolor('red')

ax.set_title("80%+ High Risk Customers Highlighted")
st.pyplot(fig)

# -----------------------
# FEATURE IMPORTANCE
# -----------------------
st.subheader("📌 Top 5 Reasons for Churn")

importance = model.feature_importances_

feat_imp = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False).head(5)

fig2, ax2 = plt.subplots()
ax2.barh(feat_imp["Feature"], feat_imp["Importance"])
ax2.invert_yaxis()

st.pyplot(fig2)