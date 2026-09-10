import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data, preprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "BankChurners.csv")

print("Loading:", data_path)

df = load_data(data_path)

X, y, scaler, features = preprocess(df)

# SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Model
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Save
model_dir = os.path.join(BASE_DIR, "models")

joblib.dump(model, os.path.join(model_dir, "xgb_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
joblib.dump(features, os.path.join(model_dir, "features.pkl"))

# ✅ Save probabilities for histogram
probs = model.predict_proba(X)[:, 1]
pd.DataFrame(probs).to_csv(os.path.join(model_dir, "probs.csv"), index=False)

print("✅ Model trained & saved successfully!")
#
# y_pred = model.predict(X_test)
#
# # Metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
#
# print("\n📊 Model Evaluation:")
# print("Accuracy :", round(accuracy, 3))
# print("Precision:", round(precision, 3))
# print("Recall   :", round(recall, 3))