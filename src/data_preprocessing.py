import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.drop(['CLIENTNUM'], axis=1)

    # Target
    df['Attrition_Flag'] = df['Attrition_Flag'].map({
        'Existing Customer': 0,
        'Attrited Customer': 1
    })

    # Encode categorical
    categorical_cols = [
        'Gender', 'Education_Level',
        'Marital_Status', 'Income_Category',
        'Card_Category'
    ]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # ✅ IMPORTANT: Strong features added
    selected_features = [
        'Customer_Age', 'Gender', 'Dependent_count',
        'Education_Level', 'Marital_Status',
        'Income_Category', 'Card_Category',
        'Total_Trans_Ct',
        'Total_Trans_Amt',
        'Months_Inactive_12_mon',
        'Total_Relationship_Count'
    ]

    X = df[selected_features]
    y = df['Attrition_Flag']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, selected_features