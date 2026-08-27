import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)

    categorical_cols = [
        "Home Ownership",
        "Loan Purpose",
        "Loan Grade",
        "Previous Default History"
    ]

    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("Loan Status (0=Approved, 1=Default)", axis=1)
    y = df["Loan Status (0=Approved, 1=Default)"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns, encoders, scaler