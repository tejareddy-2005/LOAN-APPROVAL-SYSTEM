import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

def train_model(file_path):
    print("Loading dataset...")

    df = pd.read_csv(file_path)

    categorical_cols = [
        "Home Ownership",
        "Loan Purpose",
        "Loan Grade",
        "Previous Default History"
    ]

    encoders = {}

    print("Encoding data...")

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("Loan Status (0=Approved, 1=Default)", axis=1)
    y = df["Loan Status (0=Approved, 1=Default)"]

    print("Scaling data...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training model...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05
    )

    model.fit(X_scaled, y)

    print("Saving model...")

    joblib.dump(model, "xgb_model.pkl")
    joblib.dump(encoders, "encoders.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Training completed!")

    return "Training Completed Successfully"