from src.preprocessing import load_and_preprocess
from src.model import build_model
from src.evaluation import evaluate

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import pandas as pd
import joblib

# ---------------------------
# Load and preprocess
# ---------------------------
X, y, feature_names, encoders, scaler = load_and_preprocess("data/final_dataset_readable.csv")

# Convert to DataFrame
X_df = pd.DataFrame(X, columns=feature_names)

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42
)

# ---------------------------
# 1. Logistic Regression
# ---------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\n==============================")
print("Logistic Regression Results")
print("==============================")
evaluate(y_test, y_pred_lr, "Logistic Regression")

# ---------------------------
# 2. XGBoost (MAIN MODEL ⭐)
# ---------------------------
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("\n==============================")
print("XGBoost Results (BEST MODEL)")
print("==============================")
evaluate(y_test, y_pred_xgb, "XGBoost")

# ---------------------------
# 3. CNN Model
# ---------------------------
X_train_cnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

model = build_model((X_train.shape[1], 1))

model.fit(X_train_cnn, y_train, epochs=10, batch_size=32)

y_pred_cnn = (model.predict(X_test_cnn) > 0.5).astype(int)

print("\n==============================")
print("1D-CNN Results")
print("==============================")
evaluate(y_test, y_pred_cnn, "1D-CNN")

# ---------------------------
# Save Model + Encoders + Scaler
# ---------------------------
joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModels and preprocessors saved successfully!")

# ---------------------------
# Final Summary
# ---------------------------
print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("CNN Accuracy:", accuracy_score(y_test, y_pred_cnn))