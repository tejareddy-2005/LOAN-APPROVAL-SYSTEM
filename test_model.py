import pandas as pd
import joblib

# ---------------------------
# Load trained model + encoders + scaler
# ---------------------------
model = joblib.load("xgb_model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")

print("\nEnter Loan Details:")

age = float(input("Age: "))
income = float(input("Annual Income: "))
home = input("Home Ownership (RENT/OWN/MORTGAGE): ")
emp = float(input("Employment Length: "))
purpose = input("Loan Purpose: ")
grade = input("Loan Grade (A-E): ")
loan_amt = float(input("Loan Amount: "))
interest = float(input("Interest Rate: "))
loan_ratio = float(input("Loan to Income Ratio: "))
default = input("Previous Default (Y/N): ")
cred_hist = float(input("Credit History Length: "))
dti = float(input("Debt to Income Ratio: "))
credit_util = float(input("Credit Utilization Ratio: "))
open_acc = float(input("Open Accounts: "))
late = float(input("Late Payments: "))

# ---------------------------
# Create DataFrame
# ---------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "Annual Income": income,
    "Home Ownership": home,
    "Employment Length (Years)": emp,
    "Loan Purpose": purpose,
    "Loan Grade": grade,
    "Loan Amount": loan_amt,
    "Interest Rate (%)": interest,
    "Loan to Income Ratio": loan_ratio,
    "Previous Default History": default,
    "Credit History Length (Years)": cred_hist,
    "Debt to Income Ratio": dti,
    "Credit Utilization Ratio": credit_util,
    "Number of Open Credit Accounts": open_acc,
    "Number of Late Payments": late
}])

# ---------------------------
# Apply SAME encoding
# ---------------------------
categorical_cols = [
    "Home Ownership",
    "Loan Purpose",
    "Loan Grade",
    "Previous Default History"
]

for col in categorical_cols:
    input_data[col] = encoders[col].transform(input_data[col])

# ---------------------------
# Apply SAME scaling
# ---------------------------
input_scaled = scaler.transform(input_data)

# ---------------------------
# Prediction
# ---------------------------
prediction = model.predict(input_scaled)

if prediction[0] == 0:
    print("\n✅ Loan Approved")
else:
    print("\n❌ Loan Rejected")