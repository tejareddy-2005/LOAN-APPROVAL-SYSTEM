import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from database import save_history


def show_employee_dashboard():
    st.title("👨‍💻 Employee Dashboard")

    st.subheader("Enter Loan Details")

    age = st.number_input("Age", min_value=18)
    income = st.number_input("Annual Income (₹)")
    home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
    emp = st.number_input("Employment Length")
    purpose = st.selectbox("Loan Purpose", ["PERSONAL", "MEDICAL", "EDUCATION", "VENTURE", "HOMEIMPROVEMENT"])
    grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E"])
    loan_amt = st.number_input("Loan Amount (₹)")
    interest = st.number_input("Interest Rate")
    ratio = st.number_input("Loan to Income Ratio")
    default = st.selectbox("Previous Default", ["Y", "N"])
    cred = st.number_input("Credit History Length")
    dti = st.number_input("Debt to Income Ratio")
    util = st.number_input("Credit Utilization")
    open_acc = st.number_input("Open Accounts")
    late = st.number_input("Late Payments")

    if st.button("Predict"):

        # ---------------------------
        # Load model
        # ---------------------------
        model = joblib.load("xgb_model.pkl")

        # ---------------------------
        # Encoding
        # ---------------------------
        home_map = {"RENT": 0, "OWN": 1, "MORTGAGE": 2}
        purpose_map = {"PERSONAL": 0, "MEDICAL": 1, "EDUCATION": 2, "VENTURE": 3, "HOMEIMPROVEMENT": 4}
        grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

        # ---------------------------
        # Input Data
        # ---------------------------
        input_df = pd.DataFrame([{
            "Age": age,
            "Annual Income (₹)": income,
            "Home Ownership": home_map[home],
            "Employment Length": emp,
            "Loan Purpose": purpose_map[purpose],
            "Loan Grade": grade_map[grade],
            "Loan Amount (₹)": loan_amt,
            "Interest Rate": interest,
            "Loan to Income Ratio": ratio,
            "Previous Default": 1 if default == "Y" else 0,
            "Credit History Length (Years)": cred,
            "Debt to Income Ratio": dti,
            "Credit Utilization Ratio": util,
            "Number of Open Credit Accounts": open_acc,
            "Number of Late Payments": late
        }])

        # ---------------------------
        # Correct column order
        # ---------------------------
        columns_order = [
            "Age",
            "Annual Income (₹)",
            "Home Ownership",
            "Employment Length",
            "Loan Purpose",
            "Loan Grade",
            "Loan Amount (₹)",
            "Interest Rate",
            "Loan to Income Ratio",
            "Previous Default",
            "Credit History Length (Years)",
            "Debt to Income Ratio",
            "Credit Utilization Ratio",
            "Number of Open Credit Accounts",
            "Number of Late Payments"
        ]

        input_df = input_df[columns_order]

        # ---------------------------
        # Prediction
        # ---------------------------
        prediction = model.predict(input_df)[0]

        result = "❌ Rejected" if prediction == 1 else "✅ Approved"
        st.success(result)

        # ---------------------------
        # Save History
        # ---------------------------
        save_history({
            "user": "employee",
            "prediction": result
        })

        # ---------------------------
        # SHAP Explanation
        # ---------------------------
        st.subheader("🔍 SHAP Explanation")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_df, show=False)
        st.pyplot(fig)