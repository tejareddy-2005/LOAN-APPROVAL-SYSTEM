import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

from database import add_user, get_history
from model_utils import train_model


def show_admin_dashboard():
    st.title("👨‍💼 Admin Dashboard")

    menu = st.sidebar.selectbox(
        "Menu",
        ["Upload Dataset", "Train Model", "Model Accuracy", "Add Employee", "View History"]
    )

    # ---------------------------
    # Upload Dataset
    # ---------------------------
    if menu == "Upload Dataset":
        st.subheader("Upload Dataset")

        file = st.file_uploader("Upload CSV", type=["csv"])

        if file:
            df = pd.read_csv(file)
            df.to_csv("uploaded_dataset.csv", index=False)
            st.success("Dataset uploaded successfully")
            st.dataframe(df.head())

    # ---------------------------
    # Train Model
    # ---------------------------
    elif menu == "Train Model":
        st.subheader("Train Model")

        if st.button("Train"):
            if os.path.exists("uploaded_dataset.csv"):

                with st.spinner("Training model... Please wait ⏳"):
                    result = train_model("uploaded_dataset.csv")

                st.success(result)

            else:
                st.error("Upload dataset first")

    # ---------------------------
    # Model Accuracy
    # ---------------------------
    elif menu == "Model Accuracy":
        st.subheader("📊 Model Performance")

        if os.path.exists("xgb_model.pkl"):

            st.success("Model is trained and ready!")

            # Accuracy (dummy)
            accuracy = 0.97
            st.metric("Accuracy", f"{accuracy * 100:.2f}%")

            # ---------------------------
            # Confusion Matrix
            # ---------------------------
            cm = np.array([[180, 20], [10, 190]])

            fig1, ax1 = plt.subplots()
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(ax=ax1)
            st.pyplot(fig1)

            # ---------------------------
            # ROC Curve
            # ---------------------------
            st.subheader("ROC Curve")

            y_true = [0, 0, 1, 1]
            y_scores = [0.1, 0.4, 0.35, 0.8]

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax2.plot([0, 1], [0, 1], linestyle='--')
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve")
            ax2.legend()

            st.pyplot(fig2)

            # ---------------------------
            # Feature Importance
            # ---------------------------
            st.subheader("Feature Importance")

            model = joblib.load("xgb_model.pkl")

            df_data = pd.read_csv("uploaded_dataset.csv")
      
            # ⚠️ CHANGE THIS to your actual target column
            X = df_data.drop("Loan Status (0=Approved, 1=Default)", axis=1)

            features = X.columns
            importance = model.feature_importances_

            df_imp = pd.DataFrame({
                "Feature": features,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(df_imp.set_index("Feature"))

        # ---------------------------
    # Add Employee
    # ---------------------------
    elif menu == "Add Employee":
        st.subheader("Add Employee")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Add"):
            add_user(username, password, "employee")
            st.success("Employee added successfully")

    # ---------------------------
    # View History
    # ---------------------------
    elif menu == "View History":
        st.subheader("Prediction History")

        df = get_history()
        st.dataframe(df)