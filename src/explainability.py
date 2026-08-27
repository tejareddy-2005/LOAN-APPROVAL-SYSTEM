import shap
import matplotlib.pyplot as plt

def explain_model(model, X_train, X_test):
    print("\nGenerating SHAP explanations...")

    # Use TreeExplainer for XGBoost (best)
    explainer = shap.Explainer(model, X_train)

    shap_values = explainer(X_test)

    # Summary plot (global importance)
    shap.plots.bar(shap_values)

    # Detailed plot
    shap.summary_plot(shap_values, X_test)