from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

def evaluate(y_test, y_pred, name="Model"):
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))