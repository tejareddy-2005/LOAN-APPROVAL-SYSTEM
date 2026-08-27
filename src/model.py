from sklearn.ensemble import RandomForestClassifier

def build_model(input_shape=None):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model
