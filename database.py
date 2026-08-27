import pandas as pd
import os

USER_FILE = "users.csv"

# ---------------------------
# Add User
# ---------------------------
def add_user(username, password, role):

    if not os.path.exists(USER_FILE):
        df = pd.DataFrame(columns=["username", "password", "role"])
        df.to_csv(USER_FILE, index=False)

    df = pd.read_csv(USER_FILE)

    new_user = pd.DataFrame({
        "username": [username],
        "password": [password],
        "role": [role]
    })

    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_FILE, index=False)


# ---------------------------
# Get History
# ---------------------------
def get_history():
    if not os.path.exists("history.csv"):
        return pd.DataFrame()

    return pd.read_csv("history.csv")


# ---------------------------
# Validate User
# ---------------------------
def validate_user(username, password, role):

    username = username.strip()
    password = password.strip()
    role = role.lower()

    if not os.path.exists(USER_FILE):
        return False

    df = pd.read_csv(USER_FILE)

    user = df[
        (df["username"] == username) &
        (df["password"] == password) &
        (df["role"] == role)
    ]

    return not user.empty
# ---------------------------
# Save Prediction History
# ---------------------------
def save_history(data):
    file = "history.csv"

    if not os.path.exists(file):
        df = pd.DataFrame(columns=data.keys())
        df.to_csv(file, index=False)

    df = pd.read_csv(file)

    new_row = pd.DataFrame([data])
    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(file, index=False)