import streamlit as st

# ---------------------------
# Session Setup
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None

st.markdown(
    "<h1 style='text-align: center; color: navy;'>"
    "EXPLORING FACTORS INVOLVED IN LOAN APPROVAL DECISION<br>"
    "DEEP INSIGHTS AND DATA ANALYTICS TECHNIQUES"
    "</h1>",
    unsafe_allow_html=True
)

# ---------------------------
# LOGIN PAGE
# ---------------------------
from database import validate_user

def login_page():
    st.title("🏦 Loan Approval System")
    st.subheader("Login")

    role = st.selectbox("Select Role", ["Admin", "Employee"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        role_lower = role.lower()   # ✅ fix case issue

        # ✅ Admin (optional hardcoded)
        if role == "Admin" and username == "admin" and password == "admin":
            st.session_state.logged_in = True
            st.session_state.role = "admin"
            

        # ✅ Employee from CSV
        elif validate_user(username, password, role_lower):
            st.session_state.logged_in = True
            st.session_state.role = "employee"
          

        else:
            st.error("Invalid credentials")
# ---------------------------
# LOGOUT FUNCTION
# ---------------------------
def logout():
    st.session_state.logged_in = False
    st.session_state.role = None
    st.rerun()

# ---------------------------
# MAIN ROUTING
# ---------------------------
if not st.session_state.logged_in:
    login_page()

else:
    # Logout button (TOP RIGHT)
    st.sidebar.button("🚪 Logout", on_click=logout)

    if st.session_state.role == "admin":
        from pages.admin_dashboard import show_admin_dashboard
        show_admin_dashboard()

    elif st.session_state.role == "employee":
        from pages.employee_dashboard import show_employee_dashboard
        show_employee_dashboard()