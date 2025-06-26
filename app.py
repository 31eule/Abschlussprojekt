import streamlit as st
import login
import arzt_dashboard
import patient_dashboard
from login import login

st.set_page_config(layout="wide")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    if st.session_state["role"] == "Arzt":
        arzt_dashboard.app()
    elif st.session_state["role"] == "Patient":
        patient_dashboard.app()

    

