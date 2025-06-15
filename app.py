import streamlit as st
import login
import arzt_dashboard
import patient_dashboard

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login.login()
else:
    if st.session_state["role"] == "arzt":
        arzt_dashboard.app()
    elif st.session_state["role"] == "patient":
        patient_dashboard.app()
